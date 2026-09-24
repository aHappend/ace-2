#!/usr/bin/env python3
"""Validate the active environment audit and detect a stale prior Reviewer binding."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_v_residual_value_correction_attention_v1"
PACKET = (
    ROOT
    / "evidence/shared_qk_residual_cross_term_attention_v1/latest/"
    "ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
)
AUDIT = ROOT / "research/ENVIRONMENT_AUDIT.json"
VERDICT = ROOT / "research/ENVIRONMENT_REVIEWER_VERDICT.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
CHIP_SCOPE = ROOT / "design/CHIP_SCOPE.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_matches(record: dict[str, Any], path: Path) -> bool:
    return (
        record.get("path") == path.relative_to(ROOT).as_posix()
        and record.get("bytes") == path.stat().st_size
        and record.get("sha256") == sha256(path)
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect-stale",
        action="store_true",
        help="require the prior independent environment verdict to be stale",
    )
    args = parser.parse_args()

    packet = load(PACKET)
    audit = load(AUDIT)
    verdict = load(VERDICT)
    pipeline = load(PIPELINE)
    public = load(PUBLIC_STATUS)
    scope = load(CHIP_SCOPE)

    require(packet.get("contract_id") == CONTRACT, "active architecture packet changed")
    require(packet.get("current_stage") == "architecture", "architecture packet stage changed")
    require(packet.get("implementation_authorized") is False, "architecture packet authorizes implementation")
    require(audit.get("integrity", {}).get("canonical_sha256") == canonical_sha256(audit),
            "environment audit canonical hash is corrupt")

    binding = audit.get("contract_binding", {})
    require(binding.get("active_contract_id") == CONTRACT, "environment audit contract is stale")
    require(record_matches(binding.get("architecture_packet", {}), PACKET),
            "environment audit does not bind the refreshed architecture packet")
    require(binding.get("preserved_contracts") == packet.get("preserved_contracts"),
            "environment audit changed the preserved frontier")
    require(binding.get("implementation_authorized") is False,
            "environment audit authorizes implementation")
    require(binding.get("environment_review_status") == "pending_fresh_independent_reviewer",
            "environment audit self-claims independent acceptance")
    require(pipeline.get("current_stage") == "environment", "Manager-owned stage is not environment")
    require(public.get("stage", {}).get("current_stage") == "architecture",
            "public architecture hold changed before fresh environment acceptance")

    active = scope.get("operator_owned_execution_policy", {}).get("active_successor_contract", {})
    selected = public.get("selected_replacement_contract", {})
    require(active.get("contract_id") == CONTRACT and active.get("implementation_authorized") is False,
            "CHIP_SCOPE active successor authorization is stale")
    require(selected.get("contract_id") == CONTRACT and selected.get("implementation_authorized") is False,
            "PUBLIC_STATUS active successor authorization is stale")

    audit_sha = sha256(AUDIT)
    packet_sha = sha256(PACKET)
    stale_reasons = {
        "contract_id": verdict.get("contract_id") != CONTRACT,
        "environment_audit": verdict.get("current_environment_binding", {}).get("sha256") != audit_sha,
        "architecture_packet": verdict.get("source_architecture_packet", {}).get("sha256") != packet_sha,
        "implementation_authority": verdict.get("implementation_authorized") is not False,
    }
    reviewer_verdict_stale = any(stale_reasons.values())

    result = {
        "contract_id": CONTRACT,
        "manager_owned_current_stage": pipeline.get("current_stage"),
        "public_stage_hold": public.get("stage", {}).get("current_stage"),
        "architecture_packet_sha256": packet_sha,
        "environment_audit_sha256": audit_sha,
        "environment_audit_canonical_sha256": audit.get("integrity", {}).get("canonical_sha256"),
        "environment_audit_fresh": True,
        "prior_reviewer_verdict_stale": reviewer_verdict_stale,
        "prior_reviewer_stale_reasons": stale_reasons,
        "implementation_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.expect_stale:
        require(reviewer_verdict_stale, "prior independent environment verdict unexpectedly matches the fresh audit")
        print(
            "ACE2_ENVIRONMENT_BINDING_STALE_CONFIRMED "
            "reviewer_verdict_stale=true manager_stage=environment "
            "public_stage_hold=architecture implementation_authorized=false"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
