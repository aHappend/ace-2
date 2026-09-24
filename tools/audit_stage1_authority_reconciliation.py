#!/usr/bin/env python3
"""Fail-closed audit for the Stage 1 production-attestation boundary.

This tool is intentionally read-only with respect to Manager-owned pipeline
state and official-attempt namespaces.  It can write only its requested audit
report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


PIPELINE = Path("research/PIPELINE_STATE.json")
PIPELINE_CHECKSUM = Path("research/PIPELINE_STATE.sha256")
PACKAGE = Path(
    "reference/"
    "qk_gbfp8_head64_b0_external_attestation_prerequisite_static_v30_action_root/"
    "QK_GBFP8_HEAD64_B0_EXTERNAL_ATTESTATION_PREREQUISITE_STATIC_V30_PACKAGE.json"
)
STATIC_REPORT = Path(
    "build/planner-cycle1-v30-static-recheck-20260814/"
    "V30_STATIC_PREREQUISITE_REPORT.json"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def checksum_digest(raw: bytes) -> str:
    fields = raw.decode("utf-8").strip().split()
    return fields[0] if fields else ""


def attempt_projection(pipeline: dict[str, Any]) -> dict[str, Any]:
    blocker = pipeline.get("stage1_blocker")
    blocker = blocker if isinstance(blocker, dict) else {}
    return {
        "b0": blocker.get("b0"),
        "effects": blocker.get("effects"),
        "authorization": blocker.get("authorization"),
        "status": blocker.get("status"),
    }


def projection_digest(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def production_contract_ready(package: dict[str, Any]) -> tuple[bool, list[str]]:
    trust = package.get("production_trust_boundary") or {}
    contract = package.get("production_verification_contract") or {}
    missing: list[str] = []
    required = (
        "algorithm",
        "key_id",
        "public_key_sha256",
        "canonicalization_identifier",
        "canonicalization_version",
        "signed_payload_field_order",
        "key_rotation_contract_sha256",
        "anti_replay_contract_sha256",
        "first_terminal_contract_sha256",
        "framework_verifier_endpoint",
        "immutable_receipt_source",
    )
    for field in required:
        if contract.get(field) in (None, "", []):
            missing.append(field)
    if trust.get("production_receipt_acceptance_enabled") is not True:
        missing.append("production_receipt_acceptance_enabled")
    return not missing, missing


def receipt_summary(package: dict[str, Any], root: Path) -> dict[str, Any]:
    trust = package.get("production_trust_boundary") or {}
    contract = package.get("production_verification_contract") or {}
    required = trust.get("required_attestors") or []
    roles = [str(item.get("layer")) for item in required if isinstance(item, dict)]
    source = contract.get("immutable_receipt_source")
    source_exists = False
    if isinstance(source, str) and source:
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = root / candidate
        source_exists = candidate.is_file()
    return {
        "required_roles": roles,
        "required_count": len(roles),
        "immutable_source_declared": bool(source),
        "immutable_source_exists": source_exists,
        "cryptographically_verified_count": 0,
        "per_role": {role: "MISSING_PRODUCTION_RECEIPT" for role in roles},
    }


def audit(root: Path) -> tuple[dict[str, Any], bool]:
    pipeline_path = root / PIPELINE
    checksum_path = root / PIPELINE_CHECKSUM
    package_path = root / PACKAGE
    static_report_path = root / STATIC_REPORT

    pipeline_before = pipeline_path.read_bytes()
    checksum_before = checksum_path.read_bytes()
    pipeline = json.loads(pipeline_before)
    package = load_json(package_path)
    static_report = load_json(static_report_path)

    attempt_before = attempt_projection(pipeline)
    attempt_before_digest = projection_digest(attempt_before)
    attempt_root_rel = (
        package.get("predecessor_preservation", {})
        .get("additive_0005", {})
        .get("build_attempt_root_must_be_absent")
    )
    attempt_root = root / str(attempt_root_rel) if attempt_root_rel else None
    attempt_root_before = bool(attempt_root and attempt_root.exists())

    contract_ready, missing_contract_fields = production_contract_ready(package)
    receipts = receipt_summary(package, root)
    receipts_verified = bool(
        contract_ready
        and receipts["cryptographically_verified_count"] == receipts["required_count"]
    )

    pipeline_digest = sha256_bytes(pipeline_before)
    companion_digest = checksum_digest(checksum_before)
    checksum_matches = pipeline_digest == companion_digest

    pipeline_after = pipeline_path.read_bytes()
    checksum_after = checksum_path.read_bytes()
    attempt_after = attempt_projection(json.loads(pipeline_after))
    attempt_after_digest = projection_digest(attempt_after)
    attempt_root_after = bool(attempt_root and attempt_root.exists())
    attempt_before_b0 = attempt_before.get("b0") or {}
    attempt_before_effects = attempt_before.get("effects") or {}

    no_attempt_consumed = bool(
        attempt_before_digest == attempt_after_digest
        and not attempt_root_before
        and not attempt_root_after
        and attempt_before_b0.get("executed", False) is False
        and attempt_before_effects.get("b0_effect_count", 0) == 0
        and attempt_before_effects.get("evaluator_call_count", 0) == 0
        and attempt_before_effects.get("payload_open_count", 0) == 0
        and attempt_before_effects.get("target_start_count", 0) == 0
    )
    manager_state_unchanged = pipeline_before == pipeline_after
    checksum_companion_unchanged = checksum_before == checksum_after
    accepted = bool(checksum_matches and receipts_verified)

    if accepted:
        status = "ACCEPTED"
    elif not receipts_verified:
        status = "BLOCKED_EXTERNAL_ATTESTATION"
    else:
        status = "READY_FOR_MANAGER_CHECKSUM_RECONCILIATION"

    report = {
        "schema_version": 1,
        "artifact_kind": "ace2_stage1_authority_reconciliation_audit",
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "status": status,
        "acceptance": {
            "pipeline_checksum_matches": checksum_matches,
            "authority_gate_verifies_all_three_production_receipts": receipts_verified,
            "accepted": accepted,
        },
        "pipeline": {
            "current_stage": pipeline.get("current_stage"),
            "stage1_blocker_status": (pipeline.get("stage1_blocker") or {}).get(
                "status"
            ),
            "computed_sha256": pipeline_digest,
            "companion_sha256": companion_digest,
            "manager_owned_state_unchanged_during_audit": manager_state_unchanged,
            "checksum_companion_unchanged_during_audit": checksum_companion_unchanged,
        },
        "production_contract": {
            "status": (package.get("production_verification_contract") or {}).get(
                "status"
            ),
            "ready": contract_ready,
            "missing_fields": missing_contract_fields,
            "receipt_acceptance_enabled": (
                package.get("production_trust_boundary") or {}
            ).get("production_receipt_acceptance_enabled"),
        },
        "production_receipts": receipts,
        "attempt_invariants": {
            "canonical_projection_before_sha256": attempt_before_digest,
            "canonical_projection_after_sha256": attempt_after_digest,
            "declared_attempt_root": str(attempt_root_rel or ""),
            "attempt_root_absent_before": not attempt_root_before,
            "attempt_root_absent_after": not attempt_root_after,
            "no_attempt_consumed": no_attempt_consumed,
            "projection": attempt_after,
        },
        "static_v30_evidence": {
            "status": static_report.get("status"),
            "production_receipt_verifications": (
                static_report.get("production_verification") or {}
            ).get("production_receipt_verifications"),
            "production_receipts_accepted": (
                static_report.get("production_verification") or {}
            ).get("production_receipts_accepted"),
        },
        "manager_reconciliation": {
            "eligible": receipts_verified,
            "performed_by_this_tool": False,
            "reason": (
                "Only the Manager may mutate PIPELINE_STATE and its checksum; "
                "the production receipt gate is not satisfied."
            ),
        },
        "claim_boundary": (
            "Read-only production-authority audit only. No receipt, credential, "
            "authority, runtime namespace, payload access, evaluator call, target "
            "start, official attempt, or Manager-owned state mutation is created."
        ),
    }
    return report, accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-accepted", action="store_true")
    args = parser.parse_args()

    report, accepted = audit(args.root.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = args.root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if args.require_accepted and not accepted else 0


if __name__ == "__main__":
    raise SystemExit(main())
