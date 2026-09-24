#!/usr/bin/env python3
"""Read-only consistency check for the DPRF verification integrity disposition."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from run_down_projection_residual_fusion_verification import check_replays


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
DECISION = ROOT / f"evidence/{CONTRACT}/latest/VERIFICATION_DECISION.json"
RESULTS = ROOT / "verification/RESULTS.json"
REVIEW = ROOT / "evidence/review/focused_verification_shared_down_projection_residual_fusion_v1/decision.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
ORACLE = ROOT / "reference/ORACLE_MANIFEST.json"
PROPERTIES = ROOT / "formal/ACE2_VERIFICATION_PROPERTIES.json"
PLAN = ROOT / "verification/PLAN.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"


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


def verify_artifact(record: dict[str, Any]) -> None:
    require(artifact(ROOT / record["path"]) == record, f"artifact binding differs: {record['path']}")


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


def main() -> int:
    require(load(PIPELINE).get("current_stage") == "verification", "Manager-owned stage changed")
    decision = load(DECISION)
    results = load(RESULTS)
    review = load(REVIEW)
    public = load(PUBLIC)
    verify_canonical(decision, "authoritative decision")
    verify_canonical(results, "verification results")
    verify_canonical(review, "independent L2 review")
    verify_canonical(public, "public status")

    require(decision["status"] == "integrity_no_go", "authoritative status differs")
    require(decision["technical_disposition"]["status"] == "bounded_no_go", "technical status differs")
    execution = decision["execution_contract"]
    require(execution["required_successful_execution_count"] == 1, "exact-one requirement differs")
    require(execution["observed_successful_execution_count"] == 2, "successful execution count differs")
    require(execution["satisfied"] is False, "execution contract is incorrectly marked satisfied")
    require(execution["rerun_permitted"] is False, "third execution is incorrectly permitted")

    first_baseline_record = execution["first_successful_execution"]["baseline_replay"]
    first_candidate_record = execution["first_successful_execution"]["candidate_replay"]
    second_baseline_record = execution["later_duplicate_execution"]["baseline_replay"]
    second_candidate_record = execution["later_duplicate_execution"]["candidate_replay"]
    for record in (
        first_baseline_record,
        first_candidate_record,
        second_baseline_record,
        second_candidate_record,
        decision["precheck"],
        decision["rtl_review"],
        *decision["source_hashes"],
    ):
        verify_artifact(record)

    first_baseline = load(ROOT / first_baseline_record["path"])
    first_candidate = load(ROOT / first_candidate_record["path"])
    second_baseline = load(ROOT / second_baseline_record["path"])
    second_candidate = load(ROOT / second_candidate_record["path"])
    require(first_baseline["generated_at_utc"] == "2026-08-02T00:14:50Z", "first baseline timestamp differs")
    require(first_candidate["generated_at_utc"] == "2026-08-02T00:14:50Z", "first candidate timestamp differs")
    require(second_baseline["generated_at_utc"] == "2026-08-02T00:17:24Z", "second baseline timestamp differs")
    require(second_candidate["generated_at_utc"] == "2026-08-02T00:17:24Z", "second candidate timestamp differs")
    require(strip_execution_time(first_baseline) == strip_execution_time(second_baseline), "baseline payloads differ")
    require(strip_execution_time(first_candidate) == strip_execution_time(second_candidate), "candidate payloads differ")

    quality = check_replays(first_baseline, first_candidate)
    require(quality == decision["technical_disposition"]["quality_gate"], "decision quality gate differs")
    require(quality == results["quality_gate"], "results quality gate differs")
    require(len(quality["failed_conditions"]) == 8 and quality["passed"] is False, "unexpected quality disposition")

    require(results["status"] == "integrity_no_go", "results status differs")
    require(results["stage_closing"] is False and results["candidate_accepted"] is False, "results acceptance differs")
    require(results["decision"] == artifact(DECISION), "results decision binding differs")
    require(results["independent_l2_review"] == artifact(REVIEW), "results review binding differs")
    require(results["oracle_manifest"] == artifact(ORACLE), "oracle manifest binding differs")
    require(results["property_manifest"] == artifact(PROPERTIES), "property manifest binding differs")
    require(all(item["exit_status"] == 0 for item in results["commands"]), "retained command failure recorded")
    for item in results["commands"]:
        verify_artifact(item["raw_log"])

    require(review["reviewer_role"] == "independent_l2", "reviewer role differs")
    require(review["reviewer_status"] == "done", "reviewer status differs")
    require(review["decision"] == "accept_integrity_no_go", "review decision differs")
    require(review["stage_closing"] is False, "review incorrectly closes verification")
    require(review["execution_contract_satisfied"] is False, "review incorrectly accepts exact-one")
    require(review["authoritative_discriminator_decision"] == artifact(DECISION), "review decision input differs")
    verify_artifact(review["verification_results_review_input"])
    verify_artifact(review["review_tool"])
    require("verification_results" not in review, "review contains a mutable results backlink")
    for record in review["technical_evidence"].values():
        if isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}:
            verify_artifact(record)

    latest = public["latest_verification_stage"]
    require(latest["decision"] == artifact(DECISION), "public decision binding differs")
    require(latest["independent_l2_review"] == artifact(REVIEW), "public review binding differs")
    require(public["stage"]["current_stage"] == "verification", "public stage differs")
    require(public["stage_closing"] is False, "public status incorrectly closes verification")
    require(public["ordered_supported_layer_operator_prefix"][-1] == "layer_0.v_proj", "supported prefix differs")
    require(public["first_unsupported_layer_operator"] == "layer_0.rope_q", "first unsupported operator differs")
    require(public["current_mode"] == "ADVANCE", "mode differs")

    plan_text = PLAN.read_text(encoding="utf-8")
    checkpoint_text = CHECKPOINT.read_text(encoding="utf-8")
    require("Technical status: `bounded_no_go`" in plan_text, "plan technical status missing")
    require("Execution-integrity status: `integrity_no_go`" in plan_text, "plan integrity status missing")
    require("Status: `integrity_no_go`" in checkpoint_text, "checkpoint integrity status missing")
    require("No third discriminator run was performed" in checkpoint_text, "checkpoint rerun boundary missing")

    print(
        "ACE2_DPRF_INTEGRITY_DISPOSITION_CHECK_PASS "
        f"decision_sha256={sha256(DECISION)} review_sha256={sha256(REVIEW)} "
        f"failed_conditions={len(quality['failed_conditions'])} current_stage=verification"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
