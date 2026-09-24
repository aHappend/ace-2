#!/usr/bin/env python3
"""Bind the projection-shadow candidate before its unique paired smoke."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "layer0_projection_shadow_staged_attention_v1"
PROPOSAL_SHA256 = "bb686fdb1a8e886ba80e8578db25a5db39eb58a757fc053986fc4538d6c36d92"
EVIDENCE = ROOT / "evidence" / CONTRACT / "latest"
APPROVAL = ROOT / "evidence" / "authorization" / CONTRACT / "operator_approval.json"
SOURCE_LIST = EVIDENCE / "source_hashes.txt"
CANDIDATE = EVIDENCE / "candidate_evidence.json"
SOURCES = [
    "Makefile",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    f"evidence/authorization/{CONTRACT}/operator_approval.json",
    "rtl/ace2_projection_shadow_staged_attention_core.sv",
    "tools/ace2_full_model_fixed_point.py",
    "tools/ace2_projection_shadow_reference.py",
    "tools/bind_projection_shadow_candidate.py",
    "tools/gen_projection_shadow_vectors.py",
    "verification/generated/projection_shadow_vectors.json",
    "verification/generated/projection_shadow_vectors.svh",
    "verification/tb/ace2_projection_shadow_staged_attention_tb.sv",
    "verification/test_projection_shadow_staged_attention.py",
]
LOGS = {
    "software_reference": (
        "evidence/layer0_projection_shadow_staged_attention_v1/latest/"
        "software_reference_unittest.log"
    ),
    "full_model_self_test": (
        "evidence/layer0_projection_shadow_staged_attention_v1/latest/"
        "full_model_self_test.log"
    ),
    "exact_runtime": (
        "evidence/layer0_projection_shadow_staged_attention_v1/latest/"
        "runtime_dependency.log"
    ),
    "rtl_simulation": (
        "evidence/layer0_projection_shadow_staged_attention_v1/latest/"
        "rtl_projection_shadow.log"
    ),
    "projection_lint": (
        "evidence/layer0_projection_shadow_staged_attention_v1/latest/"
        "rtl_projection_lint.log"
    ),
    "score_lint": (
        "evidence/layer0_projection_shadow_staged_attention_v1/latest/"
        "rtl_score_lint.log"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    require(
        load(ROOT / "research/PIPELINE_STATE.json").get("current_stage") == "rtl",
        "Manager-owned stage is not rtl",
    )
    require(
        sha256(ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md") == PROPOSAL_SHA256,
        "proposal hash changed",
    )
    approval = load(APPROVAL)
    require(
        approval.get("authority") == "operator"
        and approval.get("contract_id") == CONTRACT,
        "operator approval binding is missing",
    )
    for source in SOURCES:
        require((ROOT / source).is_file(), f"candidate source missing: {source}")
    for relative in LOGS.values():
        require((ROOT / relative).is_file(), f"focused log missing: {relative}")
    require("OK" in (ROOT / LOGS["software_reference"]).read_text(encoding="utf-8"),
            "software reference tests did not pass")
    require("SELF_TEST status=pass" in
            (ROOT / LOGS["full_model_self_test"]).read_text(encoding="utf-8"),
            "full-model self-test did not pass")
    require("gmpy2 2.3.1" in
            (ROOT / LOGS["exact_runtime"]).read_text(encoding="utf-8"),
            "exact runtime dependency is not bound")
    require("TB_PASS" in (ROOT / LOGS["rtl_simulation"]).read_text(encoding="utf-8"),
            "focused RTL simulation did not pass")
    require((ROOT / LOGS["projection_lint"]).stat().st_size == 0,
            "projection lint is not warning-free")
    require((ROOT / LOGS["score_lint"]).stat().st_size == 0,
            "score lint is not warning-free")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    source_hashes = {source: sha256(ROOT / source) for source in sorted(SOURCES)}
    SOURCE_LIST.write_text(
        "".join(f"{digest}  {source}\n" for source, digest in source_hashes.items()),
        encoding="utf-8",
    )
    candidate_hash = sha256(SOURCE_LIST)
    manifest = ROOT / "design/RTL_MANIFEST.json"
    ledger = ROOT / "design/PPA_FRONTIER_LEDGER.json"
    focused = {
        name: {"status": "pass", **artifact(ROOT / relative)}
        for name, relative in LOGS.items()
    }
    focused["projection_lint"]["status"] = "pass_warning_free"
    focused["score_lint"]["status"] = "pass_warning_free"
    payload = {
        "schema_version": 1,
        "candidate_id": f"projection_shadow_staged_attention_{candidate_hash[:16]}",
        "contract_id": CONTRACT,
        "proposal_sha256": PROPOSAL_SHA256,
        "recorded_at_utc": utc_now(),
        "status": "focused_native_checks_pass_operator_directed_paired_smoke_pending",
        "acceptance_claim": False,
        "accepted_publication_frontier_changed": False,
        "official_14_item_evaluation_run": False,
        "full_shell_regression_run": False,
        "canonical_sky130_ppa_run": False,
        "operator_approval": {
            "authority": "operator",
            "approved_at_utc": approval["approved_at_utc"],
            "evidence": APPROVAL.relative_to(ROOT).as_posix(),
            "evidence_sha256": sha256(APPROVAL),
            "scope": approval["scope"],
        },
        "source_binding": {
            "ordered_source_hash_list_sha256": candidate_hash,
            "source_hash_list": SOURCE_LIST.relative_to(ROOT).as_posix(),
        },
        "focused_verification": focused,
        "accepted_frontier_preservation": {
            "rtl_manifest_path": manifest.relative_to(ROOT).as_posix(),
            "rtl_manifest_sha256": sha256(manifest),
            "ppa_frontier_ledger_path": ledger.relative_to(ROOT).as_posix(),
            "ppa_frontier_ledger_sha256": sha256(ledger),
            "non_sram_area_mm2": 0.6108746272,
            "setup_slack_ns_at_100mhz": 0.1502,
        },
        "frozen_stop_rule": {
            "wikitext2_ratio_strictly_below": 13549.939049967887,
            "c4_en_512_ratio_strictly_below": 4477.990517308544,
            "failure_action": "seal_bounded_no_go_without_full_shell_ppa_or_official_14",
        },
    }
    CANDIDATE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "ACE2_PROJECTION_SHADOW_CANDIDATE_BOUND "
        f"candidate_id={payload['candidate_id']} sha256={sha256(CANDIDATE)}"
    )


if __name__ == "__main__":
    main()
