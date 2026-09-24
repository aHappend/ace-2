#!/usr/bin/env python3
"""Bind the tile-BFP candidate after its focused discriminator passes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "layer0_tile_bfp_score_attention_v1"
PROPOSAL_SHA256 = "a437a57a7d2dc35254222d818d7c2c2300fbc42e49bd60fa3001af5de4569c13"
EVIDENCE = ROOT / "evidence" / CONTRACT / "latest"
APPROVAL = ROOT / "evidence" / "authorization" / CONTRACT / "operator_approval.json"
SOURCE_LIST = EVIDENCE / "source_hashes.txt"
CANDIDATE = EVIDENCE / "candidate_evidence.json"
BASELINE = ROOT / "evidence/layer0_tile_max_delta_attention_v1/focused-baseline-full-20260801-v1/results.json"
FOCUSED = ROOT / f"evidence/{CONTRACT}/focused-candidate-full-20260801-v1/results.json"
SOURCES = [
    "Makefile",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    f"evidence/authorization/{CONTRACT}/operator_approval.json",
    "rtl/ace2_tile_bfp_score_attention_core.sv",
    "tools/ace2_full_model_fixed_point.py",
    "tools/ace2_tile_bfp_reference.py",
    "tools/bind_tile_bfp_candidate.py",
    "tools/bind_tile_bfp_rtl_state.py",
    "tools/gen_tile_bfp_attention_vectors.py",
    "tools/localize_score_to_lm_head.py",
    "verification/generated/tile_bfp_attention_vectors.json",
    "verification/generated/tile_bfp_attention_vectors.svh",
    "verification/tb/ace2_tile_bfp_attention_tb.sv",
    "verification/test_tile_bfp_attention.py",
]
LOGS = {
    "software_reference": f"evidence/{CONTRACT}/latest/software_reference_unittest.log",
    "full_model_self_test": f"evidence/{CONTRACT}/latest/full_model_self_test.log",
    "exact_runtime": f"evidence/{CONTRACT}/latest/runtime_dependency.log",
    "rtl_simulation": f"evidence/{CONTRACT}/latest/rtl_tile_bfp.log",
    "score_lint": f"evidence/{CONTRACT}/latest/rtl_score_lint.log",
    "softmax_lint": f"evidence/{CONTRACT}/latest/rtl_softmax_lint.log",
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
        and approval.get("contract_id") == CONTRACT
        and approval.get("stage_closing") is False,
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
    require((ROOT / LOGS["score_lint"]).stat().st_size == 0,
            "score lint is not warning-free")
    require((ROOT / LOGS["softmax_lint"]).stat().st_size == 0,
            "softmax lint is not warning-free")

    baseline = load(BASELINE)
    focused = load(FOCUSED)
    require(baseline["input_observations"] == focused["input_observations"],
            "focused input observations differ")
    comparisons: dict[str, Any] = {}
    for dataset in ("wikitext2", "c4_en_512"):
        score_base = baseline["comparisons"][dataset]["model.layers.0.score"]
        score_cand = focused["comparisons"][dataset]["model.layers.0.score"]
        value_base = baseline["comparisons"][dataset]["model.layers.0.attention_value"]
        value_cand = focused["comparisons"][dataset]["model.layers.0.attention_value"]
        comparisons[dataset] = {
            "score_relative_l2": {
                "baseline": score_base["relative_l2_error"],
                "candidate": score_cand["relative_l2_error"],
                "strictly_improved": score_cand["relative_l2_error"] < score_base["relative_l2_error"],
            },
            "score_centered_zero_fraction": score_cand["candidate_zero_fraction"],
            "attention_value_relative_l2": {
                "baseline": value_base["relative_l2_error"],
                "candidate": value_cand["relative_l2_error"],
                "strictly_improved": value_cand["relative_l2_error"] < value_base["relative_l2_error"],
            },
        }
        require(comparisons[dataset]["score_relative_l2"]["strictly_improved"],
                f"{dataset} score relative-L2 did not strictly improve")
        require(comparisons[dataset]["attention_value_relative_l2"]["strictly_improved"],
                f"{dataset} attention-value relative-L2 did not strictly improve")
        require(comparisons[dataset]["score_centered_zero_fraction"] < 0.10,
                f"{dataset} centered-score zero fraction is not below 0.10")

    source_hashes = {source: sha256(ROOT / source) for source in sorted(SOURCES)}
    SOURCE_LIST.write_text(
        "".join(f"{digest}  {source}\n" for source, digest in source_hashes.items()),
        encoding="utf-8",
    )
    candidate_hash = sha256(SOURCE_LIST)
    payload = {
        "schema_version": 1,
        "candidate_id": f"tile_bfp_score_attention_{candidate_hash[:16]}",
        "contract_id": CONTRACT,
        "proposal_sha256": PROPOSAL_SHA256,
        "recorded_at_utc": utc_now(),
        "status": "focused_gate_pass_unique_paired_smoke_authorized",
        "acceptance_claim": False,
        "accepted_publication_frontier_changed": False,
        "stage_closing": False,
        "operator_approval": artifact(APPROVAL),
        "source_binding": {
            "ordered_source_hash_list_sha256": candidate_hash,
            "source_hash_list": SOURCE_LIST.relative_to(ROOT).as_posix(),
            "source_hashes": source_hashes,
        },
        "focused_verification": {
            **{name: artifact(ROOT / relative) for name, relative in LOGS.items()},
            "baseline": artifact(BASELINE),
            "candidate": artifact(FOCUSED),
            "comparisons": comparisons,
            "passed": True,
        },
        "accepted_frontier_preservation": {
            "rtl_manifest_path": "design/RTL_MANIFEST.json",
            "rtl_manifest_sha256": sha256(ROOT / "design/RTL_MANIFEST.json"),
            "ppa_frontier_ledger_path": "design/PPA_FRONTIER_LEDGER.json",
            "ppa_frontier_ledger_sha256": sha256(ROOT / "design/PPA_FRONTIER_LEDGER.json"),
            "supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm",
                "layer_0.q_proj",
                "layer_0.k_proj",
                "layer_0.v_proj"
            ],
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "mode": "ADVANCE",
            "non_sram_area_mm2": 0.6108746272,
            "setup_slack_ns_at_100mhz": 0.1502,
        },
        "frozen_stop_rule": {
            "wikitext2_ratio_strictly_below": 13549.939049967887,
            "c4_en_512_ratio_strictly_below": 4477.990517308544,
            "failure_action": "seal_bounded_no_go_without_full_shell_or_ppa",
            "pass_action": "continue_same_bounded_rtl_task_through_full_regression_and_canonical_sky130_ppa",
        },
    }
    CANDIDATE.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ACE2_TILE_BFP_CANDIDATE_BOUND "
        f"candidate_id={payload['candidate_id']} sha256={sha256(CANDIDATE)}"
    )


if __name__ == "__main__":
    main()
