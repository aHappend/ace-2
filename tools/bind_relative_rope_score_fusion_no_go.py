#!/usr/bin/env python3
"""Bind the authorized relative-RoPE implementation to its smoke stop rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = "layer0_relative_rope_score_fusion_v1"
CANDIDATE_EVIDENCE = (
    ROOT / "evidence/layer0_relative_rope_score_fusion_v1/latest/candidate_evidence.json"
)
SOURCE_LIST = (
    ROOT / "evidence/layer0_relative_rope_score_fusion_v1/latest/source_hashes.txt"
)
BOUNDARY = (
    ROOT
    / "evidence/layer0_relative_rope_score_fusion_v1/latest/global_valid_key_max_contract.json"
)
SOFTWARE_LOG = (
    ROOT
    / "evidence/layer0_relative_rope_score_fusion_v1/latest/software_reference_unittest.log"
)
RTL_LOG = (
    ROOT
    / "evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_relative_rope_score_fusion.log"
)
ARITH_LINT = (
    ROOT / "evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_arithmetic_lint.log"
)
CENTER_LINT = (
    ROOT / "evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_center_lint.log"
)
BASELINE = (
    ROOT / "benchmark/raw/quality/operator-paired-smoke-20260731T081721Z/results.json"
)
CANDIDATE = (
    ROOT
    / "benchmark/raw/quality/layer0-relative-rope-score-fusion-v1-smoke-128-20260731/results.json"
)
RUN_CONTRACT = CANDIDATE.parent / "run_contract.json"
BOUNDED_NO_GO = (
    ROOT / "evidence/layer0_relative_rope_score_fusion_v1/latest/BOUNDED_NO_GO.json"
)
PROPOSAL = ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md"
RTL_MANIFEST = ROOT / "design/RTL_MANIFEST.json"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def expected_record() -> dict[str, Any]:
    evidence = load(CANDIDATE_EVIDENCE)
    baseline = load(BASELINE)
    candidate = load(CANDIDATE)
    run_contract = load(RUN_CONTRACT)
    boundary = load(BOUNDARY)

    require(evidence["contract_id"] == CONTRACT_ID, "candidate contract differs")
    require(sha256_file(PROPOSAL) == evidence["proposal_sha256"], "proposal hash differs")
    require(
        sha256_file(SOURCE_LIST)
        == evidence["source_binding"]["ordered_source_hash_list_sha256"],
        "candidate source-list hash differs",
    )
    require(boundary["status"] == "pass", "257-key boundary check did not pass")
    require("Ran 5 tests" in SOFTWARE_LOG.read_text(), "software test count differs")
    require(
        "ACE2_RELATIVE_ROPE_SCORE_FUSION_TB_PASS" in RTL_LOG.read_text(),
        "focused RTL completion marker is absent",
    )
    require(ARITH_LINT.stat().st_size == 0, "arithmetic lint emitted diagnostics")
    require(CENTER_LINT.stat().st_size == 0, "centering lint emitted diagnostics")

    accepted_rtl = candidate["artifacts"]["accepted_rtl"]
    require(
        accepted_rtl["candidate_id"] == evidence["candidate_id"],
        "paired smoke candidate ID differs",
    )
    require(
        accepted_rtl["candidate_rtl_hash"]
        == evidence["source_binding"]["ordered_source_hash_list_sha256"],
        "paired smoke source binding differs",
    )
    require(
        run_contract["candidate"]["candidate_id"] == evidence["candidate_id"],
        "run contract candidate ID differs",
    )
    require(
        candidate["input_observations"] == baseline["input_observations"],
        "paired smoke inputs are not comparable",
    )

    baseline_wiki = float(baseline["metrics"]["wikitext2"]["ratio"])
    baseline_c4 = float(baseline["metrics"]["c4_en_512"]["ratio"])
    candidate_wiki = float(candidate["metrics"]["wikitext2"]["ratio"])
    candidate_c4 = float(candidate["metrics"]["c4_en_512"]["ratio"])
    wiki_improved = candidate_wiki < baseline_wiki
    c4_improved = candidate_c4 < baseline_c4
    require(not (wiki_improved and c4_improved), "no-go binder cannot bind a passing gate")
    require(
        evidence["full_shell_regression_run"] is False
        and evidence["canonical_sky130_ppa_run"] is False
        and evidence["official_14_item_evaluation_run"] is False,
        "an expensive run was recorded before the smoke gate",
    )

    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "candidate_id": evidence["candidate_id"],
        "recorded_at_utc": candidate["generated_at_utc"],
        "status": "bounded_no_go_both_comparable_paired_smokes_regressed",
        "acceptance_claim": False,
        "stage_closing": False,
        "operator_targets_relaxed": False,
        "implementation_adequacy": {
            "coefficient_table_sha256": "62fd37a6e4dabc6abf89301dca6fb56cda61be329477155dbb7ffa02ca0fb325",
            "global_valid_key_max_257": artifact(BOUNDARY),
            "independent_scalar_tensor_tests": artifact(SOFTWARE_LOG),
            "rtl_arithmetic_and_centering_simulation": artifact(RTL_LOG),
            "rtl_arithmetic_lint": artifact(ARITH_LINT),
            "rtl_center_lint": artifact(CENTER_LINT),
            "source_binding": artifact(SOURCE_LIST),
        },
        "paired_smoke": {
            "baseline": artifact(BASELINE),
            "candidate": artifact(CANDIDATE),
            "run_contract": artifact(RUN_CONTRACT),
            "input_observations_match": True,
            "wikitext2": {
                "baseline_ratio": baseline_wiki,
                "candidate_ratio": candidate_wiki,
                "strict_improvement_required": True,
                "improved": wiki_improved,
                "regression_percent": 100.0 * (candidate_wiki / baseline_wiki - 1.0),
            },
            "c4_en_512": {
                "baseline_ratio": baseline_c4,
                "candidate_ratio": candidate_c4,
                "strict_improvement_required": True,
                "improved": c4_improved,
                "regression_percent": 100.0 * (candidate_c4 / baseline_c4 - 1.0),
            },
            "joint_gate_passed": False,
        },
        "expensive_run_lock": {
            "full_shell_regression_run": False,
            "canonical_sky130_ppa_run": False,
            "official_14_item_evaluation_run": False,
            "reason": "the mandatory comparable paired-smoke gate failed on both datasets",
        },
        "frontier_preservation": {
            "supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm",
                "layer_0.q_proj",
                "layer_0.k_proj",
                "layer_0.v_proj",
            ],
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "historical_non_sram_area_mm2": 0.6108746272,
            "historical_setup_slack_ns_at_100mhz": 0.1502,
            "candidate_ppa_claim": None,
        },
        "routing": {
            "current_stage": "rtl",
            "manager_stage_change_required": "rollback_to_architecture_before_selecting_a_distinct_replacement_contract",
            "planner_edited_pipeline_state": False,
            "required_operator_action": "approve_any_future_replacement_only_after_architecture_review",
        },
        "evidence": {
            "candidate_binding": artifact(CANDIDATE_EVIDENCE),
            "proposal": artifact(PROPOSAL),
        },
    }


def check_routing_state() -> None:
    manifest = load(RTL_MANIFEST)
    public = load(PUBLIC_STATUS)
    pipeline = load(PIPELINE_STATE)

    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(
        manifest.get("first_unsupported_layer_operator") == "layer_0.rope_q",
        "RTL manifest first unsupported operator differs",
    )
    require(
        manifest.get("architecture_selection_evidence", {}).get(
            "implementation_authorized"
        )
        is False,
        "RTL manifest still authorizes the consumed implementation",
    )
    traceability = manifest.get("traceability", {})
    require(
        traceability.get("architecture_contract_gap", {}).get("status")
        == "selected_contract_approval_consumed_candidate_rejected_requires_architecture_rollback",
        "RTL manifest architecture gap does not record the consumed no-go",
    )
    require(
        traceability.get("selected_mechanism")
        == "layer0_relative_rope_score_fusion_v1_rejected_no_authorized_replacement",
        "RTL manifest selected mechanism is stale",
    )
    rejected_modules = set(traceability.get("rejected_diagnostic_sources_traced", []))
    require(
        {
            "ace2_relative_rope_score_core",
            "ace2_global_score_center_core",
        }.issubset(rejected_modules),
        "RTL manifest omits rejected relative-RoPE modules from traceability",
    )

    public_stage = public.get("stage", {})
    require(public_stage.get("current_stage") == "rtl", "public stage differs")
    require(
        public_stage.get("current_stage_status")
        == "blocked_pending_manager_architecture_rollback_after_selected_contract_bounded_no_go",
        "public routing status differs",
    )
    require(
        public.get("selected_replacement_contract", {}).get(
            "implementation_authorized"
        )
        is False,
        "public status still authorizes the consumed implementation",
    )
    require(
        public.get("first_unsupported_layer_operator") == "layer_0.rope_q",
        "public first unsupported operator differs",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = expected_record()
    serialized = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if args.check:
        require(BOUNDED_NO_GO.is_file(), "bounded no-go evidence is missing")
        require(
            BOUNDED_NO_GO.read_text(encoding="utf-8") == serialized,
            "bounded no-go evidence differs from current hash-bound inputs",
        )
        check_routing_state()
        print("ACE2_RELATIVE_ROPE_SCORE_FUSION_NO_GO_CHECK status=pass")
        return
    BOUNDED_NO_GO.write_text(serialized, encoding="utf-8")
    print("ACE2_RELATIVE_ROPE_SCORE_FUSION_NO_GO status=bound")


if __name__ == "__main__":
    main()
