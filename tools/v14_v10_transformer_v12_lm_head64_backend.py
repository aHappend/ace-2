#!/usr/bin/env python3
"""Fresh V14 authority adapter for the corrected V13 construction algorithm.

V13's official namespace and first construction marker are immutable. This
module reuses the corrected, tensor-diagnosed algorithm only after rebinding it
to the separately sealed V14 plan, task, namespace, and quality campaign.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import v13_v10_transformer_v12_lm_head64_backend as base


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v14"
TASK_ID = "task-91ec703b42da"
MISSION_ID = "successor-of-9f4ab32999cc-v14"

PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V14_PLAN.json"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V14_ENGINEER_TASK.json"
RUNNER_PATH = ROOT / "tools/run_option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v14_stage1.py"
ADAPTER_PATH = Path(__file__).resolve()
ALGORITHM_PATH = ROOT / "tools/v13_v10_transformer_v12_lm_head64_backend.py"

V13_PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V13_PLAN.json"
V13_PLAN_SHA256 = "aa3fc2797b27c2227588f2573d9aebe924fa385baee298ea99b69912c95a3603"
V13_MARKER_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v13/construction_started.json"
V13_MARKER_SHA256 = "be54b2bfd782cbda0776c4f9c33ea557e134d72dc552d7273bdc96bb6e6c27d9"
V13_NO_GO_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v13/PRODUCT_PREFLIGHT_NO_GO.json"
V13_NO_GO_SHA256 = "8dcb494e89e1e199206cfcb337671bd1b9cb39454f23132697326aaf1ac7f90c"
V13_DIAGNOSTIC_PATH = ROOT / "build/v14-preplan-final-source-tensor1-diagnostic.json"
V13_DIAGNOSTIC_SHA256 = "fe0b47f7cbc529f19a5fd5a675477a03fc541fffc1ce330d220f2f6e7a40a54a"
V13_REVIEW_PATH = ROOT / "research/raw/specification/v13-product-quality-construction-no-go-review-replan-20260807T060033Z.json"
V13_REVIEW_SHA256 = "1299a3a26675f7ec82e76d12a38f3ec6bba43d9bc74a3759b7628fca1effdd64"

OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14"
QUALITY_DIR = OFFICIAL_ROOT / "quality-campaign-0001"

_ORIGINAL_VERIFY_FROZEN_AUTHORITY = base._verify_frozen_authority
_ORIGINAL_CANDIDATE_WORKER = base._candidate_worker
_ORIGINAL_FREEZE_QUALITY_MATRIX = base._freeze_quality_matrix


def _helper_paths_v14() -> dict[str, Path]:
    return {
        "v14_runner": RUNNER_PATH,
        "v14_authority_adapter": ADAPTER_PATH,
        "corrected_v13_algorithm_backend": ALGORITHM_PATH,
        "v10_runner": Path(base.v10.reference.__file__).resolve(),
        "v10_backend": Path(base.v10.__file__).resolve(),
        "v12_runner": Path(base.v12.reference.__file__).resolve(),
        "v12_backend": Path(base.v12.__file__).resolve(),
        "dynamic_policy_helper": Path(base.v10.v7.v3.__file__).resolve(),
        "grouped_scale32_helper": Path(base.v10.v7.grouped.__file__).resolve(),
        "source_identity_helper": ROOT / "tools/qwen_instruct_option_b.py",
        "generation_contract_helper": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
        "rtl_shell_unchanged_witness": ROOT / "rtl/ace2_shell.sv",
    }


def _verify_frozen_authority_v14() -> dict[str, Any]:
    authority = _ORIGINAL_VERIFY_FROZEN_AUTHORITY()
    base._verify_bound_file(V13_PLAN_PATH, V13_PLAN_SHA256)
    base._verify_bound_file(V13_MARKER_PATH, V13_MARKER_SHA256)
    base._verify_bound_file(V13_NO_GO_PATH, V13_NO_GO_SHA256)
    base._verify_bound_file(V13_DIAGNOSTIC_PATH, V13_DIAGNOSTIC_SHA256)
    base._verify_bound_file(V13_REVIEW_PATH, V13_REVIEW_SHA256)

    plan = base.load_json(PLAN_PATH)
    task = base.load_json(TASK_PATH)
    for record in plan["corrected_execution_binding"].values():
        base._verify_file_record(record)
    inherited = plan["inherited_v13_acceptance_contract"]
    base.require(inherited["path"] == str(V13_PLAN_PATH.relative_to(ROOT)), "V14 inherited plan path differs")
    base.require(inherited["sha256"] == V13_PLAN_SHA256, "V14 inherited plan hash differs")
    base.require(inherited["acceptance_changes"] == [], "V14 changed the V13 acceptance contract")
    base.require(plan["terminal_predecessor"]["review_record_sha256"] == V13_REVIEW_SHA256, "V14 review binding differs")
    base.require(task["predecessor"]["terminal_status"] == "PRODUCT_PREFLIGHT_NO_GO", "V14 predecessor status differs")
    base.require(task["predecessor"]["quality_campaign_started"] is False, "V14 predecessor quality state differs")

    terminal = base.load_json(V13_NO_GO_PATH)
    diagnostic = base.load_json(V13_DIAGNOSTIC_PATH)
    review = base.load_json(V13_REVIEW_PATH)
    base.require(terminal["error_type"] == "KeyError" and terminal["error"] == "'selector'", "V13 failure record differs")
    base.require(terminal["quality_campaign_started"] is False, "V13 quality campaign unexpectedly started")
    base.require(diagnostic["status"] == "PASS_PREPLAN_DIAGNOSTIC_ONLY", "corrected-source diagnostic status differs")
    base.require(diagnostic["v14_authority"]["construction_consumed"] is False, "V14 construction authority was consumed early")
    base.require(review["review_status"] == "replan_requested", "V13 Fresh Review routing differs")
    base.require(review["failure_taxonomy"] == "CONSTRUCTION_OR_CODEC_FAILURE", "V13 failure taxonomy differs")
    base.require(not OFFICIAL_ROOT.exists(), "fresh V14 namespace must remain absent before construction")

    authority.update(
        {
            "terminal_v13_plan_sha256": V13_PLAN_SHA256,
            "terminal_v13_construction_marker_sha256": V13_MARKER_SHA256,
            "terminal_v13_no_go_sha256": V13_NO_GO_SHA256,
            "corrected_source_tensor1_diagnostic_sha256": V13_DIAGNOSTIC_SHA256,
            "v13_replan_review_sha256": base.sha256_file(V13_REVIEW_PATH),
        }
    )
    return authority


def _candidate_worker_v14(output_dir: str) -> None:
    _configure()
    _ORIGINAL_CANDIDATE_WORKER(output_dir)


def _freeze_quality_matrix_v14(tokenizer: Any, core_manifest_sha256: str) -> dict[str, Any]:
    configured_plan = base.PLAN_PATH
    try:
        base.PLAN_PATH = V13_PLAN_PATH
        return _ORIGINAL_FREEZE_QUALITY_MATRIX(tokenizer, core_manifest_sha256)
    finally:
        base.PLAN_PATH = configured_plan


def _configure() -> None:
    base.CANDIDATE_ID = CANDIDATE_ID
    base.MISSION_ID = MISSION_ID
    base.TASK_ID = TASK_ID
    base.PLAN_PATH = PLAN_PATH
    base.PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
    base.TASK_PATH = TASK_PATH
    base.TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
    base.RUNNER_PATH = RUNNER_PATH
    base.BACKEND_PATH = ADAPTER_PATH
    base.OFFICIAL_ROOT = OFFICIAL_ROOT
    base.CONSTRUCTION_MARKER = OFFICIAL_ROOT / "construction_started.json"
    base.CONSTRUCTION_RECOVERY_MARKER = OFFICIAL_ROOT / "construction_recovery_started.json"
    base.CONSTRUCTOR_SELF_TEST_PATH = OFFICIAL_ROOT / "constructor_self_test.json"
    base.CONSTRUCTOR_RECOVERY_SELF_TEST_PATH = OFFICIAL_ROOT / "constructor_self_test_recovery.json"
    base.CANDIDATE_A_PATH = OFFICIAL_ROOT / "candidate_a_manifest.json"
    base.CANDIDATE_B_PATH = OFFICIAL_ROOT / "candidate_b_manifest.json"
    base.ALIAS_WITNESS_PATH = OFFICIAL_ROOT / "alias_separation_witness.json"
    base.TRANSFORMER_REPRODUCTION_PATH = OFFICIAL_ROOT / "transformer_v10_exact_reproduction.json"
    base.LM_HEAD_REPRODUCTION_PATH = OFFICIAL_ROOT / "lm_head_v12_exact_reproduction.json"
    base.ACCOUNTING_PATH = OFFICIAL_ROOT / "codec_and_byte_accounting.json"
    base.DYNAMIC_PATH = OFFICIAL_ROOT / "dynamic_preflight.json"
    base.CONTRACT_PATH = OFFICIAL_ROOT / "predeclared_contract.json"
    base.PREFLIGHT_READY_PATH = OFFICIAL_ROOT / "PRODUCT_PREFLIGHT_READY.json"
    base.PREFLIGHT_NO_GO_PATH = OFFICIAL_ROOT / "PRODUCT_PREFLIGHT_NO_GO.json"
    base.QUALITY_DIR = QUALITY_DIR
    base.FROZEN_MATRIX_PATH = QUALITY_DIR / "frozen_matrix.json"
    base.QUALITY_MARKER = QUALITY_DIR / "execution_started.json"
    base.RAW_OUTPUTS_PATH = QUALITY_DIR / "raw_outputs.jsonl"
    base.TOKEN_LATENCY_PATH = QUALITY_DIR / "token_and_latency_results.json"
    base.RUBRIC_PATH = QUALITY_DIR / "rubric_results.json"
    base.QUALITY_RESULT_PATH = QUALITY_DIR / "RESULT.json"
    base.QUALITY_SUMS_PATH = QUALITY_DIR / "SHA256SUMS"
    base.FRESH_REVIEWER_SUBMISSION_PATH = OFFICIAL_ROOT / "fresh_reviewer_submission.json"
    base._helper_paths = _helper_paths_v14
    base._verify_frozen_authority = _verify_frozen_authority_v14
    base._candidate_worker = _candidate_worker_v14
    base._freeze_quality_matrix = _freeze_quality_matrix_v14


_configure()

require = base.require
combined_self_test = base.combined_self_test
verify_result = base.verify_result


def run_all() -> int:
    """Consume the fresh V14 construction authority exactly once."""

    _configure()
    return base.run_all(recover=False)
