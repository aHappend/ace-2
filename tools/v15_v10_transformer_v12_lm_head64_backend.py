#!/usr/bin/env python3
"""Fresh V15 authority adapter with inherited-matrix restoration closure.

V13 and V14 official namespaces and construction markers are immutable. This
module reuses the corrected construction algorithm only after rebinding it to
the separately sealed V15 plan, task, namespace, and quality campaign. It also
repairs the V14 post-restoration matrix lookup without changing the inherited
V13/V14 acceptance contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import v13_v10_transformer_v12_lm_head64_backend as base


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v15"
TASK_ID = "task-c0a354f6ad25"
MISSION_ID = "successor-of-9f4ab32999cc-v15"

PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V15_PLAN.json"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V15_ENGINEER_TASK.json"
RUNNER_PATH = ROOT / "tools/run_option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v15_stage1.py"
ADAPTER_PATH = Path(__file__).resolve()
ALGORITHM_PATH = ROOT / "tools/v13_v10_transformer_v12_lm_head64_backend.py"

V13_PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V13_PLAN.json"
V13_PLAN_SHA256 = "aa3fc2797b27c2227588f2573d9aebe924fa385baee298ea99b69912c95a3603"
V14_PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V14_PLAN.json"
V14_PLAN_SHA256 = "85a9d5bf1127301b4ae384590a3baa859cfd11ee73bef09a464efbc21941185d"
V14_TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V14_ENGINEER_TASK.json"
V14_TASK_SHA256 = "59afe6316167302edb1ed68b46e09d22e16d9676daedb14fa6f6b96084c4b6ff"
V14_MARKER_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14/construction_started.json"
V14_MARKER_SHA256 = "7c0e2d86c20d0cce1a2f63c043854030afb6184eea595fff19d95cb42a34d8ed"
V14_SELF_TEST_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14/constructor_self_test.json"
V14_SELF_TEST_SHA256 = "bcacd9aa8b905a3fdd6d7dbf231f9d02d3e50922a14ae1354b8e56a5ea57c0c7"
V14_FROZEN_MATRIX_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14/quality-campaign-0001/frozen_matrix.json"
V14_FROZEN_MATRIX_SHA256 = "e4426dada0be9e7a25e72b598f865e78e3db7cc852083018ba8287e972d0ac99"
V14_NO_GO_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14/PRODUCT_PREFLIGHT_NO_GO.json"
V14_NO_GO_SHA256 = "b0cf069fd67802b77375d0fe987713e133bc2c12d990ea669d5fce56e93ad5fe"
V14_REVIEW_PATH = ROOT / "research/raw/specification/v14-product-quality-matrix-lookup-no-go-review-replan-20260807T070738Z.json"
V14_REVIEW_SHA256 = "685ab676b166c1fa7d7b26fc1c9920c4620bfeb186896c0354847dff9f117759"

OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v15"
QUALITY_DIR = OFFICIAL_ROOT / "quality-campaign-0001"

_ORIGINAL_VERIFY_FROZEN_AUTHORITY = base._verify_frozen_authority
_ORIGINAL_CANDIDATE_WORKER = base._candidate_worker
_ORIGINAL_FREEZE_QUALITY_MATRIX = base._freeze_quality_matrix
_ORIGINAL_LOAD_JSON = base.load_json
_ORIGINAL_COMBINED_SELF_TEST = base.combined_self_test


def _load_json_v15(path: Path) -> Any:
    """Expose the inherited V13 matrix through the restored V15 plan view."""

    value = _ORIGINAL_LOAD_JSON(path)
    if Path(path).resolve() != PLAN_PATH.resolve():
        return value
    inherited_matrix = _ORIGINAL_LOAD_JSON(V13_PLAN_PATH)["frozen_product_quality_matrix"]
    inherited_hash = base.canonical_sha256(inherited_matrix)
    expected_hash = value["inherited_v14_acceptance_contract"]["sections"]["frozen_product_quality_matrix"]
    base.require(inherited_hash == expected_hash, "V15 inherited matrix hash differs")
    base.require(
        inherited_hash == value["frozen_product_quality_matrix_source"]["canonical_sha256"],
        "V15 matrix source hash differs",
    )
    resolved = dict(value)
    resolved["frozen_product_quality_matrix"] = inherited_matrix
    return resolved


def _helper_paths_v15() -> dict[str, Path]:
    return {
        "v15_runner": RUNNER_PATH,
        "v15_authority_adapter": ADAPTER_PATH,
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


def _verify_frozen_authority_v15() -> dict[str, Any]:
    authority = _ORIGINAL_VERIFY_FROZEN_AUTHORITY()
    for path, digest in (
        (V13_PLAN_PATH, V13_PLAN_SHA256),
        (V14_PLAN_PATH, V14_PLAN_SHA256),
        (V14_TASK_PATH, V14_TASK_SHA256),
        (V14_MARKER_PATH, V14_MARKER_SHA256),
        (V14_SELF_TEST_PATH, V14_SELF_TEST_SHA256),
        (V14_FROZEN_MATRIX_PATH, V14_FROZEN_MATRIX_SHA256),
        (V14_NO_GO_PATH, V14_NO_GO_SHA256),
        (V14_REVIEW_PATH, V14_REVIEW_SHA256),
    ):
        base._verify_bound_file(path, digest)

    plan = base.load_json(PLAN_PATH)
    task = base.load_json(TASK_PATH)
    for record in plan["corrected_execution_binding"].values():
        base._verify_file_record(record)
    inherited = plan["inherited_v14_acceptance_contract"]
    base.require(inherited["path"] == str(V14_PLAN_PATH.relative_to(ROOT)), "V15 inherited plan path differs")
    base.require(inherited["sha256"] == V14_PLAN_SHA256, "V15 inherited plan hash differs")
    base.require(inherited["acceptance_changes"] == [], "V15 changed the V13/V14 acceptance contract")
    base.require(plan["terminal_predecessor"]["review_record_sha256"] == V14_REVIEW_SHA256, "V15 review binding differs")
    base.require(task["predecessor"]["terminal_status"] == "PRODUCT_PREFLIGHT_NO_GO", "V15 predecessor status differs")
    base.require(task["predecessor"]["quality_campaign_started"] is False, "V15 predecessor quality state differs")

    terminal = _ORIGINAL_LOAD_JSON(V14_NO_GO_PATH)
    review = _ORIGINAL_LOAD_JSON(V14_REVIEW_PATH)
    base.require(
        terminal["error_type"] == "KeyError" and terminal["error"] == "'frozen_product_quality_matrix'",
        "V14 failure record differs",
    )
    base.require(terminal["quality_campaign_started"] is False, "V14 quality campaign unexpectedly started")
    base.require(review["review_status"] == "replan_requested", "V14 Fresh Review routing differs")
    base.require(review["failure_taxonomy"] == "CONSTRUCTION_OR_CODEC_FAILURE", "V14 failure taxonomy differs")
    base.require(not OFFICIAL_ROOT.exists(), "fresh V15 namespace must remain absent before construction")

    authority.update(
        {
            "terminal_v14_plan_sha256": V14_PLAN_SHA256,
            "terminal_v14_task_sha256": V14_TASK_SHA256,
            "terminal_v14_construction_marker_sha256": V14_MARKER_SHA256,
            "terminal_v14_frozen_matrix_sha256": V14_FROZEN_MATRIX_SHA256,
            "terminal_v14_no_go_sha256": V14_NO_GO_SHA256,
            "v14_replan_review_sha256": V14_REVIEW_SHA256,
        }
    )
    return authority


def _candidate_worker_v15(output_dir: str) -> None:
    _configure()
    _ORIGINAL_CANDIDATE_WORKER(output_dir)


def _freeze_quality_matrix_v15(tokenizer: Any, core_manifest_sha256: str) -> dict[str, Any]:
    configured_plan = base.PLAN_PATH
    try:
        base.PLAN_PATH = V13_PLAN_PATH
        return _ORIGINAL_FREEZE_QUALITY_MATRIX(tokenizer, core_manifest_sha256)
    finally:
        base.PLAN_PATH = configured_plan


def _adapter_restoration_regression() -> dict[str, Any]:
    """Exercise the exact V14 failure path without creating official state."""

    base.require(not OFFICIAL_ROOT.exists(), "V15 regression found an official namespace")
    base.require(not (QUALITY_DIR / "execution_started.json").exists(), "V15 regression found a quality marker")
    configured_plan = base.PLAN_PATH
    tokenizer = base.AutoTokenizer.from_pretrained(base.SNAPSHOT, local_files_only=True, trust_remote_code=False)
    matrix = base._freeze_quality_matrix(tokenizer, "0" * 64)
    base.require(base.PLAN_PATH == configured_plan == PLAN_PATH, "V15 adapter did not restore the successor plan")
    visible_matrix = base.load_json(base.PLAN_PATH)["frozen_product_quality_matrix"]
    visible_matrix_sha256 = base.canonical_sha256(visible_matrix)
    expected_sha256 = base.load_json(PLAN_PATH)["frozen_product_quality_matrix_source"]["canonical_sha256"]
    base.require(visible_matrix_sha256 == expected_sha256, "V15 restored visible matrix hash differs")
    base.require(matrix["matrix_id"] == "ace2-v13-product-quality-v1", "V15 frozen matrix identity differs")
    base.require(not OFFICIAL_ROOT.exists(), "V15 regression created an official namespace")
    base.require(not (QUALITY_DIR / "execution_started.json").exists(), "V15 regression consumed quality authority")
    return {
        "status": "PASS",
        "successor_plan_restored": True,
        "exact_visible_matrix_lookup_exercised": True,
        "visible_matrix_sha256": visible_matrix_sha256,
        "frozen_matrix_id": matrix["matrix_id"],
        "response_count": matrix["response_count"],
        "official_namespace_created": False,
        "quality_marker_created": False,
        "claim_boundary": "non-consuming inherited-matrix adapter-restoration control-path regression only",
    }


def _combined_self_test_v15(*, allow_recovery: bool = False) -> dict[str, Any]:
    base.require(not allow_recovery, "V15 recovery is forbidden")
    result = _ORIGINAL_COMBINED_SELF_TEST(allow_recovery=False)
    regression = _adapter_restoration_regression()
    result["checks"]["inherited_matrix_adapter_restoration_regression"] = True
    result["adapter_restoration_regression"] = regression
    result["claim_boundary"] = (
        "synthetic constructor-kernel, frozen-input, and non-consuming inherited-matrix adapter-restoration "
        "regression only; no full-model, durable 312-event, product-quality, accelerator, RTL, synthesis, "
        "timing, area, or Fresh Reviewer conclusion"
    )
    return result


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
    base.load_json = _load_json_v15
    base._helper_paths = _helper_paths_v15
    base._verify_frozen_authority = _verify_frozen_authority_v15
    base._candidate_worker = _candidate_worker_v15
    base._freeze_quality_matrix = _freeze_quality_matrix_v15
    base.combined_self_test = _combined_self_test_v15


_configure()

require = base.require
combined_self_test = base.combined_self_test
verify_result = base.verify_result


def run_all() -> int:
    """Consume the fresh V15 construction authority exactly once."""

    _configure()
    return base.run_all(recover=False)
