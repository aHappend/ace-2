#!/usr/bin/env python3
"""Fail-closed audit for the ACE-2 same-architecture BF16 LoRA contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V1_CONTRACT.json"
COMPANION = CONTRACT.with_suffix(".sha256")
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
TRUNCATION_AUDIT = ROOT / "research/raw/specification/frozen-bf16-v20-generation-truncation-audit-20260807T171153Z.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, name: str, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise RuntimeError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    contract = load(CONTRACT)
    assert isinstance(contract, dict)
    companion_hash, companion_name = COMPANION.read_text(encoding="ascii").strip().split()
    require(companion_name == CONTRACT.name, "contract_companion_name_matches", checks)
    require(sha256_file(CONTRACT) == companion_hash, "contract_hash_matches", checks)
    require(
        contract["status"]
        == "OPERATOR_AUTHORIZED_SAME_ARCHITECTURE_WEIGHT_SUCCESSOR_128_TOKEN_SPEC_FROZEN",
        "contract_status_frozen",
        checks,
    )
    require(contract["authority"]["operator_authorization_verbatim"] == "全部授权", "operator_authorization_bound", checks)
    require(contract["architecture"]["architecture_change_allowed"] is False, "architecture_change_forbidden", checks)
    require(contract["pilot"]["attempt_count"] == 1, "single_pilot_attempt", checks)
    require(contract["pilot"]["weights_frozen_except_lora"] is True, "base_weights_frozen_during_pilot", checks)
    require(contract["acceptance"]["sealed_23_case_product_campaign"]["train_select_or_tune_allowed"] is False, "sealed_campaign_not_for_training_selection_or_tuning", checks)
    generation = contract["acceptance"]["generation"]
    require(generation["eos_stopped"] is True, "generation_stops_on_eos", checks)
    require(generation["primary_max_new_tokens"] == 128, "generation_primary_budget_128", checks)
    require(generation["max_new_tokens"] == 128, "generation_compatibility_budget_128", checks)
    require(generation["diagnostic_max_new_tokens"] == 256, "generation_diagnostic_budget_256", checks)
    require(
        generation["diagnostic_eligibility"]
        == "only a fresh response that reaches exactly 128 generated tokens without EOS",
        "generation_diagnostic_only_after_128_without_eos",
        checks,
    )
    require(generation["diagnostic_affects_primary_score"] is False, "diagnostic_does_not_replace_primary_score", checks)
    require(generation["historical_64_token_results_overwritten"] is False, "historical_64_results_preserved", checks)
    require(generation["product_semantics_may_hardcode_64"] is False, "product_semantics_not_hardcoded_to_64", checks)
    future_evaluation = contract["acceptance"]["future_product_evaluation"]
    require(future_evaluation["primary_comparable_max_new_tokens"] == 128, "future_product_primary_budget_128", checks)
    require(future_evaluation["diagnostic_256_only_after_primary_128_without_eos"] is True, "future_product_diagnostic_bounded", checks)
    require(future_evaluation["sealed_23_case_campaign_reuse_allowed"] is False, "sealed_campaign_reuse_forbidden", checks)
    sealed_campaign = contract["acceptance"]["sealed_23_case_product_campaign"]
    require(sealed_campaign["maximum_new_tokens"] == 64, "sealed_campaign_historical_budget_64", checks)
    require("immutable historical" in sealed_campaign["allowed_use"], "sealed_campaign_historical_only", checks)
    require(contract["stage_control"]["training_execution_in_this_bounded_node"] is False, "bounded_node_does_not_execute_training", checks)

    truncation_binding = contract["historical_truncation_audit"]
    require(sha256_file(TRUNCATION_AUDIT) == truncation_binding["audit_sha256"], "truncation_audit_hash_matches", checks)
    require(
        truncation_binding["audit_path"] == str(TRUNCATION_AUDIT.relative_to(ROOT)),
        "truncation_audit_path_matches",
        checks,
    )
    truncation = load(TRUNCATION_AUDIT)
    assert isinstance(truncation, dict)
    require(truncation["status"] == "PASS_ARTIFACT_ONLY_NO_REPLAY", "truncation_audit_pass", checks)
    require(truncation["routing_decision"]["single_authorized_sft_pilot_remains_necessary"] is True, "truncation_audit_keeps_single_sft_pilot", checks)
    require(truncation["routing_decision"]["additional_sft_scope_authorized_by_this_audit"] is False, "truncation_audit_does_not_expand_sft", checks)

    source = contract["source_model"]
    source_contract = ROOT / source["source_contract_path"]
    source_audit = ROOT / source["local_audit_path"]
    require(sha256_file(source_contract) == source["source_contract_sha256"], "source_contract_hash_matches", checks)
    require(sha256_file(source_audit) == source["local_audit_sha256"], "source_audit_hash_matches", checks)
    audit = load(source_audit)
    assert isinstance(audit, dict)
    snapshot = Path(audit["snapshot"]["path"])
    require(audit["status"] == "PASS_LOCAL_INSTRUCT_ARTIFACT_AVAILABLE", "source_audit_pass", checks)
    require(audit["required_revision"] == source["revision"], "source_revision_matches", checks)
    require(snapshot.name == source["revision"], "snapshot_directory_revision_matches", checks)
    for name, expected in source["files"].items():
        require(sha256_file(snapshot / name) == expected, f"source_file_hash_{name}", checks)
    config = load(snapshot / "config.json")
    assert isinstance(config, dict)
    for key in [
        "architectures",
        "hidden_size",
        "intermediate_size",
        "max_position_embeddings",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rope_theta",
        "tie_word_embeddings",
        "vocab_size",
    ]:
        require(config[key] == contract["architecture"][key], f"architecture_{key}", checks)

    dataset = contract["dataset"]
    generator = ROOT / dataset["generator_path"]
    manifest_path = ROOT / dataset["manifest_path"]
    require(sha256_file(generator) == dataset["generator_sha256"], "dataset_generator_hash_matches", checks)
    require(sha256_file(manifest_path) == dataset["manifest_sha256"], "dataset_manifest_hash_matches", checks)
    manifest = load(manifest_path)
    assert isinstance(manifest, dict)
    require(manifest["counts"] == {"dev": 56, "holdout": 56, "train": 280}, "dataset_counts_match", checks)
    require(manifest["integrity"]["normalized_input_hashes_unique"] is True, "dataset_inputs_unique", checks)
    require(manifest["integrity"]["template_families_disjoint"] is True, "dataset_template_families_disjoint", checks)
    require(manifest["integrity"]["sealed_campaign_prompt_hash_collisions"] == 0, "dataset_sealed_prompt_hash_collisions_zero", checks)
    for name, metadata in manifest["files"].items():
        path = manifest_path.parent / name
        require(path.stat().st_size == metadata["bytes"], f"dataset_file_size_{name}", checks)
        require(sha256_file(path) == metadata["sha256"], f"dataset_file_hash_{name}", checks)

    exclusion_path = ROOT / dataset["sealed_final_exclusion_path"]
    exclusion = load(exclusion_path)
    assert isinstance(exclusion, dict)
    sealed_path = ROOT / exclusion["sealed_campaign_path"]
    sealed = load(sealed_path)
    assert isinstance(sealed, dict)
    require(sha256_file(sealed_path) == exclusion["sealed_campaign_file_sha256"], "sealed_campaign_file_hash_matches", checks)
    require(sealed["matrix_content_sha256"] == exclusion["sealed_matrix_content_sha256"], "sealed_matrix_content_hash_matches", checks)
    require(sealed["response_count"] == 23, "sealed_response_count_23", checks)
    require(exclusion["prompt_text_inspected_by_this_freeze_tool"] is False, "freeze_tool_did_not_inspect_sealed_prompt_text", checks)

    pipeline = load(PIPELINE)
    assert isinstance(pipeline, dict)
    require(pipeline["current_stage"] == "specification", "pipeline_stage_specification", checks)
    terminal_v20 = load(
        ROOT
        / "build/stage1-option-b-post-v19-block16-affine-bf16-w4a8-v20"
        / "quality-campaign-0001/RESULT.json"
    )
    assert isinstance(terminal_v20, dict)
    require(terminal_v20["selected_policy_id"] is None, "selected_policy_id_null", checks)
    attempt_namespace = ROOT / contract["pilot"]["attempt_namespace"]
    require(not attempt_namespace.exists(), "pilot_attempt_namespace_absent", checks)

    result = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "contract_path": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256_file(CONTRACT),
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS",
        "checks": dict(sorted(checks.items())),
        "binding_next_action": "Fresh Reviewer evaluates this specification; if accepted, Manager routes the single A100 BF16 LoRA pilot without reopening model-choice approval.",
        "claim_boundary": "Specification and identity audit only; no training, holdout campaign, quantization, RTL, PPA, FPGA, U280, or product-quality result.",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "checks": len(checks), "output": str(output.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
