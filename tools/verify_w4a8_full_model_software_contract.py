#!/usr/bin/env python3
"""Verify the frozen base-vs-checkpoint176 W4A8 software contract."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "design/QWEN25_05B_W4A8_FULL_MODEL_SOFTWARE_CONTRACT_V1.json"
CONTRACT_SHA = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    / REVISION
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verify_record(record: dict[str, Any], label: str) -> None:
    require(set(record) == {"path", "sha256"}, f"{label} record fields differ")
    path = ROOT / record["path"]
    require(path.is_file(), f"{label} is missing: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"{label} hash differs")


def punctuation_only(value: str) -> bool:
    visible = [character for character in value if not character.isspace()]
    return bool(visible) and all(
        unicodedata.category(character).startswith(("P", "S")) for character in visible
    )


def verify_quality_semantics(contract: dict[str, Any]) -> None:
    evaluation = contract["evaluation_contract"]
    base = contract["model_identities"]["qwen2.5-0.5b-instruct"]
    checkpoint = contract["model_identities"]["ace2_lora_v4_checkpoint176"]
    config = load_json(ROOT / contract["frozen_references"]["quality_config"]["path"])
    manifest = load_json(
        ROOT / contract["frozen_references"]["quality_prompt_manifest"]["path"]
    )

    expected_primary_model = {
        "repository": base["repository"],
        "revision": base["revision"],
    }
    require(manifest["model"] == expected_primary_model, "quality manifest model identity differs")
    require(
        {
            "repository": config["baseline"]["model_repository"],
            "revision": config["baseline"]["model_revision"],
        }
        == expected_primary_model,
        "quality config model identity differs",
    )
    expected_model_scope = {
        "comparison": "each_contract_model_compared_only_to_its_own_bf16_control",
        "model_ids": [base["model_id"], checkpoint["model_id"]],
    }
    require(manifest["model_scope"] == expected_model_scope, "quality manifest model scope differs")
    require(
        config["baseline"]["comparison_scope"] == expected_model_scope["comparison"]
        and config["baseline"]["model_ids"] == expected_model_scope["model_ids"],
        "quality config model scope differs",
    )
    require(config["baseline"]["dtype"] == "bfloat16", "quality BF16 dtype differs")

    calibration = evaluation["calibration_set"]
    calibration_manifest = manifest["datasets"]["c4_calibration"]
    require(
        {
            "config": calibration_manifest["config"],
            "record_indices": calibration_manifest["indices"],
            "repository": calibration_manifest["repository"],
            "revision": calibration_manifest["revision"],
            "split": calibration_manifest["split"],
            "token_limit": calibration_manifest["token_limit"],
        }
        == {
            key: calibration[key]
            for key in (
                "config",
                "record_indices",
                "repository",
                "revision",
                "split",
                "token_limit",
            )
        },
        "quality calibration set differs from evaluation_contract",
    )
    require(
        calibration_manifest["field"] == "text"
        and config["activation_quantization"]["calibration_dataset"] == "c4_calibration",
        "quality calibration binding differs",
    )

    bounded = evaluation["bounded_quality_sets"]
    c4 = bounded["c4_perplexity"]
    c4_manifest = manifest["datasets"]["c4_en_512"]
    require(
        {
            "config": c4_manifest["config"],
            "record_indices": c4_manifest["indices"],
            "repository": c4_manifest["repository"],
            "revision": c4_manifest["revision"],
            "sequence_length": c4_manifest["token_limit"],
            "split": c4_manifest["split"],
        }
        == c4,
        "quality C4 set differs from evaluation_contract",
    )
    c4_config = config["evaluation"]["c4_en_512"]
    require(
        c4_config["batch_size"] == 1
        and c4_config["prompt_count"] == c4["record_indices"]["stop"] - c4["record_indices"]["start"]
        and c4_config["sequence_length"] == c4["sequence_length"],
        "quality C4 bounds differ",
    )

    wiki = bounded["wikitext2_perplexity"]
    wiki_manifest = manifest["datasets"]["wikitext2"]
    require(
        {
            "config": wiki_manifest["config"],
            "first_complete_windows": wiki_manifest["first_complete_windows"],
            "repository": wiki_manifest["repository"],
            "revision": wiki_manifest["revision"],
            "sequence_length": wiki_manifest["token_limit"],
            "split": wiki_manifest["split"],
        }
        == {
            key: wiki[key]
            for key in (
                "config",
                "first_complete_windows",
                "repository",
                "revision",
                "sequence_length",
                "split",
            )
        },
        "quality WikiText set differs from evaluation_contract",
    )
    wiki_config = config["evaluation"]["wikitext2"]
    require(
        wiki_config
        == {
            "first_complete_windows": wiki["first_complete_windows"],
            "sequence_length": wiki["sequence_length"],
            "stride": wiki["stride"],
        },
        "quality WikiText bounds differ",
    )

    lm_eval = bounded["lm_eval"]
    require(manifest["lm_eval"]["fewshot"] == lm_eval["fewshot"], "lm-eval fewshot differs")
    require(manifest["lm_eval"]["limit"] == lm_eval["limit_per_task"], "lm-eval limit differs")
    require(
        set(manifest["lm_eval"]["tasks"]) == set(lm_eval["tasks"]),
        "lm-eval task set differs",
    )
    lm_eval_config = config["evaluation"]["lm_eval"]
    require(
        lm_eval_config["limit_per_task"] == lm_eval["limit_per_task"]
        and lm_eval_config["metric"] == lm_eval["metric"],
        "quality lm-eval config differs from evaluation_contract",
    )

    gate = evaluation["quality_gate"]
    require(
        config["acceptance_thresholds"]
        == {
            "applies_independently_to_both_models": gate["applies_independently_to_both_models"],
            "c4_en_512_perplexity_ratio_max": gate[
                "c4_perplexity_ratio_to_model_specific_bf16_max"
            ],
            "lm_eval_average_normalized_accuracy_drop_percentage_points_max": gate[
                "lm_eval_average_normalized_accuracy_drop_percentage_points_max"
            ],
            "lm_eval_individual_task_drop_percentage_points_max": gate[
                "lm_eval_individual_task_drop_percentage_points_max"
            ],
            "mission_local_not_project_final_emnlp_gate": gate[
                "mission_local_not_project_final_emnlp_gate"
            ],
            "wikitext2_perplexity_ratio_max": gate[
                "wikitext2_perplexity_ratio_to_model_specific_bf16_max"
            ],
        },
        "quality thresholds differ from evaluation_contract",
    )


def main() -> None:
    require(CONTRACT.is_file(), "software contract is missing")
    raw = CONTRACT.read_bytes()
    contract = load_json(CONTRACT)
    canonical = (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode()
    require(raw == canonical, "software contract is not canonical sorted JSON")
    require(CONTRACT_SHA.is_file(), "software contract SHA256 companion is missing")
    sha_fields = CONTRACT_SHA.read_text(encoding="utf-8").strip().split()
    require(len(sha_fields) == 2, "software contract SHA256 companion differs")
    require(sha_fields[0] == sha256_file(CONTRACT), "software contract SHA256 differs")
    require(sha_fields[1] == CONTRACT.name, "software contract SHA256 filename differs")

    require(contract.get("schema_version") == 2, "software contract schema differs")
    require(
        contract.get("contract_id")
        == "qwen2.5-0.5b-base-vs-ace2-lora-v4-checkpoint176-w4a8-full-model-software-v2",
        "software contract identity differs",
    )
    require(
        contract.get("status") == "FROZEN_PRE_PTQ_V2_CONTROLS_PENDING_FRESH_L2",
        "software contract is not the corrected v2 pre-PTQ freeze",
    )
    serialized = raw.decode("utf-8")
    require("/home/" not in serialized, "software contract exposes an absolute home path")

    guards = contract["scope_guards"]
    require(guards["official_generation_attempts_consumed"] == 0, "official attempt was consumed")
    require(not guards["complete_rtl_integration_authorized"], "RTL integration was authorized")
    require(not guards["retraining_authorized"], "retraining was authorized")
    require(not guards["u280_authorized"], "U280 work was authorized")
    require(not guards["stage1_completion_claimed"], "Stage-1 completion was claimed")
    require(
        not guards["ptq_candidate_execution_authorized"],
        "PTQ candidate execution was authorized before Fresh-L2 acceptance",
    )

    for name, record in contract["frozen_references"].items():
        verify_record(record, f"frozen reference {name}")

    base = contract["model_identities"]["qwen2.5-0.5b-instruct"]
    require(base["revision"] == REVISION, "base revision differs")
    require(SNAPSHOT.is_dir(), "pinned base snapshot is missing")
    for filename, expected in base["files"].items():
        path = SNAPSHOT / filename
        require(path.is_file(), f"pinned base file is missing: {filename}")
        require(sha256_file(path) == expected, f"pinned base file differs: {filename}")
    tokenizer_config = load_json(SNAPSHOT / "tokenizer_config.json")
    template = tokenizer_config.get("chat_template")
    require(isinstance(template, str), "pinned chat template is missing")
    template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    require(template_sha == base["chat_template_sha256"], "pinned chat template differs")
    source_config = load_json(SNAPSHOT / "config.json")
    expected_config = base["config"]
    observed_config = {
        "head_dim": source_config["hidden_size"] // source_config["num_attention_heads"],
        "hidden_size": source_config["hidden_size"],
        "intermediate_size": source_config["intermediate_size"],
        "max_position_embeddings": source_config["max_position_embeddings"],
        "num_attention_heads": source_config["num_attention_heads"],
        "num_hidden_layers": source_config["num_hidden_layers"],
        "num_key_value_heads": source_config["num_key_value_heads"],
        "rms_norm_eps": source_config["rms_norm_eps"],
        "rope_theta": source_config["rope_theta"],
        "tie_word_embeddings": source_config["tie_word_embeddings"],
        "vocab_size": source_config["vocab_size"],
    }
    require(observed_config == expected_config, "pinned base geometry differs")

    lora = contract["model_identities"]["ace2_lora_v4_checkpoint176"]["adapter"]
    verify_record(lora["adapter_config"], "checkpoint-176 adapter config")
    verify_record(lora["adapter_model"], "checkpoint-176 adapter model")
    adapter_config = load_json(ROOT / lora["adapter_config"]["path"])
    require(adapter_config["r"] == lora["lora_rank"], "checkpoint-176 LoRA rank differs")
    require(adapter_config["lora_alpha"] == lora["lora_alpha"], "checkpoint-176 LoRA alpha differs")
    require(adapter_config["peft_version"] == lora["peft_version"], "checkpoint-176 PEFT version differs")
    require(sorted(adapter_config["target_modules"]) == lora["target_modules"], "checkpoint-176 target modules differ")
    require(adapter_config["use_qalora"] is False, "checkpoint-176 unexpectedly uses QALoRA")

    training_record = contract["frozen_references"]["v4_training_result"]
    training_result = load_json(ROOT / training_record["path"])
    checkpoint = training_result["checkpoint_manifests"]["4"]
    require(checkpoint["tree_sha256"] == lora["checkpoint_tree_sha256"], "checkpoint-176 tree identity differs")
    require(
        checkpoint["files"]["adapter_config.json"]["sha256"] == lora["adapter_config"]["sha256"],
        "training record adapter-config identity differs",
    )
    require(
        checkpoint["files"]["adapter_model.safetensors"]["sha256"] == lora["adapter_model"]["sha256"],
        "training record adapter-model identity differs",
    )

    artifact_policy = contract["artifact_policy"]
    require(artifact_policy["model_artifacts_generated_independently"], "independent artifacts are not required")
    require(artifact_policy["base_root"] != artifact_policy["checkpoint176_root"], "model artifact roots alias")
    require(artifact_policy["candidate_attempts_per_model"] == 1, "attempt bound differs")
    require(
        artifact_policy["bf16_control_namespace"] == "bf16-control-v2",
        "corrected BF16 control namespace differs",
    )
    require(
        artifact_policy["legacy_bf16_control_namespace"] == "bf16-control",
        "legacy BF16 control namespace differs",
    )
    required_forbidden = {
        "packed_w4_weights",
        "weight_scale32",
        "activation_scale32",
        "operator_scale32",
        "kv_cache_scale32",
        "calibration_observations",
    }
    require(set(artifact_policy["forbidden_cross_model_reuse"]) == required_forbidden, "cross-model reuse guard differs")

    candidates = contract["candidate_matrix"]
    require(len(candidates) == 4, "candidate bound differs")
    require([item["ordered_index"] for item in candidates] == list(range(4)), "candidate order differs")
    require(
        [item["candidate_id"] for item in candidates]
        == [
            "c00-rtn-absmax",
            "c01-mse-clip-grid",
            "c02-qqq-block-hessian-round",
            "c03-qserve-progressive-block-reconstruction",
        ],
        "candidate identities differ",
    )
    require(
        contract["stopping_rules"]["maximum_genuine_candidate_model_attempts"]
        == len(candidates) * 2,
        "candidate/model attempt bound differs",
    )

    shared = contract["shared_w4a8_contract"]
    require(shared["weight_w4"]["integer_range"] == [-8, 7], "W4 range differs")
    require(shared["weight_w4"]["packing_group_size"] == 128, "W4 group differs")
    require(shared["activation_a8"]["integer_range"] == [-128, 127], "A8 range differs")
    require(shared["accumulation"]["width"] == 32, "accumulator width differs")
    require(shared["scale32"]["significand_range"] == [32768, 65535], "Scale32 significand differs")
    require(shared["scale32"]["exponent_range"] == [-24, 4], "Scale32 exponent differs")
    require(shared["kv_cache"]["width"] == 8, "KV-cache width differs")
    require(shared["generation"]["use_cache"] is True, "production decode does not use KV cache")
    require(shared["generation"]["sampling"] is False, "generation is not greedy")
    require(shared["generation"]["max_new_tokens"] == 32, "generation limit differs")

    evaluation = contract["evaluation_contract"]
    verify_quality_semantics(contract)
    bf16_control = evaluation["bf16_control"]
    require(
        bf16_control["status"] == "PENDING_V2_EXECUTION",
        "corrected BF16 control state differs",
    )
    require(
        bf16_control["cache_equivalence_fields"]
        == ["generated_token_ids", "per_step_selected_token", "termination_reason"],
        "BF16 cache-equivalence fields differ",
    )
    require(
        bf16_control["cache_logit_comparison"]
        == "enforce_0p25_bound_as_a_separate_numerical_diagnostic_with_only_the_frozen_FP32_exoneration_while_semantic_equivalence_requires_exact_selected_tokens_and_termination",
        "BF16 cache-logit diagnostic policy differs",
    )
    require(
        bf16_control["semantic_equivalence_rule"]
        == "every_cache_step_selected_token_must_equal_full_prefix_and_duplicate_replays_must_match_generated_tokens_selected_tokens_and_termination",
        "BF16 semantic-equivalence rule differs",
    )
    require(
        bf16_control["bf16_logit_max_abs_difference_bound"] == 0.25,
        "BF16 numerical diagnostic bound differs",
    )
    require(
        bf16_control["fp32_exonerated_models"] == ["base"],
        "BF16 FP32-exonerated model set differs",
    )
    require(
        bf16_control["fp32_required_classification"] == "FP32_NUMERICALLY_CONSISTENT",
        "required FP32 diagnostic classification differs",
    )
    require(
        bf16_control["allowed_numerical_diagnostic_classifications"]
        == [
            "BF16_NUMERICAL_DIAGNOSTIC_WITHIN_BOUND",
            "BF16_NUMERICAL_DIAGNOSTIC_BOUND_EXCEEDED_FP32_EXONERATED",
        ],
        "BF16 numerical diagnostic classifications differ",
    )
    require(
        bf16_control["ptq_unlock_requires_fresh_l2_acceptance"],
        "Fresh-L2 PTQ unlock requirement is missing",
    )

    fp32_record = contract["frozen_references"]["fp32_cache_diagnostic"]
    fp32_evidence = load_json(ROOT / fp32_record["path"])
    require(fp32_evidence["status"] == "PASS", "FP32 cache diagnostic did not pass")
    require(
        fp32_evidence["classification"] == "FP32_NUMERICALLY_CONSISTENT",
        "FP32 cache diagnostic classification differs",
    )
    require(
        fp32_evidence["source_bf16_worst_step"]
        == {
            "logit_max_abs_difference": 0.40625,
            "logit_mismatch_count": 131985,
            "prompt_id": "g03-three-lines",
            "step_index": 16,
        },
        "FP32 diagnostic does not bind the Manager-selected BF16 worst step",
    )

    containment_record = contract["frozen_references"]["round3_ptq_containment"]
    containment = load_json(ROOT / containment_record["path"])
    require(containment["status"] == "PASS", "round-3 PTQ containment did not pass")
    require(
        containment["legacy_bf16_controls"]["base"]["classification"]
        == "INVALID_PASS_MISSING_THRESHOLD_ENFORCEMENT",
        "legacy base BF16 false pass was not contained",
    )
    require(
        containment["fixed_point_source"]["after_restore_sha256"]
        == contract["frozen_references"]["fixed_point_reference"]["sha256"],
        "round-3 fixed-point restoration does not match the frozen reference",
    )
    prompts = evaluation["canonical_generation_prompts"]
    require(len(prompts) == 8, "canonical prompt count differs")
    require(len({item["prompt_id"] for item in prompts}) == len(prompts), "canonical prompt IDs alias")
    require(all(item["user"].strip() for item in prompts), "canonical prompt is empty")
    gate = evaluation["quality_gate"]
    require(gate["applies_independently_to_both_models"], "quality gate is not per model")
    require(gate["mission_local_not_project_final_emnlp_gate"], "quality gate scope differs")

    negatives = contract["negative_evidence"]
    require(len(negatives) == 2, "Option-B negative evidence count differs")
    for index, record in enumerate(negatives):
        verify_record({"path": record["path"], "sha256": record["sha256"]}, f"negative evidence {index}")
        source = load_json(ROOT / record["path"])
        require(source.get("decoded_text") == record["decoded_text"], f"negative evidence {index} text differs")
        require(punctuation_only(record["decoded_text"]), f"negative evidence {index} is not punctuation collapse")
        require(record["classification"].startswith("REJECTED_OPTION_B_"), f"negative evidence {index} is not rejected")

    print(
        "ACE2_W4A8_FULL_MODEL_SOFTWARE_CONTRACT_PASS "
        f"contract_sha256={sha256_file(CONTRACT)} candidates={len(candidates)} models=2 prompts={len(prompts)}"
    )


if __name__ == "__main__":
    main()
