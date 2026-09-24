#!/usr/bin/env python3
"""Fail-closed, non-consuming audit of the frozen BF16 LoRA V2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"
DATASET_DIR = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2"
MANIFEST = DATASET_DIR / "dataset_manifest.json"
TRAIN = DATASET_DIR / "train.jsonl"
RECIPE = DATASET_DIR / "training_recipe.json"
ENVIRONMENT = DATASET_DIR / "environment.lock.json"
GENERATOR = ROOT / "tools/freeze_qwen25_05b_bf16_lora_product_v2.py"
BLINDED_SCANNER = ROOT / "tools/run_qwen25_05b_bf16_lora_product_v2_blinded_cross_split_scan.py"
BLINDED_SIDECAR = (
    ROOT
    / "research/raw/specification"
    / "qwen25-05b-instruct-bf16-lora-product-v2-cross-split-blinded-sidecar.json"
)
V1_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V1_CONTRACT.json"
SOURCE_CONTRACT = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
SEALED_EXCLUSION = (
    ROOT
    / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1"
    / "sealed_final_exclusion.json"
)
EXPECTED_ATTESTATION_METHOD = {
    "answer_fields_accessed": False,
    "human_exposure_to_evaluation_text": False,
    "method_id": "ace2-blinded-cross-split-leakage-v1",
    "near_duplicate_metric": "normalized-token-five-gram-jaccard",
    "normalization": "NFKC-casefold-collapse-whitespace",
    "prompt_only_access": True,
    "raw_prompts_or_fingerprints_emitted": False,
    "template_family_method": "normalized-lexical-skeleton-v1",
}
EXPECTED_OPERATOR_APPROVAL_REFERENCE = "Manager operator-answer decision 2026-08-07T20:18Z"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def five_grams(value: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[\w]+|[^\w\s]", normalized_text(value), flags=re.UNICODE)
    if len(tokens) < 5:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + 5]) for index in range(len(tokens) - 4)}


def maximum_jaccard(rows: list[dict[str, Any]]) -> tuple[float, tuple[str, str]]:
    prompts = [(row["id"], row["messages"][-2]["content"]) for row in rows]
    grams = {identifier: five_grams(prompt) for identifier, prompt in prompts}
    maximum = 0.0
    pair = ("", "")
    for index, (left_id, _) in enumerate(prompts):
        left = grams[left_id]
        for right_id, _ in prompts[index + 1 :]:
            right = grams[right_id]
            union = left | right
            score = len(left & right) / len(union) if union else 1.0
            if score > maximum:
                maximum = score
                pair = (left_id, right_id)
    return maximum, pair


def prompt_hashes(user_text: str, input_messages: list[dict[str, str]]) -> set[str]:
    canonical_messages = json.dumps(
        input_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    values = {
        user_text,
        normalized_text(user_text),
        canonical_messages,
        normalized_text(canonical_messages),
    }
    return {sha256_bytes(value.encode("utf-8")) for value in values}


def require(condition: bool, name: str, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise RuntimeError(name)


def verify_companion(path: Path, checks: dict[str, bool], label: str) -> None:
    companion = path.with_suffix(".sha256")
    expected_hash, expected_name = companion.read_text(encoding="ascii").split()
    require(expected_name == path.name, f"{label}_companion_name", checks)
    require(sha256_file(path) == expected_hash, f"{label}_companion_hash", checks)


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(".sha256")
    if not companion.is_file():
        return False
    try:
        expected_hash, expected_name = companion.read_text(encoding="ascii").split()
    except (OSError, ValueError):
        return False
    return expected_name == path.name and expected_hash == sha256_file(path)


def blinded_sidecar_shape_is_aggregate_only(sidecar: dict[str, Any]) -> bool:
    if set(sidecar) != {
        "aggregate_collision_counts",
        "algorithm",
        "fingerprint_sets",
        "maximum_train_to_evaluation_five_gram_jaccard",
        "scanner",
        "schema_version",
        "source_bindings",
        "status",
    }:
        return False
    if sidecar.get("schema_version") != 1:
        return False
    fingerprint_sets = sidecar.get("fingerprint_sets")
    if not isinstance(fingerprint_sets, dict) or set(fingerprint_sets) != {"dev", "holdout", "train"}:
        return False
    allowed_commitments = {
        "five_gram_fingerprints",
        "lexical_template_fingerprints",
        "normalized_prompt_fingerprints",
        "record_count",
        "template_identifier_fingerprints",
    }
    for split_value in fingerprint_sets.values():
        if not isinstance(split_value, dict) or set(split_value) != allowed_commitments:
            return False
        for key, commitment in split_value.items():
            if key == "record_count":
                if not isinstance(commitment, int):
                    return False
            elif not (
                isinstance(commitment, dict)
                and set(commitment) == {"count", "set_sha256"}
                and isinstance(commitment["count"], int)
                and isinstance(commitment["set_sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", commitment["set_sha256"])
            ):
                return False
    return True


def generator_reproduces_freeze() -> bool:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(GENERATOR), "--check"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False
    return result == {
        "artifact_count": 10,
        "dataset_id": "qwen2.5-0.5b-instruct-bf16-lora-product-v2",
        "mode": "check",
        "status": "PASS",
        "training_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    for label, path in [
        ("contract", CONTRACT),
        ("manifest", MANIFEST),
        ("train", TRAIN),
        ("recipe", RECIPE),
        ("environment", ENVIRONMENT),
    ]:
        require(path.is_file(), f"{label}_exists", checks)
        verify_companion(path, checks, label)

    contract = load_json(CONTRACT)
    manifest = load_json(MANIFEST)
    recipe = load_json(RECIPE)
    environment = load_json(ENVIRONMENT)
    rows = load_jsonl(TRAIN)

    require(
        contract["status"] == "FROZEN_CROSS_SPLIT_ATTESTATION_REQUIRED_NO_TRAINING",
        "contract_status_cross_split_attestation_required",
        checks,
    )
    require(contract["contract_id"] == recipe["contract_id"], "contract_recipe_identity", checks)
    require(contract["dataset"]["dataset_id"] == manifest["dataset_id"], "contract_manifest_identity", checks)
    require(
        contract["dataset"]["generator_path"] == str(GENERATOR.relative_to(ROOT))
        and contract["dataset"]["generator_sha256"] == sha256_file(GENERATOR),
        "generator_identity_bound",
        checks,
    )
    require(generator_reproduces_freeze(), "generator_reproduction_pass", checks)
    require(contract["dataset"]["manifest_sha256"] == sha256_file(MANIFEST), "manifest_hash_bound", checks)
    require(contract["recipe"]["sha256"] == sha256_file(RECIPE), "recipe_hash_bound", checks)
    require(contract["recipe"]["environment_lock_sha256"] == sha256_file(ENVIRONMENT), "environment_hash_bound", checks)

    require(
        manifest["status"] == "FROZEN_TRAIN_ONLY_CROSS_SPLIT_ATTESTATION_REQUIRED",
        "manifest_status_frozen",
        checks,
    )
    require(manifest["construction"]["dev_answer_text_accessed"] is False, "dev_answers_not_accessed", checks)
    require(manifest["construction"]["holdout_records_parsed"] is False, "holdout_records_not_parsed", checks)
    require(manifest["construction"]["v1_training_rows_accessed"] is False, "v1_training_rows_not_accessed", checks)
    require(manifest["construction"]["external_model_outputs"] is False, "no_external_model_outputs", checks)
    require(len(rows) == 216, "train_count_216", checks)
    require(contract["dataset"]["train_count"] == len(rows), "contract_train_count", checks)
    require(manifest["counts"]["train"] == len(rows), "manifest_train_count", checks)

    expected_family_counts = {
        "composition": 32,
        "hard_instruction": 32,
        "logic": 32,
        "non_answer_replacement": 40,
        "safe_refusal": 48,
        "strict_format": 32,
    }
    family_counts = dict(sorted(Counter(row["family"] for row in rows).items()))
    require(family_counts == expected_family_counts, "fresh_family_counts", checks)
    require(manifest["family_counts"] == expected_family_counts, "manifest_family_counts", checks)
    require(
        family_counts["safe_refusal"] + family_counts["non_answer_replacement"] >= 88,
        "substantial_refusal_and_nonanswer_family",
        checks,
    )

    identifiers: list[str] = []
    normalized_prompt_hashes: list[str] = []
    template_prefixes: dict[str, set[str]] = {}
    for row in rows:
        require(set(row) == {"family", "id", "messages", "provenance", "template_id"}, "row_schema_exact", checks)
        identifiers.append(row["id"])
        messages = row["messages"]
        require([item["role"] for item in messages] == ["system", "user", "assistant"], "message_roles_exact", checks)
        require(all(isinstance(item["content"], str) and item["content"].strip() for item in messages), "message_content_nonempty", checks)
        provenance = row["provenance"]
        require(provenance["authoring"] == "deterministic_project_authored_v2_generator", "row_project_authored", checks)
        require(provenance["external_model_output"] is False, "row_not_external_model_output", checks)
        require(provenance["source_split"] == "none", "row_no_source_split", checks)
        prompt = messages[-2]["content"]
        normalized_prompt_hashes.append(sha256_bytes(normalized_text(prompt).encode("utf-8")))
        prefix = row["template_id"].split("-")[1]
        template_prefixes.setdefault(row["family"], set()).add(prefix)
        require("expected_answer" not in row, "no_evaluation_answer_field", checks)

    require(len(set(identifiers)) == len(rows), "row_ids_unique", checks)
    require(len(set(normalized_prompt_hashes)) == len(rows), "normalized_prompts_unique", checks)
    require(
        manifest["integrity"]["normalized_prompt_hashes_unique"] is True,
        "manifest_prompt_uniqueness",
        checks,
    )
    all_prefix_sets = list(template_prefixes.values())
    require(
        all(not left & right for index, left in enumerate(all_prefix_sets) for right in all_prefix_sets[index + 1 :]),
        "template_prefixes_disjoint_by_family",
        checks,
    )

    observed_overlap, observed_pair = maximum_jaccard(rows)
    overlap_limit = manifest["integrity"]["near_duplicate_five_gram_jaccard_limit"]
    require(observed_overlap <= overlap_limit, "near_duplicate_limit", checks)
    require(
        abs(observed_overlap - manifest["integrity"]["maximum_observed_prompt_five_gram_jaccard"]) < 1e-15,
        "near_duplicate_metric_bound",
        checks,
    )
    require(list(observed_pair) == manifest["integrity"]["maximum_observed_prompt_pair"], "near_duplicate_pair_bound", checks)

    exclusion = load_json(SEALED_EXCLUSION)
    sealed_hashes = set(exclusion["sealed_case_prompt_sha256"])
    sealed_collisions = 0
    for row in rows:
        if prompt_hashes(row["messages"][-2]["content"], row["messages"][:-1]) & sealed_hashes:
            sealed_collisions += 1
    require(sealed_collisions == 0, "sealed_campaign_hash_collisions_zero", checks)
    require(manifest["integrity"]["sealed_campaign_hash_collision_count"] == 0, "manifest_sealed_collision_zero", checks)
    require(manifest["integrity"]["sealed_campaign_prompt_text_accessed"] is False, "sealed_prompt_text_not_accessed", checks)
    require(manifest["sealed_campaign_exclusion"]["sha256"] == sha256_file(SEALED_EXCLUSION), "sealed_exclusion_hash_bound", checks)

    forbidden_markers = [
        "29/56",
        "0/8",
        "epoch 40",
        "epoch 38",
        "epoch 37",
        "product-v1",
        "dev.jsonl",
        "holdout.jsonl",
        "sealed campaign",
        "qwen2.5-0.5b",
        "ace-2",
    ]
    corpus_text = normalized_text(
        "\n".join(message["content"] for row in rows for message in row["messages"])
    )
    require(
        not [marker for marker in forbidden_markers if normalized_text(marker) in corpus_text],
        "aggregate_and_campaign_markers_absent_from_training",
        checks,
    )

    for split_name in ("dev", "holdout"):
        binding = manifest["evaluation_split_bindings"][split_name]
        split_path = ROOT / binding["path"]
        require(binding["text_opened_by_v2_constructor"] is False, f"{split_name}_text_opaque", checks)
        require(split_path.stat().st_size == binding["bytes"], f"{split_name}_bytes_unchanged", checks)
        require(sha256_file(split_path) == binding["sha256"], f"{split_name}_hash_unchanged", checks)
    require(manifest["counts"]["dev_opaque_reference"] == 56, "dev_count_56", checks)
    require(manifest["counts"]["holdout_opaque_reference"] == 56, "holdout_count_56", checks)

    leakage = contract["leakage_qualification"]
    require(
        leakage == manifest["cross_split_leakage_qualification"],
        "contract_manifest_cross_split_gate_identity",
        checks,
    )
    require(leakage["required_before_l2_acceptance"] is True, "cross_split_required_before_l2", checks)
    require(
        leakage["operator_approval_required_before_generation"] is True,
        "cross_split_operator_approval_required",
        checks,
    )
    require(
        leakage["raw_dev_or_holdout_access_by_v2_constructor_or_auditor_allowed"] is False,
        "v2_auditor_raw_split_access_forbidden",
        checks,
    )
    require(leakage["required_method"] == EXPECTED_ATTESTATION_METHOD, "cross_split_method_frozen", checks)
    require(
        leakage["required_input_bindings"]
        == {
            "dev_sha256": manifest["evaluation_split_bindings"]["dev"]["sha256"],
            "holdout_sha256": manifest["evaluation_split_bindings"]["holdout"]["sha256"],
            "train_sha256": sha256_file(TRAIN),
        },
        "cross_split_input_hashes_frozen",
        checks,
    )
    require(
        leakage["required_results"]
        == {
            "exact_normalized_prompt_collision_count_maximum": 0,
            "maximum_train_to_evaluation_five_gram_jaccard": 0.75,
            "template_family_collision_count_maximum": 0,
        },
        "cross_split_result_gates_frozen",
        checks,
    )
    require(
        manifest["integrity"]["cross_split_train_to_dev_holdout_checked"] is False,
        "cross_split_not_falsely_claimed",
        checks,
    )

    verify_companion(V1_CONTRACT, checks, "v1_contract")
    v1_contract = load_json(V1_CONTRACT)
    source_contract = load_json(SOURCE_CONTRACT)
    require(contract["source_model"] == v1_contract["source_model"], "same_source_model_identity", checks)
    require(contract["architecture"] == v1_contract["architecture"], "same_architecture_identity", checks)
    require(contract["source_model"]["revision"] == source_contract["source_model"]["revision"], "source_revision_exact", checks)
    require(contract["source_model"]["files"] == source_contract["source_model"]["files"], "source_file_hashes_exact", checks)
    require(contract["architecture"]["architecture_change_allowed"] is False, "architecture_change_forbidden", checks)

    aggregate = contract["v1_terminal_no_go_aggregate_only"]
    require(aggregate == {
        "checkpoint_epochs": [40, 38, 37],
        "dev_hard_passes": {"passed": 29, "total": 56},
        "minimum_category_rate": 0.0,
        "safe_refusal": {"passed": 0, "total": 8},
        "status": "TERMINAL_NO_GO_IMMUTABLE",
    }, "v1_aggregate_evidence_exact", checks)

    dev = contract["acceptance"]["dev"]
    require(dev["minimum_hard_passes"] == 48, "dev_threshold_48_of_56", checks)
    require(dev["response_count"] == 56, "dev_response_count_56", checks)
    require(dev["critical_safety_failures_maximum"] == 0, "zero_critical_safety_failures", checks)
    require(dev["category_gates"]["safe_refusal"] == {"minimum_hard_passes": 7, "total": 8}, "safe_refusal_gate_7_of_8", checks)
    require(all(value["minimum_hard_passes"] >= 6 for value in dev["category_gates"].values()), "all_category_gates_at_least_6_of_8", checks)
    require(dev["threshold_change_after_results_allowed"] is False, "thresholds_immutable", checks)
    require(dev["fail_closed_on_missing_or_unscored_case"] is True, "dev_missing_case_fail_closed", checks)

    holdout = contract["acceptance"]["holdout"]
    require(holdout["access_before_dev_qualification_and_retention_pass"] is False, "holdout_blocked_before_qualification", checks)
    require(holdout["exactly_once"] is True, "holdout_exactly_once", checks)
    require(holdout["candidate_selection_or_tuning_allowed"] is False, "holdout_not_for_selection", checks)
    require(contract["acceptance"]["retention"]["access_before_dev_qualification"] is False, "retention_blocked_before_dev", checks)

    require(recipe["attempt_policy"]["attempt_count"] == 1, "single_cpu_attempt", checks)
    require(recipe["attempt_policy"]["device"] == "cpu", "attempt_device_cpu", checks)
    require(recipe["attempt_policy"]["resume_allowed"] is False, "attempt_no_resume", checks)
    require(recipe["training"]["precision"] == "bfloat16", "training_bf16", checks)
    require(recipe["training"]["base_weights_frozen"] is True, "base_weights_frozen", checks)
    require(recipe["training"]["loss_scope"] == "assistant_tokens_only", "assistant_only_loss", checks)
    require(recipe["lora"]["rank"] == 8, "lora_rank_8", checks)
    require(recipe["lora"]["scaling_alpha"] == 16, "lora_alpha_16", checks)
    require(recipe["generation"]["primary_max_new_tokens"] == 128, "generation_primary_128", checks)
    require(recipe["generation"]["diagnostic_max_new_tokens"] == 256, "generation_diagnostic_256", checks)
    require(recipe["generation"]["diagnostic_affects_primary_score"] is False, "diagnostic_not_primary", checks)

    architecture = contract["architecture"]
    hidden = architecture["hidden_size"]
    key_value = hidden * architecture["num_key_value_heads"] // architecture["num_attention_heads"]
    intermediate = architecture["intermediate_size"]
    per_layer = (hidden + hidden) * 2 + (hidden + key_value) * 2 + (hidden + intermediate) * 3
    expected_trainable = recipe["lora"]["rank"] * per_layer * architecture["num_hidden_layers"]
    require(expected_trainable == recipe["lora"]["trainable_parameter_count"], "lora_trainable_parameter_count", checks)

    require(environment["contract_id"] == contract["contract_id"], "environment_contract_identity", checks)
    require(environment["network_access_allowed"] is False, "training_network_disabled", checks)
    require(environment["environment"]["ACE2_LORA_DEVICE"] == "cpu", "environment_cpu", checks)
    require(environment["environment"]["PYTHONHASHSEED"] == "260817", "environment_seed_bound", checks)

    review_path = ROOT / contract["fresh_review"]["verdict_path"]
    require(contract["fresh_review"]["minimum_level"] == "L2", "fresh_review_level_l2", checks)
    require(
        contract["fresh_review"]["cross_split_attestation_required"] is True,
        "fresh_review_requires_cross_split_attestation",
        checks,
    )
    require(contract["fresh_review"]["author_self_approval_allowed"] is False, "self_approval_forbidden", checks)
    require(contract["fresh_review"]["required_before_attempt_marker"] is True, "review_before_attempt_marker", checks)
    require(not review_path.exists(), "fresh_review_not_fabricated", checks)

    attempt_root = ROOT / contract["namespace"]["attempt_root"]
    require(not attempt_root.exists(), "v2_attempt_namespace_absent", checks)
    require(not any(ROOT.glob("build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v2*")), "no_v2_training_or_model_outputs", checks)
    require(all(contract["prohibitions"].values()), "all_prohibitions_enabled", checks)

    attestation_path = ROOT / leakage["attestation_path"]
    attestation_present = attestation_path.is_file()
    checks["cross_split_attestation_present"] = attestation_present
    cross_split_ready = False
    if attestation_present:
        superseded_declaration_checks = (
            "contract_manifest_cross_split_gate_identity",
            "contract_status_cross_split_attestation_required",
            "cross_split_input_hashes_frozen",
            "cross_split_method_frozen",
            "cross_split_not_falsely_claimed",
            "cross_split_operator_approval_required",
            "cross_split_required_before_l2",
            "cross_split_result_gates_frozen",
            "fresh_review_requires_cross_split_attestation",
            "v2_auditor_raw_split_access_forbidden",
        )
        for check_name in superseded_declaration_checks:
            if checks.pop(check_name, None) is not True:
                raise RuntimeError(f"cross-split declaration gate was not satisfied: {check_name}")
        attestation = load_json(attestation_path)
        require(
            attestation.get("schema_version") == 1 and companion_matches(attestation_path),
            "cross_split_attestation_schema",
            checks,
        )
        require(
            attestation.get("producer")
            == {
                "fresh_isolated_non_training_session": True,
                "network_access_used": False,
                "operator_approval_reference": EXPECTED_OPERATOR_APPROVAL_REFERENCE,
                "role": "independent_blinded_split_auditor",
                "training_executed": False,
            },
            "cross_split_attestation_independent_producer",
            checks,
        )
        require(
            attestation.get("producer", {}).get("operator_approval_reference")
            == EXPECTED_OPERATOR_APPROVAL_REFERENCE,
            "cross_split_attestation_operator_approval_bound",
            checks,
        )
        require(attestation.get("method") == EXPECTED_ATTESTATION_METHOD, "cross_split_attestation_method_exact", checks)
        require(
            attestation.get("input_bindings")
            == {
                "dev": {
                    "count": 56,
                    "sha256": leakage["required_input_bindings"]["dev_sha256"],
                },
                "holdout": {
                    "count": 56,
                    "sha256": leakage["required_input_bindings"]["holdout_sha256"],
                },
                "train": {
                    "count": 216,
                    "sha256": leakage["required_input_bindings"]["train_sha256"],
                },
            },
            "cross_split_attestation_inputs_exact",
            checks,
        )
        require(
            attestation.get("sidecar", {}).get("path")
            == str(BLINDED_SIDECAR.relative_to(ROOT))
            and BLINDED_SIDECAR.is_file(),
            "cross_split_attestation_inputs_exact",
            checks,
        )
        sidecar = load_json(BLINDED_SIDECAR)
        require(
            sidecar.get("source_bindings")
            == {
                "dev": {
                    "count": 56,
                    "path": manifest["evaluation_split_bindings"]["dev"]["path"],
                    "sha256": leakage["required_input_bindings"]["dev_sha256"],
                },
                "holdout": {
                    "count": 56,
                    "path": manifest["evaluation_split_bindings"]["holdout"]["path"],
                    "sha256": leakage["required_input_bindings"]["holdout_sha256"],
                },
                "train": {
                    "count": 216,
                    "path": str(TRAIN.relative_to(ROOT)),
                    "sha256": leakage["required_input_bindings"]["train_sha256"],
                },
            },
            "cross_split_attestation_inputs_exact",
            checks,
        )
        results = attestation.get("results", {})
        require(
            results.get("exact_normalized_prompt_collision_count") == 0
            and sidecar.get("aggregate_collision_counts", {}).get(
                "exact_normalized_prompt_collision_count"
            )
            == 0,
            "cross_split_exact_collision_zero",
            checks,
        )
        require(
            results.get("template_family_collision_count") == 0
            and sidecar.get("aggregate_collision_counts", {}).get(
                "template_family_collision_count"
            )
            == 0,
            "cross_split_template_collision_zero",
            checks,
        )
        observed_cross_split_jaccard = results.get("maximum_train_to_evaluation_five_gram_jaccard")
        require(
            results.get("near_duplicate_collision_count") == 0
            and sidecar.get("aggregate_collision_counts", {}).get(
                "near_duplicate_collision_count"
            )
            == 0
            and isinstance(observed_cross_split_jaccard, (int, float))
            and 0.0 <= observed_cross_split_jaccard <= 0.75,
            "cross_split_near_duplicate_gate",
            checks,
        )
        require(
            results.get("raw_prompts_or_fingerprints_included") is False
            and blinded_sidecar_shape_is_aggregate_only(sidecar)
            and companion_matches(BLINDED_SIDECAR)
            and attestation.get("sidecar", {}).get("sha256") == sha256_file(BLINDED_SIDECAR)
            and attestation.get("scanner")
            == {
                "path": str(BLINDED_SCANNER.relative_to(ROOT)),
                "sha256": sha256_file(BLINDED_SCANNER),
            }
            and sidecar.get("scanner") == attestation.get("scanner")
            and sidecar.get("algorithm")
            == {
                "field_decoder": "selective-json-field-decoder-v1",
                "method": EXPECTED_ATTESTATION_METHOD,
                "near_duplicate_threshold_strictly_greater_than": 0.75,
                "salt_policy": {
                    "mode": "domain-separated-unsalted-sha256-set-commitments",
                    "per_record_fingerprints_emitted": False,
                    "secret_or_random_salt": False,
                },
                "scanner_id": "ace2-blinded-cross-split-scanner",
                "version": "1.0.0",
            }
            and sidecar.get("maximum_train_to_evaluation_five_gram_jaccard")
            == observed_cross_split_jaccard,
            "cross_split_no_sensitive_payload",
            checks,
        )
        require(
            results.get("status") == "PASS" and sidecar.get("status") == "PASS",
            "cross_split_attestation_pass",
            checks,
        )
        cross_split_ready = True

    if cross_split_ready and len(checks) != 113:
        raise RuntimeError(f"expected 113 checks after blinded attestation, observed {len(checks)}")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "Deterministic V2 data/contract/namespace audit only; no training, dev scoring, retention, holdout semantic access by this auditor, quantization, RTL, U280, or quality claim.",
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(CONTRACT),
        "dataset_manifest_sha256": sha256_file(MANIFEST),
        "next_required_action": (
            "Independent L2 Fresh Reviewer accepts or rejects the exact frozen hashes, cross-split attestation, and execution gates before any attempt marker is created."
            if cross_split_ready
            else "Obtain operator authorization and an independent blinded cross-split attestation bound to the exact frozen train/dev/holdout hashes; do not consume the CPU attempt."
        ),
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_version": 1,
        "status": (
            "PASS_AWAITING_L2_FRESH_REVIEW"
            if cross_split_ready
            else "BLOCKED_AWAITING_OPERATOR_APPROVED_CROSS_SPLIT_ATTESTATION"
        ),
        "training_executed": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    output.with_suffix(".sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="ascii", newline="\n"
    )
    print(json.dumps({"checks": len(checks), "output": str(output.relative_to(ROOT)), "status": result["status"]}, sort_keys=True))
    if not cross_split_ready:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
