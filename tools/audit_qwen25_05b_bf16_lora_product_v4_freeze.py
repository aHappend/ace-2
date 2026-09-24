#!/usr/bin/env python3
"""Fail-closed audit of the frozen, non-executing ACE-2 LoRA V4 package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools/freeze_qwen25_05b_bf16_lora_product_v4.py"
SELF_TEST_SOURCE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v4/self_test.py"
DATASET = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v4"
EXPECTED_IMMUTABLE = {
    "v1": {"file_count": 19, "tree_sha256": "49d5301ed33b5963096da2278be59bb595f9b2964a72dd723f4807c4c6297b2b"},
    "v2": {"file_count": 18, "tree_sha256": "5fd67e9c9fe039a47dcd47419ac97ed0871ed1ca16760de8e940d486d49c614b"},
    "v3": {"file_count": 20, "tree_sha256": "9073fcccc4ca3f09394a02278c1445840f12fa6789405a2e4dbc6369abee4c0d"},
}


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


def message_hash(messages: list[dict[str, str]], *, normalize_content: bool) -> str:
    prepared = [
        {
            "content": normalized_text(item["content"]) if normalize_content else item["content"],
            "role": item["role"],
        }
        for item in messages
    ]
    payload = json.dumps(prepared, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise RuntimeError(f"training row lacks a final assistant target: {row.get('id', '<unknown>')}")
    return messages[:-1]


def expected_input_hash_schemes(v1_count: int, patch_count: int, probe_count: int) -> dict[str, Any]:
    canonicalization = "UTF-8 JSON with ensure_ascii=false, sorted keys, and compact separators"
    prompt_scope = "prompt messages only; final training assistant target excluded"
    return {
        "content_derived_normalized_comparison": {
            "canonicalization": canonicalization,
            "content_normalization": "NFKC-casefold-collapse-whitespace",
            "message_scope": prompt_scope,
            "scheme_id": "sha256-normalized-message-json-v1",
            "uses": [
                "train input uniqueness",
                "V1 backbone to V4 patch disjointness",
                "synthetic probe input uniqueness",
                "synthetic probe to train disjointness",
            ],
        },
        "stored_input_sha256": {
            "v1_backbone": {
                "canonicalization": canonicalization,
                "content_normalization": "NFKC-casefold-collapse-whitespace",
                "message_scope": prompt_scope,
                "rows": v1_count,
                "scheme_id": "sha256-normalized-message-json-v1",
            },
            "v4_patch_and_synthetic_probes": {
                "canonicalization": canonicalization,
                "content_normalization": "none; raw message content",
                "message_scope": prompt_scope,
                "rows": {"patch": patch_count, "synthetic_probes": probe_count},
                "scheme_id": "sha256-raw-message-json-v1",
            },
        },
    }


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not companion.is_file():
        return False
    expected = f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n".encode("ascii")
    return companion.read_bytes() == expected


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/freeze-audit.json",
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    offline_root = (ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4").resolve()
    if not output.is_relative_to(offline_root):
        raise RuntimeError("audit output escaped the V4 offline namespace")

    generator = import_module(GENERATOR, "ace2_v4_generator_for_audit")
    self_test = import_module(SELF_TEST_SOURCE, "ace2_v4_self_test_for_audit")
    checks: dict[str, bool] = {}

    generator.check_final()
    checks["deterministic_final_freeze_reproduction"] = True
    freeze = load_json(DATASET / "freeze_manifest.json")
    manifest = load_json(DATASET / "dataset_manifest.json")
    recipe = load_json(DATASET / "training_recipe.json")
    contract = load_json(DATASET / "freeze_contract.json")
    v3_dev = load_json(generator.V3_CONTRACT)["acceptance_and_selection"]
    package_manifest = load_json(DATASET / "execution_package_manifest.json")
    v1_rows = load_jsonl(generator.V1_TRAIN)
    train_rows = load_jsonl(DATASET / "train.jsonl")
    patch_rows = load_jsonl(DATASET / "patch.jsonl")
    probe_rows = load_jsonl(DATASET / "synthetic_probes.jsonl")

    primary_artifacts = [
        DATASET / "train.jsonl",
        DATASET / "patch.jsonl",
        DATASET / "synthetic_probes.jsonl",
        DATASET / "freeze_contract.json",
        DATASET / "training_recipe.json",
        DATASET / "dataset_manifest.json",
        DATASET / "execution_package_manifest.json",
        DATASET / "freeze_manifest.json",
        generator.PACKAGE_CONFIG,
        generator.BLINDED_ATTESTATION,
        generator.BLINDED_SIDECAR,
        generator.SELF_TEST_RESULT,
    ]
    checks["all_companions_exact"] = all(companion_matches(path) for path in primary_artifacts)
    checks["v1_backbone_prefix_exact"] = (DATASET / "train.jsonl").read_bytes()[: generator.V1_TRAIN_BYTES] == generator.V1_TRAIN.read_bytes()
    checks["v1_backbone_manifest_exact"] = manifest["integrity"]["v1_prefix_sha256"] == generator.V1_TRAIN_SHA256 and manifest["counts"]["v1_backbone"] == 280
    checks["bounded_additive_patch_72"] = len(patch_rows) == 72 and manifest["counts"]["additive_patch"] == 72
    checks["fresh_synthetic_probes_28"] = len(probe_rows) == 28 and manifest["counts"]["synthetic_probes"] == 28
    checks["input_hash_schemes_documented"] = (
        manifest["schema_version"] == 2
        and manifest["input_hash_schemes"]
        == expected_input_hash_schemes(len(v1_rows), len(patch_rows), len(probe_rows))
    )
    v1_normalized_hashes = [
        message_hash(training_prompt_messages(row), normalize_content=True) for row in v1_rows
    ]
    patch_normalized_hashes = [
        message_hash(training_prompt_messages(row), normalize_content=True) for row in patch_rows
    ]
    train_normalized_hashes = [
        message_hash(training_prompt_messages(row), normalize_content=True) for row in train_rows
    ]
    probe_normalized_hashes = [
        message_hash(row["input_messages"], normalize_content=True) for row in probe_rows
    ]
    checks["v1_stored_hashes_match_normalized_scheme"] = all(
        row["input_sha256"] == derived for row, derived in zip(v1_rows, v1_normalized_hashes)
    )
    checks["v4_stored_hashes_match_raw_scheme"] = all(
        row["input_sha256"]
        == message_hash(training_prompt_messages(row), normalize_content=False)
        for row in patch_rows
    ) and all(
        row["input_sha256"] == message_hash(row["input_messages"], normalize_content=False)
        for row in probe_rows
    )
    checks["normalized_train_inputs_unique_from_content"] = (
        len(train_rows) == len(train_normalized_hashes) == len(set(train_normalized_hashes))
    )
    checks["normalized_v1_patch_inputs_disjoint_from_content"] = not (
        set(v1_normalized_hashes) & set(patch_normalized_hashes)
    )
    checks["normalized_probe_inputs_unique_from_content"] = (
        len(probe_rows) == len(probe_normalized_hashes) == len(set(probe_normalized_hashes))
    )
    checks["normalized_probe_inputs_disjoint_from_train_from_content"] = not (
        set(probe_normalized_hashes) & set(train_normalized_hashes)
    )
    checks["manifest_content_derived_integrity_claims_exact"] = all(
        manifest["integrity"][name] is True
        for name in (
            "normalized_probe_inputs_disjoint_from_train_from_message_content",
            "normalized_probe_inputs_unique_from_message_content",
            "normalized_train_inputs_unique_from_message_content",
            "normalized_v1_patch_inputs_disjoint_from_message_content",
            "stored_input_sha256_matches_documented_schemes",
        )
    )

    scorer = generator.import_scorer()
    checks["all_patch_targets_pass_frozen_scorer"] = all(
        scorer.hard_check(row["target_validation"], row["messages"][-1]["content"])["passed"]
        for row in patch_rows
    )
    checks["all_probe_targets_pass_frozen_scorer"] = all(
        scorer.hard_check(row, row["expected_answer"])["passed"] for row in probe_rows
    )
    checks["all_patch_targets_ascii_plain"] = all(
        row["messages"][-1]["content"].isascii() and "```" not in row["messages"][-1]["content"]
        for row in patch_rows
    )
    checks["system_message_hash_bound"] = contract["data"]["system_message"]["bytes_sha256"] == manifest["system_message"]["bytes_sha256"]
    checks["rank_alpha_prospectively_increased"] = recipe["lora"]["rank"] == 16 and recipe["lora"]["scaling_alpha"] == 32
    checks["fp32_master_states_bf16_compute"] = (
        recipe["lora"]["master_parameter_dtype"] == "float32"
        and recipe["optimizer"]["first_moment_dtype"] == "float32"
        and recipe["optimizer"]["second_moment_dtype"] == "float32"
        and recipe["training"]["base_compute"] == "bfloat16"
    )
    checks["lower_learning_rate"] = recipe["optimizer"]["learning_rate"] == 4e-5
    checks["epoch_one_candidate"] = recipe["checkpoint_policy"]["first_candidate_epoch"] == 1
    lifecycle = generator.checkpoint_lifecycle_payload()
    checks["checkpoint_lifecycle_contract_recipe_exact"] = contract["checkpoint_lifecycle"] == recipe["checkpoint_lifecycle"] == lifecycle
    checks["probe_early_stop_locks_before_dev"] = (
        recipe["early_stop"]["selection_inputs"] == lifecycle["selection_inputs"]
        and recipe["early_stop"]["checkpoint_selection_before_dev"] == lifecycle["probe_selection_rule"]
        and recipe["early_stop"]["checkpoint_lock_before_dev_access"] is True
        and recipe["early_stop"]["dev_access_before_checkpoint_lock"] is False
        and recipe["early_stop"]["stop_at_first_epoch_satisfying_every_probe_gate"] is True
        and recipe["early_stop"]["training_loss_selects_checkpoint"] is False
    )
    checks["dev_qualifies_locked_checkpoint_only"] = (
        contract["acceptance"]["dev"]["selection_source"] == lifecycle["selection_source"]
        and contract["acceptance"]["dev"]["selection_order"] == lifecycle["probe_selection_rule"]
        and contract["acceptance"]["dev"]["qualification_source"] == "frozen official dev aggregates only"
        and contract["acceptance"]["dev"]["qualification_checkpoint_count"] == 1
        and contract["acceptance"]["dev"]["checkpoint_reselection_after_results_allowed"] is False
        and recipe["checkpoint_policy"]["dev_qualification_scope"] == "locked_checkpoint_only"
        and recipe["checkpoint_policy"]["official_dev_reselection_allowed"] is False
    )
    unchanged_dev_gate_keys = (
        "category_gates",
        "critical_safety_failures_maximum",
        "minimum_hard_passes",
        "response_count",
        "threshold_change_after_results_allowed",
    )
    checks["dev_gates_unchanged"] = (
        all(contract["acceptance"]["dev"][key] == v3_dev[key] for key in unchanged_dev_gate_keys)
        and contract["acceptance"]["thresholds_changed"] is False
    )

    attestation = load_json(generator.BLINDED_ATTESTATION)
    scan_comparisons = attestation["results"]["comparisons"]
    threshold = manifest["leakage_policy"]["maximum_train_or_probe_to_evaluation_five_gram_jaccard"]
    checks["blinded_scan_exactly_once"] = attestation["method"]["scan_ordinal"] == 1
    checks["blinded_scan_pass_zero_collisions"] = attestation["results"]["status"] == "PASS" and all(
        item["exact_normalized_prompt_collision_count"] == 0
        and item["near_duplicate_collision_count"] == 0
        and item["template_family_collision_count"] == 0
        and item["maximum_five_gram_jaccard"] <= threshold
        for item in scan_comparisons.values()
    )
    checks["blinded_scan_answer_fields_unopened"] = attestation["method"]["answer_fields_accessed"] is False and attestation["method"]["human_exposure_to_evaluation_text"] is False
    expected_self_test = self_test.result_payload()
    observed_self_test = load_json(generator.SELF_TEST_RESULT)
    checks["package_self_test_exact_pass"] = observed_self_test == expected_self_test and observed_self_test["status"] == "PASS"
    checks["package_has_no_training_launcher"] = package_manifest["training_executable_included"] is False and not any((generator.PACKAGE / name).exists() for name in ("train.py", "run_once.sh", "evaluate_dev.py", "evaluate_holdout.py", "merge.py"))
    checks["official_model_namespace_absent"] = not generator.RUN_ROOT.exists()
    checks["no_attempt_or_training_authority"] = recipe["attempt_policy"]["attempt_authority_granted"] is False and recipe["attempt_policy"]["training_authority_granted"] is False and freeze["training_executed"] is False
    checks["freeze_pending_independent_l2"] = freeze["independent_l2_review"] == {"accepted": False, "status": "PENDING_FRESH_REVIEWER"}

    observed_immutable = {
        version: generator.immutable_tree(
            (
                ROOT / f"pilot/qwen25_05b_bf16_lora_product_{version}",
                ROOT / f"research/training/qwen2.5-0.5b-instruct-bf16-lora-product-{version}",
            )
        )
        for version in ("v1", "v2", "v3")
    }
    checks["v1_v2_v3_immutable"] = observed_immutable == EXPECTED_IMMUTABLE == freeze["predecessor_immutable_trees"]
    checks["freeze_artifact_bindings_exact"] = all(
        (ROOT / relative).is_file()
        and (ROOT / relative).stat().st_size == item["bytes"]
        and hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == item["sha256"]
        for relative, item in freeze["artifacts"].items()
    )

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"V4 freeze audit failed: {failed}")
    result = {
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "V4 package freeze verified; independent L2 acceptance and any later attempt authority remain absent.",
        "freeze_manifest_sha256": hashlib.sha256((DATASET / "freeze_manifest.json").read_bytes()).hexdigest(),
        "package_tree_sha256": package_manifest["tree_sha256"],
        "schema_version": 1,
        "status": "PASS_FROZEN_PENDING_INDEPENDENT_L2",
    }
    write_atomic(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(f"ACE2_LORA_V4_FREEZE_AUDIT_PASS checks={len(checks)} status={result['status']}")


if __name__ == "__main__":
    main()
