#!/usr/bin/env python3
"""Fail-closed audit of the frozen, non-consuming ACE-2 LoRA V3 package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v3"
TRAIN = DATASET / "train.jsonl"
MANIFEST = DATASET / "dataset_manifest.json"
RECIPE = DATASET / "training_recipe.json"
PACKAGE_MANIFEST = DATASET / "execution_package_manifest.json"
FREEZE_MANIFEST = DATASET / "freeze_manifest.json"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V3_PREEXECUTION_CONTRACT.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-freeze-operator-authority.json"
BLINDED_ATTESTATION = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-cross-split-blinded-attestation.json"
BLINDED_SIDECAR = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-cross-split-blinded-sidecar.json"
SYNTHETIC_ATTESTATION = ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_disjoint_attestation.json"
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v3"
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3"
EXECUTION_MANIFEST = OFFLINE / "execution-package-manifest.json"
SELF_TEST = OFFLINE / "execution-self-test.json"
L2_REVIEW = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-freeze-l2-review.json"
ATTEMPT_AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-attempt-operator-authority.json"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3"
GENERATOR = ROOT / "tools/freeze_qwen25_05b_bf16_lora_product_v3.py"
V2_RUNTIME = ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/runtime.py"
V3_RUNTIME = PACKAGE / "runtime.py"
EXPECTED_CONTRACT_SHA256 = "411bf8dfb9722bfa3d18ee87219b9b46d92f5b2a9f0c9fb10f439d89e4e3210d"
EXPECTED_CATEGORIES = {
    "arithmetic": 40,
    "concise_summary": 48,
    "context_memory": 32,
    "format_discipline": 48,
    "polite_rewrite": 40,
    "safe_refusal": 48,
    "structured_extraction": 32,
}


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


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    fields = companion.read_text(encoding="ascii").split() if companion.is_file() else []
    return fields == [sha256_file(path), path.name]


def legacy_companion_matches(path: Path) -> bool:
    companion = path.with_suffix(".sha256")
    fields = companion.read_text(encoding="ascii").split() if companion.is_file() else []
    return fields == [sha256_file(path), path.name]


def require(condition: bool, name: str, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise RuntimeError(name)


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_manifest() -> dict[str, Any]:
    generator = import_module(GENERATOR, "ace2_v3_generator_for_audit")
    return generator.package_manifest_payload()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checks: dict[str, bool] = {}

    for path in (TRAIN, MANIFEST, RECIPE, PACKAGE_MANIFEST, FREEZE_MANIFEST, AUTHORITY):
        require(path.is_file() and companion_matches(path), f"companion_exact:{path.name}", checks)
    require(sha256_file(CONTRACT) == EXPECTED_CONTRACT_SHA256, "contract_hash_exact", checks)
    require(legacy_companion_matches(CONTRACT), "contract_companion_exact", checks)

    reproduction = subprocess.run(
        [sys.executable, str(GENERATOR), "--check-final"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(reproduction.returncode == 0, "deterministic_freeze_reproduction", checks)

    contract = load_json(CONTRACT)
    manifest = load_json(MANIFEST)
    recipe = load_json(RECIPE)
    authority = load_json(AUTHORITY)
    freeze = load_json(FREEZE_MANIFEST)
    rows = load_jsonl(TRAIN)
    require(len(rows) == 288, "train_count_288", checks)
    require(dict(sorted(Counter(row["category"] for row in rows).items())) == EXPECTED_CATEGORIES, "category_mixture_exact", checks)
    require(manifest["category_counts"] == EXPECTED_CATEGORIES, "manifest_category_mixture_exact", checks)
    require(manifest["counts"]["train"] == 288, "manifest_train_count_288", checks)

    expected_schema = {
        "category",
        "id",
        "input_sha256",
        "messages",
        "provenance",
        "split",
        "target_validation",
        "template_family",
    }
    require(all(set(row) == expected_schema for row in rows), "row_schema_exact", checks)
    require(len({row["id"] for row in rows}) == 288, "row_ids_unique", checks)
    require(len({row["input_sha256"] for row in rows}) == 288, "input_hashes_unique", checks)
    require(all(row["split"] == "train" for row in rows), "row_split_train", checks)
    require(all(row["category"] in EXPECTED_CATEGORIES for row in rows), "official_category_on_every_row", checks)

    system = manifest["system_message"]["value"]
    require(system == "You are a helpful assistant.", "system_message_exact", checks)
    require(
        all(row["messages"][0] == {"content": system, "role": "system"} for row in rows),
        "system_message_byte_identical_all_rows",
        checks,
    )
    require(manifest["integrity"]["system_message_mismatch_count"] == 0, "system_message_mismatch_zero", checks)
    require(
        hashlib.sha256(normalized_text(system).encode("utf-8")).hexdigest()
        == contract["data_contract"]["system_message_normalized_sha256"],
        "system_message_normalized_hash_exact",
        checks,
    )

    v2_runtime = import_module(V2_RUNTIME, "ace2_v2_frozen_scorer")
    v3_runtime = import_module(V3_RUNTIME, "ace2_v3_runtime_for_audit")
    require(sha256_file(V2_RUNTIME) == contract["evaluation_contract"]["runtime_sha256"], "frozen_scorer_hash_exact", checks)
    require(inspect.getsource(v2_runtime.hard_check) == inspect.getsource(v3_runtime.hard_check), "v3_hard_check_source_identical", checks)
    require(inspect.getsource(v2_runtime.score_gates) == inspect.getsource(v3_runtime.score_gates), "v3_score_gates_source_identical", checks)
    target_results = [
        v2_runtime.hard_check(row["target_validation"], row["messages"][-1]["content"])
        for row in rows
    ]
    require(all(item["passed"] for item in target_results), "all_288_targets_pass_frozen_scorer", checks)
    require(sum(item["passed"] for item in target_results) == 288, "target_pass_count_288", checks)
    refusal_rows = [row for row in rows if row["category"] == "safe_refusal"]
    require(len(refusal_rows) == 48, "safe_refusal_count_48", checks)
    require(all("cannot" in row["messages"][-1]["content"] for row in refusal_rows), "safe_refusal_ascii_cannot", checks)
    require(all("’" not in row["messages"][-1]["content"] for row in refusal_rows), "safe_refusal_curly_apostrophe_absent", checks)

    provenance = [row["provenance"] for row in rows]
    require(all(item["authoring"] == "deterministic_project_authored_v3_generator" for item in provenance), "rows_project_authored", checks)
    require(all(item["external_model_output"] is False and item["source_split"] == "none" for item in provenance), "rows_no_external_or_eval_source", checks)
    require(manifest["integrity"]["prior_v1_v2_normalized_prompt_collision_count"] == 0, "no_v1_v2_prompt_replay", checks)

    require(recipe["lora"]["master_parameter_dtype"] == "float32", "lora_master_fp32", checks)
    require(recipe["optimizer"]["first_moment_dtype"] == "float32", "adam_first_moment_fp32", checks)
    require(recipe["optimizer"]["second_moment_dtype"] == "float32", "adam_second_moment_fp32", checks)
    require(recipe["training"]["base_compute"] == "bfloat16", "base_compute_bf16", checks)
    require(recipe["merge"]["floating_parameter_dtype"] == "bfloat16", "final_merge_bf16", checks)
    require(recipe["lora"]["dropout"] == 0.0, "lora_dropout_zero", checks)
    require(recipe["optimizer"]["learning_rate"] == 8e-5, "learning_rate_exact", checks)
    require(recipe["checkpoint_policy"]["candidate_epochs"] == [1, 2, 3, 4], "candidate_epochs_1_to_4", checks)
    require(recipe["checkpoint_policy"]["save_total_limit"] is None, "all_candidate_checkpoints_retained", checks)

    for section, local_name in (
        ("dev", "evaluate_dev.py"),
        ("holdout", "evaluate_holdout.py"),
        ("retention", "retention.py"),
    ):
        expected = contract["evaluation_contract"][section]["evaluator_sha256"]
        require(sha256_file(ROOT / contract["evaluation_contract"][section]["evaluator_path"]) == expected, f"frozen_{section}_evaluator_exact", checks)
        require(sha256_file(PACKAGE / local_name) == expected, f"v3_{section}_evaluator_bytes_identical", checks)

    require(package_manifest() == load_json(PACKAGE_MANIFEST), "execution_package_manifest_exact", checks)
    require(load_json(PACKAGE_MANIFEST)["tree_sha256"] == freeze["package_tree_sha256"], "package_tree_freeze_bound", checks)
    for relative, metadata in freeze["artifacts"].items():
        path = ROOT / relative
        require(path.is_file() and path.stat().st_size == metadata["bytes"] and sha256_file(path) == metadata["sha256"], f"frozen_artifact_exact:{relative}", checks)

    blinded = load_json(BLINDED_ATTESTATION)
    sidecar = load_json(BLINDED_SIDECAR)
    require(companion_matches(BLINDED_ATTESTATION) and companion_matches(BLINDED_SIDECAR), "blinded_attestation_companions_exact", checks)
    require(blinded["results"]["status"] == "PASS" and sidecar["status"] == "PASS", "blinded_scan_pass", checks)
    require(blinded["results"]["exact_normalized_prompt_collision_count"] == 0, "blinded_exact_collision_zero", checks)
    require(blinded["results"]["template_family_collision_count"] == 0, "blinded_template_collision_zero", checks)
    require(blinded["results"]["near_duplicate_collision_count"] == 0, "blinded_near_collision_zero", checks)
    require(blinded["results"]["maximum_train_to_evaluation_five_gram_jaccard"] <= 0.25, "blinded_jaccard_at_most_point25", checks)
    require(blinded["results"]["raw_prompts_or_fingerprints_included"] is False, "blinded_output_aggregate_only", checks)
    require(blinded["method"]["answer_fields_accessed"] is False and blinded["method"]["human_exposure_to_evaluation_text"] is False, "blinded_no_answer_or_human_exposure", checks)
    require(blinded["input_bindings"]["train"]["sha256"] == sha256_file(TRAIN), "blinded_train_hash_exact", checks)
    require(blinded["input_bindings"]["dev"]["sha256"] == contract["leakage_contract"]["dev_sha256"], "blinded_dev_hash_exact", checks)
    require(blinded["input_bindings"]["holdout"]["sha256"] == contract["leakage_contract"]["holdout_sha256"], "blinded_holdout_hash_exact", checks)

    synthetic = load_json(SYNTHETIC_ATTESTATION)
    require(companion_matches(SYNTHETIC_ATTESTATION), "synthetic_attestation_companion_exact", checks)
    require(synthetic["results"]["status"] == "PASS", "synthetic_disjointness_pass", checks)
    require(synthetic["results"]["exact_normalized_prompt_collision_count"] == 0, "synthetic_exact_collision_zero", checks)
    require(synthetic["results"]["template_family_collision_count"] == 0, "synthetic_template_collision_zero", checks)
    require(synthetic["results"]["near_duplicate_collision_count"] == 0, "synthetic_near_collision_zero", checks)
    require(synthetic["results"]["maximum_fixture_to_evaluation_five_gram_jaccard"] <= 0.25, "synthetic_jaccard_at_most_point25", checks)

    execution_manifest = load_json(EXECUTION_MANIFEST)
    self_test = load_json(SELF_TEST)
    require(execution_manifest == load_json(PACKAGE_MANIFEST), "offline_execution_manifest_exact", checks)
    require(self_test["status"] == "PASS", "offline_package_self_test_pass", checks)
    require(self_test["execution_tree_sha256"] == execution_manifest["tree_sha256"], "offline_self_test_tree_bound", checks)
    for required in (
        "real_model_trainable_parameter_dtypes_fp32",
        "real_model_frozen_base_dtypes_bf16",
        "real_model_forward_logits_bf16",
        "real_model_gradient_dtypes_fp32",
        "real_optimizer_adam_moments_fp32",
    ):
        require(self_test["checks"].get(required) is True, f"offline_dtype_check:{required}", checks)

    require(authority["status"] == "AUTHORIZED_NO_TRAINING_AUTHORITY", "freeze_authority_exact", checks)
    require(authority["training_or_attempt_marker_authorized"] is False, "freeze_authority_no_marker", checks)
    require(freeze["attempt_marker_creation_authorized"] is False, "freeze_manifest_no_marker_authority", checks)
    require(not RUN_ROOT.exists(), "v3_run_namespace_absent", checks)
    require(not list(ROOT.glob("build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/**/ATTEMPT_CONSUMPTION_MARKER.json")), "v3_attempt_marker_absent", checks)
    require(not ATTEMPT_AUTHORITY.exists(), "fresh_attempt_authority_not_fabricated", checks)
    require(not L2_REVIEW.exists(), "independent_l2_review_not_self_authored", checks)

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "Frozen V3 dataset/runner/leakage/self-test audit only; no attempt marker, training, dev, retention, holdout, quantization, RTL, or U280 execution.",
        "contract_sha256": sha256_file(CONTRACT),
        "execution_self_test_sha256": sha256_file(SELF_TEST),
        "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
        "next_required_action": "An independent L2 Fresh Reviewer must accept the exact freeze artifact hash map and execution interlocks before any separate attempt authority or marker may exist.",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "package_tree_sha256": freeze["package_tree_sha256"],
        "reproduction_output": reproduction.stdout.strip(),
        "schema_version": 1,
        "status": "PASS_AWAITING_INDEPENDENT_L2",
        "training_executed": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="ascii", newline="\n"
    )
    print(json.dumps({"checks": len(checks), "freeze_manifest_sha256": result["freeze_manifest_sha256"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
