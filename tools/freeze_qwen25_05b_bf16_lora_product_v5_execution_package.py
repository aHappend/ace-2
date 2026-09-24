#!/usr/bin/env python3
"""Freeze the V5 execution contract and exact-package L2 review request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v5"
RUNNER = ROOT / "pilot/qwen25_05b_bf16_lora_product_v5_runner"
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V5_EXECUTION_PACKAGE_CONTRACT.json"
MANIFEST = OFFLINE / "v5-execution-package-manifest.json"
SELF_TEST = OFFLINE / "v5-execution-package-self-test.json"
REVIEW_REQUEST = OFFLINE / "execution-package-l2-review-request.json"
CALIBRATION = ROOT / "research/diagnostics/qwen25_lora_v5_selector_calibration.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required file absent: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    companion = path.with_suffix(path.suffix + ".sha256")
    with companion.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n")


def contract_payload() -> dict[str, Any]:
    if RUN_ROOT.exists():
        raise RuntimeError("official V5 namespace must remain absent while freezing execution package")
    recipe = load_json(DATASET / "training_recipe.json")
    parent = recipe["training"]["parent_adapter_initialization"]
    parent_dir = ROOT / parent["adapter_path"]
    environment = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2/environment.lock.json"
    return {
        "attempt_authority": {
            "attempt": 1,
            "attempt_marker_creation_authorized_required": True,
            "path": "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v5-attempt-operator-authority.json",
            "required_status": "AUTHORIZE_SINGLE_V5_ATTEMPT",
            "separate_fresh_operator_authority_required": True,
        },
        "claim_boundary": "Execution-package implementation contract only. It authorizes construction, selector calibration, and non-consuming validation of the V5 package, but grants no V5 run namespace, attempt marker, training, official evaluation, merge, quantization, RTL, PPA, FPGA, or product claim.",
        "contract_id": "qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5-execution-package-v1",
        "environment_resolution": {
            "device": "cpu",
            "package_versions_cuda_runtime_and_non_seed_launch_variables_source": binding(environment),
            "pythonhashseed_conflict": {
                "environment_lock_value": load_json(environment)["environment"]["PYTHONHASHSEED"],
                "resolution": "The V5 recipe-specific seed controls V5 execution; the reused environment lock controls package versions, runtime, offline flags, and other launch variables.",
                "selected_value": recipe["determinism"]["PYTHONHASHSEED"],
                "training_recipe_value": recipe["determinism"]["PYTHONHASHSEED"],
            },
        },
        "execution_order": [
            "non_consuming_package_self_test",
            "fresh_independent_l2_exact_package_acceptance",
            "fresh_operator_attempt_authority",
            "non_consuming_launch_preflight",
            "create_attempt_marker_then_continue_v1_parent",
            "score_half_then_full_candidate_on_frozen_selector",
            "stop_predev_no_go_if_neither_candidate_passes_without_official_dev_access",
            "qualify_exactly_one_selector_locked_checkpoint_on_official_dev",
            "merge_then_retention_then_exactly_once_holdout_only_after_prior_passes",
        ],
        "failure_policy": {
            "evaluator_no_execution_classified_separately": True,
            "first_genuine_failed_gate_stops_later_stages": True,
            "no_checkpoint_fallback": True,
            "no_resume": True,
            "preserve_partial_evidence": True,
        },
        "frozen_inputs": {
            "dataset_manifest": binding(DATASET / "dataset_manifest.json"),
            "freeze_contract": binding(DATASET / "freeze_contract.json"),
            "freeze_manifest": binding(DATASET / "freeze_manifest.json"),
            "parent_adapter": binding(parent_dir / "adapter_model.safetensors"),
            "parent_adapter_config": binding(parent_dir / "adapter_config.json"),
            "selector_calibration": binding(CALIBRATION),
            "selector_cases": binding(DATASET / "selector.jsonl"),
            "train": binding(DATASET / "train.jsonl"),
            "training_recipe": binding(DATASET / "training_recipe.json"),
        },
        "independent_execution_package_review": {
            "author_self_approval_allowed": False,
            "minimum_level": "L2",
            "path": "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/execution-package-l2-acceptance.json",
            "required_before_preflight_or_attempt_marker": True,
            "required_status": "ACCEPTED_V5_EXACT_PACKAGE_NO_ATTEMPT",
        },
        "namespace": {
            "attempt": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/attempt-0001",
            "dev": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/qualification/dev",
            "holdout": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/qualification/holdout-exactly-once",
            "merged_model": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/selected/merged-bf16",
            "model_root": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5",
            "offline_root": "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5",
            "retention": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/qualification/retention",
            "selected_adapter": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/selected/adapter",
        },
        "package": {
            "frozen_non_executing_package_preserved": "pilot/qwen25_05b_bf16_lora_product_v5",
            "runner": str(RUNNER.relative_to(ROOT)),
        },
        "selector_lock": {
            "candidate_checkpoints": ["half_epoch", "full_epoch"],
            "candidate_steps": [10, 20],
            "dev_access_before_lock": False,
            "fallback_allowed": False,
            "official_dev_checkpoint_count": 1,
            "official_dev_reselection_allowed": False,
            "selection_inputs": ["fresh_disjoint_selector_cases"],
            "training_loss_selects_checkpoint": False,
        },
        "schema_version": 1,
        "source_and_evaluator": {
            "frozen_scorer": binding(ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/runtime.py"),
            "retention_amendment": binding(ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_EVALUATOR_RETENTION_AMENDMENT.json"),
            "source_contract": binding(ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"),
        },
        "stage": "v5_execution_package",
        "status": "IMPLEMENTED_PENDING_SELF_TEST_AND_INDEPENDENT_L2",
    }


def review_request_payload() -> dict[str, Any]:
    if RUN_ROOT.exists():
        raise RuntimeError("official V5 namespace must remain absent before L2 review")
    manifest = load_json(MANIFEST)
    self_test = load_json(SELF_TEST)
    if self_test.get("status") != "PASS" or self_test.get("official_namespace_exists") is not False:
        raise RuntimeError("execution package self-test has not passed marker-free")
    return {
        "acceptance_requested": "ACCEPTED_V5_EXACT_PACKAGE_NO_ATTEMPT",
        "claim_boundary": "Request for independent L2 review of the exact V5 data, selector, calibration, recipe, runner, and fail-closed interlocks; no attempt authority is requested or granted.",
        "exact_package": {
            "contract": binding(CONTRACT),
            "dataset_manifest": binding(DATASET / "dataset_manifest.json"),
            "execution_manifest": binding(MANIFEST),
            "execution_self_test": binding(SELF_TEST),
            "freeze_manifest": binding(DATASET / "freeze_manifest.json"),
            "selector": binding(DATASET / "selector.jsonl"),
            "selector_calibration": binding(CALIBRATION),
            "training_recipe": binding(DATASET / "training_recipe.json"),
        },
        "execution_tree_sha256": manifest["tree_sha256"],
        "independent_from_constructor_required": True,
        "minimum_level": "L2",
        "official_namespace_exists": RUN_ROOT.exists(),
        "schema_version": 1,
        "status": "PENDING_INDEPENDENT_L2",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-contract", action="store_true")
    action.add_argument("--write-review-request", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write_contract:
        payload = canonical(contract_payload())
        write_new(CONTRACT, payload)
        print(f"ACE2_LORA_V5_EXECUTION_CONTRACT_FROZEN {hashlib.sha256(payload).hexdigest()}")
    elif args.write_review_request:
        payload = canonical(review_request_payload())
        write_new(REVIEW_REQUEST, payload)
        print(f"ACE2_LORA_V5_L2_REVIEW_REQUEST_FROZEN {hashlib.sha256(payload).hexdigest()}")
    else:
        if CONTRACT.read_bytes() != canonical(contract_payload()):
            raise RuntimeError("V5 execution contract differs")
        if REVIEW_REQUEST.read_bytes() != canonical(review_request_payload()):
            raise RuntimeError("V5 L2 review request differs")
        if RUN_ROOT.exists():
            raise RuntimeError("official V5 namespace exists")
        markers = list((ROOT / "build").glob("qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/attempt-*/ATTEMPT_CONSUMPTION_MARKER.json"))
        if markers:
            raise RuntimeError("V5 attempt marker exists")
        print("ACE2_LORA_V5_EXACT_PACKAGE_REPRODUCES_NO_NAMESPACE_NO_MARKER")


if __name__ == "__main__":
    main()
