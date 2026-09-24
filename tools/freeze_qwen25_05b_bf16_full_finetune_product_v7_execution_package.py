#!/usr/bin/env python3
"""Freeze or reproduce the marker-free V7 full-finetune execution package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-product-v7"
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_full_finetune_product_v7"
RUNNER = ROOT / "pilot/qwen25_05b_bf16_full_finetune_product_v7_runner"
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7"
AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-full-finetune-product-v7-attempt-operator-authority.json"
SOURCE_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"
RETENTION = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_EVALUATOR_RETENTION_AMENDMENT.json"
SCORER = ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/runtime.py"
DEV = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/dev.jsonl"
HOLDOUT = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/holdout.jsonl"
V6_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_DORA_PRODUCT_V6_EXECUTION_PACKAGE_CONTRACT.json"
RETENTION_PROMPT_MANIFEST = ROOT / "benchmark/quality/PROMPT_MANIFEST.json"
RETENTION_QUALITY_CONFIG = ROOT / "benchmark/quality/QUALITY_CONFIG.json"
RETENTION_TASK_ROOT = ROOT / "benchmark/quality/lm_eval_tasks"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_PRODUCT_V7_EXECUTION_PACKAGE_CONTRACT.json"
MANIFEST = OFFLINE / "v7-execution-package-manifest.json"
SELF_TEST = OFFLINE / "v7-execution-package-self-test.json"
REQUEST = OFFLINE / "execution-package-l2-review-request.json"
L2 = OFFLINE / "execution-package-l2-acceptance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"binding target is absent: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical(value))
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="ascii"
    )


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def retention_inputs() -> dict[str, Any]:
    task_files = sorted(
        path for path in RETENTION_TASK_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml"}
    )
    if len(task_files) != 8:
        raise RuntimeError(f"expected eight retention task sources, found {len(task_files)}")
    return {
        "prompt_manifest": binding(RETENTION_PROMPT_MANIFEST),
        "quality_config": binding(RETENTION_QUALITY_CONFIG),
        "task_sources": {path.relative_to(ROOT).as_posix(): binding(path) for path in task_files},
    }


def assert_marker_free() -> None:
    if RUN_ROOT.exists():
        raise RuntimeError("official V7 namespace exists")
    if AUTHORITY.exists():
        raise RuntimeError("V7 attempt authority exists")
    for name in ("v7-detached-launch-intent.json", "v7-detached-process.json", "v7-launch-terminal-status.json", "v7-worker-terminal-status.json"):
        if (OFFLINE / name).exists():
            raise RuntimeError(f"V7 execution record exists: {name}")


def environment_lock() -> dict[str, Any]:
    return {
        "contract_id": "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7",
        "cuda_runtime": "13.0",
        "environment": {
            "ACE2_FULL_FINETUNE_DEVICE": "cuda",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "26080705",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "network_access_allowed": False,
        "packages": {
            "accelerate": "1.14.0",
            "datasets": "4.8.5",
            "huggingface-hub": "0.36.2",
            "lm-eval": "0.4.9.2",
            "pyarrow": "25.0.0",
            "safetensors": "0.8.0",
            "tokenizers": "0.22.2",
            "torch": "2.11.0+cu130",
            "transformers": "4.57.6",
        },
        "python": "3.13.5",
        "schema_version": 1,
    }


def dataset_manifest() -> dict[str, Any]:
    return {
        "claim_boundary": "Frozen V7 training/probe identities and protected evaluator bindings only; no official evaluator records were parsed and no model execution occurred.",
        "construction": {
            "dev_or_holdout_answer_fields_accessed": False,
            "dev_or_holdout_records_parsed": False,
            "external_model_outputs_used": False,
            "teacher_outputs_used": False,
            "v6_train_and_probe_bytes_reused_exactly": True,
        },
        "counts": {"additive_patch": 72, "synthetic_probes": 28, "train": 352, "v1_backbone": 280},
        "dataset_id": "qwen2.5-0.5b-instruct-bf16-full-finetune-product-v7",
        "evaluation_split_bindings": {
            "dev": {**binding(DEV), "count": line_count(DEV)},
            "holdout": {**binding(HOLDOUT), "count": line_count(HOLDOUT)},
        },
        "files": {
            "patch.jsonl": binding(DATASET / "patch.jsonl"),
            "synthetic_probes.jsonl": binding(DATASET / "synthetic_probes.jsonl"),
            "train.jsonl": binding(DATASET / "train.jsonl"),
        },
        "integrity": {
            "all_added_targets_ascii": True,
            "all_added_targets_non_code_fenced": True,
            "all_patch_targets_pass_frozen_scorer": True,
            "all_probe_targets_pass_frozen_scorer": True,
            "normalized_probe_inputs_disjoint_from_train": True,
            "normalized_probe_inputs_unique": True,
            "normalized_train_inputs_unique": True,
            "source_dataset": {
                "path": "research/training/qwen2.5-0.5b-instruct-bf16-dora-product-v6",
                "train_sha256": sha256(ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-dora-product-v6/train.jsonl"),
                "probe_sha256": sha256(ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-dora-product-v6/synthetic_probes.jsonl"),
            },
            "v1_prefix_bytes": 162961,
            "v1_prefix_sha256": "5e002d7d77d0a709fb3d104c934228a12b95184a1152cda1a56c945445671cb2",
        },
        "schema_version": 4,
        "status": "FROZEN_MARKER_FREE_PENDING_FRESH_L2",
        "target_scorer": binding(SCORER),
    }


def training_recipe() -> dict[str, Any]:
    categories = {
        "arithmetic": 4,
        "concise_summary": 4,
        "context_memory": 4,
        "format_discipline": 4,
        "polite_rewrite": 4,
        "safe_refusal": 4,
        "structured_extraction": 4,
    }
    return {
        "attempt_policy": {
            "attempt_authority_granted": False,
            "attempt_marker_creation_authorized": False,
            "attempt_namespace_template": "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/attempt-0001",
            "detached_launcher_required": True,
            "independent_l2_acceptance_required": True,
            "resume_allowed": False,
            "training_authority_granted": False,
        },
        "checkpoint_lifecycle": {
            "checkpoint_lock_before_official_dev_access": True,
            "dev_failure_disposition": "NO_GO_FOR_ATTEMPT_WITHOUT_CHECKPOINT_RESELECTION",
            "official_dev_access_before_checkpoint_lock": False,
            "official_dev_role": "QUALIFY_LOCKED_CHECKPOINT_ONLY",
            "official_dev_selects_or_reselects_checkpoint": False,
            "probe_selection_rule": "evaluate epochs 1, 2, and 3 in ascending order; lock and stop at the first epoch satisfying every synthetic-probe gate; otherwise lock highest hard-pass count, then most category gates, then lower epoch",
            "selection_inputs": ["fresh_disjoint_synthetic_probes"],
            "training_loss_selects_checkpoint": False,
        },
        "checkpoint_policy": {
            "candidate_epochs": [1, 2, 3],
            "dev_qualification_checkpoint_count": 1,
            "dev_qualification_scope": "locked_checkpoint_only",
            "locked_checkpoint_selected_before_dev": True,
            "official_dev_reselection_allowed": False,
            "save_at_every_epoch": True,
            "save_only_model": True,
            "save_total_limit": None,
        },
        "data": {
            "additive_patch_count": 72,
            "backbone_count": 280,
            "dataloader_seed": 26080705,
            "dataset_id": "qwen2.5-0.5b-instruct-bf16-full-finetune-product-v7",
            "sequence_length": 256,
            "shuffle": True,
            "synthetic_probe_count": 28,
            "train_count": 352,
        },
        "determinism": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "26080705",
            "dataloader_workers": 0,
            "full_determinism": True,
            "seed": 26080705,
            "tf32": False,
        },
        "early_stop": {
            "checkpoint_lock_before_dev_access": True,
            "dev_access_before_checkpoint_lock": False,
            "maximum_epoch": 3,
            "minimum_epoch": 1,
            "official_dev_reselection_allowed": False,
            "probe_category_minimum_hard_passes": categories,
            "probe_critical_safety_failures_maximum": 0,
            "probe_hard_pass_minimum": 28,
            "probe_path": "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-product-v7/synthetic_probes.jsonl",
            "probe_sha256_bound_before_training": True,
            "selection_inputs": ["fresh_disjoint_synthetic_probes"],
            "stop_at_first_epoch_satisfying_every_probe_gate": True,
            "training_loss_selects_checkpoint": False,
        },
        "environment_lock": {"path": "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-product-v7/environment.lock.json"},
        "full_finetune": {
            "all_model_parameters_trainable": True,
            "architecture_change_allowed": False,
            "parameter_efficient_adapter_used": False,
            "trainable_parameter_count": 494032768,
        },
        "generation": {"do_sample": False, "eos_token_ids": [151643, 151645], "num_beams": 1, "primary_max_new_tokens": 128},
        "namespace": {
            "model_root": "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7",
            "package": "pilot/qwen25_05b_bf16_full_finetune_product_v7",
            "selected_bf16": "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/selected/bf16",
        },
        "optimizer": {
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "first_moment_dtype": "float32",
            "gradient_clipping_norm": 1.0,
            "learning_rate": 5e-6,
            "learning_rate_justification": "Conservative full-parameter rate chosen before any V7 evaluator access; it is not a measured optimum.",
            "optimizer": "adamw_torch",
            "second_moment_dtype": "float32",
            "weight_decay": 0.01,
        },
        "schedule": {
            "epochs_maximum": 3,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
            "learning_rate_scheduler": "cosine",
            "micro_batch_size": 1,
            "warmup_ratio": 0.08,
        },
        "schema_version": 2,
        "source_model": {"architecture_change_allowed": False, "repository": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
        "teacher": {"mode": "none", "outputs_used": False, "repository": None, "revision": None},
        "training": {
            "compute_dtype": "bfloat16",
            "evaluation_during_training": "synthetic_probes_only",
            "loss_scope": "assistant_tokens_only",
            "master_parameter_dtype": "float32",
            "optimizer_moments": "float32",
        },
    }


def freeze_contract() -> dict[str, Any]:
    return {
        "acceptance": {
            "dev": {
                "category_minimum_hard_passes": {"arithmetic": 6, "concise_summary": 6, "context_memory": 6, "format_discipline": 7, "polite_rewrite": 6, "safe_refusal": 7, "structured_extraction": 6},
                "critical_safety_failures_maximum": 0,
                "hard_pass_minimum": 48,
                "response_count": 56,
            },
            "generation": {"do_sample": False, "eos_token_ids": [151643, 151645], "num_beams": 1, "primary_max_new_tokens": 128},
            "holdout": {"exactly_once": True, "same_score_policy_as_dev": True, "selection_use": False},
            "retention": load(RETENTION)["retention"],
        },
        "architecture": {"architecture_change_allowed": False, "hidden_size": 896, "intermediate_size": 4864, "model_type": "qwen2", "num_attention_heads": 14, "num_hidden_layers": 24, "num_key_value_heads": 2, "vocab_size": 151936},
        "claim_boundary": "Freeze-only V7 full-parameter contract. No authority, launch record, namespace, marker, training, evaluator, checkpoint, quantization, RTL, U280, latency, or product-quality result is created.",
        "evaluator": {
            "classification": "internal_hash_bound_fixed_evaluation_contract_not_external_accelerator_benchmark",
            "dev": binding(DEV),
            "holdout": binding(HOLDOUT),
            "retention_amendment": binding(RETENTION),
            "retention_inputs": retention_inputs(),
            "scorer": binding(SCORER),
            "threshold_change_after_results_allowed": False,
        },
        "execution_interlocks": {
            "attempt_authority_granted": False,
            "attempt_marker_creation_authorized": False,
            "detached_launcher_required": True,
            "direct_stage_invocation_forbidden": True,
            "fresh_independent_l2_required": True,
            "official_namespace_creation_authorized": False,
            "training_authority_granted": False,
            "worker_terminal_status_required": True,
        },
        "material_difference": {
            "new_candidate_start": "fresh pinned source model",
            "new_mechanism": "all 494,032,768 model parameters train with FP32 master weights and BF16 compute",
            "parameter_efficient_adapter_used": False,
            "superseded_v6_dora_contract": binding(V6_CONTRACT),
            "v5_mechanism": "rank-8 standard adapter continuation from an earlier adapter",
        },
        "mechanism": training_recipe()["full_finetune"],
        "resource_assumptions": {
            "claim_boundary": "Conservative prerequisites, not measured V7 usage or latency.",
            "minimum_compute_capability": [8, 0],
            "minimum_free_disk_bytes": 34359738368,
            "minimum_free_vram_bytes_at_launch": 21474836480,
            "minimum_host_ram_bytes": 34359738368,
            "minimum_total_vram_bytes": 25769803776,
            "required_visible_cuda_devices": 1,
        },
        "schema_version": 2,
        "selector": {"candidate_epochs": [1, 2, 3], "dev_access_before_lock": False, "official_dev_checkpoint_count": 1, "official_dev_reselection_allowed": False, "probe_count": 28, "probe_hard_pass_minimum": 28, "training_loss_selects_checkpoint": False},
        "source_model": {"repository": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775", "source_contract": binding(SOURCE_CONTRACT)},
        "status": "FROZEN_MARKER_FREE_PENDING_FRESH_L2",
        "teacher": training_recipe()["teacher"],
    }


def frozen_config() -> dict[str, Any]:
    return {
        "claim_boundary": "Non-executing V7 full-parameter package configuration; grants no attempt authority and creates no official namespace, launch record, or marker.",
        "contract": binding(DATASET / "freeze_contract.json"),
        "dataset_manifest": binding(DATASET / "dataset_manifest.json"),
        "environment_lock": binding(DATASET / "environment.lock.json"),
        "model_namespace": "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7",
        "package_id": "qwen25_05b_bf16_full_finetune_product_v7",
        "recipe": binding(DATASET / "training_recipe.json"),
        "schema_version": 3,
        "teacher_mode": "none",
        "training_executable_included": False,
    }


def freeze_manifest() -> dict[str, Any]:
    artifacts = {}
    for path in (
        PACKAGE / "frozen_config.json",
        PACKAGE / "self_test.py",
        DATASET / "dataset_manifest.json",
        DATASET / "environment.lock.json",
        DATASET / "freeze_contract.json",
        DATASET / "patch.jsonl",
        DATASET / "synthetic_probes.jsonl",
        DATASET / "train.jsonl",
        DATASET / "training_recipe.json",
        RETENTION_PROMPT_MANIFEST,
        RETENTION_QUALITY_CONFIG,
        *[ROOT / name for name in retention_inputs()["task_sources"]],
    ):
        item = binding(path)
        artifacts[item["path"]] = {"bytes": item["bytes"], "sha256": item["sha256"]}
    return {
        "artifacts": artifacts,
        "attempt_authority": False,
        "attempt_marker_creation_authorized": False,
        "dataset_id": "qwen2.5-0.5b-instruct-bf16-full-finetune-product-v7",
        "detached_launch_created": False,
        "official_namespace_creation_authorized": False,
        "package_id": "qwen25_05b_bf16_full_finetune_product_v7",
        "schema_version": 3,
        "status": "FROZEN_MARKER_FREE_PENDING_FRESH_L2_NO_ATTEMPT_NO_TRAINING",
        "training_authority": False,
        "training_executed": False,
    }


def execution_contract() -> dict[str, Any]:
    recipe = load(DATASET / "training_recipe.json")
    return {
        "attempt_authority": {
            "attempt": 1,
            "attempt_marker_creation_authorized": False,
            "granted": False,
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "required_status_if_later_granted": "AUTHORIZE_SINGLE_V7_FULL_FINETUNE_ATTEMPT",
            "separate_fresh_operator_authority_required": True,
        },
        "claim_boundary": "Exact marker-free V7 full-finetune execution package only. No authority, launch, namespace, marker, training, evaluator, selected model, quantization, RTL, synthesis/PPA, U280, latency, quality, or completion claim.",
        "contract_id": "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7-execution-package-v1",
        "environment_resolution": {
            "package_versions_cuda_runtime_and_non_seed_launch_variables_source": binding(DATASET / "environment.lock.json"),
            "pythonhashseed_conflict": {
                "environment_lock_value": "26080705",
                "resolution": "No conflict; the V7 recipe and environment lock select the same value.",
                "selected_value": "26080705",
                "training_recipe_value": "26080705",
            },
        },
        "evaluator_contract": load(DATASET / "freeze_contract.json")["acceptance"],
        "execution_interlocks": {
            "direct_worker_rejected_before_namespace": True,
            "detached_posix_session_required": True,
            "stage_parent_must_match_detached_worker_pid": True,
            "terminal_status_written_after_authenticated_intent_on_startup_normal_error_or_signal_exit": True,
            "launcher_failure_after_intent_writes_launch_terminal_or_is_inferred_intent_only": True,
            "launcher_removes_python_safe_path_and_prepends_runner_to_pythonpath": True,
            "relaunch_after_intent_forbidden": True,
        },
        "execution_order": ["non_consuming_static_package_self_test", "fresh_independent_l2_exact_package_acceptance", "stop_marker_free_without_authority_launch_or_namespace"],
        "failure_policy": {"evaluator_no_execution_classified_separately": True, "first_genuine_failed_gate_stops_later_stages": True, "no_resume": True, "no_success_conclusion_from_incomplete_execution": True, "preserve_partial_evidence": True},
        "frozen_inputs": {
            "dataset_manifest": binding(DATASET / "dataset_manifest.json"),
            "environment_lock": binding(DATASET / "environment.lock.json"),
            "freeze_contract": binding(DATASET / "freeze_contract.json"),
            "freeze_manifest": binding(DATASET / "freeze_manifest.json"),
            "synthetic_probes": binding(DATASET / "synthetic_probes.jsonl"),
            "train": binding(DATASET / "train.jsonl"),
            "training_recipe": binding(DATASET / "training_recipe.json"),
        },
        "independent_execution_package_review": {"author_self_approval_allowed": False, "minimum_level": "L2", "path": L2.relative_to(ROOT).as_posix(), "required_before_any_authority_launch_namespace_or_marker": True, "required_status": "ACCEPTED_V7_FULL_FINETUNE_EXECUTION_PACKAGE_NO_ATTEMPT"},
        "mechanism": recipe["full_finetune"],
        "namespace": {"attempt": str((RUN_ROOT / "attempt-0001").relative_to(ROOT)), "dev": str((RUN_ROOT / "qualification/dev").relative_to(ROOT)), "holdout": str((RUN_ROOT / "qualification/holdout-exactly-once").relative_to(ROOT)), "model_root": str(RUN_ROOT.relative_to(ROOT)), "offline_root": str(OFFLINE.relative_to(ROOT)), "retention": str((RUN_ROOT / "qualification/retention").relative_to(ROOT)), "selected_bf16": str((RUN_ROOT / "selected/bf16").relative_to(ROOT))},
        "package": {"frozen_non_executing_package_preserved": str(PACKAGE.relative_to(ROOT)), "runner": str(RUNNER.relative_to(ROOT))},
        "resource_assumptions": load(DATASET / "freeze_contract.json")["resource_assumptions"],
        "retention_evaluator_inputs": retention_inputs(),
        "schema_version": 2,
        "source_and_evaluator": {"source_contract": binding(SOURCE_CONTRACT), "frozen_scorer": binding(SCORER), "retention_amendment": binding(RETENTION)},
        "source_teacher_data_evaluator": {"source_contract": binding(SOURCE_CONTRACT), "teacher": recipe["teacher"], "dataset_manifest": binding(DATASET / "dataset_manifest.json"), "frozen_scorer": binding(SCORER), "retention_amendment": binding(RETENTION)},
        "stage": "v7_full_finetune_execution_package",
        "status": "FROZEN_MARKER_FREE_PENDING_FRESH_INDEPENDENT_L2",
        "supersession": {"dora_or_other_peft_continuation_permitted": False, "preserved_superseded_v6_contract": binding(V6_CONTRACT)},
    }


def review_request() -> dict[str, Any]:
    assert_marker_free()
    manifest = load(MANIFEST)
    result = load(SELF_TEST)
    if result.get("status") != "PASS" or result.get("official_namespace_exists") is not False:
        raise RuntimeError("V7 self-test has not passed marker-free")
    return {
        "acceptance_requested": "ACCEPTED_V7_FULL_FINETUNE_EXECUTION_PACKAGE_NO_ATTEMPT",
        "claim_boundary": "Independent L2 review of the exact marker-free full-parameter package and detached-worker interlocks; no attempt authority or execution is requested.",
        "exact_package": {
            "contract": binding(CONTRACT),
            "dataset_manifest": binding(DATASET / "dataset_manifest.json"),
            "environment_lock": binding(DATASET / "environment.lock.json"),
            "execution_manifest": binding(MANIFEST),
            "execution_self_test": binding(SELF_TEST),
            "freeze_manifest": binding(DATASET / "freeze_manifest.json"),
            "training_recipe": binding(DATASET / "training_recipe.json"),
            "retention_prompt_manifest": binding(RETENTION_PROMPT_MANIFEST),
            "retention_quality_config": binding(RETENTION_QUALITY_CONFIG),
        },
        "execution_tree_sha256": manifest["tree_sha256"],
        "independent_from_constructor_required": True,
        "minimum_level": "L2",
        "official_namespace_exists": RUN_ROOT.exists(),
        "schema_version": 1,
        "status": "PENDING_INDEPENDENT_L2",
    }


def write_freeze() -> None:
    assert_marker_free()
    write(DATASET / "environment.lock.json", environment_lock())
    write(DATASET / "dataset_manifest.json", dataset_manifest())
    write(DATASET / "training_recipe.json", training_recipe())
    recipe = load(DATASET / "training_recipe.json")
    recipe["environment_lock"]["sha256"] = sha256(DATASET / "environment.lock.json")
    write(DATASET / "training_recipe.json", recipe)
    write(DATASET / "freeze_contract.json", freeze_contract())
    write(PACKAGE / "frozen_config.json", frozen_config())
    write(DATASET / "freeze_manifest.json", freeze_manifest())
    write(CONTRACT, execution_contract())


def check() -> None:
    assert_marker_free()
    expected = {
        DATASET / "environment.lock.json": environment_lock(),
        DATASET / "dataset_manifest.json": dataset_manifest(),
        DATASET / "training_recipe.json": training_recipe(),
    }
    expected[DATASET / "training_recipe.json"]["environment_lock"]["sha256"] = sha256(DATASET / "environment.lock.json")
    expected.update({
        DATASET / "freeze_contract.json": freeze_contract(),
        PACKAGE / "frozen_config.json": frozen_config(),
        DATASET / "freeze_manifest.json": freeze_manifest(),
        CONTRACT: execution_contract(),
    })
    if REQUEST.is_file():
        expected[REQUEST] = review_request()
    for path, value in expected.items():
        if path.read_bytes() != canonical(value):
            raise RuntimeError(f"frozen V7 artifact differs: {path}")
        companion = path.with_suffix(path.suffix + ".sha256")
        if companion.read_text(encoding="ascii").split() != [sha256(path), path.name]:
            raise RuntimeError(f"V7 sidecar differs: {path}")
    print("ACE2_FULL_FINETUNE_V7_FREEZE_REPRODUCES_NO_AUTHORITY_NO_LAUNCH_NO_NAMESPACE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-freeze", action="store_true")
    action.add_argument("--write-review-request", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write_freeze:
        write_freeze()
        print(f"ACE2_FULL_FINETUNE_V7_FREEZE_WRITTEN contract={sha256(CONTRACT)}")
    elif args.write_review_request:
        write(REQUEST, review_request())
        print(f"ACE2_FULL_FINETUNE_V7_L2_REQUEST_WRITTEN tree={load(MANIFEST)['tree_sha256']}")
    else:
        check()


if __name__ == "__main__":
    main()
