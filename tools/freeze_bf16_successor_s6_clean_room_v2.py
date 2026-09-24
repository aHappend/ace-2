#!/usr/bin/env python3
"""Construct the S6 clean-room successor from the four mounted inputs only."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any


INPUT_DIR = Path("/inputs")
OUTPUT_DIR = Path("/out")
FREEZE_DATE_UTC = "2026-08-08"

INPUTS = {
    "procedural-brief.json": {
        "original_path": "design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S6_CLEAN_ROOM_REPLACEMENT_TASK.json",
        "sha256": "6243dd3bbacdf35b572b0a7c402f00318b2d2ad572e76740616fd9df4d80148a",
        "role": "procedural_brief",
    },
    "s5-aggregate-terminal-audit.json": {
        "original_path": "build/bf16-full-finetune-successor-s5/backend-terminal-audit-20260808T150208Z.json",
        "sha256": "bd7bd5c0f55bbb63b9bd66473eb95dc0306380bdd96d53c5c71600b555c192b8",
        "role": "aggregate_quality_source",
    },
    "s5-public-execution-contract.json": {
        "original_path": "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S5_EXECUTION_PACKAGE_CONTRACT.json",
        "sha256": "6b9425f027bcc50f399fdded1ac91147fbdde91c9a041e10d1e00e6b3c9c0a54",
        "role": "public_evaluator_and_mechanism_source",
    },
    "s5-public-training-recipe.json": {
        "original_path": "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s5/training_recipe.json",
        "sha256": "24a4b8acf314c0bd90a026b0a2438c424ab281a0f29c9ab9e183e9b272b42df2",
        "role": "public_recipe_source",
    },
}

CATEGORIES = [
    "arithmetic",
    "concise_summary",
    "context_memory",
    "format_discipline",
    "polite_rewrite",
    "safe_refusal",
    "structured_extraction",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_unique_key(value: Any, key: str) -> Any:
    matches: list[Any] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for node_key, node_value in node.items():
                if node_key == key:
                    matches.append(node_value)
                visit(node_value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    if len(matches) != 1:
        raise ValueError(f"expected one {key!r}, found {len(matches)}")
    return matches[0]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, metadata in INPUTS.items():
        actual = sha256(INPUT_DIR / filename)
        if actual != metadata["sha256"]:
            raise ValueError(f"input hash mismatch for {filename}: {actual}")

    brief = load_json(INPUT_DIR / "procedural-brief.json")
    aggregate = load_json(INPUT_DIR / "s5-aggregate-terminal-audit.json")
    public_contract = load_json(INPUT_DIR / "s5-public-execution-contract.json")
    recipe = load_json(INPUT_DIR / "s5-public-training-recipe.json")

    permitted = {entry["path"]: entry for entry in brief["permitted_evidence_sources"]}
    for metadata in list(INPUTS.values())[1:]:
        source = permitted.get(metadata["original_path"])
        if source is None or source["sha256"] != metadata["sha256"]:
            raise ValueError(f"brief does not permit {metadata['original_path']}")

    dev = aggregate["quality"]["dev"]
    probe_selection = aggregate["quality"]["probe_selection"]
    selected_epoch = str(aggregate["quality"]["selected_epoch"])
    evaluator_dev = public_contract["evaluator_contract"]["dev"]
    mechanism = public_contract["mechanism"]

    data_counts = {
        key: find_unique_key(recipe, key)
        for key in ("additive_patch_count", "backbone_count", "synthetic_probe_count", "train_count")
    }
    public_recipe = {
        "checkpoint_lifecycle": recipe["checkpoint_lifecycle"],
        "checkpoint_policy": recipe["checkpoint_policy"],
        "data_counts": data_counts,
        "full_finetune": recipe["full_finetune"],
        "optimizer": recipe["optimizer"],
        "schedule": recipe["schedule"],
        "training_dtype_and_loss_scope": {
            key: find_unique_key(recipe, key)
            for key in ("compute_dtype", "loss_scope", "master_parameter_dtype")
        },
    }
    projection = {
        "schema_version": 1,
        "source_scope": "ONLY_FIELDS_PERMITTED_BY_CLEAN_ROOM_TASK",
        "aggregate": {
            "failure": {"classification": aggregate["failure"]["classification"]},
            "quality": {
                "dev": {
                    key: dev[key]
                    for key in (
                        "category_passes",
                        "category_totals",
                        "critical_safety_failures",
                        "gates",
                        "hard_pass_count",
                        "response_count",
                    )
                },
                "probe_selection": {
                    epoch: {
                        key: values[key]
                        for key in (
                            "category_minimum_gates_satisfied",
                            "category_passes",
                            "category_totals",
                            "critical_safety_failures",
                            "hard_pass_count",
                        )
                    }
                    for epoch, values in probe_selection.items()
                },
                "selected_epoch": aggregate["quality"]["selected_epoch"],
            },
        },
        "public_contract": {
            "evaluator_contract": {"dev": evaluator_dev},
            "mechanism": mechanism,
        },
        "public_recipe": public_recipe,
    }

    gate_deficits = {
        category: max(
            0,
            evaluator_dev["category_gates"][category]["minimum_hard_passes"]
            - dev["category_passes"][category],
        )
        for category in CATEGORIES
    }
    selected_probe = probe_selection[selected_epoch]
    probe_thresholds = {
        category: math.ceil(
            evaluator_dev["category_gates"][category]["minimum_hard_passes"]
            * data_counts["synthetic_probe_count"]
            / evaluator_dev["response_count"]
        )
        for category in CATEGORIES
    }
    probe_overall_threshold = math.ceil(
        evaluator_dev["minimum_hard_passes"]
        * data_counts["synthetic_probe_count"]
        / evaluator_dev["response_count"]
    )

    identity_material = "|".join(
        [metadata["sha256"] for metadata in INPUTS.values()]
        + ["marker-free-category-balanced-conflict-projected-full-finetune-v2"]
    )
    identity_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, identity_material))
    spec_id = (
        "qwen2.5-0.5b-instruct-ace2-bf16-successor-s6-clean-room-v2-"
        f"category-balanced-conflict-projected-{identity_uuid}"
    )

    source_anchors = [
        {
            "path": metadata["original_path"],
            "sha256": metadata["sha256"],
            "used_fields": (
                brief["permitted_evidence_sources"][index - 1]["allowed_fields"]
                if index > 0
                else ["procedural and acceptance fields"]
            ),
        }
        for index, metadata in enumerate(INPUTS.values())
    ]

    spec = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity_derivation": {
            "algorithm": "UUIDv5(NAMESPACE_URL, ordered input hashes plus mechanism label)",
            "uuid": identity_uuid,
            "clean_room_task_hash_included": True,
            "purpose": "make the identity source-bound and necessarily new relative to a predecessor created before this replacement task",
        },
        "freeze_date_utc": FREEZE_DATE_UTC,
        "status": "FROZEN_SPECIFICATION_ONLY_NO_CONSUMING_AUTHORITY",
        "stage": "specification",
        "claim_boundary": brief["claim_boundary"],
        "source_anchors": source_anchors,
        "aggregate_evidence_summary": {
            "s5_failure_classification": aggregate["failure"]["classification"],
            "official_dev": {
                "hard_pass_count": dev["hard_pass_count"],
                "minimum_hard_passes": evaluator_dev["minimum_hard_passes"],
                "hard_pass_shortfall": evaluator_dev["minimum_hard_passes"] - dev["hard_pass_count"],
                "response_count": dev["response_count"],
                "category_passes": dev["category_passes"],
                "category_gate_deficits": gate_deficits,
                "critical_safety_failures": dev["critical_safety_failures"],
                "gate_status": dev["gates"]["status"],
            },
            "selected_probe_epoch": {
                "epoch": int(selected_epoch),
                "hard_pass_count": selected_probe["hard_pass_count"],
                "category_minimum_gates_satisfied": selected_probe[
                    "category_minimum_gates_satisfied"
                ],
                "category_passes": selected_probe["category_passes"],
                "critical_safety_failures": selected_probe["critical_safety_failures"],
            },
            "aggregate_conflict_observation": {
                "safe_refusal": "selected probe 4/4 versus official dev 0/8",
                "concise_summary": "selected probe 0/4 versus official dev 8/8",
                "structured_extraction": "selected probe 0/4 versus official dev 8/8",
                "interpretation_boundary": "category aggregates justify conflict-robust training and stricter selection; they do not identify any row or response cause",
            },
        },
        "successor_mechanism": {
            "name": "marker-free category-balanced conflict-projected full fine-tuning",
            "material_difference_from_permitted_s5_mechanism": {
                "s5_declared_mechanism": mechanism,
                "new_capabilities": [
                    "each optimizer superstep contains every evaluator category",
                    "each category has equal weight after within-category loss averaging",
                    "negative inter-category gradient components are deterministically projected away before the optimizer update",
                    "checkpoint selection has no best-effort fallback when synthetic gates fail",
                ],
            },
            "model_parameterization": {
                "all_model_parameters_trainable": mechanism["all_model_parameters_trainable"],
                "architecture_change_allowed": mechanism["architecture_change_allowed"],
                "parameter_efficient_adapter_used": mechanism[
                    "parameter_efficient_adapter_used"
                ],
                "trainable_parameter_count": mechanism["trainable_parameter_count"],
            },
            "marker_policy": {
                "special_control_tokens_added": False,
                "category_label_in_model_input": False,
                "category_label_in_target": False,
                "score_or_gate_label_in_model_input_or_target": False,
                "metadata_transport": "external sampler sidecar removed before tokenization",
                "fail_closed_if_metadata_leaks_into_tokens": True,
            },
            "category_order": CATEGORIES,
            "optimizer_superstep": {
                "micro_batch_size": public_recipe["schedule"]["micro_batch_size"],
                "microbatches": public_recipe["schedule"]["gradient_accumulation_steps"],
                "required_base_microbatches": "one from each of the seven categories",
                "eighth_microbatch": "rotating category extra for 42 steps, then one bridge row for each of the final two steps",
                "within_category_rule": "average all row losses assigned to a category before taking that category gradient",
                "bridge_rule": "split a bridge-row loss equally across its declared categories; bridge metadata never enters tokens",
                "equal_category_weight_after_averaging": True,
            },
            "conflict_projection": {
                "arithmetic_dtype": "float32",
                "raw_gradient_symbol": "g_i for category i",
                "projection_rule": "for i then j in category_order, if dot(p_i,g_j)<0 set p_i=p_i-dot(p_i,g_j)/max(norm2(g_j),1e-12)*g_j",
                "initial_value": "p_i=g_i",
                "reference_gradient_rule": "use immutable raw g_j during all projections",
                "combined_gradient": "mean(p_i for all seven categories)",
                "post_combine_gradient_clipping_norm": public_recipe["optimizer"][
                    "gradient_clipping_norm"
                ],
                "deterministic_pair_order": True,
                "missing_category_disposition": "FAIL_CLOSED_NO_OPTIMIZER_STEP",
                "nonfinite_gradient_disposition": "FAIL_CLOSED_NO_OPTIMIZER_STEP",
            },
        },
        "future_data_contract": {
            "status": "NOT_MATERIALIZED_BY_THIS_FREEZE",
            "train_count": data_counts["train_count"],
            "backbone_count": data_counts["backbone_count"],
            "additive_patch_count": data_counts["additive_patch_count"],
            "category_assignment": {
                "backbone_per_category": data_counts["backbone_count"] // len(CATEGORIES),
                "category_specific_patch_per_category": (
                    data_counts["additive_patch_count"] - 2
                )
                // len(CATEGORIES),
                "single_category_rows_per_category": (
                    data_counts["backbone_count"]
                    + data_counts["additive_patch_count"]
                    - 2
                )
                // len(CATEGORIES),
                "cross_category_bridge_rows": 2,
                "single_category_balance_required": True,
                "category_vocabulary": CATEGORIES,
            },
            "row_requirements": {
                "new_rows_created_now": 0,
                "literal_row_content_in_spec": False,
                "single_category_rows_have_exactly_one_external_category": True,
                "bridge_rows_have_at_least_two_external_categories": True,
                "category_and_bridge_metadata_not_tokenized": True,
                "deduplicate_before_freeze": True,
                "reject_train_probe_or_dev_overlap": True,
                "reject_missing_category_metadata": True,
            },
            "future_freeze_requirements": [
                "freeze byte-exact rows, tokenizer, formatter, shuffle seed, and row-order manifest before training",
                "hash every data and metadata artifact before training",
                "keep synthetic probes and official dev disjoint from training and from each other",
                "obtain separate operator authority before creating any row or execution package",
            ],
        },
        "training_contract": {
            "full_finetune": public_recipe["full_finetune"],
            "compute_dtype": public_recipe["training_dtype_and_loss_scope"]["compute_dtype"],
            "master_parameter_dtype": public_recipe["training_dtype_and_loss_scope"][
                "master_parameter_dtype"
            ],
            "loss_scope": public_recipe["training_dtype_and_loss_scope"]["loss_scope"],
            "optimizer": public_recipe["optimizer"],
            "schedule": public_recipe["schedule"],
            "derived_epoch_layout": {
                "optimizer_steps_per_epoch": data_counts["train_count"]
                // public_recipe["schedule"]["gradient_accumulation_steps"],
                "seven_step_extra_category_cycle_count": 6,
                "bridge_steps_at_epoch_end": 2,
                "every_row_used_exactly_once_per_epoch": True,
            },
            "candidate_epochs": public_recipe["checkpoint_policy"]["candidate_epochs"],
            "save_at_every_epoch": public_recipe["checkpoint_policy"]["save_at_every_epoch"],
            "save_only_model": public_recipe["checkpoint_policy"]["save_only_model"],
            "training_execution_authorized": False,
        },
        "checkpoint_selection_contract": {
            "synthetic_probe_status": "NOT_MATERIALIZED_BY_THIS_FREEZE",
            "synthetic_probe_count": data_counts["synthetic_probe_count"],
            "synthetic_probe_per_category": data_counts["synthetic_probe_count"]
            // len(CATEGORIES),
            "category_minimum_hard_passes": probe_thresholds,
            "minimum_hard_passes": probe_overall_threshold,
            "critical_safety_failures_maximum": evaluator_dev[
                "critical_safety_failures_maximum"
            ],
            "selection_order": public_recipe["checkpoint_policy"]["candidate_epochs"],
            "selection_rule": "lock and stop at the first epoch satisfying the overall threshold, every category threshold, zero critical safety failures, and complete scoring",
            "no_passing_epoch_disposition": "NO_CHECKPOINT; DO_NOT_ACCESS_OFFICIAL_DEV",
            "training_loss_selects_checkpoint": False,
            "official_dev_selects_or_reselects_checkpoint": False,
            "threshold_change_after_results_allowed": evaluator_dev[
                "threshold_change_after_results_allowed"
            ],
            "missing_or_unscored_probe_disposition": "FAIL_CLOSED_NO_CHECKPOINT",
        },
        "official_dev_contract": {
            **evaluator_dev,
            "role": "QUALIFY_LOCKED_CHECKPOINT_ONLY",
            "access_before_checkpoint_lock": False,
            "failure_disposition": "NO_GO; NO_RESELECTION; NO_RESCORE",
            "execution_authorized_by_this_freeze": False,
        },
        "fail_closed_contract": {
            "source_hash_mismatch": "STOP_CONSTRUCTION",
            "data_or_metadata_hash_mismatch": "STOP_BEFORE_TRAINING",
            "category_imbalance": "STOP_BEFORE_TRAINING",
            "marker_leakage": "STOP_BEFORE_TRAINING",
            "missing_category_in_superstep": "SKIP_UPDATE_AND_FAIL_RUN",
            "nonfinite_loss_or_gradient": "SKIP_UPDATE_AND_FAIL_RUN",
            "missing_or_unscored_probe": "NO_CHECKPOINT",
            "no_probe_qualified_epoch": "NO_CHECKPOINT_AND_NO_OFFICIAL_DEV",
            "missing_or_unscored_dev_case": "NO_GO",
            "official_dev_gate_failure": "NO_GO_WITHOUT_RESELECTION_OR_RESCORE",
        },
        "no_downstream_boundary": {
            "dataset_rows_created": False,
            "synthetic_probe_rows_created": False,
            "runner_or_execution_package_created": False,
            "operator_authority_created": False,
            "submission_intent_created": False,
            "execution_namespace_created": False,
            "backend_experiment_or_job_created": False,
            "attempt_marker_created": False,
            "model_or_evaluator_executed": False,
            "replay_rescore_retention_or_holdout_executed": False,
            "w4a8_rtl_synthesis_ppa_fpga_u280_work_created": False,
            "stage_transition_created": False,
            "next_required_action": "independent Fresh Reviewer audit of aggregate-only provenance and zero consuming state",
        },
        "prohibited_content_attestation": {
            "response_text": False,
            "token_arrays": False,
            "row_level_scores": False,
            "checkpoint_bytes": False,
            "backend_logs": False,
            "hidden_harness": False,
            "golden_output": False,
            "rejected_s6_v1_content_read_or_reused": False,
        },
    }

    manifest = {
        "schema_version": 1,
        "context_id": "bf16-successor-s6-clean-room-v2-construction",
        "freeze_date_utc": FREEZE_DATE_UTC,
        "isolation": {
            "engine": "bubblewrap",
            "namespace_mode": "--unshare-all",
            "network_namespace": "isolated",
            "repository_root_visible": False,
            "project_evidence_mount": "/inputs",
            "construction_output_mount": "/out",
            "runtime_mounts": ["/usr", "/bin", "/lib", "/lib64", "/proc", "/dev", "/tmp"],
            "procedural_tool_mount": "/tool/construct.py",
        },
        "exposed_project_evidence": [
            {
                "capsule_path": f"/inputs/{filename}",
                "original_path": metadata["original_path"],
                "sha256": metadata["sha256"],
                "role": metadata["role"],
            }
            for filename, metadata in INPUTS.items()
        ],
        "exposed_project_evidence_count": len(INPUTS),
        "construction_policy": {
            "only_three_s5_sources": True,
            "aggregate_or_public_recipe_fields_only": True,
            "forbidden_construction_inputs_visible": False,
            "network_access": False,
            "consuming_state_allowed": False,
        },
        "outputs": [
            "aggregate-only-projection.json",
            "successor-design-freeze.json",
            "access-manifest.json",
        ],
    }

    write_json(OUTPUT_DIR / "aggregate-only-projection.json", projection)
    write_json(OUTPUT_DIR / "successor-design-freeze.json", spec)
    write_json(OUTPUT_DIR / "access-manifest.json", manifest)
    for filename in manifest["outputs"]:
        print(f"{sha256(OUTPUT_DIR / filename)}  {filename}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # fail closed with a single diagnostic
        print(f"construction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
