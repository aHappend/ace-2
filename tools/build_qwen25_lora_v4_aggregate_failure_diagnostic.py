#!/usr/bin/env python3
"""Build the aggregate-only V4 failure diagnosis and V5 recommendation.

This tool intentionally reads no official dev/holdout scored rows and performs
no model execution, checkpoint inference, training, or attempt consumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research/diagnostics/qwen25_lora_v4_aggregate_failure_diagnostic.json"
OUTPUT_SHA256 = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")

CATEGORIES = (
    "arithmetic",
    "concise_summary",
    "context_memory",
    "format_discipline",
    "polite_rewrite",
    "safe_refusal",
    "structured_extraction",
)

PATHS = {
    "v1_contract": ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V1_CONTRACT.json",
    "v1_aggregate": ROOT / "research/diagnostics/qwen25_lora_v1_aggregate_dev_baseline.json",
    "v1_precision": ROOT / "research/diagnostics/qwen25_lora_v1_v2_precision_checkpoint_audit.json",
    "v1_adapter_config": ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v1/attempt-0001/"
    "checkpoints/checkpoint-18/adapter_config.json",
    "v1_adapter": ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v1/attempt-0001/"
    "checkpoints/checkpoint-18/adapter_model.safetensors",
    "v1_train": ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/train.jsonl",
    "v3_diagnostic": ROOT / "research/diagnostics/qwen25_lora_v3_aggregate_failure_diagnostic.json",
    "v4_recipe": ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v4/training_recipe.json",
    "v4_dataset": ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v4/dataset_manifest.json",
    "v4_runner": ROOT / "pilot/qwen25_05b_bf16_lora_product_v4_runner/train.py",
    "v4_training": ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/attempt-0001/training_result.json",
    "v4_lock": ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/attempt-0001/checkpoint_lock.json",
    "v4_dev": ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/qualification/dev/RESULT.json",
    "v4_terminal": ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/TERMINAL_NO_GO.json",
    "v4_review": ROOT
    / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v4-terminal-fresh-review.json",
}

PROBE_SUMMARIES = tuple(
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/attempt-0001/"
    f"probe-selection/epoch-{epoch}/summary.json"
    for epoch in range(1, 7)
)

FORBIDDEN_INPUT_FRAGMENTS = (
    "qualification/dev/scored_rows.jsonl",
    "qualification/holdout",
    "dev.jsonl",
    "holdout.jsonl",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required input is absent: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def gate_vector(category_passes: dict[str, int], gates: dict[str, dict[str, int]]) -> dict[str, bool]:
    return {
        category: category_passes[category] >= gates[category]["minimum_hard_passes"]
        for category in CATEGORIES
    }


def build() -> dict[str, object]:
    all_inputs = [*PATHS.values(), *PROBE_SUMMARIES]
    for path in all_inputs:
        relative = path.relative_to(ROOT).as_posix()
        require(
            not any(fragment in relative for fragment in FORBIDDEN_INPUT_FRAGMENTS),
            f"forbidden protected input configured: {relative}",
        )

    v1_contract = load_json(PATHS["v1_contract"])
    v1 = load_json(PATHS["v1_aggregate"])
    v1_precision = load_json(PATHS["v1_precision"])
    v1_config = load_json(PATHS["v1_adapter_config"])
    v3 = load_json(PATHS["v3_diagnostic"])
    v4_recipe = load_json(PATHS["v4_recipe"])
    v4_dataset = load_json(PATHS["v4_dataset"])
    v4_training = load_json(PATHS["v4_training"])
    v4_lock = load_json(PATHS["v4_lock"])
    v4_dev = load_json(PATHS["v4_dev"])
    v4_terminal = load_json(PATHS["v4_terminal"])
    v4_review = load_json(PATHS["v4_review"])
    probes = [load_json(path) for path in PROBE_SUMMARIES]
    v4_runner = PATHS["v4_runner"].read_text(encoding="utf-8")

    v1_best = v1["best_candidate"]
    v3_epochs = v3["dev_aggregate"]["epochs"]
    v3_best = max(v3_epochs, key=lambda item: item["hard_pass_count"])
    selected_epoch = int(v4_lock["checkpoint_epoch"])
    selected_probe = probes[selected_epoch - 1]
    dev_gates = v1_contract["acceptance"]["dev_selection"].get("category_gates")
    if dev_gates is None:
        dev_gates = load_json(
            ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"
        )["acceptance"]["dev"]["category_gates"]

    require(v1_best["epoch"] == 1 and v1_best["hard_pass_count"] == 40, "V1 baseline changed")
    require(v3_best["epoch"] == 1 and v3_best["hard_pass_count"] == 33, "V3 aggregate changed")
    require([item["hard_pass_count"] for item in probes] == [10, 13, 13, 15, 13, 14], "V4 probe sequence changed")
    require(selected_epoch == 4 and selected_probe["hard_pass_count"] == 15, "V4 lock changed")
    require(v4_dev["hard_pass_count"] == 36 and v4_dev["status"] == "NO_GO", "V4 dev result changed")
    require(v4_terminal["failure_taxonomy"] == "DEV_QUALITY_GATE_FAILURE", "V4 taxonomy changed")
    require(v4_review["review"]["status"] == "done", "V4 Fresh Review is not done")
    require(v4_training["optimizer_steps"] == 264, "V4 optimizer-step count changed")
    require(v4_training["trainable_parameter_count"] == 8_798_208, "V4 trainable count changed")
    require(v4_recipe["lora"]["rank"] == 16 and v4_recipe["lora"]["scaling_alpha"] == 32, "V4 LoRA geometry changed")
    require(v4_recipe["optimizer"]["learning_rate"] == 4e-5, "V4 learning rate changed")
    require(v4_dataset["counts"] == {"additive_patch": 72, "synthetic_probes": 28, "train": 352, "v1_backbone": 280}, "V4 dataset counts changed")
    require("result = trainer.train(resume_from_checkpoint=None)" in v4_runner, "V4 no-resume source fact changed")
    require("model = get_peft_model(" in v4_runner, "V4 fresh-adapter construction fact changed")

    v1_checkpoint = v1_precision["observed_training"]["checkpoints"]["v1"]["18"]
    require(
        v1_checkpoint["adapter"]["dtype_elements"] == {"torch.float32": 4_399_104},
        "V1 epoch-1 adapter is not the bound FP32 checkpoint",
    )
    require(v1_config["r"] == 8 and v1_config["lora_alpha"] == 16, "V1 adapter geometry changed")

    probe_checks = selected_probe["gates"]["checks"]
    probe_gate_vector = {
        category: bool(probe_checks[f"{category}.minimum_hard_passes"])
        for category in CATEGORIES
    }
    dev_gate_vector = gate_vector(v4_dev["category_passes"], dev_gates)
    category_comparison = {
        category: {
            "dev_gate_pass": dev_gate_vector[category],
            "dev_passes": v4_dev["category_passes"][category],
            "dev_total": v4_dev["category_totals"][category],
            "gate_direction_matches": probe_gate_vector[category] == dev_gate_vector[category],
            "probe_gate_pass": probe_gate_vector[category],
            "probe_passes": selected_probe["category_passes"][category],
            "probe_total": selected_probe["category_totals"][category],
            "v4_dev_minus_v1_dev": (
                v4_dev["category_passes"][category]
                - v1_best["category_passes"][category]
            ),
        }
        for category in CATEGORIES
    }
    mismatch_categories = [
        category
        for category, comparison in category_comparison.items()
        if not comparison["gate_direction_matches"]
    ]

    patch_count = v4_dataset["counts"]["additive_patch"]
    train_count = v4_dataset["counts"]["train"]
    v4_additions = v4_dataset["category_counts"]["additive_patch"]
    proposed_patch = {"arithmetic": 8, "format_discipline": 16, "safe_refusal": 16}
    proposed_patch_count = sum(proposed_patch.values())
    proposed_rehearsal_count = 280
    proposed_train_count = proposed_rehearsal_count + proposed_patch_count

    source_aggregate = v1["source"]
    source_gate_vector = gate_vector(source_aggregate["category_passes"], dev_gates)
    v1_gate_vector = gate_vector(v1_best["category_passes"], dev_gates)

    evidence_bindings = {
        name: binding(path)
        for name, path in PATHS.items()
    }
    evidence_bindings["v4_probe_summaries"] = [binding(path) for path in PROBE_SUMMARIES]

    return {
        "schema_version": 1,
        "diagnostic_id": "qwen25-lora-v4-terminal-aggregate-failure-diagnostic-v1",
        "status": "RECOMMEND_V5_SPECIFICATION_ONLY_PENDING_INDEPENDENT_L2",
        "failure_taxonomy": "DEV_QUALITY_GATE_FAILURE",
        "evidence_boundary": {
            "aggregate_dev_summaries_read": True,
            "official_dev_scored_rows_read": False,
            "official_dev_or_holdout_prompts_answers_or_responses_read": False,
            "holdout_accessed": False,
            "checkpoint_inference_replayed": False,
            "training_resumed_repaired_or_started": False,
            "attempt_namespace_or_consumption_marker_created": False,
            "rtl_quantization_synthesis_ppa_or_u280_work_performed": False,
            "fresh_inputs_used": [
                "already_frozen V4 disjoint synthetic-probe aggregate summaries"
            ],
        },
        "evidence_bindings": evidence_bindings,
        "observed_aggregate_facts": {
            "v1": {
                "best_epoch": 1,
                "hard_pass_count": 40,
                "response_count": 56,
                "category_passes": v1_best["category_passes"],
                "rank": 8,
                "scaling_alpha": 16,
                "learning_rate": 1e-4,
                "initialization": "fresh LoRA adapter on the frozen source model",
            },
            "v3": {
                "best_epoch": int(v3_best["epoch"]),
                "hard_pass_count": int(v3_best["hard_pass_count"]),
                "response_count": 56,
                "category_passes": v3_best["category_passes"],
                "rank": 8,
                "scaling_alpha": 16,
                "learning_rate": 8e-5,
                "initialization": "fresh LoRA adapter on the frozen source model",
            },
            "v4": {
                "probe_hard_pass_counts": [item["hard_pass_count"] for item in probes],
                "probe_locked_epoch": selected_epoch,
                "locked_probe_hard_pass_count": selected_probe["hard_pass_count"],
                "locked_probe_passed_its_own_gates": selected_probe["status"] == "PASS",
                "selection_used_fallback": selected_probe["status"] != "PASS",
                "dev_hard_pass_count": v4_dev["hard_pass_count"],
                "dev_response_count": v4_dev["response_count"],
                "dev_category_passes": v4_dev["category_passes"],
                "rank": v4_recipe["lora"]["rank"],
                "scaling_alpha": v4_recipe["lora"]["scaling_alpha"],
                "learning_rate": v4_recipe["optimizer"]["learning_rate"],
                "initialization": "fresh LoRA adapter on the frozen source model; no V1 adapter initialization",
                "patch_rows": patch_count,
                "train_rows": train_count,
                "patch_fraction": round(patch_count / train_count, 12),
                "patch_category_additions": v4_additions,
            },
            "comparisons": {
                "v4_dev_minus_v1_best": v4_dev["hard_pass_count"] - v1_best["hard_pass_count"],
                "v4_dev_minus_v3_best": v4_dev["hard_pass_count"] - v3_best["hard_pass_count"],
                "selected_probe_to_dev_category_comparison": category_comparison,
                "selected_probe_to_dev_gate_mismatch_count": len(mismatch_categories),
                "selected_probe_to_dev_gate_mismatch_categories": mismatch_categories,
            },
        },
        "failure_mechanism_assessment": {
            "selector_to_gate_misalignment": {
                "classification": "SUPPORTED_HIGH",
                "facts": [
                    "No V4 epoch passed the frozen synthetic-probe gate, so checkpoint selection used fallback rather than an accepted probe candidate.",
                    f"The locked epoch disagreed with official-dev category-gate direction in {len(mismatch_categories)} of seven categories.",
                    "The probe reported safe_refusal=4/4 while official dev reported 0/8.",
                    "The probe reported concise_summary=0/4 and structured_extraction=0/4 while official dev reported 8/8 for both.",
                ],
                "conclusion": "The V4 probe set and fallback ranking were not a reliable proxy for the unchanged official category gates.",
            },
            "rank_alpha_learning_rate_package": {
                "classification": "INCONCLUSIVE_NO_SUPPORT_FOR_FURTHER_CAPACITY_INCREASE",
                "facts": [
                    "V4 changed rank/alpha from 8/16 to 16/32 and learning rate from V3's 8e-5 to 4e-5 while also changing data and checkpoint selection.",
                    "V4 improved only three aggregate dev passes over V3 and remained four passes below V1, which used rank/alpha 8/16.",
                ],
                "conclusion": "Visible evidence cannot assign the V4 outcome to rank, alpha, or learning rate, and does not justify another rank increase.",
            },
            "additive_patch_mixture": {
                "classification": "NO_OBSERVED_TARGET_BENEFIT_CAUSAL_INTERFERENCE_UNRESOLVED",
                "facts": [
                    "The 72-row V4 patch was 20.4545% of training rows and added 24 format-discipline plus 24 safe-refusal rows.",
                    "Relative to V1 best aggregate dev, V4 format discipline changed from 3/8 to 2/8 and safe refusal remained 0/8; arithmetic changed from 5/8 to 2/8.",
                    "V4 simultaneously restarted from a fresh rank-16 adapter and used a different learning rate and selector.",
                ],
                "conclusion": "The patch delivered no visible targeted dev improvement under V4, but aggregate evidence cannot isolate destructive mixture interference from the simultaneous changes.",
            },
        },
        "v5_recommendation": {
            "candidate_id": "qwen2.5-0.5b-instruct-bf16-lora-product-v5-staged-v1-parent-r8a16",
            "status": "RECOMMENDATION_ONLY_NOT_FROZEN_NOT_AUTHORIZED",
            "material_mechanism_change": "continue from the immutable V1 epoch-1 FP32 adapter instead of relearning a fresh adapter from the source model",
            "parent": {
                "epoch": 1,
                "adapter": binding(PATHS["v1_adapter"]),
                "adapter_config": binding(PATHS["v1_adapter_config"]),
                "adapter_dtype": "float32",
                "rank": 8,
                "scaling_alpha": 16,
                "target_modules": sorted(v1_config["target_modules"]),
                "historical_dev_hard_pass_count": 40,
                "historical_dev_result_is_evidence_only_and_must_not_be_replayed": True,
            },
            "prospective_selector": {
                "must_be_finalized_and_hash_bound_before_training": True,
                "response_count": 56,
                "category_totals": {category: 8 for category in CATEGORIES},
                "category_minimum_hard_passes": {
                    category: dev_gates[category]["minimum_hard_passes"]
                    for category in CATEGORIES
                },
                "minimum_hard_passes": 48,
                "critical_safety_failures_maximum": 0,
                "calibration_anchors": {
                    "source": {
                        "historical_category_gate_vector": source_gate_vector,
                        "historical_hard_pass_count": source_aggregate["hard_pass_count"],
                    },
                    "v1_epoch_1": {
                        "historical_category_gate_vector": v1_gate_vector,
                        "historical_hard_pass_count": v1_best["hard_pass_count"],
                    },
                },
                "calibration_acceptance": {
                    "minimum_matching_category_gate_directions_per_anchor": 6,
                    "anchor_count": 2,
                    "required_before_selector_freeze": True,
                    "uses_only_fresh_disjoint_synthetic_inputs_and_visible_aggregate_labels": True,
                },
                "candidate_order": [
                    "first candidate satisfying every selector gate",
                    "higher hard-pass count",
                    "higher minimum category pass rate",
                    "earlier checkpoint",
                ],
                "fallback_if_no_candidate_passes": "PREDEV_NO_GO_WITH_NO_OFFICIAL_DEV_ACCESS",
                "official_dev_selects_or_reselects_checkpoint": False,
            },
            "proposed_training_bounds": {
                "status": "proposal_requires_independent_L2_and_separate_freeze_authority",
                "initialization": "load the exact V1 epoch-1 adapter as trainable; do not create a fresh adapter",
                "rank": 8,
                "scaling_alpha": 16,
                "trainable_parameter_count": 4_399_104,
                "adapter_and_optimizer_state_dtype": "float32",
                "base_compute": "bfloat16",
                "sequence_length": 512,
                "truncation_allowed": False,
                "rehearsal": {
                    "rows": proposed_rehearsal_count,
                    "source": binding(PATHS["v1_train"]),
                    "byte_identical_full_v1_corpus_required": True,
                },
                "new_patch": {
                    "maximum_rows": proposed_patch_count,
                    "proposed_category_counts": proposed_patch,
                    "all_targets_must_pass_frozen_scorer": True,
                    "dev_or_holdout_content_or_outputs_allowed": False,
                },
                "maximum_total_rows": proposed_train_count,
                "maximum_patch_fraction": round(proposed_patch_count / proposed_train_count, 12),
                "optimizer": {
                    "learning_rate": 1e-5,
                    "learning_rate_sweep_allowed": False,
                    "optimizer": "adamw_torch",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_epsilon": 1e-8,
                    "weight_decay": 0.01,
                    "gradient_clipping_norm": 1.0,
                },
                "schedule": {
                    "epochs_maximum": 1,
                    "global_batch_size": 16,
                    "micro_batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "candidate_checkpoints": ["half_epoch", "full_epoch"],
                    "additional_epochs_or_post_dev_training_allowed": False,
                },
            },
            "ordered_execution_if_later_authorized": [
                "construct and hash the small patch plus 56-case selector without reading official dev or holdout rows",
                "calibrate the selector on fresh source and V1-parent synthetic inference, then freeze it",
                "obtain independent L2 acceptance of the exact data, selector, recipe, runner, and interlocks",
                "obtain separate operator authority before creating one fresh V5 attempt marker",
                "train at most one epoch and lock only a checkpoint that passes every synthetic selector gate",
                "if no checkpoint passes, stop PREDEV_NO_GO without official dev access",
                "evaluate exactly one locked checkpoint on official dev; stop at the first failed ordered gate",
            ],
            "unchanged_official_acceptance": {
                "dev_minimum_hard_passes": 48,
                "dev_category_gates": dev_gates,
                "critical_safety_failures_maximum": 0,
                "retention_and_exactly_once_holdout_order_unchanged": True,
            },
        },
        "authority_and_review": {
            "independent_l2_required": True,
            "author_self_approval_allowed": False,
            "recommendation_is_an_execution_contract": False,
            "v5_dataset_or_selector_frozen": False,
            "v5_namespace_creation_authorized": False,
            "attempt_marker_creation_authorized": False,
            "training_authorized": False,
            "official_dev_access_authorized": False,
            "downstream_quantization_rtl_synthesis_ppa_or_u280_authorized": False,
        },
        "prohibited_conclusions": [
            "that V4's isolated failure cause is known",
            "that the 72-row patch alone caused regression",
            "that rank 16 or learning rate 4e-5 alone caused or repaired the outcome",
            "that the proposed V5 values are frozen, reviewed, authorized, trained, or accepted",
            "that any BF16 checkpoint, quantized reference, accelerator-backed chat demo, or U280 deployment now exists",
        ],
    }


def render() -> str:
    return json.dumps(build(), indent=2, sort_keys=True) + "\n"


def check() -> bool:
    expected = render()
    if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != expected:
        return False
    expected_companion = f"{hashlib.sha256(expected.encode()).hexdigest()}  {OUTPUT.name}\n"
    return OUTPUT_SHA256.is_file() and OUTPUT_SHA256.read_text(encoding="utf-8") == expected_companion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        passed = check()
        print("PASS" if passed else "FAIL")
        return 0 if passed else 1
    print(render(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
