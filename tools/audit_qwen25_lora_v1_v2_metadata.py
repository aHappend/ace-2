#!/usr/bin/env python3
"""Reproduce the aggregate-only V1/V2 precision and checkpoint audit.

This tool reads only immutable contracts, recipes, runner sources, aggregate
summaries, training logs, and checkpoint metadata. It never opens dev/holdout
inputs, generated responses, or scored rows, and it performs no model inference
or training.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V1_RUN = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v1/attempt-0001"
V2_RUN = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v2/cpu-attempt-0001"
V2_QUALIFICATION = (
    ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v2/qualification/dev"
)

V1_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V1_CONTRACT.json"
V2_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"
V2_RECIPE = (
    ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2/training_recipe.json"
)
V1_ENVIRONMENT = ROOT / "pilot/qwen25_05b_bf16_lora_product_v1/environment.lock.json"
V2_ENVIRONMENT = (
    ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2/environment.lock.json"
)
V1_TRAIN_SOURCE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v1/train.py"
V2_TRAIN_SOURCE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/train.py"
V1_RUNTIME_SOURCE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v1/common.py"
V2_RUNTIME_SOURCE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/runtime.py"

V1_STEPS = (18, 36, 54)
V2_STEPS = (108, 135, 162)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="write the deterministic audit and SHA-256 sidecar")
    mode.add_argument("--verify", type=Path, help="recompute and verify an existing audit and sidecar")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {relative(path)}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def inspect_adapter(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    dtype_elements: collections.Counter[str] = collections.Counter()
    element_count = 0
    tensor_count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            dtype_elements[str(tensor.dtype)] += tensor.numel()
            element_count += tensor.numel()
            tensor_count += 1
    return {
        "dtype_elements": dict(sorted(dtype_elements.items())),
        "element_count": element_count,
        "sha256": sha256_file(path),
        "tensor_count": tensor_count,
    }


def inspect_optimizer(path: Path) -> dict[str, Any]:
    import torch

    optimizer = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(optimizer, dict), f"invalid optimizer object: {relative(path)}")
    states = optimizer.get("state")
    groups = optimizer.get("param_groups")
    require(isinstance(states, dict), f"optimizer state missing: {relative(path)}")
    require(isinstance(groups, list), f"optimizer parameter groups missing: {relative(path)}")

    dtype_element_counts: collections.Counter[str] = collections.Counter()
    dtype_tensor_counts: collections.Counter[str] = collections.Counter()
    for state in states.values():
        require(isinstance(state, dict), f"invalid optimizer state entry: {relative(path)}")
        for name, value in state.items():
            if torch.is_tensor(value):
                key = f"{name}:{value.dtype}"
                dtype_tensor_counts[key] += 1
                dtype_element_counts[key] += value.numel()
    return {
        "dtype_element_counts": dict(sorted(dtype_element_counts.items())),
        "dtype_tensor_counts": dict(sorted(dtype_tensor_counts.items())),
        "group_learning_rates": [group.get("lr") for group in groups],
        "sha256": sha256_file(path),
        "state_entries": len(states),
    }


def inspect_checkpoints(base: Path, steps: tuple[int, ...]) -> dict[str, Any]:
    checkpoints: dict[str, Any] = {}
    for step in steps:
        directory = base / "checkpoints" / f"checkpoint-{step}"
        state_path = directory / "trainer_state.json"
        state = read_json(state_path)
        checkpoints[str(step)] = {
            "adapter": inspect_adapter(directory / "adapter_model.safetensors"),
            "epoch": float(state["epoch"]),
            "global_step": int(state["global_step"]),
            "optimizer": inspect_optimizer(directory / "optimizer.pt"),
            "trainer_state_sha256": sha256_file(state_path),
        }
    return checkpoints


def adapter_delta(base: Path, first_step: int, second_step: int) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    first_path = base / "checkpoints" / f"checkpoint-{first_step}" / "adapter_model.safetensors"
    second_path = base / "checkpoints" / f"checkpoint-{second_step}" / "adapter_model.safetensors"
    changed_elements = 0
    total_elements = 0
    nonzero_deltas: list[Any] = []
    with safe_open(first_path, framework="pt", device="cpu") as first, safe_open(
        second_path, framework="pt", device="cpu"
    ) as second:
        require(list(first.keys()) == list(second.keys()), "adapter tensor names differ")
        for name in first.keys():
            delta = (second.get_tensor(name).float() - first.get_tensor(name).float()).abs().reshape(-1)
            total_elements += delta.numel()
            changed_elements += int(torch.count_nonzero(delta))
            nonzero = delta[delta != 0]
            if nonzero.numel():
                nonzero_deltas.append(nonzero)
    require(total_elements > 0, "adapter has no elements")
    joined = torch.cat(nonzero_deltas) if nonzero_deltas else torch.empty(0)
    changed_fraction = changed_elements / total_elements
    return {
        "changed_elements": changed_elements,
        "changed_fraction": changed_fraction,
        "from_step": first_step,
        "nonzero_abs_delta_max": float(joined.max()) if joined.numel() else None,
        "nonzero_abs_delta_median": float(joined.median()) if joined.numel() else None,
        "nonzero_abs_delta_min": float(joined.min()) if joined.numel() else None,
        "to_step": second_step,
        "total_elements": total_elements,
        "unchanged_fraction": 1.0 - changed_fraction,
    }


def training_summary(path: Path) -> dict[str, Any]:
    result = read_json(path)
    loss_rows = result["loss"]["optimizer_step_losses"]
    if loss_rows and isinstance(loss_rows[0], dict):
        rows = loss_rows
    else:
        rows = [item for item in result["trainer_log_history"] if "loss" in item]
    by_epoch: dict[int, list[float]] = {}
    for item in rows:
        epoch = int(math.ceil(float(item["epoch"]) - 1e-12))
        by_epoch.setdefault(epoch, []).append(float(item["loss"]))
    return {
        "mean_optimizer_step_loss_by_epoch": {
            str(epoch): sum(values) / len(values) for epoch, values in sorted(by_epoch.items())
        },
        "optimizer_steps": int(result["optimizer_steps"]),
        "sha256": sha256_file(path),
        "trainable_parameter_dtypes": result["trainable_parameter_dtypes"],
        "trainer_train_loss": float(result["loss"]["trainer_train_loss"]),
    }


def source_semantics() -> dict[str, Any]:
    v1_train = V1_TRAIN_SOURCE.read_text(encoding="utf-8")
    v2_train = V2_TRAIN_SOURCE.read_text(encoding="utf-8")
    v1_runtime = V1_RUNTIME_SOURCE.read_text(encoding="utf-8")
    v2_runtime = V2_RUNTIME_SOURCE.read_text(encoding="utf-8")
    require("model = get_peft_model(base, lora_config)" in v1_train, "V1 PEFT construction changed")
    require("autocast_adapter_dtype=False" in v2_train, "V2 BF16 adapter override missing")
    require("trainable_dtypes == {torch.bfloat16}" in v2_train, "V2 BF16 adapter assertion missing")
    require("dtype=torch.bfloat16" in v1_runtime, "V1 BF16 base load missing")
    require("dtype=torch.bfloat16" in v2_runtime, "V2 BF16 base load missing")
    return {
        "base_compute": {"v1": "bfloat16", "v2": "bfloat16"},
        "peft_adapter_construction": {
            "v1": "default autocast_adapter_dtype behavior; observed checkpoints are FP32",
            "v2": "autocast_adapter_dtype=False plus an explicit all-BF16 trainable-parameter assertion",
        },
    }


def v2_aggregate_qualification() -> dict[str, Any]:
    epochs: dict[str, Any] = {}
    for epoch in (4, 5, 6):
        summary_path = V2_QUALIFICATION / f"epoch-{epoch}" / "summary.json"
        summary = read_json(summary_path)
        epochs[str(epoch)] = {
            "category_passes": summary["category_passes"],
            "critical_safety_failures": summary["critical_safety_failures"],
            "hard_pass_count": summary["hard_pass_count"],
            "sha256": sha256_file(summary_path),
            "status": summary["status"],
        }
    selection_path = V2_QUALIFICATION / "selection.json"
    selection = read_json(selection_path)
    return {
        "epochs": epochs,
        "selected_epoch": selection["selected_epoch"],
        "selection_sha256": sha256_file(selection_path),
        "status": selection["status"],
    }


def evidence_manifest() -> dict[str, str]:
    paths = [
        V1_CONTRACT,
        V2_CONTRACT,
        V2_RECIPE,
        V1_ENVIRONMENT,
        V2_ENVIRONMENT,
        V1_TRAIN_SOURCE,
        V2_TRAIN_SOURCE,
        V1_RUNTIME_SOURCE,
        V2_RUNTIME_SOURCE,
        V1_RUN / "training_result.json",
        V2_RUN / "training_result.json",
        V2_QUALIFICATION / "selection.json",
    ]
    paths.extend(V2_QUALIFICATION / f"epoch-{epoch}" / "summary.json" for epoch in (4, 5, 6))
    for base, steps in ((V1_RUN, V1_STEPS), (V2_RUN, V2_STEPS)):
        for step in steps:
            checkpoint = base / "checkpoints" / f"checkpoint-{step}"
            paths.extend(
                (
                    checkpoint / "adapter_model.safetensors",
                    checkpoint / "optimizer.pt",
                    checkpoint / "trainer_state.json",
                )
            )
    return {relative(path): sha256_file(path) for path in sorted(paths)}


def build_report() -> dict[str, Any]:
    v1_contract = read_json(V1_CONTRACT)
    v2_contract = read_json(V2_CONTRACT)
    v2_recipe = read_json(V2_RECIPE)
    v1_environment = read_json(V1_ENVIRONMENT)
    v2_environment = read_json(V2_ENVIRONMENT)
    require(v1_environment["packages"] == v2_environment["packages"], "V1/V2 package locks differ")
    require(v1_contract["source_model"]["revision"] == v2_contract["source_model"]["revision"], "source revisions differ")

    v1_checkpoints = inspect_checkpoints(V1_RUN, V1_STEPS)
    v2_checkpoints = inspect_checkpoints(V2_RUN, V2_STEPS)
    v1_deltas = [adapter_delta(V1_RUN, 18, 36), adapter_delta(V1_RUN, 36, 54)]
    v2_deltas = [adapter_delta(V2_RUN, 108, 135), adapter_delta(V2_RUN, 135, 162)]

    trainable_count = v1_contract["pilot"]["lora"]["trainable_parameter_count"]
    require(trainable_count == v2_recipe["lora"]["trainable_parameter_count"], "trainable counts differ")
    require(
        all(item["adapter"]["element_count"] == trainable_count for item in v1_checkpoints.values()),
        "V1 adapter element count differs",
    )
    require(
        all(item["adapter"]["element_count"] == trainable_count for item in v2_checkpoints.values()),
        "V2 adapter element count differs",
    )

    return {
        "diagnostic_id": "qwen25-lora-v1-v2-precision-checkpoint-audit-v1",
        "evidence_boundary": {
            "allowed": [
                "immutable contracts and recipes",
                "runner source and environment locks",
                "aggregate dev summaries",
                "training logs and trainer-state metadata",
                "adapter and optimizer tensor metadata",
            ],
            "excluded_and_not_opened": [
                "raw dev or holdout prompts and answers",
                "scored_rows.jsonl and per-case semantics",
                "V1/V2 reruns or edits",
                "training, inference, checkpoint selection, SPEC, RTL, quantization, and U280",
            ],
            "fresh_synthetic_probes_executed": 0,
        },
        "evidence_manifest_sha256": evidence_manifest(),
        "observed_configuration": {
            "package_lock_equal": True,
            "packages": v1_environment["packages"],
            "source_model_revision": v1_contract["source_model"]["revision"],
            "source_semantics": source_semantics(),
            "trainable_parameter_count": trainable_count,
            "v1": {
                "candidate_epochs": v1_contract["pilot"]["checkpoint_epochs"],
                "epochs": v1_contract["pilot"]["schedule"]["epochs"],
                "gradient_accumulation_steps": v1_contract["pilot"]["schedule"]["gradient_accumulation_steps"],
                "learning_rate": v1_contract["pilot"]["optimizer"]["learning_rate"],
                "lora_dropout": v1_contract["pilot"]["lora"]["dropout"],
                "micro_batch_size": v1_contract["pilot"]["schedule"]["micro_batch_size"],
                "sequence_length": v1_contract["pilot"]["loss"]["sequence_length"],
                "train_count": v1_contract["dataset"]["train_count"],
            },
            "v2": {
                "candidate_epochs": v2_recipe["checkpoint_policy"]["candidate_epochs"],
                "epochs": v2_recipe["schedule"]["epochs"],
                "gradient_accumulation_steps": v2_recipe["schedule"]["gradient_accumulation_steps"],
                "learning_rate": v2_recipe["optimizer"]["learning_rate"],
                "lora_dropout": v2_recipe["lora"]["dropout"],
                "micro_batch_size": v2_recipe["schedule"]["micro_batch_size"],
                "sequence_length": v2_recipe["data"]["sequence_length"],
                "train_count": v2_recipe["data"]["train_count"],
            },
        },
        "observed_training": {
            "adapter_deltas": {"v1": v1_deltas, "v2": v2_deltas},
            "checkpoints": {"v1": v1_checkpoints, "v2": v2_checkpoints},
            "training_results": {
                "v1": training_summary(V1_RUN / "training_result.json"),
                "v2": training_summary(V2_RUN / "training_result.json"),
            },
        },
        "v2_aggregate_dev": v2_aggregate_qualification(),
        "hypotheses": {
            "h1_bf16_adapter_storage_degraded_optimization": {
                "classification": "supported",
                "evidence_strength": "moderate for degraded update resolution; inconclusive for sole causation of the 40/56-to-24/56 quality regression",
                "mechanism": "V2 forced both adapter parameters and Adam first/second moments to BF16. Late cosine-decay steps increasingly failed to cross the BF16 storage quantum, leaving 46.5863% of elements unchanged from epoch 4 to 5 and 80.9402% unchanged from epoch 5 to 6. V1 used FP32 adapter parameters and moments, with adjacent-epoch unchanged fractions below 0.00021%.",
                "contradictory_or_null_evidence": "V2 optimizer-step training loss decreased in every epoch, so BF16 storage did not cause global divergence or total learning failure.",
                "confounders": [
                    "V1 and V2 used different corpora, train counts, sequence lengths, dropout, batch/accumulation schedules, learning rates, seeds, and epoch counts.",
                    "Adjacent-checkpoint deltas summarize accumulated updates and cannot recover individual rounded-away optimizer steps.",
                ],
            },
            "h2_early_useful_checkpoint_was_missed": {
                "classification": "inconclusive",
                "evidence_strength": "strong for a checkpoint-coverage defect; no admissible evidence remains for epochs 1-3 quality",
                "mechanism": "The six-epoch V2 run saved each epoch but save_total_limit=3 and the frozen candidate list [4,5,6] retained only late checkpoints. Epochs 1-3 are absent, so their quality cannot be tested without a prohibited V2 rerun.",
                "observed_late_trajectory": {"epoch_4": 24, "epoch_5": 21, "epoch_6": 24},
                "confounders": [
                    "Training loss alone is not a proxy for the frozen 56-case quality gate.",
                    "No fresh synthetic checkpoint probe can reconstruct deleted epoch-1/2/3 weights.",
                ],
            },
            "h3_corpus_or_answer_style_catastrophic_tradeoffs": {
                "classification": "inconclusive",
                "evidence_strength": "not tested in this bounded increment",
            },
            "h4_scorer_generation_target_style_mismatch": {
                "classification": "inconclusive",
                "evidence_strength": "not tested in this bounded increment",
            },
        },
        "prospective_v3_constraints_from_this_increment": {
            "acceptance_gate": "unchanged 48/56 dev minimum with all existing category gates",
            "checkpoint_policy": "Predeclare every epoch from 1 through the training horizon as a candidate and retain all of them until aggregate dev qualification completes; do not use save_total_limit to prune unevaluated epochs.",
            "precision": "Keep frozen base weights and BF16 base compute, but require FP32 LoRA master parameters and FP32 Adam first/second moments. Assert these dtypes after PEFT construction and in every checkpoint.",
            "remaining_before_full_v3_contract": "Adjudicate corpus/style and scorer/generation hypotheses before freezing data mixture, learning rate, or epoch horizon.",
        },
        "schema_version": 1,
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sidecar_path(report_path: Path) -> Path:
    return report_path.with_suffix(report_path.suffix + ".sha256")


def write_report(path: Path) -> None:
    payload = canonical_bytes(build_report())
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write(path, payload)
    atomic_write(sidecar_path(path), f"{digest}  {path.name}\n".encode("ascii"))
    print(f"WROTE {relative(path)} sha256={digest}")


def verify_report(path: Path) -> None:
    expected = canonical_bytes(build_report())
    observed = path.read_bytes()
    require(observed == expected, "audit bytes differ from recomputed immutable evidence")
    digest = hashlib.sha256(observed).hexdigest()
    fields = sidecar_path(path).read_text(encoding="ascii").split()
    require(fields == [digest, path.name], "audit SHA-256 sidecar differs")
    report = json.loads(observed)
    excluded_fragments = ("scored_rows", "holdout.jsonl", "dev.jsonl")
    evidence_paths = report["evidence_manifest_sha256"]
    require(
        not any(fragment in evidence_path for evidence_path in evidence_paths for fragment in excluded_fragments),
        "forbidden raw evaluation evidence entered the manifest",
    )
    require(report["hypotheses"]["h1_bf16_adapter_storage_degraded_optimization"]["classification"] == "supported", "H1 classification changed")
    require(report["hypotheses"]["h2_early_useful_checkpoint_was_missed"]["classification"] == "inconclusive", "H2 classification changed")
    print(f"QWEN25_LORA_METADATA_AUDIT_VERIFIED sha256={digest}")


def main() -> None:
    args = parse_args()
    if args.output is not None:
        write_report(args.output.resolve())
    else:
        verify_report(args.verify.resolve())


if __name__ == "__main__":
    main()
