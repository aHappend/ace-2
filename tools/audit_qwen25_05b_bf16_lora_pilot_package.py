#!/usr/bin/env python3
"""Non-consuming audit for the build-ready ACE-2 A100 LoRA pilot package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v1"
sys.path.insert(0, str(PACKAGE))

import common  # noqa: E402


def check(value: bool, name: str, checks: dict[str, bool]) -> None:
    checks[name] = bool(value)
    if not value:
        raise RuntimeError(name)


def audit() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    contract = common.load_contract()
    pipeline = common.load_json(ROOT / "research/PIPELINE_STATE.json")
    check(pipeline.get("current_stage") == "specification", "pipeline_stage_specification", checks)
    check(not common.PREFLIGHT_DIR.exists(), "a100_preflight_unconsumed", checks)
    check(not common.ATTEMPT_DIR.exists(), "pilot_attempt_unconsumed", checks)
    check(common.package_tree_sha256() != "", "package_tree_hash_available", checks)

    lock = common.load_json(common.ENVIRONMENT_LOCK)
    required = contract["execution_preflight"]["required_software"]
    mapping = {
        "accelerate": "accelerate",
        "datasets": "datasets",
        "huggingface_hub": "huggingface-hub",
        "peft": "peft",
        "safetensors": "safetensors",
        "tokenizers": "tokenizers",
        "torch": "torch",
        "transformers": "transformers",
    }
    check(lock["python"] == required["python"], "lock_python_matches_contract", checks)
    for contract_name, package_name in mapping.items():
        check(
            lock["packages"][package_name] == required[contract_name],
            f"lock_{contract_name}_matches_contract",
            checks,
        )
    check(lock["packages"]["lm-eval"] == "0.4.9.2", "lock_lm_eval_frozen", checks)
    check(lock["network_access_allowed"] is False, "lock_network_forbidden", checks)
    versions = common.environment_versions()["observed"]
    check(versions["python"] == lock["python"], "host_python_matches_lock", checks)
    check(versions["cuda_runtime"] == lock["cuda_runtime"], "host_cuda_runtime_matches_lock", checks)
    for name, expected in lock["packages"].items():
        check(versions["packages"][name] == expected, f"host_package_{name}_matches_lock", checks)

    requirements = {
        line.strip()
        for line in (PACKAGE / "requirements.lock.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    expected_requirements = {f"{name}=={version}" for name, version in lock["packages"].items()}
    check(requirements == expected_requirements, "requirements_lock_matches_environment_lock", checks)
    prerequisite = common.load_json(PACKAGE / "A100_ACCESS_PREREQUISITE.json")
    check(
        prerequisite["status"] == "WAITING_FOR_EXECUTION_ENDPOINT",
        "a100_access_status_waiting",
        checks,
    )
    check(
        prerequisite["execution"]["preflight_attempts_remaining"] == 1,
        "one_preflight_remaining",
        checks,
    )
    check(
        prerequisite["execution"]["training_attempts_remaining"] == 1,
        "one_training_attempt_remaining",
        checks,
    )
    check(
        prerequisite["execution"]["command"]
        == "bash pilot/qwen25_05b_bf16_lora_product_v1/run_once.sh",
        "access_request_command_exact",
        checks,
    )
    check(
        prerequisite["hardware"]["minimum_memory_gib"]
        == contract["execution_preflight"]["required_hardware"]["minimum_memory_gib"],
        "access_request_memory_matches_contract",
        checks,
    )

    for name in common.PACKAGE_FILES:
        check((PACKAGE / name).is_file(), f"package_file_{name}", checks)
    for path in sorted(PACKAGE.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        checks[f"python_compile_{path.name}"] = True
    shell = subprocess.run(
        ["bash", "-n", str(PACKAGE / "run_once.sh")],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    check(shell.returncode == 0, "run_once_shell_syntax", checks)
    for script in (
        "preflight.py",
        "train.py",
        "merge.py",
        "retention.py",
        "evaluate_holdout.py",
        "finalize_manual_rubric.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(PACKAGE / script), "--help"],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        check(completed.returncode == 0, f"help_is_non_consuming_{script}", checks)
    check(not common.PREFLIGHT_DIR.exists(), "help_did_not_consume_preflight", checks)
    check(not common.ATTEMPT_DIR.exists(), "help_did_not_consume_attempt", checks)

    preflight_source = (PACKAGE / "preflight.py").read_text(encoding="utf-8")
    train_source = (PACKAGE / "train.py").read_text(encoding="utf-8")
    merge_source = (PACKAGE / "merge.py").read_text(encoding="utf-8")
    retention_source = (PACKAGE / "retention.py").read_text(encoding="utf-8")
    holdout_source = (PACKAGE / "evaluate_holdout.py").read_text(encoding="utf-8")
    run_once = (PACKAGE / "run_once.sh").read_text(encoding="utf-8")
    check(
        preflight_source.index("CONSUMPTION_MARKER.json")
        < preflight_source.index("torch.cuda.is_available"),
        "preflight_marker_precedes_cuda_probe",
        checks,
    )
    check(
        "memory_bytes >= 40 * (1024**3)" in preflight_source,
        "preflight_enforces_40_gib",
        checks,
    )
    check(
        train_source.index("ATTEMPT_CONSUMPTION_MARKER.json")
        < train_source.index("load_causal_model(snapshot)"),
        "attempt_marker_precedes_model_training",
        checks,
    )
    check(
        holdout_source.index("CONSUMPTION_MARKER.json")
        < holdout_source.index("include_holdout=True"),
        "holdout_marker_precedes_holdout_hash_or_read",
        checks,
    )
    check("holdout.jsonl" not in train_source, "training_does_not_name_holdout_file", checks)
    check("holdout.jsonl" not in merge_source, "dev_selection_does_not_name_holdout_file", checks)
    check("holdout.jsonl" not in retention_source, "retention_does_not_name_holdout_file", checks)
    check("include_holdout=True" in holdout_source, "holdout_evaluator_is_only_holdout_reader", checks)
    check(
        "merged_dtypes == {torch.bfloat16}" in merge_source,
        "merge_requires_bfloat16_parameters",
        checks,
    )
    order = [
        run_once.index('"$PACKAGE/preflight.py"'),
        run_once.index('"$PACKAGE/train.py"'),
        run_once.index('"$PACKAGE/merge.py"'),
        run_once.index('"$PACKAGE/retention.py"'),
        run_once.index('"$PACKAGE/evaluate_holdout.py"'),
    ]
    check(order == sorted(order), "run_once_stage_order", checks)
    check("pip install" not in run_once, "run_once_has_no_network_install", checks)

    snapshot = common.source_snapshot(contract)
    common.verify_dataset(contract, include_holdout=False)
    tokenizer = common.load_tokenizer(snapshot, contract)
    train_rows = common.read_jsonl(common.DATASET_DIR / "train.jsonl")
    dev_rows = common.read_jsonl(common.DATASET_DIR / "dev.jsonl")
    train_encoded = common.encode_training_rows(
        tokenizer, train_rows, contract["pilot"]["loss"]["sequence_length"]
    )
    dev_encoded = common.encode_dev_rows(
        tokenizer, dev_rows, contract["pilot"]["loss"]["sequence_length"]
    )
    check(len(train_encoded) == contract["dataset"]["train_count"], "train_count_encoded", checks)
    check(len(dev_encoded) == contract["dataset"]["dev_count"], "dev_count_encoded", checks)
    check(
        max(len(item["input_ids"]) for item in train_encoded.values)
        == contract["dataset"]["maximum_tokenized_lengths"]["train_full_conversation"],
        "train_max_tokenized_length",
        checks,
    )
    dev_prompt_max = max(
        len(
            tokenizer.apply_chat_template(
                row["input_messages"], tokenize=True, add_generation_prompt=True
            )
        )
        for row in dev_rows
    )
    check(
        dev_prompt_max == contract["dataset"]["maximum_tokenized_lengths"]["dev_input"],
        "dev_max_input_length",
        checks,
    )
    labels_valid = True
    for item in train_encoded.values:
        first_target = next(
            (index for index, label in enumerate(item["labels"]) if label != -100), None
        )
        labels_valid = labels_valid and first_target is not None and all(
            label == -100 for label in item["labels"][:first_target]
        )
    check(labels_valid, "assistant_only_labels_present", checks)
    common.self_test()
    checks["hard_check_self_test"] = True

    architecture = contract["architecture"]
    hidden = architecture["hidden_size"]
    intermediate = architecture["intermediate_size"]
    head_dim = hidden // architecture["num_attention_heads"]
    kv = head_dim * architecture["num_key_value_heads"]
    rank = contract["pilot"]["lora"]["rank"]
    per_layer = rank * (
        (hidden + hidden)
        + (hidden + kv)
        + (hidden + kv)
        + (hidden + hidden)
        + (hidden + intermediate)
        + (hidden + intermediate)
        + (intermediate + hidden)
    )
    derived_trainable = per_layer * architecture["num_hidden_layers"]
    check(
        derived_trainable == contract["pilot"]["lora"]["trainable_parameter_count"],
        "derived_trainable_parameter_count",
        checks,
    )
    import gc

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        snapshot,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    lora = contract["pilot"]["lora"]
    peft_model = get_peft_model(
        base,
        LoraConfig(
            task_type=lora["task_type"],
            r=lora["rank"],
            lora_alpha=lora["scaling_alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias=lora["bias"],
            modules_to_save=None,
            inference_mode=False,
        ),
    )
    trainable = [
        (name, parameter.numel(), str(parameter.dtype))
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    check(
        sum(item[1] for item in trainable) == lora["trainable_parameter_count"],
        "actual_peft_trainable_parameter_count",
        checks,
    )
    check(all("lora_" in item[0] for item in trainable), "actual_peft_only_lora_trainable", checks)
    peft_structure = {
        "trainable_parameter_count": sum(item[1] for item in trainable),
        "trainable_parameter_dtypes": sorted({item[2] for item in trainable}),
        "trainable_tensor_count": len(trainable),
    }
    del peft_model
    del base
    gc.collect()
    check(
        contract["prohibitions"]["w4a8_quantization_before_bf16_holdout_and_retention_go"] is True,
        "quantization_remains_forbidden",
        checks,
    )
    check(
        contract["prohibitions"]["rtl_or_u280_reopen_before_fresh_reviewer_accepts_new_quantized_reference"] is True,
        "rtl_u280_remain_closed",
        checks,
    )

    return {
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "Package construction and non-consuming static audit only. No A100 connectivity probe, training, holdout read, model evaluation, quantization, RTL, synthesis, FPGA, U280, or exactly-once evidence consumption occurred.",
        "contract_sha256": common.sha256_file(common.CONTRACT_PATH),
        "package_tree_sha256": common.package_tree_sha256(),
        "peft_structure": peft_structure,
        "schema_version": 1,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit()
    result["observed_at_utc"] = common.utc_now()
    output = ROOT / args.output
    common.atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "checks": result["check_count"],
                "output": str(output.relative_to(ROOT)),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
