#!/usr/bin/env python3
"""Run one immutable, non-quality CPU smoke of the frozen BF16 LoRA training path."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v1"
sys.path.insert(0, str(PACKAGE))

from common import (  # noqa: E402
    ATTEMPT_DIR,
    CONTRACT_PATH,
    DATASET_DIR,
    PREFLIGHT_DIR,
    AssistantOnlyCollator,
    atomic_write_json,
    configure_determinism,
    encode_training_rows,
    exclusive_write_json,
    load_contract,
    load_tokenizer,
    package_tree_sha256,
    read_jsonl,
    require,
    sha256_file,
    source_snapshot,
    utc_now,
    verify_dataset,
    verify_environment,
)


SMOKE_DIR = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v1/cpu-smoke-0001"
)


class Tee:
    """Mirror complete process text output into an immutable durable log."""

    def __init__(self, primary: TextIO, durable: TextIO):
        self.primary = primary
        self.durable = durable
        self.encoding = getattr(primary, "encoding", "utf-8")
        self.errors = getattr(primary, "errors", "strict")

    def write(self, value: str) -> int:
        self.primary.write(value)
        self.durable.write(value)
        self.durable.flush()
        return len(value)

    def flush(self) -> None:
        self.primary.flush()
        self.durable.flush()

    def isatty(self) -> bool:
        return False


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def run_smoke(optimizer_steps: int, marker: dict[str, Any]) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments

    class StopAfterOptimizerSteps(TrainerCallback):
        """Preserve the full Trainer schedule while stopping after bounded steps."""

        def __init__(self, requested_steps: int):
            self.requested_steps = requested_steps
            self.planned_optimizer_steps: int | None = None

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            self.planned_optimizer_steps = int(state.max_steps)
            return control

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if state.global_step >= self.requested_steps:
                control.should_training_stop = True
            return control

    contract = load_contract()
    environment = verify_environment()
    snapshot = source_snapshot(contract)
    manifest = verify_dataset(contract, include_holdout=False)
    configure_determinism(contract)

    tokenizer = load_tokenizer(snapshot, contract)
    rows = read_jsonl(DATASET_DIR / "train.jsonl")
    require(len(rows) == contract["dataset"]["train_count"], "training row count differs")
    sequence_length = int(contract["pilot"]["loss"]["sequence_length"])
    train_dataset = encode_training_rows(tokenizer, rows, sequence_length)
    collator = AssistantOnlyCollator(tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        attn_implementation="eager",
        device_map=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.to("cpu")
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    lora = contract["pilot"]["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=lora["task_type"],
            r=lora["rank"],
            lora_alpha=lora["scaling_alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias=lora["bias"],
            modules_to_save=lora["modules_to_save"] or None,
            inference_mode=False,
        ),
    )
    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    require(trainable, "no trainable LoRA parameters")
    require(all("lora_" in name for name in trainable), "non-LoRA parameter is trainable")
    require(
        sum(trainable.values()) == lora["trainable_parameter_count"],
        "trainable parameter count differs",
    )
    model.config.use_cache = False

    schedule = contract["pilot"]["schedule"]
    optimizer = contract["pilot"]["optimizer"]
    deterministic = contract["pilot"]["determinism"]
    callback = StopAfterOptimizerSteps(optimizer_steps)
    arguments = TrainingArguments(
        output_dir=str(SMOKE_DIR / "scratch"),
        overwrite_output_dir=False,
        do_train=True,
        eval_strategy="no",
        save_strategy="no",
        per_device_train_batch_size=schedule["micro_batch_size"],
        gradient_accumulation_steps=schedule["gradient_accumulation_steps"],
        num_train_epochs=schedule["epochs"],
        learning_rate=optimizer["learning_rate"],
        weight_decay=optimizer["weight_decay"],
        adam_beta1=optimizer["adam_beta1"],
        adam_beta2=optimizer["adam_beta2"],
        adam_epsilon=optimizer["adam_epsilon"],
        max_grad_norm=optimizer["gradient_clipping_norm"],
        lr_scheduler_type=schedule["learning_rate_scheduler"],
        warmup_ratio=schedule["warmup_ratio"],
        optim=optimizer["optimizer"],
        bf16=True,
        fp16=False,
        tf32=False,
        use_cpu=True,
        seed=deterministic["seed"],
        data_seed=contract["pilot"]["data_order"]["dataloader_seed"],
        full_determinism=True,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=[],
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        log_level="info",
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[callback],
    )

    training_started = time.perf_counter()
    training_result = trainer.train(resume_from_checkpoint=None)
    training_wall_seconds = time.perf_counter() - training_started
    observed_steps = int(trainer.state.global_step)
    require(observed_steps == optimizer_steps, "smoke optimizer-step count differs")
    require(callback.planned_optimizer_steps is not None, "full schedule step count absent")
    planned_steps = callback.planned_optimizer_steps
    projected_training_seconds = training_wall_seconds * planned_steps / observed_steps
    log_history = [dict(item) for item in trainer.state.log_history]
    losses = [float(item["loss"]) for item in log_history if "loss" in item]
    require(losses, "Trainer emitted no optimizer-step loss")

    floating_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    )
    trainable_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad}
    )
    require("torch.bfloat16" in floating_dtypes, "BF16 model parameters were not exercised")
    require(trainable_dtypes == ["torch.float32"], "LoRA trainable dtype differs")

    return {
        "accepted_training_semantics": {
            "assistant_tokens_only": contract["pilot"]["loss"]["assistant_tokens_only"],
            "bf16_autocast_enabled": bool(getattr(trainer, "use_cpu_amp", False)),
            "data_order_seed": contract["pilot"]["data_order"]["dataloader_seed"],
            "determinism_seed": deterministic["seed"],
            "full_schedule_optimizer_steps": planned_steps,
            "gradient_accumulation_steps": schedule["gradient_accumulation_steps"],
            "micro_batch_size": schedule["micro_batch_size"],
            "optimizer": optimizer,
            "sequence_length_limit": sequence_length,
            "training_dtype": contract["pilot"]["training_dtype"],
        },
        "claim_boundary": (
            "Deterministic CPU training-code smoke only. No A100 preflight, pilot attempt, "
            "checkpoint, dev selection, retention, holdout access, quality result, model-success "
            "claim, quantization, or RTL evidence was produced."
        ),
        "contract_sha256": marker["contract_sha256"],
        "dataset": {
            "manifest_sha256": contract["dataset"]["manifest_sha256"],
            "train_file_sha256": manifest["files"]["train.jsonl"]["sha256"],
            "train_rows": len(rows),
        },
        "dtype_support": {
            "bf16_forward_backward_optimizer_step_supported": True,
            "cpu_amp_dtype": str(getattr(trainer, "amp_dtype", None)),
            "floating_parameter_dtypes": floating_dtypes,
            "trainable_parameter_dtypes": trainable_dtypes,
        },
        "environment": environment,
        "finished_at_utc": utc_now(),
        "host": {
            "cpu_model": cpu_model_name(),
            "logical_cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        },
        "loss": {
            "optimizer_step_losses": losses,
            "trainer_train_loss": float(training_result.metrics["train_loss"]),
        },
        "memory": {
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "network_access_performed": False,
        "optimizer_steps": {
            "observed": observed_steps,
            "requested": optimizer_steps,
        },
        "package_tree_sha256": marker["package_tree_sha256"],
        "projection": {
            "full_automated_pilot_seconds_lower_bound": projected_training_seconds,
            "full_training_minutes": projected_training_seconds / 60.0,
            "full_training_seconds": projected_training_seconds,
            "method": (
                "Linear projection from measured optimizer-step wall time across the Trainer-computed "
                "full schedule; automated dev/retention/holdout time is omitted, so the full-pilot "
                "number is a lower bound."
            ),
            "within_90_minutes": projected_training_seconds <= 90 * 60,
        },
        "quality_or_model_success_claim": False,
        "schema_version": 1,
        "source_snapshot": str(snapshot),
        "status": "PASS",
        "timing": {
            "training_wall_seconds": training_wall_seconds,
            "trainer_metrics": {
                name: float(value)
                for name, value in training_result.metrics.items()
                if isinstance(value, (int, float))
            },
        },
        "trainable_parameter_count": sum(trainable.values()),
        "trainer_log_history": log_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-steps", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()

    contract = load_contract()
    require(not PREFLIGHT_DIR.exists(), "A100 preflight namespace must remain unconsumed")
    require(not ATTEMPT_DIR.exists(), "pilot attempt namespace must remain unconsumed")
    require(not SMOKE_DIR.exists(), "CPU smoke namespace was already consumed")
    SMOKE_DIR.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_DIR.mkdir(exist_ok=False)
    marker = {
        "claim_boundary": "single non-quality CPU training-code smoke",
        "consumed_at_utc": utc_now(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "optimizer_steps": args.optimizer_steps,
        "package_tree_sha256": package_tree_sha256(),
        "schema_version": 1,
    }
    exclusive_write_json(SMOKE_DIR / "CONSUMPTION_MARKER.json", marker)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    exit_code = 1
    with (SMOKE_DIR / "console.log").open("x", encoding="utf-8", newline="\n") as console:
        sys.stdout = Tee(original_stdout, console)
        sys.stderr = Tee(original_stderr, console)
        started = time.perf_counter()
        try:
            result = run_smoke(args.optimizer_steps, marker)
            result["timing"]["end_to_end_wall_seconds"] = time.perf_counter() - started
            exit_code = 0
            print("ACE2_LORA_CPU_SMOKE_PASS")
        except Exception as exc:
            traceback.print_exc()
            result = {
                "claim_boundary": (
                    "Failed CPU training-code smoke only. No A100 preflight, pilot attempt, dev, "
                    "retention, holdout, quality, quantization, or RTL claim."
                ),
                "contract_sha256": marker["contract_sha256"],
                "elapsed_seconds": time.perf_counter() - started,
                "error": str(exc),
                "finished_at_utc": utc_now(),
                "network_access_performed": False,
                "package_tree_sha256": marker["package_tree_sha256"],
                "peak_rss_bytes": peak_rss_bytes(),
                "schema_version": 1,
                "status": "FAIL",
                "traceback": traceback.format_exc(),
            }
            print("ACE2_LORA_CPU_SMOKE_FAIL")
        atomic_write_json(SMOKE_DIR / "RESULT.json", result)
        exclusive_write_json(
            SMOKE_DIR / "EXIT_STATUS.json",
            {"exit_code": exit_code, "finished_at_utc": utc_now(), "schema_version": 1},
        )
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
