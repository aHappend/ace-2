#!/usr/bin/env python3
"""CPU_SOFTWARE_REFERENCE_ONLY_NOT_RTL_COMPLETION Qwen2.5 CPU chat reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABEL = "CPU_SOFTWARE_REFERENCE_ONLY_NOT_RTL_COMPLETION"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/home/argustest/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
DEFAULT_OUTPUT = ROOT / "build/cpu_reference_chat/latest"
MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
SMOKE_PROMPTS = (
    "Reply with exactly three words describing reliable hardware.",
    "In one short sentence, state why deterministic tests are useful.",
)
CLAIM_BOUNDARY = (
    "CPU software reference only. This is not Stage-1 completion and provides no W4A8 RTL "
    "agreement, accelerator latency, S7/U280 execution, GPU execution, or FPGA evidence."
)


class LabeledLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, key: str, value: Any) -> None:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        line = f"{LABEL} {key}={rendered}"
        self.lines.append(line)
        print(line, flush=True)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def process_command() -> tuple[list[str], str]:
    proc_cmdline = Path("/proc/self/cmdline")
    if proc_cmdline.is_file():
        argv = [
            item.decode("utf-8", errors="replace")
            for item in proc_cmdline.read_bytes().split(b"\0")
            if item
        ]
    else:
        argv = [sys.executable, *sys.argv]
    return argv, shlex.join(argv)


def require_safe_path(path: Path, role: str) -> None:
    forbidden = re.compile(r"v7[defg]", re.IGNORECASE)
    for part in path.resolve().parts:
        lowered = part.lower()
        if lowered == "ground_truth" or "authorization" in lowered or forbidden.search(lowered):
            raise ValueError(f"{role} path is inside a prohibited boundary")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cache_length(cache: Any) -> int | None:
    if cache is None:
        return None
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    try:
        return int(cache[0][0].shape[-2])
    except (IndexError, TypeError, AttributeError):
        return None


def model_file_evidence(model_dir: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for name in MODEL_FILES:
        path = model_dir / name
        resolved = path.resolve()
        evidence[name] = {
            "path": str(path),
            "resolved_path": str(resolved),
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    return evidence


def source_evidence() -> dict[str, dict[str, Any]]:
    paths = (
        ROOT / "tools/qwen25_cpu_reference_chat.py",
        ROOT / "verification/verify_qwen25_cpu_reference_chat.py",
    )
    return {
        str(path.relative_to(ROOT)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    }


def missing_prerequisite(model_dir: Path) -> str | None:
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if not missing:
        return None
    return (
        "one complete local Qwen2.5-0.5B-Instruct snapshot at --model containing "
        + ", ".join(MODEL_FILES)
        + f"; missing at {model_dir}: {', '.join(missing)}"
    )


def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt: str,
    prompt_origin: str,
    system_prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cpu")
    attention_mask = torch.ones_like(input_ids, device="cpu")
    generated_ids: list[int] = []
    cache_lengths: list[int | None] = []
    cache_classes: list[str] = []
    reuse_input_token_counts: list[int] = []
    eos_token_ids = tokenizer.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_ids = {eos_token_ids}
    else:
        eos_ids = set(eos_token_ids or [])

    started = time.perf_counter()
    with torch.inference_mode():
        prefill_started = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        prefill_seconds = time.perf_counter() - prefill_started
        past_key_values = outputs.past_key_values
        cache_lengths.append(cache_length(past_key_values))
        cache_classes.append(type(past_key_values).__name__)
        forward_calls = 1

        while len(generated_ids) < max_new_tokens:
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            if token_id in eos_ids and len(generated_ids) >= min_new_tokens:
                break
            if len(generated_ids) >= max_new_tokens:
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device="cpu")],
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            reuse_input_token_counts.append(int(next_token.numel()))
            cache_lengths.append(cache_length(past_key_values))
            cache_classes.append(type(past_key_values).__name__)
            forward_calls += 1

    generation_seconds = time.perf_counter() - started
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    observed_lengths = [value for value in cache_lengths if value is not None]
    cache_growth = all(
        current > previous for previous, current in zip(observed_lengths, observed_lengths[1:])
    )
    kv_cache_path_exercised = (
        len(generated_ids) > 1
        and len(reuse_input_token_counts) >= 1
        and all(count == 1 for count in reuse_input_token_counts)
        and len(observed_lengths) >= 2
        and cache_growth
    )

    return {
        "artifact_label": LABEL,
        "prompt": prompt,
        "prompt_origin": prompt_origin,
        "system_prompt": system_prompt,
        "prompt_token_ids": [int(value) for value in input_ids[0].tolist()],
        "prompt_token_count": int(input_ids.shape[-1]),
        "generated_token_ids": generated_ids,
        "generated_tokens": tokenizer.convert_ids_to_tokens(generated_ids),
        "generated_token_count": len(generated_ids),
        "decoded_text": decoded,
        "decoded_nonempty": bool(decoded.strip()),
        "decoded_has_alphanumeric": any(character.isalnum() for character in decoded),
        "stopped_on_eos": bool(generated_ids and generated_ids[-1] in eos_ids),
        "generation": {
            "algorithm": "manual_greedy_argmax",
            "sampling": False,
            "forward_calls": forward_calls,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
        },
        "kv_cache": {
            "use_cache_requested": True,
            "cache_classes": cache_classes,
            "cache_sequence_lengths": cache_lengths,
            "cache_reuse_calls": len(reuse_input_token_counts),
            "cache_reuse_input_token_counts": reuse_input_token_counts,
            "path_exercised": kv_cache_path_exercised,
        },
        "timing": {
            "prefill_wall_seconds": prefill_seconds,
            "generation_wall_seconds": generation_seconds,
            "seconds_per_generated_token": generation_seconds / max(1, len(generated_ids)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--system-prompt", default="You are a concise, helpful assistant.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    log = LabeledLog()
    started_at = utc_now()
    total_started = time.perf_counter()
    args = parse_args()
    model_dir = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        require_safe_path(model_dir, "model")
        require_safe_path(output_dir, "output")
        if args.max_new_tokens < 2:
            raise ValueError("--max-new-tokens must be at least 2")
        if not 2 <= args.min_new_tokens <= args.max_new_tokens:
            raise ValueError("--min-new-tokens must be between 2 and --max-new-tokens")
        if args.threads < 1:
            raise ValueError("--threads must be positive")

        prompts: list[tuple[str, str]] = []
        if args.smoke:
            prompts.extend((prompt, "deterministic_smoke") for prompt in SMOKE_PROMPTS)
        prompts.extend((prompt, "operator") for prompt in args.prompt)
        if not prompts:
            raise ValueError("provide --prompt TEXT, or use --smoke")

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        stdout_path = output_dir / "stdout.log"
        argv, exact_command = process_command()
        log.emit("artifact_label", LABEL)
        log.emit("claim_boundary", CLAIM_BOUNDARY)
        log.emit("exact_command", exact_command)
        log.emit("device", "cpu")
        log.emit("model_path", str(model_dir))

        prerequisite = missing_prerequisite(model_dir)
        if prerequisite is not None:
            log.emit("status", "MISSING_LOCAL_WEIGHT_PREREQUISITE")
            log.emit("missing_local_weight_prerequisite", prerequisite)
            stdout_path.write_text(log.text(), encoding="utf-8")
            write_json(
                output_dir / "missing_prerequisite.json",
                {
                    "artifact_label": LABEL,
                    "status": "MISSING_LOCAL_WEIGHT_PREREQUISITE",
                    "missing_local_weight_prerequisite": prerequisite,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "exact_command": exact_command,
                    "stdout": log.text(),
                },
            )
            return 3

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        transformers.utils.logging.disable_progress_bar()
        transformers.utils.logging.set_verbosity_error()
        torch.manual_seed(0)
        torch.set_num_threads(args.threads)

        log.emit("model_hashing", "started")
        model_files = model_file_evidence(model_dir)
        log.emit("model_safetensors_sha256", model_files["model.safetensors"]["sha256"])
        log.emit("model_loading", "started")
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            attn_implementation="eager",
            device_map=None,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.to("cpu")
        model.eval()
        model.config.use_cache = True
        model_load_seconds = time.perf_counter() - load_started
        parameter_device = str(next(model.parameters()).device)
        if parameter_device != "cpu":
            raise RuntimeError(f"model loaded on prohibited device {parameter_device}")
        log.emit("model_loading_wall_seconds", model_load_seconds)

        runs: list[dict[str, Any]] = []
        for index, (prompt, origin) in enumerate(prompts):
            run = generate_one(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                prompt=prompt,
                prompt_origin=origin,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
            )
            run["index"] = index
            runs.append(run)
            log.emit(f"run_{index}_prompt", prompt)
            log.emit(f"run_{index}_prompt_token_ids", run["prompt_token_ids"])
            log.emit(f"run_{index}_generated_token_ids", run["generated_token_ids"])
            log.emit(f"run_{index}_generated_token_count", run["generated_token_count"])
            log.emit(f"run_{index}_kv_cache_path_exercised", run["kv_cache"]["path_exercised"])
            log.emit(f"run_{index}_decoded_text", run["decoded_text"])
            log.emit(f"run_{index}_generation_wall_seconds", run["timing"]["generation_wall_seconds"])

        smoke_pass = all(
            run["generated_token_count"] > 1
            and run["decoded_nonempty"]
            and run["decoded_has_alphanumeric"]
            and run["kv_cache"]["path_exercised"]
            for run in runs
        )
        total_seconds = time.perf_counter() - total_started
        status = "PASS" if smoke_pass else "FAIL"
        log.emit("smoke_test_status", status)
        log.emit("manifest_path", str(manifest_path))
        log.emit("stdout_log_path", str(stdout_path))

        config = model.config
        revision = model_dir.name if model_dir.parent.name == "snapshots" else None
        manifest = {
            "artifact_label": LABEL,
            "schema_version": 1,
            "status": status,
            "claim_boundary": CLAIM_BOUNDARY,
            "claims": {
                "stage1_completion": False,
                "w4a8_rtl_agreement": False,
                "accelerator_latency": False,
                "s7_u280_execution": False,
            },
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "exact_command": exact_command,
            "process_argv": argv,
            "working_directory": str(Path.cwd()),
            "model": {
                "declared_identity": "Qwen/Qwen2.5-0.5B-Instruct",
                "local_snapshot_path": str(model_dir),
                "immutable_revision": revision,
                "architecture": list(getattr(config, "architectures", []) or []),
                "model_type": getattr(config, "model_type", None),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "parameter_dtype": str(next(model.parameters()).dtype),
                "files": model_files,
            },
            "runtime": {
                "python": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_model": cpu_model_name(),
                "logical_cpu_count": os.cpu_count(),
                "configured_torch_threads": args.threads,
                "torch": torch.__version__,
                "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "transformers": transformers.__version__,
                "tokenizers": package_version("tokenizers"),
                "safetensors": package_version("safetensors"),
            },
            "execution": {
                "device": "cpu",
                "parameter_device": parameter_device,
                "attention_implementation": "eager",
                "generation_algorithm": "manual_greedy_argmax",
                "use_cache": True,
                "network_access_performed": False,
                "gpu_execution_performed": False,
                "fpga_execution_performed": False,
                "seed": 0,
            },
            "smoke_test": {
                "status": status,
                "criteria": {
                    "generated_token_count_greater_than_one": True,
                    "decoded_text_nonempty": True,
                    "decoded_text_has_alphanumeric": True,
                    "kv_cache_reused_with_single_token_decode_steps": True,
                },
                "run_count": len(runs),
                "deterministic_prompt_count": sum(
                    run["prompt_origin"] == "deterministic_smoke" for run in runs
                ),
                "operator_prompt_count": sum(run["prompt_origin"] == "operator" for run in runs),
            },
            "timing": {
                "model_load_wall_seconds": model_load_seconds,
                "total_wall_seconds": total_seconds,
            },
            "memory": {"peak_rss_bytes": peak_rss_bytes()},
            "runs": runs,
            "source_files": source_evidence(),
            "artifacts": {
                "manifest": str(manifest_path),
                "stdout_log": str(stdout_path),
            },
            "stdout": log.text(),
        }
        stdout_path.write_text(log.text(), encoding="utf-8")
        write_json(manifest_path, manifest)
        return 0 if smoke_pass else 1
    except Exception as exc:
        log.emit("status", "ERROR")
        log.emit("error_type", type(exc).__name__)
        log.emit("error", str(exc))
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "stdout.log").write_text(log.text(), encoding="utf-8")
            write_json(
                output_dir / "failure.json",
                {
                    "artifact_label": LABEL,
                    "status": "ERROR",
                    "claim_boundary": CLAIM_BOUNDARY,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "stdout": log.text(),
                },
            )
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
