#!/usr/bin/env python3
"""Calibrate the frozen per-layer/head Q/K scales without running quality metrics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ace2_full_model_fixed_point import (
    PROMPT_MANIFEST,
    QUALITY_CONFIG,
    ROOT,
    calibrate,
    hash_records,
    hash_token_sequences,
    positive_scale,
    seed_everything,
    selected_texts,
    sha256_file,
    tokenize_prompts,
    utc_now,
    validate_runtime,
    write_json,
)


CONTRACT = "shared_qk_residual_cross_term_attention_v1"
FULL_MODEL_RUNNER = ROOT / "tools/ace2_full_model_fixed_point.py"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "benchmark/raw/quality/qk-residual-scale32-calibration-c4-validation-0-64-512"
)


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def source_contracts() -> dict[str, dict[str, Any]]:
    paths = {
        "prompt_manifest": PROMPT_MANIFEST,
        "quality_config": QUALITY_CONFIG,
        "calibration_runner": Path(__file__).resolve(),
        "full_model_runner": FULL_MODEL_RUNNER,
    }
    return {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def calibration_scope(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest["datasets"]["c4_calibration"]
    return {
        "dataset": source["repository"],
        "config": source["config"],
        "revision": source["revision"],
        "split": source["split"],
        "record_indices": copy.deepcopy(source["indices"]),
        "token_limit_per_record": source["token_limit"],
        "scale_rule": (
            "per_layer_per_q_or_k_head_maximum_absolute_observed_bf16_"
            "projection_output_divided_by_127"
        ),
    }


def observed_input(
    manifest: dict[str, Any],
    texts: list[str],
    prompts: list[torch.Tensor],
) -> dict[str, Any]:
    source = manifest["datasets"]["c4_calibration"]
    record_sha256, record_count = hash_records(texts)
    token_sha256, sequence_count, token_count = hash_token_sequences(prompts)
    return {
        "c4_calibration": {
            "config": source["config"],
            "record_count": record_count,
            "record_indices": copy.deepcopy(source["indices"]),
            "record_sha256": record_sha256,
            "repository": source["repository"],
            "revision": source["revision"],
            "split": source["split"],
            "tokenized": {
                "sequence_count": sequence_count,
                "token_count": token_count,
                "token_limit_per_record": source["token_limit"],
                "token_sequence_sha256": token_sha256,
            },
        }
    }


def qk_scale_table(
    ranges: dict[str, Any],
    run_contract_path: Path,
    scope: dict[str, Any],
    observations: dict[str, Any],
) -> dict[str, Any]:
    attention: dict[str, Any] = {}
    for layer in range(24):
        prefix = f"model.layers.{layer}.self_attn"
        q_absmax = ranges[f"{prefix}.q_proj"].output_head_absmax
        k_absmax = ranges[f"{prefix}.k_proj"].output_head_absmax
        if q_absmax is None or len(q_absmax) != 14:
            raise RuntimeError(f"layer {layer} query-head calibration is incomplete")
        if k_absmax is None or len(k_absmax) != 2:
            raise RuntimeError(f"layer {layer} key-head calibration is incomplete")
        attention[prefix] = {
            "query_projection_output_absmax": q_absmax,
            "query_projection_output_scales": [positive_scale(value) for value in q_absmax],
            "key_projection_output_absmax": k_absmax,
            "key_projection_output_scales": [positive_scale(value) for value in k_absmax],
        }
    return {
        "schema_version": 3,
        "contract_id": CONTRACT,
        "calibration_scope": scope,
        "input_observations": observations,
        "provenance": {
            "run_contract": {
                "path": run_contract_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(run_contract_path),
            },
            "quality_metrics_executed": False,
        },
        "attention": attention,
    }


def run(output_dir: Path) -> dict[str, Any]:
    total_started = time.perf_counter()
    manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(QUALITY_CONFIG.read_text(encoding="utf-8"))
    versions = validate_runtime(config)
    seed_everything(config)
    output_dir.mkdir(parents=True, exist_ok=False)

    scope = calibration_scope(manifest)
    texts = selected_texts(manifest["datasets"]["c4_calibration"])
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["repository"],
        revision=manifest["model"]["revision"],
    )
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    if tokenizer_revision is not None and tokenizer_revision != manifest["model"]["revision"]:
        raise RuntimeError(
            f"tokenizer resolved to {tokenizer_revision}, expected {manifest['model']['revision']}"
        )
    prompts = tokenize_prompts(tokenizer, texts, scope["token_limit_per_record"])
    observations = observed_input(manifest, texts, prompts)
    write_json(output_dir / "input_observations.json", observations)

    contracts = source_contracts()
    run_contract = {
        "schema_version": 1,
        "status": "frozen_before_measurement",
        "created_at_utc": utc_now(),
        "mode": "calibration_only",
        "purpose": "architecture_scale32_source_no_quality_evaluation",
        "command": [sys.executable, *sys.argv],
        "contract_id": CONTRACT,
        "model": {
            **manifest["model"],
            "tokenizer_requested_revision": manifest["model"]["revision"],
            "tokenizer_resolved_revision": tokenizer_revision,
        },
        "calibration_scope": scope,
        "public_input_slice": {
            "record_limits": {"c4_calibration": 64},
            "token_limit": scope["token_limit_per_record"],
            "observations": observations,
        },
        "activation_quantization": config["activation_quantization"],
        "seeds": config["determinism"],
        "environment": {
            "device": "cpu",
            "executable": sys.executable,
            "packages": versions,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_runtime": torch.__version__,
        },
        "source_contracts": contracts,
        "quality_metrics_executed": False,
    }
    run_contract_path = output_dir / "run_contract.json"
    write_json(run_contract_path, run_contract)

    model_load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"]["repository"],
        revision=manifest["model"]["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != manifest["model"]["revision"]:
        raise RuntimeError(
            f"model resolved to {resolved_revision}, expected {manifest['model']['revision']}"
        )
    model_load_seconds = time.perf_counter() - model_load_started

    calibration_started = time.perf_counter()
    ranges, _operator_ranges = calibrate(model, prompts)
    calibration_seconds = time.perf_counter() - calibration_started
    scales_path = output_dir / "derived_scales.json"
    write_json(
        scales_path,
        qk_scale_table(ranges, run_contract_path, scope, observations),
    )

    result = {
        "schema_version": 1,
        "status": "pass",
        "classification": "calibration_only_no_quality_claim",
        "created_at_utc": utc_now(),
        "mode": "calibration_only",
        "contract_id": CONTRACT,
        "calibration_scope": scope,
        "input_observations": observations,
        "model": {
            **manifest["model"],
            "resolved_revision": resolved_revision,
        },
        "source_contracts": contracts,
        "quality_metrics_executed": False,
        "artifacts": {
            "derived_scales": {
                "path": scales_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(scales_path),
            },
            "input_observations": {
                "path": (output_dir / "input_observations.json").relative_to(ROOT).as_posix(),
                "sha256": sha256_file(output_dir / "input_observations.json"),
            },
            "run_contract": {
                "path": run_contract_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(run_contract_path),
            },
        },
        "runtime_seconds": {
            "calibration": calibration_seconds,
            "model_load": model_load_seconds,
            "total_before_result_write": time.perf_counter() - total_started,
        },
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    result["integrity"]["canonical_sha256"] = canonical_sha256(result)
    write_json(output_dir / "results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result = run(output_dir)
    observation = result["input_observations"]["c4_calibration"]
    print(
        "ACE2_QK_SCALE_CALIBRATION_PASS "
        f"records={observation['record_count']} "
        f"tokens={observation['tokenized']['token_count']} "
        f"quality_metrics=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
