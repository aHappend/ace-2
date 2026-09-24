#!/usr/bin/env python3
"""Freeze baseline V scales from the calibration used by the focused baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    load_contracts,
    positive_scale,
    seed_everything,
    selected_texts,
    sha256_file,
    tokenize_prompts,
    utc_now,
    validate_runtime,
)


CONTRACT = "shared_v_residual_value_correction_attention_v1"
DEFAULT_OUTPUT = ROOT / "reference/generated/v_residual_scale32_calibration_source.json"
FOCUSED_CALIBRATION_RECORD_LIMIT = 1
FOCUSED_CALIBRATION_TOKEN_LIMIT = 32


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build() -> dict[str, Any]:
    manifest, config, _ = load_contracts(require_rtl_binding=False)
    versions = validate_runtime(config)
    seed_everything(config)
    model_spec = manifest["model"]
    calibration_spec = manifest["datasets"]["c4_calibration"]
    texts = selected_texts(
        calibration_spec,
        limit=FOCUSED_CALIBRATION_RECORD_LIMIT,
    )
    record_sha256, record_count = hash_records(texts)
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
    )
    prompt = tokenize_prompts(
        tokenizer,
        texts,
        calibration_spec["token_limit"],
    )[0][:, :FOCUSED_CALIBRATION_TOKEN_LIMIT]
    token_sha256, sequence_count, token_count = hash_token_sequences([prompt])
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != model_spec["revision"]:
        raise RuntimeError(
            f"model resolved to {resolved_revision}, expected {model_spec['revision']}"
        )
    ranges, _ = calibrate(model, [prompt])
    records: list[dict[str, Any]] = []
    for layer in range(24):
        module = f"model.layers.{layer}.self_attn.v_proj"
        observed_absmax = ranges[module].output_absmax
        baseline_scale = positive_scale(observed_absmax)
        records.append(
            {
                "layer": layer,
                "module": module,
                "observed_output_absmax": observed_absmax,
                "baseline_v_output_scale": baseline_scale,
                "baseline_scale_rule": "maximum_absolute_observed_bf16_value_divided_by_127",
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "classification": "calibration_only_no_quality_metric",
        "generated_at_utc": utc_now(),
        "quality_metrics_executed": False,
        "model": {
            **model_spec,
            "resolved_revision": resolved_revision,
        },
        "calibration_scope": {
            "dataset": calibration_spec["repository"],
            "config": calibration_spec["config"],
            "revision": calibration_spec["revision"],
            "split": calibration_spec["split"],
            "record_indices": {
                "start": calibration_spec["indices"]["start"],
                "stop": calibration_spec["indices"]["start"] + record_count,
            },
            "token_limit": FOCUSED_CALIBRATION_TOKEN_LIMIT,
            "purpose": "freeze_the_same_baseline_v_scale_provenance_used_by_the_128_token_focused_discriminator",
        },
        "input_observations": {
            "record_count": record_count,
            "record_sha256": record_sha256,
            "tokenized": {
                "sequence_count": sequence_count,
                "token_count": token_count,
                "token_sequence_sha256": token_sha256,
            },
        },
        "source_contracts": {
            "prompt_manifest": artifact(PROMPT_MANIFEST),
            "quality_config": artifact(QUALITY_CONFIG),
            "full_model_runner": artifact(ROOT / "tools/ace2_full_model_fixed_point.py"),
            "calibration_runner": artifact(Path(__file__).resolve()),
        },
        "records": records,
        "regeneration_command": (
            ".venv/bin/python tools/calibrate_v_residual_scale32_source.py "
            "--output reference/generated/v_residual_scale32_calibration_source.json"
        ),
        "runtime": {
            "device": "cpu",
            "packages": versions,
            "torch": torch.__version__,
        },
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    payload["integrity"]["canonical_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "ACE2_V_RESIDUAL_CALIBRATION_SOURCE_PASS "
        f"records={len(payload['records'])} hash={payload['integrity']['canonical_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
