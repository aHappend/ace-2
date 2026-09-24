#!/usr/bin/env python3
"""Measure first-layer RMSNorm output-format error without changing the contract."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ace2_full_model_fixed_point import (
    PROMPT_MANIFEST,
    QUALITY_CONFIG,
    RMS_GAIN_FRAC,
    RMS_HIDDEN_SIZE,
    RMS_INV_FRAC,
    ROOT,
    calibrate,
    fixed_rmsnorm_raw,
    hash_records,
    hash_token_sequences,
    load_contracts,
    positive_scale,
    quantize_int8,
    seed_everything,
    selected_texts,
    sha256_file,
    sha256_tensor,
    tokenize_prompts,
    utc_now,
    validate_runtime,
)
from localize_quality_divergence import compare_tensor, write_sha256s


def capture_rmsnorm_io(
    model: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    captured: dict[str, Tensor] = {}

    def capture(
        _module: nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        if captured:
            raise RuntimeError("first RMSNorm executed more than once")
        captured["input"] = inputs[0].detach().to(device="cpu", dtype=torch.float64)
        captured["output"] = output.detach().to(device="cpu", dtype=torch.float64)

    hook = model.model.layers[0].input_layernorm.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        hook.remove()
    if set(captured) != {"input", "output"}:
        raise RuntimeError("first RMSNorm input/output was not captured")
    return captured["input"], captured["output"]


def fixed_rmsnorm_prequantized(
    activations: Tensor,
    gains_q14: Tensor,
) -> tuple[Tensor, Tensor]:
    if activations.dtype != torch.int8 or activations.shape[-1] != RMS_HIDDEN_SIZE:
        raise ValueError("RMSNorm requires signed-int8 vectors of length 896")
    values = activations.to(torch.int64)
    sumsq = (values * values).sum(dim=-1, keepdim=True)
    mean_square = (sumsq + RMS_HIDDEN_SIZE // 2) // RMS_HIDDEN_SIZE
    root = torch.floor(torch.sqrt(mean_square.to(torch.float64))).to(torch.int64)
    root = (root + (root * root < mean_square).to(torch.int64)).clamp(min=1)
    inv_rms_q30 = (1 << RMS_INV_FRAC) // root
    product = values * gains_q14.to(torch.int64) * inv_rms_q30
    prequantized = product.to(torch.float64) / float(
        1 << (RMS_INV_FRAC + RMS_GAIN_FRAC)
    )
    return prequantized, inv_rms_q30


def quantize_format(value: Tensor, scale: float, qmin: int, qmax: int) -> Tensor:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("output format scale must be finite and positive")
    dtype = torch.int8 if (qmin, qmax) == (-128, 127) else torch.int16
    return torch.round(value / scale).clamp(qmin, qmax).to(dtype)


def raw_summary(value: Tensor) -> dict[str, Any]:
    flat = value.reshape(-1)
    return {
        "absmax": int(flat.to(torch.int64).abs().amax()),
        "dtype": str(value.dtype),
        "sha256": sha256_tensor(value),
        "zero_fraction": float(torch.eq(flat, 0).to(torch.float64).mean()),
    }


def source_record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def run(output_dir: Path) -> dict[str, Any]:
    manifest, config, rtl_binding = load_contracts()
    versions = validate_runtime(config)
    seed_everything(config)
    output_dir.mkdir(parents=True, exist_ok=False)

    calibration_spec = manifest["datasets"]["c4_calibration"]
    evaluation_spec = manifest["datasets"]["c4_en_512"]
    calibration_text = selected_texts(calibration_spec, limit=1)
    evaluation_text = selected_texts(evaluation_spec, limit=1)

    model_spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
    )
    calibration_prompt = tokenize_prompts(
        tokenizer,
        calibration_text,
        calibration_spec["token_limit"],
    )[0][:, :32]
    evaluation_prompt = tokenize_prompts(
        tokenizer,
        evaluation_text,
        evaluation_spec["token_limit"],
    )[0][:, :32]

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

    ranges, operator_ranges = calibrate(model, [calibration_prompt])
    if not ranges:
        raise RuntimeError("linear calibration produced no ranges")
    seed_everything(config)
    _calibration_input, calibration_output = capture_rmsnorm_io(
        model, calibration_prompt
    )
    seed_everything(config)
    evaluation_input, evaluation_output = capture_rmsnorm_io(model, evaluation_prompt)

    input_scale = positive_scale(
        operator_ranges["model.layers.0.input_layernorm.input"].absmax
    )
    source_norm = model.model.layers[0].input_layernorm
    gains_q14 = torch.round(
        source_norm.weight.detach().to(torch.float64) * (1 << RMS_GAIN_FRAC)
    ).clamp(-32768, 32767).to(torch.int16)
    quantized_input = quantize_int8(evaluation_input, input_scale)
    prequantized, inv_rms_q30 = fixed_rmsnorm_prequantized(
        quantized_input, gains_q14
    )

    accepted_raw = fixed_rmsnorm_raw(quantized_input, gains_q14)
    reconstructed_accepted = torch.round(prequantized).clamp(-128, 127).to(torch.int8)
    if not torch.equal(accepted_raw, reconstructed_accepted):
        raise AssertionError("diagnostic does not reproduce accepted RMSNorm output")

    calibration_absmax = float(calibration_output.abs().amax())
    hypotheses = {
        "accepted_unit_scale_int8": {
            "qmin": -128,
            "qmax": 127,
            "scale": 1.0,
            "status": "accepted_contract",
        },
        "calibrated_static_int8": {
            "qmin": -128,
            "qmax": 127,
            "scale": calibration_absmax / 127.0,
            "status": "diagnostic_hypothesis_not_accepted",
        },
        "calibrated_static_int16": {
            "qmin": -32768,
            "qmax": 32767,
            "scale": calibration_absmax / 32767.0,
            "status": "diagnostic_hypothesis_not_accepted",
        },
    }
    comparisons = {}
    for name, hypothesis in hypotheses.items():
        raw = quantize_format(
            prequantized,
            hypothesis["scale"],
            hypothesis["qmin"],
            hypothesis["qmax"],
        )
        decoded = raw.to(torch.float64) * hypothesis["scale"]
        comparisons[name] = {
            "decoded_vs_bf16": compare_tensor(evaluation_output, decoded),
            "format": hypothesis,
            "raw": raw_summary(raw),
        }

    calibration_record_sha256, calibration_record_count = hash_records(calibration_text)
    evaluation_record_sha256, evaluation_record_count = hash_records(evaluation_text)
    calibration_token_sha256, calibration_sequence_count, calibration_token_count = (
        hash_token_sequences([calibration_prompt])
    )
    evaluation_token_sha256, evaluation_sequence_count, evaluation_token_count = (
        hash_token_sequences([evaluation_prompt])
    )
    source_paths = [
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        ROOT / "benchmark" / "quality" / "RTL_BINDING.json",
        ROOT / "design" / "RTL_MANIFEST.json",
        ROOT / "tools" / "ace2_full_model_fixed_point.py",
        Path(__file__),
    ]
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "diagnostic_format_hypotheses_not_acceptance_evidence",
        "gate_passed": False,
        "measurement_started": False,
        "method": {
            "boundary": "model.layers.0.input_layernorm",
            "fixed_kernel": (
                "accepted int8 input quantization, ceil integer RMS, Q2.14 gain, "
                "and Q30 reciprocal held constant before output-format quantization"
            ),
            "output_scale_derivation": (
                "first frozen C4 calibration record BF16 RMSNorm output absmax "
                "divided by positive integer maximum"
            ),
            "scope": "one public calibration record and one public evaluation record, 32 tokens each",
        },
        "model": {**model_spec, "resolved_revision": resolved_revision},
        "input_observations": {
            "calibration": {
                "record_count": calibration_record_count,
                "record_index": calibration_spec["indices"]["start"],
                "record_sha256": calibration_record_sha256,
                "token_count": calibration_token_count,
                "token_sequence_count": calibration_sequence_count,
                "token_sequence_sha256": calibration_token_sha256,
            },
            "evaluation": {
                "record_count": evaluation_record_count,
                "record_index": evaluation_spec["indices"]["start"],
                "record_sha256": evaluation_record_sha256,
                "token_count": evaluation_token_count,
                "token_sequence_count": evaluation_sequence_count,
                "token_sequence_sha256": evaluation_token_sha256,
            },
        },
        "kernel_observations": {
            "calibration_bf16_output_absmax": calibration_absmax,
            "evaluation_input_scale": input_scale,
            "evaluation_inv_rms_q30": {
                "maximum": int(inv_rms_q30.max()),
                "minimum": int(inv_rms_q30.min()),
            },
            "evaluation_quantized_input": raw_summary(quantized_input),
        },
        "comparisons": comparisons,
        "thresholds": config["acceptance_thresholds"],
        "contract": {
            "accepted_rtl": rtl_binding,
            "rounding": config["arithmetic"]["rounding"],
            "saturation": config["arithmetic"]["saturation"],
            "seeds": config["determinism"],
        },
        "sources": [source_record(path) for path in source_paths],
        "runtime": {
            "device": "cpu",
            "packages": versions,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
    }
    result_path = output_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sha256s = write_sha256s(output_dir)
    print(
        "ACE2_RMSNORM_FORMAT_DIAGNOSTIC "
        f"classification={result['classification']} "
        f"sha256s_sha256={sha256_file(sha256s)}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if ROOT.resolve() not in args.output_dir.resolve().parents:
        raise SystemExit("--output-dir must be below the repository root")
    run(args.output_dir)


if __name__ == "__main__":
    main()
