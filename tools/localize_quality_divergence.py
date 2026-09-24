#!/usr/bin/env python3
"""Hash-sealed paired localization of ACE-2 BF16/W4A8 tensor divergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ace2_full_model_fixed_point import (
    FixedRMSNorm,
    PROMPT_MANIFEST,
    QUALITY_CONFIG,
    ROOT,
    calibrate,
    hash_records,
    hash_token_sequences,
    load_contracts,
    replace_fixed_operators,
    replace_linears,
    seed_everything,
    selected_texts,
    sha256_file,
    tokenize_prompts,
    utc_now,
    validate_runtime,
)


def _tensor_output(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _tensor_output(item)
            except TypeError:
                continue
    raise TypeError(f"boundary output does not contain a tensor: {type(value).__name__}")


def _ordered_boundaries(model: nn.Module) -> list[tuple[str, nn.Module]]:
    boundaries: list[tuple[str, nn.Module]] = [("model.embed_tokens", model.model.embed_tokens)]
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        boundaries.extend(
            [
                (f"{prefix}.input_layernorm", layer.input_layernorm),
                (f"{prefix}.self_attn", layer.self_attn),
                (f"{prefix}.post_attention_layernorm", layer.post_attention_layernorm),
                (f"{prefix}.mlp", layer.mlp),
                (prefix, layer),
            ]
        )
    boundaries.extend(
        [
            ("model.norm", model.model.norm),
            ("lm_head", model.lm_head),
        ]
    )
    return boundaries


def capture_boundaries(model: nn.Module, input_ids: Tensor) -> dict[str, Tensor]:
    captured: dict[str, Tensor] = {}
    hooks: list[Any] = []
    for name, module in _ordered_boundaries(model):

        def capture(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            boundary_name: str = name,
        ) -> None:
            if boundary_name in captured:
                raise RuntimeError(f"boundary executed more than once: {boundary_name}")
            captured[boundary_name] = (
                _tensor_output(output).detach().to(device="cpu", dtype=torch.float64).contiguous()
            )

        hooks.append(module.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    expected = [name for name, _module in _ordered_boundaries(model)]
    if list(captured) != expected:
        raise RuntimeError(
            f"captured boundary order differs from execution contract: "
            f"{list(captured)} != {expected}"
        )
    return captured


def _sha256_tensor(value: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare_tensor(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"boundary shape mismatch: {reference.shape} != {candidate.shape}")
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    difference = candidate_flat - reference_flat
    reference_l2 = float(torch.linalg.vector_norm(reference_flat))
    candidate_l2 = float(torch.linalg.vector_norm(candidate_flat))
    difference_l2 = float(torch.linalg.vector_norm(difference))
    norm_product = reference_l2 * candidate_l2
    cosine_similarity = (
        float(torch.dot(reference_flat, candidate_flat)) / norm_product
        if norm_product > 0
        else None
    )
    return {
        "candidate_absmax": float(candidate_flat.abs().amax()),
        "candidate_integer_fraction": float(
            torch.eq(candidate_flat, torch.round(candidate_flat)).to(torch.float64).mean()
        ),
        "candidate_l2": candidate_l2,
        "candidate_sha256": _sha256_tensor(candidate),
        "candidate_zero_fraction": float(torch.eq(candidate_flat, 0).to(torch.float64).mean()),
        "cosine_similarity": cosine_similarity,
        "difference_l2": difference_l2,
        "exact_equal": torch.equal(reference, candidate),
        "mean_absolute_error": float(difference.abs().mean()),
        "reference_absmax": float(reference_flat.abs().amax()),
        "reference_l2": reference_l2,
        "reference_sha256": _sha256_tensor(reference),
        "reference_zero_fraction": float(torch.eq(reference_flat, 0).to(torch.float64).mean()),
        "relative_l2_error": difference_l2 / reference_l2 if reference_l2 > 0 else None,
        "shape": list(reference.shape),
    }


def candidate_boundary_representations(model: nn.Module) -> dict[str, dict[str, Any]]:
    representations: dict[str, dict[str, Any]] = {}
    for name, module in _ordered_boundaries(model):
        if isinstance(module, FixedRMSNorm):
            representations[name] = {
                "dequantization_scale": module.output_scale,
                "storage": "signed_int8_in_float_container",
            }
        else:
            representations[name] = {
                "dequantization_scale": 1.0,
                "storage": "dequantized_float",
            }
    return representations


def write_sha256s(output_dir: Path) -> Path:
    manifest_path = output_dir / "SHA256SUMS"
    files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path != manifest_path
    )
    manifest_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    return manifest_path


def run(
    output_dir: Path,
    *,
    activation_scale_percentile: float | None = None,
) -> dict[str, Any]:
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
    calibration_record_sha256, calibration_record_count = hash_records(calibration_text)
    evaluation_record_sha256, evaluation_record_count = hash_records(evaluation_text)
    calibration_token_sha256, calibration_sequence_count, calibration_token_count = (
        hash_token_sequences([calibration_prompt])
    )
    evaluation_token_sha256, evaluation_sequence_count, evaluation_token_count = (
        hash_token_sequences([evaluation_prompt])
    )

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
    ranges, operator_ranges = calibrate(
        model,
        [calibration_prompt],
        activation_scale_percentile=activation_scale_percentile,
    )

    seed_everything(config)
    baseline = capture_boundaries(model, evaluation_prompt)
    replace_linears(
        model,
        ranges,
        use_percentile_scale=activation_scale_percentile is not None,
    )
    replace_fixed_operators(
        model,
        operator_ranges,
        use_percentile_scale=activation_scale_percentile is not None,
    )
    seed_everything(config)
    fixed = capture_boundaries(model, evaluation_prompt)
    representations = candidate_boundary_representations(model)

    if list(baseline) != list(fixed):
        raise RuntimeError("BF16 and W4A8 boundary sets differ")
    comparisons = {}
    for name in baseline:
        raw_candidate = fixed[name]
        representation = representations[name]
        dequantized_candidate = (
            raw_candidate * representation["dequantization_scale"]
        )
        metrics = compare_tensor(baseline[name], dequantized_candidate)
        metrics["candidate_representation"] = representation
        if representation["storage"] == "signed_int8_in_float_container":
            metrics["candidate_raw_absmax"] = float(raw_candidate.abs().amax())
            metrics["candidate_raw_saturation_fraction"] = float(
                ((raw_candidate == -128) | (raw_candidate == 127))
                .to(torch.float64)
                .mean()
            )
        comparisons[name] = metrics
    first_changed = next(
        (name for name, metrics in comparisons.items() if not metrics["exact_equal"]),
        None,
    )
    if first_changed is None:
        raise RuntimeError("paired models unexpectedly produced identical boundary tensors")

    source_paths = [
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        ROOT / "benchmark" / "quality" / "RTL_BINDING.json",
        ROOT / "design" / "RTL_MANIFEST.json",
        ROOT / "tools" / "ace2_full_model_fixed_point.py",
        Path(__file__),
    ]
    result = {
        "schema_version": 2,
        "generated_at_utc": utc_now(),
        "classification": "diagnostic_localization_not_acceptance_evidence",
        "gate_passed": False,
        "method": {
            "boundary_order": list(comparisons),
            "comparison": (
                "paired BF16 and contract-faithful W4A8 tensors from one forward pass; "
                "raw RMSNorm int8 containers are dequantized by their calibrated static "
                "per-tensor output scale before error metrics"
            ),
            "first_changed_definition": "first execution-ordered boundary whose float64 tensors are not exactly equal",
            "first_changed_boundary": first_changed,
        },
        "model": {**model_spec, "resolved_revision": resolved_revision},
        "input_observations": {
            "calibration": {
                "dataset": {
                    key: calibration_spec[key]
                    for key in ("config", "repository", "revision", "split")
                },
                "record_count": calibration_record_count,
                "record_index": calibration_spec["indices"]["start"],
                "record_sha256": calibration_record_sha256,
                "token_sequence_count": calibration_sequence_count,
                "token_sequence_sha256": calibration_token_sha256,
                "token_count": calibration_token_count,
            },
            "evaluation": {
                "dataset": {
                    key: evaluation_spec[key]
                    for key in ("config", "repository", "revision", "split")
                },
                "record_count": evaluation_record_count,
                "record_index": evaluation_spec["indices"]["start"],
                "record_sha256": evaluation_record_sha256,
                "token_sequence_count": evaluation_sequence_count,
                "token_sequence_sha256": evaluation_token_sha256,
                "token_count": evaluation_token_count,
            },
        },
        "comparisons": comparisons,
        "thresholds": config["acceptance_thresholds"],
        "contract": {
            "diagnostic_activation_scale_percentile": activation_scale_percentile,
            "accepted_rtl": rtl_binding,
            "rounding": config["arithmetic"]["rounding"],
            "saturation": config["arithmetic"]["saturation"],
            "scale_derivation": config["full_model_scope"]["attention_scale_derivation"],
            "seeds": config["determinism"],
        },
        "sources": [
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "runtime": {
            "device": "cpu",
            "packages": versions,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    result_path = output_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sha256s = write_sha256s(output_dir)
    print(
        "ACE2_QUALITY_DIVERGENCE "
        f"classification={result['classification']} "
        f"first_changed_boundary={first_changed} "
        f"sha256s_sha256={sha256_file(sha256s)}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation-scale-percentile", type=float)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if ROOT.resolve() not in args.output_dir.resolve().parents:
        raise SystemExit("--output-dir must be below the repository root")
    run(
        args.output_dir,
        activation_scale_percentile=args.activation_scale_percentile,
    )


if __name__ == "__main__":
    main()
