#!/usr/bin/env python3
"""Decompose the frozen C4 layer-0 V-projection numerical error."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from ace2_full_model_fixed_point import (
    ACTIVE_ROPE_MECHANISM,
    FixedAttention,
    FixedRMSNorm,
    PROMPT_MANIFEST,
    QUALITY_CONFIG,
    ROOT,
    W4A8Linear,
    calibrate,
    derive_multiplier,
    fixed_rmsnorm_raw,
    hash_records,
    hash_token_sequences,
    load_contracts,
    quantize_int8,
    replace_fixed_operators,
    replace_linears,
    round_shift_even,
    seed_everything,
    selected_texts,
    sha256_file,
    tokenize_prompts,
    utc_now,
    validate_runtime,
)
from localize_layer0_paired_divergence import (
    capture_layer0_inputs,
    current_source_binding,
)
from localize_quality_divergence import compare_tensor, write_sha256s


MODEL_SOURCE = ROOT / "tools" / "ace2_full_model_fixed_point.py"
LOCALIZER_SOURCE = ROOT / "tools" / "localize_layer0_paired_divergence.py"
QUALITY_LOCALIZER_SOURCE = ROOT / "tools" / "localize_quality_divergence.py"
C4_DATASETS = ("c4_calibration", "c4_en_512")


def artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_span(value: Any) -> dict[str, Any]:
    source_path = Path(inspect.getsourcefile(value) or "").resolve()
    source_lines, line_start = inspect.getsourcelines(value)
    return {
        "line_end": line_start + len(source_lines) - 1,
        "line_start": line_start,
        "path": source_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(source_path),
        "symbol": value.__qualname__,
    }


def source_range(
    path: Path,
    line_start: int,
    line_end: int,
    symbol: str,
) -> dict[str, Any]:
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    if not 1 <= line_start <= line_end <= line_count:
        raise ValueError(f"invalid source range for {path}: {line_start}-{line_end}")
    return {
        "line_end": line_end,
        "line_start": line_start,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "symbol": symbol,
    }


def tensor_coordinate(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    coordinate: list[int] = []
    remainder = flat_index
    for dimension in reversed(shape):
        coordinate.append(remainder % dimension)
        remainder //= dimension
    if remainder:
        raise ValueError("flat tensor index exceeds shape")
    return list(reversed(coordinate))


def point_difference(
    previous: Tensor,
    current: Tensor,
    flat_index: int,
) -> dict[str, Any]:
    previous_value = float(previous.reshape(-1)[flat_index])
    current_value = float(current.reshape(-1)[flat_index])
    return {
        "absolute_delta": abs(current_value - previous_value),
        "coordinate": tensor_coordinate(flat_index, tuple(previous.shape)),
        "current": current_value,
        "delta": current_value - previous_value,
        "flat_index": flat_index,
        "previous": previous_value,
    }


def incremental_metrics(
    previous: Tensor,
    current: Tensor,
    reference: Tensor,
) -> dict[str, Any]:
    if previous.shape != current.shape or previous.shape != reference.shape:
        raise ValueError("decomposition tensors must have identical shapes")
    previous = previous.detach().to(device="cpu", dtype=torch.float64)
    current = current.detach().to(device="cpu", dtype=torch.float64)
    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    difference = current - previous
    flat_difference = difference.reshape(-1)
    nonzero = torch.nonzero(torch.ne(flat_difference, 0), as_tuple=False)
    first_difference = (
        None
        if not nonzero.numel()
        else point_difference(previous, current, int(nonzero[0, 0]))
    )
    max_flat_index = int(torch.argmax(flat_difference.abs()))
    reference_l2 = float(torch.linalg.vector_norm(reference.reshape(-1)))
    difference_l2 = float(torch.linalg.vector_norm(flat_difference))
    return {
        "difference_l2": difference_l2,
        "exact_equal": torch.equal(previous, current),
        "first_difference": first_difference,
        "max_absolute_difference": point_difference(
            previous,
            current,
            max_flat_index,
        ),
        "mean_absolute_difference": float(flat_difference.abs().mean()),
        "relative_l2_to_bf16_reference": (
            difference_l2 / reference_l2 if reference_l2 > 0 else None
        ),
    }


def reshape_v(value: Tensor, num_heads: int, head_dim: int) -> Tensor:
    if value.ndim != 3:
        raise ValueError("V-projection tensor must have batch, sequence, channel axes")
    return value.view(value.shape[0], value.shape[1], num_heads, head_dim).transpose(
        1,
        2,
    )


def fp64_linear(inputs: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return F.linear(
        inputs.to(torch.float64),
        weight.to(torch.float64),
        None if bias is None else bias.to(torch.float64),
    )


def multiplier_scaled_accumulator(
    accumulator: Tensor,
    multiplier: Tensor,
    right_shift: Tensor,
    output_scale: Tensor,
) -> Tensor:
    product = accumulator * multiplier
    raw_pre_round = torch.ldexp(
        product.to(torch.float64),
        -right_shift.to(torch.int32),
    )
    return raw_pre_round * output_scale


def c4_observations(
    manifest: dict[str, Any],
    texts: dict[str, list[str]],
    prompts: dict[str, Tensor],
) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for dataset in C4_DATASETS:
        spec = manifest["datasets"][dataset]
        record_hash, record_count = hash_records(texts[dataset])
        token_hash, sequence_count, token_count = hash_token_sequences(
            [prompts[dataset]]
        )
        observations[dataset] = {
            "config": spec["config"],
            "record_count": record_count,
            "record_sha256": record_hash,
            "repository": spec["repository"],
            "revision": spec["revision"],
            "split": spec["split"],
            "tokenized": {
                "sequence_count": sequence_count,
                "token_count": token_count,
                "token_sequence_sha256": token_hash,
            },
        }
    return observations


def run(output_dir: Path, frozen_baseline_path: Path) -> dict[str, Any]:
    manifest, config, _rtl_binding = load_contracts(require_rtl_binding=False)
    versions = validate_runtime(config)
    seed_everything(config)
    output_dir.mkdir(parents=True, exist_ok=False)

    texts = {
        dataset: selected_texts(manifest["datasets"][dataset], limit=1)
        for dataset in C4_DATASETS
    }
    model_spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
    )
    frozen_baseline = json.loads(frozen_baseline_path.read_text(encoding="utf-8"))
    prompts = {
        dataset: tokenize_prompts(
            tokenizer,
            texts[dataset],
            manifest["datasets"][dataset]["token_limit"],
        )[0]
        for dataset in C4_DATASETS
    }
    for dataset, prompt in prompts.items():
        token_count = frozen_baseline["input_observations"][dataset]["tokenized"][
            "token_count"
        ]
        prompts[dataset] = prompt[:, :token_count]
    observations = c4_observations(manifest, texts, prompts)
    expected_observations = {
        dataset: frozen_baseline["input_observations"][dataset]
        for dataset in C4_DATASETS
    }
    if observations != expected_observations:
        raise RuntimeError("C4-only decomposition did not reproduce frozen inputs")

    source_paths = [
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        MODEL_SOURCE,
        LOCALIZER_SOURCE,
        QUALITY_LOCALIZER_SOURCE,
        Path(__file__),
    ]
    reference_source_binding = current_source_binding(source_paths)
    diagnostic_binding = {
        "acceptance_claimed": False,
        "binding": None,
        "candidate_id": None,
        "rtl_hash": None,
        "source_hash_list": None,
        "status": "not_applicable_reference_only_diagnostic",
    }
    run_contract = {
        "schema_version": 1,
        "status": "frozen_before_measurement",
        "created_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "scope": {
            "dataset": "c4_en_512",
            "calibration_dataset": "c4_calibration",
            "layer": 0,
            "projection": "v_proj",
            "requested_components": [
                "rmsnorm_input_quantization",
                "w4_weight_approximation",
                "accumulator_scaling",
                "final_int8_requantization",
            ],
        },
        "diagnostic_binding": diagnostic_binding,
        "reference_source_binding": reference_source_binding,
        "fixed_point_model": artifact(MODEL_SOURCE),
        "frozen_baseline": artifact(frozen_baseline_path),
        "input_observations": observations,
        "model": model_spec,
        "source_contracts": {
            "prompt_manifest_sha256": sha256_file(PROMPT_MANIFEST),
            "quality_config_sha256": sha256_file(QUALITY_CONFIG),
            "decomposer_sha256": sha256_file(Path(__file__)),
        },
    }
    run_contract_path = output_dir / "run_contract.json"
    write_json(run_contract_path, run_contract)

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
    ranges, operator_ranges = calibrate(model, [prompts["c4_calibration"]])

    seed_everything(config)
    baseline_hidden, _position_embeddings, _attention_mask = capture_layer0_inputs(
        model,
        prompts["c4_en_512"],
    )
    baseline_attention = model.model.layers[0].self_attn
    source_v_proj = baseline_attention.v_proj
    baseline_v_heads = source_v_proj.out_features // baseline_attention.head_dim
    source_weight = source_v_proj.weight.detach().to(torch.float64).clone()
    source_bias = (
        None
        if source_v_proj.bias is None
        else source_v_proj.bias.detach().to(torch.float64).clone()
    )
    with torch.inference_mode():
        bf16_reference = reshape_v(
            source_v_proj(baseline_hidden).to(torch.float64),
            baseline_v_heads,
            baseline_attention.head_dim,
        )
        fp64_reference = reshape_v(
            fp64_linear(baseline_hidden, source_weight, source_bias),
            baseline_v_heads,
            baseline_attention.head_dim,
        )

    replace_linears(
        model,
        ranges,
        rope_diagnostic_mechanism=ACTIVE_ROPE_MECHANISM,
    )
    replace_fixed_operators(
        model,
        operator_ranges,
        rope_diagnostic_mechanism=ACTIVE_ROPE_MECHANISM,
    )
    layer = model.model.layers[0]
    if not isinstance(layer.input_layernorm, FixedRMSNorm) or not isinstance(
        layer.self_attn,
        FixedAttention,
    ):
        raise TypeError("layer-0 fixed operators were not installed")
    v_proj = layer.self_attn.v_proj
    if not isinstance(v_proj, W4A8Linear):
        raise TypeError("layer-0 V projection is not W4A8Linear")

    seed_everything(config)
    fixed_hidden, _position_embeddings, _attention_mask = capture_layer0_inputs(
        model,
        prompts["c4_en_512"],
    )
    if not torch.equal(fixed_hidden, torch.round(fixed_hidden)):
        raise RuntimeError("fixed RMSNorm output is not an exact integer container")
    qinput = fixed_hidden.to(torch.int8)
    input_scale = float(layer.input_layernorm.output_scale)
    if not math.isclose(v_proj.hardware_input_scale, input_scale, rel_tol=0, abs_tol=0):
        raise RuntimeError("V projection input scale is not bound to RMSNorm output scale")
    output_scale = v_proj.output_scale_per_channel.detach().to(torch.float64)
    fixed_v_heads = v_proj.out_features // layer.self_attn.head_dim
    native_scale = input_scale * v_proj.weight_scale.detach().to(torch.float64)
    dequantized_input = qinput.to(torch.float64) * input_scale
    dequantized_weight = (
        v_proj.qweight.detach().to(torch.float64)
        * v_proj.weight_scale.detach().to(torch.float64)[:, None]
    )

    with torch.inference_mode():
        rmsnorm_input_quantized = reshape_v(
            fp64_linear(dequantized_input, source_weight, source_bias),
            fixed_v_heads,
            layer.self_attn.head_dim,
        )
        w4_weight_approximated = reshape_v(
            fp64_linear(dequantized_input, dequantized_weight, source_bias),
            fixed_v_heads,
            layer.self_attn.head_dim,
        )
        accumulator = v_proj.accumulator_quantized(qinput)
        accumulator_native = reshape_v(
            (accumulator.to(torch.float64) * native_scale).reshape(
                *qinput.shape[:-1],
                v_proj.out_features,
            ),
            fixed_v_heads,
            layer.self_attn.head_dim,
        )
        accumulator_scaled = reshape_v(
            multiplier_scaled_accumulator(
                accumulator,
                v_proj.multiplier.detach(),
                v_proj.right_shift.detach(),
                output_scale,
            ).reshape(*qinput.shape[:-1], v_proj.out_features),
            fixed_v_heads,
            layer.self_attn.head_dim,
        )
        final_raw = v_proj.requantize_accumulator(accumulator, qinput.shape[:-1])
        final_dequantized = reshape_v(
            final_raw.to(torch.float64) * output_scale,
            fixed_v_heads,
            layer.self_attn.head_dim,
        )
        actual_fixed_raw = v_proj.forward_hardware_input(fixed_hidden)

    if not torch.equal(final_raw, actual_fixed_raw):
        raise RuntimeError("instrumented final requantization differs from fixed V path")
    if not torch.equal(
        final_dequantized,
        reshape_v(
            actual_fixed_raw.to(torch.float64) * output_scale,
            fixed_v_heads,
            layer.self_attn.head_dim,
        ),
    ):
        raise RuntimeError("instrumented final dequantization differs from fixed V path")

    product = accumulator * v_proj.multiplier.detach()
    rounded_unclamped = round_shift_even(product, v_proj.right_shift.detach())
    saturated_elements = int(
        ((rounded_unclamped < -128) | (rounded_unclamped > 127)).sum()
    )
    input_storage_limit_elements = int(((qinput == -128) | (qinput == 127)).sum())
    output_storage_limit_elements = int(
        ((final_raw == -128) | (final_raw == 127)).sum()
    )

    stages = [
        (
            "bf16_reference_alignment",
            "bf16_reference",
            bf16_reference,
            fp64_reference,
            [source_span(fp64_linear)],
        ),
        (
            "rmsnorm_input_quantization",
            "fp64_reference",
            fp64_reference,
            rmsnorm_input_quantized,
            [
                source_span(quantize_int8),
                source_span(fixed_rmsnorm_raw),
                source_span(FixedRMSNorm.forward),
                source_span(W4A8Linear.forward_hardware_input),
            ],
        ),
        (
            "w4_weight_approximation",
            "rmsnorm_input_quantization",
            rmsnorm_input_quantized,
            w4_weight_approximated,
            [source_span(W4A8Linear.__init__)],
        ),
        (
            "integer_accumulator_formation",
            "w4_weight_approximation",
            w4_weight_approximated,
            accumulator_native,
            [
                source_span(W4A8Linear._metadata_for_input_scale),
                source_span(W4A8Linear.accumulator_quantized),
            ],
        ),
        (
            "accumulator_scaling",
            "integer_accumulator_formation",
            accumulator_native,
            accumulator_scaled,
            [
                source_span(W4A8Linear._metadata_for_input_scale),
                source_span(derive_multiplier),
                source_span(multiplier_scaled_accumulator),
            ],
        ),
        (
            "final_int8_requantization",
            "accumulator_scaling",
            accumulator_scaled,
            final_dequantized,
            [
                source_span(round_shift_even),
                source_span(W4A8Linear.requantize_accumulator),
            ],
        ),
    ]
    decomposition: dict[str, Any] = {}
    for name, previous_name, previous, current, provenance in stages:
        decomposition[name] = {
            "from": previous_name,
            "incremental": incremental_metrics(previous, current, bf16_reference),
            "cumulative_vs_bf16_reference": compare_tensor(
                bf16_reference,
                current,
            ),
            "source_provenance": provenance,
            "to": name,
        }

    component_sum = (
        (fp64_reference - bf16_reference)
        + (rmsnorm_input_quantized - fp64_reference)
        + (w4_weight_approximated - rmsnorm_input_quantized)
        + (accumulator_native - w4_weight_approximated)
        + (accumulator_scaled - accumulator_native)
        + (final_dequantized - accumulator_scaled)
    )
    closure = component_sum - (final_dequantized - bf16_reference)
    coordinate_zero = (0, 0, 0, 0)
    coordinate_zero_trace = {
        "coordinate": list(coordinate_zero),
        "values": {
            "bf16_reference": float(bf16_reference[coordinate_zero]),
            "fp64_reference": float(fp64_reference[coordinate_zero]),
            "rmsnorm_input_quantization": float(
                rmsnorm_input_quantized[coordinate_zero]
            ),
            "w4_weight_approximation": float(w4_weight_approximated[coordinate_zero]),
            "integer_accumulator_formation": float(accumulator_native[coordinate_zero]),
            "accumulator_scaling": float(accumulator_scaled[coordinate_zero]),
            "final_int8_requantization": float(final_dequantized[coordinate_zero]),
        },
        "final_raw": int(
            reshape_v(final_raw, fixed_v_heads, layer.self_attn.head_dim)[
                coordinate_zero
            ]
        ),
    }
    pre_round_raw = torch.ldexp(
        product.to(torch.float64),
        -v_proj.right_shift.detach().to(torch.int32),
    )
    pre_round_raw_v = reshape_v(
        pre_round_raw.reshape(*qinput.shape[:-1], v_proj.out_features),
        fixed_v_heads,
        layer.self_attn.head_dim,
    )
    rounded_unclamped_v = reshape_v(
        rounded_unclamped.reshape(*qinput.shape[:-1], v_proj.out_features),
        fixed_v_heads,
        layer.self_attn.head_dim,
    )
    final_raw_v = reshape_v(final_raw, fixed_v_heads, layer.self_attn.head_dim)
    saturation_mask = (rounded_unclamped_v < -128) | (rounded_unclamped_v > 127)
    saturation_coordinates = torch.nonzero(saturation_mask, as_tuple=False)
    accumulator_projected = accumulator.reshape(
        *qinput.shape[:-1],
        v_proj.out_features,
    )
    saturation_records: list[dict[str, Any]] = []
    for coordinate_tensor in saturation_coordinates:
        coordinate = tuple(int(item) for item in coordinate_tensor.tolist())
        output_channel = coordinate[1] * layer.self_attn.head_dim + coordinate[3]
        saturation_records.append(
            {
                "accumulator": int(
                    accumulator_projected[
                        coordinate[0],
                        coordinate[2],
                        output_channel,
                    ]
                ),
                "clamped_raw": int(final_raw_v[coordinate]),
                "coordinate": list(coordinate),
                "final_dequantized": float(final_dequantized[coordinate]),
                "multiplier": int(v_proj.multiplier[output_channel]),
                "pre_round_dequantized": float(accumulator_scaled[coordinate]),
                "pre_round_raw": float(pre_round_raw_v[coordinate]),
                "right_shift": int(v_proj.right_shift[output_channel]),
                "rounded_unclamped_raw": int(rounded_unclamped_v[coordinate]),
            }
        )
    bias_metadata = None
    if source_bias is not None and v_proj.bias_accumulator is not None:
        realized_bias = (
            v_proj.bias_accumulator.detach().to(torch.float64) * native_scale
        )
        bias_metadata = {
            "exact_bias_absmax": float(source_bias.abs().amax()),
            "quantized_bias_absmax": float(realized_bias.abs().amax()),
            "quantization": incremental_metrics(
                source_bias,
                realized_bias,
                source_bias,
            ),
        }

    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "diagnostic_c4_only_layer0_v_projection_decomposition_not_acceptance_evidence",
        "gate_passed": False,
        "scope": {
            "dataset": "c4_en_512",
            "calibration_dataset": "c4_calibration",
            "layer": 0,
            "projection": "v_proj",
            "shape": list(bf16_reference.shape),
        },
        "decomposition_order": [name for name, *_rest in stages],
        "decomposition": decomposition,
        "closure": {
            "exact_equal": torch.equal(component_sum, final_dequantized - bf16_reference),
            "max_absolute_residual": float(closure.abs().amax()),
            "within_fp64_roundoff": bool(float(closure.abs().amax()) < 1e-15),
        },
        "coordinate_traces": {
            "existing_first_v_divergence": coordinate_zero_trace,
            "final_requant_saturations": saturation_records,
        },
        "final_fixed_vs_bf16_reference": compare_tensor(
            bf16_reference,
            final_dequantized,
        ),
        "representations": {
            "bf16_reference": "executed BF16 V projection",
            "fp64_reference": "FP64 recomputation from captured BF16 tensors",
            "rmsnorm_input_quantization": (
                "dequantized fixed RMSNorm signed-int8 output with original V weight/bias"
            ),
            "w4_weight_approximation": (
                "dequantized signed-int4 V weight with exact source bias"
            ),
            "integer_accumulator_formation": (
                "signed-int32 dot product plus quantized bias at input_scale*weight_scale"
            ),
            "accumulator_scaling": (
                "unrounded accumulator*integer_multiplier/2^right_shift at output scale"
            ),
            "final_int8_requantization": (
                "ties-to-even right shift, signed-int8 clamp, and output dequantization"
            ),
        },
        "metadata": {
            "rmsnorm_output_scale": input_scale,
            "v_output_scale_min": float(output_scale.min()),
            "v_output_scale_max": float(output_scale.max()),
            "weight_scale_min": float(v_proj.weight_scale.min()),
            "weight_scale_max": float(v_proj.weight_scale.max()),
            "multiplier_min": int(v_proj.multiplier.min()),
            "multiplier_max": int(v_proj.multiplier.max()),
            "right_shift_min": int(v_proj.right_shift.min()),
            "right_shift_max": int(v_proj.right_shift.max()),
            "input_storage_limit_elements": input_storage_limit_elements,
            "output_storage_limit_elements": output_storage_limit_elements,
            "final_requant_saturated_elements": saturated_elements,
            "bias": bias_metadata,
        },
        "exact_checks": {
            "final_raw_matches_fixed_v_path": torch.equal(final_raw, actual_fixed_raw),
            "hardware_input_scale_matches_rmsnorm_output_scale": math.isclose(
                v_proj.hardware_input_scale,
                input_scale,
                rel_tol=0,
                abs_tol=0,
            ),
        },
        "input_observations": observations,
        "model": {**model_spec, "resolved_revision": resolved_revision},
        "bindings": {
            "diagnostic_rtl": diagnostic_binding,
            "reference_sources": reference_source_binding,
            "fixed_point_model": artifact(MODEL_SOURCE),
            "frozen_baseline": artifact(frozen_baseline_path),
            "frozen_inputs": observations,
        },
        "artifacts": {
            "run_contract": artifact(run_contract_path),
            "sources": [
                artifact(path) for path in source_paths
            ],
        },
        "runtime": {
            "device": "cpu",
            "packages": versions,
            "torch": torch.__version__,
        },
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    sha256s = write_sha256s(output_dir)
    print(
        "ACE2_C4_V_PROJECTION_DECOMPOSITION "
        f"final_relative_l2={result['final_fixed_vs_bf16_reference']['relative_l2_error']:.17g} "
        f"closure_max={result['closure']['max_absolute_residual']:.17g} "
        f"final_raw_exact={result['exact_checks']['final_raw_matches_fixed_v_path']} "
        f"sha256s_sha256={sha256_file(sha256s)}"
    )
    return result


def repository_file(path: Path, name: str) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    if ROOT.resolve() not in resolved.parents or not resolved.is_file():
        raise SystemExit(f"--{name} must name a repository file")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-baseline", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir = output_dir.resolve()
    if ROOT.resolve() not in output_dir.parents:
        raise SystemExit("--output-dir must remain inside the repository")
    run(
        output_dir,
        repository_file(args.frozen_baseline, "frozen-baseline"),
    )


if __name__ == "__main__":
    main()
