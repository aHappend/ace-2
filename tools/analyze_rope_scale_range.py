#!/usr/bin/env python3
"""Analyze the frozen RoPE scale range without running a quality evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALES = (
    ROOT
    / "benchmark"
    / "raw"
    / "quality"
    / "reviewer-rmsnorm-contract-l2-round3-smoke"
    / "derived_scales.json"
)
DEFAULT_RESULTS = DEFAULT_SCALES.with_name("results.json")
DEFAULT_CONFIG = ROOT / "benchmark" / "quality" / "QUALITY_CONFIG.json"
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "rmsnorm_repair" / "latest" / "rope_scale_range_analysis.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def signed_bits_for_range(minimum: int, maximum: int) -> int:
    if minimum > maximum:
        raise ValueError("invalid integer range")
    bits = 1
    while minimum < -(1 << (bits - 1)) or maximum > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def analyze(scales_path: Path, results_path: Path, config_path: Path) -> dict[str, Any]:
    scales = load_json(scales_path)
    results = load_json(results_path)
    config = load_json(config_path)
    attention = scales.get("attention")
    if not isinstance(attention, dict) or not attention:
        raise ValueError("derived scales do not contain nonempty attention metadata")

    operator_formats = config["full_model_scope"]["operator_formats"]
    scale_derivation = config["full_model_scope"]["attention_scale_derivation"]
    if operator_formats["rope_input"] != "signed_int16_Q2.13":
        raise ValueError("analyzer only supports the frozen signed-int16 Q2.13 contract")

    fraction_bits = 13
    metadata_max = 32767
    layers: list[dict[str, Any]] = []
    for name, metadata in sorted(
        attention.items(), key=lambda item: int(item[0].split(".")[2])
    ):
        conversion_q13 = metadata["rope_conversion_q13"]
        if not isinstance(conversion_q13, int) or not 1 <= conversion_q13 <= metadata_max:
            raise ValueError(f"{name} has invalid Q2.13 conversion metadata")
        requested_conversion_q13 = round(
            metadata["rope_conversion_scale"] * (1 << fraction_bits)
        )
        required_pre_rotation_bits = signed_bits_for_range(
            -128 * requested_conversion_q13,
            127 * requested_conversion_q13,
        )
        query_saturation = metadata["query_rope_q13_saturation"]
        key_saturation = metadata["key_rope_q13_saturation"]
        target_product = scale_derivation["rope_qk_output_scale_product"]
        realized_product = metadata["query_key_rope_output_scale_product"]
        layers.append(
            {
                "layer": int(name.split(".")[2]),
                "conversion_q13": conversion_q13,
                "conversion_q13_requested": requested_conversion_q13,
                "conversion_scale_requested": metadata["rope_conversion_scale"],
                "conversion_scale_realized": conversion_q13 / float(1 << fraction_bits),
                "minimum_signed_pre_rotation_bits_for_full_int8_domain": (
                    required_pre_rotation_bits
                ),
                "largest_positive_activation_code_without_pre_rotation_saturation": (
                    metadata_max // conversion_q13
                ),
                "metadata_saturated": metadata["rope_metadata_saturated"],
                "qk_output_scale_product": realized_product,
                "qk_output_scale_product_relative_error": abs(
                    realized_product - target_product
                )
                / target_product,
                "query_pre_rotation_saturation_fraction": query_saturation["fraction"],
                "key_pre_rotation_saturation_fraction": key_saturation["fraction"],
            }
        )

    query_fractions = [
        layer["query_pre_rotation_saturation_fraction"] for layer in layers
    ]
    key_fractions = [layer["key_pre_rotation_saturation_fraction"] for layer in layers]
    metadata_saturated_layers = [
        layer["layer"] for layer in layers if layer["metadata_saturated"]
    ]
    product_error_layers = [
        layer["layer"]
        for layer in layers
        if layer["qk_output_scale_product_relative_error"] > 0.01
    ]
    measured_saturation_layers = [
        layer["layer"]
        for layer in layers
        if layer["query_pre_rotation_saturation_fraction"] > 0.0
        or layer["key_pre_rotation_saturation_fraction"] > 0.0
    ]
    positive_code_limits = [
        layer["largest_positive_activation_code_without_pre_rotation_saturation"]
        for layer in layers
    ]
    required_pre_rotation_bits = [
        layer["minimum_signed_pre_rotation_bits_for_full_int8_domain"]
        for layer in layers
    ]

    metrics = results.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("smoke results do not contain metrics")
    ratios = {
        dataset: dataset_metrics["ratio"]
        for dataset, dataset_metrics in sorted(metrics.items())
        if isinstance(dataset_metrics, dict) and "ratio" in dataset_metrics
    }
    thresholds = config["acceptance_thresholds"]
    smoke_threshold_failures = {
        dataset: {
            "ratio": ratio,
            "threshold": thresholds[f"{dataset}_perplexity_ratio_max"],
        }
        for dataset, ratio in ratios.items()
        if ratio > thresholds[f"{dataset}_perplexity_ratio_max"]
    }

    all_layers_saturate = len(measured_saturation_layers) == len(layers)
    if not all_layers_saturate:
        raise ValueError("frozen evidence no longer reproduces saturation in every layer")
    if set(smoke_threshold_failures) != {"c4_en_512", "wikitext2"}:
        raise ValueError("frozen evidence no longer reproduces both smoke threshold failures")

    return {
        "schema_version": 1,
        "classification": "executable_range_analysis_not_acceptance_evidence",
        "contract": {
            "rope_input": operator_formats["rope_input"],
            "fraction_bits": fraction_bits,
            "signed_storage_bits": 16,
            "metadata_integer_max": metadata_max,
            "metadata_scale_max": metadata_max / float(1 << fraction_bits),
            "target_qk_output_scale_product": scale_derivation[
                "rope_qk_output_scale_product"
            ],
        },
        "sources": [
            {
                "path": relative_path(path),
                "sha256": sha256(path),
            }
            for path in (scales_path, results_path, config_path)
        ],
        "observations": {
            "layer_count": len(layers),
            "metadata_saturated_layers": metadata_saturated_layers,
            "qk_product_error_gt_1pct_layers": product_error_layers,
            "measured_pre_rotation_saturation_layers": measured_saturation_layers,
            "largest_positive_activation_code_without_pre_rotation_saturation_range": {
                "min": min(positive_code_limits),
                "max": max(positive_code_limits),
            },
            "minimum_signed_pre_rotation_bits_for_full_int8_domain_range": {
                "min": min(required_pre_rotation_bits),
                "max": max(required_pre_rotation_bits),
            },
            "full_signed_int8_domain_representable_in_current_storage_all_layers": all(
                bits <= 16 for bits in required_pre_rotation_bits
            ),
            "query_pre_rotation_saturation_fraction_range": {
                "min": min(query_fractions),
                "max": max(query_fractions),
            },
            "key_pre_rotation_saturation_fraction_range": {
                "min": min(key_fractions),
                "max": max(key_fractions),
            },
            "smoke_perplexity_ratios": ratios,
            "smoke_threshold_failures": smoke_threshold_failures,
        },
        "layers": layers,
        "conclusion": {
            "all_layers_have_measured_pre_rotation_saturation": all_layers_saturate,
            "current_contract_acceptance_supported": False,
            "reason": (
                "The frozen signed-int16 Q2.13 pre-rotation intermediate clips "
                "observed query or key values in every layer; changing this "
                "structurally distinct RoPE/attention contract requires a separate "
                "operator-owned bounded repair."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-scales", type=Path, default=DEFAULT_SCALES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = analyze(args.derived_scales, args.results, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"ROPE_SCALE_RANGE_ANALYSIS_PASS layers={len(output['layers'])} "
        f"output={relative_path(args.output)}"
    )


if __name__ == "__main__":
    main()
