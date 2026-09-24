#!/usr/bin/env python3
"""Generate the frozen all-layer Q/K residual Scale32 model-image table."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from ace2_quality_contracts import (
    ceil_scale32_from_ratio,
    scale32_ratio,
    unpack_scale32,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_qk_residual_cross_term_attention_v1"
CALIBRATION_DIR = ROOT / "benchmark/raw/quality/qk-residual-scale32-calibration-c4-validation-0-64-512"
SOURCE = CALIBRATION_DIR / "derived_scales.json"
SOURCE_SHA256 = "b75d2ee5820992edf7f0351c19f29cdfd2552f2eb9d2f9c0404e5056cec39e06"
INPUT_OBSERVATIONS = CALIBRATION_DIR / "input_observations.json"
INPUT_OBSERVATIONS_SHA256 = "78bbce1666b626720d5df30835e43bd5e468dac358cd8bc2491f4d0caeb71638"
RESULTS = CALIBRATION_DIR / "results.json"
RESULTS_SHA256 = "19b915c2dc47a2f6140029d29d704e0df483b67a91dae6a8a2e0d3316b51f251"
RUN_CONTRACT = CALIBRATION_DIR / "run_contract.json"
RUN_CONTRACT_SHA256 = "31462a9720e8578850db2529b897def8bc9ac8b9bd14a1cdd7f88601c21c9118"
QUALITY_CONFIG = ROOT / "benchmark/quality/QUALITY_CONFIG.json"
QUALITY_CONFIG_SHA256 = "50fe8c986c556bd1247e156c9f65de1e1137ed87542aca9112d089b15d226588"
PROMPT_MANIFEST = ROOT / "benchmark/quality/PROMPT_MANIFEST.json"
PROMPT_MANIFEST_SHA256 = "9ee394d7d344d6f14e6829cbf8bea27b5508c7fe5344d941495f5668ca32335c"
CALIBRATION_RUNNER = ROOT / "tools/calibrate_qk_residual_scale32_source.py"
CALIBRATION_RUNNER_SHA256 = "586566738bc1f8b50006009679ce96445d949249785cbe517428ea476f9bfea8"
FULL_MODEL_RUNNER = ROOT / "tools/ace2_full_model_fixed_point.py"
FULL_MODEL_RUNNER_SHA256 = "ec499efbcd5f82f25486d5d3238e41341614b2e02c67f79fc2de2f2a7c9e1a76"
GENERATOR = Path(__file__).resolve()
DEFAULT_OUTPUT = ROOT / "reference/generated/qk_residual_scale32_metadata.json"
MODEL = {
    "repository": "Qwen/Qwen2.5-0.5B",
    "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
}
CALIBRATION_SCOPE = {
    "dataset": "allenai/c4",
    "config": "en",
    "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    "split": "validation",
    "record_indices": {"start": 0, "stop": 64},
    "token_limit_per_record": 512,
    "scale_rule": "per_layer_per_q_or_k_head_maximum_absolute_observed_bf16_projection_output_divided_by_127",
}
CALIBRATION_OBSERVATION = {
    "c4_calibration": {
        "config": "en",
        "record_count": 64,
        "record_indices": {"start": 0, "stop": 64},
        "record_sha256": "ef13326324b4712c3066d5af746d7f3f29f66e728188e973ea6193ed13b5ac35",
        "repository": "allenai/c4",
        "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "split": "validation",
        "tokenized": {
            "sequence_count": 64,
            "token_count": 16940,
            "token_limit_per_record": 512,
            "token_sequence_sha256": "80bf13275aeeb986af5f3f03994c10f7278f2097de425efeb50ccceeb36b9642",
        },
    }
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"frozen input hash differs for {path.relative_to(ROOT)}: {observed}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def artifact(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    observed = sha256(path)
    if expected_sha256 is not None:
        require(observed == expected_sha256, f"frozen input hash differs for {path.relative_to(ROOT)}: {observed}")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": observed}


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def scale32_record(value: Decimal) -> tuple[int, int, int]:
    numerator, denominator = value.as_integer_ratio()
    record = ceil_scale32_from_ratio(numerator, denominator)
    significand, exponent = unpack_scale32(record)
    return record, significand, exponent


def residual_record(baseline_record: int) -> tuple[int, int, int]:
    numerator, denominator = scale32_ratio(baseline_record)
    record = ceil_scale32_from_ratio(numerator, 14 * denominator)
    significand, exponent = unpack_scale32(record)
    return record, significand, exponent


def validate_source_provenance() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_contract_sources = {
        "calibration_runner": artifact(CALIBRATION_RUNNER, CALIBRATION_RUNNER_SHA256),
        "full_model_runner": artifact(FULL_MODEL_RUNNER, FULL_MODEL_RUNNER_SHA256),
        "prompt_manifest": artifact(PROMPT_MANIFEST, PROMPT_MANIFEST_SHA256),
        "quality_config": artifact(QUALITY_CONFIG, QUALITY_CONFIG_SHA256),
    }
    artifact(SOURCE, SOURCE_SHA256)
    artifact(INPUT_OBSERVATIONS, INPUT_OBSERVATIONS_SHA256)
    artifact(RESULTS, RESULTS_SHA256)
    artifact(RUN_CONTRACT, RUN_CONTRACT_SHA256)

    prompt_manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    prompt_scope = prompt_manifest["datasets"]["c4_calibration"]
    require(prompt_manifest["model"] == MODEL, "prompt-manifest model differs from frozen calibration model")
    require(
        {
            "dataset": prompt_scope["repository"],
            "config": prompt_scope["config"],
            "revision": prompt_scope["revision"],
            "split": prompt_scope["split"],
            "record_indices": prompt_scope["indices"],
            "token_limit_per_record": prompt_scope["token_limit"],
            "scale_rule": CALIBRATION_SCOPE["scale_rule"],
        }
        == CALIBRATION_SCOPE,
        "prompt-manifest calibration slice differs from the frozen Scale32 scope",
    )
    quality_config = json.loads(QUALITY_CONFIG.read_text(encoding="utf-8"))
    require(
        quality_config["baseline"] == {
            "dtype": "bfloat16",
            "model_repository": MODEL["repository"],
            "model_revision": MODEL["revision"],
        },
        "quality-config baseline model differs from frozen calibration model",
    )
    require(
        quality_config["activation_quantization"]["scale"]
        == "maximum_absolute_observed_bf16_value_divided_by_127",
        "quality-config activation scale rule differs",
    )

    observations = json.loads(INPUT_OBSERVATIONS.read_text(encoding="utf-8"))
    require(observations == CALIBRATION_OBSERVATION, "calibration input observations differ")

    run_contract = json.loads(RUN_CONTRACT.read_text(encoding="utf-8"))
    require(run_contract["status"] == "frozen_before_measurement", "calibration run contract was not frozen before measurement")
    require(run_contract["mode"] == "calibration_only", "Scale32 source was not produced by calibration-only mode")
    require(run_contract["purpose"] == "architecture_scale32_source_no_quality_evaluation", "calibration purpose differs")
    require(run_contract["contract_id"] == CONTRACT, "calibration run contract id differs")
    require(run_contract["quality_metrics_executed"] is False, "calibration run executed quality metrics")
    require(run_contract["calibration_scope"] == CALIBRATION_SCOPE, "calibration run scope differs")
    require(run_contract["source_contracts"] == expected_contract_sources, "calibration run source hashes differ")
    require(run_contract["model"]["repository"] == MODEL["repository"], "calibration run model repository differs")
    require(run_contract["model"]["revision"] == MODEL["revision"], "calibration run model revision differs")
    require(run_contract["model"]["tokenizer_requested_revision"] == MODEL["revision"], "calibration tokenizer revision differs")
    require(run_contract["public_input_slice"] == {
        "record_limits": {"c4_calibration": 64},
        "token_limit": 512,
        "observations": CALIBRATION_OBSERVATION,
    }, "calibration run observed slice differs")
    require(
        run_contract["command"][1:] == [
            "tools/calibrate_qk_residual_scale32_source.py",
            "--output-dir",
            CALIBRATION_DIR.relative_to(ROOT).as_posix(),
        ],
        "calibration command differs from the frozen runner invocation",
    )

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    require(results["status"] == "pass", "calibration result did not pass")
    require(results["classification"] == "calibration_only_no_quality_claim", "calibration result classification differs")
    require(results["mode"] == "calibration_only", "calibration result mode differs")
    require(results["contract_id"] == CONTRACT, "calibration result contract differs")
    require(results["quality_metrics_executed"] is False, "calibration result claims quality metrics")
    require(results["calibration_scope"] == CALIBRATION_SCOPE, "calibration result scope differs")
    require(results["input_observations"] == CALIBRATION_OBSERVATION, "calibration result observations differ")
    require(results["source_contracts"] == expected_contract_sources, "calibration result source hashes differ")
    require(results["model"] == {**MODEL, "resolved_revision": MODEL["revision"]}, "calibration resolved model revision differs")
    require(results["artifacts"] == {
        "derived_scales": artifact(SOURCE, SOURCE_SHA256),
        "input_observations": artifact(INPUT_OBSERVATIONS, INPUT_OBSERVATIONS_SHA256),
        "run_contract": artifact(RUN_CONTRACT, RUN_CONTRACT_SHA256),
    }, "calibration result artifact bindings differ")
    require(
        results["integrity"]["canonical_sha256"] == canonical_sha256(results),
        "calibration result canonical hash differs",
    )

    source_plain = json.loads(SOURCE.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"), parse_float=Decimal)
    require(source_plain["schema_version"] == 3, "derived-scale source schema differs")
    require(source_plain["contract_id"] == CONTRACT, "derived-scale source contract differs")
    require(source_plain["calibration_scope"] == CALIBRATION_SCOPE, "derived-scale source scope differs")
    require(source_plain["input_observations"] == CALIBRATION_OBSERVATION, "derived-scale source observations differ")
    require(source_plain["provenance"] == {
        "run_contract": artifact(RUN_CONTRACT, RUN_CONTRACT_SHA256),
        "quality_metrics_executed": False,
    }, "derived-scale source provenance differs")
    require(len(source_plain["attention"]) == 24, "derived-scale source does not contain 24 layers")
    for layer in range(24):
        attention = source_plain["attention"][f"model.layers.{layer}.self_attn"]
        for projection, expected_heads in (("query", 14), ("key", 2)):
            absmax = attention[f"{projection}_projection_output_absmax"]
            scales = attention[f"{projection}_projection_output_scales"]
            require(len(absmax) == expected_heads, f"layer {layer} {projection} absmax head count differs")
            require(len(scales) == expected_heads, f"layer {layer} {projection} scale head count differs")
            require(
                all(scale == value / 127.0 for value, scale in zip(absmax, scales, strict=True)),
                f"layer {layer} {projection} scales do not equal observed BF16 absmax divided by 127",
            )
    return source, run_contract, results


def build() -> dict[str, Any]:
    source, run_contract, results = validate_source_provenance()
    records: list[dict[str, Any]] = []
    for layer in range(24):
        attention = source["attention"][f"model.layers.{layer}.self_attn"]
        for projection, scales in (
            ("q_proj", attention["query_projection_output_scales"]),
            ("k_proj", attention["key_projection_output_scales"]),
        ):
            expected_heads = 14 if projection == "q_proj" else 2
            if len(scales) != expected_heads:
                raise RuntimeError(f"layer {layer} {projection} head count differs")
            for head, baseline_source in enumerate(scales):
                baseline, baseline_sig, baseline_exp = scale32_record(baseline_source)
                residual, residual_sig, residual_exp = residual_record(baseline)
                records.append({
                    "layer": layer,
                    "projection": projection,
                    "head": head,
                    "baseline_source_decimal": str(baseline_source),
                    "baseline_scale32": {
                        "packed_u32": baseline,
                        "packed_hex": f"0x{baseline:08x}",
                        "significand_u16": baseline_sig,
                        "exponent_s8": baseline_exp,
                    },
                    "residual_scale32": {
                        "packed_u32": residual,
                        "packed_hex": f"0x{residual:08x}",
                        "significand_u16": residual_sig,
                        "exponent_s8": residual_exp,
                    },
                })
    if len(records) != 24 * (14 + 2):
        raise RuntimeError("metadata table does not contain 384 layer/head rows")
    table: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "model": {
            **MODEL,
            "layers": 24,
            "query_heads_per_layer": 14,
            "kv_heads_per_layer": 2,
        },
        "calibration_scope": {
            **CALIBRATION_SCOPE,
            "baseline_scale_rule": "per_layer_per_q_or_k_head_maximum_absolute_observed_bf16_projection_output_divided_by_127",
            "additional_residual_fit_prompts": "none",
        },
        "calibration_provenance": {
            "derived_scales": artifact(SOURCE, SOURCE_SHA256),
            "input_observations": artifact(INPUT_OBSERVATIONS, INPUT_OBSERVATIONS_SHA256),
            "run_contract": artifact(RUN_CONTRACT, RUN_CONTRACT_SHA256),
            "results": artifact(RESULTS, RESULTS_SHA256),
            "observed_input": CALIBRATION_OBSERVATION,
            "source_contracts": run_contract["source_contracts"],
            "result_integrity_sha256": results["integrity"]["canonical_sha256"],
            "quality_metrics_executed": False,
        },
        "derivation": {
            "baseline_scale32": "smallest_normalized_legal_Scale32_greater_than_or_equal_to_the_frozen_baseline_head_scale",
            "residual_scale32": "smallest_normalized_legal_Scale32_greater_than_or_equal_to_baseline_Scale32_divided_by_14",
            "clamp_objective": "maximize_signed4_resolution_while_guaranteeing_every_unsaturated_ties_to_even_projection_remainder_maps_without_clamp; saturated_baseline_outputs_may_clamp_and_are_counted",
            "tie_rule": "Scale32 ceiling scans exponent_minus24_through_plus4_then_uses_the_smallest_integer_significand; runtime residual division uses signed_round_to_nearest_ties_to_even",
            "baseline_preservation": "Scale32 records are immutable sideband metadata and never replace or recompute the accepted per-output multiplier_shift_zero_point or baseline q8_RoPE_base_score operations",
            "row_order": "layer_ascending_then_q_proj_heads_0_to_13_then_k_proj_heads_0_to_1",
        },
        "source_artifacts": [
            artifact(SOURCE, SOURCE_SHA256),
            artifact(INPUT_OBSERVATIONS, INPUT_OBSERVATIONS_SHA256),
            artifact(RUN_CONTRACT, RUN_CONTRACT_SHA256),
            artifact(RESULTS, RESULTS_SHA256),
            artifact(QUALITY_CONFIG, QUALITY_CONFIG_SHA256),
            artifact(PROMPT_MANIFEST, PROMPT_MANIFEST_SHA256),
            artifact(CALIBRATION_RUNNER, CALIBRATION_RUNNER_SHA256),
            artifact(FULL_MODEL_RUNNER, FULL_MODEL_RUNNER_SHA256),
            artifact(GENERATOR),
        ],
        "regeneration_command": ".venv/bin/python tools/gen_qk_residual_scale32_metadata.py --output reference/generated/qk_residual_scale32_metadata.json",
        "records": records,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    table["integrity"]["canonical_sha256"] = canonical_sha256(table)
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    table = build()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rendered = json.dumps(table, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("generated residual Scale32 metadata table is stale")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(f"ACE2_QK_RESIDUAL_SCALE32_METADATA_PASS rows={len(table['records'])} hash={table['integrity']['canonical_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
