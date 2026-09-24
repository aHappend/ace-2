#!/usr/bin/env python3
"""Localize paired BF16/W4A8 divergence through the layer-0 score tensor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from ace2_full_model_fixed_point import (
    ACTIVE_ROPE_MECHANISM,
    FixedAttention,
    FixedRMSNorm,
    PROMPT_MANIFEST,
    QUALITY_CONFIG,
    ROOT,
    calibrate,
    dynamic_rope_head_raw,
    fixed_dynamic_attention_scores_raw,
    hash_records,
    hash_token_sequences,
    load_contracts,
    repeat_kv,
    replace_fixed_operators,
    replace_linears,
    seed_everything,
    selected_texts,
    sha256_file,
    tokenize_prompts,
    tokenize_wikitext,
    utc_now,
    validate_runtime,
)
from localize_quality_divergence import compare_tensor, write_sha256s


DATASETS = ("wikitext2", "c4_en_512")
BOUNDARY_ORDER = (
    "model.layers.0.input_rmsnorm",
    "model.layers.0.q_projection",
    "model.layers.0.k_projection",
    "model.layers.0.v_projection",
    "model.layers.0.q_post_rope",
    "model.layers.0.k_post_rope",
    "model.layers.0.score",
)
MODEL_SOURCE = ROOT / "tools" / "ace2_full_model_fixed_point.py"
QUALITY_LOCALIZER_SOURCE = ROOT / "tools" / "localize_quality_divergence.py"


class _Layer0InputsCaptured(RuntimeError):
    pass


def artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def current_source_binding(paths: list[Path]) -> dict[str, Any]:
    sources = [artifact(path.resolve()) for path in paths]
    digest, count = hash_records(
        f"{item['path']}\0{item['bytes']}\0{item['sha256']}" for item in sources
    )
    return {
        "ordered_source_hash_list_sha256": digest,
        "source_count": count,
        "sources": sources,
        "status": "bound_to_current_reference_sources",
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def capture_layer0_inputs(
    model: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor | None]:
    captured: dict[str, Any] = {}
    layer = model.model.layers[0]

    def capture_attention(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        position_embeddings = kwargs.get("position_embeddings")
        if not isinstance(hidden_states, Tensor):
            raise RuntimeError("layer-0 attention did not receive hidden states")
        if (
            not isinstance(position_embeddings, tuple)
            or len(position_embeddings) != 2
            or not all(isinstance(value, Tensor) for value in position_embeddings)
        ):
            raise RuntimeError("layer-0 attention did not receive rotary embeddings")
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and not isinstance(attention_mask, Tensor):
            raise RuntimeError("layer-0 attention mask is not a tensor")
        captured["hidden_states"] = hidden_states.detach()
        captured["position_embeddings"] = tuple(
            value.detach() for value in position_embeddings
        )
        captured["attention_mask"] = (
            None if attention_mask is None else attention_mask.detach()
        )
        raise _Layer0InputsCaptured

    hook = layer.self_attn.register_forward_pre_hook(
        capture_attention,
        with_kwargs=True,
    )
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    except _Layer0InputsCaptured:
        pass
    finally:
        hook.remove()
    if set(captured) != {"hidden_states", "position_embeddings", "attention_mask"}:
        raise RuntimeError(f"incomplete layer-0 capture: {sorted(captured)}")
    return (
        captured["hidden_states"],
        captured["position_embeddings"],
        captured["attention_mask"],
    )


def valid_score_mask(score: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is not None:
        return (attention_mask[:, :, :, : score.shape[-1]] >= 0).expand_as(score)
    sequence = score.shape[-1]
    return (
        torch.ones(sequence, sequence, dtype=torch.bool, device=score.device)
        .tril()
        .view(1, 1, sequence, sequence)
        .expand_as(score)
    )


@torch.inference_mode()
def baseline_boundaries(
    attention: nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
) -> dict[str, Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_rope, key_rope = apply_rotary_pos_emb(
        query,
        key,
        *position_embeddings,
    )
    repeated_key = repeat_kv(key_rope, attention.num_key_value_groups)
    score = torch.matmul(query_rope, repeated_key.transpose(2, 3)) * attention.scaling
    valid = valid_score_mask(score, attention_mask)
    row_max = torch.where(
        valid,
        score,
        torch.full_like(score, -torch.inf),
    ).amax(dim=-1, keepdim=True)
    centered_score = score - row_max
    return {
        "model.layers.0.input_rmsnorm": hidden_states,
        "model.layers.0.q_projection": query,
        "model.layers.0.k_projection": key,
        "model.layers.0.v_projection": value,
        "model.layers.0.q_post_rope": query_rope,
        "model.layers.0.k_post_rope": key_rope,
        "model.layers.0.score": centered_score[valid],
    }


@torch.inference_mode()
def fixed_boundaries(
    attention: FixedAttention,
    norm: FixedRMSNorm,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Any],
]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query = (
        attention.q_proj.forward_hardware_input(hidden_states)
        .view(hidden_shape)
        .transpose(1, 2)
    )
    key = (
        attention.k_proj.forward_hardware_input(hidden_states)
        .view(hidden_shape)
        .transpose(1, 2)
    )
    value = (
        attention.v_proj.forward_hardware_input(hidden_states)
        .view(hidden_shape)
        .transpose(1, 2)
    )
    cos, sin = position_embeddings
    if not attention.dynamic_rope_head_scale:
        raise TypeError("focused localizer requires the active dynamic RoPE path")
    query_rope, query_scale32, query_saturations = dynamic_rope_head_raw(
        query,
        attention.query_producer_scale32,
        cos,
        sin,
    )
    key_rope, key_scale32, key_saturations = dynamic_rope_head_raw(
        key,
        attention.key_producer_scale32,
        cos,
        sin,
    )
    repeated_key = repeat_kv(key_rope, attention.num_key_value_groups)
    repeated_key_scale32 = key_scale32.repeat_interleave(
        attention.num_key_value_groups,
        dim=1,
    )
    score = fixed_dynamic_attention_scores_raw(
        query_rope,
        repeated_key,
        query_scale32,
        repeated_key_scale32,
        attention_mask,
    )
    valid = valid_score_mask(score, attention_mask)
    query_rope_scale = scale32_values(query_scale32).unsqueeze(-1).expand_as(
        query_rope
    )
    key_rope_scale = scale32_values(key_scale32).unsqueeze(-1).expand_as(key_rope)
    scale_boundaries = {
        "model.layers.0.input_rmsnorm": torch.full_like(
            hidden_states,
            norm.output_scale,
            dtype=torch.float64,
        ),
        "model.layers.0.q_projection": torch.full_like(
            query,
            attention.q_proj.output_scale,
            dtype=torch.float64,
        ),
        "model.layers.0.k_projection": torch.full_like(
            key,
            attention.k_proj.output_scale,
            dtype=torch.float64,
        ),
        "model.layers.0.v_projection": torch.full_like(
            value,
            attention.v_proj.output_scale,
            dtype=torch.float64,
        ),
        "model.layers.0.q_post_rope": query_rope_scale,
        "model.layers.0.k_post_rope": key_rope_scale,
        "model.layers.0.score": torch.full_like(
            score[valid],
            1.0 / float(1 << 9),
            dtype=torch.float64,
        ),
    }
    boundaries = {
        "model.layers.0.input_rmsnorm": (
            hidden_states.to(torch.float64) * norm.output_scale
        ),
        "model.layers.0.q_projection": (
            query.to(torch.float64) * attention.q_proj.output_scale
        ),
        "model.layers.0.k_projection": (
            key.to(torch.float64) * attention.k_proj.output_scale
        ),
        "model.layers.0.v_projection": (
            value.to(torch.float64) * attention.v_proj.output_scale
        ),
        "model.layers.0.q_post_rope": (
            query_rope.to(torch.float64) * query_rope_scale
        ),
        "model.layers.0.k_post_rope": (
            key_rope.to(torch.float64) * key_rope_scale
        ),
        "model.layers.0.score": score[valid].to(torch.float64) / float(1 << 9),
    }
    raw_boundaries = {
        "model.layers.0.input_rmsnorm": hidden_states,
        "model.layers.0.q_projection": query,
        "model.layers.0.k_projection": key,
        "model.layers.0.v_projection": value,
        "model.layers.0.q_post_rope": query_rope,
        "model.layers.0.k_post_rope": key_rope,
        "model.layers.0.score": score[valid],
    }
    metadata = {
        "input_rmsnorm_output_scale": norm.output_scale,
        "q_projection_output_scale": attention.q_proj.output_scale,
        "k_projection_output_scale": attention.k_proj.output_scale,
        "v_projection_output_scale": attention.v_proj.output_scale,
        "query_rope_scale32_max": int(query_scale32.max()),
        "query_rope_scale32_min": int(query_scale32.min()),
        "key_rope_scale32_max": int(key_scale32.max()),
        "key_rope_scale32_min": int(key_scale32.min()),
        "query_rope_saturated_elements": query_saturations,
        "key_rope_saturated_elements": key_saturations,
        "boundary_provenance": {
            "model.layers.0.input_rmsnorm": {
                "rounding": "nearest_ties_to_even",
                "saturation_range": [-128, 127],
                "scale": norm.output_scale,
                "storage": "signed_int8_in_float_container",
            },
            "model.layers.0.q_projection": {
                "rounding": "nearest_ties_to_even",
                "saturation_range": [-128, 127],
                "scale": attention.q_proj.output_scale,
                "storage": "signed_int8",
            },
            "model.layers.0.k_projection": {
                "rounding": "nearest_ties_to_even",
                "saturation_range": [-128, 127],
                "scale": attention.k_proj.output_scale,
                "storage": "signed_int8",
            },
            "model.layers.0.v_projection": {
                "rounding": "nearest_ties_to_even",
                "saturation_range": [-128, 127],
                "scale": attention.v_proj.output_scale,
                "storage": "signed_int8",
            },
            "model.layers.0.q_post_rope": {
                "rounding": "nearest_ties_to_even",
                "saturated_elements": query_saturations,
                "saturation_range": [-128, 127],
                "scale": "dynamic_scale32_per_token_per_head",
                "storage": "signed_int8",
            },
            "model.layers.0.k_post_rope": {
                "rounding": "nearest_ties_to_even",
                "saturated_elements": key_saturations,
                "saturation_range": [-128, 127],
                "scale": "dynamic_scale32_per_token_per_head",
                "storage": "signed_int8",
            },
            "model.layers.0.score": {
                "rounding": "nearest_ties_to_even",
                "saturation_range": [-32768, 0],
                "scale": 1.0 / float(1 << 9),
                "storage": "signed_int16_centered_q6_9",
            },
        },
    }
    return boundaries, raw_boundaries, scale_boundaries, metadata


def scale32_values(records: Tensor) -> Tensor:
    significand = torch.bitwise_and(records, 0xFFFF).to(torch.float64)
    exponent_u8 = torch.bitwise_and(
        torch.bitwise_right_shift(records, 16),
        0xFF,
    )
    exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
    return torch.ldexp(significand, (exponent - 15).to(torch.int32))


def tensor_coordinate(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    coordinate: list[int] = []
    remainder = flat_index
    for dimension in reversed(shape):
        coordinate.append(remainder % dimension)
        remainder //= dimension
    if remainder:
        raise ValueError("flat tensor index exceeds shape")
    return list(reversed(coordinate))


def exact_difference(
    reference: Tensor,
    candidate: Tensor,
    candidate_raw: Tensor,
    candidate_scale: Tensor,
    provenance: dict[str, Any],
) -> dict[str, Any] | None:
    mismatch = torch.ne(reference, candidate).reshape(-1)
    indices = torch.nonzero(mismatch, as_tuple=False)
    if not indices.numel():
        return None
    flat_index = int(indices[0, 0])
    reference_value = float(reference.reshape(-1)[flat_index])
    candidate_value = float(candidate.reshape(-1)[flat_index])
    raw_item = candidate_raw.reshape(-1)[flat_index].item()
    raw_value: int | float = float(raw_item)
    if raw_value.is_integer():
        raw_value = int(raw_value)
    saturation_minimum, saturation_maximum = provenance["saturation_range"]
    return {
        "absolute_error": abs(candidate_value - reference_value),
        "actual_fixed": candidate_value,
        "candidate_at_storage_limit": raw_value
        in {saturation_minimum, saturation_maximum},
        "candidate_raw": raw_value,
        "coordinate": tensor_coordinate(flat_index, tuple(reference.shape)),
        "expected_reference": reference_value,
        "flat_index": flat_index,
        "rounding": provenance["rounding"],
        "saturated_elements": provenance.get("saturated_elements"),
        "saturation_range": provenance["saturation_range"],
        "scale": float(candidate_scale.reshape(-1)[flat_index]),
        "scale_provenance": provenance["scale"],
        "storage": provenance["storage"],
    }


def compare_boundaries(
    baseline: dict[str, Tensor],
    candidate: dict[str, Tensor],
    candidate_raw: dict[str, Tensor],
    candidate_scale: dict[str, Tensor],
    provenance: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str | None, str | None]:
    if tuple(baseline) != BOUNDARY_ORDER or tuple(candidate) != BOUNDARY_ORDER:
        raise RuntimeError("layer-0 boundary order differs from the frozen contract")
    if (
        tuple(candidate_raw) != BOUNDARY_ORDER
        or tuple(candidate_scale) != BOUNDARY_ORDER
        or tuple(provenance) != BOUNDARY_ORDER
    ):
        raise RuntimeError("layer-0 provenance order differs from the frozen contract")
    comparisons: dict[str, Any] = {}
    first_exact: str | None = None
    first_material: str | None = None
    for name in BOUNDARY_ORDER:
        reference_tensor = baseline[name].detach().to(device="cpu", dtype=torch.float64)
        candidate_tensor = candidate[name].detach().to(device="cpu", dtype=torch.float64)
        metrics = compare_tensor(
            reference_tensor,
            candidate_tensor,
        )
        metrics["first_difference"] = exact_difference(
            reference_tensor,
            candidate_tensor,
            candidate_raw[name].detach().to(device="cpu"),
            candidate_scale[name].detach().to(device="cpu", dtype=torch.float64),
            provenance[name],
        )
        comparisons[name] = metrics
        if first_exact is None and metrics["first_difference"] is not None:
            first_exact = name
        if first_material is None and (
            metrics["relative_l2_error"] >= 0.25
            or (
                metrics["cosine_similarity"] is not None
                and metrics["cosine_similarity"] <= 0.95
            )
        ):
            first_material = name
    return comparisons, first_exact, first_material


def observe_inputs(
    manifest: dict[str, Any],
    texts: dict[str, list[str]],
    prompts: dict[str, Tensor],
) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for dataset in ("c4_calibration", *DATASETS):
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


def run(
    output_dir: Path,
    candidate_evidence_path: Path | None,
    frozen_baseline_path: Path,
) -> dict[str, Any]:
    manifest, config, rtl_binding = load_contracts(
        require_rtl_binding=candidate_evidence_path is not None,
        candidate_evidence_path=candidate_evidence_path
    )
    versions = validate_runtime(config)
    seed_everything(config)
    output_dir.mkdir(parents=True, exist_ok=False)

    texts = {
        "c4_calibration": selected_texts(
            manifest["datasets"]["c4_calibration"], limit=1
        ),
        "c4_en_512": selected_texts(manifest["datasets"]["c4_en_512"], limit=1),
        "wikitext2": selected_texts(manifest["datasets"]["wikitext2"], limit=16),
    }
    model_spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
    )
    frozen_baseline = json.loads(frozen_baseline_path.read_text(encoding="utf-8"))
    prompts = {
        "c4_calibration": tokenize_prompts(
            tokenizer,
            texts["c4_calibration"],
            manifest["datasets"]["c4_calibration"]["token_limit"],
        )[0],
        "c4_en_512": tokenize_prompts(
            tokenizer,
            texts["c4_en_512"],
            manifest["datasets"]["c4_en_512"]["token_limit"],
        )[0],
        "wikitext2": tokenize_wikitext(
            tokenizer,
            texts["wikitext2"],
            manifest["datasets"]["wikitext2"]["token_limit"],
            manifest["datasets"]["wikitext2"]["join"],
        )[0],
    }
    for dataset, prompt in prompts.items():
        expected_tokens = frozen_baseline["input_observations"][dataset]["tokenized"][
            "token_count"
        ]
        prompts[dataset] = prompt[:, :expected_tokens]
    observations = observe_inputs(manifest, texts, prompts)
    if observations != frozen_baseline["input_observations"]:
        raise RuntimeError("focused localization did not reproduce the frozen paired input")

    source_paths = [
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        MODEL_SOURCE,
        QUALITY_LOCALIZER_SOURCE,
        Path(__file__),
    ]
    reference_source_binding = current_source_binding(source_paths)
    diagnostic_binding = (
        {
            "acceptance_claimed": True,
            "binding": rtl_binding["binding"],
            "candidate_id": rtl_binding["candidate_id"],
            "rtl_hash": rtl_binding["candidate_rtl_hash"],
            "source_hash_list": rtl_binding["source_hash_list"],
            "status": "candidate_evidence_hash_validated",
        }
        if candidate_evidence_path is not None
        else {
            "acceptance_claimed": False,
            "binding": None,
            "candidate_id": None,
            "rtl_hash": None,
            "source_hash_list": None,
            "status": "not_applicable_reference_only_diagnostic",
        }
    )
    run_contract = {
        "schema_version": 1,
        "status": "frozen_before_measurement",
        "created_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "scope": {
            "layer": 0,
            "boundary_order": list(BOUNDARY_ORDER),
            "datasets": list(DATASETS),
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
            "localizer_sha256": sha256_file(Path(__file__)),
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

    baseline: dict[str, dict[str, Tensor]] = {}
    for dataset in DATASETS:
        seed_everything(config)
        hidden_states, position_embeddings, attention_mask = capture_layer0_inputs(
            model,
            prompts[dataset],
        )
        baseline[dataset] = baseline_boundaries(
            model.model.layers[0].self_attn,
            hidden_states,
            position_embeddings,
            attention_mask,
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
        layer.self_attn, FixedAttention
    ):
        raise TypeError("layer-0 fixed operators were not installed")

    comparisons: dict[str, Any] = {}
    first_exact: dict[str, str | None] = {}
    first_material: dict[str, str | None] = {}
    fixed_metadata: dict[str, Any] = {}
    for dataset in DATASETS:
        seed_everything(config)
        hidden_states, position_embeddings, attention_mask = capture_layer0_inputs(
            model,
            prompts[dataset],
        )
        fixed, fixed_raw, fixed_scale, fixed_metadata[dataset] = fixed_boundaries(
            layer.self_attn,
            layer.input_layernorm,
            hidden_states,
            position_embeddings,
            attention_mask,
        )
        (
            comparisons[dataset],
            first_exact[dataset],
            first_material[dataset],
        ) = compare_boundaries(
            baseline[dataset],
            fixed,
            fixed_raw,
            fixed_scale,
            fixed_metadata[dataset]["boundary_provenance"],
        )

    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "diagnostic_focused_layer0_paired_localization_not_acceptance_evidence",
        "gate_passed": False,
        "scope": {
            "layer": 0,
            "boundary_order": list(BOUNDARY_ORDER),
            "datasets": list(DATASETS),
        },
        "material_divergence_policy": {
            "cosine_similarity_max": 0.95,
            "relative_l2_error_min": 0.25,
        },
        "first_exact_divergence": first_exact,
        "first_material_divergence": first_material,
        "comparisons": comparisons,
        "fixed_metadata": fixed_metadata,
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
            "accepted_rtl": (
                rtl_binding if candidate_evidence_path is not None else None
            ),
            "run_contract": artifact(run_contract_path),
            "sources": [artifact(path) for path in source_paths],
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
        "ACE2_FOCUSED_LAYER0_PAIRED_LOCALIZATION "
        f"wiki_exact={first_exact['wikitext2']} "
        f"wiki_first={first_material['wikitext2']} "
        f"c4_exact={first_exact['c4_en_512']} "
        f"c4_first={first_material['c4_en_512']} "
        f"rtl_hash={diagnostic_binding['rtl_hash']} "
        f"binding_status={diagnostic_binding['status']} "
        f"model_sha256={sha256_file(MODEL_SOURCE)} "
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
    parser.add_argument("--candidate-evidence", type=Path)
    parser.add_argument("--frozen-baseline", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if ROOT.resolve() not in args.output_dir.resolve().parents:
        raise SystemExit("--output-dir must be below the repository root")
    run(
        args.output_dir,
        (
            None
            if args.candidate_evidence is None
            else repository_file(args.candidate_evidence, "candidate-evidence")
        ),
        repository_file(args.frozen_baseline, "frozen-baseline"),
    )


if __name__ == "__main__":
    main()
