#!/usr/bin/env python3
"""Test the narrow layer-17 per-token dynamic Q-output A8 scale repair."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_checkpoint176_layer17_mechanism as mechanism
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


STEPS = 4
LAYER = mechanism.LAYER


class DynamicQScaleGuard:
    """Re-derive only layer-17 Q projection metadata at every token position."""

    def __init__(self) -> None:
        self.original = backend.derive_layer_token
        self.records: dict[int, dict[str, torch.Tensor | float | int]] = {}

    def install(self) -> None:
        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            if layer_id == LAYER and template is not None:
                template["qkv"].pop("q", None)
            derived, next_state, next_template = self.original(
                layer_id, state, cache, template, weights, adapter
            )
            if layer_id == LAYER:
                position = derived["positions"][0]
                q = position["projections"]["q"]
                self.records[int(state["position"])] = {
                    "position": int(state["position"]),
                    "input_q": q["input_q"].detach().cpu().contiguous(),
                    "input_scale": float(q["input_scale"]),
                    "qweight": q["qweight"].detach().cpu().contiguous(),
                    "weight_scale": q["weight_scale"].detach().cpu().contiguous(),
                    "multiplier": q["multiplier"].detach().cpu().contiguous(),
                    "right_shift": q["right_shift"].detach().cpu().contiguous(),
                    "accumulator": q["accumulator"].detach().cpu().contiguous(),
                    "rounded": q["rounded"].detach().cpu().contiguous(),
                    "output_q": q["output_q"].detach().cpu().contiguous(),
                    "output_scale": float(q["output_scale"]),
                    "saturation": q["saturation"].detach().cpu().contiguous(),
                }
            return derived, next_state, next_template

        backend.derive_layer_token = derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original


def score_fixed(
    state: dict[str, Any],
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
    reference_token: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    item = backend.derive_final_rmsnorm_case(state, norm_gain)
    float_chunks = []
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        float_chunks.append(
            torch.mv(embedding[start:stop].to(torch.float32), item["float_norm"])
        )
    float_output = torch.cat(float_chunks).contiguous()
    output_scale = backend.canonical.scale_for(float_output)
    activation = item["final_q"].to(torch.int32)
    accumulator = torch.empty((backend.MODEL_OUTPUT_DOMAIN,), dtype=torch.int64)
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        accumulator[start:stop] = (
            head["qweight"][start:stop].to(torch.int32) * activation
        ).sum(dim=1, dtype=torch.int64)
    multiplier, right_shift = backend.canonical.derive_multiplier(
        float(item["final_scale"]) * head["weight_scale"] / output_scale
    )
    rounded = localizer.round_shift_even_tensor(accumulator * multiplier, right_shift)
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
    real_score = (
        accumulator.to(torch.float64)
        * float(item["final_scale"])
        * head["weight_scale"]
    )
    top = int(torch.argmax(output_q))
    real_top = int(torch.argmax(real_score))
    return (
        {
            "reference_token_id": reference_token,
            "reference_piece": localizer.decode_piece(tokenizer, reference_token),
            "fixed_top_token_id": top,
            "fixed_top_piece": localizer.decode_piece(tokenizer, top),
            "real_w4_top_token_id": real_top,
            "real_w4_top_piece": localizer.decode_piece(tokenizer, real_top),
            "reference_rank_fixed": localizer.rank_of(output_q, reference_token),
            "reference_rank_real_w4": localizer.rank_of(real_score, reference_token),
            "output_scale": float(output_scale),
            "final_rmsnorm_scale": float(item["final_scale"]),
            "lm_head_saturation_count": int(saturation.sum()),
            "fixed_top_tie_count": int((output_q == output_q.max()).sum()),
            "fixed_top_candidates": localizer.stable_top(output_q, tokenizer),
            "real_w4_top_candidates": localizer.stable_top(real_score, tokenizer),
        },
        {
            "final_q": item["final_q"],
            "accumulator": accumulator,
            "multiplier": multiplier,
            "right_shift": right_shift,
            "rounded": rounded,
            "output_q": output_q,
            "saturation": saturation,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.resolve()
    output_path = candidate / "dynamic-q-repair.json"
    localizer.require(candidate.is_dir(), "candidate absent")
    localizer.require(not output_path.exists(), "dynamic Q repair result exists")
    activation_path = candidate / "activation-localization.json"
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    localizer.require(
        activation["earliest_recovering_boundary"] == "q_projection_output",
        "activation boundary differs",
    )
    predecessor_result_path = (
        localizer.PREDECESSOR / "rtl-repair-0001/result.json"
    )
    predecessor = json.loads(predecessor_result_path.read_text(encoding="utf-8"))
    reference_tokens = [
        int(value) for value in predecessor["reference"]["generated_token_ids"]
    ]
    localizer.require(len(reference_tokens) == STEPS, "reference token span differs")
    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )

    with safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        projection_cache = localizer.FastProjectionCache(weights, adapter)
        dynamic_q = DynamicQScaleGuard()
        projection_cache.install()
        dynamic_q.install()
        caches = [backend.empty_layer_cache() for _ in range(backend.LAYERS)]
        templates: list[dict[str, Any] | None] = [None] * backend.LAYERS
        generated: list[int] = []
        steps = []
        try:
            total_positions = len(localizer.TOKEN_IDS) + STEPS - 1
            for position in range(total_positions):
                token_id = (
                    localizer.TOKEN_IDS[position]
                    if position < len(localizer.TOKEN_IDS)
                    else generated[position - len(localizer.TOKEN_IDS)]
                )
                state = backend.embedding_state(weights, token_id, position)
                for layer_id in range(backend.LAYERS):
                    derived, state, templates[layer_id] = backend.derive_layer_token(
                        layer_id,
                        state,
                        caches[layer_id],
                        templates[layer_id],
                        weights,
                        adapter,
                    )
                    del derived
                if position >= len(localizer.TOKEN_IDS) - 1:
                    step = position - (len(localizer.TOKEN_IDS) - 1)
                    record, tensors = score_fixed(
                        state,
                        norm_gain,
                        embedding,
                        head,
                        tokenizer,
                        reference_tokens[step],
                    )
                    selected = int(record["fixed_top_token_id"])
                    record.update(
                        {
                            "step": step,
                            "absolute_position": position,
                            "prefix_matches_reference_before_step": generated
                            == reference_tokens[:step],
                            "matches_reference": selected == reference_tokens[step],
                        }
                    )
                    tensor_records = {
                        name: localizer.write_tensor(
                            candidate
                            / f"cpu/dynamic-q/step-{step:02d}/lm-head-{name}.bin",
                            tensor,
                        )
                        for name, tensor in tensors.items()
                    }
                    q = dynamic_q.records[position]
                    q_records = {
                        name: localizer.write_tensor(
                            candidate
                            / f"cpu/dynamic-q/step-{step:02d}/layer17-q-{name}.bin",
                            value,
                        )
                        for name, value in q.items()
                        if isinstance(value, torch.Tensor)
                    }
                    record["artifacts"] = {
                        "lm_head": tensor_records,
                        "layer17_q": q_records,
                    }
                    record["layer17_q"] = {
                        "input_scale": float(q["input_scale"]),
                        "output_scale": float(q["output_scale"]),
                        "saturation_count": int(
                            q["saturation"].to(torch.int64).sum()
                        ),
                        "multiplier_min": int(q["multiplier"].min()),
                        "multiplier_max": int(q["multiplier"].max()),
                        "right_shift_min": int(q["right_shift"].min()),
                        "right_shift_max": int(q["right_shift"].max()),
                    }
                    steps.append(record)
                    generated.append(selected)
                    print(
                        "ACE2_DYNAMIC_Q_STEP "
                        f"step={step} position={position} selected={selected} "
                        f"reference={reference_tokens[step]} match={record['matches_reference']} "
                        f"q_scale={float(q['output_scale']):.12g}",
                        flush=True,
                    )
        finally:
            dynamic_q.restore()
            projection_cache.restore()

    matching_prefix = 0
    for row in steps:
        if not (row["prefix_matches_reference_before_step"] and row["matches_reference"]):
            break
        matching_prefix += 1
    result = {
        "schema_version": 1,
        "status": (
            "PASS_DYNAMIC_Q_SCALE_FOUR_TOKEN_CPU"
            if matching_prefix == STEPS
            else "PARTIAL_DYNAMIC_Q_SCALE_CPU"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "repair": {
            "boundary": "model.layers.17.self_attn.q_proj.output_a8_scale",
            "mechanism": (
                "derive signed-A8 Q projection output scale independently for every "
                "causal token position instead of reusing position-0 metadata"
            ),
            "weights_changed": False,
            "activation_width_changed": False,
            "accumulator_width_changed": False,
            "rounding": "round_to_nearest_even",
            "saturation": "signed_int8_clamp",
            "token_or_logit_override": False,
        },
        "reference_token_ids": reference_tokens,
        "generated_token_ids": generated,
        "decoded_text": tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "matching_greedy_prefix_tokens": matching_prefix,
        "steps": steps,
        "bindings": {
            "runner": localizer.file_record(Path(__file__).resolve()),
            "activation_localization": localizer.file_record(activation_path),
            "predecessor_result": localizer.file_record(predecessor_result_path),
        },
        "next_required_action": (
            "Generate hash-bound four-position Q-projection and rank vectors, then run focused RTL reset/stall/X/round/saturation checks."
            if matching_prefix == STEPS
            else "Preserve the first failing aligned step and localize the next dynamic activation boundary."
        ),
        "official_run_launched": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    localizer.write_json(output_path, result)
    localizer.write_sums(candidate)
    print(
        "ACE2_DYNAMIC_Q_RESULT "
        f"status={result['status']} matching_prefix={matching_prefix} "
        f"generated={generated}",
        flush=True,
    )
    return 0 if matching_prefix == STEPS else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_DYNAMIC_Q_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
