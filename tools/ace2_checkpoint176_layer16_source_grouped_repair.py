#!/usr/bin/env python3
"""Evaluate exact dual-source grouped-A8 preservation at layer 16."""

from __future__ import annotations

import argparse
import json
import os
import re
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
from tools import ace2_checkpoint176_layer17_grouped_q_probe as grouped_q
from tools import ace2_checkpoint176_layer17_mechanism as mechanism
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


SOURCE_LAYER = 16
TARGET_LAYER = mechanism.LAYER
SOURCE_GROUP_SIZES = (32, 16, 8, 4, 1)
Q_PATTERN = re.compile(r"layer17_position(\d+)_q$")
PREDECESSOR = (
    ROOT / "evidence/candidates/w4a8-chat-hidden-state-localization-v1/candidate-0002"
)


def grouped_reconstruct_scale32(
    value: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize contiguous groups to signed A8 with explicit binary32 scales."""
    localizer.require(value.numel() % group_size == 0, "source group does not divide hidden")
    reconstructed = torch.empty_like(value, dtype=torch.float64)
    quantized = torch.empty(value.numel(), dtype=torch.int8)
    scales = torch.empty(value.numel() // group_size, dtype=torch.float32)
    for group_id, start in enumerate(range(0, value.numel(), group_size)):
        stop = start + group_size
        group = value[start:stop].to(torch.float64)
        scale = torch.tensor(
            max(float(group.abs().max()) / 127.0, 1.0e-12), dtype=torch.float32
        )
        quantized[start:stop] = torch.round(group / float(scale)).clamp(-128, 127).to(
            torch.int8
        )
        reconstructed[start:stop] = quantized[start:stop].to(torch.float64) * float(scale)
        scales[group_id] = scale
    return reconstructed, quantized, scales


class TensorwideQProjectionCache(grouped_q.GroupedQProjectionCache):
    """Keep the existing one-scale A8 RMSNorm-to-Q input boundary."""

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        match = Q_PATTERN.fullmatch(name)
        if match is not None:
            position = int(match.group(1))
            float_input = self.group_inputs[position].to(torch.float32)
        return super().derive_projection(
            name, merged, input_q, input_scale, float_input, source_hashes
        )


class SourceGroupedGuard:
    def __init__(
        self,
        cache: TensorwideQProjectionCache,
        weights: Any,
        source_group_size: int,
    ) -> None:
        self.cache = cache
        self.weights = weights
        self.source_group_size = source_group_size
        self.original = backend.derive_layer_token
        self.records: dict[int, dict[str, torch.Tensor]] = {}

    def install(self) -> None:
        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            if layer_id == TARGET_LAYER:
                position_id = int(state["position"])
                source = self.records[position_id]
                gain = self.weights.get_tensor(
                    f"model.layers.{TARGET_LAYER}.input_layernorm.weight"
                ).contiguous()
                input_norm = backend.canonical.float_rmsnorm(
                    source["sum_reconstructed"].to(torch.float32), gain
                ).contiguous()
                source["input_norm"] = input_norm
                self.cache.group_inputs[position_id] = input_norm
                if template is not None:
                    template["qkv"].pop("q", None)

            derived, next_state, next_template = self.original(
                layer_id, state, cache, template, weights, adapter
            )

            if layer_id == SOURCE_LAYER:
                position = derived["positions"][0]
                attention_source = (
                    position["attention_residual"]["output"].to(torch.float64)
                    * float(position["attention_residual"]["scale"])
                ).contiguous()
                down_source = (
                    position["projections"]["down"]["output_q"].to(torch.float64)
                    * float(position["projections"]["down"]["output_scale"])
                ).contiguous()
                attention_reconstructed, attention_q, attention_scales = (
                    grouped_reconstruct_scale32(attention_source, self.source_group_size)
                )
                down_reconstructed, down_q, down_scales = grouped_reconstruct_scale32(
                    down_source, self.source_group_size
                )
                self.records[int(state["position"])] = {
                    "attention_source": attention_source,
                    "attention_q": attention_q,
                    "attention_scales": attention_scales,
                    "attention_reconstructed": attention_reconstructed,
                    "down_source": down_source,
                    "down_q": down_q,
                    "down_scales": down_scales,
                    "down_reconstructed": down_reconstructed,
                    "sum_reconstructed": (
                        attention_reconstructed + down_reconstructed
                    ).contiguous(),
                }
            return derived, next_state, next_template

        backend.derive_layer_token = derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original


def run_group(
    source_group_size: int,
    cache: TensorwideQProjectionCache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure(backend.HIDDEN)
    guard = SourceGroupedGuard(cache, weights, source_group_size)
    cache.install()
    guard.install()
    try:
        result, _ = localizer.run_cut(
            None,
            [],
            weights,
            adapter,
            norm_gain,
            embedding,
            head,
            tokenizer,
        )
    finally:
        guard.restore()
        cache.restore()

    position = len(localizer.TOKEN_IDS) - 1
    source = guard.records[position]
    q = cache.group_records[position]
    result.update(
        {
            "source_group_size": source_group_size,
            "source_group_count_per_stream": backend.HIDDEN // source_group_size,
            "q_input_group_size": backend.HIDDEN,
            "attention_scale_min": float(source["attention_scales"].min()),
            "attention_scale_max": float(source["attention_scales"].max()),
            "down_scale_min": float(source["down_scales"].min()),
            "down_scale_max": float(source["down_scales"].max()),
            "q_input_scale": float(q["input_scales"][0]),
            "q_output_scale": float(q["output_scale"]),
            "q_saturation_count": int(q["saturation"].to(torch.int64).sum()),
            "sum_reconstructed_sha256": localizer.tensor_sha256(
                source["sum_reconstructed"]
            ),
            "q_output_sha256": localizer.tensor_sha256(q["output_q"]),
        }
    )
    artifacts = {
        **source,
        "q_input_q": q["input_q"],
        "q_input_scales": q["input_scales"],
        "q_output_q": q["output_q"],
        "q_saturation": q["saturation"],
    }
    return result, artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "candidate output already exists")

    predecessor_freeze_path = PREDECESSOR / "candidate-freeze.json"
    predecessor_scan_path = PREDECESSOR / "layer-scan.json"
    predecessor_freeze = json.loads(predecessor_freeze_path.read_text(encoding="utf-8"))
    predecessor_scan = json.loads(predecessor_scan_path.read_text(encoding="utf-8"))
    prompt = predecessor_freeze["prompt"]
    localizer.require(prompt["chat_token_count"] == 30, "canonical prefix count differs")
    localizer.require(prompt["chat_token_ids"] == localizer.TOKEN_IDS, "canonical prefix differs")
    localizer.require(
        predecessor_freeze["checkpoint"]["name"] == "checkpoint-176",
        "checkpoint differs",
    )
    localizer.require(
        predecessor_freeze["reference_step0_token_id"] == localizer.REFERENCE_TOKEN,
        "reference token differs",
    )
    baseline = predecessor_scan["baseline_reproduction"]["observed"]
    baseline_rank = int(baseline["reference_rank"])

    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "mission": "w4a8-layer16-source-grouped-activation-repair",
        "classification": "candidate_only_no_official_attempt_created",
        "checkpoint": predecessor_freeze["checkpoint"],
        "prompt": prompt,
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "baseline": {
            "top_token_id": int(baseline["top_token_id"]),
            "reference_rank": baseline_rank,
        },
        "ordered_source_group_sizes": list(SOURCE_GROUP_SIZES),
        "score_policy": (
            "first restored group in declared execution order; otherwise lowest reference "
            "rank strictly better than baseline, ties by declared execution order"
        ),
        "hardware_contract": {
            "attention_residual_samples": "signed A8",
            "down_projection_samples": "signed A8",
            "source_scales": "one explicit binary32 model-derived scale per source group",
            "sum": "dequantized aligned group sum before layer17 RMSNorm",
            "rmsnorm_to_q_input": "existing tensor-wide signed A8 scale",
            "weights": "unchanged signed W4 per output row",
            "bf16_runtime_sidecar": False,
            "token_or_logit_override": False,
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "u280_or_stage2_entered": False,
        },
        "bindings": {
            "predecessor_freeze": localizer.file_record(predecessor_freeze_path),
            "predecessor_scan": localizer.file_record(predecessor_scan_path),
            "runner": localizer.file_record(Path(__file__).resolve()),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)

    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )

    trace = []
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = TensorwideQProjectionCache(weights, adapter)
        for group_size in SOURCE_GROUP_SIZES:
            record, tensors = run_group(
                group_size,
                cache,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            group_dir = output / f"groups/group-{group_size:03d}"
            artifacts = {
                name: localizer.write_tensor(group_dir / f"{name}.bin", value)
                for name, value in sorted(tensors.items())
            }
            group_result = {"metrics": record, "artifacts": artifacts}
            localizer.write_json(group_dir / "result.json", group_result)
            trace.append(record)
            print(
                "ACE2_LAYER16_SOURCE_GROUPED_CANDIDATE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} sat={record['q_saturation_count']}",
                flush=True,
            )

    restored = [item for item in trace if item["top_token_id"] == localizer.REFERENCE_TOKEN]
    improved = [item for item in trace if int(item["reference_rank"]) < baseline_rank]
    if restored:
        selected = restored[0]
        selection_reason = "restored_step0_in_declared_execution_order"
    elif improved:
        order = {group: index for index, group in enumerate(SOURCE_GROUP_SIZES)}
        selected = min(
            improved,
            key=lambda item: (
                int(item["reference_rank"]),
                order[int(item["source_group_size"])],
            ),
        )
        selection_reason = "best_honest_reference_rank_improvement"
    else:
        selected = None
        selection_reason = "no_group_improved_baseline_rank"

    restored_step0 = selected is not None and selected["top_token_id"] == localizer.REFERENCE_TOKEN
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER16_SOURCE_GROUPED_STEP0_RESTORED"
            if restored_step0
            else "PARTIAL_LAYER16_SOURCE_GROUPED_RANK_IMPROVED"
            if selected is not None
            else "PARTIAL_LAYER16_SOURCE_GROUPED_NO_IMPROVEMENT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "baseline_reference_rank": baseline_rank,
        "ordered_source_group_sizes": list(SOURCE_GROUP_SIZES),
        "trace": trace,
        "selected_source_group_size": (
            int(selected["source_group_size"]) if selected is not None else None
        ),
        "selected_reference_rank": (
            int(selected["reference_rank"]) if selected is not None else baseline_rank
        ),
        "selected_top_token_id": (
            int(selected["top_token_id"]) if selected is not None else int(baseline["top_token_id"])
        ),
        "selection_reason": selection_reason,
        "first_token_restored": bool(restored_step0),
        "next_exact_source_boundary": (
            None
            if restored_step0
            else "model.layers.17.input_rmsnorm_to_self_attn.q_proj.output"
        ),
        "next_required_action": (
            "Run frozen four-token greedy ordering, then focused quantize/sum/RMSNorm-to-Q RTL."
            if restored_step0
            else "Preserve the best honest rank and build focused RTL for the selected source group before advancing to the layer17 Q-output boundary."
            if selected is not None
            else "Preserve all negative groups and advance to the exact layer17 Q-output activation boundary."
        ),
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER16_SOURCE_GROUPED_RESULT "
        f"status={result['status']} selected={result['selected_source_group_size']} "
        f"rank={result['selected_reference_rank']} restored={result['first_token_restored']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER16_SOURCE_GROUPED_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
