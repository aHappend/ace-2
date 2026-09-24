#!/usr/bin/env python3
"""Probe grouped-A8 layer-17 input plus grouped-A8 RMSNorm-to-Q."""

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
from tools import ace2_checkpoint176_layer17_grouped_q_probe as grouped
from tools import ace2_checkpoint176_layer17_mechanism as mechanism
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


LAYER = mechanism.LAYER


def grouped_reconstruct(value: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstructed = torch.empty_like(value, dtype=torch.float64)
    quantized = torch.empty(value.numel(), dtype=torch.int8)
    scales = []
    for start in range(0, value.numel(), group_size):
        stop = start + group_size
        group = value[start:stop].to(torch.float64)
        scale = max(float(group.abs().max()) / 127.0, 1e-12)
        q = torch.round(group / scale).clamp(-128, 127).to(torch.int8)
        quantized[start:stop] = q
        reconstructed[start:stop] = q.to(torch.float64) * scale
        scales.append(scale)
    return reconstructed, quantized, torch.tensor(scales, dtype=torch.float64)


class GroupedInputQGuard:
    def __init__(
        self,
        cache: grouped.GroupedQProjectionCache,
        weights: Any,
        hidden_states: list[torch.Tensor],
    ) -> None:
        self.cache = cache
        self.weights = weights
        self.hidden_states = hidden_states
        self.original = backend.derive_layer_token
        self.input_records: dict[int, dict[str, torch.Tensor]] = {}

    def install(self) -> None:
        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            if layer_id == LAYER:
                position = int(state["position"])
                source = self.hidden_states[LAYER][position]
                reconstructed, quantized, scales = grouped_reconstruct(
                    source, self.cache.group_size
                )
                gain = self.weights.get_tensor(
                    f"model.layers.{LAYER}.input_layernorm.weight"
                ).contiguous()
                norm = backend.canonical.float_rmsnorm(
                    reconstructed.to(torch.float32), gain
                ).contiguous()
                self.cache.group_inputs[position] = norm
                self.input_records[position] = {
                    "input_q": quantized,
                    "input_scales": scales,
                    "input_reconstructed": reconstructed,
                    "input_norm": norm,
                }
                if template is not None:
                    template["qkv"].pop("q", None)
            return self.original(layer_id, state, cache, template, weights, adapter)

        backend.derive_layer_token = derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original


def run_group(
    group_size: int,
    cache: grouped.GroupedQProjectionCache,
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], GroupedInputQGuard]:
    cache.configure(group_size)
    group_guard = GroupedInputQGuard(cache, weights, hidden_states)
    require_guard = localizer.SubstitutionRequireGuard()
    cache.install()
    group_guard.install()
    require_guard.install()
    try:
        record, _ = localizer.run_cut(
            LAYER,
            hidden_states,
            weights,
            adapter,
            norm_gain,
            embedding,
            head,
            tokenizer,
        )
    finally:
        require_guard.restore()
        group_guard.restore()
        cache.restore()
    q = cache.group_records[len(localizer.TOKEN_IDS) - 1]
    input_record = group_guard.input_records[len(localizer.TOKEN_IDS) - 1]
    record.update(
        {
            "group_size": group_size,
            "groups": backend.HIDDEN // group_size,
            "input_group_scale_min": float(input_record["input_scales"].min()),
            "input_group_scale_max": float(input_record["input_scales"].max()),
            "q_group_scale_min": float(q["input_scales"].min()),
            "q_group_scale_max": float(q["input_scales"].max()),
            "q_output_scale": float(q["output_scale"]),
            "q_saturation_count": int(q["saturation"].to(torch.int64).sum()),
            "suppressed_nonfunctional_cache_effect_assertion_count": len(
                require_guard.suppressed
            ),
        }
    )
    return record, group_guard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.resolve()
    output_path = candidate / "grouped-input-q-probe.json"
    localizer.require(candidate.is_dir(), "candidate absent")
    localizer.require(not output_path.exists(), "grouped input/Q probe exists")
    prior_path = candidate / "grouped-q-probe.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    localizer.require(
        prior["status"] == "PARTIAL_GROUPED_Q_DID_NOT_RECOVER_STEP0",
        "grouped-Q-only result differs",
    )
    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    hidden_states, _ = localizer.bf16_trace(snapshot, tokenizer)

    with safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = grouped.GroupedQProjectionCache(weights, adapter)
        trace = []
        earliest = None
        winning_guard = None
        for group_size in grouped.GROUP_SIZES:
            record, group_guard = run_group(
                group_size,
                cache,
                hidden_states,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            trace.append(record)
            print(
                "ACE2_GROUPED_INPUT_Q_PROBE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} sat={record['q_saturation_count']}",
                flush=True,
            )
            if record["top_token_id"] == localizer.REFERENCE_TOKEN:
                earliest = group_size
                winning_guard = group_guard
                break
        artifacts = None
        if earliest is not None and winning_guard is not None:
            position = len(localizer.TOKEN_IDS) - 1
            tensors = {
                **{
                    f"input_{name}": value
                    for name, value in winning_guard.input_records[position].items()
                },
                **{
                    f"q_{name}": value
                    for name, value in cache.group_records[position].items()
                    if isinstance(value, torch.Tensor)
                },
            }
            artifacts = {
                name: localizer.write_tensor(
                    candidate
                    / f"cpu/grouped-input-q-probe/group-{earliest}/{name}.bin",
                    value,
                )
                for name, value in sorted(tensors.items())
            }

    result = {
        "schema_version": 1,
        "status": (
            "PASS_GROUPED_INPUT_Q_CAUSAL_REPAIR_PROBE"
            if earliest is not None
            else "PARTIAL_GROUPED_INPUT_Q_DID_NOT_RECOVER_STEP0"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "repair_family": {
            "boundary": "model.layers.16.output_to_model.layers.17.input_rmsnorm_and_q_proj",
            "weights": "unchanged signed W4 per output row",
            "layer17_input": "signed A8 per contiguous group before RMSNorm",
            "q_projection_input": "signed A8 per contiguous group after RMSNorm",
            "group_size_shared": True,
            "token_or_logit_override": False,
        },
        "first_recovering_group_size": earliest,
        "trace": trace,
        "artifacts": artifacts,
        "bindings": {
            "runner": localizer.file_record(Path(__file__).resolve()),
            "grouped_q_negative": localizer.file_record(prior_path),
            "layer_scan": localizer.file_record(candidate / "layer-scan.json"),
        },
        "next_required_action": (
            "Run four-token greedy CPU and focused grouped-scale RTL for this group size."
            if earliest is not None
            else "Preserve layer17 Q activation as the exact boundary and use a bounded higher-precision/error-feedback sidecar; no token retuning."
        ),
        "official_run_launched": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    localizer.write_json(output_path, result)
    localizer.write_sums(candidate)
    print(
        f"ACE2_GROUPED_INPUT_Q_RESULT status={result['status']} group={earliest}",
        flush=True,
    )
    return 0 if earliest is not None else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_GROUPED_INPUT_Q_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
