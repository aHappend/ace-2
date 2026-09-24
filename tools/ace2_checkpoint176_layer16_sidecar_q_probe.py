#!/usr/bin/env python3
"""Probe a hardware-available layer16 residual sidecar into layer17 Q."""

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
from tools import ace2_checkpoint176_layer17_grouped_input_q_probe as grouped_input
from tools import ace2_checkpoint176_layer17_grouped_q_probe as grouped_q
from tools import ace2_checkpoint176_layer17_mechanism as mechanism
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


SOURCE_LAYER = 16
TARGET_LAYER = mechanism.LAYER
GROUP_SIZES = (32, 16, 8, 4, 1)


class SidecarGuard:
    def __init__(
        self,
        cache: grouped_q.GroupedQProjectionCache,
        weights: Any,
    ) -> None:
        self.cache = cache
        self.weights = weights
        self.original = backend.derive_layer_token
        self.sidecars: dict[int, torch.Tensor] = {}
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
            if layer_id == TARGET_LAYER:
                position = int(state["position"])
                source = self.sidecars[position]
                reconstructed, quantized, scales = grouped_input.grouped_reconstruct(
                    source, self.cache.group_size
                )
                gain = self.weights.get_tensor(
                    f"model.layers.{TARGET_LAYER}.input_layernorm.weight"
                ).contiguous()
                norm = backend.canonical.float_rmsnorm(
                    reconstructed.to(torch.float32), gain
                ).contiguous()
                self.cache.group_inputs[position] = norm
                self.input_records[position] = {
                    "sidecar": source,
                    "input_q": quantized,
                    "input_scales": scales,
                    "input_reconstructed": reconstructed,
                    "input_norm": norm,
                }
                if template is not None:
                    template["qkv"].pop("q", None)
            derived, next_state, next_template = self.original(
                layer_id, state, cache, template, weights, adapter
            )
            if layer_id == SOURCE_LAYER:
                position = derived["positions"][0]
                attention = (
                    position["attention_residual"]["output"].to(torch.float64)
                    * float(position["attention_residual"]["scale"])
                )
                down = (
                    position["projections"]["down"]["output_q"].to(torch.float64)
                    * float(position["projections"]["down"]["output_scale"])
                )
                self.sidecars[int(state["position"])] = (attention + down).contiguous()
            return derived, next_state, next_template

        backend.derive_layer_token = derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original


def run_group(
    group_size: int,
    cache: grouped_q.GroupedQProjectionCache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], SidecarGuard]:
    cache.configure(group_size)
    sidecar = SidecarGuard(cache, weights)
    cache.install()
    sidecar.install()
    try:
        record, _ = localizer.run_cut(
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
        sidecar.restore()
        cache.restore()
    position = len(localizer.TOKEN_IDS) - 1
    q = cache.group_records[position]
    input_record = sidecar.input_records[position]
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
            "sidecar_sha256": localizer.tensor_sha256(input_record["sidecar"]),
        }
    )
    return record, sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.resolve()
    output_path = candidate / "layer16-sidecar-q-probe.json"
    localizer.require(candidate.is_dir(), "candidate absent")
    localizer.require(not output_path.exists(), "sidecar Q probe exists")
    prior_path = candidate / "grouped-input-q-probe.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    localizer.require(
        prior["first_recovering_group_size"] == 1,
        "BF16-boundary grouped input result differs",
    )
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
        cache = grouped_q.GroupedQProjectionCache(weights, adapter)
        trace = []
        earliest = None
        winner = None
        for group_size in GROUP_SIZES:
            record, sidecar = run_group(
                group_size,
                cache,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            trace.append(record)
            print(
                "ACE2_LAYER16_SIDECAR_Q_PROBE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} sat={record['q_saturation_count']}",
                flush=True,
            )
            if record["top_token_id"] == localizer.REFERENCE_TOKEN:
                earliest = group_size
                winner = sidecar
                break
        artifacts = None
        if earliest is not None and winner is not None:
            position = len(localizer.TOKEN_IDS) - 1
            tensors = {
                **{
                    f"input_{name}": value
                    for name, value in winner.input_records[position].items()
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
                    / f"cpu/layer16-sidecar-q/group-{earliest}/{name}.bin",
                    value,
                )
                for name, value in sorted(tensors.items())
            }

    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER16_SIDECAR_GROUPED_Q_PROBE"
            if earliest is not None
            else "PARTIAL_LAYER16_SIDECAR_GROUPED_Q_NO_RECOVERY"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "repair_family": {
            "source": (
                "hardware-available layer16 attention-residual A8 stream plus layer16 "
                "down-projection A8 stream before common layer-output requantization"
            ),
            "target": "layer17 RMSNorm and Q projection only",
            "weights": "unchanged signed W4 per output row",
            "activation_streams": "signed A8 with per-group scales",
            "token_or_logit_override": False,
        },
        "first_recovering_group_size": earliest,
        "trace": trace,
        "artifacts": artifacts,
        "bindings": {
            "runner": localizer.file_record(Path(__file__).resolve()),
            "bf16_grouped_input_probe": localizer.file_record(prior_path),
            "activation_localization": localizer.file_record(
                candidate / "activation-localization.json"
            ),
        },
        "next_required_action": (
            "Run four-token greedy CPU and focused sidecar/grouped-scale RTL."
            if earliest is not None
            else "Preserve layer17 Q activation boundary and the exact remaining need for a higher-precision RMSNorm/Q sidecar."
        ),
        "official_run_launched": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    localizer.write_json(output_path, result)
    localizer.write_sums(candidate)
    print(
        f"ACE2_LAYER16_SIDECAR_Q_RESULT status={result['status']} group={earliest}",
        flush=True,
    )
    return 0 if earliest is not None else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_LAYER16_SIDECAR_Q_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
