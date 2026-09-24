#!/usr/bin/env python3
"""Probe grouped-A8 RMSNorm-to-Q fusion at the causal layer-17 boundary."""

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
from tools import ace2_checkpoint176_layer17_mechanism as mechanism
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


LAYER = mechanism.LAYER
Q_PATTERN = re.compile(r"layer17_position(\d+)_q$")
GROUP_SIZES = (128, 64, 32, 16, 8, 4, 1)


class GroupedQProjectionCache(localizer.FastProjectionCache):
    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        name = f"model.layers.{LAYER}.self_attn.q_proj"
        self.q_merged = self.merged[name]
        self.q_metadata = self.metadata[self.q_merged.data_ptr()]
        self.group_size = 128
        self.group_inputs: dict[int, torch.Tensor] = {}
        self.group_records: dict[int, dict[str, Any]] = {}

    def configure(self, group_size: int) -> None:
        localizer.require(backend.HIDDEN % group_size == 0, "group size does not divide hidden")
        self.group_size = group_size
        self.group_inputs.clear()
        self.group_records.clear()

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
        if match is None:
            return super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )
        position = int(match.group(1))
        source = self.group_inputs[position].to(torch.float64)
        float_output = torch.mv(merged, float_input.to(torch.float32)).contiguous()
        output_scale = backend.canonical.scale_for(float_output)
        qweight = self.q_metadata["qweight"]
        weight_scale = self.q_metadata["weight_scale"]
        real_output = torch.zeros(qweight.shape[0], dtype=torch.float64)
        group_scales = []
        group_q = torch.empty(backend.HIDDEN, dtype=torch.int8)
        partials = []
        for start in range(0, backend.HIDDEN, self.group_size):
            stop = start + self.group_size
            group = source[start:stop]
            scale = max(float(group.abs().max()) / 127.0, 1e-12)
            activation = torch.round(group / scale).clamp(-128, 127).to(torch.int8)
            partial = torch.mv(
                qweight[:, start:stop].to(torch.int64), activation.to(torch.int64)
            )
            real_output += partial.to(torch.float64) * scale * weight_scale
            group_q[start:stop] = activation
            group_scales.append(scale)
            partials.append(partial)
        rounded = torch.round(real_output / output_scale).to(torch.int64)
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        partial_matrix = torch.stack(partials, dim=1).contiguous()
        scale_tensor = torch.tensor(group_scales, dtype=torch.float64)
        self.group_records[position] = {
            "group_size": self.group_size,
            "input_q": group_q,
            "input_scales": scale_tensor,
            "partial_accumulators": partial_matrix,
            "real_output": real_output,
            "output_q": output_q,
            "output_scale": output_scale,
            "saturation": saturation,
        }
        return {
            "name": name,
            "input_q": group_q,
            "input_scale": input_scale,
            "qweight": qweight,
            "weight_scale": weight_scale,
            "multiplier": torch.zeros(qweight.shape[0], dtype=torch.int64),
            "right_shift": torch.zeros(qweight.shape[0], dtype=torch.int64),
            "accumulator": partial_matrix.sum(dim=1),
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": output_scale,
            "saturation": saturation,
            "float_output": float_output,
            "source_hashes": source_hashes,
            "grouped_a8": True,
        }


class GroupedQGuard:
    def __init__(self, cache: GroupedQProjectionCache, weights: Any) -> None:
        self.cache = cache
        self.weights = weights
        self.original = backend.derive_layer_token

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
                gain = self.weights.get_tensor(
                    f"model.layers.{LAYER}.input_layernorm.weight"
                ).contiguous()
                fixed_hidden = (
                    state["fixed_q"].to(torch.float64) * float(state["fixed_scale"])
                ).to(torch.float32)
                self.cache.group_inputs[position] = backend.canonical.float_rmsnorm(
                    fixed_hidden, gain
                ).contiguous()
                if template is not None:
                    template["qkv"].pop("q", None)
            return self.original(layer_id, state, cache, template, weights, adapter)

        backend.derive_layer_token = derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original


def run_group(
    group_size: int,
    cache: GroupedQProjectionCache,
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> dict[str, Any]:
    cache.configure(group_size)
    grouped_guard = GroupedQGuard(cache, weights)
    require_guard = localizer.SubstitutionRequireGuard()
    cache.install()
    grouped_guard.install()
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
        grouped_guard.restore()
        cache.restore()
    q = cache.group_records[len(localizer.TOKEN_IDS) - 1]
    record.update(
        {
            "group_size": group_size,
            "groups": backend.HIDDEN // group_size,
            "q_output_scale": float(q["output_scale"]),
            "q_saturation_count": int(q["saturation"].to(torch.int64).sum()),
            "q_group_scale_min": float(q["input_scales"].min()),
            "q_group_scale_max": float(q["input_scales"].max()),
            "q_output_sha256": localizer.tensor_sha256(q["output_q"]),
            "suppressed_nonfunctional_cache_effect_assertion_count": len(
                require_guard.suppressed
            ),
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.resolve()
    output_path = candidate / "grouped-q-probe.json"
    localizer.require(candidate.is_dir(), "candidate absent")
    localizer.require(not output_path.exists(), "grouped Q probe exists")
    dynamic_path = candidate / "dynamic-q-repair.json"
    dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
    localizer.require(
        dynamic["status"] == "PARTIAL_DYNAMIC_Q_SCALE_CPU",
        "dynamic-scale negative result differs",
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
        cache = GroupedQProjectionCache(weights, adapter)
        trace = []
        earliest = None
        for group_size in GROUP_SIZES:
            record = run_group(
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
                "ACE2_GROUPED_Q_PROBE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} sat={record['q_saturation_count']}",
                flush=True,
            )
            if record["top_token_id"] == localizer.REFERENCE_TOKEN:
                earliest = group_size
                q = cache.group_records[len(localizer.TOKEN_IDS) - 1]
                artifacts = {
                    name: localizer.write_tensor(
                        candidate
                        / f"cpu/grouped-q-probe/group-{group_size}/{name}.bin",
                        value,
                    )
                    for name, value in q.items()
                    if isinstance(value, torch.Tensor)
                }
                record["artifacts"] = artifacts
                break

    result = {
        "schema_version": 1,
        "status": (
            "PASS_GROUPED_Q_CAUSAL_REPAIR_PROBE"
            if earliest is not None
            else "PARTIAL_GROUPED_Q_DID_NOT_RECOVER_STEP0"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "repair_family": {
            "boundary": "model.layers.17.input_rmsnorm_to_self_attn.q_proj",
            "weights": "unchanged signed W4 per output row",
            "activations": "signed A8 per contiguous input group",
            "accumulation": "signed integer partial dot per group, model-derived scale per group, summed before signed-A8 Q output requantization",
            "source": "RMSNorm of dequantized incoming A8 hidden state",
            "token_or_logit_override": False,
        },
        "first_recovering_group_size": earliest,
        "trace": trace,
        "bindings": {
            "runner": localizer.file_record(Path(__file__).resolve()),
            "dynamic_q_negative": localizer.file_record(dynamic_path),
            "activation_localization": localizer.file_record(
                candidate / "activation-localization.json"
            ),
        },
        "next_required_action": (
            "Run four-token greedy CPU with the first recovering grouped-A8 Q repair and emit focused RTL vectors."
            if earliest is not None
            else "Preserve the activation boundary and test a wider RMSNorm output representation; do not retune logits."
        ),
        "official_run_launched": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    localizer.write_json(output_path, result)
    localizer.write_sums(candidate)
    print(
        f"ACE2_GROUPED_Q_RESULT status={result['status']} group={earliest}",
        flush=True,
    )
    return 0 if earliest is not None else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_GROUPED_Q_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
