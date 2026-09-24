#!/usr/bin/env python3
"""Separate layer-17 W4 weight error from A8 activation error."""

from __future__ import annotations

import argparse
import gc
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
from peft import PeftModel
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


LAYER = 17
BOUNDARY = LAYER + 1
ORDERED_SUFFIXES = localizer.LINEAR_SUFFIXES


class ExactWeightProjectionCache(localizer.FastProjectionCache):
    """Use exact merged weights for selected projections while retaining A8 I/O."""

    def __init__(self, weights: Any, adapter: Any, exact_names: set[str]) -> None:
        super().__init__(weights, adapter)
        self.exact_names = exact_names
        self.name_by_merged_ptr = {
            value.data_ptr(): name for name, value in self.merged.items()
        }
        self.exact_by_qweight_ptr: dict[int, torch.Tensor] = {}

    @staticmethod
    def exact_a8_projection(
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        output_scale: float,
    ) -> dict[str, Any]:
        real_input = input_q.to(torch.float64) * float(input_scale)
        real_output = torch.mv(merged.to(torch.float64), real_input)
        output_q = torch.round(real_output / float(output_scale)).clamp(-128, 127).to(
            torch.int8
        )
        rounded = torch.round(real_output / float(output_scale)).to(torch.int64)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        return {
            "name": name,
            "input_q": input_q,
            "input_scale": input_scale,
            "accumulator": torch.zeros(output_q.numel(), dtype=torch.int64),
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": output_scale,
            "saturation": saturation,
        }

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        tensor_name = self.name_by_merged_ptr.get(merged.data_ptr())
        localizer.require(tensor_name is not None, f"projection name absent for {name}")
        if tensor_name not in self.exact_names:
            return super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )
        float_output = torch.mv(merged, float_input.to(torch.float32)).contiguous()
        output_scale = backend.canonical.scale_for(float_output)
        result = self.exact_a8_projection(
            name, merged, input_q, input_scale, output_scale
        )
        sentinel = self.metadata[merged.data_ptr()]["qweight"]
        self.exact_by_qweight_ptr[sentinel.data_ptr()] = merged
        result.update(
            {
                "qweight": sentinel,
                "weight_scale": self.metadata[merged.data_ptr()]["weight_scale"],
                "multiplier": torch.zeros(sentinel.shape[0], dtype=torch.int64),
                "right_shift": torch.zeros(sentinel.shape[0], dtype=torch.int64),
                "float_output": float_output,
                "source_hashes": source_hashes,
            }
        )
        return result

    def from_fixed_metadata(
        self,
        name: str,
        input_q: torch.Tensor,
        input_scale: float,
        qweight: torch.Tensor,
        multiplier: torch.Tensor,
        right_shift: torch.Tensor,
        output_scale: float,
    ) -> dict[str, Any]:
        merged = self.exact_by_qweight_ptr.get(qweight.data_ptr())
        if merged is None:
            return super().from_fixed_metadata(
                name,
                input_q,
                input_scale,
                qweight,
                multiplier,
                right_shift,
                output_scale,
            )
        result = self.exact_a8_projection(
            name, merged, input_q, input_scale, output_scale
        )
        result.update(
            {
                "qweight": qweight,
                "multiplier": multiplier,
                "right_shift": right_shift,
            }
        )
        return result


def w4_only_layer_output(
    snapshot: Path,
    cached: ExactWeightProjectionCache,
) -> tuple[torch.Tensor, dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        attn_implementation="eager",
        device_map=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    peft_model = PeftModel.from_pretrained(
        model,
        backend.canonical.ADAPTER_DIR,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    model = peft_model.merge_and_unload().eval()
    modules = dict(model.named_modules())
    records = []
    with torch.no_grad():
        for suffix in ORDERED_SUFFIXES:
            name = f"model.layers.{LAYER}.{suffix}"
            module = modules.get(name)
            localizer.require(isinstance(module, nn.Linear), f"merged module absent: {name}")
            metadata = cached.metadata[cached.merged[name].data_ptr()]
            dequantized = (
                metadata["qweight"].to(torch.float64)
                * metadata["weight_scale"][:, None]
            ).to(module.weight.dtype)
            module.weight.copy_(dequantized)
            records.append(
                {
                    "name": name,
                    "qweight_sha256": localizer.tensor_sha256(metadata["qweight"]),
                    "weight_scale_sha256": localizer.tensor_sha256(
                        metadata["weight_scale"]
                    ),
                }
            )
    input_ids = torch.tensor([localizer.TOKEN_IDS], dtype=torch.long)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = output.hidden_states[BOUNDARY][0].float().cpu().contiguous()
    localizer.require(
        list(hidden.shape) == [len(localizer.TOKEN_IDS), backend.HIDDEN],
        "W4-only boundary shape differs",
    )
    del model, peft_model, output, input_ids
    gc.collect()
    return hidden, {"linears": records, "boundary_sha256": localizer.tensor_sha256(hidden)}


def run_variant(
    cache: ExactWeightProjectionCache,
    exact_names: set[str],
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> dict[str, Any]:
    cache.exact_names = set(exact_names)
    cache.exact_by_qweight_ptr.clear()
    guard = localizer.SubstitutionRequireGuard()
    cache.install()
    guard.install()
    try:
        record, _final_q = localizer.run_cut(
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
        guard.restore()
        cache.restore()
    record.update(
        {
            "exact_weight_names": sorted(exact_names),
            "exact_weight_count": len(exact_names),
            "suppressed_nonfunctional_cache_effect_assertion_count": len(
                guard.suppressed
            ),
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.resolve()
    localizer.require(candidate.is_dir(), "candidate directory absent")
    localizer.require(
        not (candidate / "mechanism-split.json").exists(),
        "mechanism split already exists",
    )
    layer_scan_path = candidate / "layer-scan.json"
    layer_scan = json.loads(layer_scan_path.read_text(encoding="utf-8"))
    localizer.require(
        layer_scan["layerwise_causal_substitution"]["earliest_recovering_cut"]
        == BOUNDARY,
        "earliest layer cut differs",
    )
    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    hidden_states, _logits = localizer.bf16_trace(snapshot, tokenizer)

    with safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)

        print("ACE2_LAYER17_MECHANISM_CACHE", flush=True)
        baseline_cache = ExactWeightProjectionCache(weights, adapter, set())
        w4_only_hidden, w4_only_binding = w4_only_layer_output(
            snapshot, baseline_cache
        )
        w4_only_states = list(hidden_states)
        w4_only_states[BOUNDARY] = w4_only_hidden
        baseline_cache.install()
        baseline_guard = localizer.SubstitutionRequireGuard()
        baseline_guard.install()
        try:
            w4_only, _ = localizer.run_cut(
                BOUNDARY,
                w4_only_states,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
        finally:
            baseline_guard.restore()
            baseline_cache.restore()
        w4_only.update(
            {
                "definition": "layer17 W4-dequantized weights with BF16 activations/operators, then A8 boundary and unchanged W4A8 layers18..23/head",
                "binding": w4_only_binding,
            }
        )
        print(
            f"ACE2_LAYER17_W4_ONLY top={w4_only['top_token_id']} rank={w4_only['reference_rank']}",
            flush=True,
        )

        layer_names = {
            f"model.layers.{LAYER}.{suffix}" for suffix in ORDERED_SUFFIXES
        }
        a8_only = run_variant(
            baseline_cache,
            layer_names,
            hidden_states,
            weights,
            adapter,
            norm_gain,
            embedding,
            head,
            tokenizer,
        )
        a8_only["definition"] = (
            "layer17 exact merged weights with accepted A8 RMSNorm/attention/SiLU/residual and A8 projection I/O, then unchanged W4A8 layers18..23/head"
        )
        print(
            f"ACE2_LAYER17_A8_ONLY top={a8_only['top_token_id']} rank={a8_only['reference_rank']}",
            flush=True,
        )

        prefix_trace = []
        active: set[str] = set()
        earliest_weight_boundary = None
        for suffix in ORDERED_SUFFIXES:
            name = f"model.layers.{LAYER}.{suffix}"
            active.add(name)
            record = run_variant(
                baseline_cache,
                active,
                hidden_states,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            record["added_boundary"] = name
            prefix_trace.append(record)
            print(
                "ACE2_LAYER17_WEIGHT_PREFIX "
                f"boundary={name} top={record['top_token_id']} "
                f"rank={record['reference_rank']}",
                flush=True,
            )
            if record["top_token_id"] == localizer.REFERENCE_TOKEN:
                earliest_weight_boundary = name
                break

    if w4_only["top_token_id"] != localizer.REFERENCE_TOKEN and a8_only[
        "top_token_id"
    ] == localizer.REFERENCE_TOKEN:
        mechanism = "W4_WEIGHT_PRIMARY"
    elif w4_only["top_token_id"] == localizer.REFERENCE_TOKEN and a8_only[
        "top_token_id"
    ] != localizer.REFERENCE_TOKEN:
        mechanism = "A8_ACTIVATION_PRIMARY"
    elif w4_only["top_token_id"] != localizer.REFERENCE_TOKEN and a8_only[
        "top_token_id"
    ] != localizer.REFERENCE_TOKEN:
        mechanism = "W4_WEIGHT_AND_A8_ACTIVATION_INTERACTION"
    else:
        mechanism = "NEITHER_ISOLATED_MECHANISM_CHANGES_TOP1"

    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER17_WEIGHT_ACTIVATION_SPLIT"
            if mechanism != "NEITHER_ISOLATED_MECHANISM_CHANGES_TOP1"
            else "PARTIAL_LAYER17_SPLIT_NONDECISIVE"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "implicated_layer": LAYER,
        "causal_boundary": "model.layers.17.output",
        "mechanism": mechanism,
        "w4_weight_only": w4_only,
        "a8_activation_only": a8_only,
        "cumulative_exact_weight_prefix": {
            "definition": (
                "Execution-ordered exact-weight prefix inside layer17 while all layer17 "
                "activation operators remain on the accepted A8 path."
            ),
            "earliest_recovering_boundary": earliest_weight_boundary,
            "trace": prefix_trace,
        },
        "bindings": {
            "runner": localizer.file_record(Path(__file__).resolve()),
            "layer_scan": localizer.file_record(layer_scan_path),
            "candidate_freeze": localizer.file_record(candidate / "candidate-freeze.json"),
        },
        "next_required_action": (
            "Implement a grouped-W4 repair at the earliest recovering projection boundary and verify four aligned tokens plus focused RTL."
            if earliest_weight_boundary is not None
            else "Localize activation operators within layer17 before selecting a repair."
        ),
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    localizer.write_json(candidate / "mechanism-split.json", result)
    localizer.write_sums(candidate)
    print(
        "ACE2_LAYER17_MECHANISM_RESULT "
        f"mechanism={mechanism} boundary={earliest_weight_boundary}",
        flush=True,
    )
    return 0 if result["status"].startswith("PASS") else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_LAYER17_MECHANISM_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
