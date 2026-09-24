#!/usr/bin/env python3
"""Localize the earliest causal A8 activation boundary inside layer 17."""

from __future__ import annotations

import argparse
import gc
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
from peft import PeftModel
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_checkpoint176_layer17_mechanism as mechanism
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


LAYER = mechanism.LAYER
PROJECTION_PATTERN = re.compile(r"layer17_position(\d+)_(q|k|v|o|gate|up|down)$")


def capture_w4_only_activations(
    snapshot: Path,
    cached: mechanism.ExactWeightProjectionCache,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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
    with torch.no_grad():
        for suffix in mechanism.ORDERED_SUFFIXES:
            name = f"model.layers.{LAYER}.{suffix}"
            module = modules.get(name)
            localizer.require(isinstance(module, nn.Linear), f"module absent: {name}")
            metadata = cached.metadata[cached.merged[name].data_ptr()]
            module.weight.copy_(
                (
                    metadata["qweight"].to(torch.float64)
                    * metadata["weight_scale"][:, None]
                ).to(module.weight.dtype)
            )

    captures: dict[str, torch.Tensor] = {}
    handles = []
    layer = model.model.layers[LAYER]

    def capture_output(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            localizer.require(isinstance(value, torch.Tensor), f"capture differs: {name}")
            captures[name] = value[0].float().cpu().contiguous()

        return hook

    handles.append(layer.input_layernorm.register_forward_hook(capture_output("input_norm")))
    handles.append(layer.self_attn.q_proj.register_forward_hook(capture_output("q")))
    handles.append(layer.self_attn.k_proj.register_forward_hook(capture_output("k")))
    handles.append(layer.self_attn.v_proj.register_forward_hook(capture_output("v")))
    handles.append(layer.self_attn.o_proj.register_forward_hook(capture_output("o")))
    handles.append(layer.post_attention_layernorm.register_forward_hook(capture_output("post_norm")))
    handles.append(layer.mlp.gate_proj.register_forward_hook(capture_output("gate")))
    handles.append(layer.mlp.up_proj.register_forward_hook(capture_output("up")))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(
        lambda _module, inputs: captures.__setitem__(
            "silu", inputs[0][0].float().cpu().contiguous()
        )
    ))
    handles.append(layer.mlp.down_proj.register_forward_hook(capture_output("down")))
    handles.append(layer.register_forward_hook(capture_output("layer_output")))

    input_ids = torch.tensor([localizer.TOKEN_IDS], dtype=torch.long)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()
    localizer.require(
        int(torch.argmax(output.logits[0, -1])) == localizer.REFERENCE_TOKEN,
        "W4-only activation capture no longer recovers reference",
    )
    expected = {
        "input_norm",
        "q",
        "k",
        "v",
        "o",
        "post_norm",
        "gate",
        "up",
        "silu",
        "down",
        "layer_output",
    }
    localizer.require(set(captures) == expected, "activation capture set differs")
    binding = {
        name: {
            "shape": list(value.shape),
            "sha256": localizer.tensor_sha256(value),
        }
        for name, value in sorted(captures.items())
    }
    del model, peft_model, output, input_ids
    gc.collect()
    return captures, binding


class ActivationProjectionCache(mechanism.ExactWeightProjectionCache):
    """Inject W4-only BF16 activations while retaining accepted A8 formats."""

    def __init__(
        self,
        weights: Any,
        adapter: Any,
        captures: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(weights, adapter, set())
        self.captures = captures
        self.inject_input_norm = False
        self.inject_outputs: set[str] = set()

    def configure(self, *, input_norm: bool, outputs: set[str]) -> None:
        self.inject_input_norm = input_norm
        self.inject_outputs = set(outputs)
        self.exact_names.clear()
        self.exact_by_qweight_ptr.clear()

    def _projection_inputs(
        self, name: str, input_q: torch.Tensor, input_scale: float
    ) -> tuple[torch.Tensor, re.Match[str] | None]:
        match = PROJECTION_PATTERN.fullmatch(name)
        if (
            match is not None
            and self.inject_input_norm
            and match.group(2) in {"q", "k", "v"}
        ):
            position = int(match.group(1))
            input_q = backend.canonical.quantize_int8(
                self.captures["input_norm"][position], input_scale
            )
        return input_q, match

    def _projection_output(
        self, result: dict[str, Any], match: re.Match[str] | None
    ) -> dict[str, Any]:
        if match is None or match.group(2) not in self.inject_outputs:
            return result
        position = int(match.group(1))
        key = match.group(2)
        real = self.captures[key][position]
        result["output_q"] = backend.canonical.quantize_int8(
            real, float(result["output_scale"])
        )
        result["activation_substitution"] = {
            "boundary": key,
            "position": position,
            "source": "layer17_w4_dequant_weight_bf16_activation_capture",
        }
        return result

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        input_q, match = self._projection_inputs(name, input_q, input_scale)
        result = super().derive_projection(
            name, merged, input_q, input_scale, float_input, source_hashes
        )
        return self._projection_output(result, match)

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
        input_q, match = self._projection_inputs(name, input_q, input_scale)
        result = super().from_fixed_metadata(
            name,
            input_q,
            input_scale,
            qweight,
            multiplier,
            right_shift,
            output_scale,
        )
        return self._projection_output(result, match)


def run_variant(
    cache: ActivationProjectionCache,
    *,
    input_norm: bool,
    outputs: set[str],
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> dict[str, Any]:
    cache.configure(input_norm=input_norm, outputs=outputs)
    guard = localizer.SubstitutionRequireGuard()
    cache.install()
    guard.install()
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
        guard.restore()
        cache.restore()
    record.update(
        {
            "inject_input_norm": input_norm,
            "injected_projection_outputs": sorted(outputs),
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
    output_path = candidate / "activation-localization.json"
    localizer.require(candidate.is_dir(), "candidate absent")
    localizer.require(not output_path.exists(), "activation localization exists")
    mechanism_path = candidate / "mechanism-split.json"
    split = json.loads(mechanism_path.read_text(encoding="utf-8"))
    localizer.require(split["mechanism"] == "A8_ACTIVATION_PRIMARY", "mechanism differs")
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
        cache = ActivationProjectionCache(weights, adapter, {})
        captures, capture_binding = capture_w4_only_activations(snapshot, cache)
        cache.captures = captures
        capture_artifacts = {
            name: localizer.write_tensor(
                candidate / f"cpu/layer17-w4-only/{name}-f32le.bin", value
            )
            for name, value in sorted(captures.items())
        }

        variants = [
            ("input_rmsnorm", True, set()),
            ("q_projection_output", False, {"q"}),
            ("qk_projection_outputs", False, {"q", "k"}),
            ("qkv_projection_outputs", False, {"q", "k", "v"}),
        ]
        trace = []
        earliest = None
        for boundary, input_norm, outputs in variants:
            record = run_variant(
                cache,
                input_norm=input_norm,
                outputs=outputs,
                hidden_states=hidden_states,
                weights=weights,
                adapter=adapter,
                norm_gain=norm_gain,
                embedding=embedding,
                head=head,
                tokenizer=tokenizer,
            )
            record["boundary"] = boundary
            trace.append(record)
            print(
                "ACE2_LAYER17_ACTIVATION_BOUNDARY "
                f"boundary={boundary} top={record['top_token_id']} "
                f"rank={record['reference_rank']}",
                flush=True,
            )
            if record["top_token_id"] == localizer.REFERENCE_TOKEN:
                earliest = boundary
                break

    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER17_EARLIEST_A8_BOUNDARY_LOCALIZED"
            if earliest is not None
            else "PARTIAL_LAYER17_A8_BOUNDARY_AFTER_QKV"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "implicated_layer": LAYER,
        "mechanism": "A8_ACTIVATION_PRIMARY",
        "definition": (
            "Each row injects a W4-dequant-weight/BF16-activation tensor at one "
            "execution-ordered layer17 boundary for all 30 causal prefix positions, "
            "requantizes it to the accepted signed-A8 scale, and leaves downstream "
            "integer operators, layers18..23, final RMSNorm, and tied W4 head unchanged."
        ),
        "earliest_recovering_boundary": earliest,
        "trace": trace,
        "captures": {
            "binding": capture_binding,
            "artifacts": capture_artifacts,
        },
        "bindings": {
            "runner": localizer.file_record(Path(__file__).resolve()),
            "mechanism_split": localizer.file_record(mechanism_path),
            "layer_scan": localizer.file_record(candidate / "layer-scan.json"),
        },
        "next_required_action": (
            "Replace the implicated tensor-wide A8 boundary with a model-derived grouped activation scale and verify aligned four-token CPU/RTL behavior."
            if earliest is not None
            else "Extend activation substitution through attention compose, o-projection, post-RMSNorm, SiLU, and down-projection."
        ),
        "official_run_launched": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    localizer.write_json(output_path, result)
    localizer.write_sums(candidate)
    print(
        f"ACE2_LAYER17_ACTIVATION_RESULT status={result['status']} boundary={earliest}",
        flush=True,
    )
    return 0 if earliest is not None else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_LAYER17_ACTIVATION_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
