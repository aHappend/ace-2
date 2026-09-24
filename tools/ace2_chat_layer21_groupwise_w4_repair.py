#!/usr/bin/env python3
"""Test a packable groupwise-W4/A8 repair at the Layer-21 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_full_model_fixed_point as fixed_point
from tools.ace2_chat_all_residual_reference import (
    FROZEN_SCALES,
    MODEL_REPOSITORY,
    ROPE_MECHANISM,
    exact_process_argv,
    frozen_ranges,
    prompt_coherence,
    sha256_bytes,
    summarize_execution,
    token_piece,
    top_k,
    utc_now,
    validate_frozen_scale_binding,
)
from tools.ace2_chat_demo import (
    REVISION,
    TOKENIZER_SNAPSHOT,
    canonical_bytes,
    decode_generated,
    load_tokenizer,
    sha256_file,
    tokenize_prompt,
    write_atomic,
)
from tools.ace2_chat_layer_prefix_substitution import PrefixSubstitution, relative_l2
from tools.ace2_chat_reference_repairs import readability_gate


LAYER_INDEX = 21
PREFIX21 = (
    ROOT
    / "build/ace2_chat_diagnostics/layer-prefix-substitution-p21-two-prompts-20260805.json"
)
PREFIX22 = (
    ROOT
    / "build/ace2_chat_diagnostics/layer-prefix-substitution-p22-eight-token-two-prompts-20260805.json"
)


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def realized_scale32(record: int) -> float:
    numerator, denominator = fixed_point.scale32_ratio(record)
    return numerator / denominator


class GroupwiseW4A8Linear(nn.Module):
    """Signed-W4 groups rescaled into one signed-32 output-channel domain."""

    def __init__(
        self,
        source: nn.Linear,
        template: fixed_point.W4A8Linear,
        group_size: int,
    ) -> None:
        super().__init__()
        if source.in_features % group_size:
            raise ValueError("group size must divide the linear input width")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.group_size = group_size
        self.groups = source.in_features // group_size
        self.input_scale = float(template.input_scale)
        self.hardware_input_scale = float(template.hardware_input_scale)
        self.output_scale = template.output_scale
        self.output_head_size = template.output_head_size
        self.output_scale32_record = template.output_scale32_record
        self.input_is_quantized = template.input_is_quantized
        self.register_buffer(
            "output_head_scales",
            template.output_head_scales.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "output_scale_per_channel",
            template.output_scale_per_channel.detach().clone(),
            persistent=True,
        )
        weight = source.weight.detach().to(torch.float64).reshape(
            self.out_features, self.groups, group_size
        )
        weight_scale = weight.abs().amax(dim=2) / 7.0
        weight_scale = torch.where(
            weight_scale > 0, weight_scale, torch.ones_like(weight_scale)
        )
        qweight = torch.round(weight / weight_scale[:, :, None]).clamp(-8, 7).to(
            torch.int8
        )
        self.register_buffer("qweight", qweight, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        self.register_buffer(
            "_source_bias",
            (
                source.bias.detach().to(torch.float64)
                if source.bias is not None
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "native_scale32_records",
            torch.zeros(self.out_features, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "group_multiplier",
            torch.zeros(
                self.out_features, self.groups, dtype=torch.int64
            ),
            persistent=True,
        )
        self.register_buffer(
            "group_right_shift",
            torch.zeros(
                self.out_features, self.groups, dtype=torch.int64
            ),
            persistent=True,
        )
        self.register_buffer(
            "multiplier",
            torch.zeros(self.out_features, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "right_shift",
            torch.zeros(self.out_features, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "bias_accumulator",
            (
                torch.zeros(self.out_features, dtype=torch.int64)
                if source.bias is not None
                else None
            ),
            persistent=True,
        )
        self.bind_hardware_input_scale(self.hardware_input_scale)

    def bind_hardware_input_scale(self, input_scale: float) -> None:
        if not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("groupwise projection input scale must be positive")
        self.hardware_input_scale = float(input_scale)
        native = self.weight_scale * self.hardware_input_scale
        minimum = native.amin(dim=1)
        records = torch.tensor(
            [
                fixed_point.ceil_scale32_from_float(float(value))
                for value in minimum.detach().cpu().tolist()
            ],
            dtype=torch.int64,
            device=native.device,
        )
        common = torch.tensor(
            [realized_scale32(int(value)) for value in records.cpu().tolist()],
            dtype=torch.float64,
            device=native.device,
        )
        group_multiplier, group_right_shift = fixed_point.derive_multiplier(
            native / common[:, None]
        )
        multiplier, right_shift = fixed_point.derive_multiplier(
            common / self.output_scale_per_channel
        )
        self.native_scale32_records.copy_(records)
        self.group_multiplier.copy_(group_multiplier)
        self.group_right_shift.copy_(group_right_shift)
        self.multiplier.copy_(multiplier)
        self.right_shift.copy_(right_shift)
        if self.bias_accumulator is not None:
            if self._source_bias is None:
                raise AssertionError("groupwise bias source disappeared")
            self.bias_accumulator.copy_(
                torch.round(self._source_bias / common).to(torch.int64)
            )

    def accumulator_quantized(self, qinput: Tensor) -> Tensor:
        if qinput.dtype != torch.int8:
            raise TypeError("groupwise projection input must be signed int8")
        if qinput.shape[-1] != self.in_features:
            raise ValueError("groupwise projection input width differs")
        flat = qinput.reshape(-1, self.groups, self.group_size).to(torch.int64)
        group_accumulator = torch.einsum(
            "rgk,ogk->rog", flat, self.qweight.to(torch.int64)
        )
        converted = fixed_point.round_shift_even(
            group_accumulator * self.group_multiplier[None, :, :],
            self.group_right_shift[None, :, :],
        )
        accumulator = converted.sum(dim=2)
        if self.bias_accumulator is not None:
            accumulator = accumulator + self.bias_accumulator
        if torch.any(accumulator < -(1 << 31)) or torch.any(
            accumulator >= (1 << 31)
        ):
            raise OverflowError("groupwise common-domain accumulator exceeds signed-32")
        return accumulator

    def requantize_accumulator(
        self,
        accumulator: Tensor,
        original_shape: tuple[int, ...],
    ) -> Tensor:
        if accumulator.dtype != torch.int64:
            raise TypeError("groupwise accumulator must be signed int64 container")
        product = accumulator * self.multiplier
        output = fixed_point.round_shift_even(product, self.right_shift)
        return output.clamp(-128, 127).to(torch.int8).reshape(
            *original_shape, self.out_features
        )

    def forward_quantized(self, qinput: Tensor) -> Tensor:
        original_shape = qinput.shape[:-1]
        return self.requantize_accumulator(
            self.accumulator_quantized(qinput), original_shape
        )

    def forward_hardware_input(self, inputs: Tensor) -> Tensor:
        if inputs.dtype == torch.int8:
            qinput = inputs
        elif inputs.is_floating_point():
            if not torch.all(torch.isfinite(inputs)):
                raise ValueError("groupwise hardware input must be finite")
            if torch.any(inputs < -128) or torch.any(inputs > 127):
                raise ValueError("groupwise hardware input is outside signed int8")
            if not torch.equal(inputs, torch.round(inputs)):
                raise ValueError("groupwise hardware input must contain integers")
            qinput = inputs.to(torch.int8)
        else:
            raise TypeError("groupwise hardware input type differs")
        return self.forward_quantized(qinput)

    def forward_raw(self, inputs: Tensor) -> Tensor:
        return self.forward_quantized(
            fixed_point.quantize_int8(inputs, self.input_scale)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        raw = (
            self.forward_hardware_input(inputs)
            if self.input_is_quantized
            else self.forward_raw(inputs)
        )
        return raw.to(inputs.dtype) * self.output_scale_per_channel.to(inputs.dtype)


def pack_signed_int4(value: Tensor) -> bytes:
    flat = value.detach().cpu().to(torch.int16).reshape(-1)
    if torch.any(flat < -8) or torch.any(flat > 7):
        raise ValueError("packed W4 tensor exceeds signed-int4")
    nibble = torch.bitwise_and(flat, 0xF).to(torch.uint8)
    if nibble.numel() % 2:
        nibble = torch.cat([nibble, torch.zeros(1, dtype=torch.uint8)])
    packed = nibble[0::2] | torch.bitwise_left_shift(nibble[1::2], 4)
    return packed.numpy().tobytes()


def append_section(
    payload: bytearray,
    name: str,
    data: bytes,
) -> dict[str, Any]:
    offset = len(payload)
    payload.extend(data)
    return {
        "name": name,
        "offset_bytes": offset,
        "length_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def module_manifest(
    name: str,
    source: nn.Linear,
    baseline: fixed_point.W4A8Linear,
    candidate: GroupwiseW4A8Linear,
    payload: bytearray,
) -> dict[str, Any]:
    weight = source.weight.detach().to(torch.float64)
    baseline_weight = (
        baseline.qweight.to(torch.float64) * baseline.weight_scale[:, None]
    )
    candidate_weight = (
        candidate.qweight.to(torch.float64) * candidate.weight_scale[:, :, None]
    ).reshape_as(weight)
    sections = []
    sections.append(
        append_section(payload, f"{name}.qweight_s4", pack_signed_int4(candidate.qweight))
    )
    arrays = (
        (
            "group_multiplier_s32",
            candidate.group_multiplier.detach().cpu().numpy().astype("<i4").tobytes(),
        ),
        (
            "group_right_shift_u8",
            candidate.group_right_shift.detach().cpu().numpy().astype("u1").tobytes(),
        ),
        (
            "common_scale32_u32",
            candidate.native_scale32_records.detach().cpu().numpy().astype("<u4").tobytes(),
        ),
        (
            "output_multiplier_s32",
            candidate.multiplier.detach().cpu().numpy().astype("<i4").tobytes(),
        ),
        (
            "output_right_shift_u8",
            candidate.right_shift.detach().cpu().numpy().astype("u1").tobytes(),
        ),
    )
    for suffix, data in arrays:
        sections.append(append_section(payload, f"{name}.{suffix}", data))
    if candidate.bias_accumulator is not None:
        sections.append(
            append_section(
                payload,
                f"{name}.bias_common_s32",
                candidate.bias_accumulator.detach().cpu().numpy().astype("<i4").tobytes(),
            )
        )
    return {
        "name": name,
        "in_features": candidate.in_features,
        "out_features": candidate.out_features,
        "group_size": candidate.group_size,
        "groups_per_output": candidate.groups,
        "signed_int4_values": candidate.qweight.numel(),
        "packed_int4_bytes": (candidate.qweight.numel() + 1) // 2,
        "activation_bits": 8,
        "group_scale_metadata": "signed-int32 multiplier plus uint8 right shift",
        "common_output_channel_scale_metadata": "Scale32 uint32",
        "rounding": "round_to_nearest_ties_to_even",
        "saturation": "signed_int8_after_output_requantization",
        "baseline_per_output_weight_relative_l2": relative_l2(weight, baseline_weight),
        "candidate_groupwise_weight_relative_l2": relative_l2(weight, candidate_weight),
        "qweight_sha256": tensor_sha256(candidate.qweight),
        "group_multiplier_sha256": tensor_sha256(candidate.group_multiplier),
        "group_right_shift_sha256": tensor_sha256(candidate.group_right_shift),
        "common_scale32_sha256": tensor_sha256(candidate.native_scale32_records),
        "sections": sections,
    }


def install_layer21_repair(
    fixed_model: Any,
    source_model: Any,
    group_size: int,
) -> tuple[list[dict[str, Any]], bytes]:
    fixed_layer = fixed_model.model.layers[LAYER_INDEX]
    source_layer = source_model.model.layers[LAYER_INDEX]
    pairs = (
        ("self_attn.q_proj", fixed_layer.self_attn, source_layer.self_attn, "q_proj"),
        ("self_attn.k_proj", fixed_layer.self_attn, source_layer.self_attn, "k_proj"),
        ("self_attn.v_proj", fixed_layer.self_attn, source_layer.self_attn, "v_proj"),
        ("self_attn.o_proj", fixed_layer.self_attn, source_layer.self_attn, "o_proj"),
        ("mlp.gate_proj", fixed_layer.mlp, source_layer.mlp, "gate_proj"),
        ("mlp.up_proj", fixed_layer.mlp, source_layer.mlp, "up_proj"),
        ("mlp.down_proj", fixed_layer.mlp, source_layer.mlp, "down_proj"),
    )
    payload = bytearray()
    manifest = []
    for suffix, fixed_parent, source_parent, child in pairs:
        baseline = getattr(fixed_parent, child)
        source = getattr(source_parent, child)
        if not isinstance(baseline, fixed_point.W4A8Linear):
            raise TypeError(f"Layer-21 {suffix} is not the frozen W4A8 module")
        candidate = GroupwiseW4A8Linear(source, baseline, group_size)
        manifest.append(
            module_manifest(
                f"model.layers.{LAYER_INDEX}.{suffix}",
                source,
                baseline,
                candidate,
                payload,
            )
        )
        setattr(fixed_parent, child, candidate)
    return manifest, bytes(payload)


def greedy_generate_with_fast_fail(
    model: Any,
    tokenizer: Any,
    prompt_name: str,
    prompt_tokens: list[int],
    termination_token_ids: set[int],
    *,
    max_new_tokens: int,
    top_k_count: int,
    prefix: PrefixSubstitution,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    generated: list[int] = []
    steps = []
    fast_fail = None
    with torch.inference_mode():
        for generation_index in range(max_new_tokens):
            prefix.set_step(prompt_name, generation_index)
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
            candidates = top_k(tokenizer, logits, top_k_count)
            selected = int(candidates[0]["token_id"])
            piece = token_piece(tokenizer, selected)
            generated.append(selected)
            terminated = selected in termination_token_ids
            steps.append(
                {
                    "generation_index": generation_index,
                    "selected_token_id": selected,
                    "selected_piece": piece,
                    "terminated": terminated,
                    "top_k": candidates,
                }
            )
            if generation_index == 0 and not any(
                character.isalpha() for character in piece
            ):
                fast_fail = {
                    "triggered": True,
                    "reason": "first selected token contains no alphabetic character",
                    "selected_token_id": selected,
                    "selected_piece": piece,
                }
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[selected]], dtype=torch.long)], dim=1
            )
            if terminated:
                break
    decoded = decode_generated(tokenizer, generated)
    readability = readability_gate(decoded, generated)
    return {
        "prompt_token_count": len(prompt_tokens),
        "prompt_tokens_sha256": sha256_bytes(
            torch.tensor(prompt_tokens, dtype=torch.int32).numpy().tobytes()
        ),
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "readability_gate": readability,
        "coherence_gate": prompt_coherence(prompt_name, decoded, readability),
        "fast_fail": fast_fail or {"triggered": False},
        "terminated": bool(steps and steps[-1]["terminated"]),
        "steps": steps,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack-output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--second-prompt", default="Write one word: blue")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    started = time.perf_counter()
    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))
    ranges, operator_ranges = frozen_ranges(scales)
    tokenizer = load_tokenizer()
    tokenizations = {
        "hello_world": tokenize_prompt(tokenizer, "Hello world", ""),
        "second_prompt": tokenize_prompt(tokenizer, args.second_prompt, ""),
    }
    source_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_point.replace_linears(
        fixed_model, ranges, rope_diagnostic_mechanism=ROPE_MECHANISM
    )
    fixed_point.replace_fixed_operators(
        fixed_model,
        operator_ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
        all_projection_residual_fusion=True,
    )
    frozen_binding_before_repair = validate_frozen_scale_binding(fixed_model, scales)
    manifest, packed = install_layer21_repair(
        fixed_model, source_model, args.group_size
    )
    pack_output = args.pack_output.resolve()
    pack_output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(pack_output, packed)
    if sha256_file(pack_output) != hashlib.sha256(packed).hexdigest():
        raise RuntimeError("written groupwise-W4 pack hash differs")

    runtime = fixed_model.ace2_all_projection_residual_fusion_runtime
    prefix = PrefixSubstitution(fixed_model, source_model, LAYER_INDEX)
    prefix.install()
    termination_token_ids = {
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    generations = {}
    try:
        for prompt_name, tokenization in tokenizations.items():
            generations[prompt_name] = greedy_generate_with_fast_fail(
                fixed_model,
                tokenizer,
                prompt_name,
                tokenization["chat_template_token_ids"],
                termination_token_ids,
                max_new_tokens=args.max_new_tokens,
                top_k_count=args.top_k,
                prefix=prefix,
            )
            generation = generations[prompt_name]
            print(
                "ACE2_LAYER21_GROUPWISE_PROMPT "
                f"prompt={prompt_name} first={generation['generated_token_ids'][0]} "
                f"decoded={generation['decoded_text']!r} "
                f"readable={generation['readability_gate']['passed']} "
                f"semantic={generation['coherence_gate']['passed']} "
                f"fast_fail={generation['fast_fail']['triggered']}",
                flush=True,
            )
    finally:
        prefix.remove()

    execution = summarize_execution(runtime)
    readable = all(
        generation["readability_gate"]["passed"]
        for generation in generations.values()
    )
    semantic = all(
        generation["coherence_gate"]["passed"]
        for generation in generations.values()
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_layer21_groupwise_w4a8_repair_candidate",
        "contract": {
            "acceptance_candidate": False,
            "hardware_realizable_repair_candidate": True,
            "conditioned_exact_prefix_layers": 21,
            "repaired_layer": LAYER_INDEX,
            "repaired_projections": 7,
            "weight_format": "signed_int4_two_values_per_byte",
            "activation_format": "signed_int8_frozen_scales",
            "weight_group_size_inputs": args.group_size,
            "group_arithmetic": (
                "Each signed-int4 group dot product is signed-32. A signed-int32 "
                "multiplier and uint8 right shift convert it with round-to-nearest "
                "ties-to-even into one per-output-channel Scale32 common domain. "
                "Group contributions sum in signed-32, then the unchanged output "
                "requantizer and one-round residual fusion clamp to signed-int8."
            ),
            "readability_and_semantic_gates_are_independent": True,
            "fast_fail_rule": (
                "Stop a prompt after generation step 0 when the selected token piece "
                "contains no alphabetic character."
            ),
            "scope_limit": (
                "This tests the repaired Layer-21 W4A8 bundle on exact Layers 0 through "
                "20. It is not an all-W4A8 acceptance result."
            ),
        },
        "invocation": {"argv": exact_process_argv(), "cwd": str(Path.cwd())},
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": REVISION,
            "snapshot": str(TOKENIZER_SNAPSHOT.resolve()),
        },
        "prompts": {
            "hello_world": "Hello world",
            "second_prompt": args.second_prompt,
        },
        "repair_manifest": manifest,
        "pack": {
            "path": str(pack_output),
            "sha256": sha256_file(pack_output),
            "bytes": pack_output.stat().st_size,
            "section_count": sum(len(item["sections"]) for item in manifest),
        },
        "generations": generations,
        "frozen_binding_before_repair": frozen_binding_before_repair,
        "shadow_residual_execution": execution,
        "two_prompt_readability_passed": readable,
        "two_prompt_semantic_gate_passed": semantic,
        "status": (
            "PASS_LAYER21_GROUPWISE_W4A8_TWO_PROMPT_READABILITY"
            if readable
            else "FAIL_LAYER21_GROUPWISE_W4A8_TWO_PROMPT_READABILITY"
        ),
        "semantic_status": (
            "PASS_LAYER21_GROUPWISE_W4A8_TWO_PROMPT_SEMANTIC_GATE"
            if semantic
            else "FAIL_LAYER21_GROUPWISE_W4A8_TWO_PROMPT_SEMANTIC_GATE"
        ),
        "artifacts": {
            "prefix21": {"path": str(PREFIX21), "sha256": sha256_file(PREFIX21)},
            "prefix22": {"path": str(PREFIX22), "sha256": sha256_file(PREFIX22)},
            "frozen_scales": {
                "path": str(FROZEN_SCALES),
                "sha256": sha256_file(FROZEN_SCALES),
            },
            "fixed_point_source": {
                "path": str(ROOT / "tools/ace2_full_model_fixed_point.py"),
                "sha256": sha256_file(ROOT / "tools/ace2_full_model_fixed_point.py"),
            },
            "runner_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "wall_seconds": time.perf_counter() - started,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, canonical_bytes(result))
    print(
        "ACE2_LAYER21_GROUPWISE_RESULT "
        f"status={result['status']} semantic_status={result['semantic_status']} "
        f"wall_seconds={result['wall_seconds']:.6f} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_LAYER21_GROUPWISE_FAIL detail={error}", file=sys.stderr)
        raise
