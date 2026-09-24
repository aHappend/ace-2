#!/usr/bin/env python3
"""Freeze and run the static position-class grouped-A8 Option-B attempt.

The candidate keeps the authenticated signed-W4 per-output-row weight rule and
uses immutable Scale32 tables for three activation position classes: sink
(absolute position zero), prompt/prefill, and decode.  Runtime values may be
range-checked, but they never select, calibrate, or emit scale records.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

import run_option_b_grouped_scale32_stage1 as grouped
from discriminate_qwen_instruct_w4_weight_policies import PROMPTS, chat_input
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    TERMINATION_TOKEN_IDS,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from qwen_instruct_response_gate import evaluate as evaluate_response_gate
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM
from run_all_transformer_per32_stage1 import (
    aggregate_prompts,
    generate_reference,
    log_message,
    logit_comparison,
    prompt_binding,
    sequence_disagreements,
    utc_now,
    write_json,
)


CANDIDATE_ID = "option_b_static_position_class_group_scale_w4a8_v1"
MISSION_ID = "05f90220ffd3"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
SUCCESSOR_PATH = ROOT / "design/OPTION_B_STATIC_POSITION_CLASS_GROUP_SCALE_W4A8_V1_CONTRACT.json"
SUCCESSOR_COMPANION = ROOT / "design/OPTION_B_STATIC_POSITION_CLASS_GROUP_SCALE_W4A8_V1_CONTRACT.sha256"
CALIBRATION_CONTRACT_PATH = CALIBRATION_DIR / "calibration_contract.json"
CALIBRATION_REPORT_PATH = CALIBRATION_DIR / "calibration_report.json"
OUTPUT_DIR = ROOT / "build/stage1-option-b-static-position-class-group-scale-w4a8-v1"
TABLE_PATH = OUTPUT_DIR / "static_position_class_scale_table.json"
TABLE_COMPANION = OUTPUT_DIR / "static_position_class_scale_table.sha256"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
RUNNER_PATH = Path(__file__).resolve()
GROUPED_UTILITY_PATH = ROOT / "tools/run_option_b_grouped_scale32_stage1.py"
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16
LAYERS = 24
POSITION_CLASSES = ("sink", "prompt", "decode")
EVENT_SPECS: dict[str, tuple[int, int, str]] = {
    "input_rmsnorm": (128, 7, "bsc"),
    "q_proj": (64, 14, "bsc"),
    "k_proj": (64, 2, "bsc"),
    "v_proj": (64, 2, "bsc"),
    "rope_q": (64, 14, "bhsc"),
    "rope_k": (64, 2, "bhsc"),
    "attention_value": (128, 7, "bsc"),
    "o_proj": (128, 7, "bsc"),
    "post_attention_rmsnorm": (128, 7, "bsc"),
    "mlp_gate_proj": (128, 38, "bsc"),
    "mlp_up_proj": (128, 38, "bsc"),
    "silu_gate": (128, 38, "bsc"),
    "mlp_down_proj": (128, 7, "bsc"),
}
GROUPS_PER_LAYER = sum(spec[1] for spec in EVENT_SPECS.values())


class StaticScaleOverflow(RuntimeError):
    """A frozen table does not cover an observed runtime activation."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    ).hexdigest()


def require_project_python() -> None:
    grouped.require_project_python()


def event_name(layer: int, family: str) -> str:
    return f"layer_{layer}.{family}"


def event_family(name: str) -> str:
    return name.split(".", 1)[1]


def canonicalize(value: Tensor, layout: str) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    if layout == "bsc":
        require(value.ndim == 3, "BSC activation rank differs")
        return value, lambda restored: restored
    require(layout == "bhsc", "unknown activation layout")
    require(value.ndim == 4, "BHSC activation rank differs")
    batch, heads, sequence, lanes = value.shape
    canonical = value.permute(0, 2, 1, 3).contiguous().reshape(batch, sequence, heads * lanes)

    def restore(restored: Tensor) -> Tensor:
        return restored.reshape(batch, sequence, heads, lanes).permute(0, 2, 1, 3).contiguous()

    return canonical, restore


def class_slices(sequence: int, prompt_length: int) -> dict[str, tuple[int, int]]:
    require(1 <= prompt_length <= sequence, "prompt length is outside activation sequence")
    return {
        "sink": (0, 1),
        "prompt": (1, prompt_length),
        "decode": (prompt_length, sequence),
    }


def static_policy_cost() -> dict[str, Any]:
    weight_cost = grouped.policy_cost()
    records_per_layer = GROUPS_PER_LAYER * len(POSITION_CLASSES)
    table_bytes_per_layer = records_per_layer * 4
    table_bytes_total = table_bytes_per_layer * LAYERS
    require(GROUPS_PER_LAYER == 183, "static group count per layer differs")
    require(records_per_layer == 549, "static Scale32 records per layer differs")
    require(table_bytes_per_layer == 2_196, "static table bytes per layer differs")
    return {
        **weight_cost,
        "activation_tensor_sidecars_per_decoder_layer_token": 0,
        "activation_transport_metadata_bytes_per_decoder_layer_token": 0,
        "static_position_classes": list(POSITION_CLASSES),
        "static_activation_group_records_per_decoder_layer": records_per_layer,
        "static_activation_table_bytes_per_decoder_layer": table_bytes_per_layer,
        "static_activation_table_bytes_all_layers": table_bytes_total,
        "maximum_live_static_table_bytes": table_bytes_per_layer,
        "maximum_static_table_stream_cycles_at_128_bits": math.ceil(table_bytes_per_layer / 16),
    }


class BoundaryHooks:
    def __init__(self) -> None:
        self.handles: list[Any] = []
        self.original_rope: Callable[..., Any] | None = None
        self.active_layer: int | None = None

    def _install_common(
        self,
        model: nn.Module,
        output_callback: Callable[[str, Tensor, str], Tensor],
        input_callback: Callable[[str, Tensor, str], Tensor],
        rope_callback: Callable[[str, Tensor, str], Tensor],
    ) -> None:
        require(self.original_rope is None, "boundary hooks are already installed")
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.input_layernorm.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "input_rmsnorm"), output, "bsc"
                    )
                )
            )
            self.handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "q_proj"), output, "bsc"
                    )
                )
            )
            self.handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "k_proj"), output, "bsc"
                    )
                )
            )
            self.handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "v_proj"), output, "bsc"
                    )
                )
            )

            def set_layer(_module: nn.Module, _inputs: tuple[Any, ...], index: int = layer_index) -> None:
                require(self.active_layer is None, "nested Qwen attention activation is unsupported")
                self.active_layer = index

            def clear_layer(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                index: int = layer_index,
            ) -> Any:
                require(self.active_layer == index, "active attention layer tracking differs")
                self.active_layer = None
                return output

            self.handles.append(layer.self_attn.register_forward_pre_hook(set_layer))
            self.handles.append(layer.self_attn.register_forward_hook(clear_layer))
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    lambda _m, inputs, index=layer_index: (
                        input_callback(event_name(index, "attention_value"), inputs[0], "bsc"),
                        *inputs[1:],
                    )
                )
            )
            self.handles.append(
                layer.self_attn.o_proj.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "o_proj"), output, "bsc"
                    )
                )
            )
            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "post_attention_rmsnorm"), output, "bsc"
                    )
                )
            )
            self.handles.append(
                layer.mlp.gate_proj.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "mlp_gate_proj"), output, "bsc"
                    )
                )
            )
            self.handles.append(
                layer.mlp.up_proj.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "mlp_up_proj"), output, "bsc"
                    )
                )
            )
            self.handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    lambda _m, inputs, index=layer_index: (
                        input_callback(event_name(index, "silu_gate"), inputs[0], "bsc"),
                        *inputs[1:],
                    )
                )
            )
            self.handles.append(
                layer.mlp.down_proj.register_forward_hook(
                    lambda _m, _i, output, index=layer_index: output_callback(
                        event_name(index, "mlp_down_proj"), output, "bsc"
                    )
                )
            )

        self.original_rope = modeling_qwen2.apply_rotary_pos_emb

        def static_rope(
            q: Tensor,
            k: Tensor,
            cos: Tensor,
            sin: Tensor,
            position_ids: Any = None,
            unsqueeze_dim: int = 1,
        ) -> tuple[Tensor, Tensor]:
            require(self.original_rope is not None, "original RoPE function is missing")
            require(self.active_layer is not None, "RoPE executed outside a tracked attention layer")
            q_embed, k_embed = self.original_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
            layer_index = self.active_layer
            return (
                rope_callback(event_name(layer_index, "rope_q"), q_embed, "bhsc"),
                rope_callback(event_name(layer_index, "rope_k"), k_embed, "bhsc"),
            )

        modeling_qwen2.apply_rotary_pos_emb = static_rope

    def uninstall(self) -> None:
        if self.original_rope is not None:
            modeling_qwen2.apply_rotary_pos_emb = self.original_rope
            self.original_rope = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active_layer = None


class CalibrationObserver(BoundaryHooks):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_length: int | None = None
        self.calls: Counter[str] = Counter()
        self.maxima: dict[str, dict[str, Tensor]] = {}
        self.observed_positions: Counter[tuple[str, str]] = Counter()

    def begin_forward(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length

    def observe(self, name: str, value: Tensor, layout: str) -> Tensor:
        require(self.prompt_length is not None, "calibration prompt length is unset")
        canonical, _restore = canonicalize(value, layout)
        family = event_family(name)
        lanes, groups, expected_layout = EVENT_SPECS[family]
        require(layout == expected_layout, f"calibration layout differs: {name}")
        require(canonical.shape[-1] == lanes * groups, f"calibration width differs: {name}")
        slices = class_slices(int(canonical.shape[1]), self.prompt_length)
        event_maxima = self.maxima.setdefault(name, {})
        for position_class, (start, stop) in slices.items():
            if start == stop:
                continue
            selected = canonical[:, start:stop].detach().to(torch.float64)
            flat = selected.reshape(-1, groups, lanes)
            maxima = flat.abs().amax(dim=(0, 2)).cpu()
            if position_class in event_maxima:
                event_maxima[position_class] = torch.maximum(
                    event_maxima[position_class], maxima
                )
            else:
                event_maxima[position_class] = maxima
            self.observed_positions[(name, position_class)] += int(selected.shape[0] * selected.shape[1])
        self.calls[name] += 1
        return value

    def install(self, model: nn.Module) -> None:
        self._install_common(model, self.observe, self.observe, self.observe)

    def table_events(self) -> dict[str, Any]:
        expected = {
            event_name(layer, family)
            for layer in range(LAYERS)
            for family in EVENT_SPECS
        }
        require(set(self.calls) == expected, "calibration event closure differs")
        events: dict[str, Any] = {}
        for name in sorted(expected):
            family = event_family(name)
            lanes, groups, layout = EVENT_SPECS[family]
            classes: dict[str, Any] = {}
            for position_class in POSITION_CLASSES:
                require(
                    position_class in self.maxima[name],
                    f"calibration class was not observed: {name}:{position_class}",
                )
                maxima = self.maxima[name][position_class]
                require(maxima.numel() == groups, f"calibration group count differs: {name}")
                require(bool(torch.all(maxima > 0)), f"all-zero calibration group: {name}:{position_class}")
                records = [
                    grouped.scale32_record(float(maximum) / 127.0)
                    for maximum in maxima.tolist()
                ]
                scales = [grouped.scale32_value(record) for record in records]
                require(
                    all(float(maximum) <= 127.0 * scale for maximum, scale in zip(maxima.tolist(), scales)),
                    f"Scale32 ceiling does not cover calibration maximum: {name}:{position_class}",
                )
                classes[position_class] = {
                    "calibration_absmax": [float(value) for value in maxima.tolist()],
                    "scale32_records": records,
                    "scale_values": scales,
                    "observed_position_vectors": self.observed_positions[(name, position_class)],
                }
            events[name] = {
                "family": family,
                "layout": layout,
                "group_lanes": lanes,
                "group_count": groups,
                "classes": classes,
            }
        return events


class StaticActivationPolicy(BoundaryHooks):
    def __init__(self, table: dict[str, Any]) -> None:
        super().__init__()
        self.table = table
        self.prompt_length: int | None = None
        self.calls: Counter[str] = Counter()
        self.class_position_vectors: Counter[str] = Counter()
        self.payload_hash = hashlib.sha256()
        self.runtime_range_check_count = 0

    def begin_forward(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length

    def apply(self, name: str, value: Tensor, layout: str) -> Tensor:
        require(self.prompt_length is not None, "candidate prompt length is unset")
        record = self.table["events"][name]
        require(record["layout"] == layout, f"static table layout differs: {name}")
        lanes = int(record["group_lanes"])
        groups = int(record["group_count"])
        canonical, restore = canonicalize(value, layout)
        require(canonical.shape[-1] == lanes * groups, f"candidate width differs: {name}")
        output = torch.empty_like(canonical, dtype=torch.float64)
        slices = class_slices(int(canonical.shape[1]), self.prompt_length)
        for position_class, (start, stop) in slices.items():
            if start == stop:
                continue
            class_record = record["classes"][position_class]
            records = [int(item) for item in class_record["scale32_records"]]
            require(len(records) == groups, f"static record count differs: {name}:{position_class}")
            scales = torch.tensor(
                [grouped.scale32_value(item) for item in records],
                dtype=torch.float64,
                device=value.device,
            )
            selected = canonical[:, start:stop].detach().to(torch.float64)
            flat = selected.reshape(-1, groups, lanes)
            maxima = flat.abs().amax(dim=(0, 2))
            limits = scales * 127.0
            self.runtime_range_check_count += groups
            overflow = maxima > limits
            if bool(torch.any(overflow)):
                group_index = int(torch.nonzero(overflow, as_tuple=False)[0])
                raise StaticScaleOverflow(
                    f"fixed table overflow event={name} class={position_class} "
                    f"group={group_index} maximum={float(maxima[group_index])} "
                    f"limit={float(limits[group_index])}"
                )
            quantized = torch.round(flat / scales[None, :, None])
            require(
                bool(torch.all(quantized >= -127) and torch.all(quantized <= 127)),
                f"static activation escaped symmetric A8: {name}:{position_class}",
            )
            q8 = quantized.to(torch.int8)
            require(not bool(torch.any(q8 == -128)), "reserved signed-A8 value was produced")
            dequantized = q8.to(torch.float64) * scales[None, :, None]
            output[:, start:stop] = dequantized.reshape_as(selected)
            self.class_position_vectors[position_class] += int(selected.shape[0] * selected.shape[1])
        payload = torch.round(output.reshape(-1) / 1.0).detach().cpu().numpy().astype("<f8", copy=False)
        self.payload_hash.update(name.encode() + b"\0" + payload.tobytes(order="C"))
        self.calls[name] += 1
        return restore(output).to(value.dtype)

    def install(self, model: nn.Module) -> None:
        self._install_common(model, self.apply, self.apply, self.apply)

    def summary(self, model_forward_count: int) -> dict[str, Any]:
        expected = {
            event_name(layer, family)
            for layer in range(LAYERS)
            for family in EVENT_SPECS
        }
        require(set(self.calls) == expected, "activation event name closure differs")
        require(
            set(self.calls.values()) == {model_forward_count},
            "activation event call counts differ across boundaries",
        )
        return {
            "model_forward_count": model_forward_count,
            "named_boundary_count": len(expected),
            "events_per_model_forward": len(expected),
            "event_call_count": sum(self.calls.values()),
            "position_classes": list(POSITION_CLASSES),
            "class_position_vectors_across_all_boundaries": dict(sorted(self.class_position_vectors.items())),
            "runtime_range_check_count": self.runtime_range_check_count,
            "runtime_scale_selection_count": 0,
            "dynamic_calibration_count": 0,
            "per_token_scale_record_emission_count": 0,
            "per_token_scale_sidecar_bytes": 0,
            "reserved_minus_128_produced": False,
            "fixed_table": file_record(TABLE_PATH),
            "dequantized_payload_stream_sha256": self.payload_hash.hexdigest(),
            "calls": dict(sorted(self.calls.items())),
        }


def static_quantizer_self_test() -> dict[str, Any]:
    records = {
        "sink": [grouped.scale32_record(0.25)],
        "prompt": [grouped.scale32_record(0.125)],
        "decode": [grouped.scale32_record(0.0625)],
    }
    values = torch.tensor([[[31.75, -31.75], [15.875, -15.875], [7.9375, -7.9375]]])
    output: list[list[int]] = []
    for position_class, vector in zip(POSITION_CLASSES, values[0]):
        scale = grouped.scale32_value(records[position_class][0])
        q8 = torch.round(vector.to(torch.float64) / scale).to(torch.int8)
        output.append([int(item) for item in q8.tolist()])
    require(output == [[127, -127], [127, -127], [127, -127]], "static self-test rounding differs")
    overflow_caught = False
    try:
        scale = grouped.scale32_value(records["decode"][0])
        if 8.0 > 127.0 * scale:
            raise StaticScaleOverflow("self-test")
    except StaticScaleOverflow:
        overflow_caught = True
    require(overflow_caught, "static overflow self-test did not fail closed")
    return {
        "status": "PASS",
        "position_classes": list(POSITION_CLASSES),
        "rounding": "round_to_nearest_ties_to_even",
        "runtime_scale_selection_count": 0,
        "overflow_fail_closed": True,
        "payload_sha256": sha256_bytes(canonical_bytes(output)),
    }


def calibration_inputs(tokenizer: Any) -> tuple[list[Tensor], list[dict[str, Any]]]:
    contract = load_json(CALIBRATION_CONTRACT_PATH)
    records = contract["chat"]["prompts"]
    require(len(records) == 4, "calibration prompt count differs")
    prompts: list[Tensor] = []
    for record in records:
        ids = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        require(isinstance(ids, Tensor), "calibration tokenization did not return a tensor")
        require([int(item) for item in ids[0].tolist()] == record["token_ids"], "calibration token IDs differ")
        prompts.append(ids)
    return prompts, records


def observe_calibration_generation(
    model: nn.Module,
    input_ids: Tensor,
    observer: CalibrationObserver,
) -> list[int]:
    prompt_length = int(input_ids.shape[1])
    prefix = input_ids
    generated: list[int] = []
    with torch.inference_mode():
        for _index in range(MAX_NEW_TOKENS):
            observer.begin_forward(prompt_length)
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1]
            token = int(logits.argmax())
            generated.append(token)
            prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
            if token in TERMINATION_TOKEN_IDS:
                break
        observer.begin_forward(prompt_length)
        model(input_ids=prefix, use_cache=False)
    return generated


def prepare_table() -> dict[str, Any]:
    require_project_python()
    require(not ATTEMPT_DIR.exists(), "cannot calibrate a table after attempt creation")
    require(not TABLE_PATH.exists(), "static scale table already exists")
    verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "matrix hash differs")
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    prompts, prompt_records = calibration_inputs(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    observer = CalibrationObserver()
    observer.install(model)
    generated: list[dict[str, Any]] = []
    try:
        for record, input_ids in zip(prompt_records, prompts):
            token_ids = observe_calibration_generation(model, input_ids, observer)
            generated.append(
                {
                    "ordinal": record["ordinal"],
                    "input_token_ids_sha256": record["token_ids_sha256"],
                    "generated_token_ids": token_ids,
                    "generated_token_ids_sha256": sha256_bytes(
                        b"".join(int(value).to_bytes(4, "little") for value in token_ids)
                    ),
                }
            )
        events = observer.table_events()
    finally:
        observer.uninstall()
    del model
    gc.collect()
    table = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "classification": "offline_fixed_layer_operator_group_a8_scale_table",
        "created_at_utc": utc_now(),
        "derivation": {
            "source": "BF16 activations from the four immutable non-evaluator calibration prompts",
            "position_classes": {
                "sink": "absolute sequence position zero",
                "prompt": "positions one through original prompt length minus one",
                "decode": "positions at or beyond the original prompt length",
            },
            "group_scale_rule": "ceil normalized Scale32 of maximum absolute BF16 calibration value divided by 127",
            "rounding": "round_to_nearest_ties_to_even",
            "safety_margin_or_percentile": None,
            "official_nine_prompt_inputs_used_for_calibration": False,
            "runtime_dynamic_calibration_permitted": False,
            "runtime_scale_selection_permitted": False,
            "per_token_scale_sidecars_permitted": False,
        },
        "geometry": {
            "decoder_layers": LAYERS,
            "event_families": list(EVENT_SPECS),
            "groups_per_layer": GROUPS_PER_LAYER,
            "position_classes": list(POSITION_CLASSES),
            "records_per_layer": GROUPS_PER_LAYER * len(POSITION_CLASSES),
            "bytes_per_layer": GROUPS_PER_LAYER * len(POSITION_CLASSES) * 4,
            "bytes_all_layers": GROUPS_PER_LAYER * len(POSITION_CLASSES) * 4 * LAYERS,
        },
        "source_model": source_identity,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "offline_only": True,
        },
        "calibration_contract": file_record(CALIBRATION_CONTRACT_PATH),
        "calibration_report": file_record(CALIBRATION_REPORT_PATH),
        "calibration_generated_sequences": generated,
        "events": events,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(TABLE_PATH, table)
    TABLE_COMPANION.write_text(
        f"{sha256_file(TABLE_PATH)}  {TABLE_PATH.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )
    return table


def successor_contract() -> dict[str, Any]:
    cost = static_policy_cost()
    return {
        "schema_version": 1,
        "contract_id": CANDIDATE_ID,
        "candidate_id": CANDIDATE_ID,
        "selected_policy_id": None,
        "status": "authorized_bounded_successor_pending_execution_and_fresh_review",
        "authority": f"operator_objective_{MISSION_ID}",
        "created_at_utc": utc_now(),
        "model": {
            "repository": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "identity_manifest": file_record(
                IMAGE_DIR / "option_b_identity_manifest.json"
            ),
        },
        "policy_basis": {
            "weight_policy": "authenticated signed-W4 payload with one normalized 32-bit Scale32 record per output row",
            "activation_policy": "signed-A8 payload with immutable layer/operator/group Scale32 tables selected only by sink, prompt, or decode position class",
            "structurally_distinct_from_consumed_grouped_policy": "no runtime maximum detector, exponent selection, per-token delta, tensor sidecar, or KV scale rewrite",
            "table": file_record(TABLE_PATH),
        },
        "limits": {
            "bits_per_weight": {
                "payload_bits": 4,
                "scale_record_bits": 32,
                "minimum_weights_sharing_one_scale": 896,
                "maximum_metadata_bits_per_weight": 0.036,
                "maximum_total_stored_bits_per_weight": 4.036,
                "standard_decoder_layer_exact_bits_per_weight": 4.0271978021978025,
                "per_scalar_weight_metadata_forbidden": True,
            },
            "activation_metadata": {
                "activation_payload_bits": 8,
                "position_classes": list(POSITION_CLASSES),
                "group_lanes": [64, 128],
                "static_scale32_records_per_decoder_layer": cost["static_activation_group_records_per_decoder_layer"],
                "static_table_bytes_per_decoder_layer": cost["static_activation_table_bytes_per_decoder_layer"],
                "static_table_bytes_all_layers": cost["static_activation_table_bytes_all_layers"],
                "maximum_incremental_metadata_bytes_per_decoder_layer_token": 0,
                "per_token_or_per_tensor_sidecars_forbidden": True,
                "runtime_dynamic_calibration_forbidden": True,
            },
            "bandwidth": {
                "abstract_streaming_memory_boundary_bits": 128,
                "maximum_payload_bytes_per_cycle": 16,
                "standard_decoder_layer_weight_stream_bytes": 7_505_408,
                "maximum_static_scale_table_stream_bytes_per_layer_load": cost["static_activation_table_bytes_per_decoder_layer"],
                "maximum_static_scale_table_stream_cycles_per_layer_load": cost["maximum_static_table_stream_cycles_at_128_bits"],
                "maximum_incremental_external_activation_metadata_bytes_per_decoder_layer_token": 0,
                "unbounded_backpressure_remains_legal": True,
                "throughput_or_latency_claimed": False,
            },
            "rtl_realizability": {
                "synthesizable_integer_only": True,
                "floating_point_rtl_permitted": False,
                "new_public_ace2_shell_ports_permitted": False,
                "position_class_selection": "absolute_position_zero_else_prefill_command_else_decode_command",
                "maximum_group_lanes": 128,
                "maximum_group_records_per_event": 38,
                "maximum_live_static_table_sram_bytes": cost["maximum_live_static_table_bytes"],
                "runtime_maximum_detector_or_dynamic_exponent_selector_required": False,
                "non_sram_area_cap_mm2": 2.0,
                "frequency_floor_mhz": 100.0,
                "required_rounding": "round_to_nearest_ties_to_even",
                "saturation_or_silent_clipping_permitted": False,
                "fresh_executable_rtl_verification_required_before_capability_acceptance": True,
                "fresh_canonical_sky130_synthesis_opensta_required_before_capability_acceptance": True,
                "fresh_source_and_constraint_hashes_required_before_capability_acceptance": True,
                "independent_reviewer_acceptance_required_before_capability_acceptance": True,
            },
        },
        "candidate_scope": {
            "decoder_layers": {"first": 0, "last": 23, "count": 24},
            "transformer_linear_weight_tensors": 168,
            "activation_tensor_families": list(EVENT_SPECS),
            "final_rmsnorm_and_lm_head": "authenticated baseline arithmetic and W4 policy remain unchanged",
            "embedding_tokenizer_prompt_matrix_and_response_gate": "frozen unchanged",
            "dataset_prompt_token_or_gate_control_permitted": False,
            "alternate_policy_or_scope_substitution_permitted": False,
        },
        "attempt_budget": {
            "exact_authorized_candidate_attempts": 1,
            "attempt_id": "attempt-0001",
            "attempt_directory": ATTEMPT_DIR.relative_to(ROOT).as_posix(),
            "frozen_matrix": {
                "path": MATRIX_PATH.relative_to(ROOT).as_posix(),
                "sha256": MATRIX_SHA256,
                "prompt_count": 9,
            },
            "generation": "deterministic greedy argmax; no sampling; use_cache=false; maximum six generated tokens; termination IDs 151643 and 151645",
            "score_policy": "binary only: 9/9 complete token arrays exactly equal BF16 and 9/9 response-gate outcomes equal BF16",
            "runner_hash_must_be_frozen_before_execution": True,
            "replay_tuning_ranking_bisection_sweep_or_gate_relaxation_permitted": False,
            "network_access_permitted": False,
            "rtl_demo_ppa_u280_or_stage_transition_permitted": False,
            "selected_policy_id_must_remain_null_until_fresh_reviewer_adjudication": True,
            "on_evaluator_no_execution": "classify separately and draw no numerical correctness conclusion",
        },
        "hash_bindings": {
            "static_scale_table": file_record(TABLE_PATH),
            "calibration_contract": file_record(CALIBRATION_CONTRACT_PATH),
            "source_contract": file_record(ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"),
            "matrix": file_record(MATRIX_PATH),
            "runner": file_record(RUNNER_PATH),
            "grouped_weight_utility": file_record(GROUPED_UTILITY_PATH),
            "fixed_point_reference": file_record(ROOT / "tools/ace2_full_model_fixed_point.py"),
            "quality_contracts": file_record(ROOT / "tools/ace2_quality_contracts.py"),
            "oracle": file_record(ROOT / "tools/qwen_instruct_w4a8_oracle.py"),
            "response_gate": file_record(ROOT / "tools/qwen_instruct_response_gate.py"),
        },
        "claim_boundary": {
            "product_policy_accepted": False,
            "numerical_attempt_executed": False,
            "selected_policy_id": None,
            "rtl_capability_accepted": False,
            "ppa_or_timing_accepted": False,
            "demo_retargeted": False,
            "u280_or_stage2_authorized": False,
            "current_stage_remains": "specification",
        },
    }


def contract_artifacts() -> dict[str, Any]:
    paths = {
        "successor_contract": SUCCESSOR_PATH,
        "successor_contract_companion": SUCCESSOR_COMPANION,
        "static_scale_table": TABLE_PATH,
        "static_scale_table_companion": TABLE_COMPANION,
        "matrix": MATRIX_PATH,
        "source_contract": ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json",
        "calibration_contract": CALIBRATION_CONTRACT_PATH,
        "calibration_report": CALIBRATION_REPORT_PATH,
        "full_model_image": IMAGE_DIR / "full_model_image.bin",
        "image_manifest": IMAGE_DIR / "manifest.json",
        "identity_manifest": IMAGE_DIR / "option_b_identity_manifest.json",
        "oracle_contract": IMAGE_DIR / "oracle_contract.json",
        "response_gate_contract": IMAGE_DIR / "response_gate_contract.json",
        "runner_source": RUNNER_PATH,
        "grouped_weight_utility": GROUPED_UTILITY_PATH,
        "shared_evaluator_source": ROOT / "tools/run_all_transformer_per32_stage1.py",
        "prompt_source": ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py",
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "oracle_source": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
    }
    return {name: file_record(path) for name, path in paths.items()}


def prepare_contract() -> dict[str, Any]:
    require_project_python()
    require(TABLE_PATH.is_file(), "static scale table is missing")
    require(not ATTEMPT_DIR.exists(), "cannot prepare contract after attempt creation")
    require(not SUCCESSOR_PATH.exists(), "successor contract already exists")
    require(not CONTRACT_PATH.exists(), "predeclared contract already exists")
    verify_source_contract()
    verify_source_snapshot()
    versions = verify_versions()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "matrix hash differs")
    table_companion = TABLE_COMPANION.read_text(encoding="utf-8").strip()
    require(
        table_companion == f"{sha256_file(TABLE_PATH)}  {TABLE_PATH.relative_to(ROOT).as_posix()}",
        "static table companion differs",
    )
    successor = successor_contract()
    successor_wrapper = {
        "contract": successor,
        "contract_sha256": canonical_sha256(successor),
    }
    write_json(SUCCESSOR_PATH, successor_wrapper)
    SUCCESSOR_COMPANION.write_text(
        f"{sha256_file(SUCCESSOR_PATH)}  {SUCCESSOR_PATH.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )
    matrix = load_json(MATRIX_PATH)
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "authority": {
            "successor_contract": file_record(SUCCESSOR_PATH),
            "successor_contract_sha256": successor_wrapper["contract_sha256"],
            "exact_authorized_candidate_attempts": 1,
            "attempt_id": "attempt-0001",
            "selected_policy_id_before_execution": None,
        },
        "implementation": {
            "classification": "host_numerical_policy_reference_not_cycle_accurate_rtl",
            "weights": "all 168 transformer linears use signed-W4 with one normalized Scale32 record per output row; lm_head retains the authenticated baseline W4 rule",
            "activation_boundaries": list(EVENT_SPECS),
            "position_classes": list(POSITION_CLASSES),
            "grouping": {"q_k_v_and_rope": 64, "hidden_and_mlp": 128},
            "scale_source": file_record(TABLE_PATH),
            "runtime_scale_rule": "lookup only by layer, operator, group, and sink/prompt/decode class",
            "dynamic_calibration": False,
            "per_token_scale_sidecars": False,
            "overflow": "fail closed with evaluator-no-execution classification; no saturation or scale mutation",
            "rounding": "round_to_nearest_ties_to_even",
            "operator_arithmetic": "authenticated local Qwen BF16 host operators with frozen W4-dequantized weights and explicit fixed-table grouped-A8 boundary quantization; no RTL, latency, bandwidth, or cycle claim",
        },
        "cost": static_policy_cost(),
        "generation_and_evaluator": {
            "prompts": prompt_binding(matrix),
            "greedy_argmax": True,
            "sampling": False,
            "use_cache": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
            "eligibility": "9/9 complete token arrays exactly equal BF16 and 9/9 response-gate outcomes equal BF16",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
            "torch_num_threads": TORCH_THREADS,
        },
        "execution": {
            "exact_command": [
                "./.venv/bin/python",
                "tools/run_option_b_static_position_class_group_scale_stage1.py",
                "execute",
            ],
            "runner_sha256_frozen_before_execution": sha256_file(RUNNER_PATH),
            "network_access_permitted": False,
            "replay_tuning_ranking_bisection_sweep_or_gate_relaxation_permitted": False,
            "rtl_demo_ppa_u280_or_stage_transition_permitted": False,
            "expected_outputs": [
                f"{ATTEMPT_DIR.relative_to(ROOT).as_posix()}/execution_started.json",
                f"{ATTEMPT_DIR.relative_to(ROOT).as_posix()}/run.log",
                f"{ATTEMPT_DIR.relative_to(ROOT).as_posix()}/results.json",
                f"{ATTEMPT_DIR.relative_to(ROOT).as_posix()}/fresh_reviewer_input.json",
                f"{ATTEMPT_DIR.relative_to(ROOT).as_posix()}/SHA256SUMS",
                f"{ATTEMPT_DIR.relative_to(ROOT).as_posix()}/SHA256SUMS.sha256",
            ],
        },
        "quantizer_self_test": static_quantizer_self_test(),
        "artifacts": contract_artifacts(),
    }
    wrapper = {"contract": contract, "contract_sha256": canonical_sha256(contract)}
    write_json(CONTRACT_PATH, wrapper)
    return wrapper


def verify_successor_contract() -> dict[str, Any]:
    wrapper = load_json(SUCCESSOR_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "successor wrapper shape differs")
    contract = wrapper["contract"]
    require(wrapper["contract_sha256"] == canonical_sha256(contract), "successor canonical hash differs")
    require(contract["candidate_id"] == CANDIDATE_ID, "successor candidate differs")
    require(contract["selected_policy_id"] is None, "policy selected before review")
    require(contract["attempt_budget"]["exact_authorized_candidate_attempts"] == 1, "attempt budget differs")
    require(contract["attempt_budget"]["frozen_matrix"]["sha256"] == MATRIX_SHA256, "matrix binding differs")
    require(contract["attempt_budget"]["network_access_permitted"] is False, "network policy differs")
    require(contract["claim_boundary"]["numerical_attempt_executed"] is False, "contract already claims execution")
    companion = SUCCESSOR_COMPANION.read_text(encoding="utf-8").strip()
    require(
        companion == f"{sha256_file(SUCCESSOR_PATH)}  {SUCCESSOR_PATH.relative_to(ROOT).as_posix()}",
        "successor companion differs",
    )
    for binding in contract["hash_bindings"].values():
        path = ROOT / binding["path"]
        require(path.is_file() and sha256_file(path) == binding["sha256"], f"successor hash binding differs: {binding['path']}")
    return wrapper


def load_verified_contract(require_unconsumed: bool) -> dict[str, Any]:
    require_project_python()
    require(CONTRACT_PATH.is_file(), "predeclared contract is missing")
    wrapper = load_json(CONTRACT_PATH)
    contract = wrapper["contract"]
    require(wrapper["contract_sha256"] == canonical_sha256(contract), "predeclared contract canonical hash differs")
    require(contract["candidate_id"] == CANDIDATE_ID, "predeclared candidate differs")
    require(
        contract["execution"]["runner_sha256_frozen_before_execution"] == sha256_file(RUNNER_PATH),
        "runner changed after contract freeze",
    )
    verify_successor_contract()
    for record in contract["artifacts"].values():
        path = ROOT / record["path"]
        require(path.is_file(), f"contract artifact is missing: {record['path']}")
        require(path.stat().st_size == record["bytes"], f"contract artifact size differs: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"contract artifact hash differs: {record['path']}")
    verify_source_contract()
    verify_source_snapshot()
    verify_versions()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "matrix hash differs")
    matrix = load_json(MATRIX_PATH)
    require(prompt_binding(matrix) == contract["generation_and_evaluator"]["prompts"], "prompt binding differs")
    require(static_quantizer_self_test() == contract["quantizer_self_test"], "quantizer self-test differs")
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    if require_unconsumed:
        require(not attempts, f"exactly-once authorization is consumed: {[path.name for path in attempts]}")
    output_probe = OUTPUT_DIR / ".preflight-write-probe"
    output_probe.write_bytes(b"preflight\n")
    require(output_probe.read_bytes() == b"preflight\n", "output path readback differs")
    output_probe.unlink()
    return wrapper


def generate_candidate(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    expected: str,
    reference: dict[str, Any],
    policy: StaticActivationPolicy,
) -> tuple[dict[str, Any], int]:
    prompt_length = int(input_ids.shape[1])
    prefix = input_ids
    generated: list[int] = []
    prefix_lengths: list[int] = []
    steps: list[dict[str, Any]] = []
    aligned = True
    forwards = 0
    with torch.inference_mode():
        for index in range(MAX_NEW_TOKENS):
            prefix_lengths.append(int(prefix.shape[1]))
            policy.begin_forward(prompt_length)
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1].detach().cpu()
            forwards += 1
            token = int(logits.argmax())
            comparison = None
            if aligned and index < len(reference["_logits"]):
                comparison = logit_comparison(reference["_logits"][index], logits)
            steps.append(
                {
                    "index": index,
                    "token_id": token,
                    "incoming_prefix_aligned_with_bf16": aligned,
                    "aligned_logit_comparison": comparison,
                }
            )
            generated.append(token)
            reference_ids = reference["generated_token_ids"]
            aligned = aligned and index < len(reference_ids) and token == reference_ids[index]
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    gate = evaluate_response_gate(
        {"decoded_text": decoded, "generated_token_ids": generated}, expected
    )
    common_prefix = 0
    for left, right in zip(generated, reference["generated_token_ids"]):
        if left != right:
            break
        common_prefix += 1
    return {
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "response_gate": gate,
        "exact_bf16_sequence_match": generated == reference["generated_token_ids"],
        "common_bf16_prefix_tokens": common_prefix,
        "prefix_lengths": prefix_lengths,
        "model_token_positions_evaluated": sum(prefix_lengths),
        "steps": steps,
    }, forwards


def write_checksum_bundle(paths: list[Path]) -> None:
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths
    )
    sums_path.write_text(sums, encoding="utf-8")
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    companion.write_text(
        f"{sha256_file(sums_path)}  {sums_path.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )


def execute_once() -> int:
    wrapper = load_verified_contract(require_unconsumed=True)
    ATTEMPT_DIR.mkdir(parents=True, exist_ok=False)
    run_log_path = ATTEMPT_DIR / "run.log"
    started = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "attempt_id": "attempt-0001",
        "started_at_utc": utc_now(),
        "contract_sha256": wrapper["contract_sha256"],
        "runner_sha256": sha256_file(RUNNER_PATH),
        "matrix_sha256": MATRIX_SHA256,
        "static_scale_table_sha256": sha256_file(TABLE_PATH),
        "selected_policy_id_before_execution": None,
        "authorization_consumed": True,
    }
    write_json(ATTEMPT_DIR / "execution_started.json", started)
    started_monotonic = time.monotonic()
    with run_log_path.open("w", encoding="utf-8") as run_log:
        policy: StaticActivationPolicy | None = None
        try:
            log_message(run_log, f"official_attempt_started candidate={CANDIDATE_ID}")
            torch.set_num_threads(TORCH_THREADS)
            torch.set_num_interop_threads(1)
            source_contract = verify_source_contract()
            source_identity = verify_source_snapshot()
            versions = verify_versions()
            matrix = load_json(MATRIX_PATH)
            prompt_specs = {record["case_id"]: record for record in prompt_binding(matrix)}
            tokenizer = AutoTokenizer.from_pretrained(
                SNAPSHOT, local_files_only=True, trust_remote_code=False
            )
            inputs = {
                prompt.case_id: chat_input(tokenizer, prompt.prompt, DEFAULT_SYSTEM)
                for prompt in PROMPTS
            }

            log_message(run_log, "bf16_reference_recomputation_started prompts=9")
            reference_model = AutoModelForCausalLM.from_pretrained(
                SNAPSHOT,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval()
            references: dict[str, dict[str, Any]] = {}
            for prompt in PROMPTS:
                reference = generate_reference(
                    reference_model,
                    tokenizer,
                    inputs[prompt.case_id],
                    prompt.expected,
                )
                frozen = prompt_specs[prompt.case_id]
                require(reference["generated_token_ids"] == frozen["bf16_generated_token_ids"], f"BF16 tokens differ: {prompt.case_id}")
                require(reference["response_gate"]["status"] == frozen["bf16_response_gate_status"], f"BF16 response gate differs: {prompt.case_id}")
                references[prompt.case_id] = reference
                log_message(
                    run_log,
                    f"bf16_prompt_complete case={prompt.case_id} tokens={len(reference['generated_token_ids'])} gate={reference['response_gate']['status']}",
                )
            del reference_model
            gc.collect()

            log_message(run_log, "candidate_model_construction_started tensors=168 fixed_table_boundaries_per_layer=13")
            candidate_model = AutoModelForCausalLM.from_pretrained(
                SNAPSHOT,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval()
            weight_manifest, candidate_hashes = grouped.quantize_model_weights(candidate_model)
            table = load_json(TABLE_PATH)
            policy = StaticActivationPolicy(table)
            policy.install(candidate_model)
            log_message(
                run_log,
                "candidate_model_construction_complete "
                + f"packed_w4_sha256={candidate_hashes['transformer_packed_w4_scope_sha256']} "
                + f"scale32_sha256={candidate_hashes['transformer_weight_scale32_scope_sha256']} "
                + f"static_table_sha256={sha256_file(TABLE_PATH)}",
            )

            prompt_results: list[dict[str, Any]] = []
            model_forwards = 0
            for prompt in PROMPTS:
                reference = references[prompt.case_id]
                candidate, forwards = generate_candidate(
                    candidate_model,
                    tokenizer,
                    inputs[prompt.case_id],
                    prompt.expected,
                    reference,
                    policy,
                )
                model_forwards += forwards
                token_disagreements = sequence_disagreements(
                    reference["generated_token_ids"], candidate["generated_token_ids"]
                )
                gate_match = candidate["response_gate"]["status"] == reference["response_gate"]["status"]
                public_reference = dict(reference)
                public_reference.pop("_logits")
                prompt_results.append(
                    {
                        "case_id": prompt.case_id,
                        "prompt_sha256": prompt_specs[prompt.case_id]["prompt_sha256"],
                        "expected_sha256": prompt_specs[prompt.case_id]["expected_sha256"],
                        "bf16": public_reference,
                        "candidate": candidate,
                        "token_disagreements": token_disagreements,
                        "response_gate_outcome_match": gate_match,
                    }
                )
                log_message(
                    run_log,
                    f"candidate_prompt_complete case={prompt.case_id} tokens={len(candidate['generated_token_ids'])} "
                    f"exact={candidate['exact_bf16_sequence_match']} gate={candidate['response_gate']['status']} gate_match={gate_match}",
                )

            activation_summary = policy.summary(model_forwards)
            policy.uninstall()
            policy = None
            aggregate = aggregate_prompts(prompt_results)
            disagreement_set = [
                {
                    "case_id": prompt["case_id"],
                    "token_disagreements": prompt["token_disagreements"],
                    "bf16_response_gate_status": prompt["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": prompt["candidate"]["response_gate"]["status"],
                    "response_gate_outcome_match": prompt["response_gate_outcome_match"],
                }
                for prompt in prompt_results
                if prompt["token_disagreements"] or not prompt["response_gate_outcome_match"]
            ]
            eligible = (
                aggregate["exact_bf16_sequence_match_count"] == 9
                and aggregate["response_gate_outcome_match_count"] == 9
            )
            status = (
                "PASS_ELIGIBLE_FOR_FRESH_REVIEW"
                if eligible
                else "BLOCKED_EXACT_BF16_DISAGREEMENT"
            )
            result = {
                "schema_version": 1,
                "classification": "qwen_instruct_stage1_option_b_static_position_class_group_scale_single_candidate",
                "status": status,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "attempt_id": "attempt-0001",
                "contract": {
                    "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(CONTRACT_PATH),
                    "contract_sha256": wrapper["contract_sha256"],
                },
                "successor_contract": file_record(SUCCESSOR_PATH),
                "static_scale_table": file_record(TABLE_PATH),
                "matrix": file_record(MATRIX_PATH),
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                },
                "model": source_identity,
                "source_contract_status": source_contract["status"],
                "policy": wrapper["contract"]["implementation"],
                "cost": wrapper["contract"]["cost"],
                "weight_manifest": weight_manifest,
                "candidate_hashes": candidate_hashes,
                "activation_execution": activation_summary,
                "prompts": prompt_results,
                "aggregate": aggregate,
                "eligibility": {
                    "rule": wrapper["contract"]["generation_and_evaluator"]["eligibility"],
                    "eligible": eligible,
                    "eligible_candidate_id": CANDIDATE_ID if eligible else None,
                    "selected_policy_id": None,
                    "fresh_reviewer_required": True,
                },
                "disagreement_set": disagreement_set,
                "disposition": "READY_FOR_FRESH_REVIEW",
                "scope_guards": {
                    "candidate_attempt_count": 1,
                    "alternate_policy_executed": False,
                    "ranking_or_selection_executed": False,
                    "runtime_dynamic_calibration_executed": False,
                    "per_token_scale_sidecars_emitted": False,
                    "runner_changed_after_freeze": False,
                    "rtl_mutated": False,
                    "demo_retargeted": False,
                    "ppa_executed": False,
                    "stage2_entered": False,
                    "network_access_performed": False,
                },
                "claim_boundary": "Numerical-policy attempt only. No cycle-accurate RTL, bandwidth, timing, PPA, demo, U280, or product-acceptance claim.",
                "timing": {
                    "completed_at_utc": utc_now(),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                },
            }
            results_path = ATTEMPT_DIR / "results.json"
            write_json(results_path, result)
            reviewer_input = {
                "schema_version": 1,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "status": "READY_FOR_FRESH_REVIEW",
                "numerical_result": status,
                "contract": file_record(CONTRACT_PATH),
                "successor_contract": file_record(SUCCESSOR_PATH),
                "static_scale_table": file_record(TABLE_PATH),
                "results": file_record(results_path),
                "selected_policy_id_before_adjudication": None,
                "independent_acceptance_required": True,
                "requested_review": [
                    "contract chronology and runner hash freeze",
                    "exactly-once attempt count",
                    "fixed sink/prompt/decode table derivation and absence of dynamic calibration or sidecars",
                    "compression, bandwidth, and RTL-realizability limit arithmetic",
                    "nine-prompt exact-sequence and response-gate recomputation",
                    "set selected_policy_id only after adjudication if and only if eligible",
                ],
                "verification_commands": [
                    "./.venv/bin/python tools/run_option_b_static_position_class_group_scale_stage1.py verify-result",
                    f"sha256sum -c {ATTEMPT_DIR.relative_to(ROOT).as_posix()}/SHA256SUMS",
                ],
                "official_attempt_regeneration_forbidden": True,
            }
            reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
            write_json(reviewer_path, reviewer_input)
            log_message(
                run_log,
                f"official_attempt_complete status={status} exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 "
                f"gate_matches={aggregate['response_gate_outcome_match_count']}/9 selected_policy_id=null",
            )
            run_log.flush()
            os.fsync(run_log.fileno())
            write_checksum_bundle(
                [
                    CONTRACT_PATH,
                    ATTEMPT_DIR / "execution_started.json",
                    run_log_path,
                    results_path,
                    reviewer_path,
                ]
            )
            return 0 if eligible else 4
        except Exception as exc:
            if policy is not None:
                policy.uninstall()
            failure_class = (
                "STATIC_SCALE_RANGE_OVERFLOW"
                if isinstance(exc, StaticScaleOverflow)
                else "EVALUATOR_EXECUTION_FAILURE"
            )
            failure = {
                "schema_version": 1,
                "classification": failure_class,
                "status": "BLOCKED_EVALUATOR_EXECUTION_FAILURE",
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "attempt_id": "attempt-0001",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "numerical_correctness_conclusion": None,
                "selected_policy_id": None,
                "authorization_consumed": True,
                "failed_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
            }
            failure_path = ATTEMPT_DIR / "failure.json"
            write_json(failure_path, failure)
            log_message(
                run_log,
                f"official_attempt_failed classification={failure_class} error_type={type(exc).__name__} error={exc}",
            )
            run_log.flush()
            os.fsync(run_log.fileno())
            write_checksum_bundle(
                [
                    CONTRACT_PATH,
                    ATTEMPT_DIR / "execution_started.json",
                    run_log_path,
                    failure_path,
                ]
            )
            return 5


def parse_sums(path: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        records.append((digest, ROOT / relative))
    return records


def verify_result() -> dict[str, Any]:
    wrapper = load_verified_contract(require_unconsumed=False)
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    require([path.name for path in attempts] == ["attempt-0001"], "attempt set differs")
    started = load_json(ATTEMPT_DIR / "execution_started.json")
    require(started["authorization_consumed"] is True, "authorization consumption differs")
    require(started["runner_sha256"] == sha256_file(RUNNER_PATH), "executed runner hash differs")
    require(started["selected_policy_id_before_execution"] is None, "policy selected before execution")
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    require(
        companion.read_text(encoding="utf-8").strip()
        == f"{sha256_file(sums_path)}  {sums_path.relative_to(ROOT).as_posix()}",
        "checksum companion differs",
    )
    for digest, path in parse_sums(sums_path):
        require(path.is_file() and sha256_file(path) == digest, f"attempt artifact hash differs: {path}")
    failure_path = ATTEMPT_DIR / "failure.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        require(failure["numerical_correctness_conclusion"] is None, "execution failure drew a numerical conclusion")
        require(failure["selected_policy_id"] is None, "execution failure selected a policy")
        return {
            "status": failure["status"],
            "classification": failure["classification"],
            "candidate_id": CANDIDATE_ID,
            "attempt_count": 1,
            "eligible": None,
            "selected_policy_id": None,
            "failure_sha256": sha256_file(failure_path),
        }
    result = load_json(ATTEMPT_DIR / "results.json")
    require(result["candidate_id"] == CANDIDATE_ID, "result candidate differs")
    require(result["contract"]["contract_sha256"] == wrapper["contract_sha256"], "result contract binding differs")
    require(len(result["prompts"]) == 9, "result prompt count differs")
    require(len(result["weight_manifest"]) == 169, "weight manifest count differs")
    require(result["activation_execution"]["named_boundary_count"] == LAYERS * len(EVENT_SPECS), "activation boundary closure differs")
    require(result["activation_execution"]["dynamic_calibration_count"] == 0, "runtime calibration was executed")
    require(result["activation_execution"]["per_token_scale_sidecar_bytes"] == 0, "per-token scale sidecar was emitted")
    disagreements: list[dict[str, Any]] = []
    for prompt in result["prompts"]:
        token_disagreements = sequence_disagreements(
            prompt["bf16"]["generated_token_ids"],
            prompt["candidate"]["generated_token_ids"],
        )
        require(token_disagreements == prompt["token_disagreements"], f"token disagreement set differs: {prompt['case_id']}")
        gate_match = prompt["bf16"]["response_gate"]["status"] == prompt["candidate"]["response_gate"]["status"]
        require(gate_match == prompt["response_gate_outcome_match"], f"gate match differs: {prompt['case_id']}")
        if token_disagreements or not gate_match:
            disagreements.append(
                {
                    "case_id": prompt["case_id"],
                    "token_disagreements": token_disagreements,
                    "bf16_response_gate_status": prompt["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": prompt["candidate"]["response_gate"]["status"],
                    "response_gate_outcome_match": gate_match,
                }
            )
    require(disagreements == result["disagreement_set"], "aggregate disagreement set differs")
    eligible = not disagreements
    require(result["eligibility"]["eligible"] == eligible, "eligibility differs")
    require(result["eligibility"]["selected_policy_id"] is None, "result self-selected policy")
    return {
        "status": result["status"],
        "candidate_id": CANDIDATE_ID,
        "attempt_count": 1,
        "eligible": eligible,
        "exact_sequence_matches": result["aggregate"]["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": result["aggregate"]["response_gate_outcome_match_count"],
        "disagreement_case_ids": [record["case_id"] for record in disagreements],
        "selected_policy_id": None,
        "results_sha256": sha256_file(ATTEMPT_DIR / "results.json"),
        "reviewer_input_sha256": sha256_file(ATTEMPT_DIR / "fresh_reviewer_input.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare-table",
            "prepare-contract",
            "verify-contract",
            "execute",
            "verify-result",
            "self-test",
        ),
    )
    args = parser.parse_args()
    if args.command == "self-test":
        require_project_python()
        print("ACE2_OPTION_B_STATIC_POSITION_CLASS_SELF_TEST_PASS " + json.dumps(static_quantizer_self_test(), sort_keys=True))
        return 0
    if args.command == "prepare-table":
        table = prepare_table()
        print(
            "ACE2_OPTION_B_STATIC_POSITION_CLASS_TABLE_PREPARED "
            f"candidate={CANDIDATE_ID} table_sha256={sha256_file(TABLE_PATH)} "
            f"events={len(table['events'])}"
        )
        return 0
    if args.command == "prepare-contract":
        wrapper = prepare_contract()
        print(
            "ACE2_OPTION_B_STATIC_POSITION_CLASS_CONTRACT_PREPARED "
            f"candidate={CANDIDATE_ID} contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "verify-contract":
        wrapper = load_verified_contract(require_unconsumed=True)
        print(
            "ACE2_OPTION_B_STATIC_POSITION_CLASS_PREFLIGHT_PASS "
            f"candidate={CANDIDATE_ID} attempts=0 contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "execute":
        return execute_once()
    summary = verify_result()
    print("ACE2_OPTION_B_STATIC_POSITION_CLASS_RESULT_VERIFIED " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
