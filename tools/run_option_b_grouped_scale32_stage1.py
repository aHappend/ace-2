#!/usr/bin/env python3
"""Run the one frozen grouped-Scale32 Option-B numerical-policy attempt.

This is a host numerical-policy reference, not cycle-accurate RTL. It preserves
the authenticated per-output-row signed-W4 weight rule and applies the frozen
64/128-lane Scale32 activation policy at exactly thirteen tensor boundaries in
each of the 24 decoder layers.
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
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

from ace2_quality_contracts import (
    SCALE32_ALL_ZERO_RECORD,
    ceil_scale32_from_float,
    scale32_ratio,
    unpack_scale32,
)
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


CANDIDATE_ID = "option_b_grouped_scale32_w4a8_v1"
MISSION_ID = "aaa05dee3743"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
SUCCESSOR_PATH = ROOT / "design/OPTION_B_COMPRESSED_SUCCESSOR_CONTRACT.json"
SUCCESSOR_COMPANION = ROOT / "design/OPTION_B_COMPRESSED_SUCCESSOR_CONTRACT.sha256"
OUTPUT_DIR = ROOT / "build/stage1-option-b-grouped-scale32-w4a8-v1"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
RUNNER_PATH = Path(__file__).resolve()
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16
LAYERS = 24
HIDDEN = 896
INTERMEDIATE = 4864
HEAD = 64
EVENT_FAMILIES = (
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "rope_q",
    "rope_k",
    "attention_value",
    "o_proj",
    "post_attention_rmsnorm",
    "mlp_gate_proj",
    "mlp_up_proj",
    "silu_gate",
    "mlp_down_proj",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    ).hexdigest()


def require_project_python() -> None:
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), f"project Python is missing: {expected}")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")


def expected_transformer_tensors() -> list[dict[str, Any]]:
    geometry = (
        ("self_attn.q_proj", HIDDEN, HIDDEN),
        ("self_attn.k_proj", HIDDEN, 128),
        ("self_attn.v_proj", HIDDEN, 128),
        ("self_attn.o_proj", HIDDEN, HIDDEN),
        ("mlp.gate_proj", HIDDEN, INTERMEDIATE),
        ("mlp.up_proj", HIDDEN, INTERMEDIATE),
        ("mlp.down_proj", INTERMEDIATE, HIDDEN),
    )
    records: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        for suffix, input_features, output_features in geometry:
            records.append(
                {
                    "module": f"model.layers.{layer}.{suffix}",
                    "name": f"model.layers.{layer}.{suffix}.weight",
                    "shape": [output_features, input_features],
                }
            )
    require(len(records) == 168, "transformer linear scope differs")
    return records


def policy_cost() -> dict[str, Any]:
    tensors = expected_transformer_tensors()
    values = sum(math.prod(record["shape"]) for record in tensors)
    rows = sum(record["shape"][0] for record in tensors)
    per_layer_values = values // LAYERS
    per_layer_rows = rows // LAYERS
    packed = (values + 1) // 2
    metadata = rows * 4
    require(per_layer_values == 14_909_440, "per-layer weight count differs")
    require(per_layer_rows == 12_672, "per-layer Scale32 row count differs")
    require((per_layer_values + 1) // 2 == 7_454_720, "per-layer W4 bytes differ")
    require(per_layer_rows * 4 == 50_688, "per-layer weight metadata bytes differ")
    return {
        "transformer_linear_tensor_count": 168,
        "transformer_weight_values": values,
        "packed_w4_payload_bytes": packed,
        "weight_scale32_record_count": rows,
        "weight_scale32_metadata_bytes": metadata,
        "standard_decoder_layer_weight_count": per_layer_values,
        "standard_decoder_layer_scale32_record_count": per_layer_rows,
        "standard_decoder_layer_packed_w4_payload_bytes": 7_454_720,
        "standard_decoder_layer_weight_scale_bytes": 50_688,
        "standard_decoder_layer_total_weight_bytes": 7_505_408,
        "standard_decoder_layer_exact_bits_per_weight": 4.0271978021978025,
        "activation_tensor_sidecars_per_decoder_layer_token": len(EVENT_FAMILIES),
        "activation_transport_metadata_bytes_per_decoder_layer_token": 832,
    }


def pack_s4(qweight: Tensor) -> bytes:
    flat = qweight.detach().cpu().contiguous().reshape(-1).to(torch.int16).numpy()
    require(flat.size % 2 == 0, "W4 tensor has odd element count")
    nibble = flat.astype(np.uint8, copy=False) & np.uint8(0x0F)
    return (nibble[0::2] | (nibble[1::2] << np.uint8(4))).tobytes(order="C")


def scale32_record(value: float) -> int:
    if value == 0.0:
        return SCALE32_ALL_ZERO_RECORD
    record = int(ceil_scale32_from_float(value))
    significand, exponent = unpack_scale32(record)
    require(0x8000 <= significand <= 0xFFFF, "Scale32 significand is malformed")
    require(-24 <= exponent <= 4, "Scale32 exponent is outside the product range")
    return record


def scale32_value(record: int) -> float:
    if record == SCALE32_ALL_ZERO_RECORD:
        return math.ldexp(1.0, -24)
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def adjusted_record(base_record: int, delta: int) -> int:
    require(base_record != SCALE32_ALL_ZERO_RECORD, "dynamic base record may not be all-zero")
    significand, exponent = unpack_scale32(base_record)
    adjusted = exponent + delta
    require(-24 <= adjusted <= 4, "dynamic Scale32 exponent escaped product range")
    return int(significand | ((adjusted & 0xFF) << 16))


@dataclass
class QuantizedGroup:
    payload: Tensor
    dequantized: Tensor
    records: Tensor
    deltas: Tensor


def quantize_groups(value: Tensor, base_records: Tensor, group_lanes: int) -> QuantizedGroup:
    require(value.shape[-1] % group_lanes == 0, "activation group does not divide tensor")
    groups = value.shape[-1] // group_lanes
    require(base_records.numel() in {1, groups}, "base Scale32 record count differs")
    base = base_records.reshape(-1)
    if base.numel() == 1:
        base = base.repeat(groups)
    flat = value.detach().to(torch.float64).reshape(-1, groups, group_lanes)
    require(bool(torch.all(torch.isfinite(flat))), "activation contains non-finite values")
    maxima = flat.abs().amax(dim=2)
    output = torch.empty_like(flat, dtype=torch.int8)
    dequantized = torch.empty_like(flat, dtype=torch.float64)
    records = torch.empty(maxima.shape, dtype=torch.int64, device=maxima.device)
    deltas = torch.empty(maxima.shape, dtype=torch.int16, device=maxima.device)
    base_list = [int(item) for item in base.detach().cpu().tolist()]
    maxima_cpu = maxima.detach().cpu().tolist()
    for vector in range(len(maxima_cpu)):
        for group in range(groups):
            maximum = float(maxima_cpu[vector][group])
            base_record = base_list[group]
            require(base_record != SCALE32_ALL_ZERO_RECORD, "dynamic base record is all-zero")
            chosen: tuple[int, int, float] | None = None
            if maximum == 0.0:
                candidate = adjusted_record(base_record, 0)
                chosen = (0, candidate, scale32_value(candidate))
            else:
                for delta in range(-24, 25):
                    _sig, exponent = unpack_scale32(base_record)
                    if not -24 <= exponent + delta <= 4:
                        continue
                    candidate = adjusted_record(base_record, delta)
                    scale = scale32_value(candidate)
                    if maximum <= 127.0 * scale:
                        chosen = (delta, candidate, scale)
                        break
            require(chosen is not None, "no legal Dynamic Scale32 delta covers activation")
            delta, record, scale = chosen
            quantized = torch.round(flat[vector, group] / scale)
            require(bool(torch.all(quantized >= -127) and torch.all(quantized <= 127)), "dynamic activation escaped signed symmetric A8")
            q8 = quantized.to(torch.int8)
            require(not bool(torch.any(q8 == -128)), "reserved signed-A8 value was produced")
            output[vector, group] = q8
            dequantized[vector, group] = q8.to(torch.float64) * scale
            records[vector, group] = record
            deltas[vector, group] = delta
    shape = value.shape
    return QuantizedGroup(
        payload=output.reshape(shape),
        dequantized=dequantized.reshape(shape),
        records=records.reshape(*shape[:-1], groups),
        deltas=deltas.reshape(*shape[:-1], groups),
    )


def quantizer_self_test() -> dict[str, Any]:
    bases = torch.tensor([scale32_record(0.125), scale32_record(0.25)], dtype=torch.int64)
    values = torch.tensor(
        [[0.0, 0.125, -0.125, 0.1875, 31.5, -31.5, 0.5, -0.5]],
        dtype=torch.float64,
    )
    result = quantize_groups(values, bases, 4)
    require(result.payload.dtype == torch.int8, "self-test payload type differs")
    require(not bool(torch.any(result.payload == -128)), "self-test produced reserved value")
    require(result.deltas.tolist() == [[-6, 0]], "self-test delta selection differs")
    tie = quantize_groups(
        torch.tensor([[15.875, -15.875, 0.3125, -0.3125]], dtype=torch.float64),
        torch.tensor([scale32_record(0.125)], dtype=torch.int64),
        4,
    )
    require(
        tie.payload.tolist() == [[127, -127, 2, -2]],
        "round-to-nearest ties-to-even differs",
    )
    return {
        "status": "PASS",
        "selection": "minimum_legal_integer_delta",
        "rounding": "round_to_nearest_ties_to_even",
        "reserved_minus_128_produced": False,
        "self_test_payload_sha256": hashlib.sha256(
            result.payload.cpu().numpy().tobytes(order="C")
        ).hexdigest(),
    }


class ActivationPolicy:
    def __init__(self, scales: dict[str, Any]) -> None:
        self.scales = scales
        self.handles: list[Any] = []
        self.original_rope: Callable[..., Any] | None = None
        self.active_layer: int | None = None
        self.calls: Counter[str] = Counter()
        self.group_count = 0
        self.value_count = 0
        self.delta_min: int | None = None
        self.delta_max: int | None = None
        self.payload_hash = hashlib.sha256()
        self.record_hash = hashlib.sha256()

    @staticmethod
    def _records(values: list[float] | float) -> Tensor:
        source = values if isinstance(values, list) else [values]
        return torch.tensor([scale32_record(float(value)) for value in source], dtype=torch.int64)

    def _linear_output_records(self, name: str) -> Tensor:
        metadata = self.scales["linears"][name]
        head = [float(value) for value in metadata.get("output_head_scales") or []]
        return self._records(head if head else float(metadata["output_scale"]))

    def _operator_record(self, name: str) -> Tensor:
        return self._records(float(self.scales["operators"][name]["scale"]))

    def _apply(self, name: str, value: Tensor, records: Tensor, lanes: int) -> Tensor:
        result = quantize_groups(value, records.to(value.device), lanes)
        self.calls[name] += 1
        self.group_count += result.records.numel()
        self.value_count += result.payload.numel()
        minimum = int(result.deltas.min())
        maximum = int(result.deltas.max())
        self.delta_min = minimum if self.delta_min is None else min(self.delta_min, minimum)
        self.delta_max = maximum if self.delta_max is None else max(self.delta_max, maximum)
        payload = result.payload.detach().cpu().contiguous().numpy().astype(np.int8, copy=False)
        record = result.records.detach().cpu().contiguous().numpy().astype("<u4", copy=False)
        self.payload_hash.update(name.encode() + b"\0" + payload.tobytes(order="C"))
        self.record_hash.update(name.encode() + b"\0" + record.tobytes(order="C"))
        return result.dequantized.to(value.dtype)

    def _output_hook(self, name: str, records: Tensor, lanes: int) -> Callable[..., Tensor]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> Tensor:
            require(isinstance(output, Tensor), f"{name} output is not a tensor")
            return self._apply(name, output, records, lanes)

        return hook

    def _input_hook(self, name: str, records: Tensor, lanes: int) -> Callable[..., tuple[Any, ...]]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            require(inputs and isinstance(inputs[0], Tensor), f"{name} input is not a tensor")
            return (self._apply(name, inputs[0], records, lanes), *inputs[1:])

        return hook

    def install(self, model: nn.Module) -> None:
        require(self.original_rope is None, "activation policy is already installed")
        for layer_index, layer in enumerate(model.model.layers):
            prefix = f"model.layers.{layer_index}"
            q_name = f"{prefix}.self_attn.q_proj"
            k_name = f"{prefix}.self_attn.k_proj"
            v_name = f"{prefix}.self_attn.v_proj"
            o_name = f"{prefix}.self_attn.o_proj"
            gate_name = f"{prefix}.mlp.gate_proj"
            up_name = f"{prefix}.mlp.up_proj"
            down_name = f"{prefix}.mlp.down_proj"
            input_norm_records = self._operator_record(f"{prefix}.input_layernorm.output")
            post_norm_records = self._operator_record(f"{prefix}.post_attention_layernorm.output")
            q_records = self._linear_output_records(q_name)
            k_records = self._linear_output_records(k_name)
            v_records = self._linear_output_records(v_name)
            o_records = self._linear_output_records(o_name)
            gate_records = self._linear_output_records(gate_name)
            up_records = self._linear_output_records(up_name)
            down_records = self._linear_output_records(down_name)
            attention_value_records = self._records(float(self.scales["linears"][o_name]["input_scale"]))
            silu_records = self._records(float(self.scales["linears"][down_name]["input_scale"]))

            self.handles.append(layer.input_layernorm.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.input_rmsnorm", input_norm_records, 128)
            ))
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.q_proj", q_records, 64)
            ))
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.k_proj", k_records, 64)
            ))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.v_proj", v_records, 64)
            ))

            def set_layer(_module: nn.Module, _inputs: tuple[Any, ...], index: int = layer_index) -> None:
                require(self.active_layer is None, "nested Qwen attention activation is unsupported")
                self.active_layer = index

            def clear_layer(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> Any:
                require(self.active_layer == index, "active attention layer tracking differs")
                self.active_layer = None
                return output

            self.handles.append(layer.self_attn.register_forward_pre_hook(set_layer))
            self.handles.append(layer.self_attn.register_forward_hook(clear_layer))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                self._input_hook(f"layer_{layer_index}.attention_value", attention_value_records, 128)
            ))
            self.handles.append(layer.self_attn.o_proj.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.o_proj", o_records, 128)
            ))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.post_attention_rmsnorm", post_norm_records, 128)
            ))
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.mlp_gate_proj", gate_records, 128)
            ))
            self.handles.append(layer.mlp.up_proj.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.mlp_up_proj", up_records, 128)
            ))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                self._input_hook(f"layer_{layer_index}.silu_gate", silu_records, 128)
            ))
            self.handles.append(layer.mlp.down_proj.register_forward_hook(
                self._output_hook(f"layer_{layer_index}.mlp_down_proj", down_records, 128)
            ))

        self.original_rope = modeling_qwen2.apply_rotary_pos_emb

        def dynamic_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: Any = None, unsqueeze_dim: int = 1) -> tuple[Tensor, Tensor]:
            require(self.original_rope is not None, "original RoPE function is missing")
            require(self.active_layer is not None, "RoPE executed outside a tracked attention layer")
            q_embed, k_embed = self.original_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
            layer_index = self.active_layer
            prefix = f"model.layers.{layer_index}.self_attn"
            q_records = self._linear_output_records(f"{prefix}.q_proj")
            k_records = self._linear_output_records(f"{prefix}.k_proj")
            return (
                self._apply(f"layer_{layer_index}.rope_q", q_embed, q_records, 64),
                self._apply(f"layer_{layer_index}.rope_k", k_embed, k_records, 64),
            )

        modeling_qwen2.apply_rotary_pos_emb = dynamic_rope

    def uninstall(self) -> None:
        if self.original_rope is not None:
            modeling_qwen2.apply_rotary_pos_emb = self.original_rope
            self.original_rope = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active_layer = None

    def summary(self, model_forward_count: int) -> dict[str, Any]:
        expected_names = {
            f"layer_{layer}.{family}"
            for layer in range(LAYERS)
            for family in EVENT_FAMILIES
        }
        require(set(self.calls) == expected_names, "activation event name closure differs")
        require(set(self.calls.values()) == {model_forward_count}, "activation event call counts differ across boundaries")
        return {
            "model_forward_count": model_forward_count,
            "named_boundary_count": len(expected_names),
            "events_per_model_forward": len(expected_names),
            "event_call_count": sum(self.calls.values()),
            "group_count": self.group_count,
            "activation_value_count": self.value_count,
            "minimum_delta": self.delta_min,
            "maximum_delta": self.delta_max,
            "reserved_minus_128_produced": False,
            "payload_stream_sha256": self.payload_hash.hexdigest(),
            "scale32_record_stream_sha256": self.record_hash.hexdigest(),
            "calls": dict(sorted(self.calls.items())),
        }


def quantize_model_weights(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _ in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    manifest: list[dict[str, Any]] = []
    packed_scope = hashlib.sha256()
    scale_scope = hashlib.sha256()
    with torch.no_grad():
        for name, module in modules:
            source = module.weight.detach().to(torch.float64)
            absmax = source.abs().amax(dim=1)
            require(not bool(torch.any(absmax == 0)), f"all-zero weight row is unsupported: {name}")
            ideal = absmax / 7.0
            records = torch.tensor(
                [scale32_record(float(value)) for value in ideal.cpu().tolist()],
                dtype=torch.int64,
            )
            scales = torch.tensor(
                [scale32_value(int(value)) for value in records.tolist()],
                dtype=torch.float64,
                device=source.device,
            )
            qweight = torch.round(source / scales[:, None]).clamp(-8, 7).to(torch.int8)
            reconstructed = qweight.to(torch.float64) * scales[:, None]
            module.weight.copy_(reconstructed.to(module.weight.dtype))
            packed = pack_s4(qweight)
            scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
            record = {
                "module": name,
                "name": f"{name}.weight",
                "shape": list(module.weight.shape),
                "scope": "lm_head_authenticated_baseline_w4" if name == "lm_head" else "transformer_candidate_scope",
                "packed_w4_bytes": len(packed),
                "packed_w4_sha256": hashlib.sha256(packed).hexdigest(),
                "weight_scale32_record_count": int(records.numel()),
                "weight_scale32_bytes": len(scale_raw),
                "weight_scale32_sha256": hashlib.sha256(scale_raw).hexdigest(),
            }
            manifest.append(record)
            if name != "lm_head":
                packed_scope.update(name.encode() + b"\0" + packed)
                scale_scope.update(name.encode() + b"\0" + scale_raw)
    transformer = [record for record in manifest if record["scope"] == "transformer_candidate_scope"]
    cost = policy_cost()
    require(sum(record["packed_w4_bytes"] for record in transformer) == cost["packed_w4_payload_bytes"], "transformer packed W4 byte count differs")
    require(sum(record["weight_scale32_bytes"] for record in transformer) == cost["weight_scale32_metadata_bytes"], "transformer weight metadata byte count differs")
    return manifest, {
        "transformer_packed_w4_scope_sha256": packed_scope.hexdigest(),
        "transformer_weight_scale32_scope_sha256": scale_scope.hexdigest(),
        "weight_manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
    }


def verify_successor_contract() -> dict[str, Any]:
    wrapper = load_json(SUCCESSOR_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "successor wrapper shape differs")
    contract = wrapper["contract"]
    require(wrapper["contract_sha256"] == canonical_sha256(contract), "successor canonical hash differs")
    require(contract["selected_policy_id"] == CANDIDATE_ID, "selected policy differs")
    require(contract["status"] == "authorized_bounded_successor_pending_execution_and_fresh_review", "successor status differs")
    require(contract["attempt_budget"]["exact_authorized_candidate_attempts"] == 1, "attempt budget differs")
    require(contract["attempt_budget"]["attempt_id"] == "attempt-0001", "attempt id differs")
    require(contract["attempt_budget"]["frozen_matrix"]["sha256"] == MATRIX_SHA256, "matrix binding differs")
    require(contract["attempt_budget"]["network_access_permitted"] is False, "network policy differs")
    require(contract["claim_boundary"]["numerical_attempt_executed"] is False, "contract already claims execution")
    companion = SUCCESSOR_COMPANION.read_text(encoding="utf-8").strip()
    require(companion == f"{sha256_file(SUCCESSOR_PATH)}  design/OPTION_B_COMPRESSED_SUCCESSOR_CONTRACT.json", "successor companion differs")
    for binding in contract["hash_bindings"].values():
        path = ROOT / binding["path"]
        require(path.is_file() and sha256_file(path) == binding["sha256"], f"successor hash binding differs: {binding['path']}")
    return wrapper


def contract_artifacts() -> dict[str, Any]:
    paths = {
        "successor_contract": SUCCESSOR_PATH,
        "successor_contract_companion": SUCCESSOR_COMPANION,
        "matrix": MATRIX_PATH,
        "source_contract": ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json",
        "derived_scales": CALIBRATION_DIR / "derived_scales.json",
        "full_model_image": IMAGE_DIR / "full_model_image.bin",
        "image_manifest": IMAGE_DIR / "manifest.json",
        "identity_manifest": IMAGE_DIR / "option_b_identity_manifest.json",
        "oracle_contract": IMAGE_DIR / "oracle_contract.json",
        "response_gate_contract": IMAGE_DIR / "response_gate_contract.json",
        "runner_source": RUNNER_PATH,
        "quality_contract_source": ROOT / "tools/ace2_quality_contracts.py",
        "shared_evaluator_source": ROOT / "tools/run_all_transformer_per32_stage1.py",
        "prompt_source": ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py",
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "oracle_source": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
    }
    return {name: file_record(path) for name, path in paths.items()}


def prepare_contract() -> dict[str, Any]:
    require_project_python()
    require(not ATTEMPT_DIR.exists(), "cannot prepare contract after attempt creation")
    successor = verify_successor_contract()
    verify_source_contract()
    verify_source_snapshot()
    versions = verify_versions()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "matrix hash differs")
    self_test = quantizer_self_test()
    matrix = load_json(MATRIX_PATH)
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "authority": {
            "successor_contract": file_record(SUCCESSOR_PATH),
            "successor_contract_sha256": successor["contract_sha256"],
            "exact_authorized_candidate_attempts": 1,
            "attempt_id": "attempt-0001",
        },
        "implementation": {
            "classification": "host_numerical_policy_reference_not_cycle_accurate_rtl",
            "weights": "all 168 transformer linears use signed-W4 with one normalized Scale32 record per output row; lm_head retains the same authenticated per-row W4 rule",
            "activation_boundaries": list(EVENT_FAMILIES),
            "grouping": {"q_k_v_and_rope": 64, "hidden_and_mlp": 128},
            "selection": "preserve each frozen base Scale32 significand and choose the minimum legal exponent delta in [-24,+24] that covers every group lane in symmetric signed A8 [-127,+127]",
            "rounding": "round_to_nearest_ties_to_even",
            "operator_arithmetic": "authenticated local Qwen BF16 host operators with frozen W4-dequantized weights and explicit grouped-A8 boundary quantization; no RTL, latency, bandwidth, or cycle claim",
            "event_count_per_decoder_layer": len(EVENT_FAMILIES),
            "event_count_per_model_forward": LAYERS * len(EVENT_FAMILIES),
        },
        "cost": policy_cost(),
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
            "exact_command": ["./.venv/bin/python", "tools/run_option_b_grouped_scale32_stage1.py", "execute"],
            "runner_sha256_frozen_before_execution": sha256_file(RUNNER_PATH),
            "network_access_permitted": False,
            "replay_tuning_ranking_bisection_sweep_or_gate_relaxation_permitted": False,
            "rtl_demo_ppa_u280_or_stage_transition_permitted": False,
            "expected_outputs": [
                "build/stage1-option-b-grouped-scale32-w4a8-v1/attempt-0001/execution_started.json",
                "build/stage1-option-b-grouped-scale32-w4a8-v1/attempt-0001/run.log",
                "build/stage1-option-b-grouped-scale32-w4a8-v1/attempt-0001/results.json",
                "build/stage1-option-b-grouped-scale32-w4a8-v1/attempt-0001/fresh_reviewer_input.json",
                "build/stage1-option-b-grouped-scale32-w4a8-v1/attempt-0001/SHA256SUMS",
                "build/stage1-option-b-grouped-scale32-w4a8-v1/attempt-0001/SHA256SUMS.sha256",
            ],
        },
        "quantizer_self_test": self_test,
        "artifacts": contract_artifacts(),
    }
    wrapper = {"contract": contract, "contract_sha256": canonical_sha256(contract)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CONTRACT_PATH, wrapper)
    return wrapper


def load_verified_contract(require_unconsumed: bool) -> dict[str, Any]:
    require_project_python()
    require(CONTRACT_PATH.is_file(), "predeclared contract is missing")
    wrapper = load_json(CONTRACT_PATH)
    contract = wrapper["contract"]
    require(wrapper["contract_sha256"] == canonical_sha256(contract), "predeclared contract canonical hash differs")
    require(contract["candidate_id"] == CANDIDATE_ID, "predeclared candidate differs")
    require(contract["execution"]["runner_sha256_frozen_before_execution"] == sha256_file(RUNNER_PATH), "runner changed after contract freeze")
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
    require(quantizer_self_test() == contract["quantizer_self_test"], "quantizer self-test differs")
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
) -> tuple[dict[str, Any], int]:
    prefix = input_ids
    generated: list[int] = []
    prefix_lengths: list[int] = []
    steps: list[dict[str, Any]] = []
    aligned = True
    forwards = 0
    with torch.inference_mode():
        for index in range(MAX_NEW_TOKENS):
            prefix_lengths.append(int(prefix.shape[1]))
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
    gate = evaluate_response_gate({"decoded_text": decoded, "generated_token_ids": generated}, expected)
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
        "authorization_consumed": True,
    }
    write_json(ATTEMPT_DIR / "execution_started.json", started)
    started_monotonic = time.monotonic()
    with run_log_path.open("w", encoding="utf-8") as run_log:
        policy: ActivationPolicy | None = None
        try:
            log_message(run_log, f"official_attempt_started candidate={CANDIDATE_ID}")
            torch.set_num_threads(TORCH_THREADS)
            torch.set_num_interop_threads(1)
            source_contract = verify_source_contract()
            source_identity = verify_source_snapshot()
            versions = verify_versions()
            matrix = load_json(MATRIX_PATH)
            prompt_specs = {record["case_id"]: record for record in prompt_binding(matrix)}
            tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
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
                log_message(run_log, f"bf16_prompt_complete case={prompt.case_id} tokens={len(reference['generated_token_ids'])} gate={reference['response_gate']['status']}")
            del reference_model
            gc.collect()

            log_message(run_log, "candidate_model_construction_started tensors=168 boundaries_per_layer=13")
            scales = load_json(CALIBRATION_DIR / "derived_scales.json")
            candidate_model = AutoModelForCausalLM.from_pretrained(
                SNAPSHOT,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval()
            weight_manifest, candidate_hashes = quantize_model_weights(candidate_model)
            policy = ActivationPolicy(scales)
            policy.install(candidate_model)
            log_message(run_log, "candidate_model_construction_complete " + f"packed_w4_sha256={candidate_hashes['transformer_packed_w4_scope_sha256']} scale32_sha256={candidate_hashes['transformer_weight_scale32_scope_sha256']}")

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
                )
                model_forwards += forwards
                token_disagreements = sequence_disagreements(reference["generated_token_ids"], candidate["generated_token_ids"])
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
                log_message(run_log, f"candidate_prompt_complete case={prompt.case_id} tokens={len(candidate['generated_token_ids'])} exact={candidate['exact_bf16_sequence_match']} gate={candidate['response_gate']['status']} gate_match={gate_match}")

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
            eligible = aggregate["exact_bf16_sequence_match_count"] == 9 and aggregate["response_gate_outcome_match_count"] == 9
            status = "PASS_ELIGIBLE_FOR_FRESH_REVIEW" if eligible else "BLOCKED_EXACT_BF16_DISAGREEMENT"
            result = {
                "schema_version": 1,
                "classification": "qwen_instruct_stage1_option_b_grouped_scale32_single_candidate",
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
                "results": file_record(results_path),
                "independent_acceptance_required": True,
                "requested_review": [
                    "contract chronology and runner hash freeze",
                    "exactly-once attempt count",
                    "policy construct fidelity and cost limits",
                    "nine-prompt exact-sequence and response-gate recomputation",
                    "failure taxonomy if ineligible",
                ],
            }
            reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
            write_json(reviewer_path, reviewer_input)
            write_checksum_bundle(
                [
                    CONTRACT_PATH,
                    ATTEMPT_DIR / "execution_started.json",
                    run_log_path,
                    results_path,
                    reviewer_path,
                ]
            )
            log_message(run_log, f"official_attempt_complete status={status} exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 gate_matches={aggregate['response_gate_outcome_match_count']}/9")
            return 0 if eligible else 4
        except Exception as exc:
            if policy is not None:
                policy.uninstall()
            failure = {
                "schema_version": 1,
                "classification": "EVALUATOR_EXECUTION_FAILURE",
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
            write_json(ATTEMPT_DIR / "failure.json", failure)
            log_message(run_log, f"official_attempt_failed classification=EVALUATOR_EXECUTION_FAILURE error_type={type(exc).__name__} error={exc}")
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
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    require(companion.read_text(encoding="utf-8").strip() == f"{sha256_file(sums_path)}  {sums_path.relative_to(ROOT).as_posix()}", "checksum companion differs")
    for digest, path in parse_sums(sums_path):
        require(path.is_file() and sha256_file(path) == digest, f"attempt artifact hash differs: {path}")
    result = load_json(ATTEMPT_DIR / "results.json")
    require(result["candidate_id"] == CANDIDATE_ID, "result candidate differs")
    require(result["contract"]["contract_sha256"] == wrapper["contract_sha256"], "result contract binding differs")
    require(len(result["prompts"]) == 9, "result prompt count differs")
    require(len(result["weight_manifest"]) == 169, "weight manifest count differs")
    require(result["activation_execution"]["named_boundary_count"] == LAYERS * len(EVENT_FAMILIES), "activation boundary closure differs")
    disagreements: list[dict[str, Any]] = []
    for prompt in result["prompts"]:
        token_disagreements = sequence_disagreements(prompt["bf16"]["generated_token_ids"], prompt["candidate"]["generated_token_ids"])
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
        "results_sha256": sha256_file(ATTEMPT_DIR / "results.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-contract", "verify-contract", "execute", "verify-result", "self-test"))
    args = parser.parse_args()
    if args.command == "self-test":
        require_project_python()
        print("ACE2_OPTION_B_GROUPED_SCALE32_SELF_TEST_PASS " + json.dumps(quantizer_self_test(), sort_keys=True))
        return 0
    if args.command == "prepare-contract":
        wrapper = prepare_contract()
        print(f"ACE2_OPTION_B_GROUPED_SCALE32_CONTRACT_PREPARED candidate={CANDIDATE_ID} contract_sha256={wrapper['contract_sha256']}")
        return 0
    if args.command == "verify-contract":
        wrapper = load_verified_contract(require_unconsumed=True)
        print(f"ACE2_OPTION_B_GROUPED_SCALE32_PREFLIGHT_PASS candidate={CANDIDATE_ID} attempts=0 contract_sha256={wrapper['contract_sha256']}")
        return 0
    if args.command == "execute":
        return execute_once()
    summary = verify_result()
    print("ACE2_OPTION_B_GROUPED_SCALE32_RESULT_VERIFIED " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
