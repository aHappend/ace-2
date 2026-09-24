from __future__ import annotations

import copy
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from tools import ace2_quality_contracts as quality
from tools import ace2_full_model_fixed_point as fixed
from tools import grouped_w4a8_payload_conformance as conformance
from tools import model_hardware_contract as hardware
from tools import qualify_stage1_w4a8_software as runtime
from tools import w4a8_full_model_evaluator as evaluator


ROOT = Path(__file__).resolve().parents[1]


def runtime_template(input_channels: int, output_channels: int) -> fixed.W4A8Linear:
    template = fixed.W4A8Linear.__new__(fixed.W4A8Linear)
    nn.Module.__init__(template)
    template.in_features = input_channels
    template.out_features = output_channels
    template.input_scale = 1.0
    template.hardware_input_scale = 1.0
    template.input_is_quantized = True
    template.output_scale = 1.0
    template.output_scale32_record = quality.pack_scale32(0x8000, 0)
    template.output_head_size = None
    template.register_buffer("output_head_scales", torch.empty(0, dtype=torch.float64))
    template.register_buffer(
        "output_scale_per_channel",
        torch.ones(output_channels, dtype=torch.float64),
    )
    template.register_buffer(
        "weight_scale",
        torch.ones(output_channels, dtype=torch.float64),
    )
    template.register_buffer(
        "native_scale32_records",
        torch.full(
            (output_channels,),
            quality.pack_scale32(0x8000, 0),
            dtype=torch.int64,
        ),
    )
    multiplier, right_shift = fixed.derive_multiplier(
        torch.ones(output_channels, dtype=torch.float64)
    )
    template.register_buffer("multiplier", multiplier)
    template.register_buffer("right_shift", right_shift)
    template.register_buffer("bias_accumulator", None)
    template.register_buffer("_source_bias", None, persistent=False)
    return template


class PayloadProjectionModel(nn.Module):
    def __init__(self, q_outputs: int, down_outputs: int) -> None:
        super().__init__()
        self.q_proj = runtime_template(896, q_outputs)
        self.down_proj = runtime_template(4864, down_outputs)


class PayloadMLPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = fixed.FixedMLP.__new__(fixed.FixedMLP)
        nn.Module.__init__(self.mlp)
        self.mlp.gate_proj = runtime_template(64, 32)
        self.mlp.up_proj = runtime_template(64, 32)
        self.mlp.down_proj = runtime_template(32, 2)


class PayloadAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        source = attention_source()
        self.self_attn = fixed.FixedAttention(source, normalized_input_scale=1.0)


def attention_source(layer_index: int = 0) -> SimpleNamespace:
    source = SimpleNamespace(
        layer_idx=layer_index,
        head_dim=64,
        num_key_value_groups=7,
        q_proj=runtime_template(896, 896),
        k_proj=runtime_template(896, 128),
        v_proj=runtime_template(896, 128),
        o_proj=runtime_template(896, 896),
    )
    source.q_proj.output_scale = None
    source.q_proj.output_scale32_record = None
    source.q_proj.output_head_size = 64
    source.q_proj.output_head_scales = torch.ones(14, dtype=torch.float64)
    source.k_proj.output_scale = None
    source.k_proj.output_scale32_record = None
    source.k_proj.output_head_size = 64
    source.k_proj.output_head_scales = torch.ones(2, dtype=torch.float64)
    return source


class PayloadDecoderLayerModel(nn.Module):
    def __init__(self, layer_index: int = 0) -> None:
        super().__init__()
        source = SimpleNamespace(
            attention_type="full_attention",
            input_layernorm=SimpleNamespace(weight=torch.ones(896)),
            post_attention_layernorm=SimpleNamespace(weight=torch.ones(896)),
            self_attn=attention_source(layer_index),
            mlp=SimpleNamespace(
                gate_proj=runtime_template(896, 32),
                up_proj=runtime_template(896, 32),
                down_proj=runtime_template(32, 896),
            ),
        )
        self.layer = fixed.FixedDecoderLayer(
            source,
            input_scale=1.0,
            post_attention_scale=1.0,
            post_mlp_scale=1.0,
            input_norm_output_scale=1.0,
            post_attention_norm_output_scale=1.0,
        )


class PayloadDecoderStackModel(nn.Module):
    def __init__(self, layer_count: int = runtime.FULL_QWEN_DECODER_LAYERS) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            PayloadDecoderLayerModel(layer_index).layer
            for layer_index in range(layer_count)
        )
        self.model.norm = fixed.FixedRMSNorm(
            SimpleNamespace(weight=torch.ones(896)),
            input_scale=1.0,
            output_scale=0.5,
        )
        self.lm_head = runtime_template(
            896,
            hardware.load_descriptor("qwen2.5-0.5b")["dimensions"]["vocab_size"],
        )

    def forward_hidden(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        return self.lm_head(self.model.norm(hidden_states))


class CompactFixedAttention(fixed.FixedAttention):
    def __init__(self, layer_index: int, hidden_size: int) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_index
        self.q_proj = runtime_template(hidden_size, hidden_size)
        self.k_proj = runtime_template(hidden_size, hidden_size)
        self.v_proj = runtime_template(hidden_size, hidden_size)
        self.o_proj = runtime_template(hidden_size, hidden_size)


class CompactFixedMLP(fixed.FixedMLP):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        nn.Module.__init__(self)
        self.gate_proj = runtime_template(hidden_size, intermediate_size)
        self.up_proj = runtime_template(hidden_size, intermediate_size)
        self.down_proj = runtime_template(intermediate_size, hidden_size)


class CompactFixedDecoderLayer(fixed.FixedDecoderLayer):
    def __init__(
        self,
        layer_index: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        nn.Module.__init__(self)
        self.self_attn = CompactFixedAttention(layer_index, hidden_size)
        self.mlp = CompactFixedMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        del position_embeddings
        query = self.self_attn.q_proj.forward_hardware_input(hidden_states)
        key = self.self_attn.k_proj.forward_hardware_input(hidden_states)
        value = self.self_attn.v_proj.forward_hardware_input(hidden_states)
        self.last_key = key
        self.last_value = value
        attention_input = (
            query.to(torch.int16) + key.to(torch.int16) + value.to(torch.int16)
        ).clamp(-128, 127).to(torch.int8)
        attention = self.self_attn.o_proj.forward_hardware_input(attention_input)
        mlp = self.mlp(hidden_states)
        return (
            hidden_states.to(torch.int16)
            + attention.to(torch.int16)
            + mlp.to(torch.int16)
        ).clamp(-128, 127).to(torch.float32)


class CompactFixedRMSNorm(fixed.FixedRMSNorm):
    def __init__(self, hidden_size: int) -> None:
        nn.Module.__init__(self)
        self.input_scale = 1.0
        self.output_scale = 1.0
        self.register_buffer(
            "scaled_gains_q8",
            torch.full((hidden_size,), 1 << 8, dtype=torch.int16),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class CompactPayloadDecoderStackModel(nn.Module):
    def __init__(self, descriptor: dict) -> None:
        super().__init__()
        dimensions = descriptor["dimensions"]
        hidden_size = dimensions["hidden_size"]
        intermediate_size = dimensions["intermediate_size"]
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            CompactFixedDecoderLayer(
                layer_index,
                hidden_size,
                intermediate_size,
            )
            for layer_index in range(runtime.FULL_QWEN_DECODER_LAYERS)
        )
        self.model.norm = CompactFixedRMSNorm(hidden_size)
        self.lm_head = runtime_template(
            hidden_size,
            dimensions["vocab_size"],
        )
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace(use_cache=False)

    def forward_hidden(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        return self.lm_head(self.model.norm(hidden_states))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: SimpleNamespace | None,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        if not use_cache or logits_to_keep != 1:
            raise RuntimeError("compact package fixture requires cacheful generation")
        previous_length = (
            0
            if past_key_values is None
            else past_key_values.layers[0].keys.shape[-2]
        )
        if attention_mask.shape[-1] != previous_length + input_ids.shape[-1]:
            raise RuntimeError("compact package attention mask length differs")
        hidden_states = torch.zeros(
            (1, input_ids.shape[-1], 32),
            dtype=torch.float32,
        )
        hidden_states[:, :, 0] = (
            input_ids.remainder(17) - 8
        ).to(torch.float32)
        position_embeddings = (
            torch.ones((1, input_ids.shape[-1], 32), dtype=torch.float64),
            torch.zeros((1, input_ids.shape[-1], 32), dtype=torch.float64),
        )
        cache_layers = []
        for layer_index, layer in enumerate(self.model.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )
            key = layer.last_key.reshape(
                1, 1, input_ids.shape[-1], 32
            ).to(torch.int8)
            value = layer.last_value.reshape(
                1, 1, input_ids.shape[-1], 32
            ).to(torch.int8)
            if past_key_values is not None:
                previous = past_key_values.layers[layer_index]
                key = torch.cat((previous.keys, key), dim=-2)
                value = torch.cat((previous.values, value), dim=-2)
            layer.self_attn.ace2_key_scale32_cache = torch.full(
                key.shape[:-1],
                quality.pack_scale32(0x8000, 0),
                dtype=torch.int64,
            )
            cache_layers.append(SimpleNamespace(keys=key, values=value))
        normalized = self.model.norm(hidden_states)
        raw_logits = self.lm_head.forward_hardware_input(normalized)
        self.lm_head.last_raw_output = raw_logits
        return SimpleNamespace(
            logits=raw_logits.to(torch.float32),
            past_key_values=SimpleNamespace(layers=cache_layers),
        )


class CompactFrozenTokenizer:
    def __init__(self, system: str, prompt: str) -> None:
        self.system = system
        self.prompt = prompt

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> torch.Tensor:
        if messages != [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.prompt},
        ]:
            raise RuntimeError("frozen prompt differs")
        if not tokenize or not add_generation_prompt or return_tensors != "pt":
            raise RuntimeError("frozen generation template differs")
        encoded = (self.system + "\0" + self.prompt).encode("utf-8")
        return torch.tensor([list(encoded)], dtype=torch.long)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if not skip_special_tokens or clean_up_tokenization_spaces:
            raise RuntimeError("frozen decode policy differs")
        return " ".join(str(token_id) for token_id in token_ids)


def tensor_value_record(tensor: torch.Tensor) -> dict[str, object]:
    host = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(host.dtype),
        "shape": list(host.shape),
        "values": host.reshape(-1).tolist(),
    }


def compact_generation_tensor_trace(
    model: nn.Module,
    tokenizer: CompactFrozenTokenizer,
    prompt: dict[str, str],
    generation: dict[str, object],
) -> dict[str, object]:
    input_ids = evaluator._render_prompt_ids(tokenizer, prompt, generation)
    sequence = input_ids.clone()
    current_ids = input_ids
    past_key_values = None
    generated: list[int] = []
    steps: list[dict[str, object]] = []
    with torch.inference_mode():
        for _step in range(int(generation["max_new_tokens"])):
            output = model(
                input_ids=current_ids,
                attention_mask=torch.ones_like(sequence, dtype=torch.long),
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            raw_logits = model.lm_head.last_raw_output
            if raw_logits is None or raw_logits.dtype != torch.int8:
                raise RuntimeError("integer lm-head logits are missing")
            raw_last = raw_logits[:, -1, :].contiguous()
            selected = int(torch.argmax(raw_last, dim=-1).item())
            generated.append(selected)
            past_key_values = output.past_key_values
            if past_key_values is None or len(past_key_values.layers) != 24:
                raise RuntimeError("complete 24-layer KV cache is missing")
            layers = []
            for layer_index, (cache_layer, decoder_layer) in enumerate(
                zip(
                    past_key_values.layers,
                    model.model.layers,
                    strict=True,
                )
            ):
                scales = decoder_layer.self_attn.ace2_key_scale32_cache
                layers.append(
                    {
                        "key": tensor_value_record(cache_layer.keys),
                        "key_scale32": tensor_value_record(scales),
                        "layer_index": layer_index,
                        "value": tensor_value_record(cache_layer.values),
                    }
                )
            steps.append(
                {
                    "integer_logits": tensor_value_record(raw_last),
                    "layers": layers,
                }
            )
            next_token = torch.tensor([[selected]], dtype=torch.long)
            sequence = torch.cat((sequence, next_token), dim=1)
            current_ids = next_token
            if selected in generation["termination_token_ids"]:
                break
    return {
        "generated_token_ids": generated,
        "steps": steps,
    }


def staged_grouped_projection(
    template: fixed.W4A8Linear,
) -> runtime.GroupedW4A8Linear:
    replacement = runtime.GroupedW4A8Linear.__new__(runtime.GroupedW4A8Linear)
    nn.Module.__init__(replacement)
    replacement.in_features = template.in_features
    replacement.out_features = template.out_features
    replacement.groups = template.in_features // runtime.GROUP_SIZE
    replacement.input_is_quantized = template.input_is_quantized
    replacement.output_scale = template.output_scale
    replacement.hardware_input_scale = template.hardware_input_scale
    replacement.register_buffer(
        "output_head_scales",
        template.output_head_scales.detach().clone(),
    )
    replacement.register_buffer(
        "hardware_input_scale32_records",
        torch.empty(replacement.groups, dtype=torch.int64),
    )
    return replacement


class DeterministicGroupedProjection(runtime.GroupedW4A8Linear):
    def __init__(self, template: fixed.W4A8Linear) -> None:
        nn.Module.__init__(self)
        self.in_features = template.in_features
        self.out_features = template.out_features
        self.groups = template.in_features // runtime.GROUP_SIZE
        self.input_is_quantized = template.input_is_quantized
        self.output_scale = template.output_scale
        self.hardware_input_scale = template.hardware_input_scale
        self.calls = 0
        self.register_buffer(
            "output_head_scales",
            template.output_head_scales.detach().clone(),
        )
        self.register_buffer(
            "output_scale_per_channel",
            template.output_scale_per_channel.detach().clone(),
        )
        self.register_buffer(
            "hardware_input_scale32_records",
            torch.empty(self.groups, dtype=torch.int64),
        )
        self.bind_hardware_input_scale(template.hardware_input_scale)

    def forward_hardware_input(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros(
            (*inputs.shape[:-1], self.out_features),
            dtype=torch.int8,
            device=inputs.device,
        )

    def accumulator_quantized(self, qinput: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros(
            (qinput.reshape(-1, qinput.shape[-1]).shape[0], self.out_features),
            dtype=torch.int64,
            device=qinput.device,
        )

    def accumulator_grouped_input(
        self,
        qinput: torch.Tensor,
        _input_records: torch.Tensor,
    ) -> torch.Tensor:
        return self.accumulator_quantized(qinput)

    def requantize_accumulator(
        self,
        accumulator: torch.Tensor,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        return torch.zeros(
            (*original_shape, self.out_features),
            dtype=torch.int8,
            device=accumulator.device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw = self.forward_hardware_input(inputs)
        return raw.to(inputs.dtype) * self.output_scale_per_channel.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )


class GroupedW4A8PayloadConformanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = hardware.load_descriptor("qwen2.5-0.5b")

    def test_grouped_projection_initializes_from_source_and_template(self) -> None:
        source = nn.Linear(32, 2, bias=True, dtype=torch.float64)
        template = runtime_template(32, 2)

        projection = runtime.GroupedW4A8Linear(source, template)

        self.assertEqual(tuple(projection.qweight.shape), (2, 1, 32))
        self.assertEqual(tuple(projection.weight_scale32_records.shape), (2, 1))
        self.assertEqual(tuple(projection.bias_output.shape), (2,))
        self.assertEqual(projection.hardware_input_scale32_records.numel(), 1)

    def test_projection_grouping_is_derived_from_model_descriptor(self) -> None:
        group_size = self.descriptor["weight_layout"]["input_group_size"]
        self.assertEqual(group_size, 32)
        self.assertEqual(
            conformance.projection_input_channels(self.descriptor, "q_proj")
            // group_size,
            28,
        )
        self.assertEqual(
            conformance.projection_input_channels(self.descriptor, "down_proj")
            // group_size,
            152,
        )

    def test_payload_is_low_nibble_first_with_little_endian_scale32(self) -> None:
        group_size = self.descriptor["weight_layout"]["input_group_size"]
        weights = [list(range(-8, 8)) * 2]
        first_record = quality.pack_scale32(0x9234, -3)
        second_record = quality.pack_scale32(0xABCD, 2)
        payload = conformance.encode_payload(
            self.descriptor,
            group_size,
            weights,
            [[first_record]],
        )
        self.assertEqual(payload[conformance.HEADER.size], 0x98)
        scale_offset = conformance.HEADER.size + group_size // 2
        self.assertEqual(
            payload[scale_offset : scale_offset + 4],
            struct.pack("<I", first_record),
        )

        two_group_payload = conformance.encode_payload(
            self.descriptor,
            group_size * 2,
            [weights[0] * 2],
            [[first_record, second_record]],
        )
        decoded = conformance.decode_payload(two_group_payload, self.descriptor)
        self.assertEqual(decoded["weights"], [weights[0] * 2])
        self.assertEqual(decoded["scale32_records"], [[first_record, second_record]])

    def test_decoded_projection_is_exact_across_multiple_groups(self) -> None:
        group_size = self.descriptor["weight_layout"]["input_group_size"]
        input_channels = group_size * 3
        activations = [((index * 9) % 255) - 127 for index in range(input_channels)]
        weights = [
            [((index * 3 + output) % 16) - 8 for index in range(input_channels)]
            for output in range(2)
        ]
        records = [
            [
                quality.pack_scale32(0x8000 + output * 100 + group, -group)
                for group in range(3)
            ]
            for output in range(2)
        ]
        payload = conformance.encode_payload(
            self.descriptor, input_channels, weights, records
        )
        decoded = conformance.decode_payload(payload, self.descriptor)
        observed = conformance.evaluate_projection(
            activations,
            decoded["weights"],
            decoded["scale32_records"],
            group_size,
        )
        expected = []
        for row, row_records in zip(weights, records, strict=True):
            total = Fraction(0)
            for group, record in enumerate(row_records):
                start = group * group_size
                dot = sum(
                    activations[index] * row[index]
                    for index in range(start, start + group_size)
                )
                numerator, denominator = quality.scale32_ratio(record)
                total += Fraction(dot * numerator, denominator)
            expected.append(total)
        self.assertEqual(observed, expected)

    def test_malformed_payloads_fail_closed(self) -> None:
        group_size = self.descriptor["weight_layout"]["input_group_size"]
        record = quality.pack_scale32(0x8000, 0)
        payload = conformance.encode_payload(
            self.descriptor,
            group_size * 2,
            [[0] * (group_size * 2)],
            [[record, record]],
        )
        malformed = {}
        malformed["truncated"] = payload[:-1]
        malformed["trailing"] = payload + b"\0"
        bad_magic = bytearray(payload)
        bad_magic[0] ^= 0xFF
        malformed["magic"] = bytes(bad_magic)
        bad_group_size = bytearray(payload)
        struct.pack_into("<H", bad_group_size, 20, 64)
        malformed["descriptor layout"] = bytes(bad_group_size)
        bad_reserved = bytearray(payload)
        struct.pack_into("<H", bad_reserved, 26, 1)
        malformed["reserved header"] = bytes(bad_reserved)
        bad_scale = bytearray(payload)
        scale_offset = conformance.HEADER.size + group_size
        struct.pack_into("<I", bad_scale, scale_offset, 0x01008000)
        malformed["reserved Scale32 byte"] = bytes(bad_scale)
        for name, value in malformed.items():
            with self.subTest(name=name):
                with self.assertRaises(conformance.PayloadError):
                    conformance.decode_payload(value, self.descriptor)
        with self.assertRaisesRegex(conformance.PayloadError, "input geometry"):
            conformance.decode_payload(
                payload,
                self.descriptor,
                expected_input_channels=group_size * 3,
            )

    def test_production_runtime_loads_exact_multi_group_q_and_down_payloads(
        self,
    ) -> None:
        output_channels = 2
        model = PayloadProjectionModel(output_channels, output_channels)
        payloads = {}
        expected_outputs = {}
        activations_by_projection = {}
        input_records_by_projection = {}
        unit_record = quality.pack_scale32(0x8000, 0)
        scale_records = (
            quality.pack_scale32(0x8000, 0),
            quality.pack_scale32(0x8000, -1),
            quality.pack_scale32(0x8000, -2),
        )
        for projection in ("q_proj", "down_proj"):
            input_channels = conformance.projection_input_channels(
                self.descriptor, projection
            )
            group_count = input_channels // runtime.GROUP_SIZE
            activations = [0] * input_channels
            weights = [[0] * input_channels for _ in range(output_channels)]
            records = [
                [scale_records[group % len(scale_records)] for group in range(group_count)]
                for _ in range(output_channels)
            ]
            for group in range(group_count):
                lane = group * runtime.GROUP_SIZE
                activations[lane] = 4
                weights[0][lane] = -1 if group & 1 else 1
                weights[1][lane] = 1 if group % 3 else -1
            payloads[projection] = conformance.encode_payload(
                self.descriptor,
                input_channels,
                weights,
                records,
            )
            exact = conformance.evaluate_projection(
                activations,
                weights,
                records,
                runtime.GROUP_SIZE,
            )
            self.assertTrue(all(value.denominator == 1 for value in exact))
            expected_outputs[projection] = [
                max(-128, min(127, int(value))) for value in exact
            ]
            activations_by_projection[projection] = torch.tensor(
                activations, dtype=torch.int8
            ).reshape(1, 1, input_channels)
            input_records_by_projection[projection] = torch.full(
                (1, 1, group_count),
                unit_record,
                dtype=torch.int64,
            )

        runtime.install_payload_candidate(model, payloads, self.descriptor)

        for projection in ("q_proj", "down_proj"):
            with self.subTest(projection=projection):
                module = getattr(model, projection)
                self.assertIsInstance(module, runtime.GroupedW4A8Linear)
                self.assertEqual(
                    module.to_payload(self.descriptor, projection),
                    payloads[projection],
                )
                observed = module.forward_grouped_input(
                    activations_by_projection[projection],
                    input_records_by_projection[projection],
                )
                self.assertEqual(
                    observed.reshape(-1).tolist(),
                    expected_outputs[projection],
                )

    def test_production_runtime_payload_boundary_fails_closed(self) -> None:
        group_size = self.descriptor["weight_layout"]["input_group_size"]
        record = quality.pack_scale32(0x8000, 0)
        payload = conformance.encode_payload(
            self.descriptor,
            896,
            [[0] * 896],
            [[record] * (896 // group_size)],
        )
        template = runtime_template(896, 1)
        malformed = bytearray(payload)
        struct.pack_into("<H", malformed, 26, 1)
        with self.assertRaisesRegex(conformance.PayloadError, "reserved"):
            runtime.GroupedW4A8Linear.from_payload(
                bytes(malformed),
                self.descriptor,
                "q_proj",
                template,
            )
        with self.assertRaisesRegex(conformance.PayloadError, "input geometry"):
            runtime.GroupedW4A8Linear.from_payload(
                payload,
                self.descriptor,
                "down_proj",
                template,
            )
        model = PayloadProjectionModel(1, 1)
        with self.assertRaisesRegex(RuntimeError, "payload set differs"):
            runtime.install_payload_candidate(
                model,
                {"q_proj": payload},
                self.descriptor,
            )
        original_q = model.q_proj
        original_down = model.down_proj
        down_payload = conformance.encode_payload(
            self.descriptor,
            4864,
            [[0] * 4864],
            [[record] * (4864 // group_size)],
        )
        with self.assertRaises(conformance.PayloadError):
            runtime.install_payload_candidate(
                model,
                {
                    "q_proj": payload,
                    "down_proj": down_payload[:-1],
                },
                self.descriptor,
            )
        self.assertIs(model.q_proj, original_q)
        self.assertIs(model.down_proj, original_down)

    def test_payload_installation_enables_fixed_mlp_grouped_dynamic_silu(
        self,
    ) -> None:
        descriptor = copy.deepcopy(self.descriptor)
        descriptor["dimensions"]["hidden_size"] = 64
        descriptor["dimensions"]["intermediate_size"] = 32
        model = PayloadMLPModel()
        unit_record = quality.pack_scale32(0x8000, 0)
        gate_weights = [[0] * 64 for _ in range(32)]
        up_weights = [[0] * 64 for _ in range(32)]
        for output in range(32):
            gate_weights[output][output % 4] = 1
            up_weights[output][4 + output % 4] = 1
        down_weights = [[0] * 32 for _ in range(2)]
        down_weights[0][0] = 1
        down_weights[1][1] = 1
        payloads = {
            "mlp.gate_proj": conformance.encode_payload(
                descriptor,
                64,
                gate_weights,
                [[unit_record, unit_record] for _ in range(32)],
            ),
            "mlp.up_proj": conformance.encode_payload(
                descriptor,
                64,
                up_weights,
                [[unit_record, unit_record] for _ in range(32)],
            ),
            "mlp.down_proj": conformance.encode_payload(
                descriptor,
                32,
                down_weights,
                [[unit_record] for _ in range(2)],
            ),
        }
        hidden_states = torch.zeros((1, 1, 64), dtype=torch.float32)
        hidden_states[0, 0, :8] = torch.tensor(
            [12, -9, 7, 3, 20, 11, -7, 5],
            dtype=torch.float32,
        )

        runtime.install_payload_candidate(model, payloads, descriptor)
        gate = model.mlp.gate_proj.forward_hardware_input(hidden_states)
        up = model.mlp.up_proj.forward_hardware_input(hidden_states)
        self.assertEqual(gate[0, 0, :4].tolist(), [12, -9, 7, 3])
        self.assertEqual(up[0, 0, :4].tolist(), [20, 11, -7, 5])
        gated, records, saturation, elements = runtime.dynamic_silu_groups(
            gate,
            up,
            1.0,
            1.0,
        )
        self.assertEqual(gated[0, 0, :8].tolist(), [127, 0, -39, 11] * 2)
        self.assertEqual(records.tolist(), [[[41269]]])
        for record in records.reshape(-1).tolist():
            quality.unpack_scale32(record)
        expected = model.mlp.down_proj.forward_grouped_input(gated, records)
        self.assertEqual(expected.tolist(), [[[127, 0]]])

        for projection in (
            model.mlp.gate_proj,
            model.mlp.up_proj,
            model.mlp.down_proj,
        ):
            projection.hardware_input_calls = 0
            projection.dynamic_input_calls = 0
        accumulator, observed = model.mlp.forward_components(hidden_states)

        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(accumulator.tolist(), [[[160, 0]]])
        self.assertEqual(accumulator.dtype, torch.int32)
        self.assertEqual(model.mlp.dynamic_saturation_events, saturation)
        self.assertEqual(model.mlp.dynamic_output_elements, elements)
        self.assertEqual(model.mlp.gate_proj.hardware_input_calls, 1)
        self.assertEqual(model.mlp.up_proj.hardware_input_calls, 1)
        self.assertEqual(model.mlp.down_proj.hardware_input_calls, 0)
        self.assertEqual(model.mlp.down_proj.dynamic_input_calls, 1)

    def _attention_payloads(
        self,
        *,
        malformed_projection: str | None = None,
    ) -> dict[str, bytes]:
        unit_record = quality.pack_scale32(0x8000, 0)
        output_channels = {
            "q_proj": 896,
            "k_proj": 128,
            "v_proj": 128,
            "o_proj": 896,
        }
        payloads = {}
        for projection, outputs in output_channels.items():
            weights = []
            for output in range(outputs):
                row = [0] * 896
                row[output] = 1
                weights.append(row)
            payload = conformance.encode_payload(
                self.descriptor,
                896,
                weights,
                [[unit_record] * 28 for _ in range(outputs)],
            )
            payloads[f"self_attn.{projection}"] = (
                payload[:-1] if projection == malformed_projection else payload
            )
        return payloads

    def _decoder_layer_payloads(
        self,
        descriptor: dict,
        *,
        malformed_projection: str | None = None,
    ) -> dict[str, bytes]:
        unit_record = quality.pack_scale32(0x8000, 0)
        geometries = {
            "self_attn.q_proj": (896, 896),
            "self_attn.k_proj": (896, 128),
            "self_attn.v_proj": (896, 128),
            "self_attn.o_proj": (896, 896),
            "mlp.gate_proj": (896, 32),
            "mlp.up_proj": (896, 32),
            "mlp.down_proj": (32, 896),
        }
        payloads = {}
        for projection, (inputs, outputs) in geometries.items():
            weights = [[0] * inputs for _ in range(outputs)]
            for output, row in enumerate(weights):
                row[output % inputs] = -1 if output % 11 == 0 else 1
            payload = conformance.encode_payload(
                descriptor,
                inputs,
                weights,
                [[unit_record] * (inputs // runtime.GROUP_SIZE) for _ in range(outputs)],
            )
            name = f"layer.{projection}"
            payloads[name] = (
                payload[:-1] if projection == malformed_projection else payload
            )
        return payloads

    def test_payload_installation_executes_all_fixed_attention_projections(
        self,
    ) -> None:
        model = PayloadAttentionModel()
        attention = model.self_attn
        expected_q_metadata = attention.query_projection_scales.clone()
        expected_k_metadata = attention.key_projection_scales.clone()
        expected_score_multiplier = attention.score_multiplier.clone()

        runtime.install_payload_candidate(
            model,
            self._attention_payloads(),
            self.descriptor,
        )

        hidden_states = torch.stack(
            (
                (torch.arange(896, dtype=torch.int16) % 7) - 3,
                (torch.arange(896, dtype=torch.int16) % 11) - 5,
            ),
            dim=0,
        ).to(torch.float32).unsqueeze(0)
        cos = torch.ones((1, 2, 64), dtype=torch.float64)
        sin = torch.zeros((1, 2, 64), dtype=torch.float64)
        observed, cache = attention(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )

        query = hidden_states.to(torch.int8).reshape(1, 2, 14, 64).transpose(1, 2)
        key = hidden_states[..., :128].to(torch.int8).reshape(
            1, 2, 2, 64
        ).transpose(1, 2)
        value = hidden_states[..., :128].to(torch.int8).reshape(
            1, 2, 2, 64
        ).transpose(1, 2)
        query, _ = fixed.fixed_rope_raw_with_saturation(
            query,
            attention.rope_conversion_q9,
            cos,
            sin,
            attention.rope_output_bits,
        )
        key, _ = fixed.fixed_rope_raw_with_saturation(
            key,
            attention.rope_conversion_q9,
            cos,
            sin,
            attention.rope_output_bits,
        )
        key = fixed.repeat_kv(key, attention.num_key_value_groups)
        value = fixed.repeat_kv(value, attention.num_key_value_groups)
        scores = fixed.fixed_attention_scores_raw(
            query,
            key,
            None,
            attention.score_multiplier.reshape(1, 14, 1, 1),
            attention.score_right_shift.reshape(1, 14, 1, 1),
        )
        probabilities = fixed.fixed_softmax_raw(scores)
        expected = fixed.fixed_attention_value_raw(
            probabilities,
            value,
        ).transpose(1, 2).contiguous().reshape(1, 2, 896).to(torch.float32)

        self.assertIsNone(cache)
        self.assertTrue(torch.equal(observed, expected))
        self.assertTrue(
            torch.equal(attention.query_projection_scales, expected_q_metadata)
        )
        self.assertTrue(
            torch.equal(attention.key_projection_scales, expected_k_metadata)
        )
        self.assertTrue(
            torch.equal(attention.score_multiplier, expected_score_multiplier)
        )
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(attention, projection)
            self.assertIsInstance(module, runtime.GroupedW4A8Linear)
            self.assertEqual(module.hardware_input_calls, 1)

    def test_malformed_fixed_attention_payload_rolls_back_atomically(self) -> None:
        model = PayloadAttentionModel()
        attention = model.self_attn
        original_projections = tuple(
            getattr(attention, projection)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        original_metadata = (
            attention.query_projection_scales.clone(),
            attention.key_projection_scales.clone(),
            attention.score_multiplier.clone(),
            attention.score_right_shift.clone(),
        )

        with self.assertRaisesRegex(conformance.PayloadError, "length"):
            runtime.install_payload_candidate(
                model,
                self._attention_payloads(malformed_projection="o_proj"),
                self.descriptor,
            )

        self.assertEqual(
            tuple(
                getattr(attention, projection)
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            ),
            original_projections,
        )
        for observed, expected in zip(
            (
                attention.query_projection_scales,
                attention.key_projection_scales,
                attention.score_multiplier,
                attention.score_right_shift,
            ),
            original_metadata,
            strict=True,
        ):
            self.assertTrue(torch.equal(observed, expected))

    def test_malformed_fixed_mlp_payload_rolls_back_execution_and_projections(
        self,
    ) -> None:
        descriptor = copy.deepcopy(self.descriptor)
        descriptor["dimensions"]["hidden_size"] = 64
        descriptor["dimensions"]["intermediate_size"] = 32
        model = PayloadMLPModel()
        unit_record = quality.pack_scale32(0x8000, 0)
        payloads = {
            "mlp.gate_proj": conformance.encode_payload(
                descriptor,
                64,
                [[0] * 64 for _ in range(32)],
                [[unit_record, unit_record] for _ in range(32)],
            ),
            "mlp.up_proj": conformance.encode_payload(
                descriptor,
                64,
                [[0] * 64 for _ in range(32)],
                [[unit_record, unit_record] for _ in range(32)],
            ),
            "mlp.down_proj": conformance.encode_payload(
                descriptor,
                32,
                [[0] * 32 for _ in range(2)],
                [[unit_record] for _ in range(2)],
            )[:-1],
        }
        original_projections = (
            model.mlp.gate_proj,
            model.mlp.up_proj,
            model.mlp.down_proj,
        )
        original_forward = model.mlp.forward_components.__func__

        with self.assertRaisesRegex(conformance.PayloadError, "length"):
            runtime.install_payload_candidate(model, payloads, descriptor)

        self.assertEqual(
            (
                model.mlp.gate_proj,
                model.mlp.up_proj,
                model.mlp.down_proj,
            ),
            original_projections,
        )
        self.assertIs(model.mlp.forward_components.__func__, original_forward)
        self.assertFalse(hasattr(model.mlp, "dynamic_saturation_events"))
        self.assertFalse(hasattr(model.mlp, "dynamic_output_elements"))

    def test_payload_installed_fixed_decoder_layer_executes_all_seven_projections(
        self,
    ) -> None:
        descriptor = copy.deepcopy(self.descriptor)
        descriptor["dimensions"]["intermediate_size"] = 32
        model = PayloadDecoderLayerModel()
        projection_names = (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
        original_projections = tuple(
            model.get_submodule(f"layer.{name}") for name in projection_names
        )
        payloads = self._decoder_layer_payloads(
            descriptor,
            malformed_projection="mlp.down_proj",
        )

        with self.assertRaisesRegex(conformance.PayloadError, "length"):
            runtime.install_payload_candidate(model, payloads, descriptor)
        self.assertEqual(
            tuple(model.get_submodule(f"layer.{name}") for name in projection_names),
            original_projections,
        )

        payloads = self._decoder_layer_payloads(descriptor)
        runtime.install_payload_candidate(model, payloads, descriptor)
        layer = model.layer
        hidden_states = (
            ((torch.arange(2 * 896, dtype=torch.int16) * 5) % 13) - 6
        ).to(torch.float32).reshape(1, 2, 896)
        position_embeddings = (
            torch.ones((1, 2, 64), dtype=torch.float64),
            torch.zeros((1, 2, 64), dtype=torch.float64),
        )

        first_norm = layer.input_layernorm(hidden_states)
        attention_output, _ = layer.self_attn(
            first_norm,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        post_attention = fixed.fixed_residual_add(
            hidden_states,
            attention_output,
            layer.post_attention_scale,
        )
        second_norm = layer.post_attention_layernorm(post_attention)
        mlp_output = layer.mlp(second_norm)
        expected = fixed.fixed_residual_add(
            post_attention,
            mlp_output,
            layer.post_mlp_scale,
        )

        for name in projection_names:
            projection = model.get_submodule(f"layer.{name}")
            self.assertIsInstance(projection, runtime.GroupedW4A8Linear)
            projection.hardware_input_calls = 0
            projection.dynamic_input_calls = 0
        layer.mlp.dynamic_saturation_events = 0
        layer.mlp.dynamic_output_elements = 0
        norm_transitions = []
        handles = [
            layer.input_layernorm.register_forward_hook(
                lambda _module, _inputs, output: norm_transitions.append(
                    ("input", output.clone())
                )
            ),
            layer.post_attention_layernorm.register_forward_hook(
                lambda _module, _inputs, output: norm_transitions.append(
                    ("post_attention", output.clone())
                )
            ),
        ]
        try:
            observed = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )
            repeated = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertTrue(torch.equal(observed, expected))
        self.assertTrue(torch.equal(repeated, observed))
        self.assertEqual(
            [name for name, _output in norm_transitions],
            ["input", "post_attention", "input", "post_attention"],
        )
        self.assertTrue(torch.equal(norm_transitions[0][1], first_norm))
        self.assertTrue(torch.equal(norm_transitions[1][1], second_norm))
        for name in projection_names[:-1]:
            projection = model.get_submodule(f"layer.{name}")
            self.assertEqual(projection.hardware_input_calls, 2, name)
            self.assertEqual(projection.dynamic_input_calls, 0, name)
        down_projection = model.get_submodule("layer.mlp.down_proj")
        self.assertEqual(down_projection.hardware_input_calls, 0)
        self.assertEqual(down_projection.dynamic_input_calls, 2)

    def _full_qwen_payload_names(self) -> tuple[str, ...]:
        return runtime.full_qwen_payload_names()

    def test_full_qwen_install_is_complete_and_deterministically_ordered(
        self,
    ) -> None:
        model = PayloadDecoderStackModel()
        expected_names = self._full_qwen_payload_names()
        payloads = {
            name: name.encode("ascii")
            for name in reversed(expected_names)
        }
        decoded_names = []

        def stage(
            payload: bytes,
            _descriptor: dict,
            _projection: str,
            template: fixed.W4A8Linear,
        ) -> runtime.GroupedW4A8Linear:
            decoded_names.append(payload.decode("ascii"))
            return staged_grouped_projection(template)

        with mock.patch.object(
            runtime.GroupedW4A8Linear,
            "from_payload",
            side_effect=stage,
        ):
            runtime.install_full_qwen_payload_candidate(
                model,
                payloads,
                self.descriptor,
            )

        self.assertEqual(len(expected_names), 169)
        self.assertEqual(len(expected_names[:-1]), 168)
        self.assertEqual(decoded_names, list(expected_names))
        self.assertEqual(
            tuple(
                name
                for name, module in model.named_modules()
                if isinstance(module, runtime.GroupedW4A8Linear)
            ),
            expected_names,
        )
        expected_record = quality.ceil_scale32_from_float(
            model.model.norm.output_scale
        )
        self.assertEqual(
            model.lm_head.hardware_input_scale32_records.tolist(),
            [expected_record] * (896 // runtime.GROUP_SIZE),
        )
        self.assertTrue(model.lm_head.input_is_quantized)
        self.assertTrue(
            all(
                isinstance(layer, fixed.FixedDecoderLayer)
                for layer in model.model.layers
            )
        )

    def test_full_qwen_lm_head_failure_rolls_back_all_projections(
        self,
    ) -> None:
        model = PayloadDecoderStackModel()
        expected_names = self._full_qwen_payload_names()
        payloads = {name: name.encode("ascii") for name in expected_names}
        original_projections = tuple(
            model.get_submodule(name) for name in expected_names
        )
        original_mlp_forwards = tuple(
            layer.mlp.forward_components.__func__ for layer in model.model.layers
        )
        original_norm_metadata = (
            model.model.norm.input_scale,
            model.model.norm.output_scale,
            model.model.norm.scaled_gains_q8.clone(),
        )
        late_failure = expected_names[-1]
        payloads[late_failure] = b"\0"
        decode_payload = runtime.GroupedW4A8Linear.from_payload.__func__

        def stage(
            payload: bytes,
            descriptor: dict,
            projection: str,
            template: fixed.W4A8Linear,
        ) -> runtime.GroupedW4A8Linear:
            if payload == b"\0":
                return decode_payload(
                    runtime.GroupedW4A8Linear,
                    payload,
                    descriptor,
                    projection,
                    template,
                )
            return staged_grouped_projection(template)

        with mock.patch.object(
            runtime.GroupedW4A8Linear,
            "from_payload",
            side_effect=stage,
        ):
            with self.assertRaisesRegex(
                conformance.PayloadError,
                "header is truncated",
            ):
                runtime.install_full_qwen_payload_candidate(
                    model,
                    payloads,
                    self.descriptor,
                )

        self.assertEqual(
            tuple(model.get_submodule(name) for name in expected_names),
            original_projections,
        )
        self.assertEqual(
            tuple(
                layer.mlp.forward_components.__func__
                for layer in model.model.layers
            ),
            original_mlp_forwards,
        )
        self.assertTrue(
            all(
                not hasattr(layer.mlp, "dynamic_saturation_events")
                for layer in model.model.layers
            )
        )
        self.assertEqual(model.model.norm.input_scale, original_norm_metadata[0])
        self.assertEqual(model.model.norm.output_scale, original_norm_metadata[1])
        self.assertTrue(
            torch.equal(
                model.model.norm.scaled_gains_q8,
                original_norm_metadata[2],
            )
        )

    def test_full_qwen_install_rejects_incomplete_model_and_payload_set(
        self,
    ) -> None:
        model = PayloadDecoderStackModel()
        names = self._full_qwen_payload_names()
        payloads = {name: name.encode("ascii") for name in names}
        with self.assertRaisesRegex(RuntimeError, "payload set differs"):
            runtime.install_full_qwen_payload_candidate(
                model,
                {name: payloads[name] for name in names[:-1]},
                self.descriptor,
            )
        with self.assertRaisesRegex(RuntimeError, "exactly 24"):
            runtime.install_full_qwen_payload_candidate(
                PayloadDecoderStackModel(layer_count=23),
                payloads,
                self.descriptor,
            )
        model = PayloadDecoderStackModel()
        model.model.norm = nn.Identity()
        with self.assertRaisesRegex(RuntimeError, "production FixedRMSNorm"):
            runtime.install_full_qwen_payload_candidate(
                model,
                payloads,
                self.descriptor,
            )

    def test_full_qwen_installed_forward_is_deterministic_through_vocab_logits(
        self,
    ) -> None:
        model = PayloadDecoderStackModel()
        names = self._full_qwen_payload_names()
        payloads = {name: name.encode("ascii") for name in names}

        def stage(
            _payload: bytes,
            _descriptor: dict,
            _projection: str,
            template: fixed.W4A8Linear,
        ) -> runtime.GroupedW4A8Linear:
            return DeterministicGroupedProjection(template)

        with mock.patch.object(
            runtime.GroupedW4A8Linear,
            "from_payload",
            side_effect=stage,
        ):
            runtime.install_full_qwen_payload_candidate(
                model,
                payloads,
                self.descriptor,
            )

        hidden_states = (
            (torch.arange(896, dtype=torch.int16) % 9) - 4
        ).to(torch.float32).reshape(1, 1, 896)
        position_embeddings = (
            torch.ones((1, 1, 64), dtype=torch.float64),
            torch.zeros((1, 1, 64), dtype=torch.float64),
        )
        with torch.inference_mode():
            first = model.forward_hidden(hidden_states, position_embeddings)
            second = model.forward_hidden(hidden_states, position_embeddings)

        self.assertEqual(
            tuple(first.shape),
            (1, 1, self.descriptor["dimensions"]["vocab_size"]),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(torch.isfinite(first)))
        self.assertTrue(
            all(
                model.get_submodule(name).calls > 0
                for name in names
            )
        )

    def test_full_qwen_file_reload_preserves_four_step_frozen_generation(
        self,
    ) -> None:
        descriptor = copy.deepcopy(self.descriptor)
        descriptor["dimensions"]["hidden_size"] = 32
        descriptor["dimensions"]["intermediate_size"] = 32
        descriptor["dimensions"]["num_attention_heads"] = 1
        descriptor["dimensions"]["num_key_value_heads"] = 1
        descriptor["dimensions"]["vocab_size"] = 32
        names = self._full_qwen_payload_names()
        unit_record = quality.pack_scale32(0x8000, 0)
        payloads = {}
        for index, name in enumerate(names):
            weights = [[0] * 32 for _ in range(32)]
            for output, row in enumerate(weights):
                row[(output + index) % 32] = -1 if index & 1 else 1
            payloads[name] = conformance.encode_payload(
                descriptor,
                32,
                weights,
                [[unit_record] for _ in range(32)],
            )

        installed = CompactPayloadDecoderStackModel(descriptor)
        runtime.install_full_qwen_payload_candidate(
            installed,
            payloads,
            descriptor,
        )
        suite = json.loads(runtime.SUITE_PATH.read_text())
        frozen_prompt = suite["cases"][0]["prompt"]
        tokenizer = CompactFrozenTokenizer(
            suite["system_prompt"],
            frozen_prompt,
        )
        prompt = {
            "prompt_id": "fresh-package-reload",
            "user": frozen_prompt,
        }
        generation = {
            "canonical_system_message": suite["system_prompt"],
            "clean_up_tokenization_spaces": False,
            "max_new_tokens": 4,
            "termination_token_ids": [2**31 - 1],
        }
        installed_run = evaluator.generate_w4a8(
            installed,
            tokenizer,
            [prompt],
            generation,
        )[0]
        installed_tensor_trace = compact_generation_tensor_trace(
            installed,
            tokenizer,
            prompt,
            generation,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "model.ace2w4m1"
            runtime.export_installed_full_qwen_payload_package(
                installed,
                package_path,
                descriptor,
            )
            package = package_path.read_bytes()
            self.assertEqual(
                package,
                runtime.serialize_installed_full_qwen_payload_package(
                    installed,
                    descriptor,
                ),
            )
            decoded = conformance.decode_payload_package(package, names)
            self.assertEqual(len(decoded), 169)
            self.assertEqual(decoded, payloads)
            with self.assertRaises(FileExistsError):
                runtime.export_installed_full_qwen_payload_package(
                    installed,
                    package_path,
                    descriptor,
                )

            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import copy
import json
import sys
from pathlib import Path

from tools import model_hardware_contract as hardware
from tools import qualify_stage1_w4a8_software as runtime
from tools import w4a8_full_model_evaluator as evaluator
from verification.test_grouped_w4a8_payload_conformance import (
    CompactFrozenTokenizer,
    CompactPayloadDecoderStackModel,
    compact_generation_tensor_trace,
)

descriptor = copy.deepcopy(hardware.load_descriptor("qwen2.5-0.5b"))
descriptor["dimensions"].update({
    "hidden_size": 32,
    "intermediate_size": 32,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "vocab_size": 32,
})
model = CompactPayloadDecoderStackModel(descriptor)
runtime.load_full_qwen_payload_package(model, Path(sys.argv[1]), descriptor)
suite = json.loads(runtime.SUITE_PATH.read_text())
prompt_text = suite["cases"][0]["prompt"]
result = evaluator.generate_w4a8(
    model,
    CompactFrozenTokenizer(suite["system_prompt"], prompt_text),
    [{"prompt_id": "fresh-package-reload", "user": prompt_text}],
    {
        "canonical_system_message": suite["system_prompt"],
        "clean_up_tokenization_spaces": False,
        "max_new_tokens": 4,
        "termination_token_ids": [2**31 - 1],
    },
)[0]
tensor_trace = compact_generation_tensor_trace(
    model,
    CompactFrozenTokenizer(suite["system_prompt"], prompt_text),
    {"prompt_id": "fresh-package-reload", "user": prompt_text},
    {
        "canonical_system_message": suite["system_prompt"],
        "clean_up_tokenization_spaces": False,
        "max_new_tokens": 4,
        "termination_token_ids": [2**31 - 1],
    },
)
print(json.dumps({"run": result, "tensor_trace": tensor_trace}, sort_keys=True))
""",
                    str(package_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(child.returncode, 0, child.stderr)
            reloaded_result = json.loads(child.stdout)
            reloaded_run = reloaded_result["run"]
            reloaded_tensor_trace = reloaded_result["tensor_trace"]
            self.assertEqual(
                reloaded_run["generated_token_ids"],
                installed_run["generated_token_ids"],
            )
            self.assertEqual(len(reloaded_run["generated_token_ids"]), 4)
            self.assertEqual(
                reloaded_tensor_trace["generated_token_ids"],
                installed_tensor_trace["generated_token_ids"],
            )
            self.assertEqual(len(reloaded_tensor_trace["steps"]), 4)
            for step_index, (reloaded_step, installed_step) in enumerate(
                zip(
                    reloaded_tensor_trace["steps"],
                    installed_tensor_trace["steps"],
                    strict=True,
                )
            ):
                with self.subTest(step=step_index):
                    self.assertEqual(
                        reloaded_step["integer_logits"],
                        installed_step["integer_logits"],
                    )
                    self.assertEqual(len(reloaded_step["layers"]), 24)
                    for layer_index, (
                        reloaded_layer,
                        installed_layer,
                    ) in enumerate(
                        zip(
                            reloaded_step["layers"],
                            installed_step["layers"],
                            strict=True,
                        )
                    ):
                        self.assertEqual(
                            reloaded_layer["layer_index"],
                            layer_index,
                        )
                        self.assertEqual(
                            reloaded_layer["key"],
                            installed_layer["key"],
                        )
                        self.assertEqual(
                            reloaded_layer["value"],
                            installed_layer["value"],
                        )
                        self.assertEqual(
                            reloaded_layer["key_scale32"],
                            installed_layer["key_scale32"],
                        )
            self.assertEqual(
                reloaded_run["per_step_integer_logit_tensor_sha256"],
                installed_run["per_step_integer_logit_tensor_sha256"],
            )
            self.assertEqual(
                reloaded_run["per_step_kv_cache"],
                installed_run["per_step_kv_cache"],
            )
            self.assertEqual(
                reloaded_run["per_step_kv_cache_sha256"],
                installed_run["per_step_kv_cache_sha256"],
            )
            input_length = len(installed_run["input_token_ids"])
            for step_index, cache in enumerate(reloaded_run["per_step_kv_cache"]):
                self.assertEqual(len(cache["layers"]), 24)
                self.assertEqual(
                    [layer["layer_index"] for layer in cache["layers"]],
                    list(range(24)),
                )
                self.assertEqual(
                    {layer["sequence_length"] for layer in cache["layers"]},
                    {input_length + step_index},
                )
                for layer in cache["layers"]:
                    self.assertRegex(layer["key_sha256"], r"^[0-9a-f]{64}$")
                    self.assertRegex(layer["value_sha256"], r"^[0-9a-f]{64}$")
                    self.assertRegex(
                        layer["key_scale32_sha256"],
                        r"^[0-9a-f]{64}$",
                    )

            reordered_names = names[1:2] + names[:1] + names[2:]
            malformed_packages = {
                "truncated": package[:-1],
                "trailing": package + b"\0",
                "reordered": conformance.encode_payload_package(
                    payloads,
                    reordered_names,
                ),
                "malformed projection": conformance.encode_payload_package(
                    {**payloads, names[-1]: payloads[names[-1]][:-1]},
                    names,
                ),
            }
            for case, malformed in malformed_packages.items():
                with self.subTest(case=case):
                    malformed_path = root / f"{case}.ace2w4m1"
                    malformed_path.write_bytes(malformed)
                    untouched = CompactPayloadDecoderStackModel(descriptor)
                    original_projections = tuple(
                        untouched.get_submodule(name) for name in names
                    )
                    with self.assertRaises(
                        (conformance.PayloadError, RuntimeError)
                    ):
                        runtime.load_full_qwen_payload_package(
                            untouched,
                            malformed_path,
                            descriptor,
                        )
                    self.assertEqual(
                        tuple(untouched.get_submodule(name) for name in names),
                        original_projections,
                    )
                    self.assertTrue(
                        all(
                            not hasattr(layer.mlp, "dynamic_saturation_events")
                            for layer in untouched.model.layers
                        )
                    )

    def test_atomic_package_write_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "model.ace2w4m1"
            original_write = runtime.os.write
            call_count = 0

            def fail_after_partial_write(
                descriptor: int,
                payload: memoryview,
            ) -> int:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return original_write(descriptor, payload[:7])
                raise OSError("injected package write failure")

            with mock.patch.object(runtime.os, "write", fail_after_partial_write):
                with self.assertRaisesRegex(
                    OSError,
                    "injected package write failure",
                ):
                    runtime._write_atomic_exclusive_bytes(
                        destination,
                        b"deterministic-package-bytes",
                    )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_local_conformance_cli_is_deterministic(self) -> None:
        command = [
            sys.executable,
            "tools/grouped_w4a8_payload_conformance.py",
            "--check",
        ]
        first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        self.assertEqual(first.stdout, second.stdout)
        report = json.loads(first.stdout)
        self.assertEqual(
            report["status"], "PASS_GROUPED_W4A8_PAYLOAD_CONFORMANCE"
        )
        self.assertEqual(
            [(case["projection"], case["group_count"]) for case in report["cases"]],
            [("q_proj", 28), ("down_proj", 152)],
        )
        self.assertNotIn(str(ROOT).encode(), first.stdout)


if __name__ == "__main__":
    unittest.main()
