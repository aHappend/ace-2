#!/usr/bin/env python3
"""Qualify one immutable, software-only Stage-1 W4A8 repair candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from tools import ace2_full_model_fixed_point as fixed
    from tools import grouped_w4a8_payload_conformance as grouped_payload
    from tools.ace2_quality_contracts import (
        ceil_scale32_from_float,
        scale32_ratio,
        unpack_scale32,
    )
else:
    import ace2_full_model_fixed_point as fixed
    import grouped_w4a8_payload_conformance as grouped_payload
    from ace2_quality_contracts import (
        ceil_scale32_from_float,
        scale32_ratio,
        unpack_scale32,
    )
from ace2_stage1_chat_product import readable_output_acceptance
from qwen_instruct_option_b import (
    REPOSITORY,
    REVISION,
    ROOT,
    SNAPSHOT,
    verify_source_snapshot,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM, reconstruct_fixed_model


SUITE_PATH = ROOT / "verification/fixtures/stage1_w4a8_software_quality_suite.json"
CALIBRATION_PATH = ROOT / "verification/fixtures/stage1_w4a8_software_calibration.json"
ACCEPTED_REPORT = (
    ROOT
    / "build/ace2_chat_demo/stage1-coherence-offline-repair-20260821T160620Z"
    / "root_cause_repair_report.json"
)
ACCEPTED_REPORT_SHA256 = "0b44973034dd78aef0d954e8406faa0aa24de5dedcd48ad86a0add0cd8f7f1c0"
DERIVED_SCALES_PATH = (
    ROOT
    / "evidence/verification/qwen2.5-0.5b-instruct-w4a8-calibration-v1"
    / "derived_scales.json"
)
SMOKE_TOOL_PATH = ROOT / "tools/smoke_stage1_w4a8_candidate.py"
GROUP_SIZE = 32
CANDIDATE_ID = "w4a8-g32-dynamic-silu-scale32-v1"
TERMINATION_TOKEN_IDS = frozenset((151643, 151645))
PROJECTION_FAMILIES = frozenset(
    ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
)
DECODER_PROJECTION_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
FULL_QWEN_DECODER_LAYERS = 24
FULL_QWEN_DECODER_PROJECTIONS = (
    FULL_QWEN_DECODER_LAYERS * len(DECODER_PROJECTION_PATHS)
)
FULL_QWEN_PAYLOAD_PROJECTIONS = FULL_QWEN_DECODER_PROJECTIONS + 1
PROJECTION_AXES = {
    "accumulated_dots": ["row", "output_channel", "input_group"],
    "activations": ["batch", "sequence", "input_channel"],
    "input_scale32_records": ["batch", "sequence", "input_group"],
    "output": ["batch", "sequence", "output_channel"],
    "packed_signed_int4_weights": [
        "output_channel",
        "input_group",
        "packed_group_lane",
    ],
    "requantization_multiplier": ["row", "output_channel", "input_group"],
    "requantization_right_shift": ["row", "output_channel", "input_group"],
    "signed_int4_weights": ["output_channel", "input_group", "group_lane"],
    "weight_scales": ["output_channel", "input_group"],
    "weight_scale32_records": ["output_channel", "input_group"],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, logical_path: Path | None = None) -> dict[str, Any]:
    recorded_path = path if logical_path is None else logical_path
    return {
        "bytes": path.stat().st_size,
        "path": recorded_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        payload = memoryview(canonical_bytes(value))
        while payload:
            payload = payload[os.write(descriptor, payload) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic_exclusive_bytes(path: Path, payload: bytes) -> None:
    require(isinstance(path, Path), "package destination must be a Path")
    require(type(payload) is bytes, "package payload must be bytes")
    require(path.parent.is_dir(), "package destination directory is missing")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "package write made no progress")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary_path, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def validate_frozen_inputs(suite: dict[str, Any], calibration: dict[str, Any]) -> None:
    require(suite["system_prompt"] == DEFAULT_SYSTEM, "canonical system prompt differs")
    require(len(suite["cases"]) >= 6, "qualification suite is not diverse")
    require(
        len({case["instruction_type"] for case in suite["cases"]}) >= 6,
        "qualification instruction types are not diverse",
    )
    require(
        sum(case["prompt"] == "What are registers used for?" for case in suite["cases"]) == 1,
        "accepted rejected-prompt regression must appear exactly once",
    )
    policy = suite["freeze_policy"]
    require(policy["candidate_output_inspection_before_freeze"] is False, "suite freeze leaked")
    require(policy["candidate_specific_prompt_editing"] is False, "candidate prompt editing enabled")
    require(policy["per_prompt_scales"] is False, "per-prompt scales enabled")
    require(policy["generated_tokens"] == 4 and policy["sampling"] is False, "generation policy differs")
    require(len(calibration["records"]) >= 8, "calibration source is not diverse")


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT), "immutable artifact path is outside the repository")
    return resolved.relative_to(ROOT).as_posix()


def qualification_command(
    freeze_manifest_path: Path,
    output_root: Path,
    smoke_evidence_path: Path,
    task_id: str,
) -> str:
    return (
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:tools "
        "./.venv/bin/python tools/qualify_stage1_w4a8_software.py "
        f"--freeze-manifest {repository_path(freeze_manifest_path)} "
        f"--output-dir {repository_path(output_root)} "
        f"--smoke-evidence {repository_path(smoke_evidence_path)} --task-id {task_id}"
    )


def validate_smoke_evidence(smoke_evidence_path: Path) -> dict[str, Any]:
    require(smoke_evidence_path.is_file(), "candidate installation smoke evidence is missing")
    evidence = load_json(smoke_evidence_path)
    require(evidence.get("schema_version") == 1, "candidate installation smoke schema differs")
    require(evidence.get("status") == "PASSED", "candidate installation smoke did not pass")
    require(
        set(evidence.get("projection_families", {})) == PROJECTION_FAMILIES,
        "candidate installation smoke projection coverage differs",
    )
    require(
        all(
            evidence["projection_families"][name]["call_count"] > 0
            for name in PROJECTION_FAMILIES
        ),
        "candidate installation smoke did not call every projection family",
    )
    require(
        evidence.get("forward", {}).get("finite_logits") is True,
        "candidate installation smoke lacks a finite logit step",
    )
    require(
        evidence.get("malformed_contract", {}).get("all_rejected") is True,
        "candidate installation smoke did not fail closed",
    )
    require(
        evidence.get("model_source") == verify_source_snapshot(),
        "candidate installation smoke model source differs",
    )
    require(
        evidence.get("qualification_source") == file_record(Path(__file__).resolve()),
        "candidate installation smoke qualification source differs",
    )
    require(
        evidence.get("smoke_source") == file_record(SMOKE_TOOL_PATH),
        "candidate installation smoke source differs",
    )
    return file_record(smoke_evidence_path)


def frozen_boundary(
    suite: dict[str, Any],
    calibration: dict[str, Any],
    freeze_manifest_path: Path,
    output_root: Path,
    smoke_evidence_path: Path,
    task_id: str,
) -> dict[str, Any]:
    validate_frozen_inputs(suite, calibration)
    require(task_id.startswith("stage1-w4a8-software-quality-"), "qualification task identity differs")
    require(not task_id.endswith("20260821t162637z"), "first failed task identity was reused")
    require(not task_id.endswith("20260821t163315z"), "second failed task identity was reused")
    require(not task_id.endswith("20260821t165500z"), "third failed task identity was reused")
    smoke_evidence = validate_smoke_evidence(smoke_evidence_path)
    source_paths = (
        Path(__file__).resolve(),
        SMOKE_TOOL_PATH,
        ROOT / "tools/ace2_full_model_fixed_point.py",
        ROOT / "tools/ace2_quality_contracts.py",
        ROOT / "tools/ace2_stage1_chat_product.py",
        ROOT / "tools/qwen_instruct_option_b.py",
        ROOT / "tools/qwen_instruct_w4a8_oracle.py",
    )
    source_snapshot = verify_source_snapshot()
    return {
        "accepted_root_cause_report": file_record(ACCEPTED_REPORT),
        "anti_leakage_policy": suite["freeze_policy"],
        "calibration": file_record(CALIBRATION_PATH),
        "calibration_record_count": len(calibration["records"]),
        "candidate_qualification_inference_performed_before_freeze": False,
        "candidate_smoke_performed_before_freeze": True,
        "candidate_matrix": [
            {
                "activation": "signed int8 with Scale32 group conversion",
                "candidate_id": CANDIDATE_ID,
                "group_size": GROUP_SIZE,
                "rounding": "ties_to_even",
                "saturation": "signed_int8",
                "weight": "signed packed int4",
            }
        ],
        "canonical_chat_template": {
            "add_generation_prompt": True,
            "sha256": source_snapshot["chat_template_sha256"],
        },
        "derived_scale_source": file_record(DERIVED_SCALES_PATH),
        "evaluator": file_record(ROOT / "tools/ace2_stage1_chat_product.py"),
        "execution": {
            "exact_command": qualification_command(
                freeze_manifest_path,
                output_root,
                smoke_evidence_path,
                task_id,
            ),
            "freeze_manifest": repository_path(freeze_manifest_path),
            "output_dir": repository_path(output_root),
            "smoke_evidence": repository_path(smoke_evidence_path),
            "task_id": task_id,
        },
        "generation_policy": {
            "add_generation_prompt": True,
            "generated_tokens": 4,
            "greedy_argmax": True,
            "sampling": False,
            "use_cache": False,
        },
        "model": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "source_files": source_snapshot,
        },
        "schema_version": 2,
        "smoke_prerequisite": smoke_evidence,
        "software_sources": {
            path.relative_to(ROOT).as_posix(): file_record(path) for path in source_paths
        },
        "status": "FROZEN_BEFORE_CANDIDATE_INFERENCE",
        "suite": file_record(SUITE_PATH),
        "suite_case_count": len(suite["cases"]),
    }


def scale32_value(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def pack_signed_int4(qweight: Tensor) -> Tensor:
    require(qweight.dtype == torch.int8, "signed-int4 weights are not int8 codes")
    require(
        qweight.ndim == 3 and qweight.shape[2] == GROUP_SIZE,
        "signed-int4 weight geometry differs",
    )
    require(
        bool(torch.all((qweight >= -8) & (qweight <= 7))),
        "signed-int4 weight code is outside -8..7",
    )
    codes = (qweight.to(torch.int16) & 0xF).to(torch.uint8)
    return codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)


def unpack_signed_int4(packed: Tensor, output_channels: int, groups: int) -> Tensor:
    require(packed.dtype == torch.uint8, "packed signed-int4 payload is not uint8")
    require(
        packed.shape == (output_channels, groups, GROUP_SIZE // 2),
        "packed signed-int4 geometry differs",
    )
    codes = torch.empty(
        (output_channels, groups, GROUP_SIZE),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[:, :, 0::2] = packed & 0xF
    codes[:, :, 1::2] = packed >> 4
    signed = codes.to(torch.int8)
    signed[signed >= 8] -= 16
    return signed


class GroupedW4A8Linear(fixed.W4A8Linear):
    """Per-32 W4 projection with Scale32 A8 group conversion and one final saturation."""

    def __init__(self, source: nn.Linear, template: fixed.W4A8Linear) -> None:
        nn.Module.__init__(self)
        require(source.in_features % GROUP_SIZE == 0, "linear input width is not divisible by 32")
        groups = source.in_features // GROUP_SIZE
        grouped = source.weight.detach().to(torch.float64).reshape(
            source.out_features, groups, GROUP_SIZE
        )
        raw_scales = grouped.abs().amax(dim=2) / 7.0
        raw_scales = torch.where(raw_scales > 0, raw_scales, torch.ones_like(raw_scales))
        records = torch.tensor(
            [
                [ceil_scale32_from_float(float(value)) for value in output_scales]
                for output_scales in raw_scales.tolist()
            ],
            dtype=torch.int64,
            device=source.weight.device,
        )
        scales = self._scales_from_records(records)
        qweight = torch.round(grouped / scales[:, :, None]).clamp(-8, 7).to(torch.int8)
        self._initialize(
            template,
            qweight,
            records,
            source.bias,
        )

    @classmethod
    def from_payload(
        cls,
        payload: bytes,
        descriptor: Mapping[str, Any],
        projection: str,
        template: fixed.W4A8Linear,
    ) -> GroupedW4A8Linear:
        input_channels = grouped_payload.projection_input_channels(
            descriptor, projection
        )
        decoded = grouped_payload.decode_payload(
            payload,
            descriptor,
            expected_input_channels=input_channels,
            expected_output_channels=template.out_features,
        )
        if decoded["input_channels"] != decoded["group_count"] * GROUP_SIZE:
            raise grouped_payload.PayloadError(
                "projection payload group size differs from the production runtime"
            )
        device = template.output_scale_per_channel.device
        qweight = torch.tensor(
            decoded["weights"],
            dtype=torch.int8,
            device=device,
        ).reshape(
            decoded["output_channels"],
            decoded["group_count"],
            GROUP_SIZE,
        )
        records = torch.tensor(
            decoded["scale32_records"],
            dtype=torch.int64,
            device=device,
        )
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module._initialize(
            template,
            qweight,
            records,
            getattr(template, "_source_bias", None),
        )
        return module

    def to_payload(
        self,
        descriptor: Mapping[str, Any],
        projection: str,
        *,
        expected_output_channels: int | None = None,
    ) -> bytes:
        input_channels = grouped_payload.projection_input_channels(
            descriptor, projection
        )
        require(
            self.in_features == input_channels,
            "grouped projection input geometry differs from the model descriptor",
        )
        if expected_output_channels is not None:
            require(
                self.out_features == expected_output_channels,
                "grouped projection output geometry differs from the model descriptor",
            )
        self._validate_static_layout(validate_values=True)
        return grouped_payload.encode_payload(
            descriptor,
            input_channels,
            self.qweight.reshape(self.out_features, input_channels)
            .detach()
            .cpu()
            .tolist(),
            self.weight_scale32_records.detach().cpu().tolist(),
        )

    @staticmethod
    def _scales_from_records(records: Tensor) -> Tensor:
        return torch.tensor(
            [
                [scale32_value(int(record)) for record in output_records]
                for output_records in records.detach().cpu().tolist()
            ],
            dtype=torch.float64,
            device=records.device,
        )

    def _initialize(
        self,
        template: fixed.W4A8Linear,
        qweight: Tensor,
        records: Tensor,
        source_bias: Tensor | None,
    ) -> None:
        require(
            qweight.ndim == 3 and qweight.shape[2] == GROUP_SIZE,
            "signed-int4 weight geometry differs",
        )
        self.out_features = qweight.shape[0]
        self.groups = qweight.shape[1]
        self.in_features = self.groups * GROUP_SIZE
        self.input_scale = float(template.input_scale)
        self.hardware_input_scale = float(template.hardware_input_scale)
        self.input_is_quantized = bool(template.input_is_quantized)
        self.output_scale = template.output_scale
        self.output_scale32_record = template.output_scale32_record
        self.output_head_size = template.output_head_size
        self.register_buffer("output_head_scales", template.output_head_scales.detach().clone())
        self.register_buffer(
            "output_scale_per_channel",
            template.output_scale_per_channel.detach().to(torch.float64).clone(),
        )
        self.register_buffer("qweight", qweight.detach().clone())
        self.register_buffer("weight_scale32_records", records.detach().clone())
        self.register_buffer("weight_scale", self._scales_from_records(records))
        self.register_buffer(
            "bias_output",
            (
                None
                if source_bias is None
                else torch.round(
                    source_bias.detach().to(
                        device=self.output_scale_per_channel.device,
                        dtype=torch.float64,
                    )
                    / self.output_scale_per_channel
                ).to(torch.int64)
            ),
        )
        self.register_buffer(
            "hardware_input_scale32_records",
            torch.empty(
                self.groups,
                dtype=torch.int64,
                device=self.output_scale_per_channel.device,
            ),
        )
        self.saturation_events = 0
        self.output_elements = 0
        self.hardware_input_calls = 0
        self.exact_float_carrier_conversions = 0
        self.bound_scale32_associations = 0
        self.dynamic_input_calls = 0
        self.bind_hardware_input_scale(self.hardware_input_scale)
        self._validate_static_layout(validate_values=True)

    def _validate_static_layout(self, *, validate_values: bool = False) -> None:
        require(
            self.qweight.shape == (self.out_features, self.groups, GROUP_SIZE),
            "signed-int4 weight geometry differs",
        )
        require(self.qweight.dtype == torch.int8, "signed-int4 weights are not int8 codes")
        require(
            self.weight_scale32_records.shape == (self.out_features, self.groups),
            "weight Scale32 record geometry differs",
        )
        require(
            self.weight_scale32_records.dtype == torch.int64,
            "weight Scale32 records are not int64",
        )
        require(
            self.weight_scale.shape == (self.out_features, self.groups),
            "weight scale geometry differs",
        )
        require(
            self.weight_scale.dtype == torch.float64,
            "weight scales are not float64",
        )
        require(
            self.output_scale_per_channel.shape == (self.out_features,),
            "output-channel scale geometry differs",
        )
        if validate_values:
            require(
                bool(torch.all((self.qweight >= -8) & (self.qweight <= 7))),
                "signed-int4 weight code is outside -8..7",
            )
            for output_records in self.weight_scale32_records.detach().cpu().tolist():
                for record in output_records:
                    unpack_scale32(record)

    def _validate_projection_layout(
        self,
        qinput: Tensor,
        input_records: Tensor,
    ) -> tuple[int, int]:
        self._validate_static_layout()
        require(qinput.dtype == torch.int8, "grouped projection input is not int8")
        require(qinput.ndim == 3, "activation geometry is not [batch, sequence, input]")
        batch, sequence, input_channels = qinput.shape
        require(input_channels == self.in_features, "activation input width differs")
        require(
            input_records.shape == (batch, sequence, self.groups),
            "dynamic input Scale32 geometry differs",
        )
        require(
            input_records.dtype == torch.int64,
            "dynamic input Scale32 records are not int64",
        )
        for record in input_records.detach().cpu().reshape(-1).tolist():
            unpack_scale32(record)
        return batch, sequence

    def bind_hardware_input_scale32_records(self, records: Tensor) -> None:
        require(records.dtype == torch.int64, "hardware input Scale32 records are not int64")
        require(
            records.shape == (self.groups,),
            "hardware input Scale32 record geometry differs",
        )
        for record in records.detach().cpu().tolist():
            unpack_scale32(record)
        self.hardware_input_scale32_records.copy_(
            records.to(self.hardware_input_scale32_records.device)
        )

    def bind_hardware_input_scale(self, input_scale: float) -> None:
        require(math.isfinite(input_scale) and input_scale > 0, "invalid hardware input scale")
        self.hardware_input_scale = float(input_scale)
        self.bind_hardware_input_scale32_records(
            torch.full(
                (self.groups,),
                ceil_scale32_from_float(input_scale),
                dtype=torch.int64,
                device=self.hardware_input_scale32_records.device,
            )
        )

    def _checked_hardware_input(self, inputs: Tensor) -> Tensor:
        require(inputs.ndim == 3, "activation geometry is not [batch, sequence, input]")
        require(inputs.shape[-1] == self.in_features, "activation input width differs")
        if inputs.dtype == torch.int8:
            return inputs
        require(
            inputs.is_floating_point(),
            "hardware projection input must be signed int8 or an exact float container",
        )
        require(bool(torch.all(torch.isfinite(inputs))), "hardware projection input must be finite")
        require(
            bool(torch.all((inputs >= -128) & (inputs <= 127))),
            "hardware projection input is outside signed int8",
        )
        require(
            torch.equal(inputs, torch.round(inputs)),
            "hardware projection input must contain exact integers",
        )
        self.exact_float_carrier_conversions += 1
        return inputs.to(torch.int8)

    def _bound_hardware_input_records(self, inputs: Tensor) -> Tensor:
        records = getattr(self, "hardware_input_scale32_records", None)
        require(isinstance(records, Tensor), "hardware input Scale32 metadata is missing")
        require(records.dtype == torch.int64, "hardware input Scale32 records are not int64")
        require(
            records.shape == (self.groups,),
            "hardware input Scale32 record geometry differs",
        )
        require(
            records.device == inputs.device,
            "hardware input Scale32 records are on a different device",
        )
        for record in records.detach().cpu().tolist():
            unpack_scale32(record)
        return records.view(1, 1, self.groups).expand(
            inputs.shape[0], inputs.shape[1], self.groups
        )

    def _accumulate(self, qinput: Tensor, input_records: Tensor) -> Tensor:
        batch, sequence = self._validate_projection_layout(qinput, input_records)
        activation_rows = qinput.unflatten(
            -1, (self.groups, GROUP_SIZE)
        ).flatten(0, 1).to(torch.int64)
        input_record_rows = input_records.flatten(0, 1)
        dots = torch.einsum(
            "rgk,ogk->rog",
            activation_rows,
            self.qweight.to(torch.int64),
        )
        expected_metadata_shape = (batch * sequence, self.out_features, self.groups)
        require(dots.shape == expected_metadata_shape, "accumulated dot geometry differs")
        input_scales = torch.tensor(
            [
                [scale32_value(int(record)) for record in row]
                for row in input_record_rows.detach().cpu().tolist()
            ],
            dtype=torch.float64,
            device=dots.device,
        )
        require(
            input_scales.shape == (batch * sequence, self.groups),
            "input scale row geometry differs",
        )
        native = (
            input_scales[:, None, :]
            * self.weight_scale.to(dots.device)[None, :, :]
            / self.output_scale_per_channel.to(dots.device)[None, :, None]
        )
        require(native.shape == expected_metadata_shape, "requantization scale geometry differs")
        multiplier, right_shift = fixed.derive_multiplier(native)
        require(
            multiplier.shape == expected_metadata_shape,
            "requantization multiplier geometry differs",
        )
        require(
            right_shift.shape == expected_metadata_shape,
            "requantization right-shift geometry differs",
        )
        products = dots * multiplier.to(dots.device)
        right_shift = right_shift.to(dots.device)
        common_shift = right_shift.amax(dim=2, keepdim=True)
        shift_delta = common_shift - right_shift
        require(bool(torch.all(shift_delta >= 0)), "common right-shift delta is negative")
        absolute_bound = (
            products.abs().to(torch.float64)
            * torch.exp2(shift_delta.to(torch.float64))
        ).sum(dim=2)
        require(
            float(absolute_bound.amax()) < float(1 << 63),
            "common-shift numerator exceeds signed-64",
        )
        numerator = (products << shift_delta).sum(dim=2)
        converted = fixed.round_shift_even(numerator, common_shift.squeeze(2))
        if self.bias_output is not None:
            converted = converted + self.bias_output.to(converted.device)
        require(
            converted.shape == (batch * sequence, self.out_features),
            "projection accumulator geometry differs",
        )
        require(
            bool(torch.all((converted >= -(1 << 31)) & (converted < (1 << 31)))),
            "projection accumulator exceeds signed-32",
        )
        return converted

    def requantize_accumulator(
        self,
        accumulator: Tensor,
        original_shape: tuple[int, ...],
    ) -> Tensor:
        require(accumulator.dtype == torch.int64, "projection accumulator is not signed int64")
        require(len(original_shape) == 2, "projection output geometry is not [batch, sequence]")
        require(
            accumulator.shape == (math.prod(original_shape), self.out_features),
            "projection accumulator geometry differs",
        )
        require(
            bool(torch.all((accumulator >= -(1 << 31)) & (accumulator < (1 << 31)))),
            "projection accumulator exceeds signed-32",
        )
        self.saturation_events += int(((accumulator < -128) | (accumulator > 127)).sum())
        self.output_elements += accumulator.numel()
        return accumulator.clamp(-128, 127).to(torch.int8).reshape(
            *original_shape, self.out_features
        )

    def _project(self, qinput: Tensor, input_records: Tensor) -> Tensor:
        accumulator = self._accumulate(qinput, input_records)
        return self.requantize_accumulator(accumulator, tuple(qinput.shape[:-1]))

    def _hardware_accumulator(self, qinput: Tensor) -> Tensor:
        records = self._bound_hardware_input_records(qinput)
        accumulator = self._accumulate(qinput, records)
        self.hardware_input_calls += 1
        self.bound_scale32_associations += 1
        return accumulator

    def accumulator_quantized(self, qinput: Tensor) -> Tensor:
        require(qinput.dtype == torch.int8, "projection input must be signed int8")
        checked = self._checked_hardware_input(qinput)
        return self._hardware_accumulator(checked)

    def accumulator_grouped_input(
        self,
        qinput: Tensor,
        input_records: Tensor,
    ) -> Tensor:
        accumulator = self._accumulate(qinput, input_records)
        self.dynamic_input_calls += 1
        return accumulator

    def forward_grouped_input(self, qinput: Tensor, input_records: Tensor) -> Tensor:
        accumulator = self.accumulator_grouped_input(qinput, input_records)
        return self.requantize_accumulator(accumulator, tuple(qinput.shape[:-1]))

    def forward_hardware_input(self, inputs: Tensor) -> Tensor:
        qinput = self._checked_hardware_input(inputs)
        accumulator = self._hardware_accumulator(qinput)
        return self.requantize_accumulator(accumulator, tuple(qinput.shape[:-1]))

    def forward_raw(self, inputs: Tensor) -> Tensor:
        qinput = fixed.quantize_int8(inputs, self.input_scale).to(torch.int8)
        record = ceil_scale32_from_float(self.input_scale)
        records = torch.full(
            (*qinput.shape[:-1], self.groups),
            record,
            dtype=torch.int64,
            device=qinput.device,
        )
        return self._project(qinput, records)

    def forward(self, inputs: Tensor) -> Tensor:
        raw = (
            self.forward_hardware_input(inputs)
            if self.input_is_quantized
            else self.forward_raw(inputs)
        )
        return raw.to(inputs.dtype) * self.output_scale_per_channel.to(inputs.device, inputs.dtype)


def _install_payload_replacements(
    fixed_model: nn.Module,
    payloads: Mapping[str, bytes],
    descriptor: Mapping[str, Any],
    replacements: list[tuple[str, fixed.W4A8Linear]],
    *,
    final_norm: fixed.FixedRMSNorm | None = None,
) -> None:
    expected_names = {name for name, _ in replacements}
    require(set(payloads) == expected_names, "projection payload set differs")
    staged = []
    replacement_by_name = {}
    for name, template in replacements:
        projection = name.rsplit(".", 1)[-1]
        replacement = GroupedW4A8Linear.from_payload(
            payloads[name],
            descriptor,
            projection,
            template,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = fixed_model.get_submodule(parent_name) if parent_name else fixed_model
        staged.append((parent, child_name, replacement))
        replacement_by_name[name] = replacement
    if final_norm is not None:
        lm_head = replacement_by_name.get("lm_head")
        require(
            isinstance(lm_head, GroupedW4A8Linear),
            "full-Qwen grouped lm_head is missing",
        )
        lm_head.bind_hardware_input_scale(final_norm.output_scale)
        lm_head.input_is_quantized = True
        expected_record = ceil_scale32_from_float(final_norm.output_scale)
        require(
            torch.equal(
                lm_head.hardware_input_scale32_records,
                torch.full_like(
                    lm_head.hardware_input_scale32_records,
                    expected_record,
                ),
            ),
            "final RMSNorm-to-lm_head Scale32 metadata differs",
        )
    for name, module in fixed_model.named_modules():
        if not isinstance(module, fixed.FixedAttention):
            continue
        prefix = f"{name}." if name else ""
        projections = tuple(
            replacement_by_name.get(f"{prefix}{projection}")
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        require(
            all(isinstance(projection, GroupedW4A8Linear) for projection in projections),
            "fixed attention grouped projection set differs",
        )
        q_proj, k_proj, v_proj, o_proj = projections
        if q_proj.output_head_scales.numel():
            require(
                torch.equal(
                    q_proj.output_head_scales,
                    module.query_projection_scales.to(
                        device=q_proj.output_head_scales.device
                    ),
                ),
                "fixed attention Q projection metadata differs",
            )
        if k_proj.output_head_scales.numel():
            require(
                torch.equal(
                    k_proj.output_head_scales,
                    module.key_projection_scales.to(
                        device=k_proj.output_head_scales.device
                    ),
                ),
                "fixed attention K projection metadata differs",
            )
        require(
            v_proj.output_scale is not None
            and o_proj.hardware_input_scale == v_proj.output_scale,
            "fixed attention V-to-O scale metadata differs",
        )
    staged_mlps = []
    for name, module in fixed_model.named_modules():
        if not isinstance(module, fixed.FixedMLP):
            continue
        prefix = f"{name}." if name else ""
        projections = tuple(
            replacement_by_name.get(f"{prefix}{projection}")
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        require(
            all(isinstance(projection, GroupedW4A8Linear) for projection in projections),
            "fixed MLP grouped projection set differs",
        )
        staged_mlps.append(module)
    for parent, child_name, replacement in staged:
        setattr(parent, child_name, replacement)
    for module in staged_mlps:
        module.dynamic_saturation_events = 0
        module.dynamic_output_elements = 0
        module.forward_components = types.MethodType(
            grouped_mlp_forward_components, module
        )


def install_payload_candidate(
    fixed_model: nn.Module,
    payloads: Mapping[str, bytes],
    descriptor: Mapping[str, Any],
) -> None:
    replacements = [
        (name, module)
        for name, module in fixed_model.named_modules()
        if isinstance(module, fixed.W4A8Linear)
    ]
    _install_payload_replacements(fixed_model, payloads, descriptor, replacements)


def install_full_qwen_payload_candidate(
    fixed_model: nn.Module,
    payloads: Mapping[str, bytes],
    descriptor: Mapping[str, Any],
) -> None:
    dimensions = descriptor.get("dimensions")
    require(isinstance(dimensions, Mapping), "model descriptor dimensions are missing")
    require(
        dimensions.get("num_hidden_layers") == FULL_QWEN_DECODER_LAYERS,
        "model descriptor must bind exactly 24 decoder layers",
    )
    hidden_size = dimensions.get("hidden_size")
    vocab_size = dimensions.get("vocab_size")
    require(
        type(hidden_size) is int and hidden_size > 0,
        "model descriptor hidden size is invalid",
    )
    require(
        type(vocab_size) is int and vocab_size > 0,
        "model descriptor vocabulary size is invalid",
    )
    model = getattr(fixed_model, "model", None)
    layers = getattr(model, "layers", None)
    require(isinstance(layers, nn.ModuleList), "full-Qwen decoder layers are missing")
    require(
        len(layers) == FULL_QWEN_DECODER_LAYERS,
        "full-Qwen model must contain exactly 24 decoder layers",
    )
    replacements = []
    for layer_index, layer in enumerate(layers):
        require(
            isinstance(layer, fixed.FixedDecoderLayer),
            f"model layer {layer_index} is not a production FixedDecoderLayer",
        )
        require(
            layer.self_attn.layer_idx == layer_index,
            f"model layer {layer_index} execution index differs",
        )
        for projection_path in DECODER_PROJECTION_PATHS:
            template = layer.get_submodule(projection_path)
            require(
                isinstance(template, fixed.W4A8Linear),
                f"model.layers.{layer_index}.{projection_path} is not W4A8",
            )
            replacements.append(
                (
                    f"model.layers.{layer_index}.{projection_path}",
                    template,
                )
            )
    require(
        len(replacements) == FULL_QWEN_DECODER_PROJECTIONS,
        "full-Qwen decoder projection count differs",
    )
    final_norm = getattr(model, "norm", None)
    require(
        isinstance(final_norm, fixed.FixedRMSNorm),
        "full-Qwen final norm is not a production FixedRMSNorm",
    )
    require(
        final_norm.scaled_gains_q8.dtype == torch.int16
        and final_norm.scaled_gains_q8.shape == (hidden_size,),
        "full-Qwen final RMSNorm metadata geometry differs",
    )
    require(
        math.isfinite(final_norm.input_scale)
        and final_norm.input_scale > 0
        and math.isfinite(final_norm.output_scale)
        and final_norm.output_scale > 0,
        "full-Qwen final RMSNorm scale metadata is invalid",
    )
    lm_head = getattr(fixed_model, "lm_head", None)
    require(
        isinstance(lm_head, fixed.W4A8Linear),
        "full-Qwen lm_head is not W4A8",
    )
    require(
        lm_head.in_features == hidden_size and lm_head.out_features == vocab_size,
        "full-Qwen lm_head geometry differs",
    )
    replacements.append(("lm_head", lm_head))
    require(
        len(replacements) == FULL_QWEN_PAYLOAD_PROJECTIONS,
        "full-Qwen projection count differs",
    )
    _install_payload_replacements(
        fixed_model,
        payloads,
        descriptor,
        replacements,
        final_norm=final_norm,
    )


def full_qwen_payload_names() -> tuple[str, ...]:
    return (
        *(
            f"model.layers.{layer_index}.{projection}"
            for layer_index in range(FULL_QWEN_DECODER_LAYERS)
            for projection in DECODER_PROJECTION_PATHS
        ),
        "lm_head",
    )


def serialize_installed_full_qwen_payload_package(
    fixed_model: nn.Module,
    descriptor: Mapping[str, Any],
) -> bytes:
    dimensions = descriptor.get("dimensions")
    require(isinstance(dimensions, Mapping), "model descriptor dimensions are missing")
    require(
        dimensions.get("num_hidden_layers") == FULL_QWEN_DECODER_LAYERS,
        "model descriptor must bind exactly 24 decoder layers",
    )
    hidden_size = dimensions.get("hidden_size")
    intermediate_size = dimensions.get("intermediate_size")
    attention_heads = dimensions.get("num_attention_heads")
    key_value_heads = dimensions.get("num_key_value_heads")
    vocab_size = dimensions.get("vocab_size")
    require(
        all(
            type(value) is int and value > 0
            for value in (
                hidden_size,
                intermediate_size,
                attention_heads,
                key_value_heads,
                vocab_size,
            )
        ),
        "model descriptor projection dimensions are invalid",
    )
    require(
        hidden_size % attention_heads == 0,
        "model descriptor attention head geometry differs",
    )
    key_value_width = key_value_heads * (hidden_size // attention_heads)
    expected_outputs = {
        "q_proj": hidden_size,
        "k_proj": key_value_width,
        "v_proj": key_value_width,
        "o_proj": hidden_size,
        "gate_proj": intermediate_size,
        "up_proj": intermediate_size,
        "down_proj": hidden_size,
        "lm_head": vocab_size,
    }
    model = getattr(fixed_model, "model", None)
    layers = getattr(model, "layers", None)
    require(
        isinstance(layers, nn.ModuleList)
        and len(layers) == FULL_QWEN_DECODER_LAYERS,
        "full-Qwen model must contain exactly 24 decoder layers",
    )
    payloads = {}
    for name in full_qwen_payload_names():
        try:
            module = fixed_model.get_submodule(name)
        except AttributeError as exc:
            raise RuntimeError(f"full-Qwen grouped projection is missing: {name}") from exc
        require(
            isinstance(module, GroupedW4A8Linear),
            f"full-Qwen grouped projection is missing: {name}",
        )
        projection = name.rsplit(".", 1)[-1]
        payloads[name] = module.to_payload(
            descriptor,
            projection,
            expected_output_channels=expected_outputs[projection],
        )
    return grouped_payload.encode_payload_package(
        payloads,
        full_qwen_payload_names(),
    )


def install_full_qwen_payload_package(
    fixed_model: nn.Module,
    package: bytes,
    descriptor: Mapping[str, Any],
) -> None:
    payloads = grouped_payload.decode_payload_package(
        package,
        full_qwen_payload_names(),
    )
    install_full_qwen_payload_candidate(fixed_model, payloads, descriptor)


def export_installed_full_qwen_payload_package(
    fixed_model: nn.Module,
    path: Path,
    descriptor: Mapping[str, Any],
) -> None:
    package = serialize_installed_full_qwen_payload_package(
        fixed_model,
        descriptor,
    )
    _write_atomic_exclusive_bytes(path, package)


def load_full_qwen_payload_package(
    fixed_model: nn.Module,
    path: Path,
    descriptor: Mapping[str, Any],
) -> None:
    require(isinstance(path, Path), "package source must be a Path")
    package = path.read_bytes()
    install_full_qwen_payload_package(fixed_model, package, descriptor)


def dynamic_silu_groups(
    gate: Tensor,
    up: Tensor,
    gate_scale: float,
    up_scale: float,
) -> tuple[Tensor, Tensor, int, int]:
    gate_q9 = torch.round(gate.to(torch.float64) * gate_scale * (1 << 9))
    up_q9 = torch.round(up.to(torch.float64) * up_scale * (1 << 9))
    gate_q9 = gate_q9.clamp(-32768, 32767).to(torch.int64)
    up_q9 = up_q9.clamp(-32768, 32767).to(torch.int64)
    table = torch.tensor(fixed.SILU_LUT, dtype=torch.int64, device=gate.device)
    product = table[(gate_q9 >> 6).clamp(-64, 64) + 64] * up_q9
    grouped = product.reshape(*product.shape[:-1], -1, GROUP_SIZE)
    absmax = grouped.abs().amax(dim=-1)
    records = torch.tensor(
        [
            ceil_scale32_from_float(max(1.0, float(value)) / ((1 << 21) * 127.0))
            for value in absmax.detach().cpu().reshape(-1).tolist()
        ],
        dtype=torch.int64,
        device=product.device,
    ).reshape(absmax.shape)
    scales = torch.tensor(
        [scale32_value(int(value)) for value in records.detach().cpu().reshape(-1).tolist()],
        dtype=torch.float64,
        device=product.device,
    ).reshape(records.shape)
    multiplier, right_shift = fixed.derive_multiplier(1.0 / ((1 << 21) * scales))
    unbounded = fixed.round_shift_even(
        grouped * multiplier[..., None],
        right_shift[..., None],
    )
    saturation = int(((unbounded < -128) | (unbounded > 127)).sum())
    output = unbounded.clamp(-128, 127).to(torch.int8).reshape_as(product)
    return output, records, saturation, output.numel()


def grouped_mlp_forward_components(
    self: fixed.FixedMLP,
    hidden_states: Tensor,
) -> tuple[Tensor, Tensor]:
    gate = self.gate_proj.forward_hardware_input(hidden_states)
    up = self.up_proj.forward_hardware_input(hidden_states)
    gated, records, saturation, elements = dynamic_silu_groups(
        gate,
        up,
        float(self.gate_proj.output_scale),
        float(self.up_proj.output_scale),
    )
    self.dynamic_saturation_events += saturation
    self.dynamic_output_elements += elements
    original_shape = tuple(gated.shape[:-1])
    accumulator = self.down_proj.accumulator_grouped_input(gated, records)
    output = self.down_proj.requantize_accumulator(accumulator, original_shape)
    return accumulator.to(torch.int32).reshape(*original_shape, -1), output


def install_candidate(fixed_model: nn.Module, source_model: nn.Module) -> None:
    source_linears = {
        name: module for name, module in source_model.named_modules() if isinstance(module, nn.Linear)
    }
    replacements = [
        (name, module)
        for name, module in fixed_model.named_modules()
        if isinstance(module, fixed.W4A8Linear)
    ]
    require(set(name for name, _ in replacements) == set(source_linears), "linear sets differ")
    for name, template in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = fixed_model.get_submodule(parent_name) if parent_name else fixed_model
        replacement = GroupedW4A8Linear(source_linears[name], template)
        replacement.bind_hardware_input_scale(template.hardware_input_scale)
        setattr(parent, child_name, replacement)
    for layer in fixed_model.model.layers:
        layer.mlp.dynamic_saturation_events = 0
        layer.mlp.dynamic_output_elements = 0
        layer.mlp.forward_components = types.MethodType(
            grouped_mlp_forward_components, layer.mlp
        )


def render(tokenizer: Any, system: str, prompt: str) -> Tensor:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def top_k(logits: Tensor, count: int = 5) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.to(torch.float64), count)
    return [
        {"logit": float(value), "token_id": int(index)}
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def generate(model: nn.Module, tokenizer: Any, system: str, prompt: str) -> dict[str, Any]:
    sequence = render(tokenizer, system, prompt)
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    layer_outputs: dict[str, Tensor] = {}
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, output, index=index: layer_outputs.__setitem__(
                str(index), output[0].detach().cpu().to(torch.float64)
            )
        )
        for index, layer in enumerate(model.model.layers)
    ]
    try:
        with torch.inference_mode():
            for index in range(4):
                logits = model(input_ids=sequence, use_cache=False).logits[0, -1]
                token = int(torch.argmax(logits))
                generated.append(token)
                steps.append(
                    {
                        "index": index,
                        "selected_token_id": token,
                        "top_k": top_k(logits),
                    }
                )
                sequence = torch.cat((sequence, torch.tensor([[token]], dtype=torch.long)), dim=1)
    finally:
        for hook in hooks:
            hook.remove()
    decoded = tokenizer.decode(
        [token for token in generated if token not in TERMINATION_TOKEN_IDS],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "_layer_final_hidden_values": {
            name: value[0, -1].clone() for name, value in layer_outputs.items()
        },
        "decoded_text": decoded,
        "generated_token_ids": generated,
        "input_token_ids": render(tokenizer, system, prompt).reshape(-1).tolist(),
        "layer_final_hidden": {
            name: {
                "absmax": float(value[0, -1].abs().amax()),
                "sha256": hashlib.sha256(value[0, -1].numpy().tobytes()).hexdigest(),
            }
            for name, value in sorted(layer_outputs.items(), key=lambda item: int(item[0]))
        },
        "steps": steps,
    }


def compare_runs(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    step_comparisons = []
    for bf16, w4a8 in zip(reference["steps"], candidate["steps"], strict=True):
        bf_ids = [item["token_id"] for item in bf16["top_k"]]
        w_ids = [item["token_id"] for item in w4a8["top_k"]]
        step_comparisons.append(
            {
                "argmax_agreement": bf16["selected_token_id"] == w4a8["selected_token_id"],
                "bf16_top_k": bf16["top_k"],
                "top_k_token_overlap": len(set(bf_ids) & set(w_ids)),
                "w4a8_top_k": w4a8["top_k"],
            }
        )
    reference_layers = reference["_layer_final_hidden_values"]
    candidate_layers = candidate["_layer_final_hidden_values"]
    require(set(reference_layers) == set(candidate_layers), "compared layer sets differ")
    layer_errors = []
    for name in sorted(reference_layers, key=int):
        delta = candidate_layers[name] - reference_layers[name]
        layer_errors.append(
            {
                "candidate_absmax": float(candidate_layers[name].abs().amax()),
                "layer": int(name),
                "max_abs_error": float(delta.abs().amax()),
                "mean_abs_error": float(delta.abs().mean()),
                "reference_absmax": float(reference_layers[name].abs().amax()),
                "root_mean_square_error": float(torch.sqrt(torch.mean(delta * delta))),
            }
        )
    earliest = next(
        (entry["layer"] for entry in layer_errors if entry["max_abs_error"] != 0.0),
        None,
    )
    return {
        "earliest_layer_with_activation_divergence": earliest,
        "layer_activation_error": layer_errors,
        "steps": step_comparisons,
    }


def pack_candidate(
    model: nn.Module,
    output_root: Path,
    logical_output_root: Path,
) -> dict[str, Any]:
    payload_path = output_root / "packed-w4.bin"
    scales_path = output_root / "weight-scale32.bin"
    layout: list[dict[str, Any]] = []
    payload_offset = 0
    scale_offset = 0
    with payload_path.open("xb") as payload_file, scales_path.open("xb") as scale_file:
        for name, module in sorted(model.named_modules()):
            if not isinstance(module, GroupedW4A8Linear):
                continue
            module._validate_static_layout(validate_values=True)
            packed = pack_signed_int4(module.qweight.detach().cpu())
            payload = packed.contiguous().numpy().tobytes()
            scale_payload = b"".join(
                struct.pack("<I", int(record))
                for output_records in module.weight_scale32_records.tolist()
                for record in output_records
            )
            payload_file.write(payload)
            scale_file.write(scale_payload)
            layout.append(
                {
                    "groups": module.groups,
                    "in_features": module.in_features,
                    "name": name,
                    "out_features": module.out_features,
                    "packed_weight_axes": PROJECTION_AXES["packed_signed_int4_weights"],
                    "payload_bytes": len(payload),
                    "payload_offset": payload_offset,
                    "scale32_axes": PROJECTION_AXES["weight_scale32_records"],
                    "scale32_bytes": len(scale_payload),
                    "scale32_offset": scale_offset,
                }
            )
            payload_offset += len(payload)
            scale_offset += len(scale_payload)
    return {
        "layout": layout,
        "packed_w4": file_record(payload_path, logical_output_root / payload_path.name),
        "weight_scale32": file_record(scales_path, logical_output_root / scales_path.name),
    }


def verify_round_trip(model: nn.Module, packed: dict[str, Any], output_root: Path) -> None:
    payload = (output_root / "packed-w4.bin").read_bytes()
    scales = (output_root / "weight-scale32.bin").read_bytes()
    modules = dict(model.named_modules())
    for entry in packed["layout"]:
        module = modules[entry["name"]]
        start = entry["payload_offset"]
        raw = torch.frombuffer(
            bytearray(payload[start : start + entry["payload_bytes"]]), dtype=torch.uint8
        )
        packed_weights = raw.unflatten(
            0,
            (entry["out_features"], entry["groups"], GROUP_SIZE // 2),
        )
        signed = unpack_signed_int4(
            packed_weights,
            entry["out_features"],
            entry["groups"],
        )
        require(torch.equal(signed, module.qweight.cpu()), "W4 round trip differs")
        scale_start = entry["scale32_offset"]
        observed = [
            item[0]
            for item in struct.iter_unpack(
                "<I", scales[scale_start : scale_start + entry["scale32_bytes"]]
            )
        ]
        expected = [
            int(record)
            for output_records in module.weight_scale32_records.tolist()
            for record in output_records
        ]
        require(observed == expected, "Scale32 round trip differs")
        for record in observed:
            unpack_scale32(record)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-evidence", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    require(sha256_file(ACCEPTED_REPORT) == ACCEPTED_REPORT_SHA256, "accepted report differs")
    suite = load_json(SUITE_PATH)
    calibration = load_json(CALIBRATION_PATH)
    freeze_manifest_path = args.freeze_manifest.resolve()
    output_root = args.output_dir.resolve()
    smoke_evidence_path = args.smoke_evidence.resolve()
    expected_freeze = frozen_boundary(
        suite,
        calibration,
        freeze_manifest_path,
        output_root,
        smoke_evidence_path,
        args.task_id,
    )
    if args.freeze_only:
        require(not freeze_manifest_path.exists(), "immutable freeze manifest already exists")
        require(not output_root.exists(), "immutable qualification output already exists")
        write_exclusive(freeze_manifest_path, expected_freeze)
        print(f"ACE2_W4A8_SOFTWARE_FREEZE status=FROZEN output={freeze_manifest_path}")
        return 0
    require(freeze_manifest_path.is_file(), "pre-tuning freeze manifest is missing")
    require(
        canonical_bytes(load_json(freeze_manifest_path)) == canonical_bytes(expected_freeze),
        "pre-tuning freeze manifest differs from current inputs",
    )
    require(not output_root.exists(), "immutable qualification output already exists")
    temporary = output_root.with_name("." + output_root.name + ".tmp")
    require(not temporary.exists(), "qualification temporary output exists")
    temporary.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    bf16_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_model = reconstruct_fixed_model(
        load_json(DERIVED_SCALES_PATH)
    ).eval()
    baseline_runs = []
    bf16_runs = []
    for case in suite["cases"]:
        bf16_runs.append(generate(bf16_model, tokenizer, suite["system_prompt"], case["prompt"]))
        baseline_runs.append(generate(fixed_model, tokenizer, suite["system_prompt"], case["prompt"]))
    install_candidate(fixed_model, bf16_model)
    cases: list[dict[str, Any]] = []
    for case, bf16, baseline in zip(
        suite["cases"], bf16_runs, baseline_runs, strict=True
    ):
        candidate = generate(fixed_model, tokenizer, suite["system_prompt"], case["prompt"])
        baseline_comparison = compare_runs(bf16, baseline)
        candidate_comparison = compare_runs(bf16, candidate)
        for run in (bf16, baseline, candidate):
            del run["_layer_final_hidden_values"]
        acceptance = readable_output_acceptance(
            case["prompt"], candidate["decoded_text"], candidate["generated_token_ids"]
        )
        cases.append(
            {
                "bf16": bf16,
                "candidate": candidate,
                "comparison": candidate_comparison,
                "current_w4a8_control": baseline,
                "current_w4a8_control_comparison": baseline_comparison,
                "current_w4a8_control_evaluator": readable_output_acceptance(
                    case["prompt"],
                    baseline["decoded_text"],
                    baseline["generated_token_ids"],
                ),
                "evaluator": acceptance,
                "id": case["id"],
                "instruction_type": case["instruction_type"],
                "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
            }
        )
    packed = pack_candidate(fixed_model, temporary, output_root)
    verify_round_trip(fixed_model, packed, temporary)
    linear_modules = [
        module for module in fixed_model.modules() if isinstance(module, GroupedW4A8Linear)
    ]
    dynamic_mlps = list(fixed_model.model.layers)
    saturation = {
        "linear_output_elements": sum(module.output_elements for module in linear_modules),
        "linear_saturation_events": sum(module.saturation_events for module in linear_modules),
        "silu_output_elements": sum(layer.mlp.dynamic_output_elements for layer in dynamic_mlps),
        "silu_saturation_events": sum(layer.mlp.dynamic_saturation_events for layer in dynamic_mlps),
    }
    saturation["linear_saturation_fraction"] = (
        saturation["linear_saturation_events"] / saturation["linear_output_elements"]
    )
    saturation["silu_saturation_fraction"] = (
        saturation["silu_saturation_events"] / saturation["silu_output_elements"]
    )
    all_passed = all(case["evaluator"]["accepted"] for case in cases)
    report = {
        "backend_loadability": {
            "existing_packed_w4_payload": True,
            "existing_per_output_channel_scale_contract": False,
            "precise_bounded_interface_change": (
                "extend each projection layout record with groups=in_features/32 and "
                "consume one little-endian Scale32 weight record per output-channel/group; "
                "the existing low-nibble-first W4 payload, signed int8 ports, Scale32 "
                "multiplier/right-shift, ties-to-even rounding, and final int8 saturation remain unchanged"
            ),
            "round_trip_verified": True,
        },
        "candidate": {
            "activation": "signed int8, static at existing boundaries and per-token/per-32 after SiLU",
            "candidate_id": CANDIDATE_ID,
            "group_size": GROUP_SIZE,
            "rounding": "ties_to_even",
            "saturation": "signed_int8_after_group_merge",
            "weight": "signed packed int4 per-output-channel/per-32-input-group",
        },
        "canonical_projection_axes": PROJECTION_AXES,
        "cases": cases,
        "frozen_inputs": {
            "accepted_root_cause_report": file_record(ACCEPTED_REPORT),
            "calibration": file_record(CALIBRATION_PATH),
            "calibration_record_count": len(calibration["records"]),
            "pre_tuning_freeze_manifest": file_record(freeze_manifest_path),
            "smoke_prerequisite": file_record(smoke_evidence_path),
            "suite": file_record(SUITE_PATH),
            "suite_case_count": len(suite["cases"]),
        },
        "generation_policy": {
            "add_generation_prompt": True,
            "canonical_chat_template": True,
            "generated_tokens": 4,
            "greedy_argmax": True,
            "sampling": False,
            "use_cache": False,
        },
        "model": {
            "repository": REPOSITORY,
            "revision": REVISION,
        },
        "integer_semantics": {
            "candidate_execution": (
                "signed int4 dot products with signed int8 activations, Scale32-realized "
                "multiplier/right-shift, ties-to-even rounding, and signed int8 saturation"
            ),
            "packed_weight_or_scale_round_trip_mismatch_count": 0,
            "software_hardware_oracle": file_record(
                ROOT / "tools/ace2_full_model_fixed_point.py"
            ),
            "unexplained_mismatch_count": 0,
        },
        "packed_candidate": packed,
        "saturation": saturation,
        "schema_version": 1,
        "scope_guards": {
            "bf16_candidate_or_fallback": False,
            "consuming_rtl_or_icarus_executed": False,
            "network_access_performed": False,
            "ppa_s7_stage2_u280_executed": False,
            "product_prompt_selected": False,
            "stage_transition": "SKIPPED",
        },
        "status": "QUALIFIED_PENDING_INDEPENDENT_L2" if all_passed else "REJECTED_BY_FROZEN_SUITE",
        "unchanged_prompt_aware_evaluator": file_record(
            ROOT / "tools/ace2_stage1_chat_product.py"
        ),
    }
    write_exclusive(temporary / "software_qualification_report.json", report)
    manifest = {
        "candidate_id": CANDIDATE_ID,
        "candidate_status": report["status"],
        "pre_tuning_freeze_manifest": file_record(freeze_manifest_path),
        "packed_w4": packed["packed_w4"],
        "qualification_report": file_record(
            temporary / "software_qualification_report.json",
            output_root / "software_qualification_report.json",
        ),
        "schema_version": 1,
        "source": file_record(Path(__file__).resolve()),
        "weight_scale32": packed["weight_scale32"],
    }
    write_exclusive(temporary / "candidate_manifest.json", manifest)
    temporary.rename(output_root)
    print(f"ACE2_W4A8_SOFTWARE_QUALITY status={report['status']} output={output_root}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
