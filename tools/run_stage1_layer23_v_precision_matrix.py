#!/usr/bin/env python3
"""Run and verify the frozen layer-23 V-projection precision matrix."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import peft
import tokenizers
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import diagnose_stage1_layer23_qkv_families as diagnosis


MISSION_ID = "stage1vprecision01"
LAYER = 23
TARGET_TOKEN = 39814
TOP_K = 8
FORMAT_ORDER = ("w4a8", "w8a8", "w4a16", "w8a16")
FORMAT_BITS = {
    "w4a8": (4, 8, 21),
    "w8a8": (8, 8, 25),
    "w4a16": (4, 16, 29),
    "w8a16": (8, 16, 33),
}
OUTPUT = (
    ROOT
    / "evidence/candidates/stage1-layer23-v-precision-matrix-v1/"
    "nonofficial-matrix-0001"
)
FREEZE = OUTPUT / "candidate-freeze.json"
RESULT = OUTPUT / "result.json"
REVIEW_REQUEST = OUTPUT / "fresh-l2-review-request.json"
PRE_REPAIR_REVIEW_REQUEST = (
    OUTPUT / "fresh-l2-review-request.pre-provenance-repair.json"
)
EXECUTION_RUNNER = OUTPUT / "matrix-execution-runner.py"
PROVENANCE_REPAIR = OUTPUT / "output-scale-provenance-repair.json"
SUMS = OUTPUT / "SHA256SUMS"
LAUNCH_FAILURE = OUTPUT / "launch-failure-0001.json"
CONTRACT_REQUIREMENT = (
    ROOT
    / "design/model_hardware_contracts/"
    "qwen2.5-0.5b-layer23-v-precision-requirement.json"
)
ACCEPTED_DIAGNOSIS = ROOT / "diagnosis/stage1layer23qkvfamilies01/result.json"
ACCEPTED_DIAGNOSIS_SHA256 = (
    "4e6fa9c101f2d2a4c09a80fb1883665043a5d8126ebaa7d4fa862bf602aa4856"
)
ACCEPTED_DIAGNOSIS_SUMS = (
    ROOT / "diagnosis/stage1layer23qkvfamilies01/SHA256SUMS"
)
ACCEPTED_DIAGNOSIS_SUMS_SHA256 = (
    "8ed4a1e7c8e23623c6c0e2e4e622a4c0ac2900443c284786dbc825cc423901b9"
)
PRIOR_CANDIDATE_SUMS = (
    ROOT
    / "evidence/candidates/stage1-layer23-qkv-grouped-scale-v1/"
    "candidate-0001/SHA256SUMS"
)
PRIOR_CANDIDATE_SUMS_SHA256 = (
    "30e281746e1f1be7e0cbac0d68a6e69970cbf9907e1bc8cb0fa172bd4b6a2b3f"
)
BF16_LOGITS = (
    ROOT
    / "diagnosis/stage1qualitydiag01/logits/"
    "checkpoint176-bf16-greedy-step-00-f32le.bin"
)
V_PATTERN = re.compile(r"layer23_position(?P<position>\d+)_v")
EXPECTED_BASELINE = {
    "logits_sha256": "0b80dbb919618cb7f39a8a862cac652ba684176a520865020595bd048085ea17",
    "target_rank": 15,
    "top8_overlap_count": 0,
    "jensen_shannon_divergence_nats": 0.6701229933515445,
    "relative_l2": 0.9423817179633847,
    "top_token_id": 55725,
}
RETAINED_A8_OUTPUT_SCALE = 0.06586176203930472
CAPTURED_BF16_V_ORACLE_SCALE = 0.06348425196850394
EXECUTION_RUNNER_SHA256 = (
    "5e04ca938a518e77275d59d01ed1f664d820fb8173ddbac815d31a2bb7848855"
)
LOCALIZER_SHA256 = (
    "e6962c5140d6c36a77b692a219bc480e3fd8ad39e9c421f7ab236fb4665f99ab"
)
PRIOR_CAUSAL_RESULT_SHA256 = (
    "0e35e7ccafa5dc90324af4dee91e4a9c7d91075055965b6894c0adb750c1bff4"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return diagnosis.prior.localizer.file_record(path)


def write_tensor(path: Path, value: torch.Tensor, dtype: str) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(diagnosis.prior.localizer.tensor_bytes(tensor))
    record = diagnosis.prior.localizer.file_record(path)
    record.update({"dtype": dtype, "shape": list(tensor.shape)})
    return record


def verify_sum_tree(sums_path: Path, expected_hash: str) -> None:
    localizer = diagnosis.prior.localizer
    require(localizer.sha256_file(sums_path) == expected_hash, "protected SHA256SUMS changed")
    for line in sums_path.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = sums_path.parent / relative
        require(path.is_file(), f"protected artifact missing: {relative}")
        require(
            localizer.sha256_file(path) == expected,
            f"protected artifact changed: {relative}",
        )


def protected_records() -> dict[str, Any]:
    localizer = diagnosis.prior.localizer
    return {
        "accepted_diagnosis": localizer.file_record(ACCEPTED_DIAGNOSIS),
        "accepted_diagnosis_sums": localizer.file_record(ACCEPTED_DIAGNOSIS_SUMS),
        "prior_candidate_sums": localizer.file_record(PRIOR_CANDIDATE_SUMS),
        "sealed_attempt_0002": diagnosis.prior.sealed_records(),
    }


def configuration_record(freeze: dict[str, Any], mode: str) -> dict[str, Any]:
    weight_bits, input_bits, accumulator_bits = FORMAT_BITS[mode]
    value = {
        "format": mode,
        "layer": LAYER,
        "component": "v",
        "weight_bits": weight_bits,
        "input_activation_bits": input_bits,
        "projection_output_bits": 8,
        "accumulator_bits": accumulator_bits,
        "quantizers": copy.deepcopy(freeze["formats"][mode]),
        "scale_semantics": copy.deepcopy(freeze["scale_semantics"]),
    }
    return {
        "value": value,
        "sha256": diagnosis.prior.localizer.sha256_bytes(canonical_bytes(value)),
    }


class PrecisionMatrixCache(diagnosis.prior.localizer.FastProjectionCache):
    """Change only layer-23 V weight/input precision and retain its A8 output."""

    def __init__(
        self,
        weights: Any,
        adapter: Any,
        captures: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(weights, adapter)
        self.captures = captures
        self.mode: str | None = None
        self.records: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}

    def configure(self, mode: str) -> None:
        require(mode in FORMAT_ORDER, "format outside frozen matrix")
        self.mode = mode
        self.records = []
        self.state = {}

    def _input(
        self,
        position: int,
        default_q: torch.Tensor,
        default_scale: float,
    ) -> tuple[torch.Tensor, float]:
        require(self.mode is not None, "precision format is not configured")
        input_bits = FORMAT_BITS[self.mode][1]
        if input_bits == 8:
            return default_q, float(default_scale)
        if "input_scale" not in self.state:
            self.state["baseline_input_scale"] = float(default_scale)
            self.state["input_scale"] = float(default_scale) * 127.0 / 32767.0
        scale = float(self.state["input_scale"])
        source = self.captures["input_norm"][position].to(torch.float64)
        quantized = torch.round(source / scale).clamp(-32768, 32767).to(torch.int16)
        return quantized, scale

    def _weight(
        self, merged: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        require(self.mode is not None, "precision format is not configured")
        weight_bits = FORMAT_BITS[self.mode][0]
        metadata = self.metadata.get(merged.data_ptr())
        require(metadata is not None, "V projection metadata is missing")
        if weight_bits == 4:
            return metadata["qweight"], metadata["weight_scale"]
        weight = merged.to(torch.float64)
        scale = weight.abs().amax(dim=1) / 127.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight = torch.round(weight / scale[:, None]).clamp(-128, 127).to(torch.int8)
        return qweight, scale

    def _fixed_precision(
        self,
        name: str,
        input_q: torch.Tensor,
        input_scale: float,
        qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        multiplier: torch.Tensor,
        right_shift: torch.Tensor,
        output_scale: float,
    ) -> dict[str, Any]:
        require(self.mode is not None, "precision format is not configured")
        accumulator = torch.mv(qweight.to(torch.int64), input_q.to(torch.int64))
        accumulator_bits = FORMAT_BITS[self.mode][2]
        lower = -(1 << (accumulator_bits - 1))
        upper = 1 << (accumulator_bits - 1)
        require(
            bool(torch.all(accumulator >= lower))
            and bool(torch.all(accumulator < upper)),
            f"{name} exceeds declared signed {accumulator_bits}-bit accumulator",
        )
        product = accumulator * multiplier
        rounded = diagnosis.prior.localizer.round_shift_even_tensor(
            product, right_shift
        )
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        return {
            "name": name,
            "input_q": input_q,
            "input_scale": input_scale,
            "qweight": qweight,
            "weight_scale": weight_scale,
            "multiplier": multiplier,
            "right_shift": right_shift,
            "accumulator": accumulator,
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": output_scale,
            "saturation": saturation,
        }

    def _record(self, position: int, result: dict[str, Any]) -> None:
        if position == 0:
            self.state.update(
                {
                    "qweight": result["qweight"].detach().cpu().contiguous(),
                    "weight_scale": result["weight_scale"].detach().cpu().contiguous(),
                    "multiplier": result["multiplier"].detach().cpu().contiguous(),
                    "right_shift": result["right_shift"].detach().cpu().contiguous(),
                    "input_scale": float(result["input_scale"]),
                    "output_scale": float(result["output_scale"]),
                }
            )
        self.records.append(
            {
                "position": position,
                "input_q": result["input_q"].detach().cpu().contiguous(),
                "output_q": result["output_q"].detach().cpu().contiguous(),
                "accumulator": result["accumulator"].detach().cpu().contiguous(),
            }
        )

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        match = V_PATTERN.fullmatch(name)
        if match is None:
            return super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )
        require(self.mode is not None, "precision format is not configured")
        position = int(match.group("position"))
        if self.mode == "w4a8":
            result = super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )
        else:
            candidate_input, candidate_input_scale = self._input(
                position, input_q, input_scale
            )
            qweight, weight_scale = self._weight(merged)
            float_output = torch.mv(
                merged, float_input.to(torch.float32)
            ).contiguous()
            output_scale = float(
                diagnosis.prior.backend.canonical.scale_for(float_output)
            )
            multiplier, right_shift = (
                diagnosis.prior.backend.canonical.derive_multiplier(
                    candidate_input_scale * weight_scale / output_scale
                )
            )
            result = self._fixed_precision(
                name,
                candidate_input,
                candidate_input_scale,
                qweight,
                weight_scale,
                multiplier,
                right_shift,
                output_scale,
            )
            result.update(
                {
                    "float_output": float_output,
                    "source_hashes": source_hashes,
                }
            )
        self._record(position, result)
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
        match = V_PATTERN.fullmatch(name)
        if match is None:
            return super().from_fixed_metadata(
                name,
                input_q,
                input_scale,
                qweight,
                multiplier,
                right_shift,
                output_scale,
            )
        require(self.mode is not None, "precision format is not configured")
        position = int(match.group("position"))
        if self.mode == "w4a8":
            result = super().from_fixed_metadata(
                name,
                input_q,
                input_scale,
                qweight,
                multiplier,
                right_shift,
                output_scale,
            )
            result["weight_scale"] = self.state["weight_scale"]
        else:
            candidate_input, candidate_input_scale = self._input(
                position, input_q, input_scale
            )
            result = self._fixed_precision(
                name,
                candidate_input,
                candidate_input_scale,
                qweight,
                self.state["weight_scale"],
                multiplier,
                right_shift,
                output_scale,
            )
        self._record(position, result)
        return result

    def evidence(self, mode: str, oracle_v: torch.Tensor) -> dict[str, Any]:
        require(len(self.records) == 34, f"{mode} did not produce 34 V outputs")
        require(
            [record["position"] for record in self.records] == list(range(34)),
            f"{mode} V positions are not ordered",
        )
        input_q = torch.stack([record["input_q"] for record in self.records])
        output_q = torch.stack([record["output_q"] for record in self.records])
        accumulator = torch.stack(
            [record["accumulator"] for record in self.records]
        )
        output_scale = float(self.state["output_scale"])
        dequantized = output_q.to(torch.float64) * output_scale
        reference = oracle_v.to(torch.float64)
        denominator = torch.linalg.vector_norm(reference)
        relative_l2 = float(
            torch.linalg.vector_norm(dequantized - reference) / denominator
        )
        directory = OUTPUT / "formats" / mode
        input_dtype = "int8" if FORMAT_BITS[mode][1] == 8 else "int16_le"
        evidence = {
            "qweight": write_tensor(
                directory / "v_qweight_s8.bin",
                self.state["qweight"].to(torch.int8),
                "int8",
            ),
            "weight_scale": write_tensor(
                directory / "v_weight_scale_f64le.bin",
                self.state["weight_scale"].to(torch.float64),
                "float64_le",
            ),
            "input_q": write_tensor(
                directory
                / (
                    "v_input_s8.bin"
                    if FORMAT_BITS[mode][1] == 8
                    else "v_input_s16le.bin"
                ),
                input_q,
                input_dtype,
            ),
            "multiplier": write_tensor(
                directory / "v_multiplier_s32le.bin",
                self.state["multiplier"].to(torch.int32),
                "int32_le",
            ),
            "right_shift": write_tensor(
                directory / "v_right_shift_u8.bin",
                self.state["right_shift"].to(torch.uint8),
                "uint8",
            ),
            "output_q": write_tensor(
                directory / "v_output_s8.bin", output_q, "int8"
            ),
            "input_scale": float(self.state["input_scale"]),
            "output_scale": output_scale,
            "accumulator_observed_min": int(accumulator.min().item()),
            "accumulator_observed_max": int(accumulator.max().item()),
            "accumulator_sha256": diagnosis.prior.localizer.tensor_sha256(
                accumulator
            ),
            "v_projection_relative_l2_vs_captured_bf16": relative_l2,
        }
        if "baseline_input_scale" in self.state:
            evidence["baseline_a8_input_scale"] = float(
                self.state["baseline_input_scale"]
            )
        return evidence


def summary_metrics(record: dict[str, Any]) -> dict[str, Any]:
    comparison = record["comparison_vs_checkpoint176_bf16"]
    rank = int(comparison["reference_top_rank_in_candidate"])
    top_token = int(record["top_token_id"])
    return {
        "target_token_id": TARGET_TOKEN,
        "target_token_rank": rank,
        "target_token_top1": rank == 1 and top_token == TARGET_TOKEN,
        "top_token_id": top_token,
        "top8_overlap_count": int(comparison["top8_overlap_count"]),
        "jensen_shannon_divergence_nats": float(
            comparison["jensen_shannon_divergence_nats"]
        ),
        "relative_l2": float(comparison["raw_score_error"]["relative_l2"]),
        "comparison_vs_checkpoint176_bf16": comparison,
    }


def decision_checks(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    rank_limit = max(TOP_K, int(baseline["target_token_rank"]) // 10)
    exact_recovery = (
        candidate["target_token_rank"] == 1
        and candidate["target_token_top1"]
        and candidate["top8_overlap_count"] >= baseline["top8_overlap_count"]
        and candidate["jensen_shannon_divergence_nats"]
        <= baseline["jensen_shannon_divergence_nats"]
    )
    distribution_recovery = (
        candidate["target_token_rank"] <= rank_limit
        and candidate["top8_overlap_count"] > baseline["top8_overlap_count"]
        and candidate["jensen_shannon_divergence_nats"]
        <= baseline["jensen_shannon_divergence_nats"] - 0.05
    )
    return {
        "exact_rank1_without_top8_or_jsd_regression": exact_recovery,
        "material_rank_top8_jsd_recovery": distribution_recovery,
        "materially_improves": exact_recovery or distribution_recovery,
        "rank_limit": rank_limit,
    }


def run_format(
    cache: PrecisionMatrixCache,
    mode: str,
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
    bf16_logits: torch.Tensor,
    oracle_v: torch.Tensor,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    cache.configure(mode)
    label = f"layer23-v-{mode}"
    record, _ = diagnosis.prior.run_cut_record(
        LAYER,
        label,
        hidden_states,
        weights,
        adapter,
        norm_gain,
        embedding,
        head,
        tokenizer,
        bf16_logits,
    )
    metrics = summary_metrics(record)
    evidence = cache.evidence(mode, oracle_v)
    config = configuration_record(freeze, mode)
    print(
        "ACE2_LAYER23_V_PRECISION "
        f"format={mode} top={metrics['top_token_id']} "
        f"rank={metrics['target_token_rank']} "
        f"overlap={metrics['top8_overlap_count']} "
        f"jsd={metrics['jensen_shannon_divergence_nats']:.12f} "
        f"rel_l2={metrics['relative_l2']:.12f}",
        flush=True,
    )
    return {
        "format": mode,
        "configuration": config,
        "metrics": metrics,
        "logits": record["accepted_w4a8_logits"],
        "v_projection_evidence": evidence,
        "cost": {
            "weight_bytes": freeze["cost_model"]["weight_bytes"][mode],
            "activation_bytes_per_v_input": freeze["cost_model"][
                "activation_bytes_per_v_input"
            ][mode],
            "projection_output_bytes": freeze["cost_model"][
                "projection_output_bytes"
            ],
            "macs_per_v_projection": freeze["cost_model"][
                "macs_per_v_projection"
            ],
            "operand_bit_product": freeze["cost_model"]["operand_bit_product"][
                mode
            ],
            "accumulator_bits": freeze["cost_model"]["accumulator_bits"][mode],
        },
    }


def validate_freeze(freeze: dict[str, Any]) -> None:
    require(freeze["mission_id"] == MISSION_ID, "freeze mission changed")
    require(
        tuple(freeze["candidate_order"]) == FORMAT_ORDER,
        "frozen format order changed",
    )
    require(
        freeze["matrix_execution_count"] == 1,
        "matrix execution count is not one",
    )
    require(
        freeze["evaluation"]["target_token_id"] == TARGET_TOKEN
        and freeze["evaluation"]["top_k"] == TOP_K
        and freeze["evaluation"]["generation_position"] == 0,
        "frozen evaluator changed",
    )
    require(
        freeze["accepted_diagnosis"]["result_sha256"]
        == ACCEPTED_DIAGNOSIS_SHA256,
        "accepted diagnosis binding changed",
    )
    require(
        freeze["oracle"]["bf16_logits_sha256"]
        == diagnosis.prior.BF16_LOGITS_SHA256,
        "BF16 logits oracle binding changed",
    )
    require(
        freeze["oracle"]["captured_v_sha256"]
        == "44122f1d7938b82fca1203557cd0ffa85f5d6516f36dd8e9d3e5c427cce37f2e",
        "BF16 V oracle binding changed",
    )


def generate() -> None:
    require(FREEZE.is_file(), "candidate freeze is missing")
    require(not RESULT.exists(), "precision matrix result already exists")
    require(not SUMS.exists(), "precision matrix was already finalized")
    freeze = read_json(FREEZE)
    validate_freeze(freeze)
    localizer = diagnosis.prior.localizer
    require(
        localizer.sha256_file(ACCEPTED_DIAGNOSIS)
        == ACCEPTED_DIAGNOSIS_SHA256,
        "accepted diagnosis changed",
    )
    verify_sum_tree(ACCEPTED_DIAGNOSIS_SUMS, ACCEPTED_DIAGNOSIS_SUMS_SHA256)
    verify_sum_tree(PRIOR_CANDIDATE_SUMS, PRIOR_CANDIDATE_SUMS_SHA256)
    require(
        localizer.sha256_file(BF16_LOGITS)
        == diagnosis.prior.BF16_LOGITS_SHA256,
        "BF16 logits oracle changed",
    )
    protected_before = protected_records()
    started = time.monotonic()
    snapshot = diagnosis.prior.generation_runner.resolve_snapshot()
    diagnosis.prior.backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    token_ids = diagnosis.prior.generation_runner.canonical_chat_token_ids(
        tokenizer, diagnosis.prior.PROMPT
    )
    require(len(token_ids) == 34, "frozen prompt token count changed")
    require(
        localizer.sha256_bytes(canonical_bytes(token_ids))
        == diagnosis.prior.TOKEN_IDS_SHA256,
        "frozen token IDs changed",
    )
    require(
        localizer.sha256_bytes(diagnosis.prior.PROMPT.encode("utf-8"))
        == diagnosis.prior.PROMPT_SHA256,
        "frozen prompt changed",
    )
    diagnosis.prior.localizer.PROMPT = diagnosis.prior.PROMPT
    diagnosis.prior.localizer.TOKEN_IDS = token_ids
    diagnosis.prior.localizer.REFERENCE_TOKEN = TARGET_TOKEN

    print("ACE2_LAYER23_V_PRECISION_BF16_CAPTURE", flush=True)
    layer_input, captures, bf16_logits = diagnosis.capture_layer_context(
        snapshot, token_ids
    )
    require(
        localizer.tensor_sha256(captures["v"])
        == freeze["oracle"]["captured_v_sha256"],
        "captured BF16 V projection changed",
    )
    oracle_record = write_tensor(
        OUTPUT / "oracle/layer23-v-bf16-projection-f32le.bin",
        captures["v"].to(torch.float32),
        "float32_le",
    )
    hidden_states = [torch.empty(0)] * LAYER + [layer_input]
    format_records: list[dict[str, Any]] = []
    suppressed_messages: list[str] = []
    prior_output = diagnosis.prior.OUTPUT
    diagnosis.prior.OUTPUT = OUTPUT
    try:
        with safe_open(
            diagnosis.prior.backend.canonical.MODEL,
            framework="pt",
            device="cpu",
        ) as weights, safe_open(
            diagnosis.prior.backend.canonical.ADAPTER,
            framework="pt",
            device="cpu",
        ) as adapter:
            embedding = weights.get_tensor(
                "model.embed_tokens.weight"
            ).contiguous()
            norm_gain = weights.get_tensor("model.norm.weight").contiguous()
            head = localizer.derive_lm_head_weights(embedding)
            cache = PrecisionMatrixCache(weights, adapter, captures)
            guard = localizer.SubstitutionRequireGuard()
            cache.install()
            guard.install()
            try:
                for mode in FORMAT_ORDER:
                    format_records.append(
                        run_format(
                            cache,
                            mode,
                            hidden_states,
                            weights,
                            adapter,
                            norm_gain,
                            embedding,
                            head,
                            tokenizer,
                            bf16_logits,
                            captures["v"],
                            freeze,
                        )
                    )
            finally:
                suppressed_messages.extend(guard.suppressed)
                guard.restore()
                cache.restore()
    finally:
        diagnosis.prior.OUTPUT = prior_output

    baseline = format_records[0]["metrics"]
    require(
        format_records[0]["logits"]["sha256"]
        == EXPECTED_BASELINE["logits_sha256"]
        and baseline["target_token_rank"] == EXPECTED_BASELINE["target_rank"]
        and baseline["top8_overlap_count"]
        == EXPECTED_BASELINE["top8_overlap_count"]
        and math.isclose(
            baseline["jensen_shannon_divergence_nats"],
            EXPECTED_BASELINE["jensen_shannon_divergence_nats"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            baseline["relative_l2"],
            EXPECTED_BASELINE["relative_l2"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and baseline["top_token_id"] == EXPECTED_BASELINE["top_token_id"],
        "unchanged W4A8 baseline did not reproduce the accepted cut-23 record",
    )
    baseline_output_scale = format_records[0]["v_projection_evidence"][
        "output_scale"
    ]
    for record in format_records[1:]:
        require(
            record["v_projection_evidence"]["output_scale"]
            == baseline_output_scale,
            f"{record['format']} changed the retained A8 output scale",
        )
        record["decision"] = decision_checks(baseline, record["metrics"])
    passing = [
        record
        for record in format_records[1:]
        if record["decision"]["materially_improves"]
    ]
    selected = passing[0] if passing else None
    protected_after = protected_records()
    require(
        protected_after == protected_before,
        "protected diagnosis, candidate, or attempt-0002 evidence changed",
    )
    verify_sum_tree(ACCEPTED_DIAGNOSIS_SUMS, ACCEPTED_DIAGNOSIS_SUMS_SHA256)
    verify_sum_tree(PRIOR_CANDIDATE_SUMS, PRIOR_CANDIDATE_SUMS_SHA256)
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": (
            "MINIMUM_PRECISION_CANDIDATE_SELECTED_PENDING_FRESH_L2"
            if selected is not None
            else "PRECISION_MATRIX_NEGATIVE_RESULT_PENDING_FRESH_L2"
        ),
        "classification": "bounded_nonofficial_software_only_precision_matrix",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution": {
            "matrix_execution_count": 1,
            "format_order": list(FORMAT_ORDER),
            "prompt_token_count": len(token_ids),
            "generation_position": 0,
            "layer_cut": LAYER,
            "elapsed_wall_seconds": time.monotonic() - started,
        },
        "bindings": {
            "freeze": localizer.file_record(FREEZE),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "no_execution_launch_failure": localizer.file_record(LAUNCH_FAILURE),
            "accepted_diagnosis": localizer.file_record(ACCEPTED_DIAGNOSIS),
            "accepted_diagnosis_fresh_l2_review_sha256": freeze[
                "accepted_diagnosis"
            ]["fresh_l2_review_sha256"],
            "bf16_logits": localizer.file_record(BF16_LOGITS),
            "captured_bf16_v_projection": oracle_record,
            "protected_before": protected_before,
            "protected_after": protected_after,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
        },
        "baseline": format_records[0],
        "candidates": format_records[1:],
        "selection": {
            "policy": freeze["evaluation"]["material_improvement"],
            "precision_order": list(FORMAT_ORDER[1:]),
            "selected_format": selected["format"] if selected else None,
            "selected_configuration_sha256": (
                selected["configuration"]["sha256"] if selected else None
            ),
            "negative_result": selected is None,
        },
        "cost_scope": freeze["cost_model"]["scope"],
        "suppressed_nonfunctional_cache_effect_assertions": {
            "count": len(suppressed_messages),
            "unique_messages": sorted(set(suppressed_messages)),
        },
        "independent_review_required": True,
        "scope_guards": copy.deepcopy(freeze["scope_guards"]),
    }
    write_json(RESULT, result)
    print(
        "ACE2_LAYER23_V_PRECISION_RESULT "
        f"selected={result['selection']['selected_format']} "
        f"negative={result['selection']['negative_result']} "
        f"output={localizer.public_path(OUTPUT)}",
        flush=True,
    )


def all_format_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [result["baseline"], *result["candidates"]]


def finalize() -> None:
    require(RESULT.is_file(), "precision matrix result is missing")
    require(CONTRACT_REQUIREMENT.is_file(), "model hardware requirement is missing")
    require(EXECUTION_RUNNER.is_file(), "matrix execution runner archive is missing")
    require(PROVENANCE_REPAIR.is_file(), "output-scale provenance repair is missing")
    result = read_json(RESULT)
    requirement = read_json(CONTRACT_REQUIREMENT)
    localizer = diagnosis.prior.localizer
    require(
        requirement["evidence"]["matrix_result_sha256"]
        == localizer.sha256_file(RESULT),
        "model hardware requirement is not bound to the matrix result",
    )
    require(
        requirement["required_v_projection_format"]
        == result["selection"]["selected_format"],
        "model hardware requirement selection differs",
    )
    request = {
        "schema_version": 2,
        "mission_id": MISSION_ID,
        "review_type": "Fresh-L2",
        "status": "PENDING_INDEPENDENT_REVIEW",
        "result": localizer.file_record(RESULT),
        "candidate_freeze": localizer.file_record(FREEZE),
        "matrix_execution_runner": localizer.file_record(EXECUTION_RUNNER),
        "output_scale_provenance_repair": localizer.file_record(
            PROVENANCE_REPAIR
        ),
        "metadata_repair_verifier": localizer.file_record(Path(__file__).resolve()),
        "model_hardware_contract_requirement": localizer.file_record(
            CONTRACT_REQUIREMENT
        ),
        "requested_checks": [
            "Verify the accepted diagnosis and immutable protected-tree bindings.",
            "Recompute every format metric from retained logits against the frozen BF16 logits.",
            "Confirm the original matrix runner is preserved byte-for-byte and no candidate was rerun.",
            "Confirm 0.06586176203930472 is the retained canonical W4A8 merged-FP32 output scale held across all formats and positions.",
            "Independently derive 0.06348425196850394 from the captured BF16 V oracle and confirm that oracle did not source the retained A8 scale.",
            "Recompute each captured-BF16 V-projection relative L2 using the retained A8 outputs and actual retained scale.",
            "Confirm exact W4/W8, A8/A16, accumulator, A8-output, rounding, saturation, and layout semantics.",
            "Confirm analytical storage and operand-width deltas and that they make no PPA claim.",
            "Confirm the frozen material-improvement rule selects the first passing format or retains the negative result.",
            "Confirm the Model Hardware Contract records a pending requirement, not current RTL support.",
            "Confirm no RTL, official, sealed, attempt-0003, retry, PPA, Stage-2, proprietary-tool, or FPGA artifact changed.",
        ],
    }
    write_json(REVIEW_REQUEST, request)
    localizer.write_sums(OUTPUT)
    verify()


def verify() -> None:
    require(RESULT.is_file(), "precision matrix result is missing")
    require(REVIEW_REQUEST.is_file(), "Fresh-L2 review request is missing")
    require(SUMS.is_file(), "precision matrix SHA256SUMS is missing")
    require(CONTRACT_REQUIREMENT.is_file(), "model hardware requirement is missing")
    require(EXECUTION_RUNNER.is_file(), "matrix execution runner archive is missing")
    require(PROVENANCE_REPAIR.is_file(), "output-scale provenance repair is missing")
    localizer = diagnosis.prior.localizer
    for line in SUMS.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = OUTPUT / relative
        require(path.is_file(), f"summed artifact is missing: {relative}")
        require(
            localizer.sha256_file(path) == expected,
            f"summed artifact changed: {relative}",
        )
    freeze = read_json(FREEZE)
    validate_freeze(freeze)
    result = read_json(RESULT)
    require(result["mission_id"] == MISSION_ID, "result mission changed")
    require(
        result["execution"]["matrix_execution_count"] == 1
        and tuple(result["execution"]["format_order"]) == FORMAT_ORDER,
        "result matrix contract changed",
    )
    require(
        result["bindings"]["freeze"] == localizer.file_record(FREEZE),
        "result freeze binding changed",
    )
    recorded_runner = result["bindings"]["runner"]
    execution_runner = localizer.file_record(EXECUTION_RUNNER)
    require(
        recorded_runner["path"] == "tools/run_stage1_layer23_v_precision_matrix.py"
        and recorded_runner["bytes"] == execution_runner["bytes"] == 39957
        and recorded_runner["sha256"]
        == execution_runner["sha256"]
        == EXECUTION_RUNNER_SHA256,
        "executed matrix runner is not preserved byte-for-byte",
    )
    require(
        result["bindings"]["no_execution_launch_failure"]
        == localizer.file_record(LAUNCH_FAILURE),
        "no-execution launch failure binding changed",
    )
    require(
        result["bindings"]["accepted_diagnosis"]
        == localizer.file_record(ACCEPTED_DIAGNOSIS),
        "accepted diagnosis result changed",
    )
    verify_sum_tree(ACCEPTED_DIAGNOSIS_SUMS, ACCEPTED_DIAGNOSIS_SUMS_SHA256)
    verify_sum_tree(PRIOR_CANDIDATE_SUMS, PRIOR_CANDIDATE_SUMS_SHA256)
    require(
        protected_records()
        == result["bindings"]["protected_before"]
        == result["bindings"]["protected_after"],
        "protected evidence binding changed",
    )
    reference = np.fromfile(BF16_LOGITS, dtype="<f4")
    require(
        reference.shape == (diagnosis.prior.backend.MODEL_OUTPUT_DOMAIN,),
        "BF16 logits shape changed",
    )
    oracle_record = result["bindings"]["captured_bf16_v_projection"]
    oracle_path = ROOT / oracle_record["path"]
    require(
        localizer.sha256_file(oracle_path) == oracle_record["sha256"],
        "captured BF16 V oracle changed",
    )
    oracle = torch.from_numpy(
        np.fromfile(oracle_path, dtype="<f4").reshape(34, 128).copy()
    ).to(torch.float64)
    captured_bf16_v_oracle_scale = float(oracle[0].abs().max().item() / 127.0)
    require(
        captured_bf16_v_oracle_scale == CAPTURED_BF16_V_ORACLE_SCALE
        and captured_bf16_v_oracle_scale != RETAINED_A8_OUTPUT_SCALE,
        "captured BF16 V oracle scale is not distinct from the retained A8 scale",
    )
    repair = read_json(PROVENANCE_REPAIR)
    proof = repair["immutable_proof_chain"]
    require(
        repair["mission_id"] == MISSION_ID
        and repair["classification"]
        == "metadata_evidence_only_provenance_repair"
        and repair["repair_scope"]["candidate_executions_added"] == 0
        and repair["repair_scope"]["retained_logits_or_tensors_changed"] is False
        and repair["matrix_disposition"]["matrix_execution_count"] == 1
        and repair["matrix_disposition"]["misconfigured"] is False
        and repair["matrix_disposition"]["negative_result_retained"] is True,
        "output-scale repair scope or matrix disposition changed",
    )
    require(
        proof["matrix_result"] == localizer.file_record(RESULT)
        and proof["candidate_freeze"] == localizer.file_record(FREEZE)
        and proof["execution_runner"] == execution_runner
        and proof["accepted_diagnosis"]
        == localizer.file_record(ACCEPTED_DIAGNOSIS)
        and proof["accepted_diagnosis_runner"]
        == localizer.file_record(
            ROOT / "tools/diagnose_stage1_layer23_qkv_families.py"
        )
        and proof["prior_causal_diagnosis"]
        == localizer.file_record(
            ROOT / "diagnosis/stage1samepromptcausal01/result.json"
        )
        and proof["hidden_state_localizer"]
        == localizer.file_record(
            ROOT / "tools/ace2_checkpoint176_hidden_state_localizer.py"
        )
        and proof["captured_bf16_v_projection"] == oracle_record
        and proof["prior_review_request"]
        == localizer.file_record(PRE_REPAIR_REVIEW_REQUEST),
        "immutable output-scale proof chain changed",
    )
    require(
        proof["hidden_state_localizer"]["sha256"] == LOCALIZER_SHA256
        and proof["prior_causal_diagnosis"]["sha256"]
        == PRIOR_CAUSAL_RESULT_SHA256,
        "bound canonical W4A8 derivation source changed",
    )
    execution_source = EXECUTION_RUNNER.read_text(encoding="utf-8")
    localizer_source = (
        ROOT / "tools/ace2_checkpoint176_hidden_state_localizer.py"
    ).read_text(encoding="utf-8")
    require(
        'if self.mode == "w4a8":' in execution_source
        and "result = super().derive_projection(" in execution_source
        and "merged, float_input.to(torch.float32)" in execution_source
        and "record[\"v_projection_evidence\"][\"output_scale\"]"
        in execution_source
        and "== baseline_output_scale" in execution_source,
        "matrix runner no longer proves the retained W4A8 scale path",
    )
    require(
        "base.to(torch.float32)" in localizer_source
        and "lora_b.to(torch.float32), lora_a.to(torch.float32)"
        in localizer_source
        and "torch.mv(merged, float_input.to(torch.float32))" in localizer_source
        and "output_scale = backend.canonical.scale_for(float_output)"
        in localizer_source,
        "localizer no longer proves the merged-FP32 output-scale derivation",
    )
    require(
        repair["retained_a8_output_scale"]["value"]
        == RETAINED_A8_OUTPUT_SCALE
        and repair["captured_bf16_v_oracle_scale"]["value"]
        == captured_bf16_v_oracle_scale
        and repair["captured_bf16_v_oracle_scale"]["role"]
        == (
            "frozen comparison oracle only; not the source of the retained A8 "
            "requantization scale"
        ),
        "repaired output-scale semantics changed",
    )
    records = all_format_records(result)
    require(
        [record["format"] for record in records] == list(FORMAT_ORDER),
        "format record order changed",
    )
    for record in records:
        mode = record["format"]
        require(
            record["configuration"] == configuration_record(freeze, mode),
            f"{mode} quantizer configuration changed",
        )
        logits_record = record["logits"]
        logits_path = ROOT / logits_record["path"]
        require(
            localizer.sha256_file(logits_path) == logits_record["sha256"],
            f"{mode} logits changed",
        )
        scores = np.fromfile(logits_path, dtype="<f8")
        recomputed = diagnosis.prior.quality.score_comparison(reference, scores)
        require(
            recomputed
            == record["metrics"]["comparison_vs_checkpoint176_bf16"],
            f"{mode} logits metrics changed",
        )
        expected_metrics = {
            **record["metrics"],
            "target_token_rank": int(
                recomputed["reference_top_rank_in_candidate"]
            ),
            "target_token_top1": (
                int(recomputed["reference_top_rank_in_candidate"]) == 1
                and int(recomputed["candidate_top_token_id"]) == TARGET_TOKEN
            ),
            "top_token_id": int(recomputed["candidate_top_token_id"]),
            "top8_overlap_count": int(recomputed["top8_overlap_count"]),
            "jensen_shannon_divergence_nats": float(
                recomputed["jensen_shannon_divergence_nats"]
            ),
            "relative_l2": float(recomputed["raw_score_error"]["relative_l2"]),
        }
        require(
            expected_metrics == record["metrics"],
            f"{mode} metric summary changed",
        )
        projection = record["v_projection_evidence"]
        require(
            projection["output_scale"] == RETAINED_A8_OUTPUT_SCALE,
            f"{mode} retained A8 output scale changed",
        )
        output_path = ROOT / projection["output_q"]["path"]
        require(
            localizer.sha256_file(output_path)
            == projection["output_q"]["sha256"],
            f"{mode} V output changed",
        )
        output = torch.from_numpy(
            np.fromfile(output_path, dtype=np.int8).reshape(34, 128).copy()
        ).to(torch.float64)
        relative_l2 = float(
            torch.linalg.vector_norm(
                output * float(projection["output_scale"]) - oracle
            )
            / torch.linalg.vector_norm(oracle)
        )
        require(
            math.isclose(
                relative_l2,
                projection["v_projection_relative_l2_vs_captured_bf16"],
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            f"{mode} V projection relative L2 changed",
        )
    baseline = result["baseline"]["metrics"]
    for record in result["candidates"]:
        require(
            record["decision"] == decision_checks(baseline, record["metrics"]),
            f"{record['format']} selection checks changed",
        )
    passing = [
        record
        for record in result["candidates"]
        if record["decision"]["materially_improves"]
    ]
    selected = passing[0] if passing else None
    require(
        result["selection"]["selected_format"]
        == (selected["format"] if selected else None)
        and result["selection"]["negative_result"] == (selected is None),
        "minimum-precision selection changed",
    )
    requirement = read_json(CONTRACT_REQUIREMENT)
    require(
        requirement["evidence"]["matrix_result_sha256"]
        == localizer.sha256_file(RESULT)
        and requirement["evidence"]["output_scale_provenance_repair"]
        == localizer.file_record(PROVENANCE_REPAIR)
        and requirement["evidence"]["matrix_execution_runner"]
        == execution_runner
        and requirement["required_v_projection_format"]
        == result["selection"]["selected_format"]
        and requirement["current_rtl_support"] is False,
        "model hardware requirement binding changed",
    )
    retained_output = requirement["retained_projection_output"]
    require(
        retained_output["output_scale"] == RETAINED_A8_OUTPUT_SCALE
        and retained_output["captured_bf16_v_oracle_scale"]
        == captured_bf16_v_oracle_scale
        and retained_output["output_scale_provenance"]
        == "unchanged canonical W4A8 merged-FP32 V-projection at position 0",
        "model hardware requirement repeats the contradicted scale semantics",
    )
    request = read_json(REVIEW_REQUEST)
    require(
        request["schema_version"] == 2
        and request["status"] == "PENDING_INDEPENDENT_REVIEW"
        and request["result"] == localizer.file_record(RESULT)
        and request["candidate_freeze"] == localizer.file_record(FREEZE)
        and request["matrix_execution_runner"] == execution_runner
        and request["output_scale_provenance_repair"]
        == localizer.file_record(PROVENANCE_REPAIR)
        and request["metadata_repair_verifier"]
        == localizer.file_record(Path(__file__).resolve())
        and request["model_hardware_contract_requirement"]
        == localizer.file_record(CONTRACT_REQUIREMENT),
        "Fresh-L2 review request binding changed",
    )
    require(
        not (
            ROOT
            / "evidence/verification/rtl-arbitrary-text-generation-v1/"
            "attempt-0003"
        ).exists(),
        "attempt-0003 exists",
    )
    print(
        "PASS stage1vprecision01 "
        f"selected={result['selection']['selected_format']} "
        f"negative={result['selection']['negative_result']} "
        f"formats={len(records)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    elif args.finalize:
        finalize()
    else:
        generate()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ACE2_LAYER23_V_PRECISION_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
