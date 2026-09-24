#!/usr/bin/env python3
"""Diagnose low-rank structure in the frozen layer-23 V projection residual."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
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

from tools import run_stage1_layer23_v_joint_scale_search as joint


diagnosis = joint.diagnosis
precision = joint.precision

MISSION_ID = "stage1vresidualstructure01"
LAYER = 23
TARGET_TOKEN = 39814
TOP_K = 8
RANKS = (1, 2, 4, 8)
CALIBRATION_POSITIONS = tuple(range(17))
HELD_OUT_POSITIONS = tuple(range(17, 34))
PINV_RTOL = 1.0e-6
LOW_RANK_ENERGY_THRESHOLD = 0.90
OUTPUT = (
    ROOT
    / "evidence/diagnostics/stage1-layer23-v-residual-structure-v1/"
    "nonofficial-diagnosis-0001"
)
FREEZE = OUTPUT / "diagnosis-freeze.json"
RESULT = OUTPUT / "result.json"
REVIEW_REQUEST = OUTPUT / "fresh-l2-review-request.json"
SUMS = OUTPUT / "SHA256SUMS"
EXECUTION_RUNNER = OUTPUT / "residual-structure-execution-runner.py"
AUTHORITY_CONSUMPTION = OUTPUT / "authorization-consumed.json"
ACCEPTED_DIAGNOSIS = ROOT / "diagnosis/stage1layer23qkvfamilies01/result.json"
ACCEPTED_DIAGNOSIS_SUMS = (
    ROOT / "diagnosis/stage1layer23qkvfamilies01/SHA256SUMS"
)
ACCEPTED_JOINT_RESULT = (
    ROOT
    / "evidence/candidates/stage1-layer23-v-joint-scale-search-v1/"
    "nonofficial-search-0001/result.json"
)
ACCEPTED_JOINT_SUMS = ACCEPTED_JOINT_RESULT.parent / "SHA256SUMS"
BF16_LOGITS = (
    ROOT
    / "diagnosis/stage1qualitydiag01/logits/"
    "checkpoint176-bf16-greedy-step-00-f32le.bin"
)
EXPECTED_BASELINE = {
    "logits_sha256": "0b80dbb919618cb7f39a8a862cac652ba684176a520865020595bd048085ea17",
    "target_rank": 15,
    "top8_overlap_count": 0,
    "jensen_shannon_divergence_nats": 0.6701229933515445,
    "relative_l2": 0.9423817179633847,
    "top_token_id": 55725,
}
SCOPE_GUARDS = {
    "attempt_0003": False,
    "candidate_or_rtl_promotion": False,
    "focused_or_full_rtl": False,
    "official_preflight_or_run": False,
    "ppa_or_stage2": False,
    "replay_attempt_0002": False,
    "specification_audit_refresh": False,
}


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


def consume_authority(value: Any) -> dict[str, Any]:
    AUTHORITY_CONSUMPTION.parent.mkdir(parents=True, exist_ok=True)
    with AUTHORITY_CONSUMPTION.open("xb") as stream:
        stream.write(canonical_bytes(value))
    return diagnosis.prior.localizer.file_record(AUTHORITY_CONSUMPTION)


def write_tensor(
    path: Path, value: torch.Tensor, dtype: str
) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(diagnosis.prior.localizer.tensor_bytes(tensor))
    record = diagnosis.prior.localizer.file_record(path)
    record.update({"dtype": dtype, "shape": list(tensor.shape)})
    return record


def load_tensor(record: dict[str, Any]) -> torch.Tensor:
    dtype_map = {
        "float64_le": np.dtype("<f8"),
        "float32_le": np.dtype("<f4"),
        "int8": np.dtype("i1"),
    }
    dtype = dtype_map[record["dtype"]]
    values = np.fromfile(ROOT / record["path"], dtype=dtype).copy()
    return torch.from_numpy(values).reshape(record["shape"])


def verify_sum_tree(path: Path, expected_hash: str) -> None:
    localizer = diagnosis.prior.localizer
    require(
        localizer.sha256_file(path) == expected_hash,
        f"{path.name} hash changed",
    )
    for line in path.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        artifact = path.parent / relative
        require(artifact.is_file(), f"protected artifact missing: {relative}")
        require(
            localizer.sha256_file(artifact) == expected,
            f"protected artifact changed: {relative}",
        )


def protected_records() -> dict[str, Any]:
    localizer = diagnosis.prior.localizer
    return {
        "accepted_qkv_diagnosis": localizer.file_record(ACCEPTED_DIAGNOSIS),
        "accepted_qkv_diagnosis_sums": localizer.file_record(
            ACCEPTED_DIAGNOSIS_SUMS
        ),
        "accepted_joint_scale_negative": localizer.file_record(
            ACCEPTED_JOINT_RESULT
        ),
        "accepted_joint_scale_negative_sums": localizer.file_record(
            ACCEPTED_JOINT_SUMS
        ),
        "bf16_logits": localizer.file_record(BF16_LOGITS),
    }


def validate_freeze(freeze: dict[str, Any]) -> None:
    require(freeze["mission_id"] == MISSION_ID, "freeze mission changed")
    require(
        freeze["classification"]
        == "bounded_nonofficial_software_only_layer23_v_residual_diagnosis",
        "freeze classification changed",
    )
    require(freeze["ranks"] == list(RANKS), "frozen ranks changed")
    require(
        freeze["partition"]["calibration_positions"]
        == list(CALIBRATION_POSITIONS)
        and freeze["partition"]["held_out_positions"]
        == list(HELD_OUT_POSITIONS),
        "calibration/held-out partition changed",
    )
    require(
        freeze["fit_policy"]
        == {
            "basis": (
                "right singular vectors of the calibration-only "
                "BF16-minus-W4A8 V output residual"
            ),
            "coefficient_predictor": (
                "minimum-norm linear map from existing dequantized A8 "
                "V-projection inputs to calibration residual coefficients"
            ),
            "intercept": False,
            "pinv_rtol": PINV_RTOL,
            "retuning": False,
        },
        "frozen fit policy changed",
    )
    require(
        freeze["decision_policy"]
        == {
            "low_rank_calibration_energy_threshold": LOW_RANK_ENERGY_THRESHOLD,
            "logit_gate": (
                "rank 1 with no top-8/JSD regression, or rank <= 8 with "
                "strict top-8 improvement and JSD improvement >= 0.05 nats"
            ),
            "held_out_gate": (
                "corrected A8 V projection relative-L2 no worse than baseline"
            ),
            "promotion": False,
        },
        "frozen decision policy changed",
    )
    require(
        freeze["scope_guards"] == SCOPE_GUARDS,
        "frozen scope guards changed",
    )


class ResidualCorrectionCache(diagnosis.prior.localizer.FastProjectionCache):
    """Add one frozen predicted residual before the retained V A8 output."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        self.correction: torch.Tensor | None = None
        self.records: list[dict[str, Any]] = []

    def configure(self, correction: torch.Tensor | None) -> None:
        if correction is not None:
            require(
                correction.shape == (34, 128),
                "residual correction shape changed",
            )
        self.correction = correction
        self.records = []

    def _apply(
        self, name: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        match = joint.V_PATTERN.fullmatch(name)
        if match is None:
            return result
        position = int(match.group("position"))
        output_scale = float(result["output_scale"])
        baseline = result["output_q"].to(torch.float64) * output_scale
        correction = (
            torch.zeros_like(baseline)
            if self.correction is None
            else self.correction[position].to(torch.float64)
        )
        source = baseline + correction
        if self.correction is not None:
            result["output_q"] = diagnosis.prior.backend.canonical.quantize_int8(
                source.to(torch.float32), output_scale
            )
        corrected = result["output_q"].to(torch.float64) * output_scale
        self.records.append(
            {
                "position": position,
                "input_dequantized": (
                    result["input_q"].to(torch.float64)
                    * float(result["input_scale"])
                )
                .cpu()
                .contiguous(),
                "baseline_output": baseline.cpu().contiguous(),
                "predicted_correction": correction.cpu().contiguous(),
                "corrected_output": corrected.cpu().contiguous(),
                "output_q": result["output_q"].cpu().contiguous(),
                "output_scale": output_scale,
            }
        )
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
        result = super().derive_projection(
            name,
            merged,
            input_q,
            input_scale,
            float_input,
            source_hashes,
        )
        return self._apply(name, result)

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
        result = super().from_fixed_metadata(
            name,
            input_q,
            input_scale,
            qweight,
            multiplier,
            right_shift,
            output_scale,
        )
        return self._apply(name, result)

    def stacked(self) -> dict[str, torch.Tensor | float]:
        require(len(self.records) == 34, "V path did not produce 34 outputs")
        require(
            [int(item["position"]) for item in self.records] == list(range(34)),
            "V positions are not ordered",
        )
        output_scales = {
            float(item["output_scale"]) for item in self.records
        }
        require(len(output_scales) == 1, "V A8 output scale changed by position")
        return {
            key: torch.stack([item[key] for item in self.records])
            for key in (
                "input_dequantized",
                "baseline_output",
                "predicted_correction",
                "corrected_output",
                "output_q",
            )
        } | {"output_scale": output_scales.pop()}


def run_downstream(
    cache: ResidualCorrectionCache,
    label: str,
    correction: torch.Tensor | None,
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
    bf16_logits: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | float]]:
    cache.configure(correction)
    record, _ = diagnosis.prior.run_cut_record(
        LAYER,
        f"layer23-v-residual-{label}",
        hidden_states,
        weights,
        adapter,
        norm_gain,
        embedding,
        head,
        tokenizer,
        bf16_logits,
    )
    return record, cache.stacked()


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    difference = candidate.to(torch.float64) - reference.to(torch.float64)
    denominator = torch.linalg.vector_norm(reference.to(torch.float64))
    require(float(denominator.item()) > 0.0, "relative-L2 denominator is zero")
    return float(torch.linalg.vector_norm(difference).item() / denominator.item())


def error_summary(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, Any]:
    difference = candidate.to(torch.float64) - reference.to(torch.float64)
    return {
        "count": difference.numel(),
        "relative_l2": relative_l2(candidate, reference),
        "rmse": float(torch.sqrt(torch.mean(difference * difference)).item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


def split_projection_summary(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, Any]:
    calibration = torch.tensor(CALIBRATION_POSITIONS, dtype=torch.int64)
    held_out = torch.tensor(HELD_OUT_POSITIONS, dtype=torch.int64)
    return {
        "calibration": error_summary(
            candidate.index_select(0, calibration),
            reference.index_select(0, calibration),
        ),
        "held_out": error_summary(
            candidate.index_select(0, held_out),
            reference.index_select(0, held_out),
        ),
        "full_34_positions": error_summary(candidate, reference),
    }


def energy_fraction(residual: torch.Tensor, approximation: torch.Tensor) -> float:
    energy = float(torch.sum(residual.to(torch.float64) ** 2).item())
    require(energy > 0.0, "residual energy is zero")
    error = float(
        torch.sum(
            (residual.to(torch.float64) - approximation.to(torch.float64)) ** 2
        ).item()
    )
    return 1.0 - error / energy


def concentration(energy: torch.Tensor, count: int) -> float:
    total = float(energy.sum().item())
    require(total > 0.0, "channel energy is zero")
    return float(torch.topk(energy, min(count, energy.numel())).values.sum().item() / total)


def residual_structure(
    residual: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    calibration = residual[list(CALIBRATION_POSITIONS)].to(torch.float64)
    held_out = residual[list(HELD_OUT_POSITIONS)].to(torch.float64)
    _, singular_values, right_vectors = torch.linalg.svd(
        calibration, full_matrices=False
    )
    singular_energy = singular_values * singular_values
    total_energy = float(singular_energy.sum().item())
    cumulative = torch.cumsum(singular_energy, dim=0) / total_energy
    channel_energy = torch.sum(calibration * calibration, dim=0)
    channel_rms = torch.sqrt(torch.mean(calibration * calibration, dim=0))
    channel_max = torch.max(torch.abs(calibration), dim=0).values
    median = torch.median(channel_rms)
    mad = torch.median(torch.abs(channel_rms - median))
    threshold = median + 3.0 * 1.4826 * mad
    outliers = torch.nonzero(channel_rms > threshold).flatten()
    top_channels = torch.argsort(channel_energy, descending=True)[:16]
    position_norms = torch.linalg.vector_norm(calibration, dim=1)
    top_positions = torch.argsort(position_norms, descending=True)[:8]
    held_out_energy = float(torch.sum(held_out * held_out).item())
    held_out_basis_energy = {}
    for rank in RANKS:
        basis = right_vectors[:rank]
        oracle_projection = (held_out @ basis.T) @ basis
        held_out_basis_energy[str(rank)] = (
            float(torch.sum(oracle_projection * oracle_projection).item())
            / held_out_energy
        )
    structure = {
        "matrix_shape": list(calibration.shape),
        "singular_values": [float(value) for value in singular_values.tolist()],
        "singular_energy_fractions": [
            float(value) for value in (singular_energy / total_energy).tolist()
        ],
        "cumulative_explained_energy": [
            float(value) for value in cumulative.tolist()
        ],
        "rank_explained_energy": {
            str(rank): float(cumulative[rank - 1].item()) for rank in RANKS
        },
        "held_out_oracle_energy_in_calibration_basis": held_out_basis_energy,
        "effective_rank_90_percent": next(
            (
                index + 1
                for index, value in enumerate(cumulative.tolist())
                if value >= LOW_RANK_ENERGY_THRESHOLD
            ),
            len(cumulative),
        ),
        "channel_structure": {
            "energy_concentration": {
                f"top_{count}": concentration(channel_energy, count)
                for count in (1, 4, 8, 16)
            },
            "outlier_policy": (
                "calibration channel RMS greater than median plus "
                "3 times 1.4826 MAD"
            ),
            "median_rms": float(median.item()),
            "mad_rms": float(mad.item()),
            "outlier_threshold_rms": float(threshold.item()),
            "outlier_channel_indices": [int(value) for value in outliers.tolist()],
            "top_channels": [
                {
                    "channel": int(index),
                    "energy_fraction": float(
                        channel_energy[index].item() / channel_energy.sum().item()
                    ),
                    "rms": float(channel_rms[index].item()),
                    "maximum_absolute_error": float(channel_max[index].item()),
                }
                for index in top_channels.tolist()
            ],
        },
        "position_structure": {
            "top_calibration_positions": [
                {
                    "position": int(CALIBRATION_POSITIONS[index]),
                    "residual_l2": float(position_norms[index].item()),
                }
                for index in top_positions.tolist()
            ]
        },
    }
    return structure, singular_values, right_vectors


def cost_estimate(rank: int) -> dict[str, Any]:
    input_width = 896
    output_width = 128
    weight_values = rank * (input_width + output_width)
    scale_records = rank + output_width
    return {
        "estimate_only": True,
        "integer_execution_evaluated": False,
        "hypothetical_datapath": (
            "signed-A8 input, two signed-W4 low-rank matrices, signed-A8 "
            "rank intermediate, signed-32 accumulators, and retained V A8 output"
        ),
        "rank": rank,
        "weight_values": weight_values,
        "w4_weight_payload_bytes": (weight_values + 1) // 2,
        "scale32_records": scale_records,
        "scale32_metadata_bytes": 4 * scale_records,
        "rank_activation_bytes": rank,
        "estimated_total_parameter_and_scale_bytes": (
            (weight_values + 1) // 2 + 4 * scale_records
        ),
        "macs_per_token": rank * (input_width + output_width),
        "first_factor_macs": input_width * rank,
        "second_factor_macs": output_width * rank,
        "requantization_multiplier_count": scale_records,
        "accumulator_width_bits": 32,
    }


def decision(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_projection: dict[str, Any],
    candidate_projection: dict[str, Any],
) -> dict[str, Any]:
    exact = (
        candidate_metrics["target_token_rank"] == 1
        and candidate_metrics["target_token_top1"]
        and candidate_metrics["top8_overlap_count"]
        >= baseline_metrics["top8_overlap_count"]
        and candidate_metrics["jensen_shannon_divergence_nats"]
        <= baseline_metrics["jensen_shannon_divergence_nats"]
    )
    material = (
        candidate_metrics["target_token_rank"] <= TOP_K
        and candidate_metrics["top8_overlap_count"]
        > baseline_metrics["top8_overlap_count"]
        and candidate_metrics["jensen_shannon_divergence_nats"]
        <= baseline_metrics["jensen_shannon_divergence_nats"] - 0.05
    )
    baseline_held_out = baseline_projection["held_out"]["relative_l2"]
    candidate_held_out = candidate_projection["held_out"]["relative_l2"]
    return {
        "exact_rank1_without_top8_or_jsd_regression": exact,
        "material_rank_top8_jsd_recovery": material,
        "held_out_projection_relative_l2": candidate_held_out,
        "baseline_held_out_projection_relative_l2": baseline_held_out,
        "held_out_no_regression": candidate_held_out <= baseline_held_out,
        "diagnostic_pass": (exact or material)
        and candidate_held_out <= baseline_held_out,
        "promotion_eligible": False,
    }


def generate_sums() -> None:
    localizer = diagnosis.prior.localizer
    paths = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path != SUMS
    )
    lines = [
        f"{localizer.sha256_file(path)}  {path.relative_to(OUTPUT)}"
        for path in paths
    ]
    SUMS.write_text("\n".join(lines) + "\n", encoding="ascii")


def generate() -> None:
    require(FREEZE.is_file(), "diagnosis freeze is missing")
    require(not RESULT.exists(), "residual diagnosis result already exists")
    require(not SUMS.exists(), "residual diagnosis was already finalized")
    require(
        not AUTHORITY_CONSUMPTION.exists(),
        "bounded diagnosis authority was already consumed",
    )
    freeze = read_json(FREEZE)
    validate_freeze(freeze)
    localizer = diagnosis.prior.localizer
    require(
        localizer.sha256_file(Path(__file__).resolve())
        == freeze["runner"]["sha256"],
        "preregistered runner hash changed",
    )
    verify_sum_tree(
        ACCEPTED_DIAGNOSIS_SUMS,
        freeze["bindings"]["accepted_qkv_diagnosis_sums_sha256"],
    )
    verify_sum_tree(
        ACCEPTED_JOINT_SUMS,
        freeze["bindings"]["accepted_joint_scale_negative_sums_sha256"],
    )
    require(
        localizer.sha256_file(ACCEPTED_DIAGNOSIS)
        == freeze["bindings"]["accepted_qkv_diagnosis_result_sha256"],
        "accepted QKV diagnosis changed",
    )
    require(
        localizer.sha256_file(ACCEPTED_JOINT_RESULT)
        == freeze["bindings"]["accepted_joint_scale_negative_result_sha256"],
        "accepted joint-scale negative changed",
    )
    require(
        localizer.sha256_file(BF16_LOGITS)
        == freeze["bindings"]["bf16_logits_sha256"],
        "BF16 logits changed",
    )
    authority_consumption = consume_authority(
        {
            "schema_version": 1,
            "authority_id": freeze["authority"]["authority_id"],
            "mission_id": MISSION_ID,
            "consumed_at_utc": datetime.now(UTC).isoformat(),
            "status": "CONSUMED_EXACTLY_ONCE",
            "freeze": localizer.file_record(FREEZE),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "scope": freeze["authority"]["scope"],
        }
    )
    protected_before = protected_records()
    started = time.monotonic()
    snapshot = diagnosis.prior.generation_runner.resolve_snapshot()
    diagnosis.prior.backend.configure_snapshot(snapshot)
    canonical = diagnosis.prior.backend.canonical
    require(
        localizer.sha256_file(canonical.MODEL) == canonical.MODEL_SHA256,
        "base model hash changed",
    )
    require(
        localizer.sha256_file(canonical.ADAPTER) == canonical.ADAPTER_SHA256,
        "checkpoint-176 adapter hash changed",
    )
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
        == freeze["prompt"]["token_ids_sha256"],
        "frozen token IDs changed",
    )
    require(
        localizer.sha256_bytes(diagnosis.prior.PROMPT.encode("utf-8"))
        == freeze["prompt"]["sha256"],
        "frozen prompt changed",
    )
    diagnosis.prior.localizer.PROMPT = diagnosis.prior.PROMPT
    diagnosis.prior.localizer.TOKEN_IDS = token_ids
    diagnosis.prior.localizer.REFERENCE_TOKEN = TARGET_TOKEN

    print("ACE2_LAYER23_V_RESIDUAL_BF16_CAPTURE", flush=True)
    layer_input, captures, bf16_logits = diagnosis.capture_layer_context(
        snapshot, token_ids
    )
    require(
        localizer.tensor_sha256(captures["input_norm"])
        == freeze["inputs"]["input_norm_sha256"]
        and localizer.tensor_sha256(captures["v"])
        == freeze["inputs"]["v_oracle_sha256"],
        "accepted frozen V tensors changed",
    )
    hidden_states = [torch.empty(0)] * LAYER + [layer_input]
    suppressed_messages: list[str] = []
    prior_output = diagnosis.prior.OUTPUT
    diagnosis.prior.OUTPUT = OUTPUT
    try:
        with safe_open(
            canonical.MODEL, framework="pt", device="cpu"
        ) as weights, safe_open(
            canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            embedding = weights.get_tensor(
                "model.embed_tokens.weight"
            ).contiguous()
            norm_gain = weights.get_tensor("model.norm.weight").contiguous()
            head = localizer.derive_lm_head_weights(embedding)
            cache = ResidualCorrectionCache(weights, adapter)
            guard = diagnosis.prior.localizer.SubstitutionRequireGuard()
            cache.install()
            guard.install()
            try:
                baseline_record, baseline_state = run_downstream(
                    cache,
                    "baseline-w4a8",
                    None,
                    hidden_states,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                    bf16_logits,
                )
                baseline_output = baseline_state["corrected_output"]
                input_dequantized = baseline_state["input_dequantized"]
                require(
                    isinstance(baseline_output, torch.Tensor)
                    and isinstance(input_dequantized, torch.Tensor),
                    "baseline tensor state changed",
                )
                oracle_v = captures["v"].to(torch.float64)
                residual = oracle_v - baseline_output.to(torch.float64)
                structure, singular_values, right_vectors = residual_structure(
                    residual
                )
                calibration_input = input_dequantized[
                    list(CALIBRATION_POSITIONS)
                ].to(torch.float64)
                calibration_residual = residual[
                    list(CALIBRATION_POSITIONS)
                ]
                candidate_runs = []
                factor_records = {}
                for rank in RANKS:
                    basis = right_vectors[:rank].contiguous()
                    coefficients = calibration_residual @ basis.T
                    left_factor = (
                        torch.linalg.pinv(
                            calibration_input, rtol=PINV_RTOL
                        )
                        @ coefficients
                    ).contiguous()
                    predicted = (
                        input_dequantized.to(torch.float64)
                        @ left_factor
                        @ basis
                    ).contiguous()
                    record, state = run_downstream(
                        cache,
                        f"rank-{rank:02d}",
                        predicted,
                        hidden_states,
                        weights,
                        adapter,
                        norm_gain,
                        embedding,
                        head,
                        tokenizer,
                        bf16_logits,
                    )
                    corrected_output = state["corrected_output"]
                    require(
                        isinstance(corrected_output, torch.Tensor),
                        "corrected V output state changed",
                    )
                    factor_directory = OUTPUT / "factors" / f"rank-{rank:02d}"
                    factor_records[str(rank)] = {
                        "left_factor": write_tensor(
                            factor_directory / "input_to_rank_f64le.bin",
                            left_factor,
                            "float64_le",
                        ),
                        "right_basis": write_tensor(
                            factor_directory / "rank_to_channel_f64le.bin",
                            basis,
                            "float64_le",
                        ),
                        "predicted_residual": write_tensor(
                            factor_directory / "predicted_residual_f64le.bin",
                            predicted,
                            "float64_le",
                        ),
                        "corrected_v_output": write_tensor(
                            factor_directory / "corrected_v_output_f64le.bin",
                            corrected_output.to(torch.float64),
                            "float64_le",
                        ),
                    }
                    metrics = precision.summary_metrics(record)
                    projection = split_projection_summary(
                        corrected_output.to(torch.float64), oracle_v
                    )
                    fit = {
                        "calibration_predicted_residual_relative_l2": relative_l2(
                            predicted[list(CALIBRATION_POSITIONS)],
                            calibration_residual,
                        ),
                        "held_out_predicted_residual_relative_l2": relative_l2(
                            predicted[list(HELD_OUT_POSITIONS)],
                            residual[list(HELD_OUT_POSITIONS)],
                        ),
                        "calibration_predicted_explained_energy": energy_fraction(
                            calibration_residual,
                            predicted[list(CALIBRATION_POSITIONS)],
                        ),
                        "held_out_predicted_explained_energy": energy_fraction(
                            residual[list(HELD_OUT_POSITIONS)],
                            predicted[list(HELD_OUT_POSITIONS)],
                        ),
                    }
                    candidate_runs.append(
                        {
                            "rank": rank,
                            "metrics": metrics,
                            "logits": record["accepted_w4a8_logits"],
                            "projection_output": projection,
                            "fit_and_overfit_separation": {
                                **fit,
                                "relative_l2_generalization_gap": (
                                    fit[
                                        "held_out_predicted_residual_relative_l2"
                                    ]
                                    - fit[
                                        "calibration_predicted_residual_relative_l2"
                                    ]
                                ),
                                "held_out_positions_used_for_fit": False,
                            },
                            "cost_estimate": cost_estimate(rank),
                            "artifacts": factor_records[str(rank)],
                        }
                    )
            finally:
                suppressed_messages.extend(guard.suppressed)
                guard.restore()
                cache.restore()
    finally:
        diagnosis.prior.OUTPUT = prior_output

    baseline_metrics = precision.summary_metrics(baseline_record)
    require(
        baseline_record["accepted_w4a8_logits"]["sha256"]
        == EXPECTED_BASELINE["logits_sha256"]
        and baseline_metrics["target_token_rank"]
        == EXPECTED_BASELINE["target_rank"]
        and baseline_metrics["top8_overlap_count"]
        == EXPECTED_BASELINE["top8_overlap_count"]
        and math.isclose(
            baseline_metrics["jensen_shannon_divergence_nats"],
            EXPECTED_BASELINE["jensen_shannon_divergence_nats"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            baseline_metrics["relative_l2"],
            EXPECTED_BASELINE["relative_l2"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and baseline_metrics["top_token_id"]
        == EXPECTED_BASELINE["top_token_id"],
        "Fresh-L2-accepted cut-23 baseline did not reproduce",
    )
    baseline_projection = split_projection_summary(
        baseline_output.to(torch.float64), oracle_v
    )
    for candidate in candidate_runs:
        candidate["decision"] = decision(
            baseline_metrics,
            candidate["metrics"],
            baseline_projection,
            candidate["projection_output"],
        )
    winners = [
        candidate
        for candidate in candidate_runs
        if candidate["decision"]["diagnostic_pass"]
    ]
    rank8_energy = structure["rank_explained_energy"]["8"]
    if winners:
        diagnosis_class = "LOW_RANK_CORRECTION_SIGNAL_WITH_HELD_OUT_SUPPORT"
    elif rank8_energy >= LOW_RANK_ENERGY_THRESHOLD:
        diagnosis_class = (
            "LOW_RANK_CALIBRATION_STRUCTURE_WITHOUT_HELD_OUT_QUALITY_RECOVERY"
        )
    else:
        diagnosis_class = "HIGH_RANK_OR_DIFFUSE_V_RESIDUAL"

    tensors = {
        "input_dequantized": write_tensor(
            OUTPUT / "tensors/v_input_dequantized_f64le.bin",
            input_dequantized.to(torch.float64),
            "float64_le",
        ),
        "bf16_v_output": write_tensor(
            OUTPUT / "tensors/bf16_v_output_f64le.bin",
            oracle_v,
            "float64_le",
        ),
        "baseline_w4a8_v_output": write_tensor(
            OUTPUT / "tensors/baseline_w4a8_v_output_f64le.bin",
            baseline_output.to(torch.float64),
            "float64_le",
        ),
        "bf16_minus_w4a8_residual": write_tensor(
            OUTPUT / "tensors/bf16_minus_w4a8_residual_f64le.bin",
            residual,
            "float64_le",
        ),
        "singular_values": write_tensor(
            OUTPUT / "tensors/calibration_singular_values_f64le.bin",
            singular_values,
            "float64_le",
        ),
        "right_singular_vectors": write_tensor(
            OUTPUT / "tensors/calibration_right_singular_vectors_f64le.bin",
            right_vectors,
            "float64_le",
        ),
    }
    protected_after = protected_records()
    require(
        protected_before == protected_after,
        "accepted predecessor evidence changed",
    )
    shutil.copyfile(Path(__file__).resolve(), EXECUTION_RUNNER)
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": "V_RESIDUAL_STRUCTURE_DIAGNOSIS_PENDING_FRESH_L2",
        "classification": (
            "bounded_nonofficial_software_only_layer23_v_residual_diagnosis"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution": {
            "ranks": list(RANKS),
            "layer_cut": LAYER,
            "prompt_token_count": len(token_ids),
            "generation_position": 0,
            "elapsed_wall_seconds": time.monotonic() - started,
        },
        "bindings": {
            "freeze": localizer.file_record(FREEZE),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "archived_execution_runner": localizer.file_record(EXECUTION_RUNNER),
            "authorization_consumption": authority_consumption,
            "protected_before": protected_before,
            "protected_after": protected_after,
            "base_model": {
                "bytes": canonical.MODEL.stat().st_size,
                "sha256": canonical.MODEL_SHA256,
            },
            "checkpoint_176_adapter": {
                "bytes": canonical.ADAPTER.stat().st_size,
                "sha256": canonical.ADAPTER_SHA256,
            },
            "captured_tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": localizer.tensor_sha256(value),
                }
                for name, value in sorted(captures.items())
            },
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
        },
        "partition": copy.deepcopy(freeze["partition"]),
        "fit_policy": copy.deepcopy(freeze["fit_policy"]),
        "baseline": {
            "metrics": baseline_metrics,
            "logits": baseline_record["accepted_w4a8_logits"],
            "projection_output": baseline_projection,
        },
        "residual_structure": structure,
        "candidates": candidate_runs,
        "conclusion": {
            "diagnosis": diagnosis_class,
            "rank8_calibration_explained_energy": rank8_energy,
            "small_structured_correction_supported": bool(winners),
            "supported_ranks": [
                int(candidate["rank"]) for candidate in winners
            ],
            "candidate_promotion_authorized": False,
            "next_boundary": (
                "Any correction architecture requires a separate reviewed "
                "hardware-candidate task."
            ),
        },
        "failure_taxonomy": (
            None
            if winners
            else {
                "class": "LOW_RANK_RESIDUAL_CORRECTION_NO_HELD_OUT_RECOVERY",
                "root_cause_hypothesis": (
                    "Calibration residual structure does not translate into a "
                    "small correction that preserves held-out V projection and "
                    "restores downstream token quality."
                ),
                "regression": (
                    "Retain ranks 1/2/4/8, the 0-16 versus 17-33 split, and "
                    "the unchanged cut-23 logit metrics for any structurally "
                    "different correction proposal."
                ),
            }
        ),
        "artifacts": {
            "tensors": tensors,
            "rank_factors": factor_records,
        },
        "suppressed_nonfunctional_cache_effect_assertions": {
            "count": len(suppressed_messages),
            "unique_messages": sorted(set(suppressed_messages)),
        },
        "independent_review_required": True,
        "scope_guards": copy.deepcopy(SCOPE_GUARDS),
    }
    write_json(RESULT, result)
    review = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "review_type": "Fresh-L2",
        "status": "PENDING_INDEPENDENT_REVIEW",
        "result": localizer.file_record(RESULT),
        "diagnosis_freeze": localizer.file_record(FREEZE),
        "execution_runner": localizer.file_record(EXECUTION_RUNNER),
        "reproduction_command": (
            ".venv/bin/python "
            "tools/diagnose_stage1_layer23_v_residual_structure.py "
            "--verify-only"
        ),
        "required_checks": [
            "Recompute the calibration-only V residual SVD and channel/outlier statistics.",
            "Confirm only positions 0-16 determine every rank factor and predictor.",
            "Recompute rank-1/2/4/8 predicted corrections and held-out projection errors.",
            "Recompute token-39814 rank, top-8 overlap, JSD, and logit relative-L2.",
            "Confirm cost values are estimates and no integer candidate was executed.",
            "Confirm no RTL, official attempt, attempt-0003, PPA, or Stage-2 artifact was produced.",
        ],
    }
    write_json(REVIEW_REQUEST, review)
    generate_sums()
    print(
        "ACE2_LAYER23_V_RESIDUAL_RESULT "
        f"diagnosis={diagnosis_class} "
        f"supported_ranks={result['conclusion']['supported_ranks']} "
        f"output={localizer.public_path(OUTPUT)}",
        flush=True,
    )


def verify() -> None:
    require(RESULT.is_file(), "residual diagnosis result is missing")
    require(REVIEW_REQUEST.is_file(), "Fresh-L2 review request is missing")
    require(EXECUTION_RUNNER.is_file(), "archived execution runner is missing")
    require(
        AUTHORITY_CONSUMPTION.is_file(),
        "authority consumption marker is missing",
    )
    require(SUMS.is_file(), "residual diagnosis SHA256SUMS is missing")
    localizer = diagnosis.prior.localizer
    for line in SUMS.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        artifact = OUTPUT / relative
        require(artifact.is_file(), f"summed artifact missing: {relative}")
        require(
            localizer.sha256_file(artifact) == expected,
            f"summed artifact hash changed: {relative}",
        )
    freeze = read_json(FREEZE)
    validate_freeze(freeze)
    result = read_json(RESULT)
    require(
        result["scope_guards"] == SCOPE_GUARDS,
        "result scope guards changed",
    )
    require(
        result["bindings"]["runner"]["sha256"] == freeze["runner"]["sha256"]
        and result["bindings"]["archived_execution_runner"]["sha256"]
        == freeze["runner"]["sha256"],
        "runner binding changed",
    )
    consumption = read_json(AUTHORITY_CONSUMPTION)
    require(
        consumption["status"] == "CONSUMED_EXACTLY_ONCE"
        and consumption["authority_id"] == freeze["authority"]["authority_id"]
        and consumption["freeze"] == localizer.file_record(FREEZE)
        and consumption["runner"]["bytes"]
        == localizer.file_record(EXECUTION_RUNNER)["bytes"]
        and consumption["runner"]["sha256"]
        == localizer.file_record(EXECUTION_RUNNER)["sha256"]
        and result["bindings"]["authorization_consumption"]
        == localizer.file_record(AUTHORITY_CONSUMPTION),
        "authority consumption binding changed",
    )
    verify_sum_tree(
        ACCEPTED_DIAGNOSIS_SUMS,
        freeze["bindings"]["accepted_qkv_diagnosis_sums_sha256"],
    )
    verify_sum_tree(
        ACCEPTED_JOINT_SUMS,
        freeze["bindings"]["accepted_joint_scale_negative_sums_sha256"],
    )
    require(
        protected_records()
        == result["bindings"]["protected_before"]
        == result["bindings"]["protected_after"],
        "accepted predecessor evidence changed",
    )
    residual = load_tensor(
        result["artifacts"]["tensors"]["bf16_minus_w4a8_residual"]
    ).to(torch.float64)
    input_dequantized = load_tensor(
        result["artifacts"]["tensors"]["input_dequantized"]
    ).to(torch.float64)
    oracle = load_tensor(
        result["artifacts"]["tensors"]["bf16_v_output"]
    ).to(torch.float64)
    baseline_output = load_tensor(
        result["artifacts"]["tensors"]["baseline_w4a8_v_output"]
    ).to(torch.float64)
    require(
        torch.equal(residual, oracle - baseline_output),
        "retained residual tensor changed",
    )
    recomputed_structure, singular_values, right_vectors = residual_structure(
        residual
    )
    require(
        recomputed_structure == result["residual_structure"],
        "residual structure metrics changed",
    )
    require(
        torch.equal(
            singular_values,
            load_tensor(result["artifacts"]["tensors"]["singular_values"]).to(
                torch.float64
            ),
        )
        and torch.equal(
            right_vectors,
            load_tensor(
                result["artifacts"]["tensors"]["right_singular_vectors"]
            ).to(torch.float64),
        ),
        "retained SVD tensors changed",
    )
    reference = np.fromfile(BF16_LOGITS, dtype="<f4")
    baseline_projection = split_projection_summary(baseline_output, oracle)
    require(
        baseline_projection == result["baseline"]["projection_output"],
        "baseline projection metrics changed",
    )
    for candidate in result["candidates"]:
        rank = int(candidate["rank"])
        factors = result["artifacts"]["rank_factors"][str(rank)]
        left = load_tensor(factors["left_factor"]).to(torch.float64)
        basis = load_tensor(factors["right_basis"]).to(torch.float64)
        predicted = load_tensor(factors["predicted_residual"]).to(torch.float64)
        corrected = load_tensor(factors["corrected_v_output"]).to(torch.float64)
        require(
            torch.equal(predicted, input_dequantized @ left @ basis),
            f"rank-{rank} predicted residual changed",
        )
        logits = np.fromfile(ROOT / candidate["logits"]["path"], dtype="<f8")
        comparison = diagnosis.prior.quality.score_comparison(reference, logits)
        require(
            comparison
            == candidate["metrics"]["comparison_vs_checkpoint176_bf16"],
            f"rank-{rank} logit metrics changed",
        )
        projection = split_projection_summary(corrected, oracle)
        require(
            projection == candidate["projection_output"],
            f"rank-{rank} projection metrics changed",
        )
        require(
            candidate["decision"]
            == decision(
                result["baseline"]["metrics"],
                candidate["metrics"],
                baseline_projection,
                projection,
            ),
            f"rank-{rank} decision changed",
        )
        require(
            candidate["cost_estimate"] == cost_estimate(rank),
            f"rank-{rank} cost estimate changed",
        )
    require(
        not (
            ROOT
            / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0003"
        ).exists(),
        "attempt-0003 exists",
    )
    request = read_json(REVIEW_REQUEST)
    require(
        request["status"] == "PENDING_INDEPENDENT_REVIEW"
        and request["result"] == localizer.file_record(RESULT)
        and request["diagnosis_freeze"] == localizer.file_record(FREEZE)
        and request["execution_runner"]
        == localizer.file_record(EXECUTION_RUNNER),
        "Fresh-L2 review request binding changed",
    )
    print(
        "PASS stage1vresidualstructure01 "
        f"diagnosis={result['conclusion']['diagnosis']} "
        f"rank8_energy={result['conclusion']['rank8_calibration_explained_energy']:.9f} "
        f"supported_ranks={result['conclusion']['supported_ranks']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        generate()
        verify()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ACE2_LAYER23_V_RESIDUAL_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
