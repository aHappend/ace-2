#!/usr/bin/env python3
"""Search grouped signed-A8 Scale32 at layer-17 RMSNorm output/Q input."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
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
from tools import ace2_checkpoint176_layer16_source_grouped_scale32_search as layer16
from tools import ace2_checkpoint176_layer17_q_output_grouped_scale32_search as qoutput
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    scale32_ratio,
)
from tools.ace2_rmsnorm_reference import (
    derive_rmsnorm_output_scale,
    derive_scaled_gains_q8,
    reference_rmsnorm,
)


MISSION = "w4a8-layer17-rmsnorm-output-source-grouped-repair-v1"
TARGET_LAYER = layer16.TARGET_LAYER
SOURCE_GROUP_SIZE = 1
Q_OUTPUT_GROUP_SIZE = 128
RMS_OUTPUT_GROUP_SIZES = (448, 128, 64)
HEAD_DIM = backend.HEAD_DIM
Q_PATTERN = re.compile(r"layer17_position(\d+)_q$")
PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer17-q-output-source-grouped-repair-v1/"
    "candidate-0001"
)
PREDECESSOR_SEARCH_REVIEW = ROOT / "build/l2-review-layer17-q-output-search-r1/result.json"
PREDECESSOR_RTL_REVIEW = ROOT / "build/l2-review-layer17-q-output-rtl-r1/result.json"
RTL_DEP = ROOT / "rtl/ace2_rmsnorm_core.sv"
RTL = ROOT / "rtl/ace2_layer17_rmsnorm_output_grouped_scale32_core.sv"
RTL_TOP = "ace2_layer17_rmsnorm_output_grouped_scale32_core"


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _rms_group_metadata(
    calibrated_output: torch.Tensor,
    weights: torch.Tensor,
    group_size: int,
) -> tuple[tuple[int, ...], list[int]]:
    scales = []
    gains = []
    for start in range(0, backend.HIDDEN, group_size):
        stop = start + group_size
        group_weights = weights[start:stop].to(torch.float64).tolist()
        calibrated_absmax = float(
            calibrated_output[start:stop].to(torch.float64).abs().max()
        )
        output_scale = derive_rmsnorm_output_scale(
            group_weights, max(calibrated_absmax, 1.0e-12)
        )
        scale32 = ceil_scale32_from_float(output_scale)
        scales.append(scale32)
        gains.extend(derive_scaled_gains_q8(group_weights, _scale32_float(scale32)))
    return tuple(scales), gains


def _projection_reals(
    partials: torch.Tensor,
    input_scale32: tuple[int, ...],
    weight_scale32: tuple[int, ...],
) -> list[Fraction]:
    input_ratios = [scale32_ratio(record) for record in input_scale32]
    result = []
    for row in range(partials.shape[0]):
        activation = Fraction(0, 1)
        for group, (numerator, denominator) in enumerate(input_ratios):
            activation += Fraction(
                int(partials[row, group]) * numerator, denominator
            )
        weight_numerator, weight_denominator = scale32_ratio(weight_scale32[row])
        result.append(
            activation * Fraction(weight_numerator, weight_denominator)
        )
    return result


def _output_group_scales(
    values: list[Fraction], group_size: int
) -> tuple[int, ...]:
    scales = []
    for start in range(0, len(values), group_size):
        maximum = max((abs(value) for value in values[start : start + group_size]))
        scales.append(
            ceil_scale32_from_ratio(maximum.numerator, maximum.denominator * 127)
        )
    return tuple(scales)


def _quantize_fraction(value: Fraction, scale32: int) -> int:
    scale_numerator, scale_denominator = scale32_ratio(scale32)
    return round_divide_even_signed(
        value.numerator * scale_denominator,
        value.denominator * scale_numerator,
    )


class GroupedRmsnormQInputScale32Cache(qoutput.GroupedQOutputScale32Cache):
    """Replace only the layer-17 RMSNorm output/Q-input quantizer."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        self.rms_output_group_size = RMS_OUTPUT_GROUP_SIZES[0]
        self.rms_weight = weights.get_tensor(
            f"model.layers.{TARGET_LAYER}.input_layernorm.weight"
        ).contiguous()

    def configure(self, rms_output_group_size: int) -> None:
        localizer.require(
            backend.HIDDEN % rms_output_group_size == 0,
            "RMSNorm-output group does not divide hidden",
        )
        localizer.require(
            rms_output_group_size % HEAD_DIM == 0,
            "RMSNorm-output group must contain complete Q-head channels",
        )
        self.rms_output_group_size = rms_output_group_size
        super().configure(Q_OUTPUT_GROUP_SIZE)

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
        record = self.position_records[position]
        record["tensor_rms_gain_s16_q8"] = record["rms_gain_s16_q8"].clone()
        record["tensor_rms_output_s8"] = record["rms_output_s8"].clone()
        record["tensor_q_input_scale32"] = int(record["q_input_scale32"])

        group_scales, gains = _rms_group_metadata(
            record["binary_input_norm"], self.rms_weight, self.rms_output_group_size
        )
        rms = reference_rmsnorm(
            record["sum_s8"].to(torch.int64).tolist(), gains
        )
        rms_output_s8 = torch.tensor(rms.outputs, dtype=torch.int8)
        group_scale_tensor = torch.tensor(group_scales, dtype=torch.int64)

        qweight = self.q_metadata["qweight"]
        partials = []
        for start in range(0, backend.HIDDEN, self.rms_output_group_size):
            stop = start + self.rms_output_group_size
            partials.append(
                torch.mv(
                    qweight[:, start:stop].to(torch.int64),
                    rms_output_s8[start:stop].to(torch.int64),
                )
            )
        partial_matrix = torch.stack(partials, dim=1).contiguous()
        localizer.require(
            bool(torch.all(partial_matrix >= -(1 << 31)))
            and bool(torch.all(partial_matrix < (1 << 31))),
            f"{name} partial accumulator overflow",
        )

        real_output = _projection_reals(
            partial_matrix, group_scales, self.q_weight_scale32
        )
        q_output_scales = _output_group_scales(real_output, Q_OUTPUT_GROUP_SIZE)
        rounded_values = [
            _quantize_fraction(
                value, q_output_scales[row // Q_OUTPUT_GROUP_SIZE]
            )
            for row, value in enumerate(real_output)
        ]
        rounded = torch.tensor(rounded_values, dtype=torch.int64)
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        q_output_scale_tensor = torch.tensor(q_output_scales, dtype=torch.int64)
        head_scales = torch.tensor(
            [
                q_output_scales[(head * HEAD_DIM) // Q_OUTPUT_GROUP_SIZE]
                for head in range(backend.Q_HEADS)
            ],
            dtype=torch.int64,
        )
        tensor_scale = _output_group_scales(real_output, backend.HIDDEN)[0]
        real_output_tensor = torch.tensor(
            [float(value) for value in real_output], dtype=torch.float32
        )

        record.update(
            {
                "rms_output_group_size": self.rms_output_group_size,
                "rms_output_group_scale32": group_scale_tensor,
                "rms_weight_f32": self.rms_weight.to(torch.float32),
                "rms_gain_s16_q8": torch.tensor(gains, dtype=torch.int16),
                "rms_output_s8": rms_output_s8,
                "rms_sumsq": rms.sumsq,
                "rms_inv_q30": rms.inv_rms_q30,
                "rms_saturation": rms.saturation_seen,
                "q_weight_s4": qweight,
                "q_weight_scale32": torch.tensor(
                    self.q_weight_scale32, dtype=torch.int64
                ),
                "q_partial_accumulator_s32": partial_matrix,
                "q_output_group_scale32": q_output_scale_tensor,
                "q_head_scale32": head_scales,
                "q_rounded_s64": rounded,
                "q_output_s8": output_q,
                "q_saturation": saturation,
                "q_tensor_output_scale32": tensor_scale,
                "binary_q_float_output": real_output_tensor,
            }
        )
        return {
            "name": name,
            "input_q": rms_output_s8,
            "input_scale": _scale32_float(group_scales[0]),
            "qweight": qweight,
            "weight_scale": self.q_metadata["weight_scale"],
            "multiplier": torch.zeros(qweight.shape[0], dtype=torch.int64),
            "right_shift": torch.zeros(qweight.shape[0], dtype=torch.int64),
            "accumulator": partial_matrix.sum(dim=1),
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": _scale32_float(int(head_scales[0])),
            "saturation": saturation,
            "float_output": real_output_tensor,
            "source_hashes": source_hashes,
            "exact_grouped_rmsnorm_q_input_scale32": True,
        }


def _artifacts(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    artifacts = {
        name: value for name, value in record.items() if isinstance(value, torch.Tensor)
    }
    for name in (
        "attention_base_scale32",
        "down_base_scale32",
        "sum_scale32",
        "tensor_q_input_scale32",
        "q_tensor_output_scale32",
        "rms_sumsq",
        "rms_inv_q30",
    ):
        artifacts[name] = torch.tensor([int(record[name])], dtype=torch.int64)
    artifacts["rms_saturation"] = torch.tensor(
        [int(record["rms_saturation"])], dtype=torch.uint8
    )
    return artifacts


def run_group(
    rms_output_group_size: int,
    cache: GroupedRmsnormQInputScale32Cache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure(rms_output_group_size)
    source_guard = layer16.ExactScale32SourceGuard(
        cache, weights, SOURCE_GROUP_SIZE
    )
    score_guard = qoutput.QHeadScaleScoreGuard(cache)
    cache.install()
    score_guard.install()
    source_guard.install()
    try:
        result, _ = localizer.run_cut(
            None, [], weights, adapter, norm_gain, embedding, head, tokenizer
        )
    finally:
        source_guard.restore()
        score_guard.restore()
        cache.restore()

    expected_scores = backend.Q_HEADS * (2 * len(localizer.TOKEN_IDS) - 1)
    localizer.require(
        score_guard.replacements == expected_scores,
        "accepted grouped Q-output scale did not reach every layer-17 score",
    )
    position = len(localizer.TOKEN_IDS) - 1
    exact = source_guard.records[position]
    rms_scales = exact["rms_output_group_scale32"]
    q_scales = exact["q_output_group_scale32"]
    result.update(
        {
            "source_group_size": SOURCE_GROUP_SIZE,
            "rms_output_group_size": rms_output_group_size,
            "rms_output_group_count": backend.HIDDEN // rms_output_group_size,
            "rms_output_group_scale32_min": f"0x{int(rms_scales.min()):08x}",
            "rms_output_group_scale32_max": f"0x{int(rms_scales.max()):08x}",
            "rms_output_s8_sha256": localizer.tensor_sha256(
                exact["rms_output_s8"]
            ),
            "rms_saturation": bool(exact["rms_saturation"]),
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "q_output_group_count": backend.HIDDEN // Q_OUTPUT_GROUP_SIZE,
            "q_output_group_scale32_min": f"0x{int(q_scales.min()):08x}",
            "q_output_group_scale32_max": f"0x{int(q_scales.max()):08x}",
            "q_output_s8_sha256": localizer.tensor_sha256(exact["q_output_s8"]),
            "q_saturation_count": int(
                exact["q_saturation"].to(torch.int64).sum()
            ),
            "layer17_score_scale_replacements": score_guard.replacements,
        }
    )
    return result, _artifacts(exact)


def _rtl_preflight(output: Path) -> dict[str, Any]:
    log = output / "rtl-interface-preflight.log"
    with tempfile.TemporaryDirectory(
        prefix="ace2-layer17-rmsnorm-output-", dir=ROOT / "build"
    ) as temporary:
        executable = Path(temporary) / "preflight.vvp"
        command = [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            RTL_TOP,
            "-o",
            str(executable),
            str(RTL_DEP),
            str(RTL),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log.write_text(completed.stdout, encoding="utf-8")
    localizer.require(completed.returncode == 0, "frozen RTL interface did not compile")
    localizer.require(not completed.stdout.strip(), "RTL interface preflight emitted warnings")
    return {
        "status": "PASS_FROZEN_PUBLIC_RTL_INTERFACE_COMPILES_WARNING_CLEAN",
        "command": command,
        "log": localizer.file_record(log),
        "rtl_dependency": localizer.file_record(RTL_DEP),
        "rtl": localizer.file_record(RTL),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "candidate output already exists")

    predecessor_freeze_path = PREDECESSOR / "candidate-freeze.json"
    predecessor_result_path = PREDECESSOR / "result.json"
    predecessor_freeze = json.loads(predecessor_freeze_path.read_text(encoding="utf-8"))
    predecessor_result = json.loads(predecessor_result_path.read_text(encoding="utf-8"))
    search_review = json.loads(PREDECESSOR_SEARCH_REVIEW.read_text(encoding="utf-8"))
    rtl_review = json.loads(PREDECESSOR_RTL_REVIEW.read_text(encoding="utf-8"))
    localizer.require(
        predecessor_result["selected_q_output_group_size"] == Q_OUTPUT_GROUP_SIZE,
        "predecessor no longer selects Q-output group 128",
    )
    localizer.require(
        search_review["selected_q_output_group_size"] == Q_OUTPUT_GROUP_SIZE
        and search_review["selected_reference_rank"]
        == predecessor_result["selected_reference_rank"],
        "independent Q-output search review differs",
    )
    localizer.require(
        rtl_review["status"]
        == "PASS_FOCUSED_LAYER17_Q_OUTPUT_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT",
        "independent Q-output RTL review is absent",
    )
    localizer.require(
        predecessor_freeze["prompt"]["chat_token_ids"] == localizer.TOKEN_IDS,
        "canonical 30-token prefix differs",
    )
    baseline_rank = int(predecessor_result["selected_reference_rank"])
    baseline_top = int(predecessor_result["selected_top_token_id"])

    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "mission": MISSION,
        "classification": "candidate_only_no_official_attempt_created",
        "checkpoint": predecessor_freeze["checkpoint"],
        "prompt": predecessor_freeze["prompt"],
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "baseline": {
            "layer16_source_group_size": SOURCE_GROUP_SIZE,
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "top_token_id": baseline_top,
            "reference_rank": baseline_rank,
        },
        "ordered_rms_output_group_sizes": list(RMS_OUTPUT_GROUP_SIZES),
        "score_policy": (
            "first restored group in declared execution order; otherwise lowest "
            "reference rank strictly better than the independently accepted "
            "layer17 Q-output group-128 baseline, ties by declared execution order"
        ),
        "numeric_contract": {
            "boundary": "model.layers.17.input_rmsnorm.output_to_self_attn.q_proj.input",
            "weights": "unchanged signed W4",
            "carried_activation_samples": "signed A8",
            "rms_output_group_scale": (
                "smallest legal Scale32 not below both calibrated group absmax/127 "
                "and signed-Q7.8 gain representability floor"
            ),
            "rms_rounding": "signed round-to-nearest ties-to-even",
            "rms_saturation": "signed int8 clamp",
            "q_projection": (
                "signed W4 by signed-A8 partial dot per RMS output group with exact "
                "Scale32 rational accumulation"
            ),
            "q_output_dependency": "independently accepted group-128 Scale32",
            "bf16_runtime_sidecar": False,
            "wider_activation_payload": False,
            "token_or_logit_override": False,
        },
        "rtl_public_contract": {
            "top": RTL_TOP,
            "dependency": "ace2_rmsnorm_core",
            "activation_ports": "16 signed-A8 lanes per beat",
            "metadata_ports": "16 signed-Q7.8 gains plus Scale32/group id/last",
            "reset": "asynchronous active-low reset; synchronous clear",
            "ports": [
                "clk_i", "rst_ni", "clear_i", "start_valid_i", "start_ready_o",
                "in_valid_i", "in_ready_o", "in_data_i:16xs8", "scale_valid_i",
                "scale_ready_o", "scale_act_data_i:16xs8", "gain_data_i:16xs16",
                "group_scale32_i:u32", "group_id_i:u4", "last_i", "out_valid_o",
                "out_ready_i", "out_data_o:16xs8", "group_scale32_o:u32",
                "group_id_o:u4", "last_o", "done_valid_o", "done_ready_i",
                "sumsq_o:u48", "inv_rms_q30_o:u32", "saturation_seen_o",
            ],
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "u280_or_stage2_entered": False,
        },
        "bindings": {
            "predecessor_freeze": localizer.file_record(predecessor_freeze_path),
            "predecessor_result": localizer.file_record(predecessor_result_path),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "independent_search_review": localizer.file_record(PREDECESSOR_SEARCH_REVIEW),
            "independent_rtl_review": localizer.file_record(PREDECESSOR_RTL_REVIEW),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "rtl_dependency": localizer.file_record(RTL_DEP),
            "rtl": localizer.file_record(RTL),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)
    preflight = _rtl_preflight(output)
    localizer.write_json(output / "rtl-interface-preflight.json", preflight)

    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )

    trace = []
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = GroupedRmsnormQInputScale32Cache(weights, adapter)
        for group_size in RMS_OUTPUT_GROUP_SIZES:
            record, tensors = run_group(
                group_size,
                cache,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            group_dir = output / f"groups/group-{group_size:03d}"
            artifacts = {
                name: localizer.write_tensor(group_dir / f"{name}.bin", value)
                for name, value in sorted(tensors.items())
            }
            localizer.write_json(
                group_dir / "result.json", {"metrics": record, "artifacts": artifacts}
            )
            trace.append(record)
            print(
                "ACE2_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_CANDIDATE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} sat={record['q_saturation_count']}",
                flush=True,
            )

    order = {group: index for index, group in enumerate(RMS_OUTPUT_GROUP_SIZES)}
    restored = [item for item in trace if item["top_token_id"] == localizer.REFERENCE_TOKEN]
    improved = [item for item in trace if int(item["reference_rank"]) < baseline_rank]
    best_observed = min(
        trace,
        key=lambda item: (
            int(item["reference_rank"]),
            order[int(item["rms_output_group_size"])],
        ),
    )
    if restored:
        selected = restored[0]
        selection_reason = "restored_step0_in_declared_execution_order"
    elif improved:
        selected = min(
            improved,
            key=lambda item: (
                int(item["reference_rank"]),
                order[int(item["rms_output_group_size"])],
            ),
        )
        selection_reason = "best_honest_reference_rank_improvement"
    else:
        selected = None
        selection_reason = "no_group_improved_independently_accepted_q_output_baseline"

    validation = selected if selected is not None else best_observed
    restored_step0 = selected is not None and selected["top_token_id"] == localizer.REFERENCE_TOKEN
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_STEP0_RESTORED"
            if restored_step0
            else "PARTIAL_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_RANK_IMPROVED"
            if selected is not None
            else "PARTIAL_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_NO_IMPROVEMENT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "baseline_reference_rank": baseline_rank,
        "baseline_top_token_id": baseline_top,
        "ordered_rms_output_group_sizes": list(RMS_OUTPUT_GROUP_SIZES),
        "trace": trace,
        "selected_rms_output_group_size": (
            int(selected["rms_output_group_size"]) if selected is not None else None
        ),
        "selected_reference_rank": (
            int(selected["reference_rank"]) if selected is not None else baseline_rank
        ),
        "selected_top_token_id": (
            int(selected["top_token_id"]) if selected is not None else baseline_top
        ),
        "selection_reason": selection_reason,
        "best_observed_rms_output_group_size": int(
            best_observed["rms_output_group_size"]
        ),
        "best_observed_reference_rank": int(best_observed["reference_rank"]),
        "best_observed_top_token_id": int(best_observed["top_token_id"]),
        "rtl_validation_rms_output_group_size": int(
            validation["rms_output_group_size"]
        ),
        "first_token_restored": bool(restored_step0),
        "next_exact_upstream_boundary": (
            None
            if restored_step0
            else "model.layers.16.output_to_model.layers.17.input_rmsnorm.input"
        ),
        "next_required_action": (
            "Bind the selected grouped RMSNorm-output Scale32 payload to focused RTL."
            if selected is not None
            else "Validate the best null result in focused RTL and preserve the next upstream RMSNorm input boundary."
        ),
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "rtl_preflight": localizer.file_record(output / "rtl-interface-preflight.json"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_RESULT "
        f"status={result['status']} selected={result['selected_rms_output_group_size']} "
        f"best={result['best_observed_rms_output_group_size']} "
        f"rank={result['best_observed_reference_rank']} restored={result['first_token_restored']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
