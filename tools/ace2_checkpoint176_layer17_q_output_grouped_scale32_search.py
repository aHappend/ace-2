#!/usr/bin/env python3
"""Search exact grouped signed-A8 quantization at layer-17 Q output."""

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
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    scale32_ratio,
)


MISSION = "w4a8-layer17-q-output-source-grouped-repair-v1"
TARGET_LAYER = layer16.TARGET_LAYER
SOURCE_GROUP_SIZE = 1
Q_OUTPUT_GROUP_SIZES = (448, 128, 64)
HEAD_DIM = backend.HEAD_DIM
Q_PATTERN = re.compile(r"layer17_position(\d+)_q$")
SCORE_PATTERN = re.compile(
    r"layer17_position(?P<position>\d+)_head(?P<head>\d+)(?:_zero_cached_k)?$"
)
PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0002"
)
RTL = ROOT / "rtl/ace2_layer17_q_output_grouped_scale32_core.sv"
RTL_TOP = "ace2_layer17_q_output_grouped_scale32_core"


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _group_scales_from_accumulator(
    accumulator: torch.Tensor,
    input_scale32: int,
    weight_scale32: tuple[int, ...],
    group_size: int,
) -> tuple[int, ...]:
    input_num, input_den = scale32_ratio(input_scale32)
    scales = []
    for start in range(0, accumulator.numel(), group_size):
        maximum = Fraction(0, 1)
        for row in range(start, start + group_size):
            weight_num, weight_den = scale32_ratio(weight_scale32[row])
            value = Fraction(
                abs(int(accumulator[row])) * input_num * weight_num,
                input_den * weight_den,
            )
            maximum = max(maximum, value)
        localizer.require(maximum > 0, "zero Q-output group is unsupported")
        scales.append(
            ceil_scale32_from_ratio(
                maximum.numerator,
                maximum.denominator * 127,
            )
        )
    return tuple(scales)


class GroupedQOutputScale32Cache(layer16.ExactScale32QProjectionCache):
    """Replace only the layer-17 Q accumulator-to-A8 output quantizer."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        self.output_group_size = Q_OUTPUT_GROUP_SIZES[0]

    def configure(self, output_group_size: int) -> None:
        localizer.require(
            backend.HIDDEN % output_group_size == 0,
            "Q-output group does not divide hidden",
        )
        localizer.require(
            output_group_size % HEAD_DIM == 0,
            "Q-output group must preserve one scale per complete attention head",
        )
        self.output_group_size = output_group_size
        super().configure()

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
        exact_input_q = record["rms_output_s8"]
        input_scale32 = int(record["q_input_scale32"])
        qweight = self.q_metadata["qweight"]
        accumulator = torch.mv(qweight.to(torch.int64), exact_input_q.to(torch.int64))
        localizer.require(
            bool(torch.all(accumulator >= -(1 << 31)))
            and bool(torch.all(accumulator < (1 << 31))),
            f"{name} accumulator overflow",
        )

        group_scales = _group_scales_from_accumulator(
            accumulator,
            input_scale32,
            self.q_weight_scale32,
            self.output_group_size,
        )
        multiplier_values = []
        shift_values = []
        for row, weight_scale32 in enumerate(self.q_weight_scale32):
            multiplier, shift = layer16._derive_scale32_multiplier(
                input_scale32,
                weight_scale32,
                group_scales[row // self.output_group_size],
            )
            multiplier_values.append(multiplier)
            shift_values.append(shift)
        multiplier = torch.tensor(multiplier_values, dtype=torch.int64)
        right_shift = torch.tensor(shift_values, dtype=torch.int64)
        rounded = localizer.round_shift_even_tensor(accumulator * multiplier, right_shift)
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        group_scale_tensor = torch.tensor(group_scales, dtype=torch.int64)
        head_scales = torch.tensor(
            [group_scales[(head * HEAD_DIM) // self.output_group_size]
             for head in range(backend.Q_HEADS)],
            dtype=torch.int64,
        )

        binary_input_norm = record["binary_input_norm"].to(torch.float32)
        float_output = torch.mv(merged, binary_input_norm).contiguous()
        tensor_output_scale32 = ceil_scale32_from_float(
            float(backend.canonical.scale_for(float_output))
        )
        record.update(
            {
                "q_weight_s4": qweight,
                "q_weight_scale32": torch.tensor(
                    self.q_weight_scale32, dtype=torch.int64
                ),
                "q_accumulator_s32": accumulator,
                "q_output_group_scale32": group_scale_tensor,
                "q_head_scale32": head_scales,
                "q_multiplier_s32": multiplier,
                "q_shift_u6": right_shift,
                "q_rounded_s64": rounded,
                "q_output_s8": output_q,
                "q_saturation": saturation,
                "q_tensor_output_scale32": tensor_output_scale32,
                "q_output_group_size": self.output_group_size,
                "binary_q_float_output": float_output,
            }
        )
        return {
            "name": name,
            "input_q": exact_input_q,
            "input_scale": _scale32_float(input_scale32),
            "qweight": qweight,
            "weight_scale": self.q_metadata["weight_scale"],
            "multiplier": multiplier,
            "right_shift": right_shift,
            "accumulator": accumulator,
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": _scale32_float(int(head_scales[0])),
            "saturation": saturation,
            "float_output": float_output,
            "source_hashes": source_hashes,
            "exact_grouped_q_output_scale32": True,
        }


class QHeadScaleScoreGuard:
    """Consume the explicit grouped Q scale at the existing per-head score boundary."""

    def __init__(self, cache: GroupedQOutputScale32Cache) -> None:
        self.cache = cache
        self.original = backend.canonical.reference_attention_score
        self.replacements = 0

    def install(self) -> None:
        def score(case: Any) -> Any:
            match = SCORE_PATTERN.fullmatch(case.name)
            if match is None:
                return self.original(case)
            position = int(match.group("position"))
            head = int(match.group("head"))
            record = self.cache.position_records[position]
            query_scale32 = int(record["q_head_scale32"][head])
            replaced = backend.canonical.AttentionScoreCase(
                case.name,
                case.q_values,
                case.k_values,
                _scale32_float(query_scale32),
                case.key_scale,
            )
            self.replacements += 1
            return self.original(replaced)

        backend.canonical.reference_attention_score = score

    def restore(self) -> None:
        backend.canonical.reference_attention_score = self.original


def _artifacts(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors = {
        name: value for name, value in record.items() if isinstance(value, torch.Tensor)
    }
    for name in (
        "attention_base_scale32",
        "down_base_scale32",
        "sum_scale32",
        "q_input_scale32",
        "q_tensor_output_scale32",
        "rms_sumsq",
        "rms_inv_q30",
    ):
        tensors[name] = torch.tensor([int(record[name])], dtype=torch.int64)
    tensors["rms_saturation"] = torch.tensor(
        [int(record["rms_saturation"])], dtype=torch.uint8
    )
    return tensors


def run_group(
    output_group_size: int,
    cache: GroupedQOutputScale32Cache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure(output_group_size)
    source_guard = layer16.ExactScale32SourceGuard(
        cache, weights, SOURCE_GROUP_SIZE
    )
    score_guard = QHeadScaleScoreGuard(cache)
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
        "grouped Q scale did not reach every layer-17 attention score",
    )
    position = len(localizer.TOKEN_IDS) - 1
    exact = source_guard.records[position]
    scales = exact["q_output_group_scale32"]
    result.update(
        {
            "q_output_group_size": output_group_size,
            "q_output_group_count": backend.HIDDEN // output_group_size,
            "q_output_group_scale32_min": f"0x{int(scales.min()):08x}",
            "q_output_group_scale32_max": f"0x{int(scales.max()):08x}",
            "q_tensor_output_scale32": f"0x{int(exact['q_tensor_output_scale32']):08x}",
            "q_input_scale32": f"0x{int(exact['q_input_scale32']):08x}",
            "q_saturation_count": int(
                exact["q_saturation"].to(torch.int64).sum()
            ),
            "q_output_s8_sha256": localizer.tensor_sha256(exact["q_output_s8"]),
            "q_accumulator_s32_sha256": localizer.tensor_sha256(
                exact["q_accumulator_s32"]
            ),
            "layer17_score_scale_replacements": score_guard.replacements,
            "source_group_size": SOURCE_GROUP_SIZE,
        }
    )
    return result, _artifacts(exact)


def _rtl_preflight(output: Path) -> dict[str, Any]:
    log = output / "rtl-interface-preflight.log"
    with tempfile.TemporaryDirectory(prefix="ace2-layer17-q-output-", dir=ROOT / "build") as temporary:
        executable = Path(temporary) / "preflight.vvp"
        command = [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            RTL_TOP,
            "-o",
            str(executable),
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
    localizer.require(
        predecessor_result["selected_source_group_size"] == SOURCE_GROUP_SIZE,
        "predecessor no longer selects source group 1",
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
            "source_group_size": SOURCE_GROUP_SIZE,
            "top_token_id": baseline_top,
            "reference_rank": baseline_rank,
        },
        "ordered_q_output_group_sizes": list(Q_OUTPUT_GROUP_SIZES),
        "score_policy": (
            "first restored group in declared execution order; otherwise lowest "
            "reference rank strictly better than the frozen layer-16 group-1 baseline, "
            "ties by declared execution order"
        ),
        "numeric_contract": {
            "boundary": "model.layers.17.input_rmsnorm_to_self_attn.q_proj.output",
            "weights": "unchanged signed W4",
            "carried_q_samples": "signed A8",
            "group_scale": (
                "smallest legal Scale32 not below exact max(abs(accumulator * "
                "input_scale32 * row_weight_scale32))/127"
            ),
            "rounding": "signed round-to-nearest ties-to-even",
            "saturation": "signed int8 clamp",
            "downstream_scale_use": "one explicit Scale32 per complete Q-head group",
            "bf16_runtime_sidecar": False,
            "token_or_logit_override": False,
        },
        "rtl_public_contract": {
            "top": RTL_TOP,
            "latency": "one registered ready/valid stage",
            "reset": "asynchronous active-low reset; synchronous clear",
            "ports": [
                "clk_i", "rst_ni", "clear_i", "in_valid_i", "in_ready_o",
                "accumulator_i:s32", "multiplier_i:s32", "right_shift_i:u6",
                "group_scale32_i:u32", "channel_i:u10", "last_i",
                "out_valid_o", "out_ready_i", "q_o:s8",
                "group_scale32_o:u32", "channel_o:u10", "last_o",
                "saturation_o",
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
            "runner": localizer.file_record(Path(__file__).resolve()),
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
        cache = GroupedQOutputScale32Cache(weights, adapter)
        for group_size in Q_OUTPUT_GROUP_SIZES:
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
                "ACE2_LAYER17_Q_OUTPUT_GROUPED_SCALE32_CANDIDATE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} sat={record['q_saturation_count']}",
                flush=True,
            )

    restored = [item for item in trace if item["top_token_id"] == localizer.REFERENCE_TOKEN]
    improved = [item for item in trace if int(item["reference_rank"]) < baseline_rank]
    order = {group: index for index, group in enumerate(Q_OUTPUT_GROUP_SIZES)}
    if restored:
        selected = restored[0]
        selection_reason = "restored_step0_in_declared_execution_order"
    elif improved:
        selected = min(
            improved,
            key=lambda item: (
                int(item["reference_rank"]),
                order[int(item["q_output_group_size"])],
            ),
        )
        selection_reason = "best_honest_reference_rank_improvement"
    else:
        selected = None
        selection_reason = "no_group_improved_frozen_layer16_baseline"

    restored_step0 = selected is not None and selected["top_token_id"] == localizer.REFERENCE_TOKEN
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER17_Q_OUTPUT_GROUPED_SCALE32_STEP0_RESTORED"
            if restored_step0
            else "PARTIAL_LAYER17_Q_OUTPUT_GROUPED_SCALE32_RANK_IMPROVED"
            if selected is not None
            else "PARTIAL_LAYER17_Q_OUTPUT_GROUPED_SCALE32_NO_IMPROVEMENT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "baseline_reference_rank": baseline_rank,
        "baseline_top_token_id": baseline_top,
        "ordered_q_output_group_sizes": list(Q_OUTPUT_GROUP_SIZES),
        "trace": trace,
        "selected_q_output_group_size": (
            int(selected["q_output_group_size"]) if selected is not None else None
        ),
        "selected_reference_rank": (
            int(selected["reference_rank"]) if selected is not None else baseline_rank
        ),
        "selected_top_token_id": (
            int(selected["top_token_id"]) if selected is not None else baseline_top
        ),
        "selection_reason": selection_reason,
        "first_token_restored": bool(restored_step0),
        "next_exact_upstream_boundary": (
            None
            if restored_step0
            else "model.layers.17.input_rmsnorm.output_to_self_attn.q_proj.input"
        ),
        "next_required_action": (
            "Bind the selected grouped Q-output Scale32 payload to focused warning/X-clean RTL."
            if selected is not None
            else "Preserve all outcomes and advance to grouped layer-17 RMSNorm-output quantization."
        ),
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "rtl_preflight": localizer.file_record(
                output / "rtl-interface-preflight.json"
            ),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER17_Q_OUTPUT_GROUPED_SCALE32_RESULT "
        f"status={result['status']} selected={result['selected_q_output_group_size']} "
        f"rank={result['selected_reference_rank']} restored={result['first_token_restored']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER17_Q_OUTPUT_GROUPED_SCALE32_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
