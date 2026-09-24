#!/usr/bin/env python3
"""Search bounded Scale32 contracts at the layer-16 attention residual sum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
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
from tools import ace2_checkpoint176_layer16_down_proj_accumulator_grouped_scale32_search as accumulator
from tools import ace2_checkpoint176_layer16_down_proj_output_grouped_scale32_search as downoutput
from tools import ace2_checkpoint176_layer16_source_grouped_scale32_search as layer16
from tools import ace2_checkpoint176_layer17_q_output_grouped_scale32_search as qoutput
from tools import ace2_checkpoint176_layer17_rmsnorm_input_grouped_scale32_search as rmsinput
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import ceil_scale32_from_float, scale32_ratio
from tools.ace2_rmsnorm_reference import RmsNormResult, derive_scaled_gains_q8


MISSION = "w4a8-layer16-post-attention-residual-sum-scale-alignment-repair-v1"
BOUNDARY = "model.layers.16.self_attn.output_to_post_attention_residual_sum.scale_alignment"
NEXT_BOUNDARY = "model.layers.16.self_attn.o_proj.accumulator_to_signed_a8.scale_alignment"
ATTENTION_SOURCE_GROUP_SIZE = 1
DOWN_OUTPUT_GROUP_SIZE = 4
Q_OUTPUT_GROUP_SIZE = 128
OUTPUT_GROUP_SIZES = (896, 32, 16, 8, 4, 1)

PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer16-residual-sum-dual-scale-repair-v1/"
    "candidate-0006"
)
PREDECESSOR_FOCUSED = (
    ROOT
    / "evidence/verification/w4a8-layer16-residual-sum-dual-scale-repair-v1/"
    "candidate-0006-focused-rtl-r2"
)
PREDECESSOR_L2_SEARCH = ROOT / "build/l2-review-layer16-residual-sum-dual-scale-r1/result.json"
PREDECESSOR_L2_FOCUSED = ROOT / "build/l2-review-layer16-residual-sum-dual-scale-focused-r1/result.json"
PREDECESSOR_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-layer16-residual-sum-dual-scale-repair-v1/round-0001.json"
)
RTL = ROOT / "rtl/ace2_layer16_post_attention_residual_sum_scale_alignment_core.sv"
ARITHMETIC_RTL = ROOT / "rtl/ace2_layer16_residual_sum_dual_scale_core.sv"
RTL_TOP = "ace2_layer16_post_attention_residual_sum_scale_alignment_core"
PROTECTED_HASHES = dict(downoutput.PROTECTED_HASHES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _verify_manifest(directory: Path) -> int:
    completed = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=directory,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    localizer.require(completed.returncode == 0, f"manifest verification failed: {directory}")
    return len([line for line in completed.stdout.splitlines() if line.endswith(": OK")])


def _rtl_preflight() -> dict[str, Any]:
    results = {}
    with tempfile.TemporaryDirectory(prefix="ace2-layer16-post-attention-", dir=ROOT / "build") as temporary:
        executable = Path(temporary) / "preflight.vvp"
        commands = {
            "iverilog": [
                "iverilog", "-g2012", "-Wall", "-Wno-timescale", "-s", RTL_TOP,
                "-o", str(executable), str(ARITHMETIC_RTL), str(RTL),
            ],
            "verilator": [
                "verilator", "--lint-only", "-Wall", "-Wno-fatal",
                "--top-module", RTL_TOP, str(ARITHMETIC_RTL), str(RTL),
            ],
        }
        for name, command in commands.items():
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            localizer.require(completed.returncode == 0, f"{name} preflight failed: {completed.stdout}")
            localizer.require(not completed.stdout.strip(), f"{name} preflight warned: {completed.stdout}")
            results[name] = {"command": command, "returncode": completed.returncode}
    return {
        "schema_version": 1,
        "top": RTL_TOP,
        "rtl": localizer.file_record(RTL),
        "arithmetic_rtl": localizer.file_record(ARITHMETIC_RTL),
        "checks": results,
    }


def _tool_versions() -> dict[str, str]:
    commands = {
        "python": [sys.executable, "--version"],
        "iverilog": ["iverilog", "-V"],
        "verilator": ["verilator", "--version"],
    }
    versions = {"torch": torch.__version__}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        localizer.require(completed.returncode == 0, f"tool version failed: {name}")
        versions[name] = completed.stdout.strip().splitlines()[0]
    return versions


def _real_values(
    samples: torch.Tensor, scales: tuple[int, ...], group_size: int
) -> list[Fraction]:
    values = []
    for index, sample in enumerate(samples.to(torch.int64).tolist()):
        numerator, denominator = scale32_ratio(scales[index // group_size])
        values.append(Fraction(int(sample) * numerator, denominator))
    return values


class PostAttentionAlignmentHook:
    """Replace only the first layer-16 residual add and its RMSNorm consumption."""

    def __init__(self, output_group_size: int) -> None:
        self.output_group_size = output_group_size
        self.original_derive = backend.derive_layer_token
        self.original_projection = backend.canonical.derive_projection
        self.original_residual_add = backend.canonical.reference_residual_add
        self.original_rmsnorm = backend.canonical.reference_rmsnorm
        self.active: dict[str, Any] | None = None
        self.records: dict[int, dict[str, Any]] = {}

    def _projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        result = self.original_projection(
            name, merged, input_q, input_scale, float_input, source_hashes
        )
        if self.active is not None and name == self.active["o_name"]:
            self.active["attention_s8"] = result["output_q"].contiguous()
            self.active["attention_scale32"] = ceil_scale32_from_float(
                float(result["output_scale"])
            )
        return result

    def _residual_add(self, lhs: list[int], rhs: list[int]) -> tuple[list[int], bool]:
        active = self.active
        if active is None or active["residual_add_seen"]:
            return self.original_residual_add(lhs, rhs)
        active["residual_add_seen"] = True
        if self.output_group_size == backend.HIDDEN:
            return self.original_residual_add(lhs, rhs)

        residual_s8 = active["residual_s8"]
        attention_s8 = active.get("attention_s8")
        localizer.require(attention_s8 is not None, "layer-16 attention output was not captured")
        residual_num, residual_den = scale32_ratio(active["residual_scale32"])
        attention_num, attention_den = scale32_ratio(active["attention_scale32"])
        values = [
            Fraction(int(residual) * residual_num, residual_den)
            + Fraction(int(attention) * attention_num, attention_den)
            for residual, attention in zip(
                residual_s8.to(torch.int64).tolist(),
                attention_s8.to(torch.int64).tolist(),
                strict=True,
            )
        ]
        output_scales = rmsinput._group_scales(values, self.output_group_size)
        output_s8, saturation = rmsinput._quantize_groups(
            values, output_scales, self.output_group_size
        )
        active.update(
            {
                "values": values,
                "output_scales": output_scales,
                "output_s8": output_s8,
                "saturation": saturation,
                "residual_replaced": True,
            }
        )
        return output_s8.to(torch.int64).tolist(), bool(saturation.any())

    def _rmsnorm(self, activations: list[int], gains: list[int]) -> RmsNormResult:
        active = self.active
        if (
            active is None
            or not active.get("residual_replaced", False)
            or active.get("post_rmsnorm_replaced", False)
        ):
            return self.original_rmsnorm(activations, gains)
        grouped = rmsinput._grouped_rmsnorm(
            active["output_s8"],
            active["output_scales"],
            self.output_group_size,
            gains,
        )
        active["post_rmsnorm_replaced"] = True
        active["post_rmsnorm"] = grouped
        return RmsNormResult(
            grouped["outputs"].to(torch.int64).tolist(),
            int(grouped["sumsq"]),
            int(grouped["inv_rms_q30"]),
            bool(grouped["saturation"].any()),
        )

    def _derive(
        self,
        layer_id: int,
        state: dict[str, Any],
        cache: dict[str, Any],
        template: dict[str, Any] | None,
        weights: Any,
        adapter: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if layer_id != layer16.SOURCE_LAYER:
            return self.original_derive(layer_id, state, cache, template, weights, adapter)
        position = int(state["position"])
        localizer.require(self.active is None, "nested layer-16 alignment context")
        self.active = {
            "position": position,
            "o_name": f"layer{layer_id}_position{position}_o",
            "residual_s8": state["fixed_q"].contiguous(),
            "residual_scale32": ceil_scale32_from_float(float(state["fixed_scale"])),
            "residual_add_seen": False,
            "residual_replaced": False,
            "post_rmsnorm_replaced": False,
        }
        try:
            derived, next_state, next_template = self.original_derive(
                layer_id, state, cache, template, weights, adapter
            )
            active = self.active
            localizer.require(active.get("attention_s8") is not None, "attention operand missing")
            position_record = derived["positions"][0]
            if self.output_group_size == backend.HIDDEN:
                output_scale = ceil_scale32_from_float(
                    float(position_record["attention_residual"]["scale"])
                )
                active["output_scales"] = (output_scale,)
                active["output_s8"] = position_record["attention_residual"]["output"].contiguous()
                active["saturation"] = torch.full(
                    (backend.HIDDEN,),
                    int(position_record["attention_residual"]["saturation"]),
                    dtype=torch.uint8,
                )
            else:
                position_record["attention_residual"].update(
                    {
                        "lhs": active["residual_s8"],
                        "rhs": active["attention_s8"],
                        "output": active["output_s8"],
                        "saturation": bool(active["saturation"].any()),
                        "residual_scale32": active["residual_scale32"],
                        "attention_scale32": active["attention_scale32"],
                        "output_group_size": self.output_group_size,
                        "output_group_scale32": torch.tensor(
                            active["output_scales"], dtype=torch.int64
                        ),
                    }
                )
            active["post_attention_output_group_size"] = self.output_group_size
            active["post_attention_output_group_scale32"] = torch.tensor(
                active["output_scales"], dtype=torch.int64
            )
            active["post_attention_sum_s8"] = active["output_s8"]
            active["post_attention_sum_saturation"] = active["saturation"]
            active["residual_scale32_expanded"] = torch.full(
                (backend.HIDDEN,), active["residual_scale32"], dtype=torch.int64
            )
            active["attention_scale32_expanded"] = torch.full(
                (backend.HIDDEN,), active["attention_scale32"], dtype=torch.int64
            )
            active["post_attention_rmsnorm_output_scale32"] = ceil_scale32_from_float(
                float(position_record["post_norm_scale"])
            )
            self.records[position] = dict(active)
            return derived, next_state, next_template
        finally:
            self.active = None

    def install(self) -> None:
        backend.canonical.derive_projection = self._projection
        backend.canonical.reference_residual_add = self._residual_add
        backend.canonical.reference_rmsnorm = self._rmsnorm
        backend.derive_layer_token = self._derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original_derive
        backend.canonical.reference_rmsnorm = self.original_rmsnorm
        backend.canonical.reference_residual_add = self.original_residual_add
        backend.canonical.derive_projection = self.original_projection


class PostAttentionSourceGuard(accumulator.DownProjAccumulatorSourceGuard):
    """Feed the repaired post-attention residual into accepted group-1/group-4/Q-128."""

    def __init__(
        self,
        cache: accumulator.GroupedDownProjAccumulatorScale32Cache,
        weights: Any,
        alignment: PostAttentionAlignmentHook,
    ) -> None:
        super().__init__(cache, weights, DOWN_OUTPUT_GROUP_SIZE)
        self.alignment = alignment

    def _replace_record(self, position: int) -> None:
        record = self.records[position]
        boundary = self.alignment.records[position]
        boundary_scales = tuple(
            int(value)
            for value in boundary["post_attention_output_group_scale32"].tolist()
        )
        boundary_values = _real_values(
            boundary["post_attention_sum_s8"],
            boundary_scales,
            self.alignment.output_group_size,
        )
        attention_scales = rmsinput._group_scales(boundary_values, ATTENTION_SOURCE_GROUP_SIZE)
        attention_group_s8, attention_saturation = rmsinput._quantize_groups(
            boundary_values, attention_scales, ATTENTION_SOURCE_GROUP_SIZE
        )
        down_group_s8 = record["down_output_s8"]
        down_scales = tuple(
            int(value) for value in record["down_output_group_scale32"].tolist()
        )

        binary_attention = accumulator._group_dequantize(
            attention_group_s8, attention_scales, ATTENTION_SOURCE_GROUP_SIZE
        )
        binary_down = accumulator._group_dequantize(
            down_group_s8, down_scales, DOWN_OUTPUT_GROUP_SIZE
        )
        binary_sum = (binary_attention + binary_down).contiguous()
        gain = self.weights.get_tensor(
            f"model.layers.{layer16.TARGET_LAYER}.input_layernorm.weight"
        ).contiguous()
        binary_input_norm = backend.canonical.float_rmsnorm(
            binary_sum.to(torch.float32), gain
        ).contiguous()
        binary_q_input_scale = max(
            float(binary_input_norm.to(torch.float64).abs().max()) / 127.0,
            1.0e-12,
        )
        q_input_scale32 = ceil_scale32_from_float(binary_q_input_scale)
        final_values = downoutput._sum_reals(
            attention_group_s8,
            down_group_s8,
            attention_scales,
            down_scales,
            DOWN_OUTPUT_GROUP_SIZE,
        )
        sum_scale32 = downoutput._tensor_scale(final_values)
        sum_s8, sum_saturation = downoutput._quantize_values(final_values, sum_scale32)
        gains = derive_scaled_gains_q8(
            gain.to(torch.float64).tolist(), _scale32_float(q_input_scale32)
        )
        rms = backend.canonical.reference_rmsnorm(
            sum_s8.to(torch.int64).tolist(), gains
        )

        record.update(boundary)
        record.update(
            {
                "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
                "attention_source_s8": boundary["post_attention_sum_s8"],
                "attention_base_scale32": int(boundary_scales[0]),
                "attention_group_scale32": torch.tensor(
                    attention_scales, dtype=torch.int64
                ),
                "attention_group_s8": attention_group_s8,
                "attention_saturation": attention_saturation,
                "down_group_s8": down_group_s8,
                "down_group_scale32": record["down_output_group_scale32"],
                "sum_scale32": sum_scale32,
                "sum_s8": sum_s8,
                "sum_saturation": sum_saturation,
                "rms_gain_s16_q8": torch.tensor(gains, dtype=torch.int16),
                "rms_output_s8": torch.tensor(rms.outputs, dtype=torch.int8),
                "rms_sumsq": rms.sumsq,
                "rms_inv_q30": rms.inv_rms_q30,
                "rms_saturation": rms.saturation_seen,
                "q_input_scale32": q_input_scale32,
                "binary_attention_q": attention_group_s8,
                "binary_down_q": down_group_s8,
                "binary_attention_scales": torch.tensor(
                    [_scale32_float(value) for value in attention_scales],
                    dtype=torch.float32,
                ),
                "binary_down_scales": torch.tensor(
                    [_scale32_float(value) for value in down_scales],
                    dtype=torch.float32,
                ),
                "binary_sum_reconstructed": binary_sum,
                "binary_input_norm": binary_input_norm,
                "binary_q_input_scale": binary_q_input_scale,
            }
        )

    def install(self) -> None:
        super().install()
        derive_with_down = backend.derive_layer_token

        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            result = derive_with_down(layer_id, state, cache, template, weights, adapter)
            if layer_id == layer16.SOURCE_LAYER:
                self._replace_record(int(state["position"]))
            return result

        backend.derive_layer_token = derive


def _artifacts(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    artifacts = {
        name: value for name, value in record.items() if isinstance(value, torch.Tensor)
    }
    scalar_names = (
        "residual_scale32",
        "attention_scale32",
        "post_attention_rmsnorm_output_scale32",
        "down_input_scale32",
        "down_tensor_output_scale32",
        "sum_scale32",
        "q_input_scale32",
        "q_tensor_output_scale32",
        "rms_sumsq",
        "rms_inv_q30",
    )
    for name in scalar_names:
        if name in record:
            artifacts[name] = torch.tensor([int(record[name])], dtype=torch.int64)
    artifacts["rms_saturation"] = torch.tensor(
        [int(record["rms_saturation"])], dtype=torch.uint8
    )
    return artifacts


def run_contract(
    output_group_size: int,
    cache: accumulator.GroupedDownProjAccumulatorScale32Cache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure_down(DOWN_OUTPUT_GROUP_SIZE)
    cache.install()
    alignment = PostAttentionAlignmentHook(output_group_size)
    alignment.install()
    source_guard = (
        accumulator.DownProjAccumulatorSourceGuard(
            cache, weights, DOWN_OUTPUT_GROUP_SIZE
        )
        if output_group_size == backend.HIDDEN
        else PostAttentionSourceGuard(cache, weights, alignment)
    )
    score_guard = qoutput.QHeadScaleScoreGuard(cache)
    score_guard.install()
    source_guard.install()
    try:
        result, _ = localizer.run_cut(
            None, [], weights, adapter, norm_gain, embedding, head, tokenizer
        )
    finally:
        source_guard.restore()
        score_guard.restore()
        alignment.restore()
        cache.restore()

    expected_scores = backend.Q_HEADS * (2 * len(localizer.TOKEN_IDS) - 1)
    localizer.require(
        score_guard.replacements == expected_scores,
        "accepted grouped Q scale did not reach every layer-17 score",
    )
    exact = source_guard.records[len(localizer.TOKEN_IDS) - 1]
    exact.update(alignment.records[len(localizer.TOKEN_IDS) - 1])
    output_scales = exact["post_attention_output_group_scale32"]
    q_scales = exact["q_output_group_scale32"]
    result.update(
        {
            "contract": "control" if output_group_size == backend.HIDDEN else f"group-{output_group_size}",
            "control_contract": output_group_size == backend.HIDDEN,
            "output_group_size": output_group_size,
            "output_group_count": backend.HIDDEN // output_group_size,
            "output_group_scale32_min": f"0x{int(output_scales.min()):08x}",
            "output_group_scale32_max": f"0x{int(output_scales.max()):08x}",
            "residual_s8_sha256": localizer.tensor_sha256(exact["residual_s8"]),
            "residual_scale32": f"0x{int(exact['residual_scale32']):08x}",
            "attention_s8_sha256": localizer.tensor_sha256(exact["attention_s8"]),
            "attention_scale32": f"0x{int(exact['attention_scale32']):08x}",
            "post_attention_sum_s8_sha256": localizer.tensor_sha256(
                exact["post_attention_sum_s8"]
            ),
            "post_attention_sum_saturation_count": int(
                exact["post_attention_sum_saturation"].sum()
            ),
            "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
            "down_output_group_size": DOWN_OUTPUT_GROUP_SIZE,
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "down_s8_sha256": localizer.tensor_sha256(exact["down_output_s8"]),
            "down_scale32_sha256": localizer.tensor_sha256(
                exact["down_output_group_scale32"]
            ),
            "q_output_group_scale32_min": f"0x{int(q_scales.min()):08x}",
            "q_output_group_scale32_max": f"0x{int(q_scales.max()):08x}",
            "q_output_s8_sha256": localizer.tensor_sha256(exact["q_output_s8"]),
            "q_saturation_count": int(exact["q_saturation"].sum()),
            "layer17_score_scale_replacements": score_guard.replacements,
        }
    )
    return result, _artifacts(exact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "candidate output already exists")

    for relative, expected in PROTECTED_HASHES.items():
        localizer.require(_sha256(ROOT / relative) == expected, f"protected hash differs: {relative}")

    predecessor_result = json.loads((PREDECESSOR / "result.json").read_text(encoding="utf-8"))
    predecessor_focused = json.loads((PREDECESSOR_FOCUSED / "result.json").read_text(encoding="utf-8"))
    l2_search = json.loads(PREDECESSOR_L2_SEARCH.read_text(encoding="utf-8"))
    l2_focused = json.loads(PREDECESSOR_L2_FOCUSED.read_text(encoding="utf-8"))
    handoff = json.loads(PREDECESSOR_HANDOFF.read_text(encoding="utf-8"))
    predecessor_members = _verify_manifest(PREDECESSOR)
    focused_members = _verify_manifest(PREDECESSOR_FOCUSED)
    for record, label in ((predecessor_result, "candidate-0006"), (l2_search, "Fresh-L2 search")):
        localizer.require(
            record["control"]["reference_rank"] == 31879
            and record["control"]["top_token_id"] == 37865
            and record["best_non_control_output_group_size"] == 32
            and record["best_non_control_reference_rank"] == 34409,
            f"{label} predecessor outcome differs",
        )
        localizer.require(
            [item["sum_output_group_size"] for item in record["failures"]] == [8, 4, 1],
            f"{label} smaller-group failures differ",
        )
    for record, label in ((predecessor_focused, "producer focused"), (l2_focused, "Fresh-L2 focused")):
        localizer.require(
            record["metrics"]["outputs"] == 896
            and record["checks"]["iverilog_warning_clean"]
            and record["checks"]["verilator_warning_clean"]
            and record["checks"]["xz_clean_every_checked_cycle"],
            f"{label} predecessor RTL evidence differs",
        )
    localizer.require(
        handoff["producer_role"] == "reviewer"
        and handoff["review"]["status"] == "done"
        and "31,879" in handoff["review"]["reason"]
        and "34,409" in handoff["review"]["reason"]
        and "896 vectors" in handoff["review"]["reason"],
        "sealed Fresh-L2 predecessor acceptance is absent",
    )

    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mission": MISSION,
        "classification": "candidate_only_no_official_attempt_created",
        "checkpoint": json.loads((PREDECESSOR / "candidate-freeze.json").read_text())["checkpoint"],
        "prompt": json.loads((PREDECESSOR / "candidate-freeze.json").read_text())["prompt"],
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "accepted_control": {
            "top_token_id": 37865,
            "reference_rank": 31879,
            "attention_source_group_size": 1,
            "down_output_group_size": 4,
            "q_output_group_size": 128,
        },
        "ordered_output_group_sizes": list(OUTPUT_GROUP_SIZES),
        "score_policy": (
            "execute every frozen contract; control must exactly reproduce rank 31879/top 37865; "
            "select the first token-9707 restoration in order, otherwise the lowest non-control "
            "rank strictly below 31879, ties by declared order"
        ),
        "numeric_contract": {
            "boundary": BOUNDARY,
            "residual_operand": "layer-16 input residual signed A8 plus explicit Scale32",
            "attention_operand": "layer-16 self-attention o_proj output signed A8 plus independent explicit Scale32",
            "alignment": "multiply both A8 operands by their Scale32 significands, align to one bounded common exponent, and add once",
            "requantization": "one exact integer divide by output Scale32, ties-to-even, then signed-A8 saturation",
            "downstream_paths": "accepted attention source group-1, down_proj group-4, and layer-17 Q-output group-128 retained",
            "accepted_w4_bytes_and_scales_unchanged": True,
            "bf16_runtime_sidecar": False,
            "wider_carried_activation_payload": False,
            "token_or_logit_override": False,
        },
        "rtl_public_contract": {
            "top": RTL_TOP,
            "parameters": [],
            "latency": "ten arithmetic cycles for valid descriptors; output holds under backpressure",
            "reset": "asynchronous active-low reset; synchronous clear",
            "ports": [
                "clk_i", "rst_ni", "clear_i", "in_valid_i", "in_ready_o",
                "residual_s8_i:s8", "residual_scale32_i:u32", "attention_s8_i:s8",
                "attention_scale32_i:u32", "output_scale32_i:u32", "channel_i:u10", "last_i",
                "out_valid_o", "out_ready_i", "sum_s8_o:s8", "output_scale32_o:u32",
                "channel_o:u10", "last_o", "saturation_o", "descriptor_error_o",
                "numeric_overflow_o", "common_exponent_s8_o:s8", "latency_cycles_u5_o:u5",
            ],
        },
        "tool_versions": _tool_versions(),
        "protected_hashes": PROTECTED_HASHES,
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "u280_or_stage2_entered": False,
        },
        "bindings": {
            "predecessor_result": localizer.file_record(PREDECESSOR / "result.json"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "predecessor_members_verified": predecessor_members,
            "predecessor_focused_result": localizer.file_record(PREDECESSOR_FOCUSED / "result.json"),
            "predecessor_focused_sha256s": localizer.file_record(PREDECESSOR_FOCUSED / "SHA256SUMS"),
            "predecessor_focused_members_verified": focused_members,
            "fresh_l2_search": localizer.file_record(PREDECESSOR_L2_SEARCH),
            "fresh_l2_focused": localizer.file_record(PREDECESSOR_L2_FOCUSED),
            "fresh_l2_handoff": _external_record(PREDECESSOR_HANDOFF),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "rtl": localizer.file_record(RTL),
            "arithmetic_rtl": localizer.file_record(ARITHMETIC_RTL),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)
    localizer.write_json(output / "rtl-interface-preflight.json", _rtl_preflight())

    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    localizer.require(
        generation_runner.canonical_chat_token_ids(tokenizer, "Hello") == localizer.TOKEN_IDS,
        "canonical tokenizer prefix differs",
    )

    trace = []
    failures = []
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = accumulator.GroupedDownProjAccumulatorScale32Cache(weights, adapter)
        for group_size in OUTPUT_GROUP_SIZES:
            group_dir = output / f"contracts/group-{group_size:03d}"
            try:
                record, tensors = run_contract(
                    group_size, cache, weights, adapter, norm_gain, embedding, head, tokenizer
                )
                if group_size == backend.HIDDEN:
                    localizer.require(
                        record["top_token_id"] == 37865 and record["reference_rank"] == 31879,
                        "exact accepted control outcome differs",
                    )
            except Exception as error:
                failure = {
                    "output_group_size": group_size,
                    "classification": (
                        "CONTROL_REPRODUCTION_FAILURE"
                        if group_size == backend.HIDDEN
                        else "CANDIDATE_EXECUTION_FAILURE"
                    ),
                    "detail": str(error),
                    "root_cause_hypothesis": "The frozen post-attention output-group contract could not be consumed by the unchanged downstream evaluator.",
                    "regression": "Preserve this failure and continue the frozen order without changing evaluator behavior or expected values.",
                }
                localizer.write_json(group_dir / "failure.json", failure)
                failures.append(failure)
                print(
                    f"ACE2_LAYER16_POST_ATTENTION_ALIGNMENT_CANDIDATE_FAIL group={group_size} detail={error}",
                    flush=True,
                )
                if group_size == backend.HIDDEN:
                    raise
                continue
            artifacts = {
                name: localizer.write_tensor(group_dir / f"{name}.bin", value)
                for name, value in sorted(tensors.items())
            }
            localizer.write_json(group_dir / "result.json", {"metrics": record, "artifacts": artifacts})
            trace.append(record)
            print(
                f"ACE2_LAYER16_POST_ATTENTION_ALIGNMENT_CANDIDATE group={group_size} "
                f"top={record['top_token_id']} rank={record['reference_rank']} "
                f"sat={record['post_attention_sum_saturation_count']}",
                flush=True,
            )

    localizer.require(trace and trace[0]["control_contract"], "exact control did not complete first")
    control = trace[0]
    non_control = [record for record in trace if not record["control_contract"]]
    localizer.require(non_control, "no repair contract completed")
    order = {group: index for index, group in enumerate(OUTPUT_GROUP_SIZES)}
    restored = [record for record in non_control if record["top_token_id"] == localizer.REFERENCE_TOKEN]
    improved = [record for record in non_control if int(record["reference_rank"]) < 31879]
    best_non_control = min(
        non_control,
        key=lambda record: (
            int(record["reference_rank"]), order[int(record["output_group_size"])]
        ),
    )
    if restored:
        selected = restored[0]
        selection_reason = "restored_token_9707_in_declared_execution_order"
    elif improved:
        selected = min(
            improved,
            key=lambda record: (
                int(record["reference_rank"]), order[int(record["output_group_size"])]
            ),
        )
        selection_reason = "best_independent_rank_improvement_over_31879"
    else:
        selected = None
        selection_reason = "no_non_control_contract_improved_rank_31879_or_restored_token_9707"

    validation = selected if selected is not None else best_non_control
    restored_token = selected is not None and selected["top_token_id"] == localizer.REFERENCE_TOKEN
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER16_POST_ATTENTION_ALIGNMENT_TOKEN_9707_RESTORED"
            if restored_token
            else "PARTIAL_LAYER16_POST_ATTENTION_ALIGNMENT_RANK_IMPROVED"
            if selected is not None
            else "PARTIAL_LAYER16_POST_ATTENTION_ALIGNMENT_NO_IMPROVEMENT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": BOUNDARY,
        "baseline_reference_rank": 31879,
        "baseline_top_token_id": 37865,
        "ordered_output_group_sizes": list(OUTPUT_GROUP_SIZES),
        "control": control,
        "trace": trace,
        "failures": failures,
        "selected_output_group_size": int(selected["output_group_size"]) if selected else None,
        "selected_reference_rank": int(selected["reference_rank"]) if selected else 31879,
        "selected_top_token_id": int(selected["top_token_id"]) if selected else 37865,
        "selection_reason": selection_reason,
        "best_non_control_output_group_size": int(best_non_control["output_group_size"]),
        "best_non_control_reference_rank": int(best_non_control["reference_rank"]),
        "best_non_control_top_token_id": int(best_non_control["top_token_id"]),
        "rtl_validation_output_group_size": int(validation["output_group_size"]),
        "first_token_restored": bool(restored_token),
        "next_exact_upstream_scale_alignment_boundary": None if selected else NEXT_BOUNDARY,
        "prequeue_required": selected is None,
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "rtl_preflight": localizer.file_record(output / "rtl-interface-preflight.json"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "predecessor_focused_sha256s": localizer.file_record(PREDECESSOR_FOCUSED / "SHA256SUMS"),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        f"ACE2_LAYER16_POST_ATTENTION_ALIGNMENT_RESULT status={result['status']} "
        f"selected={result['selected_output_group_size']} "
        f"best_non_control={result['best_non_control_output_group_size']} "
        f"rank={result['best_non_control_reference_rank']} restored={result['first_token_restored']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER16_POST_ATTENTION_ALIGNMENT_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
