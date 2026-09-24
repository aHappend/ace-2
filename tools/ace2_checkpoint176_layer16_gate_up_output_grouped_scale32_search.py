#!/usr/bin/env python3
"""Search grouped Scale32 gate/up output requantization at layer 16."""

from __future__ import annotations

import argparse
import hashlib
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
from tools import ace2_checkpoint176_layer16_post_attention_group1_remaining_error_localization as predecessor
from tools import ace2_checkpoint176_layer16_source_grouped_scale32_search as layer16
from tools import ace2_checkpoint176_layer17_q_output_grouped_scale32_search as qoutput
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import (
    SCALE32_ALL_ZERO_RECORD,
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    round_divide_even_unsigned,
    scale32_ratio,
)


MISSION = "w4a8-layer16-gate-up-output-grouped-a8-repair-v1"
BOUNDARY = "model.layers.16.mlp.gate_up_proj.output_to_signed_a8"
BASELINE_RANK = 3252
BASELINE_TOP_TOKEN = 12
POST_RMS_RANK = 3426
GATE_UP_DIAGNOSTIC_RANK = 2843
INTERMEDIATE = backend.INTERMEDIATE
POSITIONS = len(localizer.TOKEN_IDS)
CONTRACTS = (
    ("control", INTERMEDIATE),
    ("group-256", 256),
    ("group-128", 128),
    ("group-064", 64),
)
GATE_UP_PATTERN = re.compile(r"layer16_position(?P<position>\d+)_(?P<stream>gate|up)$")
SILU_PATTERN = re.compile(r"layer16_position(?P<position>\d+)_silu$")
PREDECESSOR_CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-post-attention-group1-remaining-error-"
    "localization-v1/candidate-0004"
)
PREDECESSOR_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-layer16-post-attention-group1-remaining-error-localization-v1/"
    "round-0001.json"
)
FRESH_REPRODUCTION = ROOT / (
    "build/l2-review-layer16-post-attention-group1-remaining-error-localization-r1/"
    "result.json"
)
RTL = ROOT / "rtl/ace2_layer16_gate_up_output_grouped_scale32_core.sv"
RTL_TOP = "ace2_layer16_gate_up_output_grouped_scale32_core"
PROTECTED_HASHES = dict(predecessor.PROTECTED_HASHES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(directory: Path) -> int:
    count = 0
    for line in (directory / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        localizer.require(
            _sha256(directory / relative) == expected,
            f"sealed predecessor manifest mismatch: {relative}",
        )
        count += 1
    localizer.require(count > 0, "sealed predecessor manifest is empty")
    return count


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _gate_predecessor() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    review = json.loads(PREDECESSOR_REVIEW.read_text(encoding="utf-8"))
    result_path = PREDECESSOR_CANDIDATE / "result.json"
    freeze_path = PREDECESSOR_CANDIDATE / "candidate-freeze.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    fresh = json.loads(FRESH_REPRODUCTION.read_text(encoding="utf-8"))
    localizer.require(
        review["producer_role"] == "reviewer"
        and review["review"]["status"] == "done",
        "Fresh-L2 predecessor decision is not accepted",
    )
    for observed in (result, fresh):
        localizer.require(
            int(observed["control"]["reference_rank"]) == BASELINE_RANK
            and int(observed["control"]["top_token_id"]) == BASELINE_TOP_TOKEN,
            "Fresh-L2 control rank/token differs",
        )
        trace = {item["diagnostic_cut"]: item for item in observed["trace"]}
        localizer.require(
            int(trace["post_attention_rmsnorm"]["reference_rank"]) == POST_RMS_RANK,
            "Fresh-L2 post-attention RMSNorm diagnostic differs",
        )
        localizer.require(
            int(trace["gate_up_projections"]["reference_rank"])
            == GATE_UP_DIAGNOSTIC_RANK
            and int(trace["gate_up_projections"]["top_token_id"])
            == BASELINE_TOP_TOKEN,
            "Fresh-L2 gate/up diagnostic differs",
        )
        localizer.require(
            observed["control_exact_repeat"]["passed"] is True,
            "Fresh-L2 exact control repeat is absent",
        )
    localizer.require(
        result["next_exact_repair_boundary"] == BOUNDARY
        and fresh["next_exact_repair_boundary"] == BOUNDARY,
        "Fresh-L2 repair boundary differs",
    )
    localizer.require(
        freeze["prompt"]["chat_token_ids"] == localizer.TOKEN_IDS,
        "canonical 30-token prefix differs",
    )
    members = _verify_manifest(PREDECESSOR_CANDIDATE)
    for relative, expected in PROTECTED_HASHES.items():
        localizer.require(_sha256(ROOT / relative) == expected, f"protected hash differs: {relative}")
    return result, freeze, review, members


def _derive_multiplier(
    input_scale: float, weight_scale: float, output_scale32: int
) -> tuple[int, int]:
    input_num, input_den = float(input_scale).as_integer_ratio()
    weight_num, weight_den = float(weight_scale).as_integer_ratio()
    output_num, output_den = scale32_ratio(output_scale32)
    numerator = input_num * weight_num * output_den
    denominator = input_den * weight_den * output_num
    for shift in range(63, -1, -1):
        multiplier = round_divide_even_unsigned(numerator << shift, denominator)
        if multiplier <= (1 << 31) - 1:
            return multiplier, shift
    raise OverflowError("gate/up output multiplier is not representable")


def _group_scales(
    accumulator: torch.Tensor,
    input_scale: float,
    weight_scale: torch.Tensor,
    group_size: int,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    input_num, input_den = float(input_scale).as_integer_ratio()
    records: list[int] = []
    zero_groups: list[bool] = []
    for start in range(0, accumulator.numel(), group_size):
        maximum = Fraction(0, 1)
        for row in range(start, start + group_size):
            weight_num, weight_den = float(weight_scale[row]).as_integer_ratio()
            maximum = max(
                maximum,
                Fraction(
                    abs(int(accumulator[row])) * input_num * weight_num,
                    input_den * weight_den,
                ),
            )
        if maximum == 0:
            records.append(SCALE32_ALL_ZERO_RECORD)
            zero_groups.append(True)
        else:
            records.append(
                ceil_scale32_from_ratio(maximum.numerator, maximum.denominator * 127)
            )
            zero_groups.append(False)
    return tuple(records), tuple(zero_groups)


def _expanded_scales(records: torch.Tensor, group_size: int) -> torch.Tensor:
    values = torch.tensor(
        [_scale32_float(int(value)) for value in records.tolist()], dtype=torch.float64
    )
    return values.repeat_interleave(group_size)


class GroupedGateUpCache(predecessor.BoundaryCache):
    """Change only layer-16 gate/up accumulator-to-signed-A8 quantization."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter, {})
        self.contract_id = "control"
        self.gate_up_output_group_size = INTERMEDIATE
        self.gate_up_records: dict[int, dict[str, dict[str, Any]]] = {}
        self.stream_metadata: dict[str, dict[str, Any]] = {}
        for stream in ("gate", "up"):
            name = f"model.layers.16.mlp.{stream}_proj"
            merged = self.merged[name]
            metadata = self.metadata[merged.data_ptr()]
            self.stream_metadata[stream] = metadata

    def configure_contract(self, contract_id: str, output_group_size: int) -> None:
        localizer.require(
            (contract_id, output_group_size) in CONTRACTS,
            "gate/up output contract is not frozen",
        )
        localizer.require(
            INTERMEDIATE % output_group_size == 0,
            "gate/up output group does not divide 4,864 channels",
        )
        self.contract_id = contract_id
        self.gate_up_output_group_size = output_group_size
        self.gate_up_records.clear()
        self.configure_boundary("control")

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
            name, merged, input_q, input_scale, float_input, source_hashes
        )
        match = GATE_UP_PATTERN.fullmatch(name)
        if match is None:
            return result

        position = int(match.group("position"))
        stream = match.group("stream")
        metadata = self.stream_metadata[stream]
        localizer.require(
            self.metadata[merged.data_ptr()] is metadata,
            f"layer-16 {stream} W4 metadata differs",
        )
        accumulator = result["accumulator"].to(torch.int64).contiguous()
        weight_scale = metadata["weight_scale"].to(torch.float64).contiguous()
        control_output_scale = float(result["output_scale"])

        if self.contract_id == "control":
            scale_records = (ceil_scale32_from_float(control_output_scale),)
            zero_groups = (False,)
            group_scales = torch.tensor(scale_records, dtype=torch.int64)
            multiplier = result["multiplier"].to(torch.int64).contiguous()
            right_shift = result["right_shift"].to(torch.int64).contiguous()
            rounded = result["rounded"].to(torch.int64).contiguous()
            output_q = result["output_q"].to(torch.int8).contiguous()
            saturation = result["saturation"].to(torch.uint8).contiguous()
        else:
            scale_records, zero_groups = _group_scales(
                accumulator, input_scale, weight_scale, self.gate_up_output_group_size
            )
            group_scales = torch.tensor(scale_records, dtype=torch.int64)
            multipliers: list[int] = []
            shifts: list[int] = []
            for row, row_weight_scale in enumerate(weight_scale.tolist()):
                group = row // self.gate_up_output_group_size
                if zero_groups[group]:
                    localizer.require(
                        int(accumulator[row]) == 0,
                        "zero gate/up group contains a nonzero accumulator",
                    )
                    multiplier_value, shift_value = 0, 0
                else:
                    multiplier_value, shift_value = _derive_multiplier(
                        input_scale,
                        float(row_weight_scale),
                        scale_records[group],
                    )
                multipliers.append(multiplier_value)
                shifts.append(shift_value)
            multiplier = torch.tensor(multipliers, dtype=torch.int64)
            right_shift = torch.tensor(shifts, dtype=torch.int64)
            rounded = localizer.round_shift_even_tensor(
                accumulator * multiplier, right_shift
            )
            output_q = rounded.clamp(-128, 127).to(torch.int8)
            saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)

        scale_vector = _expanded_scales(group_scales, self.gate_up_output_group_size)
        localizer.require(
            scale_vector.numel() == INTERMEDIATE,
            "expanded gate/up Scale32 metadata shape differs",
        )
        stream_record = {
            "input_s8": input_q.to(torch.int8).contiguous(),
            "input_scale_f64": torch.tensor([float(input_scale)], dtype=torch.float64),
            "weight_s4_sha256": localizer.tensor_sha256(metadata["qweight"]),
            "weight_scale_f64": weight_scale,
            "weight_scale_sha256": localizer.tensor_sha256(weight_scale),
            "accumulator_s32": accumulator,
            "output_group_scale32": group_scales,
            "multiplier_s32": multiplier,
            "shift_u6": right_shift,
            "rounded_s64": rounded,
            "output_s8": output_q,
            "saturation": saturation,
            "control_output_scale_f64": torch.tensor(
                [control_output_scale], dtype=torch.float64
            ),
            "output_group_size": self.gate_up_output_group_size,
        }
        self.gate_up_records.setdefault(position, {})[stream] = stream_record
        result.update(
            {
                "multiplier": multiplier,
                "right_shift": right_shift,
                "rounded": rounded,
                "output_q": output_q,
                "output_scale": scale_vector,
                "saturation": saturation,
                "output_group_scale32": group_scales,
                "output_group_size": self.gate_up_output_group_size,
                "output_scale_source": "explicit_scale32_metadata_only",
                "exact_grouped_gate_up_output_scale32": True,
            }
        )
        return result


class SiluScale32ConsumptionGuard:
    """Prove SiLU consumes gate/up A8 plus explicit Scale32 metadata."""

    def __init__(self, cache: GroupedGateUpCache) -> None:
        self.cache = cache
        self.original = backend.canonical.reference_silu_gate
        self.replacements = 0

    def install(self) -> None:
        def guarded(case: Any) -> Any:
            match = SILU_PATTERN.fullmatch(case.name)
            if match is not None:
                position = int(match.group("position"))
                record = self.cache.gate_up_records[position]
                for stream, observed in (
                    ("gate", case.gate_q6_9),
                    ("up", case.up_q6_9),
                ):
                    item = record[stream]
                    scale_vector = _expanded_scales(
                        item["output_group_scale32"], item["output_group_size"]
                    )
                    expected = (
                        torch.round(
                            item["output_s8"].to(torch.float64)
                            * scale_vector
                            * (1 << 9)
                        )
                        .clamp(-32768, 32767)
                        .to(torch.int16)
                        .to(torch.int64)
                        .tolist()
                    )
                    localizer.require(
                        list(observed) == expected,
                        f"SiLU did not consume {stream} A8 plus Scale32 metadata",
                    )
                self.replacements += 1
            return self.original(case)

        backend.canonical.reference_silu_gate = guarded

    def restore(self) -> None:
        backend.canonical.reference_silu_gate = self.original


def _bundle(cache: GroupedGateUpCache, source_records: dict[int, dict[str, Any]]) -> dict[str, torch.Tensor]:
    tensors = predecessor._bundle_records(source_records)
    for stream in ("gate", "up"):
        records = [cache.gate_up_records[position][stream] for position in range(POSITIONS)]
        for source, target in (
            ("input_s8", f"{stream}_input_s8"),
            ("input_scale_f64", f"{stream}_input_scale_f64"),
            ("accumulator_s32", f"{stream}_accumulator_s32"),
            ("output_group_scale32", f"{stream}_output_group_scale32"),
            ("multiplier_s32", f"{stream}_multiplier_s32"),
            ("shift_u6", f"{stream}_shift_u6"),
            ("rounded_s64", f"{stream}_rounded_s64"),
            ("output_s8", f"{stream}_output_s8"),
            ("saturation", f"{stream}_saturation"),
            ("control_output_scale_f64", f"{stream}_control_output_scale_f64"),
        ):
            tensors[target] = torch.stack([record[source] for record in records]).contiguous()
        tensors[f"{stream}_weight_scale_f64"] = records[0]["weight_scale_f64"]
    return tensors


def _run_contract(
    contract_id: str,
    output_group_size: int,
    cache: GroupedGateUpCache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure_contract(contract_id, output_group_size)
    cache.install()
    alignment = predecessor.postattn.PostAttentionAlignmentHook(
        predecessor.ATTENTION_OUTPUT_GROUP_SIZE
    )
    alignment.install()
    source_guard = predecessor.BoundarySourceGuard(
        cache, weights, alignment, {}, "control"
    )
    score_guard = qoutput.QHeadScaleScoreGuard(cache)
    silu_guard = SiluScale32ConsumptionGuard(cache)
    score_guard.install()
    silu_guard.install()
    source_guard.install()
    try:
        result, _ = localizer.run_cut(
            None, [], weights, adapter, norm_gain, embedding, head, tokenizer
        )
    finally:
        source_guard.restore()
        silu_guard.restore()
        score_guard.restore()
        alignment.restore()
        cache.restore()

    localizer.require(
        silu_guard.replacements == POSITIONS,
        "Scale32 gate/up streams did not reach every layer-16 SiLU",
    )
    localizer.require(
        score_guard.replacements == backend.Q_HEADS * (2 * POSITIONS - 1),
        "accepted layer-17 Q group scales did not reach every score",
    )
    final = cache.gate_up_records[POSITIONS - 1]
    bundle = _bundle(cache, source_guard.records)
    result.update(
        {
            "contract_id": contract_id,
            "gate_output_group_size": output_group_size,
            "up_output_group_size": output_group_size,
            "gate_output_group_count": INTERMEDIATE // output_group_size,
            "up_output_group_count": INTERMEDIATE // output_group_size,
            "gate_output_group_scale32_min": f"0x{int(final['gate']['output_group_scale32'].min()):08x}",
            "gate_output_group_scale32_max": f"0x{int(final['gate']['output_group_scale32'].max()):08x}",
            "up_output_group_scale32_min": f"0x{int(final['up']['output_group_scale32'].min()):08x}",
            "up_output_group_scale32_max": f"0x{int(final['up']['output_group_scale32'].max()):08x}",
            "gate_saturation_count": int(final["gate"]["saturation"].sum()),
            "up_saturation_count": int(final["up"]["saturation"].sum()),
            "gate_output_s8_sha256": localizer.tensor_sha256(final["gate"]["output_s8"]),
            "up_output_s8_sha256": localizer.tensor_sha256(final["up"]["output_s8"]),
            "gate_accumulator_s32_sha256": localizer.tensor_sha256(
                final["gate"]["accumulator_s32"]
            ),
            "up_accumulator_s32_sha256": localizer.tensor_sha256(
                final["up"]["accumulator_s32"]
            ),
            "gate_weight_s4_sha256": final["gate"]["weight_s4_sha256"],
            "up_weight_s4_sha256": final["up"]["weight_s4_sha256"],
            "gate_weight_scale_sha256": final["gate"]["weight_scale_sha256"],
            "up_weight_scale_sha256": final["up"]["weight_scale_sha256"],
            "silu_scale32_consumptions": silu_guard.replacements,
            "layer17_score_scale_replacements": score_guard.replacements,
            "reproducibility_signature": {
                name: localizer.tensor_sha256(value) for name, value in sorted(bundle.items())
            },
            "runtime_bf16_gate_up_sidecar": False,
            "downstream_gate_up_source": "signed_A8_plus_explicit_separate_Scale32_metadata",
        }
    )
    return result, bundle


def _write_contract(
    directory: Path, metrics: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> None:
    artifacts = {
        name: localizer.write_tensor(directory / f"{name}.bin", value)
        for name, value in sorted(tensors.items())
    }
    localizer.write_json(directory / "result.json", {"metrics": metrics, "artifacts": artifacts})


def _exact_repeat(first: dict[str, Any], repeat: dict[str, Any]) -> None:
    localizer.require(
        int(first["top_token_id"]) == int(repeat["top_token_id"])
        and int(first["reference_rank"]) == int(repeat["reference_rank"])
        and first["reproducibility_signature"] == repeat["reproducibility_signature"],
        f"gate/up contract did not reproduce exactly: {first['contract_id']}",
    )


def _rtl_preflight(output: Path) -> dict[str, Any]:
    log = output / "rtl-interface-preflight.log"
    with tempfile.TemporaryDirectory(prefix="ace2-layer16-gate-up-output-", dir=ROOT / "build") as temporary:
        executable = Path(temporary) / "preflight.vvp"
        command = [
            "iverilog",
            "-g2012",
            "-Wall",
            "-Wno-timescale",
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
    localizer.require(completed.returncode == 0, "frozen gate/up RTL interface did not compile")
    localizer.require(not completed.stdout.strip(), "gate/up RTL interface preflight emitted warnings")
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

    predecessor_result, predecessor_freeze, review, manifest_members = _gate_predecessor()
    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "mission": MISSION,
        "classification": "candidate_only_no_official_attempt_created",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": predecessor_freeze["checkpoint"],
        "prompt": predecessor_freeze["prompt"],
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "fresh_l2_gate": {
            "control_reference_rank": BASELINE_RANK,
            "control_top_token_id": BASELINE_TOP_TOKEN,
            "post_attention_rmsnorm_reference_rank": POST_RMS_RANK,
            "gate_up_projection_reference_rank": GATE_UP_DIAGNOSTIC_RANK,
            "gate_up_projection_top_token_id": BASELINE_TOP_TOKEN,
            "review_status": review["review"]["status"],
        },
        "fixed_contracts": {
            "layer16_post_attention_output_group_size": 1,
            "layer16_down_proj_output_group_size": 4,
            "layer17_q_output_group_size": 128,
            "w4_bytes_and_scales": "unchanged",
            "carried_activation": "signed A8",
            "intermediate_channels": INTERMEDIATE,
        },
        "ordered_gate_up_output_contracts": [
            {"contract_id": contract_id, "output_group_size": group_size}
            for contract_id, group_size in CONTRACTS
        ],
        "score_policy": (
            "run every frozen contract; select the first exact token-9707 restoration in "
            "execution order, otherwise the independently repeated lowest rank strictly "
            "better than 3,252, otherwise retain the exact repeated control as best-null"
        ),
        "numeric_contract": {
            "boundary": BOUNDARY,
            "control": (
                "unchanged accumulator/multiplier/shift/output A8 with one explicit "
                "Scale32 record per gate/up stream"
            ),
            "grouped_candidates": (
                "smallest Scale32 not below exact max(abs(accumulator * frozen input "
                "scale * frozen row W4 scale))/127 per output group"
            ),
            "gate_scale_metadata": "separate explicit Scale32 stream",
            "up_scale_metadata": "separate explicit Scale32 stream",
            "rounding": "signed round-to-nearest ties-to-even",
            "saturation": "signed int8 clamp",
            "silu_and_downstream_input": "signed A8 plus explicit Scale32 metadata only",
            "bf16_runtime_sidecar": False,
            "token_or_logit_override": False,
        },
        "rtl_public_contract": {
            "top": RTL_TOP,
            "latency": "one registered ready/valid stage",
            "reset": "asynchronous active-low reset; synchronous clear",
            "ports": [
                "clk_i", "rst_ni", "clear_i", "in_valid_i", "in_ready_o",
                "stream_i", "accumulator_i:s32", "multiplier_i:s32",
                "right_shift_i:u6", "group_scale32_i:u32", "channel_i:u13",
                "last_i", "out_valid_o", "out_ready_i", "stream_o", "q_o:s8",
                "group_scale32_o:u32", "channel_o:u13", "last_o", "saturation_o",
            ],
        },
        "protected_hashes": PROTECTED_HASHES,
        "bindings": {
            "predecessor_result": localizer.file_record(PREDECESSOR_CANDIDATE / "result.json"),
            "predecessor_freeze": localizer.file_record(PREDECESSOR_CANDIDATE / "candidate-freeze.json"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR_CANDIDATE / "SHA256SUMS"),
            "fresh_l2_reproduction": localizer.file_record(FRESH_REPRODUCTION),
            "fresh_l2_review": localizer.file_record(PREDECESSOR_REVIEW),
            "predecessor_manifest_members_verified": manifest_members,
            "runner": localizer.file_record(Path(__file__).resolve()),
            "rtl": localizer.file_record(RTL),
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "gpu_or_u280_used": False,
            "stage2_entered": False,
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)
    localizer.write_json(output / "rtl-interface-preflight.json", _rtl_preflight(output))

    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    started = time.monotonic()
    trace: list[dict[str, Any]] = []
    first_artifacts: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = GroupedGateUpCache(weights, adapter)

        for index, (contract_id, group_size) in enumerate(CONTRACTS):
            record, tensors = _run_contract(
                contract_id,
                group_size,
                cache,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            contract_dir = output / "contracts" / f"{index:02d}-{contract_id}"
            _write_contract(contract_dir, record, tensors)
            trace.append(record)
            first_artifacts[contract_id] = tensors
            print(
                "ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_CANDIDATE "
                f"contract={contract_id} group={group_size} "
                f"top={record['top_token_id']} rank={record['reference_rank']} "
                f"gate_sat={record['gate_saturation_count']} "
                f"up_sat={record['up_saturation_count']}",
                flush=True,
            )

        control = trace[0]
        localizer.require(
            int(control["reference_rank"]) == BASELINE_RANK
            and int(control["top_token_id"]) == BASELINE_TOP_TOKEN,
            "exact gate/up control did not reproduce rank 3,252/top token 12",
        )
        repeat_control, repeat_control_tensors = _run_contract(
            CONTRACTS[0][0],
            CONTRACTS[0][1],
            cache,
            weights,
            adapter,
            norm_gain,
            embedding,
            head,
            tokenizer,
        )
        _exact_repeat(control, repeat_control)
        _write_contract(output / "repeats" / "control", repeat_control, repeat_control_tensors)

        restored = [item for item in trace[1:] if int(item["top_token_id"]) == localizer.REFERENCE_TOKEN]
        improved = [item for item in trace[1:] if int(item["reference_rank"]) < BASELINE_RANK]
        order = {contract_id: index for index, (contract_id, _group) in enumerate(CONTRACTS)}
        if restored:
            provisional = min(restored, key=lambda item: order[item["contract_id"]])
            selection_reason = "restored_token_9707_in_declared_execution_order"
        elif improved:
            provisional = min(
                improved,
                key=lambda item: (
                    int(item["reference_rank"]),
                    order[item["contract_id"]],
                ),
            )
            selection_reason = "best_independently_reproduced_rank_improvement"
        else:
            provisional = control
            selection_reason = "best_null_exact_control_retained"

        selected_contract = str(provisional["contract_id"])
        selected_group = int(provisional["gate_output_group_size"])
        if selected_contract == "control":
            repeated = repeat_control
        else:
            repeated, repeated_tensors = _run_contract(
                selected_contract,
                selected_group,
                cache,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            _exact_repeat(provisional, repeated)
            _write_contract(
                output / "repeats" / f"selected-{selected_contract}",
                repeated,
                repeated_tensors,
            )

    selected = provisional
    restored_step0 = int(selected["top_token_id"]) == localizer.REFERENCE_TOKEN
    rank_improved = int(selected["reference_rank"]) < BASELINE_RANK
    negative = not restored_step0 and not rank_improved
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_TOKEN_9707_RESTORED"
            if restored_step0
            else "PARTIAL_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_RANK_IMPROVED"
            if rank_improved
            else "PARTIAL_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_BEST_NULL_CONTROL"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": BOUNDARY,
        "baseline_reference_rank": BASELINE_RANK,
        "baseline_top_token_id": BASELINE_TOP_TOKEN,
        "ordered_gate_up_output_contracts": [
            {"contract_id": contract_id, "output_group_size": group_size}
            for contract_id, group_size in CONTRACTS
        ],
        "trace": trace,
        "control_exact_repeat": {
            "passed": True,
            "reference_rank": int(repeat_control["reference_rank"]),
            "top_token_id": int(repeat_control["top_token_id"]),
            "reproducibility_signature": repeat_control["reproducibility_signature"],
        },
        "selected_contract_id": selected_contract,
        "selected_gate_output_group_size": selected_group,
        "selected_up_output_group_size": selected_group,
        "selected_reference_rank": int(selected["reference_rank"]),
        "selected_top_token_id": int(selected["top_token_id"]),
        "selected_exact_repeat_passed": True,
        "selection_reason": selection_reason,
        "first_token_restored": restored_step0,
        "rank_improved": rank_improved,
        "best_null_retained": negative,
        "next_exact_causal_boundary": (
            "model.layers.16.mlp.silu_product.output_to_signed_a8" if negative else None
        ),
        "prequeued_repair": (
            {
                "boundary": "model.layers.16.mlp.silu_product.output_to_signed_a8",
                "scope": "bounded signed-A8/Scale32 SiLU-product output requantization only",
                "status": "prequeued_not_executed",
            }
            if negative
            else None
        ),
        "runtime_bf16_gate_up_sidecar": False,
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "rtl_preflight": localizer.file_record(output / "rtl-interface-preflight.json"),
            "predecessor_result": localizer.file_record(PREDECESSOR_CANDIDATE / "result.json"),
            "fresh_l2_reproduction": localizer.file_record(FRESH_REPRODUCTION),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_RESULT "
        f"status={result['status']} selected={selected_contract} group={selected_group} "
        f"rank={result['selected_reference_rank']} top={result['selected_top_token_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
