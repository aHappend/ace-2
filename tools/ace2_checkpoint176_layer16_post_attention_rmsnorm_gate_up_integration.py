#!/usr/bin/env python3
"""Rank frozen layer-16 RMSNorm-output Scale32 contracts through accepted gate/up W4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
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
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_checkpoint176_layer16_gate_up_output_grouped_scale32_search as gateup
from tools import ace2_checkpoint176_layer16_post_attention_group1_remaining_error_localization as predecessor
from tools import ace2_checkpoint176_layer17_q_output_grouped_scale32_search as qoutput
from tools import gen_layer16_post_attention_rmsnorm_runtime_metadata_vectors as runtimegen
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    pack_scale32,
    scale32_ratio,
    unpack_scale32,
)


MISSION = "w4a8-layer16-post-attention-rmsnorm-output-signed-a8-repair-v1"
BOUNDARY = "model.layers.16.post_attention_layernorm.output_to_signed_a8"
BASELINE_RANK = 3252
BASELINE_TOP_TOKEN = 12
POSITIONS = len(localizer.TOKEN_IDS)
HIDDEN = backend.HIDDEN
INTERMEDIATE = backend.INTERMEDIATE
CONTRACTS = (
    ("control", None),
    ("group-896", 896),
    ("group-128", 128),
    ("group-064", 64),
)
GATE_UP_PATTERN = re.compile(
    r"layer16_position(?P<position>\d+)_(?P<stream>gate|up)$"
)
SOURCE_CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
SOURCE_CONTROL = SOURCE_CANDIDATE / "contracts/00-control"
RUNTIME_R3 = ROOT / (
    "verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata"
)
RUNTIME_R3_RESULT = ROOT / (
    "build/layer16-post-attention-rmsnorm-runtime-metadata-focused-r3/result.json"
)
FRESH_RUNTIME_R3_RESULT = ROOT / (
    "build/l2-review-layer16-post-attention-rmsnorm-runtime-metadata-focused-r1/"
    "result.json"
)
PREQUEUE = ROOT / (
    "evidence/prequeue/w4a8-layer16-post-attention-rmsnorm-output-signed-a8-"
    "repair-v1/runtime-metadata-core-integration-prequeue.json"
)
ACCEPTED_W4 = ROOT / (
    "evidence/verification/lora-v4-two-token-24layer-rtl-v1/attempt-0003/"
    "layers/layer-16/tensors"
)
GATE_W4 = ACCEPTED_W4 / "shared_gate_qweight_s4_in_s8.bin"
UP_W4 = ACCEPTED_W4 / "shared_up_qweight_s4_in_s8.bin"

EXPECTED_BINDINGS = {
    SOURCE_CANDIDATE / "result.json": (
        "f734ae5e80bafed36fd434d65d1caec1794d6fa28ec6756c88092227726e6a18"
    ),
    RUNTIME_R3_RESULT: (
        "90e7f61ff6d69d273a9c1b0194e16e063ba4e9e358d4e62d146947d80ca27068"
    ),
    FRESH_RUNTIME_R3_RESULT: (
        "ecf7141f62bebd808891b012c8f443fb4977ebd863f22baa474bf40171426dde"
    ),
    PREQUEUE: (
        "49c52abb96a9544f9d6c590de4adf6599a03e637bd4f593ec7978f321380c359"
    ),
    GATE_W4: (
        "62b84b7c76932d35478e26925196eee04ca2e7776b43550801f60a90ad115953"
    ),
    UP_W4: (
        "0b8ee972b123324d28d2055bcf1b39720a5f97dc822e3a2789d9752599d11fa3"
    ),
}

ACCEPTED_CHAIN_HASHES = {
    "post_attention_sum_s8": (
        "4f129de929030ee6a0f877092b90fb5f8d8a2265554c31793ef76acc8cde9317"
    ),
    "post_attention_output_group_scale32": (
        "5de7b5d6f26fb6690202e15f90a23808d4633346299671ec7d319ee9d98f9caf"
    ),
    "down_input_s8": (
        "b840d32315a14a89e9e2dd7ae721b4f37181be64c48890a86501f9c087937823"
    ),
    "down_input_scale32": (
        "afbc3e83b7561e1fd30d9aba45ce7f31bfdde4d22b75b7ae2f048418b658d2d6"
    ),
    "down_weight_s4": (
        "58abe01a09f61573707b9af98b105da07ad0056eb05a8cac5343452ab15e40bb"
    ),
    "down_weight_scale32": (
        "6bef8f087de7842aa664168e6c3beefe172778500c658fb5ecedf4174f0d8f2f"
    ),
    "down_output_group_scale32": (
        "1307f94cb033f0f47b054a839a8cc8cb2a06e5161f9e969610299bc38e120223"
    ),
    "q_input_scale32": (
        "6b4dbde878f11aa1227b541bb913959a4d263aba895940450555f7e71ef2e96f"
    ),
    "q_output_group_scale32": (
        "039d2a587cc824ea3f7bd6c943105b66a9df2d9d011b74337ef7cfec8af068ca"
    ),
    "q_output_s8": (
        "adc31d483a7c7c6dc26903219e337ec550267d62145409c84adeb066d4ebb671"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_exact(path: Path, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    localizer.require(result.size == expected, f"{path.name} element count differs")
    return result.reshape(shape)


def read_hex(path: Path, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray([int(value, 16) for value in path.read_text().split()], dtype=dtype)
    expected = int(np.prod(shape))
    localizer.require(raw.size == expected, f"{path.name} hex element count differs")
    return raw.reshape(shape)


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _verify_sources() -> dict[str, Any]:
    for path, expected in EXPECTED_BINDINGS.items():
        localizer.require(sha256(path) == expected, f"immutable binding differs: {path}")
    candidate_check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=SOURCE_CANDIDATE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    localizer.require(candidate_check.returncode == 0, candidate_check.stdout)
    candidate = json.loads((SOURCE_CANDIDATE / "result.json").read_text())
    localizer.require(
        int(candidate["selected_reference_rank"]) == BASELINE_RANK
        and int(candidate["selected_top_token_id"]) == BASELINE_TOP_TOKEN
        and candidate["control_exact_repeat"]["passed"] is True,
        "preserved null control differs",
    )
    manifest = json.loads((RUNTIME_R3 / "manifest.json").read_text())
    localizer.require(
        manifest["executed_contract"] == "group-128-power2-scale32"
        and int(manifest["observed"]["nonzero_outputs"]) == 25463
        and int(manifest["observed"]["saturation_count"]) == 0,
        "focused runtime r3 manifest differs",
    )
    return {
        "candidate_manifest_check": candidate_check.stdout.rstrip(),
        "candidate": candidate,
        "runtime_manifest": manifest,
    }


def _runtime_tensors(
    weights: Any,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    activation = read_exact(
        SOURCE_CONTROL / "post_attention_sum_s8.bin", "i1", (POSITIONS, HIDDEN)
    )
    input_scale_i64 = read_exact(
        SOURCE_CONTROL / "post_attention_output_group_scale32.bin",
        "<i8",
        (POSITIONS, HIDDEN),
    )
    localizer.require(
        bool(np.all((input_scale_i64 >= 0) & (input_scale_i64 <= 0xFFFFFFFF))),
        "accepted input Scale32 escaped unsigned storage",
    )
    input_scale = input_scale_i64.astype(np.uint32)
    gain_f32 = (
        weights.get_tensor("model.layers.16.post_attention_layernorm.weight")
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    gain_integers, gain_scales = runtimegen.encode_gain_metadata(
        gain_f32.astype(np.float64).tolist()
    )
    common = {
        "input_activation_s8": torch.from_numpy(activation.copy()).to(torch.int8),
        "input_scale32": torch.from_numpy(input_scale.astype(np.int64)),
        "gain_s16": torch.tensor(gain_integers, dtype=torch.int16),
        "gain_scale32": torch.tensor(gain_scales, dtype=torch.int64),
    }
    contracts: dict[str, dict[str, torch.Tensor]] = {}
    original_group_size = runtimegen.GROUP_SIZE
    try:
        for contract_id, group_size in CONTRACTS[1:]:
            assert group_size is not None
            runtimegen.GROUP_SIZE = group_size
            groups = HIDDEN // group_size
            aligned = np.empty((POSITIONS, HIDDEN), dtype=np.int16)
            qgain = np.empty((POSITIONS, HIDDEN), dtype=np.int16)
            output = np.empty((POSITIONS, HIDDEN), dtype=np.int8)
            saturation = np.empty((POSITIONS, HIDDEN), dtype=np.uint8)
            output_scale = np.empty((POSITIONS, groups), dtype=np.uint32)
            common_exponent = np.empty(POSITIONS, dtype=np.int8)
            rebase_shift = np.empty(POSITIONS, dtype=np.uint8)
            sumsq = np.empty(POSITIONS, dtype=np.uint64)
            mean_square = np.empty(POSITIONS, dtype=np.uint64)
            rms_ceil = np.empty(POSITIONS, dtype=np.uint16)
            inv_rms = np.empty(POSITIONS, dtype=np.uint32)
            for position in range(POSITIONS):
                result = runtimegen.runtime_reference(
                    activation[position].astype(int).tolist(),
                    input_scale[position].astype(int).tolist(),
                    gain_integers,
                    gain_scales,
                )
                aligned[position] = np.asarray(result.aligned, dtype=np.int16)
                qgain[position] = np.asarray(result.qgains, dtype=np.int16)
                output[position] = np.asarray(result.outputs, dtype=np.int8)
                saturation[position] = np.asarray(result.saturation, dtype=np.uint8)
                output_scale[position] = np.asarray(result.output_scales, dtype=np.uint32)
                common_exponent[position] = result.common_exponent
                rebase_shift[position] = result.rebase_shift
                sumsq[position] = result.sumsq
                mean_square[position] = result.mean_square
                rms_ceil[position] = result.rms_ceil
                inv_rms[position] = result.inv_rms_q30
            localizer.require(np.count_nonzero(output) > 0, f"{contract_id} remains zero")
            localizer.require(int(saturation.sum()) == 0, f"{contract_id} saturated")
            contracts[contract_id] = {
                "aligned_s12": torch.from_numpy(aligned.copy()),
                "qgain_s16_q8": torch.from_numpy(qgain.copy()),
                "output_s8": torch.from_numpy(output.copy()),
                "output_scale32": torch.from_numpy(output_scale.astype(np.int64)),
                "saturation": torch.from_numpy(saturation.copy()),
                "common_exponent_s8": torch.from_numpy(common_exponent.copy()),
                "rebase_shift_u6": torch.from_numpy(rebase_shift.copy()),
                "sumsq_u48": torch.from_numpy(sumsq.astype(np.int64)),
                "mean_square_u48": torch.from_numpy(mean_square.astype(np.int64)),
                "rms_ceil_u12": torch.from_numpy(rms_ceil.copy()),
                "inv_rms_q30": torch.from_numpy(inv_rms.astype(np.int64)),
            }
    finally:
        runtimegen.GROUP_SIZE = original_group_size

    r3 = contracts["group-128"]
    expected_output = read_hex(
        RUNTIME_R3 / "expected_s8.hex", np.uint8, (POSITIONS, HIDDEN)
    ).view(np.int8)
    expected_scale = read_hex(
        RUNTIME_R3 / "expected_output_scale32.hex", np.uint32, (POSITIONS, 7)
    )
    localizer.require(
        np.array_equal(r3["output_s8"].numpy(), expected_output)
        and np.array_equal(r3["output_scale32"].numpy(), expected_scale.astype(np.int64)),
        "group-128 runtime payload differs from focused r3",
    )
    return contracts, common


def _write_rtl_vectors(
    directory: Path,
    contract_id: str,
    group_size: int,
    runtime: dict[str, torch.Tensor],
    common: dict[str, torch.Tensor],
) -> None:
    directory.mkdir(parents=True)
    vectors: dict[str, tuple[torch.Tensor, int]] = {
        "input_activation_s8.hex": (common["input_activation_s8"], 8),
        "input_scale32.hex": (common["input_scale32"], 32),
        "gain_s16.hex": (common["gain_s16"], 16),
        "gain_scale32.hex": (common["gain_scale32"], 32),
        "expected_aligned_s12.hex": (runtime["aligned_s12"], 12),
        "expected_qgain_s16_q8.hex": (runtime["qgain_s16_q8"], 16),
        "expected_output_scale32.hex": (runtime["output_scale32"], 32),
        "expected_s8.hex": (runtime["output_s8"], 8),
        "expected_saturation.hex": (runtime["saturation"], 8),
        "expected_common_exponent_s8.hex": (runtime["common_exponent_s8"], 8),
        "expected_rebase_shift_u6.hex": (runtime["rebase_shift_u6"], 8),
        "expected_sumsq_u48.hex": (runtime["sumsq_u48"], 48),
        "expected_mean_square_u48.hex": (runtime["mean_square_u48"], 48),
        "expected_rms_ceil_u12.hex": (runtime["rms_ceil_u12"], 12),
        "expected_inv_rms_q30.hex": (runtime["inv_rms_q30"], 31),
    }
    for name, (tensor, width) in vectors.items():
        runtimegen.write_hex(directory / name, tensor.reshape(-1).tolist(), width)
    nonzero = int(torch.count_nonzero(runtime["output_s8"]))
    groups = HIDDEN // group_size
    (directory / "ace2_layer16_post_attention_rmsnorm_runtime_metadata_constants.svh").write_text(
        "\n".join(
            (
                f"localparam integer MODEL_POSITIONS = {POSITIONS};",
                f"localparam integer MODEL_HIDDEN = {HIDDEN};",
                f"localparam integer MODEL_SAMPLES = {POSITIONS * HIDDEN};",
                f"localparam integer MODEL_OUTPUT_GROUP_SIZE = {group_size};",
                f"localparam integer MODEL_GROUPS_PER_POSITION = {groups};",
                f"localparam integer MODEL_SCALE_RECORDS = {POSITIONS * groups};",
                f"localparam integer MODEL_NONZERO_OUTPUTS = {nonzero};",
                "localparam integer MODEL_SATURATIONS = 0;",
                "",
            )
        ),
        encoding="ascii",
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_id": contract_id,
                "output_group_size": group_size,
                "positions": POSITIONS,
                "samples": POSITIONS * HIDDEN,
                "nonzero_outputs": nonzero,
                "saturations": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    members = sorted(path for path in directory.iterdir() if path.is_file())
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in members
            if path.name != "SHA256SUMS"
        ),
        encoding="ascii",
    )


class RmsnormInputCache(gateup.GroupedGateUpCache):
    """Change only layer-16 gate/up inputs while retaining gate/up output control."""

    def __init__(
        self,
        weights: Any,
        adapter: Any,
        runtime_contracts: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        super().__init__(weights, adapter)
        self.runtime_contracts = runtime_contracts
        self.rms_contract_id = "control"
        self.rms_group_size: int | None = None

    def configure_rms_contract(self, contract_id: str, group_size: int | None) -> None:
        localizer.require((contract_id, group_size) in CONTRACTS, "RMS contract is not frozen")
        gateup.GroupedGateUpCache.configure_contract(self, "control", INTERMEDIATE)
        self.rms_contract_id = contract_id
        self.rms_group_size = group_size

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
        record = self.gate_up_records[position][stream]
        if self.rms_contract_id == "control":
            scale32 = ceil_scale32_from_float(float(input_scale))
            record.update(
                {
                    "input_group_scale32": torch.tensor([scale32], dtype=torch.int64),
                    "input_common_scale32": torch.tensor([scale32], dtype=torch.int64),
                    "input_shift_u6": torch.zeros(1, dtype=torch.int64),
                    "aligned_input_s16": input_q.to(torch.int16).contiguous(),
                    "input_group_size": HIDDEN,
                }
            )
            return result

        assert self.rms_group_size is not None
        runtime = self.runtime_contracts[self.rms_contract_id]
        repaired_q = runtime["output_s8"][position].to(torch.int8).contiguous()
        group_scale32 = runtime["output_scale32"][position].to(torch.int64).contiguous()
        exponents: list[int] = []
        for value in group_scale32.tolist():
            significand, exponent = unpack_scale32(int(value))
            localizer.require(significand == 0x8000, "runtime output scale is not power-of-two")
            exponents.append(exponent)
        common_exponent = min(exponents)
        shifts = torch.tensor(
            [exponent - common_exponent for exponent in exponents], dtype=torch.int64
        )
        aligned = repaired_q.to(torch.int64).clone()
        for group, shift in enumerate(shifts.tolist()):
            start = group * self.rms_group_size
            stop = start + self.rms_group_size
            aligned[start:stop] <<= int(shift)
        localizer.require(
            int(aligned.abs().max()) <= 2048,
            "gate/up aligned RMSNorm input escaped bounded signed-12 magnitude",
        )
        metadata = self.stream_metadata[stream]
        qweight = metadata["qweight"].to(torch.int64)
        accumulator = torch.mv(qweight, aligned)
        localizer.require(
            bool(torch.all(accumulator >= -(1 << 31)))
            and bool(torch.all(accumulator < (1 << 31))),
            f"{stream} accepted-W4 accumulator overflow",
        )
        common_scale32 = pack_scale32(0x8000, common_exponent)
        common_scale = _scale32_float(common_scale32)
        control_output_scale = float(record["control_output_scale_f64"][0])
        multiplier, right_shift = backend.canonical.derive_multiplier(
            common_scale * metadata["weight_scale"] / control_output_scale
        )
        multiplier = multiplier.to(torch.int64).contiguous()
        right_shift = right_shift.to(torch.int64).contiguous()
        rounded = localizer.round_shift_even_tensor(
            accumulator * multiplier, right_shift
        )
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        record.update(
            {
                "input_s8": repaired_q,
                "input_scale_f64": torch.tensor([common_scale], dtype=torch.float64),
                "input_group_scale32": group_scale32,
                "input_common_scale32": torch.tensor([common_scale32], dtype=torch.int64),
                "input_shift_u6": shifts,
                "aligned_input_s16": aligned.to(torch.int16),
                "input_group_size": self.rms_group_size,
                "accumulator_s32": accumulator,
                "multiplier_s32": multiplier,
                "shift_u6": right_shift,
                "rounded_s64": rounded,
                "output_s8": output_q,
                "saturation": saturation,
            }
        )
        result.update(
            {
                "input_q": repaired_q,
                "input_scale": common_scale,
                "accumulator": accumulator,
                "multiplier": multiplier,
                "right_shift": right_shift,
                "rounded": rounded,
                "output_q": output_q,
                "saturation": saturation,
                "rms_input_group_scale32": group_scale32,
                "rms_input_common_scale32": common_scale32,
                "rms_input_group_size": self.rms_group_size,
                "exact_power2_group_alignment": True,
            }
        )
        return result


def _bundle(
    cache: RmsnormInputCache,
    source_records: dict[int, dict[str, Any]],
    runtime: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    tensors = gateup._bundle(cache, source_records)
    for stream in ("gate", "up"):
        records = [cache.gate_up_records[position][stream] for position in range(POSITIONS)]
        for source, target in (
            ("input_group_scale32", f"{stream}_input_group_scale32"),
            ("input_common_scale32", f"{stream}_input_common_scale32"),
            ("input_shift_u6", f"{stream}_input_shift_u6"),
            ("aligned_input_s16", f"{stream}_aligned_input_s16"),
        ):
            tensors[target] = torch.stack([record[source] for record in records]).contiguous()
    if runtime is not None:
        tensors["rms_output_s8"] = runtime["output_s8"].contiguous()
        tensors["rms_output_group_scale32"] = runtime["output_scale32"].contiguous()
        tensors["rms_aligned_s12"] = runtime["aligned_s12"].contiguous()
        tensors["rms_qgain_s16_q8"] = runtime["qgain_s16_q8"].contiguous()
        tensors["rms_saturation"] = runtime["saturation"].contiguous()
        tensors["rms_common_exponent_s8"] = runtime["common_exponent_s8"].contiguous()
        tensors["rms_rebase_shift_u6"] = runtime["rebase_shift_u6"].contiguous()
        tensors["rms_sumsq_u48"] = runtime["sumsq_u48"].contiguous()
        tensors["rms_mean_square_u48"] = runtime["mean_square_u48"].contiguous()
        tensors["rms_ceil_u12"] = runtime["rms_ceil_u12"].contiguous()
        tensors["rms_inv_q30"] = runtime["inv_rms_q30"].contiguous()
    return tensors


def _run_contract(
    contract_id: str,
    group_size: int | None,
    cache: RmsnormInputCache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure_rms_contract(contract_id, group_size)
    cache.install()
    alignment = predecessor.postattn.PostAttentionAlignmentHook(
        predecessor.ATTENTION_OUTPUT_GROUP_SIZE
    )
    alignment.install()
    source_guard = predecessor.BoundarySourceGuard(
        cache, weights, alignment, {}, "control"
    )
    score_guard = qoutput.QHeadScaleScoreGuard(cache)
    silu_guard = gateup.SiluScale32ConsumptionGuard(cache)
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
        "gate/up Scale32 outputs did not reach every layer-16 SiLU",
    )
    localizer.require(
        score_guard.replacements == backend.Q_HEADS * (2 * POSITIONS - 1),
        "accepted layer-17 group-128 Q scales did not reach every score",
    )
    runtime = None if contract_id == "control" else cache.runtime_contracts[contract_id]
    bundle = _bundle(cache, source_guard.records, runtime)
    final = {stream: cache.gate_up_records[POSITIONS - 1][stream] for stream in ("gate", "up")}
    signature = {
        name: localizer.tensor_sha256(value) for name, value in sorted(bundle.items())
    }
    chain = {
        name: signature[name]
        for name in ACCEPTED_CHAIN_HASHES
        if name in signature
    }
    result.update(
        {
            "contract_id": contract_id,
            "output_group_size": group_size,
            "runtime_nonzero_outputs": (
                0 if runtime is None else int(torch.count_nonzero(runtime["output_s8"]))
            ),
            "runtime_saturation_count": (
                0 if runtime is None else int(runtime["saturation"].sum())
            ),
            "gate_input_nonzero_count": int(torch.count_nonzero(bundle["gate_input_s8"])),
            "up_input_nonzero_count": int(torch.count_nonzero(bundle["up_input_s8"])),
            "gate_accumulator_nonzero_count": int(
                torch.count_nonzero(bundle["gate_accumulator_s32"])
            ),
            "up_accumulator_nonzero_count": int(
                torch.count_nonzero(bundle["up_accumulator_s32"])
            ),
            "gate_output_nonzero_count": int(torch.count_nonzero(bundle["gate_output_s8"])),
            "up_output_nonzero_count": int(torch.count_nonzero(bundle["up_output_s8"])),
            "gate_accumulator_min": int(bundle["gate_accumulator_s32"].min()),
            "gate_accumulator_max": int(bundle["gate_accumulator_s32"].max()),
            "up_accumulator_min": int(bundle["up_accumulator_s32"].min()),
            "up_accumulator_max": int(bundle["up_accumulator_s32"].max()),
            "gate_weight_s4_sha256": final["gate"]["weight_s4_sha256"],
            "up_weight_s4_sha256": final["up"]["weight_s4_sha256"],
            "gate_weight_scale_sha256": final["gate"]["weight_scale_sha256"],
            "up_weight_scale_sha256": final["up"]["weight_scale_sha256"],
            "accepted_chain_observed_hashes": chain,
            "accepted_chain_hash_matches": {
                name: chain.get(name) == expected
                for name, expected in ACCEPTED_CHAIN_HASHES.items()
            },
            "silu_scale32_consumptions": silu_guard.replacements,
            "layer17_score_scale_replacements": score_guard.replacements,
            "reproducibility_signature": signature,
            "runtime_bf16_rmsnorm_sidecar": False,
            "accepted_w4_payload_consumed": True,
            "gate_up_accumulator_alignment": (
                "sum(q_s8[channel] * w_s4[row,channel] * "
                "2^(group_exponent-min_group_exponent))"
            ),
        }
    )
    if contract_id != "control":
        localizer.require(
            result["gate_input_nonzero_count"] > 0
            and result["up_input_nonzero_count"] > 0
            and result["gate_accumulator_nonzero_count"] > 0
            and result["up_accumulator_nonzero_count"] > 0,
            f"{contract_id} lacks nonzero accepted-W4 causality",
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
        f"contract did not reproduce exactly: {first['contract_id']}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "output already exists")

    sources = _verify_sources()
    output.mkdir(parents=True)
    interface = {
        "schema_version": 1,
        "mission": MISSION,
        "boundary": BOUNDARY,
        "channel_order": "0_through_895_then_last",
        "positions": POSITIONS,
        "hidden_channels": HIDDEN,
        "ordered_contracts": [
            {"contract_id": contract_id, "output_group_size": group_size}
            for contract_id, group_size in CONTRACTS
        ],
        "runtime_core": {
            "top": "ace2_layer16_post_attention_rmsnorm_runtime_metadata_core",
            "parameters": {
                "HIDDEN_SIZE": 896,
                "MAX_REBASED_MAGNITUDE": 2047,
                "OUTPUT_GROUP_SIZE": [896, 128, 64],
            },
            "protocol": "ready/valid channel frame with explicit position/channel/last",
            "input": "signed-A8 plus per-channel Scale32 and immutable gain integer/Scale32",
            "output": "signed-A8 plus one power-of-two Scale32 per selected output group",
        },
        "gate_up_handoff": {
            "w4_payload": "accepted immutable signed W4 bytes and row scales",
            "alignment": (
                "choose minimum live power-of-two input exponent per frame; left-shift each "
                "signed-A8 lane by its group exponent delta before the signed-W4 dot product"
            ),
            "accumulator": "signed 32-bit exact integer accumulator for each of 4,864 rows",
            "gate_and_up": "separate accumulators over identical signed-A8/Scale32 inputs",
        },
        "scope_guards": {
            "runtime_sidecars": False,
            "w4_mutation": False,
            "silu_contract_change": False,
            "official_run": False,
            "u280_or_stage2": False,
        },
    }
    localizer.write_json(output / "interface-contract.json", interface)

    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    started = time.monotonic()
    trace: list[dict[str, Any]] = []
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        runtime_contracts, runtime_common = _runtime_tensors(weights)
        for contract_id, group_size in CONTRACTS[1:]:
            assert group_size is not None
            _write_rtl_vectors(
                output / "rtl-vectors" / contract_id,
                contract_id,
                group_size,
                runtime_contracts[contract_id],
                runtime_common,
            )
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = RmsnormInputCache(weights, adapter, runtime_contracts)

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
            _write_contract(output / "contracts" / f"{index:02d}-{contract_id}", record, tensors)
            trace.append(record)
            print(
                "ACE2_LAYER16_POST_ATTENTION_RMSNORM_GATE_UP_CONTRACT "
                f"contract={contract_id} group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} gate_input_nz={record['gate_input_nonzero_count']} "
                f"gate_acc_nz={record['gate_accumulator_nonzero_count']} "
                f"up_acc_nz={record['up_accumulator_nonzero_count']}",
                flush=True,
            )

        control = trace[0]
        localizer.require(
            int(control["reference_rank"]) == BASELINE_RANK
            and int(control["top_token_id"]) == BASELINE_TOP_TOKEN,
            "exact rank-3252/top-token-12 control did not reproduce first",
        )
        repeat_control, repeat_tensors = _run_contract(
            "control",
            None,
            cache,
            weights,
            adapter,
            norm_gain,
            embedding,
            head,
            tokenizer,
        )
        _exact_repeat(control, repeat_control)
        _write_contract(output / "repeats/control", repeat_control, repeat_tensors)

    restored = [
        item for item in trace[1:] if int(item["top_token_id"]) == localizer.REFERENCE_TOKEN
    ]
    improved = [item for item in trace[1:] if int(item["reference_rank"]) < BASELINE_RANK]
    order = {contract_id: index for index, (contract_id, _group) in enumerate(CONTRACTS)}
    if restored:
        selected = min(restored, key=lambda item: order[item["contract_id"]])
        selection_reason = "restored_token_9707_in_frozen_execution_order"
    elif improved:
        selected = min(
            improved,
            key=lambda item: (int(item["reference_rank"]), order[item["contract_id"]]),
        )
        selection_reason = "lowest_strict_rank_improvement"
    else:
        selected = control
        selection_reason = "exact_control_retained_as_best_null"
    runtime_best = min(
        trace[1:],
        key=lambda item: (int(item["reference_rank"]), order[item["contract_id"]]),
    )
    selected_top = int(selected["top_token_id"])
    selected_rank = int(selected["reference_rank"])
    token_restored = selected_top == localizer.REFERENCE_TOKEN
    rank_improved = selected_rank < BASELINE_RANK
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": (
            "PASS_LAYER16_POST_ATTENTION_RMSNORM_TOKEN_9707_RESTORED"
            if token_restored
            else "PARTIAL_LAYER16_POST_ATTENTION_RMSNORM_RANK_IMPROVED"
            if rank_improved
            else "PARTIAL_LAYER16_POST_ATTENTION_RMSNORM_ALL_RUNTIME_CONTRACTS_NEGATIVE"
        ),
        "classification": "bounded_candidate_only_no_official_attempt",
        "mission": MISSION,
        "boundary": BOUNDARY,
        "baseline_reference_rank": BASELINE_RANK,
        "baseline_top_token_id": BASELINE_TOP_TOKEN,
        "ordered_contracts": [
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
        "selected_contract_id": selected["contract_id"],
        "selected_output_group_size": selected["output_group_size"],
        "selected_reference_rank": selected_rank,
        "selected_top_token_id": selected_top,
        "selection_reason": selection_reason,
        "first_token_restored": token_restored,
        "rank_improved": rank_improved,
        "best_null_retained": not token_restored and not rank_improved,
        "focused_rtl_contract_id": runtime_best["contract_id"],
        "focused_rtl_output_group_size": runtime_best["output_group_size"],
        "focused_rtl_reference_rank": int(runtime_best["reference_rank"]),
        "focused_rtl_top_token_id": int(runtime_best["top_token_id"]),
        "group_128_reference_rank": int(
            next(item for item in trace if item["contract_id"] == "group-128")[
                "reference_rank"
            ]
        ),
        "group_128_top_token_id": int(
            next(item for item in trace if item["contract_id"] == "group-128")[
                "top_token_id"
            ]
        ),
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "interface_contract": file_record(output / "interface-contract.json"),
            "source_candidate_result": file_record(SOURCE_CANDIDATE / "result.json"),
            "runtime_r3_result": file_record(RUNTIME_R3_RESULT),
            "fresh_runtime_r3_result": file_record(FRESH_RUNTIME_R3_RESULT),
            "sealed_prequeue": file_record(PREQUEUE),
            "gate_w4": file_record(GATE_W4),
            "up_w4": file_record(UP_W4),
            "runner": file_record(Path(__file__).resolve()),
            "candidate_manifest_check": sources["candidate_manifest_check"],
        },
        "checks": {
            "control_rank_3252_top_12_first": True,
            "every_frozen_contract_executed_in_order": True,
            "group_128_payload_matches_focused_r3": True,
            "all_runtime_contracts_nonzero_inputs": all(
                int(item["gate_input_nonzero_count"]) > 0
                and int(item["up_input_nonzero_count"]) > 0
                for item in trace[1:]
            ),
            "all_runtime_contracts_nonzero_accumulators": all(
                int(item["gate_accumulator_nonzero_count"]) > 0
                and int(item["up_accumulator_nonzero_count"]) > 0
                for item in trace[1:]
            ),
            "all_accumulators_s32": all(
                int(item["gate_accumulator_min"]) >= -(1 << 31)
                and int(item["gate_accumulator_max"]) < (1 << 31)
                and int(item["up_accumulator_min"]) >= -(1 << 31)
                and int(item["up_accumulator_max"]) < (1 << 31)
                for item in trace[1:]
            ),
            "accepted_gate_w4_hash_unchanged": True,
            "accepted_up_w4_hash_unchanged": True,
            "runtime_bf16_sidecar_absent": True,
            "official_run_absent": True,
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "gate_up_w4_mutated": False,
            "runtime_bf16_sidecar": False,
            "runtime_precomputed_metadata_sidecar": False,
            "silu_contract_changed": False,
            "carried_activation_wider_than_a8": False,
            "u280_or_stage2_entered": False,
            "independent_fresh_l2_required": True,
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER16_POST_ATTENTION_RMSNORM_GATE_UP_RESULT "
        f"status={result['status']} selected={result['selected_contract_id']} "
        f"rank={selected_rank} top={selected_top} focused={result['focused_rtl_contract_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
