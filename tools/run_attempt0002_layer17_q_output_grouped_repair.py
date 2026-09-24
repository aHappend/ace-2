#!/usr/bin/env python3
"""Run the nonofficial attempt-0002 layer-17 grouped-Q-output candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import peft
import torch
import transformers
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_checkpoint176_layer16_source_grouped_scale32_search as scale32
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    scale32_ratio,
)


ATTEMPT = ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0002"
DIAGNOSIS = ROOT / "diagnosis/stage1qualitydiag01"
PRIOR_CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer17-q-output-source-grouped-repair-v1"
    / "candidate-0001"
)
AUTHORIZED_ROOT = (
    ROOT
    / "evidence/candidates"
    / "w4a8-layer17-q-output-grouped-attempt0002-repair-v1"
)
ADAPTER = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4"
    / "attempt-0001/checkpoints/checkpoint-176"
)
RTL = ROOT / "rtl/ace2_layer17_q_output_grouped_scale32_core.sv"
TB = ROOT / "verification/tb/ace2_layer17_q_output_grouped_scale32_four_token_tb.sv"
RTL_TOP = "ace2_layer17_q_output_grouped_scale32_core"
TB_TOP = "ace2_layer17_q_output_grouped_scale32_four_token_tb"

STEPS = 4
TARGET_LAYER = 17
GROUP_SIZE = 128
GROUP_COUNT = backend.HIDDEN // GROUP_SIZE
PROMPT_TOKEN_COUNT = 34
PROMPT_TOKEN_IDS_SHA256 = (
    "dbdff6c533c804f645b0a76b8a44cf85e9b61f52bcce053e81189bd82f009e0d"
)
PROMPT_SHA256 = "90eaeeb4312de67b46fda2c43a80f3bfeb0ac20d0dc222a65f390c9407607c36"
SEALED_W4A8_TOKEN_IDS = [26614, 43895, 92464, 76521]
EXPECTED_BF16_GREEDY_TOKEN_IDS = [39814, 11, 1588, 594]
Q_PATTERN = re.compile(r"layer17_position(\d+)_q$")
SCORE_PATTERN = re.compile(
    r"layer17_position(?P<position>\d+)_head(?P<head>\d+)(?:_zero_cached_k)?$"
)
RTL_PASS_PATTERN = re.compile(
    r"ACE2_LAYER17_Q_OUTPUT_GROUPED_SCALE32_FOUR_TOKEN_RTL_PASS "
    r"directed=(?P<directed>\d+) outputs=(?P<outputs>\d+) "
    r"groups=(?P<groups>\d+) saturations=(?P<saturations>\d+) "
    r"input_stalls=(?P<input_stalls>\d+) output_stalls=(?P<output_stalls>\d+) "
    r"cycles=(?P<cycles>\d+) xz_clean=1 reset_recovery=1 clear_recovery=1"
)


class CandidateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def public_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": public_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": sha256_bytes(value.numpy().tobytes()),
    }


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return file_record(path)


def write_tensor(path: Path, value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tensor.numpy().tobytes())
    record = file_record(path)
    record.update({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    return record


def write_numpy(path: Path, value: np.ndarray, dtype: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes())
    record = file_record(path)
    record.update({"dtype": str(array.dtype), "shape": list(array.shape)})
    return record


def write_hex(path: Path, values: Iterable[int], width: int) -> dict[str, Any]:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )
    return file_record(path)


def write_sums(output: Path) -> None:
    members = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in members
        ),
        encoding="ascii",
    )


def run_process(
    command: list[str],
    log: Path,
    *,
    cwd: Path = ROOT,
    require_quiet: bool = False,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    require(
        completed.returncode == 0,
        f"command failed ({completed.returncode}): {' '.join(command)}",
    )
    if require_quiet:
        require(not completed.stdout.strip(), f"command emitted diagnostics: {' '.join(command)}")
    return completed.stdout


def verify_sums(directory: Path, log: Path) -> None:
    run_process(["sha256sum", "-c", "SHA256SUMS"], log, cwd=directory)


def verify_repo_sums(directory: Path, log: Path) -> None:
    run_process(
        ["sha256sum", "-c", public_path(directory / "SHA256SUMS")],
        log,
        cwd=ROOT,
    )


def find_dicts(value: Any, predicate: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if predicate(value):
            found.append(value)
        for child in value.values():
            found.extend(find_dicts(child, predicate))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_dicts(child, predicate))
    return found


def prompt_token_ids() -> list[int]:
    launch = json.loads(
        (ATTEMPT / "launch_provenance.json").read_text(encoding="utf-8")
    )
    contracts = find_dicts(
        launch,
        lambda item: (
            item.get("prompt_sha256") == PROMPT_SHA256
            and "prompt_token_ids" in item
        ),
    )
    contracts = [
        item
        for item in contracts
        if (
            "prompt_token_ids" in item
            and "chat_template_sha256" in item
            and "generation_bounds" in item
            and "source_files" in item
        )
    ]
    require(contracts, "sealed tokenizer contract is absent")
    identities = {
        sha256_bytes(
            canonical_bytes(
                {
                    "prompt_token_ids": item["prompt_token_ids"],
                    "chat_template_sha256": item["chat_template_sha256"],
                    "source_files": item["source_files"],
                    "generation_bounds": item["generation_bounds"],
                    "revision": item["revision"],
                }
            )
        )
        for item in contracts
    }
    require(len(identities) == 1, "sealed tokenizer contracts disagree")
    token_ids = list(map(int, contracts[0]["prompt_token_ids"]))
    require(len(token_ids) == PROMPT_TOKEN_COUNT, "sealed prompt token count differs")
    require(
        sha256_bytes(canonical_bytes(token_ids)) == PROMPT_TOKEN_IDS_SHA256,
        "sealed prompt token identity differs",
    )
    return token_ids


def validate_output(output: Path) -> None:
    require(output.parent == AUTHORIZED_ROOT, "output is outside authorized repair root")
    require(
        output.name.startswith("nonofficial-repair-"),
        "output must use a nonofficial-repair namespace",
    )
    require(ATTEMPT not in output.parents, "output overlaps sealed attempt")


def tool_versions() -> dict[str, str]:
    commands = {
        "iverilog": ["iverilog", "-V"],
        "vvp": ["vvp", "-V"],
        "verilator": ["verilator", "--version"],
    }
    versions: dict[str, str] = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        require(completed.returncode == 0, f"{name} version query failed")
        versions[name] = completed.stdout.strip()
    versions.update(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": importlib.metadata.version("tokenizers"),
            "safetensors": importlib.metadata.version("safetensors"),
        }
    )
    return versions


def preflight(output: Path) -> int:
    validate_output(output)
    require(not output.exists(), "nonofficial repair namespace already exists")
    output.mkdir(parents=True)
    logs = output / "preflight-logs"

    verify_sums(ATTEMPT, logs / "attempt-0002-sha256-check.log")
    verify_repo_sums(DIAGNOSIS, logs / "diagnosis-sha256-check.log")
    prompt_token_ids()

    diagnosis = json.loads((DIAGNOSIS / "result.json").read_text(encoding="utf-8"))
    prior = json.loads((PRIOR_CANDIDATE / "result.json").read_text(encoding="utf-8"))
    require(
        diagnosis["diagnosis"]["smallest_repair_task"].startswith(
            "Replace the tensor-wide signed-A8 scale at the layer-17 q-projection"
        ),
        "accepted diagnosis no longer selects this repair",
    )
    require(
        prior["selected_q_output_group_size"] == GROUP_SIZE,
        "frozen grouped-Q size differs",
    )

    compile_log = logs / "rtl-interface-compile.log"
    lint_log = logs / "rtl-interface-verilator.log"
    with tempfile.TemporaryDirectory(prefix="rtl-preflight-", dir=output) as temporary:
        image = Path(temporary) / "core.vvp"
        run_process(
            [
                "iverilog",
                "-g2012",
                "-Wall",
                "-s",
                RTL_TOP,
                "-o",
                str(image),
                str(RTL),
            ],
            compile_log,
            require_quiet=True,
        )
    run_process(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            RTL_TOP,
            str(RTL),
        ],
        lint_log,
        require_quiet=True,
    )

    contract = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mission": "attempt0002-layer17-q-output-grouped-repair-v1",
        "classification": "nonofficial_repair_candidate_no_official_attempt",
        "frozen_benchmark": {
            "source_attempt": "attempt-0002",
            "prompt_sha256": PROMPT_SHA256,
            "prompt_token_ids_sha256": PROMPT_TOKEN_IDS_SHA256,
            "prompt_token_count": PROMPT_TOKEN_COUNT,
            "prompt_text_persisted": False,
            "positions": STEPS,
            "aligned_feedback_token_ids": SEALED_W4A8_TOKEN_IDS,
            "bf16_expected_greedy_token_ids": EXPECTED_BF16_GREEDY_TOKEN_IDS,
            "selection": "greedy over 151936 outputs; lowest token ID wins exact ties",
            "score_policy": (
                "record all four same-prefix BF16/W4A8 top-1, rank, overlap, and "
                "divergence results without a pass threshold or selective rerun"
            ),
        },
        "repair_contract": {
            "only_modified_numeric_boundary": (
                "model.layers.17.self_attn.q_proj.accumulator_to_signed_a8_output"
            ),
            "q_output_group_size": GROUP_SIZE,
            "q_output_group_count": GROUP_COUNT,
            "group_alignment": "two complete 64-channel Q heads per group",
            "input_activation_samples": "unchanged signed A8",
            "weights": "unchanged signed W4 per output row",
            "group_scale": (
                "smallest legal Scale32 not below the model-derived maximum absolute "
                "dequantized Q accumulator in each group divided by 127"
            ),
            "rounding": "signed round-to-nearest ties-to-even",
            "saturation": "signed int8 clamp",
            "downstream_scale_binding": "explicit group Scale32 selected per Q head",
            "bf16_runtime_sidecar": False,
            "token_or_logit_override": False,
        },
        "rtl_public_contract": {
            "top": RTL_TOP,
            "ports": [
                "clk_i",
                "rst_ni",
                "clear_i",
                "in_valid_i",
                "in_ready_o",
                "accumulator_i:s32",
                "multiplier_i:s32",
                "right_shift_i:u6",
                "group_scale32_i:u32",
                "channel_i:u10",
                "last_i",
                "out_valid_o",
                "out_ready_i",
                "q_o:s8",
                "group_scale32_o:u32",
                "channel_o:u10",
                "last_o",
                "saturation_o",
            ],
            "preflight": "warning-clean Icarus compile and Verilator lint",
        },
        "collision_and_restoration": {
            "preflight_requires_absent_namespace": True,
            "execute_requires_exact_preflight_sha256s": True,
            "failed_execution_namespace_is_never_reused": True,
            "python_backend_hooks_restored_in_finally": True,
            "sealed_attempt_is_read_only_and_checked_before_and_after": True,
        },
        "bindings": {
            "attempt_sha256s": file_record(ATTEMPT / "SHA256SUMS"),
            "attempt_manifest": file_record(ATTEMPT / "manifest.json"),
            "attempt_authorization": file_record(ATTEMPT / "authorization.json"),
            "attempt_launch_provenance": file_record(
                ATTEMPT / "launch_provenance.json"
            ),
            "diagnosis_result": file_record(DIAGNOSIS / "result.json"),
            "diagnosis_sha256s": file_record(DIAGNOSIS / "SHA256SUMS"),
            "prior_group_selection": file_record(PRIOR_CANDIDATE / "result.json"),
            "runner": file_record(Path(__file__).resolve()),
            "rtl": file_record(RTL),
            "testbench": file_record(TB),
        },
        "tool_versions": tool_versions(),
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_created": False,
            "attempt_0002_mutated": False,
            "sealed_evidence_mutated": False,
        },
    }
    write_json(output / "candidate-contract.json", contract)
    result = {
        "schema_version": 1,
        "status": "PASS_NONOFFICIAL_REPAIR_PREFLIGHT",
        "classification": "preflight_only_no_model_candidate_execution",
        "public_rtl_contract_compiles_warning_clean": True,
        "grouped_scale_binding_frozen": True,
        "output_collision_policy_frozen": True,
        "restoration_path_frozen": True,
        "official_attempt_created": False,
        "bindings": {
            "contract": file_record(output / "candidate-contract.json"),
            "compile_log": file_record(compile_log),
            "lint_log": file_record(lint_log),
        },
    }
    write_json(output / "preflight-result.json", result)
    write_sums(output)
    print(
        "ACE2_ATTEMPT0002_GROUPED_Q_PREFLIGHT_PASS "
        f"group={GROUP_SIZE} output={public_path(output)}"
    )
    return 0


def stable_order(scores: np.ndarray) -> np.ndarray:
    ids = np.arange(scores.size, dtype=np.int64)
    return np.lexsort((ids, -scores))


def rank_of(scores: np.ndarray, token_id: int) -> int:
    selected = scores[token_id]
    ids = np.arange(scores.size, dtype=np.int64)
    return (
        int(
            np.count_nonzero(
                (scores > selected) | ((scores == selected) & (ids < token_id))
            )
        )
        + 1
    )


def summarize_scores(scores: np.ndarray, tokenizer: Any) -> dict[str, Any]:
    require(
        scores.shape == (backend.MODEL_OUTPUT_DOMAIN,),
        "score vector shape differs",
    )
    order = stable_order(scores)
    top_ids = order[:8]
    return {
        "top_token_id": int(top_ids[0]),
        "top_piece": tokenizer.decode(
            [int(top_ids[0])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "top_score": float(scores[top_ids[0]]),
        "top_tie_count": int(np.count_nonzero(scores == scores[top_ids[0]])),
        "top_candidates": [
            {"token_id": int(token_id), "score": float(scores[token_id])}
            for token_id in top_ids
        ],
    }


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    left_prob = np.exp(left64 - float(left64.max()))
    right_prob = np.exp(right64 - float(right64.max()))
    left_prob /= float(left_prob.sum())
    right_prob /= float(right_prob.sum())
    midpoint = 0.5 * (left_prob + right_prob)
    return float(
        0.5 * np.sum(left_prob * np.log(left_prob / midpoint))
        + 0.5 * np.sum(right_prob * np.log(right_prob / midpoint))
    )


def compare_scores(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference_order = stable_order(reference)
    candidate_order = stable_order(candidate)
    reference_top = int(reference_order[0])
    candidate_top = int(candidate_order[0])
    centered_reference = reference.astype(np.float64) - float(reference.mean())
    centered_candidate = candidate.astype(np.float64) - float(candidate.mean())
    denominator = float(
        np.linalg.norm(centered_reference) * np.linalg.norm(centered_candidate)
    )
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    reference_norm = float(np.linalg.norm(reference.astype(np.float64)))
    return {
        "top1_match": reference_top == candidate_top,
        "bf16_top_token_id": reference_top,
        "bf16_top_rank_in_w4a8": rank_of(candidate, reference_top),
        "w4a8_top_token_id": candidate_top,
        "w4a8_top_rank_in_bf16": rank_of(reference, candidate_top),
        "top8_overlap_count": len(
            set(map(int, reference_order[:8])) & set(map(int, candidate_order[:8]))
        ),
        "centered_cosine_similarity": (
            float(np.dot(centered_reference, centered_candidate) / denominator)
            if denominator
            else 0.0
        ),
        "relative_l2": (
            float(np.linalg.norm(delta) / reference_norm) if reference_norm else math.inf
        ),
        "jensen_shannon_divergence_nats": js_divergence(reference, candidate),
    }


def run_bf16_sequence(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    feedback_ids: list[int] | None,
    output: Path,
    label: str,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    past = None
    selected_ids: list[int] = []
    scores: list[np.ndarray] = []
    records = []
    with torch.inference_mode():
        for step in range(STEPS):
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = (
                result.logits[0, -1]
                .detach()
                .to(torch.float32)
                .cpu()
                .contiguous()
                .numpy()
            )
            summary = summarize_scores(logits, tokenizer)
            selected = int(summary["top_token_id"])
            selected_ids.append(selected)
            scores.append(logits)
            artifact = write_numpy(
                output / f"software/bf16/{label}-step-{step:02d}-logits-f32le.bin",
                logits,
                "<f4",
            )
            feedback = selected if feedback_ids is None else int(feedback_ids[step])
            cache_length = int(result.past_key_values.get_seq_length())
            records.append(
                {
                    "generation_index": step,
                    "context_token_count": len(prompt_ids) + step,
                    "cache_length_after_forward": cache_length,
                    "selected_token_id": selected,
                    "feedback_token_id": feedback if step < STEPS - 1 else None,
                    "ranking": summary,
                    "logits": artifact,
                }
            )
            past = result.past_key_values
            if step < STEPS - 1:
                input_ids = torch.tensor([[feedback]], dtype=torch.long)
                attention_mask = torch.ones(
                    (1, len(prompt_ids) + step + 1), dtype=torch.long
                )
    require(
        [item["cache_length_after_forward"] for item in records]
        == [len(prompt_ids) + step for step in range(STEPS)],
        f"BF16 {label} cache progression differs",
    )
    return (
        {
            "mode": label,
            "selected_token_ids": selected_ids,
            "feedback_policy": (
                "independent_greedy"
                if feedback_ids is None
                else "sealed_attempt0002_same_prefix"
            ),
            "decoded_selected_text": tokenizer.decode(
                selected_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "steps": records,
        },
        scores,
    )


def group_scales(
    accumulator: torch.Tensor,
    input_scale32: int,
    weight_scale32: tuple[int, ...],
) -> tuple[int, ...]:
    input_num, input_den = scale32_ratio(input_scale32)
    scales = []
    for start in range(0, accumulator.numel(), GROUP_SIZE):
        maximum = Fraction(0, 1)
        for row in range(start, start + GROUP_SIZE):
            weight_num, weight_den = scale32_ratio(weight_scale32[row])
            maximum = max(
                maximum,
                Fraction(
                    abs(int(accumulator[row])) * input_num * weight_num,
                    input_den * weight_den,
                ),
            )
        require(maximum > 0, "zero grouped Q output is unsupported")
        scales.append(
            ceil_scale32_from_ratio(
                maximum.numerator,
                maximum.denominator * 127,
            )
        )
    return tuple(scales)


def scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


class GroupedQOutputCache(localizer.FastProjectionCache):
    """Change only layer-17 Q accumulator-to-A8 output metadata and samples."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        self.position_records: dict[int, dict[str, Any]] = {}
        self.replacements = 0

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        baseline = super().derive_projection(
            name, merged, input_q, input_scale, float_input, source_hashes
        )
        match = Q_PATTERN.fullmatch(name)
        if match is None:
            return baseline

        position = int(match.group(1))
        input_scale32 = ceil_scale32_from_float(float(input_scale))
        weight_scale32 = tuple(
            ceil_scale32_from_float(float(value))
            for value in baseline["weight_scale"].tolist()
        )
        accumulator = baseline["accumulator"].to(torch.int64)
        scales = group_scales(accumulator, input_scale32, weight_scale32)
        multiplier_values = []
        shift_values = []
        for row, weight_record in enumerate(weight_scale32):
            multiplier, shift = scale32._derive_scale32_multiplier(
                input_scale32,
                weight_record,
                scales[row // GROUP_SIZE],
            )
            multiplier_values.append(multiplier)
            shift_values.append(shift)
        multiplier = torch.tensor(multiplier_values, dtype=torch.int64)
        right_shift = torch.tensor(shift_values, dtype=torch.int64)
        rounded = localizer.round_shift_even_tensor(
            accumulator * multiplier, right_shift
        )
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        group_scale32 = torch.tensor(scales, dtype=torch.int64)
        head_scale32 = torch.tensor(
            [
                scales[(head * backend.HEAD_DIM) // GROUP_SIZE]
                for head in range(backend.Q_HEADS)
            ],
            dtype=torch.int64,
        )
        record = {
            "position": position,
            "input_q": input_q.detach().cpu().contiguous(),
            "input_scale": float(input_scale),
            "input_scale32": input_scale32,
            "qweight": baseline["qweight"].detach().cpu().contiguous(),
            "weight_scale": baseline["weight_scale"].detach().cpu().contiguous(),
            "weight_scale32": torch.tensor(weight_scale32, dtype=torch.int64),
            "accumulator": accumulator.detach().cpu().contiguous(),
            "baseline_output_scale": float(baseline["output_scale"]),
            "baseline_output_q": baseline["output_q"].detach().cpu().contiguous(),
            "group_scale32": group_scale32,
            "head_scale32": head_scale32,
            "multiplier": multiplier,
            "right_shift": right_shift,
            "rounded": rounded.detach().cpu().contiguous(),
            "output_q": output_q.detach().cpu().contiguous(),
            "saturation": saturation.detach().cpu().contiguous(),
        }
        self.position_records[position] = record
        self.replacements += 1
        result = dict(baseline)
        result.update(
            {
                "multiplier": multiplier,
                "right_shift": right_shift,
                "accumulator": accumulator,
                "rounded": rounded,
                "output_q": output_q,
                "output_scale": scale32_float(int(head_scale32[0])),
                "saturation": saturation,
                "grouped_q_output_scale32": True,
            }
        )
        return result


class PerPositionQGuard:
    def __init__(self) -> None:
        self.original = backend.derive_layer_token
        self.template_invalidations = 0

    def install(self) -> None:
        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            if layer_id == TARGET_LAYER and template is not None:
                if "q" in template["qkv"]:
                    self.template_invalidations += 1
                template["qkv"].pop("q", None)
            return self.original(layer_id, state, cache, template, weights, adapter)

        backend.derive_layer_token = derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original


class QHeadScaleGuard:
    def __init__(self, cache: GroupedQOutputCache) -> None:
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
            replacement = backend.canonical.AttentionScoreCase(
                case.name,
                case.q_values,
                case.k_values,
                scale32_float(int(record["head_scale32"][head])),
                case.key_scale,
            )
            self.replacements += 1
            return self.original(replacement)

        backend.canonical.reference_attention_score = score

    def restore(self) -> None:
        backend.canonical.reference_attention_score = self.original


def derive_lm_head_weights(embedding: torch.Tensor) -> dict[str, torch.Tensor]:
    return localizer.derive_lm_head_weights(embedding)


def score_w4a8(
    state: dict[str, Any],
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
) -> tuple[np.ndarray, dict[str, torch.Tensor | float]]:
    item = backend.derive_final_rmsnorm_case(state, norm_gain)
    float_chunks = []
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        float_chunks.append(
            torch.mv(embedding[start:stop].to(torch.float32), item["float_norm"])
        )
    float_output = torch.cat(float_chunks).contiguous()
    output_scale = float(backend.canonical.scale_for(float_output))
    activation = item["final_q"].to(torch.int32)
    accumulator = torch.empty((backend.MODEL_OUTPUT_DOMAIN,), dtype=torch.int64)
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        accumulator[start:stop] = (
            head["qweight"][start:stop].to(torch.int32) * activation
        ).sum(dim=1, dtype=torch.int64)
    multiplier, right_shift = backend.canonical.derive_multiplier(
        float(item["final_scale"]) * head["weight_scale"] / output_scale
    )
    rounded = localizer.round_shift_even_tensor(
        accumulator * multiplier, right_shift
    )
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
    scores = output_q.to(torch.float64).numpy() * output_scale
    return scores, {
        "final_q": item["final_q"].detach().cpu().contiguous(),
        "output_q": output_q.detach().cpu().contiguous(),
        "output_scale": output_scale,
        "saturation": saturation.detach().cpu().contiguous(),
    }


def run_w4a8_aligned(
    prompt_ids: list[int],
    tokenizer: Any,
    weights: Any,
    adapter: Any,
    output: Path,
) -> tuple[dict[str, Any], list[np.ndarray], list[dict[str, Any]]]:
    embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
    norm_gain = weights.get_tensor("model.norm.weight").contiguous()
    head = derive_lm_head_weights(embedding)
    cache = GroupedQOutputCache(weights, adapter)
    q_guard = PerPositionQGuard()
    score_guard = QHeadScaleGuard(cache)
    layer_caches = [backend.empty_layer_cache() for _ in range(backend.LAYERS)]
    templates: list[dict[str, Any] | None] = [None] * backend.LAYERS
    selected_ids: list[int] = []
    scores: list[np.ndarray] = []
    steps = []
    scored_q_records = []
    started = time.monotonic()
    cache.install()
    q_guard.install()
    score_guard.install()
    try:
        total_positions = len(prompt_ids) + STEPS - 1
        with torch.no_grad():
            for position in range(total_positions):
                token_id = (
                    prompt_ids[position]
                    if position < len(prompt_ids)
                    else SEALED_W4A8_TOKEN_IDS[position - len(prompt_ids)]
                )
                state = backend.embedding_state(weights, token_id, position)
                for layer_id in range(backend.LAYERS):
                    _derived, state, templates[layer_id] = backend.derive_layer_token(
                        layer_id,
                        state,
                        layer_caches[layer_id],
                        templates[layer_id],
                        weights,
                        adapter,
                    )
                    del _derived
                if position < len(prompt_ids) - 1:
                    continue
                step = position - (len(prompt_ids) - 1)
                score_values, head_tensors = score_w4a8(
                    state, norm_gain, embedding, head
                )
                ranking = summarize_scores(score_values, tokenizer)
                selected_ids.append(int(ranking["top_token_id"]))
                scores.append(score_values)
                q_record = cache.position_records[position]
                scored_q_records.append(q_record)
                step_dir = output / f"software/w4a8/step-{step:02d}"
                artifacts = {
                    "lm_head_output_s8": write_tensor(
                        step_dir / "lm-head-output-s8.bin",
                        head_tensors["output_q"],
                    ),
                    "lm_head_output_scale_f64le": write_numpy(
                        step_dir / "lm-head-output-scale-f64le.bin",
                        np.asarray([head_tensors["output_scale"]]),
                        "<f8",
                    ),
                    "final_rmsnorm_s8": write_tensor(
                        step_dir / "final-rmsnorm-s8.bin",
                        head_tensors["final_q"],
                    ),
                    "q_input_s8": write_tensor(
                        step_dir / "layer17-q-input-s8.bin", q_record["input_q"]
                    ),
                    "q_accumulator_s32": write_tensor(
                        step_dir / "layer17-q-accumulator-s32.bin",
                        q_record["accumulator"],
                    ),
                    "q_multiplier_s32": write_tensor(
                        step_dir / "layer17-q-multiplier-s32.bin",
                        q_record["multiplier"],
                    ),
                    "q_shift_u6": write_tensor(
                        step_dir / "layer17-q-shift-u6.bin",
                        q_record["right_shift"],
                    ),
                    "q_group_scale32": write_tensor(
                        step_dir / "layer17-q-group-scale32.bin",
                        q_record["group_scale32"],
                    ),
                    "q_output_s8": write_tensor(
                        step_dir / "layer17-q-output-s8.bin",
                        q_record["output_q"],
                    ),
                    "q_saturation": write_tensor(
                        step_dir / "layer17-q-saturation.bin",
                        q_record["saturation"],
                    ),
                    "q_baseline_tensor_output_s8": write_tensor(
                        step_dir / "layer17-q-baseline-tensor-output-s8.bin",
                        q_record["baseline_output_q"],
                    ),
                }
                steps.append(
                    {
                        "generation_index": step,
                        "absolute_position": position,
                        "context_token_count": position + 1,
                        "selected_token_id": int(ranking["top_token_id"]),
                        "feedback_token_id": (
                            SEALED_W4A8_TOKEN_IDS[step]
                            if step < STEPS - 1
                            else None
                        ),
                        "ranking": ranking,
                        "layer17_q": {
                            "input_q": tensor_record(q_record["input_q"]),
                            "qweight": tensor_record(q_record["qweight"]),
                            "input_scale": q_record["input_scale"],
                            "input_scale32": f"0x{q_record['input_scale32']:08x}",
                            "baseline_tensor_output_scale": q_record[
                                "baseline_output_scale"
                            ],
                            "group_size": GROUP_SIZE,
                            "group_count": GROUP_COUNT,
                            "group_scale32_min": (
                                f"0x{int(q_record['group_scale32'].min()):08x}"
                            ),
                            "group_scale32_max": (
                                f"0x{int(q_record['group_scale32'].max()):08x}"
                            ),
                            "baseline_vs_grouped_output_mismatch_count": int(
                                (
                                    q_record["baseline_output_q"]
                                    != q_record["output_q"]
                                ).sum()
                            ),
                            "saturation_count": int(
                                q_record["saturation"].to(torch.int64).sum()
                            ),
                        },
                        "artifacts": artifacts,
                    }
                )
                print(
                    "ACE2_ATTEMPT0002_GROUPED_Q_SOFTWARE_STEP "
                    f"step={step} selected={ranking['top_token_id']} "
                    f"groups={GROUP_COUNT}",
                    flush=True,
                )
    finally:
        score_guard.restore()
        q_guard.restore()
        cache.restore()

    expected_positions = len(prompt_ids) + STEPS - 1
    expected_scores = backend.Q_HEADS * (2 * expected_positions - 1)
    require(cache.replacements == expected_positions, "layer-17 Q replacement count differs")
    require(
        q_guard.template_invalidations == expected_positions - 1,
        "layer-17 Q template invalidation count differs",
    )
    require(
        score_guard.replacements == expected_scores,
        "group Scale32 did not reach every layer-17 Q-head score",
    )
    return (
        {
            "mode": "same_prefix_teacher_forced",
            "selected_token_ids": selected_ids,
            "feedback_token_ids": SEALED_W4A8_TOKEN_IDS,
            "decoded_selected_text": tokenizer.decode(
                selected_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "steps": steps,
            "numeric_scope_audit": {
                "processed_positions": expected_positions,
                "layer17_q_output_replacements": cache.replacements,
                "layer17_q_template_invalidations": q_guard.template_invalidations,
                "layer17_attention_score_scale_replacements": score_guard.replacements,
                "expected_attention_score_scale_replacements": expected_scores,
                "all_other_projection_outputs": "baseline FastProjectionCache behavior",
            },
            "elapsed_wall_seconds": time.monotonic() - started,
        },
        scores,
        scored_q_records,
    )


def rne_numpy(values: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index in range(values.size):
        value = int(values[index])
        shift = int(shifts[index])
        if shift == 0:
            result[index] = value
            continue
        magnitude = abs(value)
        base = magnitude >> shift
        remainder = magnitude & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and (base & 1)):
            base += 1
        result[index] = -base if value < 0 else base
    return result


def write_rtl_vectors(
    output: Path, records: list[dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    vector_dir = output / "focused-rtl/vectors"
    accumulators = []
    multipliers = []
    shifts = []
    scales = []
    expected = []
    saturation = []
    for record in records:
        accumulator = record["accumulator"].numpy().astype(np.int64)
        multiplier = record["multiplier"].numpy().astype(np.int64)
        shift = record["right_shift"].numpy().astype(np.int64)
        group_scale = record["group_scale32"].numpy().astype(np.int64)
        output_q = record["output_q"].numpy().astype(np.int8)
        saturated = record["saturation"].numpy().astype(np.uint8)
        require(
            np.all((accumulator >= -(1 << 31)) & (accumulator < (1 << 31))),
            "Q accumulator exceeds RTL signed-32 input",
        )
        require(
            np.all((multiplier >= 0) & (multiplier <= (1 << 31) - 1)),
            "Q multiplier exceeds RTL signed-32 input",
        )
        require(np.all((shift >= 0) & (shift <= 63)), "Q shift exceeds RTL u6")
        require(
            np.all((group_scale >= 0) & (group_scale <= 0xFFFFFFFF)),
            "Q group Scale32 exceeds RTL u32",
        )
        recomputed = rne_numpy(accumulator * multiplier, shift)
        recomputed_q = np.clip(recomputed, -128, 127).astype(np.int8)
        recomputed_sat = ((recomputed < -128) | (recomputed > 127)).astype(
            np.uint8
        )
        require(np.array_equal(recomputed_q, output_q), "independent Q output differs")
        require(
            np.array_equal(recomputed_sat, saturated),
            "independent Q saturation differs",
        )
        accumulators.extend(map(int, accumulator))
        multipliers.extend(map(int, multiplier))
        shifts.extend(map(int, shift))
        scales.extend(map(int, group_scale))
        expected.extend(map(int, output_q))
        saturation.extend(map(int, saturated))

    artifacts = {
        "accumulator": write_hex(
            vector_dir / "q_accumulator_s32.hex", accumulators, 32
        ),
        "multiplier": write_hex(
            vector_dir / "q_multiplier_s32.hex", multipliers, 32
        ),
        "shift": write_hex(vector_dir / "q_shift_u6.hex", shifts, 8),
        "group_scale32": write_hex(
            vector_dir / "q_group_scale32.hex", scales, 32
        ),
        "expected": write_hex(vector_dir / "q_expected_s8.hex", expected, 8),
        "saturation": write_hex(
            vector_dir / "q_expected_saturation.hex", saturation, 8
        ),
    }
    manifest = {
        "schema_version": 1,
        "classification": "nonofficial_four_token_focused_rtl_vectors",
        "steps": STEPS,
        "outputs_per_step": backend.HIDDEN,
        "total_outputs": STEPS * backend.HIDDEN,
        "group_size": GROUP_SIZE,
        "groups_per_step": GROUP_COUNT,
        "total_group_scale_records": STEPS * GROUP_COUNT,
        "expected_saturation_count": int(sum(saturation)),
        "independent_numpy_recomputation": "exact",
        "artifacts": artifacts,
    }
    write_json(vector_dir / "manifest.json", manifest)
    return manifest, int(sum(saturation))


def run_focused_rtl(
    output: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest, saturation_count = write_rtl_vectors(output, records)
    rtl_output = output / "focused-rtl"
    logs = rtl_output / "logs"
    sim = rtl_output / "sim"
    sim.mkdir(parents=True)
    image = sim / "four-token.vvp"
    compile_stdout = run_process(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-Wno-timescale",
            "-s",
            TB_TOP,
            "-o",
            str(image),
            str(RTL),
            str(TB),
        ],
        logs / "iverilog-compile.log",
        require_quiet=True,
    )
    require(not compile_stdout, "Icarus compile output is nonempty")
    simulation_stdout = run_process(
        [
            "vvp",
            str(image),
            f"+VECTOR_DIR={public_path(rtl_output / 'vectors')}",
        ],
        logs / "simulation.log",
    )
    match = RTL_PASS_PATTERN.search(simulation_stdout)
    require(match is not None, "focused four-token RTL PASS marker is absent")
    metrics = {key: int(value) for key, value in match.groupdict().items()}
    require(metrics["directed"] == 6, "RTL directed count differs")
    require(
        metrics["outputs"] == STEPS * backend.HIDDEN,
        "RTL output count differs",
    )
    require(
        metrics["groups"] == STEPS * GROUP_COUNT,
        "RTL group count differs",
    )
    require(metrics["saturations"] == saturation_count, "RTL saturation count differs")
    run_process(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            RTL_TOP,
            str(RTL),
        ],
        logs / "verilator-lint.log",
        require_quiet=True,
    )
    result = {
        "schema_version": 1,
        "status": "PASS_FOCUSED_FOUR_TOKEN_GROUPED_Q_RTL_REFERENCE_AGREEMENT",
        "classification": "nonofficial_candidate_no_official_attempt",
        "numeric_boundary": (
            "layer17 Q signed-s32 accumulator to grouped-Scale32 signed-A8 output"
        ),
        "metrics": metrics,
        "checks": {
            "independent_numpy_reference_exact": True,
            "all_four_model_positions_exact": True,
            "signed_ties_to_even": True,
            "positive_and_negative_saturation": True,
            "input_backpressure": True,
            "output_backpressure_and_stability": True,
            "asynchronous_reset_recovery": True,
            "synchronous_clear_recovery": True,
            "xz_clean_every_checked_cycle": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
        },
        "bindings": {
            "vectors": file_record(rtl_output / "vectors/manifest.json"),
            "rtl": file_record(RTL),
            "testbench": file_record(TB),
            "compile_log": file_record(logs / "iverilog-compile.log"),
            "simulation_log": file_record(logs / "simulation.log"),
            "lint_log": file_record(logs / "verilator-lint.log"),
        },
    }
    write_json(rtl_output / "result.json", result)
    return result


def verify_preflight(output: Path) -> dict[str, Any]:
    validate_output(output)
    require(output.is_dir(), "preflight namespace is absent")
    forbidden = [
        output / "execution-start.json",
        output / "software-result.json",
        output / "result.json",
        output / "failure.json",
        output / "focused-rtl",
    ]
    require(not any(path.exists() for path in forbidden), "repair namespace was already executed")
    verify_sums(output, output / "preflight-recheck.log")
    contract = json.loads((output / "candidate-contract.json").read_text(encoding="utf-8"))
    require(
        contract["bindings"]["runner"]["sha256"]
        == sha256_file(Path(__file__).resolve()),
        "runner changed after preflight",
    )
    require(
        contract["bindings"]["rtl"]["sha256"] == sha256_file(RTL),
        "RTL changed after preflight",
    )
    require(
        contract["bindings"]["testbench"]["sha256"] == sha256_file(TB),
        "testbench changed after preflight",
    )
    require(
        contract["repair_contract"]["q_output_group_size"] == GROUP_SIZE,
        "preflight group size differs",
    )
    return contract


def execute(output: Path) -> int:
    contract = verify_preflight(output)
    write_json(
        output / "execution-start.json",
        {
            "schema_version": 1,
            "started_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "classification": "nonofficial_repair_execution",
            "contract": file_record(output / "candidate-contract.json"),
            "official_attempt_created": False,
        },
    )
    verify_sums(
        ATTEMPT,
        output / "execution-logs/attempt-0002-before-sha256-check.log",
    )
    verify_repo_sums(
        DIAGNOSIS,
        output / "execution-logs/diagnosis-before-sha256-check.log",
    )
    prompt_ids = prompt_token_ids()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )

    bf16_started = time.monotonic()
    bf16_model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        attn_implementation="eager",
        device_map=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    bf16_model = PeftModel.from_pretrained(
        bf16_model,
        ADAPTER,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    ).eval()
    bf16_aligned, bf16_aligned_scores = run_bf16_sequence(
        bf16_model,
        tokenizer,
        prompt_ids,
        SEALED_W4A8_TOKEN_IDS,
        output,
        "aligned",
    )
    bf16_greedy, _bf16_greedy_scores = run_bf16_sequence(
        bf16_model,
        tokenizer,
        prompt_ids,
        None,
        output,
        "greedy",
    )
    require(
        bf16_greedy["selected_token_ids"] == EXPECTED_BF16_GREEDY_TOKEN_IDS,
        "fresh BF16 greedy reference differs from accepted diagnosis",
    )
    bf16_elapsed = time.monotonic() - bf16_started
    del bf16_model, _bf16_greedy_scores
    gc.collect()

    w4a8_started = time.monotonic()
    with safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        w4a8, w4a8_scores, q_records = run_w4a8_aligned(
            prompt_ids, tokenizer, weights, adapter, output
        )
    w4a8_elapsed = time.monotonic() - w4a8_started

    comparisons = []
    for step in range(STEPS):
        comparison = compare_scores(
            bf16_aligned_scores[step], w4a8_scores[step]
        )
        comparison.update(
            {
                "generation_index": step,
                "identical_prefix_token_count": len(prompt_ids) + step,
                "feedback_prefix": "sealed_attempt0002_w4a8_tokens",
            }
        )
        comparisons.append(comparison)
    top1_matches = sum(int(item["top1_match"]) for item in comparisons)
    software_result = {
        "schema_version": 1,
        "status": "COMPLETED_ALIGNED_FOUR_TOKEN_BF16_W4A8_COMPARISON",
        "classification": "nonofficial_candidate_no_official_attempt",
        "bf16": {
            "aligned_same_prefix": bf16_aligned,
            "independent_greedy": bf16_greedy,
            "elapsed_wall_seconds": bf16_elapsed,
        },
        "w4a8_grouped_layer17_q_output": w4a8,
        "same_prefix_comparisons": comparisons,
        "summary": {
            "aligned_positions": STEPS,
            "bf16_top1_matches": top1_matches,
            "bf16_top1_mismatches": STEPS - top1_matches,
            "bf16_greedy_token_ids": bf16_greedy["selected_token_ids"],
            "w4a8_aligned_selected_token_ids": w4a8["selected_token_ids"],
            "quality_threshold_applied": False,
            "selective_rerun_performed": False,
        },
        "elapsed_wall_seconds": {
            "bf16": bf16_elapsed,
            "w4a8": w4a8_elapsed,
        },
        "bindings": {
            "contract": file_record(output / "candidate-contract.json"),
            "attempt_manifest": file_record(ATTEMPT / "manifest.json"),
            "diagnosis": file_record(DIAGNOSIS / "result.json"),
        },
    }
    write_json(output / "software-result.json", software_result)

    rtl_result = run_focused_rtl(output, q_records)
    verify_sums(
        ATTEMPT,
        output / "execution-logs/attempt-0002-after-sha256-check.log",
    )
    verify_repo_sums(
        DIAGNOSIS,
        output / "execution-logs/diagnosis-after-sha256-check.log",
    )

    result = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": (
            "PASS_NONOFFICIAL_GROUPED_Q_CANDIDATE_BF16_TOP1_4_OF_4_RTL_EXACT"
            if top1_matches == STEPS
            else "COMPLETED_NONOFFICIAL_GROUPED_Q_CANDIDATE_"
            f"BF16_TOP1_{top1_matches}_OF_4_RTL_EXACT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "software": {
            "aligned_positions": STEPS,
            "bf16_top1_matches": top1_matches,
            "bf16_top1_mismatches": STEPS - top1_matches,
            "result": file_record(output / "software-result.json"),
        },
        "rtl": {
            "status": rtl_result["status"],
            "outputs_checked": rtl_result["metrics"]["outputs"],
            "result": file_record(output / "focused-rtl/result.json"),
        },
        "claim_boundary": (
            "Four same-prefix BF16/W4A8 software positions and focused layer-17 "
            "grouped-Q-output RTL/reference agreement only. This is not an official "
            "attempt, independent acceptance, synthesis, timing, area, or product claim."
        ),
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_created": False,
            "attempt_0002_sha256s_valid_before_and_after": True,
            "diagnosis_sha256s_valid_before_and_after": True,
            "sealed_evidence_mutated": False,
            "quality_threshold_changed": False,
            "selective_rerun_performed": False,
            "independent_review_still_required": True,
        },
        "bindings": {
            "contract": file_record(output / "candidate-contract.json"),
            "software_result": file_record(output / "software-result.json"),
            "rtl_result": file_record(output / "focused-rtl/result.json"),
            "runner": file_record(Path(__file__).resolve()),
            "rtl": file_record(RTL),
            "testbench": file_record(TB),
        },
    }
    write_json(output / "result.json", result)
    write_sums(output)
    print(
        "ACE2_ATTEMPT0002_GROUPED_Q_CANDIDATE_COMPLETE "
        f"bf16_top1={top1_matches}/4 rtl_outputs={rtl_result['metrics']['outputs']} "
        f"official_attempt_created=false output={public_path(output)}",
        flush=True,
    )
    return 0


def record_failure(output: Path, error: Exception, phase: str) -> None:
    if not output.is_dir() or (output / "result.json").exists():
        return
    failure = {
        "schema_version": 1,
        "status": f"FAILED_NONOFFICIAL_REPAIR_{phase.upper()}",
        "classification": "wiring_or_configuration_failure_not_model_evidence",
        "failure_taxonomy": "candidate_harness_or_environment_failure",
        "phase": phase,
        "root_cause_hypothesis": str(error),
        "required_regression": (
            "Correct the concrete harness/environment defect and execute a new "
            "nonofficial-repair namespace with the unchanged frozen contract."
        ),
        "namespace_reusable": False,
        "official_attempt_created": False,
    }
    write_json(output / "failure.json", failure)
    write_sums(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    try:
        return preflight(output) if args.preflight_only else execute(output)
    except Exception as error:
        record_failure(output, error, "preflight" if args.preflight_only else "execution")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
