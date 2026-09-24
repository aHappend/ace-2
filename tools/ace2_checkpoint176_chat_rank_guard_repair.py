#!/usr/bin/env python3
"""Candidate-only checkpoint-176 W4A8 LM-head rank-resolution repair."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import peft
import tokenizers
import torch
import transformers
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


CANONICAL_PROMPT = "Hello"
CANONICAL_TOKEN_IDS = list(generation_runner.CANONICAL_HELLO_CHAT_TOKEN_IDS)
CONTINUATION_STEPS = 4
RANK_GUARD_BITS = 8
RANK_SCORE_WIDTH = 16
TOP_K = 8
FORBIDDEN_OUTPUT = ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1"
RTL_SOURCE = ROOT / "rtl/ace2_lm_head_rank_guard_core.sv"
RTL_TB = ROOT / "verification/tb/ace2_lm_head_rank_guard_tb.sv"


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
    return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": public_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return file_record(path)


def write_text(path: Path, value: str) -> dict[str, Any]:
    return write_bytes(path, value.encode("utf-8"))


def write_json(path: Path, value: Any) -> dict[str, Any]:
    return write_bytes(path, canonical_bytes(value))


def tensor_record(path: Path, value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.numpy().tobytes()
    record = write_bytes(path, raw)
    record.update({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    return record


def tensor_sha256(value: torch.Tensor) -> str:
    return sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def decode_piece(tokenizer: Any, token_id: int) -> str:
    if token_id >= len(tokenizer):
        return "<UNMAPPED>"
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def stable_top_ids(values: torch.Tensor, count: int) -> list[int]:
    return [
        int(value)
        for value in torch.argsort(values, descending=True, stable=True)[:count]
    ]


def rank_of(values: torch.Tensor, token_id: int) -> int:
    selected = values[token_id]
    ids = torch.arange(values.numel(), device=values.device)
    better = (values > selected) | ((values == selected) & (ids < token_id))
    return int(better.sum().item()) + 1


def top_candidates(tokenizer: Any, values: torch.Tensor, count: int) -> list[dict[str, Any]]:
    return [
        {
            "token_id": token_id,
            "piece": decode_piece(tokenizer, token_id),
            "score": float(values[token_id]),
        }
        for token_id in stable_top_ids(values, count)
    ]


def reference_trace(
    output: Path,
    snapshot: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    print("ACE2_RANK_GUARD_REFERENCE_LOAD", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        attn_implementation="eager",
        device_map=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        model,
        backend.canonical.ADAPTER_DIR,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    input_ids = torch.tensor([CANONICAL_TOKEN_IDS], dtype=torch.long)
    past = None
    generated: list[int] = []
    steps = []
    with torch.inference_mode():
        for step in range(CONTINUATION_STEPS):
            result = model(
                input_ids=input_ids,
                attention_mask=torch.ones(
                    (1, len(CANONICAL_TOKEN_IDS) + step), dtype=torch.long
                ),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = result.logits[0, -1].float().cpu().contiguous()
            selected = int(torch.argmax(logits).item())
            artifact = tensor_record(
                output / f"cpu/reference/step-{step:02d}-logits-f32le.bin", logits
            )
            steps.append(
                {
                    "step": step,
                    "selected_token_id": selected,
                    "selected_piece": decode_piece(tokenizer, selected),
                    "top_candidates": top_candidates(tokenizer, logits, TOP_K),
                    "logits": artifact,
                    "cache_sequence_length_after_call": int(
                        result.past_key_values.get_seq_length()
                    ),
                }
            )
            generated.append(selected)
            print(
                f"ACE2_RANK_GUARD_REFERENCE_STEP step={step} token={selected} "
                f"piece={decode_piece(tokenizer, selected)!r}",
                flush=True,
            )
            past = result.past_key_values
            input_ids = torch.tensor([[selected]], dtype=torch.long)
    decoded = tokenizer.decode(
        generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    del model, past, input_ids
    gc.collect()
    return {"generated_token_ids": generated, "decoded_text": decoded, "steps": steps}


def derive_lm_head_weights(embedding: torch.Tensor) -> dict[str, Any]:
    output_count = backend.MODEL_OUTPUT_DOMAIN
    qweight = torch.empty((output_count, backend.HIDDEN), dtype=torch.int8)
    weight_scale = torch.empty((output_count,), dtype=torch.float64)
    for start in range(0, output_count, 2048):
        stop = min(output_count, start + 2048)
        rows = embedding[start:stop].to(torch.float64)
        scale = rows.abs().amax(dim=1) / 7.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight[start:stop] = torch.round(rows / scale[:, None]).clamp(-8, 7).to(
            torch.int8
        )
        weight_scale[start:stop] = scale
    return {
        "qweight": qweight,
        "weight_scale": weight_scale,
        "qweight_sha256": tensor_sha256(qweight),
        "weight_scale_sha256": tensor_sha256(weight_scale),
    }


def score_step(
    output: Path,
    step: int,
    state: dict[str, Any],
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, Any],
    tokenizer: Any,
    reference_token: int,
) -> tuple[dict[str, Any], torch.Tensor, int]:
    item = backend.derive_final_rmsnorm_case(state, norm_gain)
    float_chunks = []
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        float_chunks.append(
            torch.mv(
                embedding[start:stop].to(torch.float32), item["float_norm"]
            )
        )
    float_output = torch.cat(float_chunks).contiguous()
    output_scale = backend.canonical.scale_for(float_output)
    activation = item["final_q"].to(torch.int32)
    accumulator = torch.empty((backend.MODEL_OUTPUT_DOMAIN,), dtype=torch.int64)
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        accumulator[start:stop] = (
            head["qweight"][start:stop].to(torch.int32) * activation
        ).sum(dim=1, dtype=torch.int64)
    require(
        bool(torch.all(accumulator >= -(1 << 31)))
        and bool(torch.all(accumulator < (1 << 31)),),
        "LM-head accumulator exceeds signed-32",
    )

    real_score = (
        accumulator.to(torch.float64)
        * float(item["final_scale"])
        * head["weight_scale"]
    )
    base_multiplier, base_shift = backend.canonical.derive_multiplier(
        float(item["final_scale"]) * head["weight_scale"] / output_scale
    )
    base_rounded = backend.lm_head.round_outputs(
        accumulator, base_multiplier, base_shift
    )
    base_saturation = (base_rounded < -128) | (base_rounded > 127)
    base_output = base_rounded.clamp(-128, 127).to(torch.int8)

    guard_scale = output_scale / float(1 << RANK_GUARD_BITS)
    guard_multiplier, guard_shift = backend.canonical.derive_multiplier(
        float(item["final_scale"]) * head["weight_scale"] / guard_scale
    )
    guard_rounded = backend.lm_head.round_outputs(
        accumulator, guard_multiplier, guard_shift
    )
    guard_min = -(1 << (RANK_SCORE_WIDTH - 1))
    guard_max = (1 << (RANK_SCORE_WIDTH - 1)) - 1
    guard_saturation = (guard_rounded < guard_min) | (guard_rounded > guard_max)
    guard_score = guard_rounded.clamp(guard_min, guard_max).to(torch.int16)

    real_top = int(torch.argmax(real_score).item())
    rounded_top = int(torch.argmax(base_rounded).item())
    baseline_top = int(torch.argmax(base_output).item())
    guard_top = int(torch.argmax(guard_score).item())
    artifacts = {
        "final_rmsnorm_s8": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/final-rmsnorm-s8.bin",
            item["final_q"],
        ),
        "accumulator_s64le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/lm-head-accumulator-s64le.bin",
            accumulator,
        ),
        "real_score_f64le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/lm-head-real-score-f64le.bin",
            real_score,
        ),
        "baseline_multiplier_s64le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/baseline-multiplier-s64le.bin",
            base_multiplier,
        ),
        "baseline_shift_s64le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/baseline-shift-s64le.bin",
            base_shift,
        ),
        "baseline_rounded_s64le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/baseline-rounded-s64le.bin",
            base_rounded,
        ),
        "baseline_output_s8": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/baseline-output-s8.bin",
            base_output,
        ),
        "guard_multiplier_s64le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/guard-multiplier-s64le.bin",
            guard_multiplier,
        ),
        "guard_shift_s64le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/guard-shift-s64le.bin",
            guard_shift,
        ),
        "guard_rounded_s64le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/guard-rounded-s64le.bin",
            guard_rounded,
        ),
        "guard_score_s16le": tensor_record(
            output / f"cpu/w4a8/step-{step:02d}/guard-score-s16le.bin",
            guard_score,
        ),
    }
    record = {
        "step": step,
        "prefix_matches_reference_before_step": True,
        "reference_token_id": reference_token,
        "reference_piece": decode_piece(tokenizer, reference_token),
        "final_rmsnorm": {
            "input_scale": float(item["input_scale"]),
            "output_scale": float(item["final_scale"]),
            "sumsq": int(item["rmsnorm_result"].sumsq),
            "inv_rms_q30": int(item["rmsnorm_result"].inv_rms_q30),
            "saturation_seen": bool(item["rmsnorm_result"].saturation_seen),
        },
        "lm_head": {
            "baseline_output_scale": float(output_scale),
            "guard_output_scale": float(guard_scale),
            "guard_fractional_bits": RANK_GUARD_BITS,
            "accumulator_min": int(accumulator.min().item()),
            "accumulator_max": int(accumulator.max().item()),
            "real_w4_top_token_id": real_top,
            "real_w4_top_piece": decode_piece(tokenizer, real_top),
            "baseline_rounded_top_token_id": rounded_top,
            "baseline_int8_top_token_id": baseline_top,
            "baseline_int8_top_piece": decode_piece(tokenizer, baseline_top),
            "guard_top_token_id": guard_top,
            "guard_top_piece": decode_piece(tokenizer, guard_top),
            "reference_rank_in_real_w4": rank_of(real_score, reference_token),
            "reference_rank_in_baseline_rounded": rank_of(
                base_rounded, reference_token
            ),
            "reference_rank_in_baseline_int8": rank_of(base_output, reference_token),
            "reference_rank_in_guard": rank_of(guard_score, reference_token),
            "baseline_top_tie_count": int(
                (base_output == base_output.max()).sum().item()
            ),
            "guard_top_tie_count": int(
                (guard_score == guard_score.max()).sum().item()
            ),
            "baseline_saturation_count": int(base_saturation.sum().item()),
            "guard_saturation_count": int(guard_saturation.sum().item()),
            "real_w4_top_candidates": top_candidates(tokenizer, real_score, TOP_K),
            "baseline_top_candidates": top_candidates(
                tokenizer, base_output, TOP_K
            ),
            "guard_top_candidates": top_candidates(tokenizer, guard_score, TOP_K),
        },
        "candidate_matches_reference": guard_top == reference_token,
        "artifacts": artifacts,
    }
    return record, guard_score, guard_top


def w4a8_trace(
    output: Path,
    snapshot: Path,
    tokenizer: Any,
    reference: dict[str, Any],
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    backend.configure_snapshot(snapshot)
    caches = [backend.empty_layer_cache() for _ in range(backend.LAYERS)]
    templates: list[dict[str, Any] | None] = [None] * backend.LAYERS
    generated: list[int] = []
    steps = []
    scores: list[torch.Tensor] = []
    with torch.no_grad(), safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        print("ACE2_RANK_GUARD_W4_HEAD_QUANTIZE", flush=True)
        head = derive_lm_head_weights(embedding)
        state: dict[str, Any] | None = None
        total_positions = len(CANONICAL_TOKEN_IDS) + CONTINUATION_STEPS - 1
        for position in range(total_positions):
            if position < len(CANONICAL_TOKEN_IDS):
                token_id = CANONICAL_TOKEN_IDS[position]
            else:
                token_id = generated[position - len(CANONICAL_TOKEN_IDS)]
            state = backend.embedding_state(weights, token_id, position)
            for layer_id in range(backend.LAYERS):
                derived, state, templates[layer_id] = backend.derive_layer_token(
                    layer_id,
                    state,
                    caches[layer_id],
                    templates[layer_id],
                    weights,
                    adapter,
                )
                del derived
            if position % 5 == 4 or position == len(CANONICAL_TOKEN_IDS) - 1:
                print(
                    f"ACE2_RANK_GUARD_W4_POSITION position={position} "
                    f"of={total_positions - 1}",
                    flush=True,
                )
            if position >= len(CANONICAL_TOKEN_IDS) - 1:
                step = position - (len(CANONICAL_TOKEN_IDS) - 1)
                reference_token = int(reference["generated_token_ids"][step])
                prefix_match = generated == reference["generated_token_ids"][:step]
                record, guard_score, selected = score_step(
                    output,
                    step,
                    state,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                    reference_token,
                )
                record["prefix_matches_reference_before_step"] = prefix_match
                generated.append(selected)
                steps.append(record)
                scores.append(guard_score)
                print(
                    "ACE2_RANK_GUARD_W4_STEP "
                    f"step={step} baseline={record['lm_head']['baseline_int8_top_token_id']} "
                    f"guard={selected} reference={reference_token} "
                    f"match={selected == reference_token}",
                    flush=True,
                )
        decoded = tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    return (
        {
            "generated_token_ids": generated,
            "decoded_text": decoded,
            "steps": steps,
            "lm_head_weight_quantization": {
                "format": "symmetric signed int4 per output row",
                "qweight_sha256": head["qweight_sha256"],
                "weight_scale_sha256": head["weight_scale_sha256"],
            },
        },
        scores,
    )


def write_rtl_vectors(output: Path, scores: list[torch.Tensor], selected: list[int]) -> dict[str, Any]:
    score_path = output / "rtl/vectors/rank-scores-s16.hex"
    expected_path = output / "rtl/vectors/expected-token-u18.hex"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="ascii") as stream:
        for step_scores in scores:
            for value in step_scores.to(torch.int64).tolist():
                stream.write(f"{int(value) & 0xffff:04x}\n")
    with expected_path.open("w", encoding="ascii") as stream:
        for token_id in selected:
            stream.write(f"{token_id:05x}\n")
    return {"scores": file_record(score_path), "expected": file_record(expected_path)}


def tool_version(command: list[str]) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return result.stdout.splitlines()[0] if result.stdout.splitlines() else "unknown"


def run_rtl(
    output: Path,
    vectors: dict[str, Any],
    *,
    evidence_subdir: str = "rtl",
) -> dict[str, Any]:
    working = output / evidence_subdir
    sim = working / "sim/ace2_lm_head_rank_guard_tb.vvp"
    logs = working / "logs"
    sim.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    compile_command = [
        "iverilog",
        "-g2012",
        "-Wall",
        "-s",
        "ace2_lm_head_rank_guard_tb",
        "-o",
        public_path(sim),
        public_path(RTL_SOURCE),
        public_path(RTL_TB),
    ]
    compile_result = subprocess.run(
        compile_command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    compile_stdout = write_text(logs / "iverilog.stdout.log", compile_result.stdout)
    compile_stderr = write_text(logs / "iverilog.stderr.log", compile_result.stderr)
    require(compile_result.returncode == 0, "Icarus compilation failed")
    require(not compile_result.stderr.strip(), "Icarus emitted warnings")

    run_command = [
        "vvp",
        public_path(sim),
        f"+SCORES={vectors['scores']['path']}",
        f"+EXPECTED={vectors['expected']['path']}",
    ]
    run_result = subprocess.run(
        run_command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    run_stdout = write_text(logs / "vvp.stdout.log", run_result.stdout)
    run_stderr = write_text(logs / "vvp.stderr.log", run_result.stderr)
    require(run_result.returncode == 0, "rank-guard RTL simulation failed")
    require(not run_result.stderr.strip(), "rank-guard RTL simulation emitted stderr")
    require(
        f"ACE2_LM_HEAD_RANK_GUARD_PASS steps=4 tokens_per_step={backend.MODEL_OUTPUT_DOMAIN} xz=0"
        in run_result.stdout,
        "rank-guard RTL PASS marker is absent",
    )
    selected = [
        int(match.group(1))
        for match in re.finditer(
            r"ACE2_LM_HEAD_RANK_GUARD_STEP step=\d+ token=(\d+) score=-?\d+",
            run_result.stdout,
        )
    ]
    require(len(selected) == CONTINUATION_STEPS, "RTL emitted the wrong step count")
    return {
        "status": "PASS_FOCUSED_RTL_RANK_GUARD_X_CLEAN",
        "compile_command": compile_command,
        "run_command": run_command,
        "selected_token_ids": selected,
        "warnings": 0,
        "xz_observations": 0,
        "artifacts": {
            "simulation": file_record(sim),
            "compile_stdout": compile_stdout,
            "compile_stderr": compile_stderr,
            "run_stdout": run_stdout,
            "run_stderr": run_stderr,
        },
    }


def load_tensor(path: Path, dtype: torch.dtype) -> torch.Tensor:
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).clone()


def replay_existing(output: Path) -> int:
    require(output.is_dir(), "existing candidate directory is missing")
    require((output / "candidate-freeze.json").is_file(), "candidate freeze is missing")
    repair = output / "rtl-repair-0001"
    require(not repair.exists(), "rtl-repair-0001 already exists")
    repair.mkdir(parents=True)

    snapshot = generation_runner.resolve_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    reference_ids: list[int] = []
    candidate_ids: list[int] = []
    steps: list[dict[str, Any]] = []
    for step in range(CONTINUATION_STEPS):
        reference_logits = load_tensor(
            output / f"cpu/reference/step-{step:02d}-logits-f32le.bin",
            torch.float32,
        )
        real_score = load_tensor(
            output / f"cpu/w4a8/step-{step:02d}/lm-head-real-score-f64le.bin",
            torch.float64,
        )
        baseline_rounded = load_tensor(
            output / f"cpu/w4a8/step-{step:02d}/baseline-rounded-s64le.bin",
            torch.int64,
        )
        baseline_output = load_tensor(
            output / f"cpu/w4a8/step-{step:02d}/baseline-output-s8.bin",
            torch.int8,
        )
        guard_score = load_tensor(
            output / f"cpu/w4a8/step-{step:02d}/guard-score-s16le.bin",
            torch.int16,
        )
        for name, value in (
            ("reference_logits", reference_logits),
            ("real_score", real_score),
            ("baseline_rounded", baseline_rounded),
            ("baseline_output", baseline_output),
            ("guard_score", guard_score),
        ):
            require(
                value.numel() == backend.MODEL_OUTPUT_DOMAIN,
                f"{name} step {step} has the wrong model domain",
            )
        reference_token = int(torch.argmax(reference_logits).item())
        real_top = int(torch.argmax(real_score).item())
        rounded_top = int(torch.argmax(baseline_rounded).item())
        baseline_top = int(torch.argmax(baseline_output).item())
        guard_top = int(torch.argmax(guard_score).item())
        reference_ids.append(reference_token)
        candidate_ids.append(guard_top)
        steps.append(
            {
                "step": step,
                "prefix_matches_reference_before_step": candidate_ids[:-1]
                == reference_ids[:-1],
                "reference_token_id": reference_token,
                "reference_piece": decode_piece(tokenizer, reference_token),
                "candidate_matches_reference": guard_top == reference_token,
                "lm_head": {
                    "real_w4_top_token_id": real_top,
                    "real_w4_top_piece": decode_piece(tokenizer, real_top),
                    "baseline_rounded_top_token_id": rounded_top,
                    "baseline_int8_top_token_id": baseline_top,
                    "baseline_int8_top_piece": decode_piece(tokenizer, baseline_top),
                    "guard_top_token_id": guard_top,
                    "guard_top_piece": decode_piece(tokenizer, guard_top),
                    "reference_rank_in_real_w4": rank_of(real_score, reference_token),
                    "reference_rank_in_baseline_rounded": rank_of(
                        baseline_rounded, reference_token
                    ),
                    "reference_rank_in_baseline_int8": rank_of(
                        baseline_output, reference_token
                    ),
                    "reference_rank_in_guard": rank_of(guard_score, reference_token),
                    "baseline_top_tie_count": int(
                        (baseline_output == baseline_output.max()).sum().item()
                    ),
                    "guard_top_tie_count": int(
                        (guard_score == guard_score.max()).sum().item()
                    ),
                    "real_w4_top_candidates": top_candidates(
                        tokenizer, real_score, TOP_K
                    ),
                    "baseline_top_candidates": top_candidates(
                        tokenizer, baseline_output, TOP_K
                    ),
                    "guard_top_candidates": top_candidates(
                        tokenizer, guard_score, TOP_K
                    ),
                },
                "artifacts": {
                    "reference_logits": file_record(
                        output / f"cpu/reference/step-{step:02d}-logits-f32le.bin"
                    ),
                    "real_w4_scores": file_record(
                        output / f"cpu/w4a8/step-{step:02d}/lm-head-real-score-f64le.bin"
                    ),
                    "baseline_rounded": file_record(
                        output / f"cpu/w4a8/step-{step:02d}/baseline-rounded-s64le.bin"
                    ),
                    "baseline_output": file_record(
                        output / f"cpu/w4a8/step-{step:02d}/baseline-output-s8.bin"
                    ),
                    "guard_score": file_record(
                        output / f"cpu/w4a8/step-{step:02d}/guard-score-s16le.bin"
                    ),
                    "baseline_multiplier": file_record(
                        output / f"cpu/w4a8/step-{step:02d}/baseline-multiplier-s64le.bin"
                    ),
                    "baseline_shift": file_record(
                        output / f"cpu/w4a8/step-{step:02d}/baseline-shift-s64le.bin"
                    ),
                    "guard_multiplier": file_record(
                        output / f"cpu/w4a8/step-{step:02d}/guard-multiplier-s64le.bin"
                    ),
                    "guard_shift": file_record(
                        output / f"cpu/w4a8/step-{step:02d}/guard-shift-s64le.bin"
                    ),
                },
            }
        )

    vectors = {
        "scores": file_record(output / "rtl/vectors/rank-scores-s16.hex"),
        "expected": file_record(output / "rtl/vectors/expected-token-u18.hex"),
    }
    rtl = run_rtl(output, vectors, evidence_subdir="rtl-repair-0001")
    require(rtl["selected_token_ids"] == candidate_ids, "RTL replay differs from CPU guard")
    localization = localize(steps[0])
    agreement = [
        bool(step["candidate_matches_reference"] and step["prefix_matches_reference_before_step"])
        for step in steps
    ]
    matching_prefix = 0
    for matches in agreement:
        if not matches:
            break
        matching_prefix += 1
    result = {
        "schema_version": 1,
        "status": "PARTIAL_CANDIDATE_UPSTREAM_W4A8_BLOCKER_RTL_REPLAY_PASS",
        "classification": "candidate_only_no_official_attempt_created",
        "localization": localization,
        "repair_evaluated": {
            "boundary": "lm_head_global_requantization",
            "guard_fractional_bits": RANK_GUARD_BITS,
            "rank_score_width": RANK_SCORE_WIDTH,
            "outcome": "NOT_CAUSAL_AT_STEP0",
            "token_specific_overrides": False,
        },
        "reference": {
            "generated_token_ids": reference_ids,
            "decoded_text": tokenizer.decode(
                reference_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        },
        "candidate_w4a8": {
            "generated_token_ids": candidate_ids,
            "decoded_text": tokenizer.decode(
                candidate_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "steps": steps,
        },
        "token_agreement": {
            "per_step": agreement,
            "matching_greedy_prefix_tokens": matching_prefix,
            "required_tokens": CONTINUATION_STEPS,
            "full_four_token_agreement": False,
        },
        "rtl": rtl,
        "initial_rtl_failure": {
            "failure_taxonomy": "verification_harness_domain_and_xcheck_mismatch",
            "root_cause_hypothesis_confirmed": (
                "The testbench used 152064 rows instead of the frozen backend's "
                "151936 rows, and Icarus reported false-positive unknowns for the "
                "$isunknown concatenation."
            ),
            "regression": (
                "Replay all 151936 scores for each of four steps with reduction-X "
                "checks and require warning-clean Icarus plus xz=0."
            ),
            "preserved_logs": {
                "stdout": file_record(output / "rtl/logs/vvp.stdout.log"),
                "stderr": file_record(output / "rtl/logs/vvp.stderr.log"),
            },
        },
        "remaining_blocker": (
            "At step 0 the dequantized W4xA8 LM-head accumulator already ranks token "
            f"{steps[0]['lm_head']['real_w4_top_token_id']} above checkpoint-176 BF16 "
            f"token {steps[0]['reference_token_id']}; the earliest causal boundary is "
            "upstream of LM-head global requantization and still requires checkpoint-176 "
            "layer-by-layer causal substitution."
        ),
        "evidence_hygiene_limit": (
            "The original process crashed before serializing scalar per-step scale "
            "metadata; exact multiplier, shift, accumulator, score, and output tensors "
            "are preserved, but this repair replay does not reconstruct those scalar labels."
        ),
        "sources_after_rtl_repair": {
            "runner": file_record(Path(__file__).resolve()),
            "rank_guard_rtl": file_record(RTL_SOURCE),
            "rank_guard_tb": file_record(RTL_TB),
        },
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "ppa_claimed": False,
        "independent_review_required": True,
    }
    result_path = repair / "result.json"
    write_json(result_path, result)
    members = sorted(
        path for path in repair.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    write_text(
        repair / "SHA256SUMS",
        "".join(
            f"{sha256_file(path)}  {path.relative_to(repair).as_posix()}\n"
            for path in members
        ),
    )
    print(
        "ACE2_RANK_GUARD_REPLAY_RESULT "
        f"status={result['status']} reference={reference_ids} candidate={candidate_ids} "
        f"output={public_path(repair)}",
        flush=True,
    )
    return 4


def localize(step0: dict[str, Any]) -> dict[str, Any]:
    head = step0["lm_head"]
    reference = int(step0["reference_token_id"])
    if head["real_w4_top_token_id"] != reference:
        return {
            "localized": False,
            "earliest_boundary": "before_lm_head_global_requantization",
            "failure_class": "upstream_w4a8_weight_or_activation_ordering_loss",
            "evidence": (
                "The intended token is not rank 1 in dequantized W4xA8 accumulator "
                "scores, so an LM-head output-scale repair cannot be causal."
            ),
        }
    if head["baseline_rounded_top_token_id"] != reference:
        return {
            "localized": True,
            "earliest_boundary": "lm_head_global_int8_requantization_rounding",
            "failure_class": "common_scale_resolution_loss",
            "evidence": (
                "Dequantized W4xA8 accumulator scores preserve the BF16 top token, "
                "but rounding into the baseline global int8 logit scale changes rank 1."
            ),
        }
    if head["baseline_int8_top_token_id"] != reference:
        return {
            "localized": True,
            "earliest_boundary": "lm_head_int8_saturation",
            "failure_class": "saturation_tie",
            "evidence": (
                "The pre-clamp fixed-point score preserves rank 1 and the signed-int8 "
                "clamp changes it."
            ),
        }
    return {
        "localized": False,
        "earliest_boundary": "no_step0_inversion_reproduced",
        "failure_class": "baseline_not_reproduced",
        "evidence": "The fresh CPU baseline did not reproduce a step-0 inversion.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-existing", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    forbidden = FORBIDDEN_OUTPUT.resolve()
    require(output != forbidden and forbidden not in output.parents, "candidate output overlaps official evidence")
    if args.replay_existing:
        return replay_existing(output)
    require(not output.exists(), "candidate output directory already exists")
    output.mkdir(parents=True)

    snapshot = generation_runner.resolve_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    observed_tokens = generation_runner.canonical_chat_token_ids(tokenizer, CANONICAL_PROMPT)
    require(observed_tokens == CANONICAL_TOKEN_IDS, "canonical chat prefix changed")
    require(len(tokenizer) <= backend.MODEL_OUTPUT_DOMAIN, "tokenizer exceeds model domain")

    freeze = {
        "schema_version": 1,
        "classification": "candidate_only_no_official_run",
        "checkpoint": {
            "name": "checkpoint-176",
            "tree_sha256": backend.canonical.CHECKPOINT_TREE_SHA256,
            "adapter_model_sha256": sha256_file(backend.canonical.ADAPTER),
            "adapter_config_sha256": sha256_file(backend.canonical.ADAPTER_CONFIG),
        },
        "base_model": {
            "repository": generation_runner.MODEL_REPOSITORY,
            "revision": generation_runner.REVISION,
            "model_sha256": sha256_file(snapshot / "model.safetensors"),
        },
        "tokenizer": {
            "chat_template_sha256": generation_runner.CHAT_TEMPLATE_SHA256,
            "tokenizer_json_sha256": sha256_file(snapshot / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(snapshot / "tokenizer_config.json"),
            "length": len(tokenizer),
        },
        "prompt": {
            "utf8_hex": CANONICAL_PROMPT.encode("utf-8").hex(),
            "sha256": sha256_bytes(CANONICAL_PROMPT.encode("utf-8")),
            "chat_token_ids": CANONICAL_TOKEN_IDS,
        },
        "decoding": {
            "strategy": "greedy_stable_low_token_id_tie_break",
            "continuation_tokens": CONTINUATION_STEPS,
            "use_cache_reference": True,
            "w4a8_cache_semantics": "accepted_sequence_backend",
        },
        "baseline": {
            "weights": "symmetric_signed_w4_per_output_row",
            "activations": "symmetric_signed_a8",
            "lm_head_output": "global_signed_int8_scale",
        },
        "candidate_repair": {
            "boundary": "lm_head_rank_requantization",
            "guard_fractional_bits": RANK_GUARD_BITS,
            "rank_score_width": RANK_SCORE_WIDTH,
            "weights_and_activations_unchanged": True,
            "token_specific_overrides": False,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
            "iverilog": tool_version(["iverilog", "-V"]),
            "vvp": tool_version(["vvp", "-V"]),
        },
        "sources": {
            "candidate_runner": file_record(Path(__file__).resolve()),
            "sequence_backend": file_record(ROOT / "tools/rtl_arbitrary_text_generation_backend.py"),
            "projection_reference": file_record(ROOT / "tools/run_lora_v4_layer0_full_rtl.py"),
            "rank_guard_rtl": file_record(RTL_SOURCE),
            "rank_guard_tb": file_record(RTL_TB),
        },
    }
    write_json(output / "candidate-freeze.json", freeze)

    reference = reference_trace(output, snapshot, tokenizer)
    w4a8, score_vectors = w4a8_trace(output, snapshot, tokenizer, reference)
    selected = [int(item) for item in w4a8["generated_token_ids"]]
    vectors = write_rtl_vectors(output, score_vectors, selected)
    rtl = run_rtl(output, vectors)
    require(rtl["selected_token_ids"] == selected, "RTL selector differs from CPU guard scores")

    localization = localize(w4a8["steps"][0])
    agreement = [
        bool(step["candidate_matches_reference"] and step["prefix_matches_reference_before_step"])
        for step in w4a8["steps"]
    ]
    matching_prefix = 0
    for matches in agreement:
        if not matches:
            break
        matching_prefix += 1
    passed = bool(localization["localized"] and matching_prefix >= CONTINUATION_STEPS)
    blocker = None
    if not localization["localized"]:
        blocker = localization["evidence"]
    elif matching_prefix < CONTINUATION_STEPS:
        failed_step = matching_prefix
        step = w4a8["steps"][failed_step]
        blocker = (
            f"step {failed_step} guard rank selected token "
            f"{step['lm_head']['guard_top_token_id']} while checkpoint-176 BF16 selected "
            f"{step['reference_token_id']}"
        )

    result = {
        "schema_version": 1,
        "status": (
            "PASS_CANDIDATE_FOUR_TOKEN_RANK_GUARD_CPU_RTL"
            if passed
            else "PARTIAL_CANDIDATE_RANK_GUARD"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "localization": localization,
        "repair": {
            "description": (
                "Retain eight fractional rank bits at the LM-head common-scale "
                "requantization boundary and stream signed-16 scores into a stable "
                "argmax; W4 weights, A8 activations, accumulators, and token domain "
                "remain unchanged."
            ),
            "guard_fractional_bits": RANK_GUARD_BITS,
            "rank_score_width": RANK_SCORE_WIDTH,
            "token_specific_overrides": False,
        },
        "reference": reference,
        "candidate_w4a8": w4a8,
        "token_agreement": {
            "per_step": agreement,
            "matching_greedy_prefix_tokens": matching_prefix,
            "required_tokens": CONTINUATION_STEPS,
            "full_four_token_agreement": matching_prefix >= CONTINUATION_STEPS,
        },
        "rtl": rtl,
        "vectors": vectors,
        "remaining_blocker": blocker,
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "ppa_claimed": False,
        "independent_review_required": True,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    write_json(output / "result.json", result)
    manifest_members = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
        for path in manifest_members
    )
    write_text(output / "SHA256SUMS", sums)
    print(
        "ACE2_RANK_GUARD_RESULT "
        f"status={result['status']} matching_prefix={matching_prefix} "
        f"reference={reference['generated_token_ids']} candidate={selected} "
        f"output={public_path(output)}",
        flush=True,
    )
    return 0 if passed else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_RANK_GUARD_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
