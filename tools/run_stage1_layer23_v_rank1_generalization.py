#!/usr/bin/env python3
"""Run the frozen rank-1 layer-23 V correction on unseen prompts."""

from __future__ import annotations

import argparse
import copy
import gc
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
from peft import PeftModel
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import diagnose_stage1_layer23_v_residual_structure as residual
from tools import run_stage1_layer23_v_rank1_integer_candidate as candidate


prior = residual.diagnosis.prior

MISSION_ID = "stage1vrank1int-generalization01"
LAYER = 23
TOP_K = 8
DECODE_STEPS = 2
OUTPUT = (
    ROOT
    / "evidence/candidates/stage1-layer23-v-rank1-integer-correction-v1/"
    "nonofficial-generalization-0001"
)
FREEZE = OUTPUT / "generalization-freeze.json"
RESULT = OUTPUT / "result.json"
REVIEW_REQUEST = OUTPUT / "fresh-l2-review-request.json"
SUMS = OUTPUT / "SHA256SUMS"
EXECUTION_RUNNER = OUTPUT / "rank1-generalization-execution-runner.py"

ACCEPTED_RTL_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/dc94c6bf8107/round-0001.json"
)
ACCEPTED_RTL_REVIEW_SHA256 = (
    "946d3d03a7ad9d52658f1d1570a3e1e0d22ad77267af1a69dd306d040a068097"
)
ACCEPTED_RTL_RESULT_SHA256 = (
    "1a19b068aaf6824ae29bf0ed4710c0015cb617c3b7fb62ead35f96d17ae4f77f"
)
CANDIDATE_FREEZE_SHA256 = (
    "c00e607a567ebc1f02854e9763f9722eeb985bfaf97962401f48414c73dfd0f4"
)
CANDIDATE_RESULT_SHA256 = (
    "fa4547a5884a070966a5c0c1fea8997b2aef51d22d40431df34f719cce6ca037"
)
CANDIDATE_SUMS_SHA256 = (
    "885cc74b701604626a8d4d85c763fbbccab4589241850689e88b9692e61189db"
)

PROMPTS = (
    {
        "prompt_id": "concise-factual-hbm",
        "category": "concise_factual",
        "text": "Define HBM briefly.",
    },
    {
        "prompt_id": "instruction-ready-valid",
        "category": "instruction",
        "text": "Two ready/valid checks?",
    },
    {
        "prompt_id": "code-reasoning-arithmetic",
        "category": "code_reasoning",
        "text": "3*3+4*4 equals?",
    },
    {
        "prompt_id": "conversational-debugger",
        "category": "conversational",
        "text": "Cheer up a tired debugger.",
    },
)

SCOPE_GUARDS = {
    "attempt_0002_replay": False,
    "attempt_0003": False,
    "factor_refit_or_rescale": False,
    "held_out_feedback_tuning": False,
    "official_preflight_or_run": False,
    "rtl_modified_or_simulated": False,
    "synthesis_or_ppa": False,
    "u280_or_xrt": False,
}

DECISION_POLICY = {
    "fixed_eval_steps": list(range(DECODE_STEPS)),
    "prompt_improvement": (
        "A prompt improves only if aggregate corrected V relative-L2 across the "
        "fixed BF16-greedy teacher-forced decode contexts is lower than baseline "
        "and at least one aggregate logit metric does not regress "
        "(mean JSD, mean reference-token rank, or mean top-8 overlap)."
    ),
    "clear_majority": "at least 3 of 4 preregistered prompts improve",
    "catastrophic_regression": (
        "Any fixed evaluation record is catastrophic if corrected V relative-L2 "
        "exceeds 1.25x baseline, JSD exceeds baseline by more than 0.25 nats, "
        "or reference-token rank is worse than both 4x baseline and baseline+32."
    ),
    "positive_result_scope": (
        "A positive nonofficial result may only request Fresh-L2 review and a "
        "separate descriptor-bound integration task; this runner never promotes "
        "or tunes the factors."
    ),
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
    return prior.localizer.file_record(path)


def sha256_file(path: Path) -> str:
    return prior.localizer.sha256_file(path)


def sha256_bytes(raw: bytes) -> str:
    return prior.localizer.sha256_bytes(raw)


def write_tensor(path: Path, value: torch.Tensor, dtype: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = value.detach().cpu().contiguous()
    if dtype == "float32_le":
        raw = tensor.to(torch.float32).numpy().astype("<f4").tobytes()
    elif dtype == "float64_le":
        raw = tensor.to(torch.float64).numpy().astype("<f8").tobytes()
    elif dtype == "int8":
        raw = tensor.to(torch.int8).numpy().astype("i1").tobytes()
    elif dtype == "int32_le":
        raw = tensor.to(torch.int32).numpy().astype("<i4").tobytes()
    else:
        raise ValueError(f"unsupported tensor dtype {dtype}")
    path.write_bytes(raw)
    record = prior.localizer.file_record(path)
    record.update({"dtype": dtype, "shape": list(tensor.shape)})
    return record


def load_tensor(record: dict[str, Any]) -> torch.Tensor:
    dtype_map = {
        "float32_le": np.dtype("<f4"),
        "float64_le": np.dtype("<f8"),
        "int8": np.dtype("i1"),
        "int32_le": np.dtype("<i4"),
    }
    path = ROOT / record["path"]
    values = np.fromfile(path, dtype=dtype_map[record["dtype"]]).copy()
    return torch.from_numpy(values).reshape(record["shape"])


def generate_sums() -> None:
    paths = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path != SUMS
    )
    body = "".join(
        f"{sha256_file(path)}  {path.relative_to(OUTPUT).as_posix()}\n"
        for path in paths
    )
    SUMS.write_text(body, encoding="ascii")


def verify_sum_tree() -> None:
    require(SUMS.is_file(), "generalization SHA256SUMS is missing")
    for line in SUMS.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = OUTPUT / relative
        require(path.is_file(), f"summed artifact missing: {relative}")
        require(sha256_file(path) == expected, f"summed artifact changed: {relative}")


def external_file_record(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required binding is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_candidate_sums() -> None:
    require(
        sha256_file(candidate.SUMS) == CANDIDATE_SUMS_SHA256,
        "candidate SHA256SUMS changed",
    )
    for line in candidate.SUMS.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = candidate.OUTPUT / relative
        require(path.is_file(), f"candidate artifact missing: {relative}")
        require(sha256_file(path) == expected, f"candidate artifact changed: {relative}")


def validate_authority() -> dict[str, Any]:
    review = external_file_record(ACCEPTED_RTL_REVIEW)
    require(
        review["sha256"] == ACCEPTED_RTL_REVIEW_SHA256,
        "Fresh-L2 RTL evidence review hash changed",
    )
    review_json = read_json(ACCEPTED_RTL_REVIEW)
    require(
        review_json["producer_role"] == "reviewer"
        and review_json["review"]["status"] == "done"
        and ACCEPTED_RTL_RESULT_SHA256 in review_json["review"]["reason"],
        "Fresh-L2 did not accept the bound RTL/reference evidence",
    )
    require(
        sha256_file(candidate.FREEZE) == CANDIDATE_FREEZE_SHA256,
        "candidate freeze hash changed",
    )
    require(
        sha256_file(candidate.RESULT) == CANDIDATE_RESULT_SHA256,
        "candidate result hash changed",
    )
    require(
        sha256_file(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-integer-correction-rtl-v1/"
            "latest/RESULT.json"
        )
        == ACCEPTED_RTL_RESULT_SHA256,
        "accepted RTL/reference RESULT hash changed",
    )
    verify_candidate_sums()
    candidate.verify()
    require(
        not (
            ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0003"
        ).exists(),
        "attempt-0003 exists",
    )
    return {
        "accepted_fresh_l2_review": review,
        "accepted_rtl_reference_result": prior.localizer.file_record(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-integer-correction-rtl-v1/"
            "latest/RESULT.json"
        ),
        "candidate_freeze": prior.localizer.file_record(candidate.FREEZE),
        "candidate_result": prior.localizer.file_record(candidate.RESULT),
        "candidate_sums": prior.localizer.file_record(candidate.SUMS),
    }


def prompt_manifest(tokenizer: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    categories = {item["category"] for item in PROMPTS}
    require(
        categories
        == {"concise_factual", "instruction", "code_reasoning", "conversational"},
        "prompt category coverage changed",
    )
    for index, item in enumerate(PROMPTS):
        prompt = item["text"]
        token_ids = prior.generation_runner.canonical_chat_token_ids(tokenizer, prompt)
        prompt_hash = sha256_bytes(prompt.encode("utf-8"))
        require(prompt_hash not in seen_hashes, "duplicate prompt hash")
        seen_hashes.add(prompt_hash)
        require(
            len(token_ids) + DECODE_STEPS - 1 <= prior.backend.MAX_CONTEXT_TOKENS,
            f"{item['prompt_id']} exceeds backend context bound",
        )
        records.append(
            {
                "index": index,
                "prompt_id": item["prompt_id"],
                "category": item["category"],
                "prompt_sha256": prompt_hash,
                "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                "prompt_text_persisted_in_freeze": False,
                "token_count": len(token_ids),
                "token_ids_sha256": sha256_bytes(canonical_bytes(token_ids)),
                "max_context_tokens_with_decode": len(token_ids) + DECODE_STEPS - 1,
            }
        )
    return records


def build_freeze(tokenizer: Any) -> dict[str, Any]:
    authority = validate_authority()
    return {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": "FROZEN_PENDING_EVALUATION",
        "classification": (
            "bounded_nonofficial_software_first_multi_prompt_generalization"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": {
            "basis": (
                "Supervisor conditional next step after Fresh-L2 accepted RESULT.json "
                f"SHA-256 {ACCEPTED_RTL_RESULT_SHA256}"
            ),
            "official_execution_authorized": False,
            "rtl_modification_authorized": False,
            "synthesis_ppa_authorized": False,
            "u280_xrt_authorized": False,
        },
        "prompt_manifest": prompt_manifest(tokenizer),
        "evaluation_policy": {
            "decode_steps": DECODE_STEPS,
            "fixed_eval_steps": list(range(DECODE_STEPS)),
            "decode_context": (
                "teacher-forced on BF16 greedy tokens; baseline and frozen correction "
                "are scored on the same context at each fixed step"
            ),
            "selection": "greedy argmax over 151936-token logits; sampling disabled",
            "top_k": TOP_K,
            "layer_cut": LAYER,
            "component": "self_attn.v_proj",
            "factor_update_policy": "no refit, no rescale, no prompt-specific tuning",
        },
        "decision_policy": copy.deepcopy(DECISION_POLICY),
        "bindings": {
            **authority,
            "runner": prior.localizer.file_record(Path(__file__).resolve()),
            "base_model": {
                "repository": prior.generation_runner.MODEL_REPOSITORY,
                "revision": prior.generation_runner.REVISION,
                "model_sha256": prior.localizer.sha256_file(
                    prior.backend.canonical.MODEL
                ),
            },
            "checkpoint_176_adapter": {
                "adapter_model_sha256": prior.localizer.sha256_file(
                    prior.backend.canonical.ADAPTER
                ),
                "adapter_config_sha256": prior.localizer.sha256_file(
                    prior.backend.canonical.ADAPTER_CONFIG
                ),
                "checkpoint_tree_sha256": prior.backend.canonical.CHECKPOINT_TREE_SHA256,
            },
        },
        "scope_guards": copy.deepcopy(SCOPE_GUARDS),
    }


def validate_freeze(freeze: dict[str, Any], tokenizer: Any) -> None:
    require(freeze["mission_id"] == MISSION_ID, "freeze mission changed")
    require(
        freeze["classification"]
        == "bounded_nonofficial_software_first_multi_prompt_generalization",
        "freeze classification changed",
    )
    require(freeze["scope_guards"] == SCOPE_GUARDS, "freeze scope guards changed")
    require(freeze["decision_policy"] == DECISION_POLICY, "decision policy changed")
    require(
        freeze["evaluation_policy"]["fixed_eval_steps"] == list(range(DECODE_STEPS)),
        "fixed evaluation positions changed",
    )
    require(
        freeze["prompt_manifest"] == prompt_manifest(tokenizer),
        "preregistered prompt manifest changed",
    )
    validate_authority()
    require(
        freeze["bindings"]["runner"]
        == prior.localizer.file_record(Path(__file__).resolve()),
        "runner binding changed after preregistration",
    )


def preregister(tokenizer: Any) -> dict[str, Any]:
    if FREEZE.exists():
        freeze = read_json(FREEZE)
        validate_freeze(freeze, tokenizer)
        return freeze
    require(not RESULT.exists(), "result exists before preregistration")
    require(not SUMS.exists(), "SHA256SUMS exists before preregistration")
    value = build_freeze(tokenizer)
    write_json(FREEZE, value)
    print(
        "ACE2_RANK1_GENERALIZATION_FREEZE "
        f"sha256={sha256_file(FREEZE)} output={prior.localizer.public_path(OUTPUT)}",
        flush=True,
    )
    return value


def capture_layer23_context(
    model: Any, token_ids: list[int]
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    layer = model.model.model.layers[LAYER]
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def capture_layer_input(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        require(bool(inputs) and isinstance(inputs[0], torch.Tensor), "no layer input")
        captures["layer_input"] = inputs[0][0].float().cpu().contiguous()

    def capture_output(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            require(isinstance(value, torch.Tensor), f"missing BF16 {name}")
            captures[name] = value[0].float().cpu().contiguous()

        return hook

    handles.append(layer.register_forward_pre_hook(capture_layer_input))
    handles.append(layer.input_layernorm.register_forward_hook(capture_output("input_norm")))
    handles.append(layer.self_attn.v_proj.register_forward_hook(capture_output("v")))
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()
    logits = output.logits[0, -1].float().cpu().contiguous()
    require(
        set(captures) == {"layer_input", "input_norm", "v"},
        "BF16 capture set changed",
    )
    require(
        all(value.shape[0] == len(token_ids) for value in captures.values()),
        "BF16 capture token count changed",
    )
    layer_input = captures.pop("layer_input")
    del output, input_ids
    return layer_input, captures, logits


class VariableResidualCorrectionCache(prior.localizer.FastProjectionCache):
    def __init__(self, weights: Any, adapter: Any, token_count: int) -> None:
        super().__init__(weights, adapter)
        self.token_count = token_count
        self.correction: torch.Tensor | None = None
        self.records: list[dict[str, Any]] = []

    def configure(self, correction: torch.Tensor | None, token_count: int) -> None:
        require(token_count == self.token_count, "token count changed during run")
        if correction is not None:
            require(
                list(correction.shape) == [token_count, 128],
                "residual correction shape changed",
            )
        self.correction = correction
        self.records = []

    def _apply(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
        match = residual.joint.V_PATTERN.fullmatch(name)
        if match is None:
            return result
        position = int(match.group("position"))
        require(0 <= position < self.token_count, "V position outside context")
        output_scale = float(result["output_scale"])
        baseline = result["output_q"].to(torch.float64) * output_scale
        correction = (
            torch.zeros_like(baseline)
            if self.correction is None
            else self.correction[position].to(torch.float64)
        )
        if self.correction is not None:
            result["output_q"] = prior.backend.canonical.quantize_int8(
                (baseline + correction).to(torch.float32), output_scale
            )
        corrected = result["output_q"].to(torch.float64) * output_scale
        self.records.append(
            {
                "position": position,
                "input_dequantized": (
                    result["input_q"].to(torch.float64) * float(result["input_scale"])
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
            name, merged, input_q, input_scale, float_input, source_hashes
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
            name, input_q, input_scale, qweight, multiplier, right_shift, output_scale
        )
        return self._apply(name, result)

    def stacked(self) -> dict[str, torch.Tensor | float]:
        require(len(self.records) == self.token_count, "V path record count changed")
        require(
            [int(item["position"]) for item in self.records] == list(range(self.token_count)),
            "V positions are not ordered",
        )
        scales = {float(item["output_scale"]) for item in self.records}
        require(len(scales) == 1, "V A8 output scale changed by position")
        return {
            key: torch.stack([item[key] for item in self.records])
            for key in (
                "input_dequantized",
                "baseline_output",
                "predicted_correction",
                "corrected_output",
                "output_q",
            )
        } | {"output_scale": scales.pop()}


def error_summary(candidate_tensor: torch.Tensor, reference_tensor: torch.Tensor) -> dict[str, Any]:
    candidate_f64 = candidate_tensor.to(torch.float64)
    reference_f64 = reference_tensor.to(torch.float64)
    difference = candidate_f64 - reference_f64
    error_energy = float(torch.sum(difference * difference).item())
    reference_energy = float(torch.sum(reference_f64 * reference_f64).item())
    require(reference_energy > 0.0, "relative-L2 denominator is zero")
    return {
        "count": int(difference.numel()),
        "squared_error_sum": error_energy,
        "reference_energy": reference_energy,
        "relative_l2": math.sqrt(error_energy / reference_energy),
        "rmse": float(torch.sqrt(torch.mean(difference * difference)).item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


def combine_error_summaries(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(item["count"]) for item in items)
    error_energy = sum(float(item["squared_error_sum"]) for item in items)
    reference_energy = sum(float(item["reference_energy"]) for item in items)
    require(count > 0 and reference_energy > 0.0, "empty aggregate error summary")
    return {
        "count": count,
        "squared_error_sum": error_energy,
        "reference_energy": reference_energy,
        "relative_l2": math.sqrt(error_energy / reference_energy),
    }


def score_metrics(record: dict[str, Any], reference_token: int) -> dict[str, Any]:
    comparison = record["comparison_vs_checkpoint176_bf16"]
    rank = int(comparison["reference_top_rank_in_candidate"])
    top_token = int(record["top_token_id"])
    return {
        "reference_token_id": int(reference_token),
        "reference_token_rank": rank,
        "reference_token_top1": rank == 1 and top_token == int(reference_token),
        "top_token_id": top_token,
        "top8_overlap_count": int(comparison["top8_overlap_count"]),
        "jensen_shannon_divergence_nats": float(
            comparison["jensen_shannon_divergence_nats"]
        ),
        "relative_l2": float(comparison["raw_score_error"]["relative_l2"]),
        "comparison_vs_bf16_oracle": comparison,
    }


def mean(values: list[float]) -> float:
    require(bool(values), "empty mean")
    return float(sum(values) / len(values))


def run_cut(
    cache: VariableResidualCorrectionCache,
    correction: torch.Tensor | None,
    token_ids: list[int],
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
    bf16_logits: torch.Tensor,
    label: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | float]]:
    cache.configure(correction, len(token_ids))
    record, _ = prior.run_cut_record(
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
    return record, cache.stacked()


def tensor_artifacts(
    step_dir: Path,
    *,
    bf16_logits: torch.Tensor,
    bf16_v: torch.Tensor,
    input_dequantized: torch.Tensor,
    baseline_v: torch.Tensor,
    correction: torch.Tensor,
    corrected_v: torch.Tensor,
) -> dict[str, Any]:
    return {
        "bf16_logits": write_tensor(step_dir / "bf16-logits-f32le.bin", bf16_logits, "float32_le"),
        "bf16_v_output": write_tensor(step_dir / "bf16-layer23-v-f32le.bin", bf16_v, "float32_le"),
        "input_dequantized": write_tensor(
            step_dir / "v-input-dequantized-f64le.bin",
            input_dequantized,
            "float64_le",
        ),
        "baseline_v_output": write_tensor(
            step_dir / "baseline-v-output-f64le.bin", baseline_v, "float64_le"
        ),
        "predicted_residual": write_tensor(
            step_dir / "rank1-predicted-residual-f64le.bin",
            correction,
            "float64_le",
        ),
        "corrected_v_output": write_tensor(
            step_dir / "rank1-corrected-v-output-f64le.bin",
            corrected_v,
            "float64_le",
        ),
    }


def evaluate_contexts(
    *,
    freeze: dict[str, Any],
    tokenizer: Any,
    contexts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    suppressed_messages: list[str] = []
    prior_output = prior.OUTPUT
    try:
        with safe_open(
            prior.backend.canonical.MODEL, framework="pt", device="cpu"
        ) as weights, safe_open(
            prior.backend.canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
            norm_gain = weights.get_tensor("model.norm.weight").contiguous()
            head = prior.localizer.derive_lm_head_weights(embedding)
            guard = prior.localizer.SubstitutionRequireGuard()
            guard.install()
            try:
                for context in contexts:
                    token_ids = context["token_ids"]
                    reference_token = int(context["reference_token_id"])
                    prior.localizer.TOKEN_IDS = token_ids
                    prior.localizer.REFERENCE_TOKEN = reference_token
                    prior.REFERENCE_TOKEN = reference_token
                    prompt_dir = (
                        OUTPUT
                        / "prompts"
                        / context["prompt_id"]
                        / f"step-{context['decode_step']:02d}"
                    )
                    prior.OUTPUT = prompt_dir
                    hidden_states = [torch.empty(0)] * LAYER + [context["layer_input"]]
                    cache = VariableResidualCorrectionCache(weights, adapter, len(token_ids))
                    cache.install()
                    try:
                        baseline_record, baseline_state = run_cut(
                            cache,
                            None,
                            token_ids,
                            hidden_states,
                            weights,
                            adapter,
                            norm_gain,
                            embedding,
                            head,
                            tokenizer,
                            context["bf16_logits"],
                            "baseline-w4a8",
                        )
                        integer_evidence, correction, corrected = candidate.integer_arithmetic(
                            freeze=read_json(candidate.FREEZE),
                            input_dequantized=baseline_state["input_dequantized"].to(torch.float64),
                            baseline_output=baseline_state["baseline_output"].to(torch.float64),
                            write_artifacts=False,
                        )
                        corrected_record, corrected_state = run_cut(
                            cache,
                            correction,
                            token_ids,
                            hidden_states,
                            weights,
                            adapter,
                            norm_gain,
                            embedding,
                            head,
                            tokenizer,
                            context["bf16_logits"],
                            "rank1-integer-s8",
                        )
                    finally:
                        cache.restore()

                    require(
                        torch.equal(
                            corrected_state["corrected_output"].to(torch.float64),
                            corrected,
                        ),
                        "corrected V replay differs from frozen integer arithmetic",
                    )
                    artifacts = tensor_artifacts(
                        prompt_dir,
                        bf16_logits=context["bf16_logits"],
                        bf16_v=context["captures"]["v"],
                        input_dequantized=baseline_state["input_dequantized"].to(torch.float64),
                        baseline_v=baseline_state["baseline_output"].to(torch.float64),
                        correction=correction,
                        corrected_v=corrected,
                    )
                    baseline_v_error = error_summary(
                        baseline_state["baseline_output"].to(torch.float64),
                        context["captures"]["v"].to(torch.float64),
                    )
                    corrected_v_error = error_summary(
                        corrected,
                        context["captures"]["v"].to(torch.float64),
                    )
                    baseline_metrics = score_metrics(baseline_record, reference_token)
                    corrected_metrics = score_metrics(corrected_record, reference_token)
                    record = {
                        "prompt_id": context["prompt_id"],
                        "category": context["category"],
                        "decode_step": int(context["decode_step"]),
                        "context_token_count": len(token_ids),
                        "reference_token_id": reference_token,
                        "reference_piece": prior.localizer.decode_piece(tokenizer, reference_token),
                        "baseline": {
                            "metrics": baseline_metrics,
                            "logits": baseline_record["accepted_w4a8_logits"],
                            "v_relative_l2": baseline_v_error,
                            "top_piece": baseline_record["top_piece"],
                        },
                        "rank1_integer": {
                            "metrics": corrected_metrics,
                            "logits": corrected_record["accepted_w4a8_logits"],
                            "v_relative_l2": corrected_v_error,
                            "integer_evidence": integer_evidence,
                            "top_piece": corrected_record["top_piece"],
                        },
                        "artifacts": artifacts,
                        "improvement_flags": {
                            "v_relative_l2_improved": (
                                corrected_v_error["relative_l2"]
                                < baseline_v_error["relative_l2"]
                            ),
                            "reference_rank_improved_or_equal": (
                                corrected_metrics["reference_token_rank"]
                                <= baseline_metrics["reference_token_rank"]
                            ),
                            "top8_overlap_improved_or_equal": (
                                corrected_metrics["top8_overlap_count"]
                                >= baseline_metrics["top8_overlap_count"]
                            ),
                            "jsd_improved_or_equal": (
                                corrected_metrics["jensen_shannon_divergence_nats"]
                                <= baseline_metrics["jensen_shannon_divergence_nats"]
                            ),
                        },
                    }
                    record["catastrophic_regression"] = catastrophic(record)
                    records.append(record)
                    print(
                        "ACE2_RANK1_GENERALIZATION_STEP "
                        f"prompt={context['prompt_id']} step={context['decode_step']} "
                        f"base_v={baseline_v_error['relative_l2']:.6f} "
                        f"rank1_v={corrected_v_error['relative_l2']:.6f} "
                        f"base_rank={baseline_metrics['reference_token_rank']} "
                        f"rank1_rank={corrected_metrics['reference_token_rank']} "
                        f"base_jsd={baseline_metrics['jensen_shannon_divergence_nats']:.6f} "
                        f"rank1_jsd={corrected_metrics['jensen_shannon_divergence_nats']:.6f}",
                        flush=True,
                    )
            finally:
                suppressed_messages.extend(guard.suppressed)
                guard.restore()
    finally:
        prior.OUTPUT = prior_output
    return records, suppressed_messages


def catastrophic(record: dict[str, Any]) -> bool:
    baseline_v = float(record["baseline"]["v_relative_l2"]["relative_l2"])
    candidate_v = float(record["rank1_integer"]["v_relative_l2"]["relative_l2"])
    baseline_metrics = record["baseline"]["metrics"]
    candidate_metrics = record["rank1_integer"]["metrics"]
    baseline_rank = int(baseline_metrics["reference_token_rank"])
    candidate_rank = int(candidate_metrics["reference_token_rank"])
    return (
        candidate_v > 1.25 * baseline_v
        or candidate_metrics["jensen_shannon_divergence_nats"]
        > baseline_metrics["jensen_shannon_divergence_nats"] + 0.25
        or (
            candidate_rank > 4 * baseline_rank
            and candidate_rank > baseline_rank + 32
        )
    )


def prompt_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_id = records[0]["prompt_id"]
    category = records[0]["category"]
    baseline_v = combine_error_summaries(
        [record["baseline"]["v_relative_l2"] for record in records]
    )
    corrected_v = combine_error_summaries(
        [record["rank1_integer"]["v_relative_l2"] for record in records]
    )
    baseline_jsd = mean(
        [
            record["baseline"]["metrics"]["jensen_shannon_divergence_nats"]
            for record in records
        ]
    )
    corrected_jsd = mean(
        [
            record["rank1_integer"]["metrics"]["jensen_shannon_divergence_nats"]
            for record in records
        ]
    )
    baseline_rank = mean(
        [
            float(record["baseline"]["metrics"]["reference_token_rank"])
            for record in records
        ]
    )
    corrected_rank = mean(
        [
            float(record["rank1_integer"]["metrics"]["reference_token_rank"])
            for record in records
        ]
    )
    baseline_top8 = mean(
        [
            float(record["baseline"]["metrics"]["top8_overlap_count"])
            for record in records
        ]
    )
    corrected_top8 = mean(
        [
            float(record["rank1_integer"]["metrics"]["top8_overlap_count"])
            for record in records
        ]
    )
    no_catastrophic = not any(record["catastrophic_regression"] for record in records)
    prompt_improved = (
        corrected_v["relative_l2"] < baseline_v["relative_l2"]
        and no_catastrophic
        and (
            corrected_jsd <= baseline_jsd
            or corrected_rank <= baseline_rank
            or corrected_top8 >= baseline_top8
        )
    )
    return {
        "prompt_id": prompt_id,
        "category": category,
        "fixed_eval_steps": [int(record["decode_step"]) for record in records],
        "baseline_v_relative_l2": baseline_v,
        "rank1_v_relative_l2": corrected_v,
        "mean_baseline_jsd": baseline_jsd,
        "mean_rank1_jsd": corrected_jsd,
        "mean_baseline_reference_rank": baseline_rank,
        "mean_rank1_reference_rank": corrected_rank,
        "mean_baseline_top8_overlap": baseline_top8,
        "mean_rank1_top8_overlap": corrected_top8,
        "catastrophic_regression": not no_catastrophic,
        "prompt_improved": prompt_improved,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_prompt.setdefault(record["prompt_id"], []).append(record)
    summaries = [
        prompt_summary(sorted(items, key=lambda item: int(item["decode_step"])))
        for _prompt_id, items in sorted(by_prompt.items())
    ]
    improved = [item for item in summaries if item["prompt_improved"]]
    catastrophic_count = sum(
        1 for record in records if bool(record["catastrophic_regression"])
    )
    aggregate_baseline_v = combine_error_summaries(
        [record["baseline"]["v_relative_l2"] for record in records]
    )
    aggregate_corrected_v = combine_error_summaries(
        [record["rank1_integer"]["v_relative_l2"] for record in records]
    )
    accepted = len(improved) >= 3 and catastrophic_count == 0
    return {
        "prompt_count": len(summaries),
        "evaluation_record_count": len(records),
        "per_prompt": summaries,
        "improved_prompt_count": len(improved),
        "catastrophic_record_count": catastrophic_count,
        "aggregate_baseline_v_relative_l2": aggregate_baseline_v,
        "aggregate_rank1_v_relative_l2": aggregate_corrected_v,
        "mean_baseline_jsd": mean(
            [
                record["baseline"]["metrics"]["jensen_shannon_divergence_nats"]
                for record in records
            ]
        ),
        "mean_rank1_jsd": mean(
            [
                record["rank1_integer"]["metrics"][
                    "jensen_shannon_divergence_nats"
                ]
                for record in records
            ]
        ),
        "mean_baseline_reference_rank": mean(
            [
                float(record["baseline"]["metrics"]["reference_token_rank"])
                for record in records
            ]
        ),
        "mean_rank1_reference_rank": mean(
            [
                float(record["rank1_integer"]["metrics"]["reference_token_rank"])
                for record in records
            ]
        ),
        "mean_baseline_top8_overlap": mean(
            [
                float(record["baseline"]["metrics"]["top8_overlap_count"])
                for record in records
            ]
        ),
        "mean_rank1_top8_overlap": mean(
            [
                float(record["rank1_integer"]["metrics"]["top8_overlap_count"])
                for record in records
            ]
        ),
        "accepted_nonofficial_generalization": accepted,
        "decision": (
            "POSITIVE_PENDING_FRESH_L2_REVIEW"
            if accepted
            else "NEGATIVE_TERMINAL_GENERAL_PROMOTION_REJECTED"
        ),
    }


def build_contexts(tokenizer: Any) -> list[dict[str, Any]]:
    snapshot = prior.generation_runner.resolve_snapshot()
    prior.backend.configure_snapshot(snapshot)
    print("ACE2_RANK1_GENERALIZATION_BF16_CAPTURE_START", flush=True)
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
        prior.backend.canonical.ADAPTER_DIR,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    contexts: list[dict[str, Any]] = []
    try:
        for item in PROMPTS:
            token_ids = prior.generation_runner.canonical_chat_token_ids(
                tokenizer, item["text"]
            )
            bf16_decode_tokens: list[int] = []
            for step in range(DECODE_STEPS):
                context_ids = [*token_ids, *bf16_decode_tokens]
                require(
                    len(context_ids) <= prior.backend.MAX_CONTEXT_TOKENS,
                    "context exceeded backend bound",
                )
                layer_input, captures, logits = capture_layer23_context(model, context_ids)
                reference_token = int(torch.argmax(logits).item())
                contexts.append(
                    {
                        "prompt_id": item["prompt_id"],
                        "category": item["category"],
                        "decode_step": step,
                        "token_ids": context_ids,
                        "bf16_prefix_generated_token_ids": list(bf16_decode_tokens),
                        "reference_token_id": reference_token,
                        "layer_input": layer_input,
                        "captures": captures,
                        "bf16_logits": logits,
                    }
                )
                bf16_decode_tokens.append(reference_token)
            print(
                "ACE2_RANK1_GENERALIZATION_BF16_PROMPT "
                f"prompt={item['prompt_id']} token_count={len(token_ids)} "
                f"bf16_tokens={','.join(str(v) for v in bf16_decode_tokens)}",
                flush=True,
            )
    finally:
        del model
        gc.collect()
    return contexts


def generate() -> None:
    require(not RESULT.exists(), "generalization result already exists")
    require(not SUMS.exists(), "generalization SHA256SUMS already exists")
    started = time.monotonic()
    snapshot = prior.generation_runner.resolve_snapshot()
    prior.backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    freeze = preregister(tokenizer)
    protected_before = validate_authority()
    contexts = build_contexts(tokenizer)
    records, suppressed_messages = evaluate_contexts(
        freeze=freeze, tokenizer=tokenizer, contexts=contexts
    )
    aggregate_result = aggregate(records)
    protected_after = validate_authority()
    require(protected_before == protected_after, "protected predecessor evidence changed")
    shutil.copyfile(Path(__file__).resolve(), EXECUTION_RUNNER)
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": aggregate_result["decision"],
        "classification": (
            "bounded_nonofficial_software_first_multi_prompt_generalization"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution": {
            "elapsed_wall_seconds": time.monotonic() - started,
            "prompt_count": len(PROMPTS),
            "decode_steps": DECODE_STEPS,
            "fixed_eval_steps": list(range(DECODE_STEPS)),
            "layer_cut": LAYER,
            "context_policy": (
                "baseline and frozen correction are scored on BF16-greedy "
                "teacher-forced contexts; no W4A8 closed-loop official generation"
            ),
            "freeze_existed_before_downstream_evaluation": True,
        },
        "bindings": {
            "freeze": prior.localizer.file_record(FREEZE),
            "runner": prior.localizer.file_record(Path(__file__).resolve()),
            "archived_execution_runner": prior.localizer.file_record(EXECUTION_RUNNER),
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
        "prompt_manifest": copy.deepcopy(freeze["prompt_manifest"]),
        "evaluation_policy": copy.deepcopy(freeze["evaluation_policy"]),
        "decision_policy": copy.deepcopy(freeze["decision_policy"]),
        "records": records,
        "aggregate": aggregate_result,
        "short_nonofficial_decode_comparison": [
            {
                "prompt_id": prompt_id,
                "bf16_reference_token_ids": [
                    int(item["reference_token_id"]) for item in sorted(items, key=lambda v: v["decode_step"])
                ],
                "baseline_top_token_ids": [
                    int(item["baseline"]["metrics"]["top_token_id"])
                    for item in sorted(items, key=lambda v: v["decode_step"])
                ],
                "rank1_top_token_ids": [
                    int(item["rank1_integer"]["metrics"]["top_token_id"])
                    for item in sorted(items, key=lambda v: v["decode_step"])
                ],
                "comparison_scope": (
                    "software-only token comparison on BF16-greedy teacher-forced "
                    "contexts, not product chat completion"
                ),
            }
            for prompt_id, items in sorted(
                {
                    prompt_id: [r for r in records if r["prompt_id"] == prompt_id]
                    for prompt_id in {r["prompt_id"] for r in records}
                }.items()
            )
        ],
        "suppressed_nonfunctional_cache_effect_assertions": {
            "count": len(suppressed_messages),
            "unique_messages": sorted(set(suppressed_messages)),
        },
        "scope_guards": copy.deepcopy(SCOPE_GUARDS),
        "no_claims": {
            "official_attempt": False,
            "attempt_0003": False,
            "rtl_modification_or_simulation": False,
            "formal": False,
            "synthesis_or_ppa": False,
            "fpga_u280_xrt": False,
            "product_completion": False,
        },
        "independent_fresh_l2_review_required": True,
    }
    write_json(RESULT, result)
    review = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "review_type": "Fresh-L2",
        "status": "PENDING_INDEPENDENT_REVIEW",
        "result": prior.localizer.file_record(RESULT),
        "freeze": prior.localizer.file_record(FREEZE),
        "execution_runner": prior.localizer.file_record(EXECUTION_RUNNER),
        "reproduction_command": (
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
            "tools/run_stage1_layer23_v_rank1_generalization.py --verify-only"
        ),
        "requested_checks": [
            "Confirm Fresh-L2 acceptance of RESULT.json SHA-256 1a19b068aaf6824ae29bf0ed4710c0015cb617c3b7fb62ead35f96d17ae4f77f.",
            "Confirm generalization-freeze.json predates downstream evaluation and fixes four prompt hashes/categories plus two evaluation steps.",
            "Confirm candidate factor, Scale32, rounding, saturation, and layer-23 placement hashes are unchanged.",
            "Recompute per-record V relative-L2, target/reference token rank, top-8 overlap, JSD, saturation, and overflow from stored tensors/logits.",
            "Confirm no RTL, official attempt, attempt-0003, synthesis/PPA, U280/XRT, factor refit, rescale, or prompt feedback tuning occurred.",
        ],
    }
    write_json(REVIEW_REQUEST, review)
    generate_sums()
    print(
        "ACE2_RANK1_GENERALIZATION_RESULT "
        f"status={result['status']} "
        f"improved_prompts={aggregate_result['improved_prompt_count']}/"
        f"{aggregate_result['prompt_count']} "
        f"catastrophic_records={aggregate_result['catastrophic_record_count']} "
        f"base_v={aggregate_result['aggregate_baseline_v_relative_l2']['relative_l2']:.6f} "
        f"rank1_v={aggregate_result['aggregate_rank1_v_relative_l2']['relative_l2']:.6f} "
        f"output={prior.localizer.public_path(OUTPUT)}",
        flush=True,
    )


def recompute_record(record: dict[str, Any]) -> dict[str, Any]:
    artifacts = record["artifacts"]
    freeze = read_json(candidate.FREEZE)
    integer_evidence, correction, corrected = candidate.integer_arithmetic(
        freeze=freeze,
        input_dequantized=load_tensor(artifacts["input_dequantized"]).to(torch.float64),
        baseline_output=load_tensor(artifacts["baseline_v_output"]).to(torch.float64),
        write_artifacts=False,
    )
    require(
        torch.equal(correction, load_tensor(artifacts["predicted_residual"]).to(torch.float64)),
        "stored predicted residual changed",
    )
    require(
        torch.equal(corrected, load_tensor(artifacts["corrected_v_output"]).to(torch.float64)),
        "stored corrected V changed",
    )
    bf16_v = load_tensor(artifacts["bf16_v_output"]).to(torch.float64)
    baseline_v = load_tensor(artifacts["baseline_v_output"]).to(torch.float64)
    bf16_logits = np.fromfile(ROOT / artifacts["bf16_logits"]["path"], dtype="<f4")
    baseline_logits = np.fromfile(
        ROOT / record["baseline"]["logits"]["path"], dtype="<f8"
    )
    corrected_logits = np.fromfile(
        ROOT / record["rank1_integer"]["logits"]["path"], dtype="<f8"
    )
    baseline_comparison = prior.quality.score_comparison(bf16_logits, baseline_logits)
    corrected_comparison = prior.quality.score_comparison(bf16_logits, corrected_logits)
    reference_token = int(np.argmax(bf16_logits))
    baseline_metrics = {
        "reference_token_id": reference_token,
        "reference_token_rank": int(
            baseline_comparison["reference_top_rank_in_candidate"]
        ),
        "reference_token_top1": (
            int(baseline_comparison["reference_top_rank_in_candidate"]) == 1
            and int(baseline_comparison["candidate_top_token_id"]) == reference_token
        ),
        "top_token_id": int(baseline_comparison["candidate_top_token_id"]),
        "top8_overlap_count": int(baseline_comparison["top8_overlap_count"]),
        "jensen_shannon_divergence_nats": float(
            baseline_comparison["jensen_shannon_divergence_nats"]
        ),
        "relative_l2": float(baseline_comparison["raw_score_error"]["relative_l2"]),
        "comparison_vs_bf16_oracle": baseline_comparison,
    }
    corrected_metrics = {
        "reference_token_id": reference_token,
        "reference_token_rank": int(
            corrected_comparison["reference_top_rank_in_candidate"]
        ),
        "reference_token_top1": (
            int(corrected_comparison["reference_top_rank_in_candidate"]) == 1
            and int(corrected_comparison["candidate_top_token_id"]) == reference_token
        ),
        "top_token_id": int(corrected_comparison["candidate_top_token_id"]),
        "top8_overlap_count": int(corrected_comparison["top8_overlap_count"]),
        "jensen_shannon_divergence_nats": float(
            corrected_comparison["jensen_shannon_divergence_nats"]
        ),
        "relative_l2": float(corrected_comparison["raw_score_error"]["relative_l2"]),
        "comparison_vs_bf16_oracle": corrected_comparison,
    }
    rebuilt = copy.deepcopy(record)
    rebuilt["baseline"]["metrics"] = baseline_metrics
    rebuilt["rank1_integer"]["metrics"] = corrected_metrics
    rebuilt["baseline"]["v_relative_l2"] = error_summary(baseline_v, bf16_v)
    rebuilt["rank1_integer"]["v_relative_l2"] = error_summary(corrected, bf16_v)
    rebuilt["rank1_integer"]["integer_evidence"] = integer_evidence
    rebuilt["improvement_flags"] = {
        "v_relative_l2_improved": (
            rebuilt["rank1_integer"]["v_relative_l2"]["relative_l2"]
            < rebuilt["baseline"]["v_relative_l2"]["relative_l2"]
        ),
        "reference_rank_improved_or_equal": (
            corrected_metrics["reference_token_rank"]
            <= baseline_metrics["reference_token_rank"]
        ),
        "top8_overlap_improved_or_equal": (
            corrected_metrics["top8_overlap_count"]
            >= baseline_metrics["top8_overlap_count"]
        ),
        "jsd_improved_or_equal": (
            corrected_metrics["jensen_shannon_divergence_nats"]
            <= baseline_metrics["jensen_shannon_divergence_nats"]
        ),
    }
    rebuilt["catastrophic_regression"] = catastrophic(rebuilt)
    return rebuilt


def verify() -> None:
    require(FREEZE.is_file(), "generalization freeze is missing")
    require(RESULT.is_file(), "generalization result is missing")
    require(REVIEW_REQUEST.is_file(), "Fresh-L2 review request is missing")
    require(EXECUTION_RUNNER.is_file(), "archived execution runner is missing")
    verify_sum_tree()
    snapshot = prior.generation_runner.resolve_snapshot()
    prior.backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    freeze = read_json(FREEZE)
    validate_freeze(freeze, tokenizer)
    result = read_json(RESULT)
    require(result["scope_guards"] == SCOPE_GUARDS, "result scope guards changed")
    require(result["bindings"]["freeze"] == prior.localizer.file_record(FREEZE), "freeze binding changed")
    require(
        result["bindings"]["runner"] == prior.localizer.file_record(Path(__file__).resolve())
        and result["bindings"]["archived_execution_runner"]
        == prior.localizer.file_record(EXECUTION_RUNNER),
        "runner binding changed",
    )
    recomputed_records = [recompute_record(record) for record in result["records"]]
    for stored, recomputed in zip(result["records"], recomputed_records, strict=True):
        require(
            stored["baseline"]["metrics"] == recomputed["baseline"]["metrics"],
            "baseline metrics changed",
        )
        require(
            stored["rank1_integer"]["metrics"]
            == recomputed["rank1_integer"]["metrics"],
            "rank1 metrics changed",
        )
        require(
            stored["baseline"]["v_relative_l2"]
            == recomputed["baseline"]["v_relative_l2"],
            "baseline V relative-L2 changed",
        )
        require(
            stored["rank1_integer"]["v_relative_l2"]
            == recomputed["rank1_integer"]["v_relative_l2"],
            "rank1 V relative-L2 changed",
        )
        require(
            stored["rank1_integer"]["integer_evidence"]
            == recomputed["rank1_integer"]["integer_evidence"],
            "integer evidence changed",
        )
        require(
            stored["catastrophic_regression"] == recomputed["catastrophic_regression"],
            "catastrophic decision changed",
        )
    recomputed_aggregate = aggregate(recomputed_records)
    require(recomputed_aggregate == result["aggregate"], "aggregate decision changed")
    request = read_json(REVIEW_REQUEST)
    require(
        request["status"] == "PENDING_INDEPENDENT_REVIEW"
        and request["result"] == prior.localizer.file_record(RESULT)
        and request["freeze"] == prior.localizer.file_record(FREEZE)
        and request["execution_runner"] == prior.localizer.file_record(EXECUTION_RUNNER),
        "Fresh-L2 review request binding changed",
    )
    require(
        not (
            ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0003"
        ).exists(),
        "attempt-0003 exists",
    )
    print(
        "PASS stage1vrank1int-generalization01 "
        f"status={result['status']} "
        f"improved_prompts={result['aggregate']['improved_prompt_count']}/"
        f"{result['aggregate']['prompt_count']} "
        f"catastrophic_records={result['aggregate']['catastrophic_record_count']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Torch CPU thread count.",
    )
    args = parser.parse_args()
    require(args.threads >= 1, "--threads must be positive")
    torch.set_num_threads(args.threads)
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
            f"ACE2_RANK1_GENERALIZATION_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
