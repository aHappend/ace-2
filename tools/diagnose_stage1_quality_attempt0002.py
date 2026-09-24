#!/usr/bin/env python3
"""Localize attempt-0002 output quality without replaying the official run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = (
    ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0002"
)
ADAPTER = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4"
    / "attempt-0001/checkpoints/checkpoint-176"
)
OUT = ROOT / "diagnosis/stage1qualitydiag01"
RESULT = OUT / "result.json"
SUMS = OUT / "SHA256SUMS"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT_SHA256 = "90eaeeb4312de67b46fda2c43a80f3bfeb0ac20d0dc222a65f390c9407607c36"
PROMPT_TOKEN_IDS_SHA256 = (
    "dbdff6c533c804f645b0a76b8a44cf85e9b61f52bcce053e81189bd82f009e0d"
)
CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
MODEL_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
ADAPTER_SHA256 = "c9476d6ccc687a4d68de1951eb889b689daaf9c5d3822b4c5acfc26a1cd38a28"
ADAPTER_CONFIG_SHA256 = (
    "9261b3480065c0d1bdf8b0a33b01279f1cfdb285ca7aaf3aa850fb62ab894664"
)
CHECKPOINT_TREE_SHA256 = (
    "8cee9b9343e49b66ee2ea427578f87335ffa259919356cdd119cd8712ff7eeee"
)
TOKENIZER_JSON_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
SEALED_TOKEN_IDS = [26614, 43895, 92464, 76521]
MODEL_OUTPUT_DOMAIN = 151936
TOP_K = 8


class DiagnosisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosisError(message)


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return file_record(path)


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


def save_f32(path: Path, values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values, dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes())
    record = file_record(path)
    record.update({"dtype": "float32_le", "shape": list(array.shape)})
    return record


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


def decode_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def summarize_scores(
    scores: np.ndarray, tokenizer: Any, target_ids: list[int]
) -> dict[str, Any]:
    require(scores.shape == (MODEL_OUTPUT_DOMAIN,), "unexpected logits shape")
    order = stable_order(scores)
    top_ids = order[:TOP_K]
    top_score = float(scores[top_ids[0]])
    runner_up_score = float(scores[top_ids[1]])
    top_ties = np.flatnonzero(scores == top_score)
    return {
        "top_candidates": [
            {
                "token_id": int(token_id),
                "piece": decode_piece(tokenizer, int(token_id)),
                "score": float(scores[token_id]),
            }
            for token_id in top_ids
        ],
        "top_token_id": int(top_ids[0]),
        "top_piece": decode_piece(tokenizer, int(top_ids[0])),
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "top_margin": top_score - runner_up_score,
        "top_tie_count": int(top_ties.size),
        "target_ranks": {
            str(token_id): rank_of(scores, token_id) for token_id in target_ids
        },
        "target_scores": {
            str(token_id): float(scores[token_id]) for token_id in target_ids
        },
    }


def cache_length(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def load_model(with_adapter: bool) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        attn_implementation="eager",
        device_map=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if with_adapter:
        model = PeftModel.from_pretrained(
            model,
            ADAPTER,
            is_trainable=False,
            autocast_adapter_dtype=False,
            low_cpu_mem_usage=True,
        )
    model.eval()
    return model


def run_sequence(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    feedback_ids: list[int] | None,
    label: str,
    mode: str,
    capture_hidden: bool,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray | None]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    records: list[dict[str, Any]] = []
    logits_values: list[np.ndarray] = []
    hidden_capture: np.ndarray | None = None
    chosen_ids: list[int] = []
    past = None

    with torch.inference_mode():
        for step in range(4):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=capture_hidden and step == 0,
                return_dict=True,
            )
            logits = (
                output.logits[0, -1]
                .detach()
                .to(torch.float32)
                .cpu()
                .contiguous()
                .numpy()
            )
            require(
                logits.shape == (MODEL_OUTPUT_DOMAIN,),
                f"{label} step {step} emitted the wrong vocabulary shape",
            )
            logits_values.append(logits)
            artifact = save_f32(
                OUT / "logits" / f"{label}-{mode}-step-{step:02d}-f32le.bin",
                logits,
            )
            summary = summarize_scores(logits, tokenizer, SEALED_TOKEN_IDS)
            if feedback_ids is None:
                chosen = int(summary["top_token_id"])
            else:
                chosen = int(feedback_ids[step])
            chosen_ids.append(chosen)
            records.append(
                {
                    "generation_index": step,
                    "context_token_count": len(prompt_ids) + step,
                    "cache_length_after_forward": cache_length(
                        output.past_key_values
                    ),
                    "feedback_token_id": chosen if step < 3 else None,
                    "logits": artifact,
                    "ranking": summary,
                }
            )
            if capture_hidden and step == 0:
                require(output.hidden_states is not None, "hidden states were not captured")
                hidden_capture = (
                    torch.stack(
                        [state[0, -1].to(torch.float32) for state in output.hidden_states]
                    )
                    .cpu()
                    .contiguous()
                    .numpy()
                )
            past = output.past_key_values
            if step < 3:
                input_ids = torch.tensor([[chosen]], dtype=torch.long)
                attention_mask = torch.ones(
                    (1, len(prompt_ids) + step + 1), dtype=torch.long
                )

    require(
        [item["cache_length_after_forward"] for item in records]
        == [len(prompt_ids) + step for step in range(4)],
        f"{label} {mode} cache length did not advance by one token",
    )
    return (
        {
            "mode": mode,
            "use_cache": True,
            "feedback_policy": (
                "sealed_w4a8_teacher_forced"
                if feedback_ids is not None
                else "independent_greedy_lowest_token_id_tie_break"
            ),
            "generated_token_ids": chosen_ids,
            "decoded_text": tokenizer.decode(
                chosen_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "steps": records,
        },
        logits_values,
        hidden_capture,
    )


def vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    delta = candidate64 - reference64
    reference_norm = float(np.linalg.norm(reference64))
    candidate_norm = float(np.linalg.norm(candidate64))
    denominator = reference_norm * candidate_norm
    return {
        "relative_l2": float(np.linalg.norm(delta) / reference_norm),
        "cosine_similarity": (
            float(np.dot(reference64, candidate64) / denominator)
            if denominator
            else 0.0
        ),
        "rmse": float(math.sqrt(float(np.mean(delta * delta)))),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
    }


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first64 = first.astype(np.float64)
    second64 = second.astype(np.float64)
    first_exp = np.exp(first64 - np.max(first64))
    second_exp = np.exp(second64 - np.max(second64))
    first_prob = first_exp / np.sum(first_exp)
    second_prob = second_exp / np.sum(second_exp)
    midpoint = (first_prob + second_prob) * 0.5
    first_mask = first_prob > 0
    second_mask = second_prob > 0
    return float(
        0.5
        * np.sum(
            first_prob[first_mask]
            * (np.log(first_prob[first_mask]) - np.log(midpoint[first_mask]))
        )
        + 0.5
        * np.sum(
            second_prob[second_mask]
            * (np.log(second_prob[second_mask]) - np.log(midpoint[second_mask]))
        )
    )


def score_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference_order = stable_order(reference)
    candidate_order = stable_order(candidate)
    reference_top = int(reference_order[0])
    candidate_top = int(candidate_order[0])
    centered_reference = reference.astype(np.float64) - float(np.mean(reference))
    centered_candidate = candidate.astype(np.float64) - float(np.mean(candidate))
    denominator = float(
        np.linalg.norm(centered_reference) * np.linalg.norm(centered_candidate)
    )
    return {
        "top1_match": reference_top == candidate_top,
        "reference_top_token_id": reference_top,
        "reference_top_rank_in_candidate": rank_of(candidate, reference_top),
        "candidate_top_token_id": candidate_top,
        "candidate_top_rank_in_reference": rank_of(reference, candidate_top),
        "top8_overlap_count": len(
            set(map(int, reference_order[:TOP_K]))
            & set(map(int, candidate_order[:TOP_K]))
        ),
        "centered_cosine_similarity": (
            float(np.dot(centered_reference, centered_candidate) / denominator)
            if denominator
            else 0.0
        ),
        "jensen_shannon_divergence_nats": js_divergence(reference, candidate),
        "raw_score_error": vector_metrics(reference, candidate),
    }


def load_w4a8_steps(tokenizer: Any) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    records: list[dict[str, Any]] = []
    scores_list: list[np.ndarray] = []
    for step, expected_token in enumerate(SEALED_TOKEN_IDS):
        head_dir = ATTEMPT / "heads" / f"step-{step:02d}"
        execution_path = head_dir / "head_execution.json"
        execution = read_json(execution_path)
        output_path = head_dir / "tensors/lm_head_output_s8.bin"
        scale_path = head_dir / "tensors/lm_head_output_scale_f64le.bin"
        raw = np.fromfile(output_path, dtype=np.int8)
        scale_values = np.fromfile(scale_path, dtype="<f8")
        require(raw.shape == (MODEL_OUTPUT_DOMAIN,), "sealed logits shape changed")
        require(scale_values.shape == (1,), "sealed logit scale shape changed")
        scores = raw.astype(np.float64) * float(scale_values[0])
        summary = summarize_scores(scores, tokenizer, SEALED_TOKEN_IDS)
        require(
            summary["top_token_id"] == expected_token,
            f"sealed W4A8 step {step} top token changed",
        )
        require(
            execution["lm_head"]["top_token"] == expected_token,
            f"sealed RTL step {step} selected token changed",
        )
        require(
            execution["lm_head"]["integer_mismatches"] == 0
            and execution["final_rmsnorm"]["integer_mismatches"] == 0,
            f"sealed RTL/reference mismatch at step {step}",
        )
        require(
            execution["lm_head"]["selected_token_agreement"]
            and execution["lm_head"]["selected_logit_agreement"],
            f"sealed RTL selection disagreement at step {step}",
        )
        records.append(
            {
                "generation_index": step,
                "selected_token_id": expected_token,
                "output_s8": file_record(output_path),
                "output_scale_f64le": file_record(scale_path),
                "head_execution": file_record(execution_path),
                "ranking": summary,
                "rtl_full_vector_integer_mismatches": execution["lm_head"][
                    "integer_mismatches"
                ],
                "rtl_final_rmsnorm_integer_mismatches": execution["final_rmsnorm"][
                    "integer_mismatches"
                ],
            }
        )
        scores_list.append(scores)
    return records, scores_list


def hidden_boundary_trace(
    bf16_hidden: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        bf16_hidden.shape == (25, 896),
        "checkpoint-176 hidden-state capture shape changed",
    )
    trace: list[dict[str, Any]] = []
    for layer in range(24):
        tensor_dir = ATTEMPT / "tokens" / "position-33" / f"layer-{layer:02d}" / "tensors"
        hidden_path = tensor_dir / "input_hidden_s8.bin"
        scale_path = tensor_dir / "input_scale_f64le.bin"
        hidden_s8 = np.fromfile(hidden_path, dtype=np.int8)
        scale_values = np.fromfile(scale_path, dtype="<f8")
        require(hidden_s8.shape == (896,), f"layer {layer} hidden shape changed")
        require(scale_values.shape == (1,), f"layer {layer} scale shape changed")
        dequantized = hidden_s8.astype(np.float64) * float(scale_values[0])
        trace.append(
            {
                "boundary": f"model.layers.{layer}.input",
                "layer": layer,
                "w4a8_input_hidden_s8": file_record(hidden_path),
                "w4a8_input_scale_f64le": file_record(scale_path),
                "metrics_vs_checkpoint176_bf16": vector_metrics(
                    bf16_hidden[layer], dequantized
                ),
            }
        )

    final_path = ATTEMPT / "heads/step-00/tensors/final_rmsnorm_s8.bin"
    final_scale_path = (
        ATTEMPT / "heads/step-00/tensors/final_rmsnorm_scale_f64le.bin"
    )
    final_s8 = np.fromfile(final_path, dtype=np.int8)
    final_scale = np.fromfile(final_scale_path, dtype="<f8")
    require(final_s8.shape == (896,), "sealed final RMSNorm shape changed")
    require(final_scale.shape == (1,), "sealed final RMSNorm scale shape changed")
    final_dequantized = final_s8.astype(np.float64) * float(final_scale[0])
    final_record = {
        "boundary": "model.final_rmsnorm.output",
        "w4a8_final_rmsnorm_s8": file_record(final_path),
        "w4a8_final_rmsnorm_scale_f64le": file_record(final_scale_path),
        "metrics_vs_checkpoint176_bf16": vector_metrics(
            bf16_hidden[-1], final_dequantized
        ),
    }
    largest = max(
        trace,
        key=lambda item: item["metrics_vs_checkpoint176_bf16"]["relative_l2"],
    )
    summary = {
        "first_compared_boundary": trace[0]["boundary"],
        "first_relative_l2": trace[0]["metrics_vs_checkpoint176_bf16"][
            "relative_l2"
        ],
        "largest_layer_input_relative_l2_boundary": largest["boundary"],
        "largest_layer_input_relative_l2": largest[
            "metrics_vs_checkpoint176_bf16"
        ]["relative_l2"],
        "final_rmsnorm_relative_l2": final_record[
            "metrics_vs_checkpoint176_bf16"
        ]["relative_l2"],
        "interpretation": (
            "The trace measures accumulated numerical drift, not causal recovery; "
            "the step-0 final-logit rank reversal is the decisive same-prompt boundary."
        ),
    }
    return {"layer_inputs": trace, "final_rmsnorm": final_record}, summary


def relevant_sealed_records() -> dict[str, dict[str, Any]]:
    return {
        name: file_record(ATTEMPT / name)
        for name in (
            "authorization.json",
            "launch_provenance.json",
            "manifest.json",
            "run_summary.json",
            "status.json",
            "SHA256SUMS",
        )
    }


def generate() -> None:
    sealed_before = relevant_sealed_records()
    authorization = read_json(ATTEMPT / "authorization.json")
    launch = read_json(ATTEMPT / "launch_provenance.json")
    manifest = read_json(ATTEMPT / "manifest.json")
    summary = read_json(ATTEMPT / "run_summary.json")
    tokenizer_contracts = find_dicts(
        launch,
        lambda item: item.get("prompt_sha256") == PROMPT_SHA256
        and "prompt_token_ids" in item,
    )
    require(tokenizer_contracts, "sealed prompt contract is missing")
    prompt_id_variants = {
        sha256_bytes(canonical_bytes(item["prompt_token_ids"]))
        for item in tokenizer_contracts
    }
    require(
        prompt_id_variants == {PROMPT_TOKEN_IDS_SHA256},
        "duplicated sealed prompt bindings disagree",
    )
    full_tokenizer_contracts = [
        item
        for item in tokenizer_contracts
        if "source_files" in item
        and "chat_template_sha256" in item
        and "generation_bounds" in item
    ]
    require(
        full_tokenizer_contracts,
        "full sealed tokenizer contract is missing",
    )
    full_contract_identities = {
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
        for item in full_tokenizer_contracts
    }
    require(
        len(full_contract_identities) == 1,
        "duplicated full tokenizer contracts disagree",
    )
    tokenizer_contract = full_tokenizer_contracts[0]
    prompt_ids = list(map(int, tokenizer_contract["prompt_token_ids"]))
    require(len(prompt_ids) == 34, "sealed prompt token count changed")
    require(
        sha256_bytes(canonical_bytes(prompt_ids)) == PROMPT_TOKEN_IDS_SHA256,
        "sealed prompt token identity changed",
    )
    require(
        authorization["prompt_sha256"] == PROMPT_SHA256
        and authorization["prompt_token_ids_sha256"] == PROMPT_TOKEN_IDS_SHA256,
        "authorization prompt binding changed",
    )
    require(
        manifest["generated_token_ids"] == SEALED_TOKEN_IDS
        and summary["generated_token_ids"] == SEALED_TOKEN_IDS,
        "sealed output token IDs changed",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    require(len(tokenizer) == 151665, "tokenizer length changed")
    require(tokenizer.vocab_size == 151643, "tokenizer vocabulary changed")
    require(
        sha256_bytes(tokenizer.chat_template.encode("utf-8")) == CHAT_TEMPLATE_SHA256,
        "chat template changed",
    )
    require(
        tokenizer_contract["chat_template_sha256"] == CHAT_TEMPLATE_SHA256,
        "sealed chat template binding changed",
    )
    require(
        tokenizer_contract["source_files"]["tokenizer.json"]["sha256"]
        == TOKENIZER_JSON_SHA256
        and tokenizer_contract["source_files"]["tokenizer_config.json"]["sha256"]
        == TOKENIZER_CONFIG_SHA256,
        "sealed tokenizer source binding changed",
    )

    w4a8_records, w4a8_scores = load_w4a8_steps(tokenizer)

    base_model = load_model(with_adapter=False)
    base_teacher, base_teacher_logits, _ = run_sequence(
        base_model,
        tokenizer,
        prompt_ids,
        SEALED_TOKEN_IDS,
        "base-bf16",
        "teacher-forced",
        False,
    )
    base_greedy, _, _ = run_sequence(
        base_model,
        tokenizer,
        prompt_ids,
        None,
        "base-bf16",
        "greedy",
        False,
    )
    del base_model
    gc.collect()

    checkpoint_model = load_model(with_adapter=True)
    checkpoint_teacher, checkpoint_teacher_logits, checkpoint_hidden = run_sequence(
        checkpoint_model,
        tokenizer,
        prompt_ids,
        SEALED_TOKEN_IDS,
        "checkpoint176-bf16",
        "teacher-forced",
        True,
    )
    checkpoint_greedy, _, _ = run_sequence(
        checkpoint_model,
        tokenizer,
        prompt_ids,
        None,
        "checkpoint176-bf16",
        "greedy",
        False,
    )
    del checkpoint_model
    gc.collect()
    require(checkpoint_hidden is not None, "checkpoint hidden-state capture missing")
    hidden_artifact = save_f32(
        OUT / "hidden/checkpoint176-bf16-position33-boundaries-f32le.bin",
        checkpoint_hidden,
    )
    boundary_trace, boundary_summary = hidden_boundary_trace(checkpoint_hidden)

    teacher_comparisons: list[dict[str, Any]] = []
    for step in range(4):
        teacher_comparisons.append(
            {
                "generation_index": step,
                "identical_prefix_token_count": len(prompt_ids) + step,
                "base_bf16_vs_checkpoint176_bf16": score_comparison(
                    checkpoint_teacher_logits[step], base_teacher_logits[step]
                ),
                "checkpoint176_bf16_vs_w4a8": score_comparison(
                    checkpoint_teacher_logits[step], w4a8_scores[step]
                ),
            }
        )

    first = teacher_comparisons[0]["checkpoint176_bf16_vs_w4a8"]
    require(not first["top1_match"], "step-0 BF16 and W4A8 unexpectedly agree")
    require(
        all(
            record["rtl_full_vector_integer_mismatches"] == 0
            and record["rtl_final_rmsnorm_integer_mismatches"] == 0
            for record in w4a8_records
        ),
        "sealed RTL/reference parity changed",
    )
    require(
        checkpoint_teacher["steps"][0]["logits"]["sha256"]
        == checkpoint_greedy["steps"][0]["logits"]["sha256"],
        "checkpoint step-0 logits depend on feedback mode",
    )
    require(
        base_teacher["steps"][0]["logits"]["sha256"]
        == base_greedy["steps"][0]["logits"]["sha256"],
        "base step-0 logits depend on feedback mode",
    )

    prior_layer_scan = (
        ROOT
        / "evidence/candidates/w4a8-chat-hidden-state-localization-v1"
        / "candidate-0002/layer-scan.json"
    )
    prior_mechanism = (
        ROOT
        / "evidence/candidates/w4a8-chat-hidden-state-localization-v1"
        / "candidate-0002/mechanism-split.json"
    )
    prior_activation = (
        ROOT
        / "evidence/candidates/w4a8-chat-hidden-state-localization-v1"
        / "candidate-0002/activation-localization.json"
    )

    result = {
        "schema_version": 1,
        "status": "PASS_STAGE1_QUALITY_FAILURE_LOCALIZED_AWAITING_INDEPENDENT_REVIEW",
        "classification": "candidate_only_diagnosis_no_official_execution",
        "scope_guards": {
            "attempt_0002_read_only": True,
            "new_official_attempt_created": False,
            "official_preflight_launched": False,
            "retry_replay_or_resume_launched": False,
            "rtl_modified": False,
            "stage2_entered": False,
        },
        "frozen_contract": {
            "attempt": "attempt-0002",
            "prompt": {
                "sha256": PROMPT_SHA256,
                "token_ids_sha256": PROMPT_TOKEN_IDS_SHA256,
                "token_count": len(prompt_ids),
                "text_persisted": False,
            },
            "tokenizer": {
                "repository": MODEL_ID,
                "revision": REVISION,
                "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
                "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
                "chat_template_sha256": CHAT_TEMPLATE_SHA256,
                "length": len(tokenizer),
                "vocab_size": tokenizer.vocab_size,
                "special_token_ids": list(map(int, tokenizer.all_special_ids)),
                "special_tokens_map_sha256": sha256_bytes(
                    canonical_bytes(tokenizer.special_tokens_map)
                ),
            },
            "generation": {
                "max_new_tokens": 4,
                "termination": "fixed_four_tokens_no_eos_observed",
                "eos_token_id": tokenizer.eos_token_id,
                "sealed_output_contains_eos": tokenizer.eos_token_id
                in SEALED_TOKEN_IDS,
                "selection": (
                    "greedy over all 151936 outputs; lowest token ID wins exact ties"
                ),
                "sampling": False,
                "kv_state_loop": (
                    "prefill all frozen prompt IDs with use_cache=True, then append "
                    "one feedback token per generation position"
                ),
            },
            "model": {
                "base_model_sha256": MODEL_SHA256,
                "checkpoint176_adapter_sha256": ADAPTER_SHA256,
                "checkpoint176_adapter_config_sha256": ADAPTER_CONFIG_SHA256,
                "checkpoint176_tree_sha256": CHECKPOINT_TREE_SHA256,
            },
            "sealed_artifacts_before": sealed_before,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": importlib.metadata.version("tokenizers"),
            "safetensors": importlib.metadata.version("safetensors"),
            "device": "cpu",
            "attention_implementation": "eager",
            "model_dtype": "torch.bfloat16",
        },
        "conditions": {
            "base_bf16": {
                "checkpoint": "base model without LoRA",
                "teacher_forced_on_sealed_prefix": base_teacher,
                "independent_greedy": base_greedy,
            },
            "checkpoint176_bf16": {
                "checkpoint": "base model plus checkpoint-176 LoRA",
                "teacher_forced_on_sealed_prefix": checkpoint_teacher,
                "independent_greedy": checkpoint_greedy,
                "position33_hidden_boundaries": hidden_artifact,
            },
            "checkpoint176_w4a8_software_and_sealed_rtl": {
                "generated_token_ids": SEALED_TOKEN_IDS,
                "decoded_text": manifest["decoded_text"],
                "steps": w4a8_records,
                "software_rtl_full_vector_mismatch_count": sum(
                    item["rtl_full_vector_integer_mismatches"]
                    for item in w4a8_records
                ),
            },
        },
        "same_prefix_teacher_forced_comparisons": teacher_comparisons,
        "same_prompt_boundary_trace": boundary_trace,
        "same_prompt_boundary_summary": boundary_summary,
        "first_decisive_divergence": {
            "generation_index": 0,
            "boundary": "final_logits_before_any_generated_token_feedback",
            "identical_prompt_token_count": len(prompt_ids),
            "checkpoint176_bf16_top_token_id": first["reference_top_token_id"],
            "w4a8_top_token_id": first["candidate_top_token_id"],
            "checkpoint176_bf16_top_rank_in_w4a8": first[
                "reference_top_rank_in_candidate"
            ],
            "w4a8_top_rank_in_checkpoint176_bf16": first[
                "candidate_top_rank_in_reference"
            ],
            "top8_overlap_count": first["top8_overlap_count"],
            "jensen_shannon_divergence_nats": first[
                "jensen_shannon_divergence_nats"
            ],
            "reason_decisive": (
                "The rank reversal occurs on the identical frozen prompt before "
                "generated-token feedback, while all 151936 software/RTL outputs match."
            ),
        },
        "diagnosis": {
            "dominant_fault": "quantization_calibration",
            "excluded_as_dominant": {
                "checkpoint_or_lora_quality": (
                    "checkpoint-176 BF16 produces its own greedy continuation before "
                    "W4A8 changes the first selected token"
                ),
                "tokenizer_or_chat_template": (
                    "all conditions consume the same frozen post-template token IDs "
                    "under the same tokenizer revision and special-token map"
                ),
                "generation_or_runtime_semantics": (
                    "the first mismatch precedes generated-token feedback; cache lengths "
                    "advance identically, and teacher-forced comparisons retain one prefix"
                ),
                "rtl": (
                    "sealed RTL equals the checkpoint-176 W4A8 software reference over "
                    "every full-vocabulary output at all four positions"
                ),
            },
            "narrowest_same_prompt_support": "checkpoint176_bf16_to_w4a8_final_logits",
            "cross_prompt_causal_support": {
                "limitation": (
                    "These causal layer-17 artifacts use the canonical Hello prompt, "
                    "not the sealed attempt-0002 prompt; they narrow the repair but do "
                    "not replace the same-prompt final-logit attribution."
                ),
                "layer_scan": file_record(prior_layer_scan),
                "mechanism_split": file_record(prior_mechanism),
                "activation_localization": file_record(prior_activation),
                "earliest_recovering_layer_cut": 18,
                "implicated_layer": 17,
                "mechanism": "tensor_wide_a8_activation_quantization",
                "earliest_recovering_operator_boundary": (
                    "model.layers.17.self_attn.q_proj.output"
                ),
            },
            "smallest_repair_task": (
                "Replace the tensor-wide signed-A8 scale at the layer-17 q-projection "
                "output with model-derived grouped activation scaling, then run only a "
                "nonofficial four-token aligned BF16/W4A8 software-and-focused-RTL "
                "candidate check; do not authorize another official attempt."
            ),
        },
        "source": file_record(Path(__file__)),
    }
    sealed_after = relevant_sealed_records()
    require(sealed_after == sealed_before, "sealed attempt changed during diagnosis")
    result["frozen_contract"]["sealed_artifacts_after"] = sealed_after
    write_json(RESULT, result)

    generated_files = sorted(
        path
        for path in OUT.rglob("*")
        if path.is_file() and path != SUMS
    )
    SUMS.write_text(
        "".join(
            f"{sha256_file(path)}  {public_path(path)}\n" for path in generated_files
        ),
        encoding="utf-8",
    )


def verify() -> None:
    require(RESULT.is_file(), "diagnosis result is missing")
    require(SUMS.is_file(), "diagnosis SHA256SUMS is missing")
    expected_paths: set[str] = set()
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"missing diagnosis artifact: {relative}")
        require(sha256_file(path) == digest, f"diagnosis hash mismatch: {relative}")
        expected_paths.add(relative)
    actual_paths = {
        public_path(path)
        for path in OUT.rglob("*")
        if path.is_file() and path != SUMS
    }
    require(expected_paths == actual_paths, "diagnosis artifact set changed")

    result = read_json(RESULT)
    require(
        result["status"]
        == "PASS_STAGE1_QUALITY_FAILURE_LOCALIZED_AWAITING_INDEPENDENT_REVIEW",
        "diagnosis status changed",
    )
    require(
        result["diagnosis"]["dominant_fault"] == "quantization_calibration",
        "dominant-fault classification changed",
    )
    require(
        result["first_decisive_divergence"]["generation_index"] == 0,
        "first decisive divergence changed",
    )
    require(
        result["first_decisive_divergence"]["checkpoint176_bf16_top_token_id"]
        != result["first_decisive_divergence"]["w4a8_top_token_id"],
        "decisive rank reversal disappeared",
    )
    require(
        result["conditions"]["checkpoint176_w4a8_software_and_sealed_rtl"][
            "software_rtl_full_vector_mismatch_count"
        ]
        == 0,
        "sealed software/RTL parity changed",
    )
    require(
        not any(
            result["scope_guards"][key]
            for key in (
                "new_official_attempt_created",
                "official_preflight_launched",
                "retry_replay_or_resume_launched",
                "rtl_modified",
                "stage2_entered",
            )
        ),
        "scope guard changed",
    )
    for phase in ("before", "after"):
        for record in result["frozen_contract"][f"sealed_artifacts_{phase}"].values():
            path = ROOT / record["path"]
            require(sha256_file(path) == record["sha256"], f"sealed hash changed: {path}")

    print(
        "PASS stage1qualitydiag01 "
        f"result_sha256={sha256_file(RESULT)} "
        f"sums_sha256={sha256_file(SUMS)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        generate()


if __name__ == "__main__":
    main()
