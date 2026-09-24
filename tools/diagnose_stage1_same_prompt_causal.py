#!/usr/bin/env python3
"""Causally localize the checkpoint-176 W4A8 position-0 quality failure."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
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

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import diagnose_stage1_quality_attempt0002 as quality
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner


MISSION_ID = "stage1samepromptcausal01"
PROMPT = "Tell me a joke."
REFERENCE_TOKEN = 39814
BASELINE_TOKEN = 26614
COARSE_CUTS = (4, 8, 12, 16, 20, 24)
TOP_K = 8
OUTPUT = ROOT / "diagnosis/stage1samepromptcausal01"
RESULT = OUTPUT / "result.json"
SUMS = OUTPUT / "SHA256SUMS"
REVIEW_REQUEST = OUTPUT / "fresh-l2-review-request.json"
PRIOR_DIAGNOSIS = ROOT / "diagnosis/stage1qualitydiag01/result.json"
SEALED_ATTEMPT = (
    ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0002"
)

PROMPT_SHA256 = "90eaeeb4312de67b46fda2c43a80f3bfeb0ac20d0dc222a65f390c9407607c36"
TOKEN_IDS_SHA256 = "dbdff6c533c804f645b0a76b8a44cf85e9b61f52bcce053e81189bd82f009e0d"
CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
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
BF16_LOGITS_SHA256 = (
    "1910fe7196b34ac858b2d5b8406ee94df00ef39e5f285ba7ed56381de66b7614"
)
PRIOR_DIAGNOSIS_SHA256 = (
    "9ec27ee399eeaa6780ce7ab0c6f28250f9771d437c1a3953d6994a55fc7c59df"
)

OPERATOR_VARIANTS = (
    ("input_rmsnorm_output", "input_norm", frozenset()),
    ("q_projection_output", None, frozenset({"q"})),
    ("qk_projection_outputs", None, frozenset({"q", "k"})),
    ("qkv_projection_outputs", None, frozenset({"q", "k", "v"})),
    ("o_projection_output", None, frozenset({"o"})),
    ("post_attention_rmsnorm_output", "post_norm", frozenset()),
    ("gate_projection_output", None, frozenset({"gate"})),
    ("gate_up_projection_outputs", None, frozenset({"gate", "up"})),
    ("silu_product_output", "silu", frozenset()),
    ("down_projection_output", None, frozenset({"down"})),
)


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
    return localizer.file_record(path)


def write_logits(path: Path, values: torch.Tensor) -> dict[str, Any]:
    array = values.detach().cpu().to(torch.float64).contiguous().numpy().astype("<f8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes())
    record = localizer.file_record(path)
    record.update({"dtype": "float64_le", "shape": list(array.shape)})
    return record


def sealed_records() -> dict[str, dict[str, Any]]:
    paths = {
        "manifest": SEALED_ATTEMPT / "manifest.json",
        "run_summary": SEALED_ATTEMPT / "run_summary.json",
        "step0_final_rmsnorm_s8": (
            SEALED_ATTEMPT / "heads/step-00/tensors/final_rmsnorm_s8.bin"
        ),
        "step0_final_rmsnorm_scale": (
            SEALED_ATTEMPT / "heads/step-00/tensors/final_rmsnorm_scale_f64le.bin"
        ),
        "step0_lm_head_output_s8": (
            SEALED_ATTEMPT / "heads/step-00/tensors/lm_head_output_s8.bin"
        ),
        "step0_lm_head_output_scale": (
            SEALED_ATTEMPT / "heads/step-00/tensors/lm_head_output_scale_f64le.bin"
        ),
    }
    return {name: localizer.file_record(path) for name, path in paths.items()}


def accepted_lm_head_scores(
    item: dict[str, Any],
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    float_outputs = []
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        float_outputs.append(
            torch.mv(
                embedding[start:stop].to(torch.float32),
                item["float_norm"],
            )
        )
    float_output = torch.cat(float_outputs).contiguous()
    output_scale = float(backend.canonical.scale_for(float_output))
    multiplier, right_shift = backend.canonical.derive_multiplier(
        float(item["final_scale"]) * head["weight_scale"] / output_scale
    )
    activation = item["final_q"].to(torch.int32)
    accumulator = torch.empty((backend.MODEL_OUTPUT_DOMAIN,), dtype=torch.int64)
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        accumulator[start:stop] = (
            head["qweight"][start:stop].to(torch.int32) * activation
        ).sum(dim=1, dtype=torch.int64)
    require(
        bool(torch.all(accumulator >= -(1 << 31)))
        and bool(torch.all(accumulator < (1 << 31))),
        "LM-head accumulator overflow",
    )
    rounded = backend.lm_head.round_outputs(accumulator, multiplier, right_shift)
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    scores = output_q.to(torch.float64) * output_scale
    return scores.contiguous(), output_q.contiguous(), output_scale


def comparison_record(
    record: dict[str, Any],
    item: dict[str, Any],
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    bf16_logits: torch.Tensor,
    tokenizer: Any,
    label: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    scores, output_q, output_scale = accepted_lm_head_scores(
        item, embedding, head
    )
    comparison = quality.score_comparison(
        bf16_logits.numpy(), scores.numpy()
    )
    top_token = int(comparison["candidate_top_token_id"])
    record.update(
        {
            "top_token_id": top_token,
            "top_piece": localizer.decode_piece(tokenizer, top_token),
            "reference_rank": int(
                comparison["reference_top_rank_in_candidate"]
            ),
            "reference_logit": float(scores[REFERENCE_TOKEN]),
            "top_logit": float(scores[top_token]),
            "top_candidates": localizer.stable_top(scores, tokenizer, TOP_K),
            "lm_head_output_scale": output_scale,
            "lm_head_output_s8_sha256": localizer.tensor_sha256(output_q),
            "accepted_w4a8_logits": write_logits(
                OUTPUT / "logits" / f"{label}-f64le.bin", scores
            ),
            "comparison_vs_checkpoint176_bf16": comparison,
        }
    )
    return record, output_q


def materially_improves(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    baseline_comparison = baseline["comparison_vs_checkpoint176_bf16"]
    candidate_comparison = candidate["comparison_vs_checkpoint176_bf16"]
    baseline_rank = int(baseline_comparison["reference_top_rank_in_candidate"])
    candidate_rank = int(candidate_comparison["reference_top_rank_in_candidate"])
    baseline_js = float(baseline_comparison["jensen_shannon_divergence_nats"])
    candidate_js = float(candidate_comparison["jensen_shannon_divergence_nats"])
    return (
        candidate_rank <= max(TOP_K, baseline_rank // 10)
        and int(candidate_comparison["top8_overlap_count"])
        > int(baseline_comparison["top8_overlap_count"])
        and candidate_js <= baseline_js - 0.05
    )


def capture_bf16_activations(
    snapshot: Path,
    layer_index: int,
    token_ids: list[int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
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
    layer = model.model.model.layers[layer_index]
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def capture_output(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            require(isinstance(value, torch.Tensor), f"BF16 capture differs: {name}")
            captures[name] = value[0].float().cpu().contiguous()

        return hook

    handles.append(
        layer.input_layernorm.register_forward_hook(capture_output("input_norm"))
    )
    handles.append(
        layer.self_attn.q_proj.register_forward_hook(capture_output("q"))
    )
    handles.append(
        layer.self_attn.k_proj.register_forward_hook(capture_output("k"))
    )
    handles.append(
        layer.self_attn.v_proj.register_forward_hook(capture_output("v"))
    )
    handles.append(
        layer.self_attn.o_proj.register_forward_hook(capture_output("o"))
    )
    handles.append(
        layer.post_attention_layernorm.register_forward_hook(
            capture_output("post_norm")
        )
    )
    handles.append(layer.mlp.gate_proj.register_forward_hook(capture_output("gate")))
    handles.append(layer.mlp.up_proj.register_forward_hook(capture_output("up")))

    def capture_silu(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        captures["silu"] = inputs[0][0].float().cpu().contiguous()

    handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture_silu))
    handles.append(layer.mlp.down_proj.register_forward_hook(capture_output("down")))
    handles.append(layer.register_forward_hook(capture_output("layer_output")))

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
    require(int(torch.argmax(logits)) == REFERENCE_TOKEN, "captured BF16 top token changed")
    require(
        localizer.tensor_sha256(logits) == BF16_LOGITS_SHA256,
        "captured BF16 logits differ from frozen diagnosis",
    )
    require(
        set(captures)
        == {
            "input_norm",
            "q",
            "k",
            "v",
            "o",
            "post_norm",
            "gate",
            "up",
            "silu",
            "down",
            "layer_output",
        },
        "BF16 activation capture set differs",
    )
    del model, output, input_ids
    gc.collect()
    return captures, logits


class BoundaryProjectionCache(localizer.FastProjectionCache):
    """Inject exact BF16 activations at one execution-ordered layer boundary."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        self.layer_index = -1
        self.pattern = re.compile(r"a^")
        self.captures: dict[str, torch.Tensor] = {}
        self.input_boundary: str | None = None
        self.output_boundaries: set[str] = set()

    def configure(
        self,
        layer_index: int,
        captures: dict[str, torch.Tensor],
        input_boundary: str | None,
        output_boundaries: frozenset[str],
    ) -> None:
        self.layer_index = layer_index
        self.pattern = re.compile(
            rf"layer{layer_index}_position(\d+)_(q|k|v|o|gate|up|down)"
        )
        self.captures = captures
        self.input_boundary = input_boundary
        self.output_boundaries = set(output_boundaries)

    def clear_substitution(self) -> None:
        self.layer_index = -1
        self.pattern = re.compile(r"a^")
        self.captures = {}
        self.input_boundary = None
        self.output_boundaries.clear()

    def _projection_inputs(
        self,
        name: str,
        input_q: torch.Tensor,
        input_scale: float,
    ) -> tuple[torch.Tensor, re.Match[str] | None]:
        match = self.pattern.fullmatch(name)
        if match is None or self.input_boundary is None:
            return input_q, match
        key = match.group(2)
        expected_keys = {
            "input_norm": {"q", "k", "v"},
            "post_norm": {"gate", "up"},
            "silu": {"down"},
        }[self.input_boundary]
        if key in expected_keys:
            position = int(match.group(1))
            input_q = backend.canonical.quantize_int8(
                self.captures[self.input_boundary][position], input_scale
            )
        return input_q, match

    def _projection_output(
        self,
        result: dict[str, Any],
        match: re.Match[str] | None,
    ) -> dict[str, Any]:
        if match is None or match.group(2) not in self.output_boundaries:
            return result
        position = int(match.group(1))
        key = match.group(2)
        result["output_q"] = backend.canonical.quantize_int8(
            self.captures[key][position], float(result["output_scale"])
        )
        result["same_prompt_causal_substitution"] = {
            "layer": self.layer_index,
            "boundary": key,
            "position": position,
            "source": "checkpoint176_exact_bf16_activation",
        }
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
        input_q, match = self._projection_inputs(name, input_q, input_scale)
        result = super().derive_projection(
            name, merged, input_q, input_scale, float_input, source_hashes
        )
        return self._projection_output(result, match)

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
        input_q, match = self._projection_inputs(name, input_q, input_scale)
        result = super().from_fixed_metadata(
            name,
            input_q,
            input_scale,
            qweight,
            multiplier,
            right_shift,
            output_scale,
        )
        return self._projection_output(result, match)


def run_cut_record(
    cut: int | None,
    label: str,
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
    bf16_logits: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    require(cut is None or 0 <= cut <= backend.LAYERS, "layer cut outside model")
    start_layer = 0 if cut is None else cut
    caches = [backend.empty_layer_cache() for _ in range(backend.LAYERS)]
    templates: list[dict[str, Any] | None] = [None] * backend.LAYERS
    state: dict[str, Any] | None = None
    started = time.monotonic()
    with torch.no_grad():
        for position, token_id in enumerate(localizer.TOKEN_IDS):
            if cut is None:
                state = backend.embedding_state(weights, token_id, position)
            else:
                state = localizer.substitute_state(
                    hidden_states[cut][position], position, token_id
                )
            for layer_id in range(start_layer, backend.LAYERS):
                _derived, state, templates[layer_id] = backend.derive_layer_token(
                    layer_id,
                    state,
                    caches[layer_id],
                    templates[layer_id],
                    weights,
                    adapter,
                )
                del _derived
    require(state is not None, "layer cut produced no final state")
    item = backend.derive_final_rmsnorm_case(state, norm_gain)
    record = {
        "cut": "baseline_w4a8" if cut is None else cut,
        "substitution": (
            "none"
            if cut is None
            else (
                f"checkpoint-176 BF16 hidden_states[{cut}] requantized to "
                "signed A8 for every frozen prefix position"
            )
        ),
        "layers_executed_per_position": backend.LAYERS - start_layer,
        "elapsed_wall_seconds": time.monotonic() - started,
        "final_rmsnorm_q_sha256": localizer.tensor_sha256(item["final_q"]),
        "final_rmsnorm_scale": float(item["final_scale"]),
        "layer_output_q_sha256": localizer.tensor_sha256(state["fixed_q"]),
        "layer_output_scale": float(state["fixed_scale"]),
    }
    return comparison_record(
        record, item, embedding, head, bf16_logits, tokenizer, label
    )


def choose_refinement_anchor(
    baseline: dict[str, Any], coarse: list[dict[str, Any]]
) -> tuple[int | None, str | None]:
    exact = [record for record in coarse if record["top_token_id"] == REFERENCE_TOKEN]
    if exact:
        return int(exact[0]["cut"]), "exact_token_recovery"
    material = [record for record in coarse if materially_improves(baseline, record)]
    if material:
        return int(material[0]["cut"]), "material_rank_and_distribution_recovery"
    return None, None


def minimal_repair(layer_index: int, boundary: str | None) -> str:
    if boundary == "q_projection_output":
        return (
            f"Candidate-only: replace the tensor-wide signed-A8 scale at "
            f"model.layers.{layer_index}.self_attn.q_proj.output with model-derived "
            "grouped activation scaling; leave W4 weights, other boundaries, the "
            "streaming-memory contract, and sealed attempt-0002 unchanged."
        )
    if boundary is None:
        return (
            f"No operator-level repair is justified at model.layers.{layer_index}; "
            "extend causal substitution only from the last tested boundary."
        )
    return (
        f"Candidate-only: confine a grouped activation-scale experiment to "
        f"model.layers.{layer_index} boundary {boundary}; do not change W4 weights or "
        "any sealed or official artifact."
    )


def prepare_frozen_contract(
    snapshot: Path,
    tokenizer: Any,
    token_ids: list[int],
    sealed_before: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    require(localizer.sha256_bytes(PROMPT.encode()) == PROMPT_SHA256, "prompt hash changed")
    require(
        localizer.sha256_bytes(canonical_bytes(token_ids)) == TOKEN_IDS_SHA256,
        "prompt token-ID hash changed",
    )
    require(len(token_ids) == 34, "prompt token count changed")
    require(
        localizer.sha256_bytes(tokenizer.chat_template.encode()) == CHAT_TEMPLATE_SHA256,
        "chat template hash changed",
    )
    require(localizer.sha256_file(snapshot / "model.safetensors") == MODEL_SHA256, "model hash changed")
    require(localizer.sha256_file(backend.canonical.ADAPTER) == ADAPTER_SHA256, "adapter hash changed")
    require(
        localizer.sha256_file(backend.canonical.ADAPTER_CONFIG)
        == ADAPTER_CONFIG_SHA256,
        "adapter config hash changed",
    )
    require(
        backend.canonical.CHECKPOINT_TREE_SHA256 == CHECKPOINT_TREE_SHA256,
        "checkpoint tree hash changed",
    )
    require(
        localizer.sha256_file(snapshot / "tokenizer.json") == TOKENIZER_JSON_SHA256,
        "tokenizer.json hash changed",
    )
    require(
        localizer.sha256_file(snapshot / "tokenizer_config.json")
        == TOKENIZER_CONFIG_SHA256,
        "tokenizer config hash changed",
    )
    require(
        localizer.sha256_file(PRIOR_DIAGNOSIS) == PRIOR_DIAGNOSIS_SHA256,
        "prior diagnosis hash changed",
    )
    return {
        "mission_id": MISSION_ID,
        "classification": "bounded_nonofficial_software_only_causal_diagnostic",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prompt": {
            "sha256": PROMPT_SHA256,
            "token_ids_sha256": TOKEN_IDS_SHA256,
            "token_count": len(token_ids),
            "text_persisted": False,
        },
        "tokenizer": {
            "repository": generation_runner.MODEL_REPOSITORY,
            "revision": generation_runner.REVISION,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "length": len(tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "special_token_ids": list(map(int, tokenizer.all_special_ids)),
        },
        "model": {
            "repository": generation_runner.MODEL_REPOSITORY,
            "revision": generation_runner.REVISION,
            "model_sha256": MODEL_SHA256,
            "checkpoint": "checkpoint-176",
            "adapter_model_sha256": ADAPTER_SHA256,
            "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
            "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
        },
        "generation": {
            "position": 0,
            "selection": "greedy over 151936 outputs; lowest token ID wins exact ties",
            "sampling": False,
            "checkpoint176_bf16_token_id": REFERENCE_TOKEN,
            "checkpoint176_w4a8_token_id": BASELINE_TOKEN,
        },
        "comparison_policy": {
            "source": localizer.file_record(
                ROOT / "tools/diagnose_stage1_quality_attempt0002.py"
            ),
            "top_k": TOP_K,
            "target_token_rank": "one-based stable descending rank with lower token ID tie-break",
            "top_k_overlap": "set intersection count over stable top-8 token IDs",
            "jensen_shannon_divergence": "natural-log softmax probability JSD in nats",
            "material_recovery": (
                "target rank improves by at least 10x to rank <= max(8, baseline_rank//10), "
                "top-8 overlap strictly increases, and JSD decreases by at least 0.05 nats"
            ),
            "l2_policy": (
                "relative L2 is retained as a descriptive logit metric only and never "
                "selects a causal recovery boundary"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
        },
        "bindings": {
            "prior_same_prompt_diagnosis": localizer.file_record(PRIOR_DIAGNOSIS),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "w4a8_backend": localizer.file_record(
                ROOT / "tools/rtl_arbitrary_text_generation_backend.py"
            ),
            "hidden_state_localizer": localizer.file_record(
                ROOT / "tools/ace2_checkpoint176_hidden_state_localizer.py"
            ),
        },
        "sealed_attempt_before": sealed_before,
        "scope_guards": {
            "official_attempt_created": False,
            "official_preflight_created": False,
            "attempt_0003_created": False,
            "full_rtl_generation_run": False,
            "stage2_entered": False,
        },
    }


def generate() -> None:
    require(not OUTPUT.exists(), "same-prompt causal output already exists")
    OUTPUT.mkdir(parents=True)
    started = time.monotonic()
    sealed_before = sealed_records()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    token_ids = generation_runner.canonical_chat_token_ids(tokenizer, PROMPT)
    frozen = prepare_frozen_contract(snapshot, tokenizer, token_ids, sealed_before)

    localizer.PROMPT = PROMPT
    localizer.TOKEN_IDS = token_ids
    localizer.REFERENCE_TOKEN = REFERENCE_TOKEN
    print("ACE2_SAME_PROMPT_CAUSAL_BF16_TRACE", flush=True)
    hidden_states, bf16_logits = localizer.bf16_trace(snapshot, tokenizer)
    require(
        localizer.tensor_sha256(bf16_logits) == BF16_LOGITS_SHA256,
        "BF16 logits differ from stage1qualitydiag01",
    )

    coarse_trace: list[dict[str, Any]] = []
    refinement_trace: list[dict[str, Any]] = []
    operator_trace: list[dict[str, Any]] = []
    suppressed_messages: list[str] = []
    capture_binding: dict[str, Any] = {}
    earliest_cut: int | None = None
    earliest_material_cut: int | None = None
    earliest_operator: str | None = None
    operator_material_boundary: str | None = None

    with safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = BoundaryProjectionCache(weights, adapter)
        guard = localizer.SubstitutionRequireGuard()
        cache.install()
        guard.install()
        try:
            baseline, baseline_output_q = run_cut_record(
                None,
                "baseline-w4a8",
                hidden_states,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
                bf16_logits,
            )
            sealed_final = SEALED_ATTEMPT / "heads/step-00/tensors/final_rmsnorm_s8.bin"
            sealed_output = SEALED_ATTEMPT / "heads/step-00/tensors/lm_head_output_s8.bin"
            sealed_scale = np.fromfile(
                SEALED_ATTEMPT
                / "heads/step-00/tensors/lm_head_output_scale_f64le.bin",
                dtype="<f8",
            )
            require(
                baseline["final_rmsnorm_q_sha256"]
                == localizer.sha256_file(sealed_final),
                "fast W4A8 final RMSNorm differs from sealed attempt-0002",
            )
            require(
                localizer.tensor_sha256(baseline_output_q)
                == localizer.sha256_file(sealed_output),
                "fast W4A8 LM-head output differs from sealed attempt-0002",
            )
            require(
                sealed_scale.shape == (1,)
                and float(sealed_scale[0]) == baseline["lm_head_output_scale"],
                "fast W4A8 LM-head scale differs from sealed attempt-0002",
            )
            require(
                baseline["top_token_id"] == BASELINE_TOKEN,
                "fast W4A8 top token differs from sealed attempt-0002",
            )

            for cut in COARSE_CUTS:
                record, _ = run_cut_record(
                    cut,
                    f"layer-cut-{cut:02d}",
                    hidden_states,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                    bf16_logits,
                )
                coarse_trace.append(record)
                print(
                    "ACE2_SAME_PROMPT_COARSE "
                    f"cut={cut} top={record['top_token_id']} "
                    f"rank={record['reference_rank']} "
                    f"overlap={record['comparison_vs_checkpoint176_bf16']['top8_overlap_count']} "
                    f"js={record['comparison_vs_checkpoint176_bf16']['jensen_shannon_divergence_nats']:.9f}",
                    flush=True,
                )

            anchor, anchor_reason = choose_refinement_anchor(baseline, coarse_trace)
            if anchor is not None:
                previous = max((cut for cut in COARSE_CUTS if cut < anchor), default=0)
                refine_cuts = list(range(previous + 1, anchor))
            else:
                refine_cuts = [
                    cut
                    for cut in range(1, backend.LAYERS + 1)
                    if cut not in COARSE_CUTS
                ]
                anchor_reason = "fallback_exhaustive_no_coarse_recovery"

            for cut in refine_cuts:
                record, _ = run_cut_record(
                    cut,
                    f"layer-cut-{cut:02d}",
                    hidden_states,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                    bf16_logits,
                )
                refinement_trace.append(record)
                print(
                    "ACE2_SAME_PROMPT_REFINE "
                    f"cut={cut} top={record['top_token_id']} "
                    f"rank={record['reference_rank']} "
                    f"overlap={record['comparison_vs_checkpoint176_bf16']['top8_overlap_count']} "
                    f"js={record['comparison_vs_checkpoint176_bf16']['jensen_shannon_divergence_nats']:.9f}",
                    flush=True,
                )

            all_cuts = sorted(
                coarse_trace + refinement_trace, key=lambda item: int(item["cut"])
            )
            exact_cuts = [
                int(record["cut"])
                for record in all_cuts
                if record["top_token_id"] == REFERENCE_TOKEN
            ]
            material_cuts = [
                int(record["cut"])
                for record in all_cuts
                if materially_improves(baseline, record)
            ]
            earliest_cut = min(exact_cuts) if exact_cuts else None
            earliest_material_cut = min(material_cuts) if material_cuts else None
            selected_cut = (
                earliest_cut
                if earliest_cut is not None
                else earliest_material_cut
            )
        finally:
            suppressed_messages.extend(guard.suppressed)
            guard.restore()
            cache.restore()

        if selected_cut is not None and selected_cut > 0:
            implicated_layer = selected_cut - 1
            print(
                f"ACE2_SAME_PROMPT_CAPTURE layer={implicated_layer}",
                flush=True,
            )
            captures, captured_logits = capture_bf16_activations(
                snapshot, implicated_layer, token_ids
            )
            require(
                torch.equal(captured_logits, bf16_logits),
                "operator capture BF16 logits differ from layer trace",
            )
            capture_binding = {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": localizer.tensor_sha256(value),
                }
                for name, value in sorted(captures.items())
            }

            guard = localizer.SubstitutionRequireGuard()
            cache.install()
            guard.install()
            try:
                for boundary, input_boundary, output_boundaries in OPERATOR_VARIANTS:
                    cache.configure(
                        implicated_layer,
                        captures,
                        input_boundary,
                        output_boundaries,
                    )
                    record, _ = run_cut_record(
                        implicated_layer,
                        f"operator-{implicated_layer:02d}-{boundary}",
                        hidden_states,
                        weights,
                        adapter,
                        norm_gain,
                        embedding,
                        head,
                        tokenizer,
                        bf16_logits,
                    )
                    record.update(
                        {
                            "boundary": boundary,
                            "input_activation_substitution": input_boundary,
                            "projection_output_substitutions": sorted(
                                output_boundaries
                            ),
                        }
                    )
                    operator_trace.append(record)
                    print(
                        "ACE2_SAME_PROMPT_OPERATOR "
                        f"layer={implicated_layer} boundary={boundary} "
                        f"top={record['top_token_id']} rank={record['reference_rank']} "
                        f"overlap={record['comparison_vs_checkpoint176_bf16']['top8_overlap_count']} "
                        f"js={record['comparison_vs_checkpoint176_bf16']['jensen_shannon_divergence_nats']:.9f}",
                        flush=True,
                    )
                    if (
                        operator_material_boundary is None
                        and materially_improves(baseline, record)
                    ):
                        operator_material_boundary = boundary
                    if record["top_token_id"] == REFERENCE_TOKEN:
                        earliest_operator = boundary
                        break
            finally:
                cache.clear_substitution()
                suppressed_messages.extend(guard.suppressed)
                guard.restore()
                cache.restore()
        else:
            implicated_layer = None

    sealed_after = sealed_records()
    require(sealed_after == sealed_before, "sealed attempt-0002 changed during diagnosis")
    status = (
        "RECOVERED_EXACT_TOKEN_AND_OPERATOR_BOUNDARY"
        if earliest_cut is not None and earliest_operator is not None
        else (
            "RECOVERED_LAYER_CUT_WITHOUT_OPERATOR_RECOVERY"
            if earliest_cut is not None
            else (
                "MATERIAL_LAYER_RECOVERY_WITHOUT_EXACT_TOKEN"
                if earliest_material_cut is not None
                else "NO_CAUSAL_RECOVERY_OBSERVED"
            )
        )
    )
    selected_operator = earliest_operator or operator_material_boundary
    result = {
        "schema_version": 1,
        "status": status,
        "classification": "candidate_only_diagnosis_no_official_execution",
        "frozen_contract": frozen,
        "bf16_reference": {
            "token_id": REFERENCE_TOKEN,
            "logits_sha256": BF16_LOGITS_SHA256,
            "source": localizer.file_record(
                ROOT
                / "diagnosis/stage1qualitydiag01/logits/"
                "checkpoint176-bf16-greedy-step-00-f32le.bin"
            ),
        },
        "layer_cut_scan": {
            "method": (
                "Run cuts 4,8,12,16,20,24 first; refine every cut after the previous "
                "coarse boundary through the earliest exact/material anchor. If no "
                "coarse recovery exists, exhaustively test all remaining cuts."
            ),
            "coarse_cuts": list(COARSE_CUTS),
            "baseline": baseline,
            "coarse_trace": coarse_trace,
            "refinement_anchor": anchor,
            "refinement_anchor_reason": anchor_reason,
            "refinement_trace": refinement_trace,
            "earliest_exact_token_recovering_cut": earliest_cut,
            "earliest_materially_recovering_cut": earliest_material_cut,
            "implicated_layer": implicated_layer,
        },
        "operator_boundary_scan": {
            "definition": (
                "At the implicated layer, start from the exact checkpoint-176 BF16 "
                "layer input for every one of the 34 prefix positions, inject exact "
                "BF16 activations at the named execution boundary, requantize to the "
                "unchanged signed-A8 consumer scale, and leave all downstream W4A8 "
                "operators, final RMSNorm, and tied W4 head unchanged."
            ),
            "stop_policy": (
                "Stop at the first exact token-39814 recovery; retain the earliest "
                "material rank/top-8/JSD recovery if no exact recovery is observed."
            ),
            "tested_boundaries": operator_trace,
            "earliest_exact_token_recovering_boundary": earliest_operator,
            "earliest_materially_recovering_boundary": operator_material_boundary,
            "bf16_capture_bindings": capture_binding,
        },
        "causal_conclusion": {
            "earliest_recovering_cut": earliest_cut or earliest_material_cut,
            "recovery_kind": (
                "exact_token"
                if earliest_cut is not None
                else (
                    "material_rank_and_distribution"
                    if earliest_material_cut is not None
                    else "none"
                )
            ),
            "narrowest_supported_operator_boundary": selected_operator,
            "l2_maximum_used_as_causal_evidence": False,
            "minimal_repair_recommendation": (
                minimal_repair(implicated_layer, selected_operator)
                if implicated_layer is not None
                else "No repair is supported by this scan."
            ),
        },
        "suppressed_nonfunctional_cache_effect_assertions": {
            "count": len(suppressed_messages),
            "unique_messages": sorted(set(suppressed_messages)),
            "scope": (
                "Only diagnostic zero-effect assertions invalidated by hidden-state "
                "substitution were bypassed; numerical and context checks remained active."
            ),
        },
        "sealed_attempt_after": sealed_after,
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    write_json(RESULT, result)
    review_request = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "review_type": "Fresh-L2",
        "status": "PENDING_INDEPENDENT_REVIEW",
        "result": localizer.file_record(RESULT),
        "source": localizer.file_record(Path(__file__).resolve()),
        "requested_checks": [
            "Recompute frozen prompt, tokenizer, model, adapter, and checkpoint bindings.",
            "Confirm the W4A8 baseline final RMSNorm and LM-head output byte-match sealed attempt-0002.",
            "Recompute token rank, top-8 overlap, and Jensen-Shannon divergence from every persisted logits file.",
            "Confirm the coarse-before-refinement order and earliest recovering layer cut.",
            "Confirm the operator injection semantics and narrowest supported recovering boundary.",
            "Confirm sealed attempt-0002 and official namespaces were not modified.",
        ],
    }
    write_json(REVIEW_REQUEST, review_request)
    localizer.write_sums(OUTPUT)
    print(
        "ACE2_SAME_PROMPT_CAUSAL_RESULT "
        f"status={status} cut={earliest_cut or earliest_material_cut} "
        f"boundary={selected_operator} output={localizer.public_path(OUTPUT)}",
        flush=True,
    )


def records_with_logits(result: dict[str, Any]) -> list[dict[str, Any]]:
    scan = result["layer_cut_scan"]
    records = [scan["baseline"], *scan["coarse_trace"], *scan["refinement_trace"]]
    records.extend(result["operator_boundary_scan"]["tested_boundaries"])
    return records


def verify() -> None:
    require(RESULT.is_file(), "same-prompt causal result is missing")
    require(REVIEW_REQUEST.is_file(), "Fresh-L2 review request is missing")
    require(SUMS.is_file(), "same-prompt causal SHA256SUMS is missing")
    for line in SUMS.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = OUTPUT / relative
        require(path.is_file(), f"summed artifact is missing: {relative}")
        require(
            localizer.sha256_file(path) == expected,
            f"summed artifact hash differs: {relative}",
        )

    result = read_json(RESULT)
    contract = result["frozen_contract"]
    require(contract["mission_id"] == MISSION_ID, "mission binding changed")
    require(contract["prompt"]["sha256"] == PROMPT_SHA256, "prompt binding changed")
    require(
        contract["prompt"]["token_ids_sha256"] == TOKEN_IDS_SHA256,
        "token-ID binding changed",
    )
    require(
        contract["model"]["model_sha256"] == MODEL_SHA256
        and contract["model"]["adapter_model_sha256"] == ADAPTER_SHA256
        and contract["model"]["checkpoint_tree_sha256"] == CHECKPOINT_TREE_SHA256,
        "model binding changed",
    )
    require(
        result["bf16_reference"]["logits_sha256"] == BF16_LOGITS_SHA256,
        "BF16 logits binding changed",
    )
    reference_path = ROOT / result["bf16_reference"]["source"]["path"]
    require(
        localizer.sha256_file(reference_path) == BF16_LOGITS_SHA256,
        "frozen BF16 logits artifact changed",
    )
    reference = np.fromfile(reference_path, dtype="<f4")
    require(
        reference.shape == (backend.MODEL_OUTPUT_DOMAIN,),
        "BF16 reference logits shape changed",
    )
    for record in records_with_logits(result):
        logits_record = record["accepted_w4a8_logits"]
        path = ROOT / logits_record["path"]
        require(localizer.sha256_file(path) == logits_record["sha256"], "logits hash changed")
        scores = np.fromfile(path, dtype="<f8")
        require(
            scores.shape == (backend.MODEL_OUTPUT_DOMAIN,),
            "candidate logits shape changed",
        )
        recomputed = quality.score_comparison(reference, scores)
        require(
            recomputed == record["comparison_vs_checkpoint176_bf16"],
            f"comparison metrics changed at {record['cut']}",
        )
        require(
            int(recomputed["candidate_top_token_id"]) == record["top_token_id"],
            f"top token changed at {record['cut']}",
        )
        require(
            int(recomputed["reference_top_rank_in_candidate"])
            == record["reference_rank"],
            f"reference rank changed at {record['cut']}",
        )

    scan = result["layer_cut_scan"]
    require(scan["coarse_cuts"] == list(COARSE_CUTS), "coarse cut contract changed")
    exact = sorted(
        int(record["cut"])
        for record in [*scan["coarse_trace"], *scan["refinement_trace"]]
        if record["top_token_id"] == REFERENCE_TOKEN
    )
    require(
        scan["earliest_exact_token_recovering_cut"]
        == (exact[0] if exact else None),
        "earliest exact cut changed",
    )
    operator = result["operator_boundary_scan"]
    exact_boundaries = [
        record["boundary"]
        for record in operator["tested_boundaries"]
        if record["top_token_id"] == REFERENCE_TOKEN
    ]
    require(
        operator["earliest_exact_token_recovering_boundary"]
        == (exact_boundaries[0] if exact_boundaries else None),
        "earliest exact operator boundary changed",
    )
    require(
        sealed_records() == contract["sealed_attempt_before"]
        == result["sealed_attempt_after"],
        "sealed attempt-0002 binding changed",
    )
    request = read_json(REVIEW_REQUEST)
    require(
        request["status"] == "PENDING_INDEPENDENT_REVIEW",
        "Fresh-L2 review status changed without a review",
    )
    require(
        request["result"] == localizer.file_record(RESULT),
        "Fresh-L2 request result binding changed",
    )
    require(
        not (ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0003").exists(),
        "attempt-0003 exists",
    )
    print(
        "PASS stage1samepromptcausal01 "
        f"status={result['status']} "
        f"cut={result['causal_conclusion']['earliest_recovering_cut']} "
        f"boundary={result['causal_conclusion']['narrowest_supported_operator_boundary']}",
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
            f"ACE2_SAME_PROMPT_CAUSAL_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
