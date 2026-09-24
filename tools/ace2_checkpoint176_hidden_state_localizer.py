#!/usr/bin/env python3
"""Candidate-only causal hidden-state localization for checkpoint-176 W4A8.

This runner never enters the official arbitrary-text namespace.  It accelerates
the accepted sequence backend by caching merged/W4 projection metadata, then
substitutes checkpoint-176 BF16 hidden states at one causal layer boundary for
all positions of the canonical 30-token chat prefix.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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


PROMPT = "Hello"
TOKEN_IDS = list(generation_runner.CANONICAL_HELLO_CHAT_TOKEN_IDS)
REFERENCE_TOKEN = 9707
BASELINE_TOKEN = 123781
PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-chat-ranking-repair-v1/candidate-0001"
)
FORBIDDEN = ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1"
LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class LocalizerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalizerError(message)


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


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def tensor_sha256(value: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(value))


def public_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": public_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return file_record(path)


def write_tensor(path: Path, value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tensor_bytes(tensor))
    record = file_record(path)
    record.update({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    return record


def rank_of(values: torch.Tensor, token_id: int) -> int:
    selected = values[token_id]
    ids = torch.arange(values.numel(), device=values.device)
    better = (values > selected) | ((values == selected) & (ids < token_id))
    return int(better.sum().item()) + 1


def decode_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def stable_top(values: torch.Tensor, tokenizer: Any, count: int = 8) -> list[dict[str, Any]]:
    ids = torch.argsort(values, descending=True, stable=True)[:count].tolist()
    return [
        {
            "token_id": int(token_id),
            "piece": decode_piece(tokenizer, int(token_id)),
            "score": float(values[int(token_id)]),
        }
        for token_id in ids
    ]


def round_shift_even_tensor(value: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Vectorized signed round-to-nearest-even matching canonical.round_shift_even."""
    require(value.dtype == torch.int64 and shift.dtype == torch.int64, "RNE dtype differs")
    require(bool(torch.all(shift >= 0)), "negative RNE shift")
    result = value.clone()
    active = shift > 0
    if not bool(active.any()):
        return result
    selected = value[active]
    selected_shift = shift[active]
    magnitude = selected.abs()
    base = torch.bitwise_right_shift(magnitude, selected_shift)
    remainder = magnitude - torch.bitwise_left_shift(base, selected_shift)
    half = torch.bitwise_left_shift(
        torch.ones_like(selected_shift), selected_shift - 1
    )
    increment = (remainder > half) | (
        (remainder == half) & torch.bitwise_and(base, 1).bool()
    )
    rounded = base + increment.to(torch.int64)
    rounded = torch.where(selected < 0, -rounded, rounded)
    result[active] = rounded
    return result


class FastProjectionCache:
    """Exact accepted projection arithmetic with immutable cached W4 metadata."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        self.merged: dict[str, torch.Tensor] = {}
        self.metadata: dict[int, dict[str, torch.Tensor]] = {}
        self.original_merge = backend.canonical.merge_projection
        self.original_derive = backend.canonical.derive_projection
        self.original_fixed = backend.frontier.projection_from_fixed_metadata
        alpha = float(backend.canonical.LORA_ALPHA)
        rank = float(backend.canonical.LORA_RANK)
        for layer_id in range(backend.LAYERS):
            for suffix in LINEAR_SUFFIXES:
                name = f"model.layers.{layer_id}.{suffix}"
                base = weights.get_tensor(name + ".weight").contiguous()
                prefix = "base_model.model." + name
                lora_a = adapter.get_tensor(prefix + ".lora_A.weight").contiguous()
                lora_b = adapter.get_tensor(prefix + ".lora_B.weight").contiguous()
                merged = (
                    base.to(torch.float32)
                    + (alpha / rank)
                    * torch.matmul(lora_b.to(torch.float32), lora_a.to(torch.float32))
                ).contiguous()
                weight = merged.to(torch.float64)
                weight_scale = weight.abs().amax(dim=1) / 7.0
                weight_scale = torch.where(
                    weight_scale > 0, weight_scale, torch.ones_like(weight_scale)
                )
                qweight = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(
                    torch.int8
                )
                self.merged[name] = merged
                self.metadata[merged.data_ptr()] = {
                    "qweight": qweight,
                    "weight_scale": weight_scale,
                }

    def install(self) -> None:
        backend.canonical.merge_projection = self.merge_projection
        backend.canonical.derive_projection = self.derive_projection
        backend.frontier.projection_from_fixed_metadata = self.from_fixed_metadata

    def restore(self) -> None:
        backend.canonical.merge_projection = self.original_merge
        backend.canonical.derive_projection = self.original_derive
        backend.frontier.projection_from_fixed_metadata = self.original_fixed

    def merge_projection(
        self, _weights: Any, _adapter: Any, tensor_name: str
    ) -> tuple[torch.Tensor, dict[str, str]]:
        require(tensor_name in self.merged, f"uncached projection {tensor_name}")
        return self.merged[tensor_name], {"localizer_cache": "checkpoint176_merged_f32"}

    @staticmethod
    def _fixed(
        name: str,
        input_q: torch.Tensor,
        input_scale: float,
        qweight: torch.Tensor,
        multiplier: torch.Tensor,
        right_shift: torch.Tensor,
        output_scale: float,
    ) -> dict[str, Any]:
        accumulator = torch.mv(qweight.to(torch.int64), input_q.to(torch.int64))
        require(
            bool(torch.all(accumulator >= -(1 << 31)))
            and bool(torch.all(accumulator < (1 << 31))),
            f"{name} accumulator overflow",
        )
        product = accumulator * multiplier
        rounded = round_shift_even_tensor(product, right_shift)
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        return {
            "name": name,
            "input_q": input_q,
            "input_scale": input_scale,
            "qweight": qweight,
            "multiplier": multiplier,
            "right_shift": right_shift,
            "accumulator": accumulator,
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": output_scale,
            "saturation": saturation,
        }

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
        return self._fixed(
            name,
            input_q,
            input_scale,
            qweight,
            multiplier,
            right_shift,
            output_scale,
        )

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        metadata = self.metadata.get(merged.data_ptr())
        require(metadata is not None, f"projection metadata absent for {name}")
        float_output = torch.mv(merged, float_input.to(torch.float32)).contiguous()
        output_scale = backend.canonical.scale_for(float_output)
        multiplier, right_shift = backend.canonical.derive_multiplier(
            input_scale * metadata["weight_scale"] / output_scale
        )
        result = self._fixed(
            name,
            input_q,
            input_scale,
            metadata["qweight"],
            multiplier,
            right_shift,
            output_scale,
        )
        result.update(
            {
                "weight_scale": metadata["weight_scale"],
                "float_output": float_output,
                "source_hashes": source_hashes,
            }
        )
        return result


class SubstitutionRequireGuard:
    """Suppress only cache-data-effect probes invalidated by causal substitution."""

    CACHE_EFFECT = re.compile(
        r"layer \d+ decode (?:assigns no cached probability|has no cached-[KV] data effect)"
    )

    def __init__(self) -> None:
        self.original = backend.require
        self.suppressed: list[str] = []

    def install(self) -> None:
        def guarded(condition: bool, message: str) -> None:
            if not condition and self.CACHE_EFFECT.fullmatch(message):
                self.suppressed.append(message)
                return
            self.original(condition, message)

        backend.require = guarded

    def restore(self) -> None:
        backend.require = self.original


def bf16_trace(snapshot: Path, tokenizer: Any) -> tuple[list[torch.Tensor], torch.Tensor]:
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
    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    logits = output.logits[0, -1].float().cpu().contiguous()
    require(int(torch.argmax(logits)) == REFERENCE_TOKEN, "BF16 step-0 token changed")
    hidden_states = [
        item[0].float().cpu().contiguous() for item in output.hidden_states
    ]
    require(len(hidden_states) == backend.LAYERS + 1, "BF16 hidden-state count differs")
    require(
        all(list(item.shape) == [len(TOKEN_IDS), backend.HIDDEN] for item in hidden_states),
        "BF16 hidden-state shape differs",
    )
    del model, output, input_ids
    gc.collect()
    return hidden_states, logits


def substitute_state(hidden: torch.Tensor, position: int, token_id: int) -> dict[str, Any]:
    scale = backend.canonical.scale_for(hidden)
    return {
        "position": position,
        "token_id": token_id,
        "fixed_q": backend.canonical.quantize_int8(hidden, scale),
        "fixed_scale": scale,
        "float_hidden": hidden.to(torch.float32).contiguous(),
    }


def derive_lm_head_weights(embedding: torch.Tensor) -> dict[str, torch.Tensor]:
    qweight = torch.empty(
        (backend.MODEL_OUTPUT_DOMAIN, backend.HIDDEN), dtype=torch.int8
    )
    weight_scale = torch.empty((backend.MODEL_OUTPUT_DOMAIN,), dtype=torch.float64)
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 2048):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 2048)
        rows = embedding[start:stop].to(torch.float64)
        scale = rows.abs().amax(dim=1) / 7.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight[start:stop] = torch.round(rows / scale[:, None]).clamp(-8, 7).to(
            torch.int8
        )
        weight_scale[start:stop] = scale
    return {"qweight": qweight, "weight_scale": weight_scale}


def score_state(
    state: dict[str, Any],
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], torch.Tensor]:
    item = backend.derive_final_rmsnorm_case(state, norm_gain)
    activation = item["final_q"].to(torch.int32)
    accumulator = torch.empty((backend.MODEL_OUTPUT_DOMAIN,), dtype=torch.int64)
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        accumulator[start:stop] = (
            head["qweight"][start:stop].to(torch.int32) * activation
        ).sum(dim=1, dtype=torch.int64)
    real_score = (
        accumulator.to(torch.float64)
        * float(item["final_scale"])
        * head["weight_scale"]
    )
    top = int(torch.argmax(real_score))
    return (
        {
            "top_token_id": top,
            "top_piece": decode_piece(tokenizer, top),
            "reference_rank": rank_of(real_score, REFERENCE_TOKEN),
            "reference_score": float(real_score[REFERENCE_TOKEN]),
            "top_score": float(real_score[top]),
            "final_rmsnorm_q_sha256": tensor_sha256(item["final_q"]),
            "final_rmsnorm_scale": float(item["final_scale"]),
            "layer_output_q_sha256": tensor_sha256(state["fixed_q"]),
            "layer_output_scale": float(state["fixed_scale"]),
            "top_candidates": stable_top(real_score, tokenizer),
        },
        item["final_q"],
    )


def run_cut(
    cut: int | None,
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], torch.Tensor]:
    require(cut is None or 0 <= cut <= backend.LAYERS, "layer cut outside model")
    start_layer = 0 if cut is None else cut
    caches = [backend.empty_layer_cache() for _ in range(backend.LAYERS)]
    templates: list[dict[str, Any] | None] = [None] * backend.LAYERS
    state: dict[str, Any] | None = None
    started = time.monotonic()
    with torch.no_grad():
        for position, token_id in enumerate(TOKEN_IDS):
            if cut is None:
                state = backend.embedding_state(weights, token_id, position)
            else:
                state = substitute_state(
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
    require(state is not None, "no final state")
    score, final_q = score_state(state, norm_gain, embedding, head, tokenizer)
    score.update(
        {
            "cut": "baseline_w4a8" if cut is None else cut,
            "substitution": (
                "none"
                if cut is None
                else f"BF16 hidden_states[{cut}] requantized to signed A8 for every prefix position"
            ),
            "layers_executed_per_position": backend.LAYERS - start_layer,
            "elapsed_wall_seconds": time.monotonic() - started,
        }
    )
    return score, final_q


def write_sums(output: Path) -> None:
    members = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    body = "".join(
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
        for path in members
    )
    (output / "SHA256SUMS").write_text(body, encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cut", type=int, default=backend.LAYERS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(ROOT.resolve() in output.parents, "output must be repository-relative")
    require(output != FORBIDDEN.resolve() and FORBIDDEN.resolve() not in output.parents, "official evidence overlap")
    require(not output.exists(), "candidate output already exists")
    require(1 <= args.max_cut <= backend.LAYERS, "max-cut differs")
    output.mkdir(parents=True)
    started = time.monotonic()

    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    require(
        generation_runner.canonical_chat_token_ids(tokenizer, PROMPT) == TOKEN_IDS,
        "canonical chat tokens changed",
    )
    predecessor_result = PREDECESSOR / "rtl-repair-0001/result.json"
    predecessor_sums = PREDECESSOR / "rtl-repair-0001/SHA256SUMS"
    predecessor_final = PREDECESSOR / "cpu/w4a8/step-00/final-rmsnorm-s8.bin"
    freeze = {
        "schema_version": 1,
        "classification": "candidate_only_hidden_state_localization_no_official_run",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mission": "w4a8-chat-hidden-state-localization-v1",
        "candidate": "candidate-0002",
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
        "prompt": {
            "utf8_hex": PROMPT.encode().hex(),
            "sha256": sha256_bytes(PROMPT.encode()),
            "chat_token_ids": TOKEN_IDS,
            "chat_token_count": len(TOKEN_IDS),
        },
        "reference_step0_token_id": REFERENCE_TOKEN,
        "predecessor_negative_candidate": {
            "result": file_record(predecessor_result),
            "sha256s": file_record(predecessor_sums),
            "step0_final_rmsnorm": file_record(predecessor_final),
            "preserved_baseline_token_id": BASELINE_TOKEN,
        },
        "localization_method": {
            "kind": "progressive_causal_layer_boundary_substitution",
            "definition": (
                "For cut k, checkpoint-176 BF16 hidden_states[k] is substituted for every "
                "one of the 30 causal prefix positions, requantized to signed A8, and layers "
                "k..23 plus final RMSNorm and tied W4 LM head execute unchanged."
            ),
            "token_specific_overrides": False,
            "logit_specific_overrides": False,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "sources": {
            "runner": file_record(Path(__file__).resolve()),
            "sequence_backend": file_record(ROOT / "tools/rtl_arbitrary_text_generation_backend.py"),
            "predecessor_runner": file_record(ROOT / "tools/ace2_checkpoint176_chat_rank_guard_repair.py"),
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_specification_modified": False,
            "u280_or_stage2_entered": False,
        },
    }
    write_json(output / "candidate-freeze.json", freeze)

    print("ACE2_HIDDEN_LOCALIZER_BF16_LOAD", flush=True)
    hidden_states, bf16_logits = bf16_trace(snapshot, tokenizer)
    write_tensor(output / "cpu/reference/step-00-logits-f32le.bin", bf16_logits)

    print("ACE2_HIDDEN_LOCALIZER_W4_CACHE", flush=True)
    with safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        projection_cache = FastProjectionCache(weights, adapter)
        require_guard = SubstitutionRequireGuard()
        projection_cache.install()
        require_guard.install()
        try:
            head = derive_lm_head_weights(embedding)
            baseline, baseline_final_q = run_cut(
                None,
                hidden_states,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            expected_final_hash = sha256_file(predecessor_final)
            require(
                baseline["final_rmsnorm_q_sha256"] == expected_final_hash,
                "fast backend final RMSNorm differs from candidate-0001",
            )
            require(
                baseline["top_token_id"] == BASELINE_TOKEN,
                "fast backend step-0 top token differs from candidate-0001",
            )
            write_tensor(
                output / "cpu/baseline/step-00-final-rmsnorm-s8.bin",
                baseline_final_q,
            )
            trace = [baseline]
            earliest = None
            for cut in range(1, args.max_cut + 1):
                record, final_q = run_cut(
                    cut,
                    hidden_states,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                )
                trace.append(record)
                print(
                    "ACE2_HIDDEN_LOCALIZER_CUT "
                    f"cut={cut} top={record['top_token_id']} "
                    f"reference_rank={record['reference_rank']} "
                    f"seconds={record['elapsed_wall_seconds']:.3f}",
                    flush=True,
                )
                if record["top_token_id"] == REFERENCE_TOKEN:
                    earliest = cut
                    write_tensor(
                        output
                        / f"cpu/layer-cuts/cut-{cut:02d}-step-00-final-rmsnorm-s8.bin",
                        final_q,
                    )
                    break
        finally:
            require_guard.restore()
            projection_cache.restore()

    result = {
        "schema_version": 1,
        "status": (
            "PASS_EARLIEST_CAUSAL_LAYER_CUT_LOCALIZED"
            if earliest is not None
            else "PARTIAL_NO_LAYER_CUT_RECOVERED_REFERENCE_WITHIN_BOUND"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "baseline_reproduction": {
            "passed": True,
            "expected_token_id": BASELINE_TOKEN,
            "expected_final_rmsnorm_sha256": sha256_file(predecessor_final),
            "observed": baseline,
        },
        "bf16_reference": {
            "token_id": REFERENCE_TOKEN,
            "piece": decode_piece(tokenizer, REFERENCE_TOKEN),
            "logits": file_record(output / "cpu/reference/step-00-logits-f32le.bin"),
        },
        "layerwise_causal_substitution": {
            "earliest_recovering_cut": earliest,
            "earliest_implicated_layer": None if earliest is None else earliest - 1,
            "trace": trace,
            "suppressed_nonfunctional_cache_effect_assertions": {
                "count": len(require_guard.suppressed),
                "unique_messages": sorted(set(require_guard.suppressed)),
                "scope": (
                    "Only diagnostic zero-effect assertions were bypassed; causal cache "
                    "length, context bounds, and all numerical checks remained active."
                ),
            },
        },
        "next_required_action": (
            "Separate W4 weight and A8 activation error inside the implicated layer, then repair its earliest operator boundary."
            if earliest is not None
            else "Extend causal attribution through final RMSNorm and tied W4 LM-head weight substitution."
        ),
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    write_json(output / "layer-scan.json", result)
    write_sums(output)
    print(
        "ACE2_HIDDEN_LOCALIZER_RESULT "
        f"status={result['status']} earliest_cut={earliest} "
        f"output={public_path(output)}",
        flush=True,
    )
    return 0 if earliest is not None else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_HIDDEN_LOCALIZER_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
