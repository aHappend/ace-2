#!/usr/bin/env python3
"""Non-consuming BF16/W4-only/W4A8 diagnosis for the immutable V20 regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

import run_e248e307675a_block32_affine_fp32_w4a8 as campaign_backend
from qwen_instruct_option_b import SNAPSHOT, file_record, load_json, require, verify_source_snapshot, verify_versions


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
SOURCE_BASELINE = ROOT / "research/probes/qwen-instruct-source-relative-quality-v2-repair1-20260807T111432Z.json"
V20_ROOT = ROOT / "build/stage1-option-b-post-v19-block16-affine-bf16-w4a8-v20"
V20_RAW = V20_ROOT / "quality-campaign-0001/raw_outputs.jsonl"
V20_RUBRIC = V20_ROOT / "quality-campaign-0001/source_relative_rubric.json"
V20_MANIFEST = V20_ROOT / "candidate_b_manifest.json"
V20_REVIEW = ROOT / "research/raw/specification/v20-block16-affine-bf16-quality-no-go-review-replan-20260807T161101Z.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
BASE_SCALES = campaign_backend.BASE_SCALES
BLOCK_WIDTH = 16
ROW_CHUNK = 256
EXPECTED_LINEAR_COUNT = 169
EXPECTED_SSE = 766.2914187726615
ALIGNED_TRACE_STEPS = 8
MAX_GENERATED_TOKENS = 40
QUALITY_THREADS = 32
DynamicPolicy = campaign_backend.quality.v10.v7.v3.AuditedDynamicPolicy


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(value: Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(order="C")


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def token_ids_sha256(values: list[int]) -> str:
    raw = b"".join(int(value).to_bytes(4, "little", signed=False) for value in values)
    return hashlib.sha256(raw).hexdigest()


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def immutable_tree_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    body = {"file_count": len(records), "files": records}
    return {**body, "manifest_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()}


def load_case() -> dict[str, Any]:
    baseline = load_json(SOURCE_BASELINE)
    source = next(item for item in baseline["responses"] if item["case_id"] == "concise_summary")
    source_rubric = next(item for item in baseline["rubric"]["responses"] if item["case_id"] == "concise_summary")
    candidate = None
    for line in V20_RAW.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("case_id") == "concise_summary":
            candidate = record
            break
    require(candidate is not None, "V20 concise_summary record is missing")
    v20_rubric = load_json(V20_RUBRIC)
    candidate_rubric = next(item for item in v20_rubric["responses"] if item["case_id"] == "concise_summary")
    require(source["input_token_ids"] == candidate["input_token_ids"], "source and V20 input IDs differ")
    require(source_rubric["action_pass"] is True, "source concise_summary is not a frozen pass")
    require(candidate_rubric["action_pass"] is False, "V20 concise_summary is not a frozen failure")
    return {
        "input_ids": [int(value) for value in source["input_token_ids"]],
        "source_tokens": [int(value) for value in source["generated_token_ids"]],
        "candidate_tokens": [int(value) for value in candidate["generated_token_ids"]],
        "source_record": {
            "input_token_ids_sha256": source["input_token_ids_sha256"],
            "generated_token_ids_sha256": source["generated_token_ids_sha256"],
            "decoded_text_sha256": source["decoded_text_sha256"],
            "action_pass": True,
        },
        "candidate_record": {
            "input_token_ids_sha256": candidate["input_token_ids_sha256"],
            "generated_token_ids_sha256": candidate["generated_token_ids_sha256"],
            "decoded_text_sha256": candidate["decoded_text_sha256"],
            "action_pass": False,
        },
    }


def encode_block16(grouped: Tensor) -> Tensor:
    require(grouped.dtype == torch.float32 and grouped.shape[-1] == BLOCK_WIDTH, "codec input differs")
    minima = grouped.amin(dim=-1)
    maxima = grouped.amax(dim=-1)
    offsets_bf16 = minima.to(torch.bfloat16)
    offsets = offsets_bf16.to(torch.float32)
    raw_steps = (maxima - offsets) / 15.0
    nonconstant = maxima != offsets
    steps_bf16 = torch.where(
        nonconstant,
        raw_steps.to(torch.bfloat16),
        torch.zeros_like(raw_steps).to(torch.bfloat16),
    )
    steps = steps_bf16.to(torch.float32)
    require(bool(torch.isfinite(offsets).all() and torch.isfinite(steps).all()), "non-finite V20 metadata")
    require(bool((steps[nonconstant] > 0).all()), "nonconstant V20 block has zero step")
    denominator = torch.where(nonconstant, steps, torch.ones_like(steps))
    indices = torch.round((grouped - offsets.unsqueeze(-1)) / denominator.unsqueeze(-1))
    indices = indices.clamp(0, 15).to(torch.uint8)
    indices = torch.where(nonconstant.unsqueeze(-1), indices, torch.zeros_like(indices))
    return (offsets.unsqueeze(-1) + indices.to(torch.float32) * steps.unsqueeze(-1)).to(torch.bfloat16)


def quantize_v20_model(model: nn.Module) -> dict[str, Any]:
    embedding = model.model.embed_tokens.weight
    require(model.lm_head.weight is embedding, "source lm_head alias differs")
    embedding_hash = tensor_sha256(embedding)
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head split failed")

    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == EXPECTED_LINEAR_COUNT, "V20 linear count differs")
    aggregate = hashlib.sha256()
    total_sse = 0.0
    started = time.monotonic()
    with torch.no_grad():
        for name, module in modules:
            weight = module.weight
            require(weight.dtype == torch.bfloat16 and weight.ndim == 2, f"weight format differs: {name}")
            rows, width = (int(value) for value in weight.shape)
            require(width % BLOCK_WIDTH == 0, f"block width differs: {name}")
            module_digest = hashlib.sha256()
            for row_start in range(0, rows, ROW_CHUNK):
                row_stop = min(rows, row_start + ROW_CHUNK)
                source = weight[row_start:row_stop].detach().cpu().to(torch.float32)
                grouped = source.reshape(row_stop - row_start, width // BLOCK_WIDTH, BLOCK_WIDTH)
                reconstructed = encode_block16(grouped).reshape(row_stop - row_start, width)
                difference = source - reconstructed.to(torch.float32)
                total_sse += float(torch.sum(difference * difference, dtype=torch.float64))
                weight[row_start:row_stop].copy_(reconstructed)
                module_digest.update(tensor_bytes(reconstructed))
            aggregate.update(name.encode() + b"\0" + module_digest.digest())
    manifest = load_json(V20_MANIFEST)
    require(abs(total_sse - EXPECTED_SSE) <= 1e-9, "V20 reconstruction SSE differs")
    require(
        aggregate.hexdigest() == manifest["aggregate_tensor_hashes"]["reconstruction"],
        "V20 reconstruction aggregate hash differs",
    )
    require(tensor_sha256(embedding) == embedding_hash == manifest["embedding_bf16_sha256"], "embedding changed")
    return {
        "linear_tensor_count": len(modules),
        "aggregate_literal_bf16_sse": total_sse,
        "aggregate_reconstruction_sha256": aggregate.hexdigest(),
        "embedding_bf16_sha256": embedding_hash,
        "wall_seconds": time.monotonic() - started,
    }


def forward_logits(model: nn.Module, input_ids: Tensor) -> Tensor:
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
    require(bool(torch.isfinite(logits).all()), "non-finite logits")
    return logits.detach().cpu().to(torch.float64)


def top_stats(logits: Tensor) -> dict[str, Any]:
    top = torch.topk(logits, k=5)
    return {
        "top_token_id": int(logits.argmax()),
        "top_five_token_ids": [int(value) for value in top.indices],
        "top_five_logits": [float(value) for value in top.values],
        "top_one_margin": float(top.values[0] - top.values[1]),
    }


def comparison(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    difference = candidate - reference
    denominator = torch.linalg.vector_norm(reference)
    ref_top = int(reference.argmax())
    cand_top = int(candidate.argmax())
    return {
        "argmax_equal": ref_top == cand_top,
        "reference_top_token_id": ref_top,
        "candidate_top_token_id": cand_top,
        "reference_top_logit_in_candidate": float(candidate[ref_top]),
        "candidate_top_logit_in_reference": float(reference[cand_top]),
        "maximum_absolute_logit_error": float(difference.abs().amax()),
        "mean_absolute_logit_error": float(difference.abs().mean()),
        "relative_l2_logit_error": None if float(denominator) == 0.0 else float(torch.linalg.vector_norm(difference) / denominator),
        "top_one_margin_delta": top_stats(candidate)["top_one_margin"] - top_stats(reference)["top_one_margin"],
    }


def generate(model: nn.Module, input_ids: Tensor, termination_ids: set[int]) -> list[int]:
    current = input_ids
    attention_mask = torch.ones_like(input_ids)
    cache: Any | None = None
    generated: list[int] = []
    with torch.inference_mode():
        for _index in range(MAX_GENERATED_TOKENS):
            output = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            logits = output.logits[0, -1]
            require(bool(torch.isfinite(logits).all()), "generation produced non-finite logits")
            token = int(logits.argmax())
            generated.append(token)
            cache = output.past_key_values
            require(cache is not None, "generation cache is missing")
            if token in termination_ids:
                break
            current = torch.tensor([[token]], dtype=input_ids.dtype)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1)
    return generated


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def exact_linear_hook(reference: nn.Linear) -> Callable[[nn.Module, tuple[Any, ...], Any], Tensor]:
    def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> Tensor:
        return F.linear(inputs[0], reference.weight, reference.bias)

    return hook


def evaluate_weight_subset(
    candidate: nn.Module,
    reference_modules: dict[str, nn.Module],
    candidate_modules: dict[str, nn.Module],
    ordered_linears: list[str],
    active_w4: set[str],
    input_ids: Tensor,
) -> Tensor:
    handles = []
    for name in ordered_linears:
        if name not in active_w4:
            reference = reference_modules[name]
            module = candidate_modules[name]
            require(isinstance(reference, nn.Linear) and isinstance(module, nn.Linear), f"linear override differs: {name}")
            handles.append(module.register_forward_hook(exact_linear_hook(reference)))
    try:
        return forward_logits(candidate, input_ids)
    finally:
        for handle in handles:
            handle.remove()


def ordered_weight_groups(model: nn.Module) -> list[tuple[str, list[str]]]:
    groups = []
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        names = [
            f"{prefix}.self_attn.q_proj",
            f"{prefix}.self_attn.k_proj",
            f"{prefix}.self_attn.v_proj",
            f"{prefix}.self_attn.o_proj",
            f"{prefix}.mlp.gate_proj",
            f"{prefix}.mlp.up_proj",
            f"{prefix}.mlp.down_proj",
        ]
        groups.append((f"layer_{index}", names))
    groups.append(("lm_head", ["lm_head"]))
    return groups


def localize_weight_boundary(reference: nn.Module, candidate: nn.Module, input_ids: Tensor) -> dict[str, Any]:
    reference_logits = forward_logits(reference, input_ids)
    reference_token = int(reference_logits.argmax())
    reference_modules = dict(reference.named_modules())
    candidate_modules = dict(candidate.named_modules())
    groups = ordered_weight_groups(candidate)
    ordered = [name for _group, names in groups for name in names]
    require(len(ordered) == EXPECTED_LINEAR_COUNT, "weight localization linear count differs")
    active: set[str] = set()
    baseline = evaluate_weight_subset(candidate, reference_modules, candidate_modules, ordered, active, input_ids)
    require(int(baseline.argmax()) == reference_token, "all-source override does not recover BF16 token")
    layer_trace = []
    responsible_group: tuple[str, list[str]] | None = None
    active_before_group: set[str] = set()
    for group_name, names in groups:
        before = set(active)
        active.update(names)
        logits = evaluate_weight_subset(candidate, reference_modules, candidate_modules, ordered, active, input_ids)
        token = int(logits.argmax())
        layer_trace.append({"group": group_name, "active_w4_linear_count": len(active), "top_token_id": token, "differs_from_bf16": token != reference_token})
        if responsible_group is None and token != reference_token:
            responsible_group = (group_name, names)
            active_before_group = before
            break
    if responsible_group is None:
        return {"status": "NO_CUMULATIVE_WEIGHT_PREFIX_CHANGED_ARGMAX", "layer_trace": layer_trace}

    boundary_trace = []
    active = set(active_before_group)
    responsible_boundary = None
    for name in responsible_group[1]:
        before_logits = evaluate_weight_subset(candidate, reference_modules, candidate_modules, ordered, active, input_ids)
        active.add(name)
        after_logits = evaluate_weight_subset(candidate, reference_modules, candidate_modules, ordered, active, input_ids)
        record = {
            "boundary": name,
            "active_w4_linear_count": len(active),
            "before_top_token_id": int(before_logits.argmax()),
            "after_top_token_id": int(after_logits.argmax()),
            "after_differs_from_bf16": int(after_logits.argmax()) != reference_token,
            "after_vs_bf16": comparison(reference_logits, after_logits),
        }
        boundary_trace.append(record)
        if responsible_boundary is None and record["after_differs_from_bf16"]:
            responsible_boundary = name
            break
    require(responsible_boundary is not None, "responsible weight group has no responsible boundary")
    return {
        "status": "CAUSAL_CUMULATIVE_WEIGHT_PREFIX_LOCALIZED",
        "definition": "first execution-ordered W4 linear prefix that changes the aligned BF16 argmax while every later linear is overridden with the exact BF16 weight",
        "responsible_group": responsible_group[0],
        "responsible_boundary": responsible_boundary,
        "layer_trace": layer_trace,
        "boundary_trace": boundary_trace,
    }


class SelectiveDynamicPolicy(DynamicPolicy):
    def __init__(self, scales: dict[str, Any], active_events: set[str]) -> None:
        super().__init__(scales)
        self.active_events = active_events

    def _apply(self, name: str, value: Tensor, records: Tensor, lanes: int) -> Tensor:
        if name not in self.active_events:
            return value
        return super()._apply(name, value, records, lanes)


def dynamic_event_groups(layer_count: int) -> list[tuple[str, list[str]]]:
    families = [
        "input_rmsnorm",
        "q_proj",
        "k_proj",
        "v_proj",
        "rope_q",
        "rope_k",
        "attention_value",
        "o_proj",
        "post_attention_rmsnorm",
        "mlp_gate_proj",
        "mlp_up_proj",
        "silu_gate",
        "mlp_down_proj",
    ]
    return [(f"layer_{index}", [f"layer_{index}.{family}" for family in families]) for index in range(layer_count)]


def evaluate_dynamic_subset(model: nn.Module, scales: dict[str, Any], events: set[str], input_ids: Tensor) -> Tensor:
    policy = SelectiveDynamicPolicy(scales, events)
    policy.install(model)
    try:
        return forward_logits(model, input_ids)
    finally:
        policy.uninstall()


def localize_dynamic_boundary(model: nn.Module, input_ids: Tensor, scales: dict[str, Any]) -> dict[str, Any]:
    w4_logits = forward_logits(model, input_ids)
    w4_token = int(w4_logits.argmax())
    groups = dynamic_event_groups(len(model.model.layers))
    active: set[str] = set()
    layer_trace = []
    responsible_group: tuple[str, list[str]] | None = None
    active_before_group: set[str] = set()
    for group_name, names in groups:
        before = set(active)
        active.update(names)
        logits = evaluate_dynamic_subset(model, scales, active, input_ids)
        token = int(logits.argmax())
        layer_trace.append({"group": group_name, "active_dynamic_event_count": len(active), "top_token_id": token, "differs_from_w4_only": token != w4_token})
        if responsible_group is None and token != w4_token:
            responsible_group = (group_name, names)
            active_before_group = before
            break
    if responsible_group is None:
        return {"status": "NO_CUMULATIVE_DYNAMIC_PREFIX_CHANGED_ARGMAX", "layer_trace": layer_trace}

    active = set(active_before_group)
    boundary_trace = []
    responsible_boundary = None
    for name in responsible_group[1]:
        before_logits = evaluate_dynamic_subset(model, scales, active, input_ids)
        active.add(name)
        after_logits = evaluate_dynamic_subset(model, scales, active, input_ids)
        record = {
            "boundary": name,
            "active_dynamic_event_count": len(active),
            "before_top_token_id": int(before_logits.argmax()),
            "after_top_token_id": int(after_logits.argmax()),
            "after_differs_from_w4_only": int(after_logits.argmax()) != w4_token,
            "after_vs_w4_only": comparison(w4_logits, after_logits),
        }
        boundary_trace.append(record)
        if responsible_boundary is None and record["after_differs_from_w4_only"]:
            responsible_boundary = name
            break
    require(responsible_boundary is not None, "responsible dynamic group has no responsible boundary")
    return {
        "status": "CAUSAL_CUMULATIVE_DYNAMIC_PREFIX_LOCALIZED",
        "definition": "first execution-ordered Dynamic Scale32 event prefix that changes the W4-only aligned argmax",
        "responsible_group": responsible_group[0],
        "responsible_boundary": responsible_boundary,
        "layer_trace": layer_trace,
        "boundary_trace": boundary_trace,
    }


def synthetic_ids(length: int, seed: str, vocab_size: int, excluded: set[int]) -> list[int]:
    values = []
    counter = 0
    while len(values) < length:
        digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
        for offset in range(0, len(digest), 4):
            token = int.from_bytes(digest[offset : offset + 4], "little") % vocab_size
            if token not in excluded:
                values.append(token)
                if len(values) == length:
                    break
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    require(ROOT.resolve() in output.resolve().parents, "output must be below repository root")
    require(not output.exists() and not output.with_suffix(output.suffix + ".sha256").exists(), "output already exists")

    started = time.monotonic()
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": verify_versions(),
        "torch_num_threads_for_quality": QUALITY_THREADS,
    }
    source_identity = verify_source_snapshot()
    pipeline_before = load_json(PIPELINE_STATE)
    require(pipeline_before.get("current_stage") == "specification", "pipeline stage differs")
    require(pipeline_before.get("selected_policy_id") is None, "selected policy is not null")
    case = load_case()
    immutable_before = immutable_tree_manifest(V20_ROOT)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    candidate = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    reconstruction = quantize_v20_model(candidate)
    torch.set_num_threads(QUALITY_THREADS)

    prompt = torch.tensor([case["input_ids"]], dtype=torch.long)
    termination_ids = {int(value) for value in campaign_backend.source_identity.TERMINATION_TOKEN_IDS}
    bf16_tokens = generate(reference, prompt, termination_ids)
    w4_tokens = generate(candidate, prompt, termination_ids)
    full_policy = DynamicPolicy(load_json(BASE_SCALES))
    full_policy.install(candidate)
    try:
        w4a8_tokens = generate(candidate, prompt, termination_ids)
    finally:
        full_policy.uninstall()
    require(bf16_tokens == case["source_tokens"], "fresh BF16 sequence does not reproduce frozen source")
    require(w4a8_tokens == case["candidate_tokens"], "fresh W4A8 sequence does not reproduce immutable V20")

    prefixes = []
    for step in range(min(ALIGNED_TRACE_STEPS, len(case["source_tokens"]))):
        prefixes.append(torch.tensor([case["input_ids"] + case["source_tokens"][:step]], dtype=torch.long))
    bf16_logits = [forward_logits(reference, prefix) for prefix in prefixes]
    w4_logits = [forward_logits(candidate, prefix) for prefix in prefixes]
    policy = DynamicPolicy(load_json(BASE_SCALES))
    policy.install(candidate)
    try:
        w4a8_logits = [forward_logits(candidate, prefix) for prefix in prefixes]
    finally:
        policy.uninstall()

    aligned_trace = []
    for step, (bf16, w4, w4a8) in enumerate(zip(bf16_logits, w4_logits, w4a8_logits, strict=True)):
        aligned_trace.append(
            {
                "generation_step": step,
                "incoming_prefix_token_ids_sha256": token_ids_sha256(case["input_ids"] + case["source_tokens"][:step]),
                "bf16": top_stats(bf16),
                "w4_only": top_stats(w4),
                "w4a8": top_stats(w4a8),
                "w4_only_vs_bf16": comparison(bf16, w4),
                "w4a8_vs_bf16": comparison(bf16, w4a8),
                "w4a8_vs_w4_only": comparison(w4, w4a8),
            }
        )

    w4_aligned_divergence = next((row["generation_step"] for row in aligned_trace if not row["w4_only_vs_bf16"]["argmax_equal"]), None)
    w4a8_aligned_divergence = next((row["generation_step"] for row in aligned_trace if not row["w4a8_vs_bf16"]["argmax_equal"]), None)
    activation_aligned_divergence = next((row["generation_step"] for row in aligned_trace if not row["w4a8_vs_w4_only"]["argmax_equal"]), None)

    greedy_dynamic_trace = []
    greedy_dynamic_limit = first_difference(w4_tokens, w4a8_tokens)
    if greedy_dynamic_limit is None:
        greedy_dynamic_limit = min(len(w4_tokens), len(w4a8_tokens)) - 1
    for step in range(max(0, greedy_dynamic_limit) + 1):
        common_prefix = case["input_ids"] + w4_tokens[:step]
        require(w4_tokens[:step] == w4a8_tokens[:step], "dynamic greedy prefix is not common")
        prefix = torch.tensor([common_prefix], dtype=torch.long)
        w4 = forward_logits(candidate, prefix)
        greedy_policy = DynamicPolicy(load_json(BASE_SCALES))
        greedy_policy.install(candidate)
        try:
            w4a8 = forward_logits(candidate, prefix)
        finally:
            greedy_policy.uninstall()
        greedy_dynamic_trace.append(
            {
                "generation_step": step,
                "incoming_prefix_token_ids_sha256": token_ids_sha256(common_prefix),
                "w4_only": top_stats(w4),
                "w4a8": top_stats(w4a8),
                "w4a8_vs_w4_only": comparison(w4, w4a8),
            }
        )
    greedy_dynamic_divergence = next(
        (row["generation_step"] for row in greedy_dynamic_trace if not row["w4a8_vs_w4_only"]["argmax_equal"]),
        None,
    )

    weight_localization = None
    if w4_aligned_divergence is not None:
        weight_localization = localize_weight_boundary(reference, candidate, prefixes[w4_aligned_divergence])
    dynamic_localization = None
    if greedy_dynamic_divergence is not None:
        dynamic_prefix = torch.tensor(
            [case["input_ids"] + w4_tokens[:greedy_dynamic_divergence]],
            dtype=torch.long,
        )
        dynamic_localization = localize_dynamic_boundary(candidate, dynamic_prefix, load_json(BASE_SCALES))

    excluded = termination_ids | {
        int(value)
        for value in (
            getattr(reference.config, "bos_token_id", None),
            getattr(reference.config, "eos_token_id", None),
            getattr(reference.config, "pad_token_id", None),
        )
        if value is not None
    }
    controls = []
    for label, length, seed in (("synthetic_32", 32, "ace2-v20-control-a"), ("synthetic_128", 128, "ace2-v20-control-b")):
        ids = synthetic_ids(length, seed, int(reference.config.vocab_size), excluded)
        tensor = torch.tensor([ids], dtype=torch.long)
        ref_logits = forward_logits(reference, tensor)
        candidate_logits = forward_logits(candidate, tensor)
        control_policy = DynamicPolicy(load_json(BASE_SCALES))
        control_policy.install(candidate)
        try:
            dynamic_logits = forward_logits(candidate, tensor)
        finally:
            control_policy.uninstall()
        controls.append(
            {
                "id": label,
                "source": "deterministic_random_legal_token_ids_independent_of_prompt_and_answer_text",
                "token_count": length,
                "token_ids_sha256": token_ids_sha256(ids),
                "bf16": top_stats(ref_logits),
                "w4_only": top_stats(candidate_logits),
                "w4a8": top_stats(dynamic_logits),
                "w4_only_vs_bf16": comparison(ref_logits, candidate_logits),
                "w4a8_vs_w4_only": comparison(candidate_logits, dynamic_logits),
            }
        )

    bf16_intrinsic = bf16_tokens != case["source_tokens"] or case["source_record"]["action_pass"] is not True
    v20_source_divergence = first_difference(bf16_tokens, w4a8_tokens)
    w4_source_divergence = first_difference(bf16_tokens, w4_tokens)
    dynamic_sequence_divergence = first_difference(w4_tokens, w4a8_tokens)
    weight_primary = (
        w4_aligned_divergence is not None
        and w4a8_aligned_divergence == w4_aligned_divergence
        and aligned_trace[w4_aligned_divergence]["w4_only"]["top_token_id"]
        == aligned_trace[w4_aligned_divergence]["w4a8"]["top_token_id"]
        and (activation_aligned_divergence is None or activation_aligned_divergence > w4_aligned_divergence)
    )
    if bf16_intrinsic:
        classification = "PINNED_BF16_INTRINSIC_OR_REPRODUCTION_DRIFT"
    elif weight_primary and dynamic_sequence_divergence is not None and dynamic_sequence_divergence > w4_aligned_divergence:
        classification = "V20_W4_WEIGHT_PRIMARY_WITH_LATER_DYNAMIC_SCALE32_AMPLIFICATION"
    elif weight_primary:
        classification = "V20_W4_WEIGHT_RECONSTRUCTION_SPECIFIC"
    elif w4_aligned_divergence is None and w4a8_aligned_divergence is not None:
        classification = "DYNAMIC_SCALE32_ACTIVATION_SPECIFIC"
    elif w4_aligned_divergence is not None and activation_aligned_divergence is not None and activation_aligned_divergence <= w4_aligned_divergence:
        classification = "W4_WEIGHT_AND_DYNAMIC_SCALE32_INTERACTION"
    else:
        classification = "UNRESOLVED_PINNED_MODEL_OR_QUANTIZATION_MECHANISM"

    pipeline_after = load_json(PIPELINE_STATE)
    immutable_after = immutable_tree_manifest(V20_ROOT)
    require(pipeline_after == pipeline_before, "pipeline state changed during diagnosis")
    require(immutable_after == immutable_before, "immutable V20 tree changed during diagnosis")
    result = {
        "schema_version": 1,
        "classification": "non_consuming_post_v20_quantization_divergence_diagnosis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "mechanism_class": classification,
            "bf16_intrinsic_failure": bf16_intrinsic,
            "w4_weight_only_first_source_divergence_step": w4_source_divergence,
            "w4a8_first_source_divergence_step": v20_source_divergence,
            "dynamic_scale32_first_incremental_greedy_divergence_step": dynamic_sequence_divergence,
            "w4_weight_is_primary_at_v20_source_divergence": weight_primary,
            "dynamic_scale32_changes_argmax_at_or_before_v20_source_divergence": (
                dynamic_sequence_divergence is not None
                and v20_source_divergence is not None
                and dynamic_sequence_divergence <= v20_source_divergence
            ),
            "fresh_v21_quantization_successor_justified": classification in {
                "V20_W4_WEIGHT_PRIMARY_WITH_LATER_DYNAMIC_SCALE32_AMPLIFICATION",
                "V20_W4_WEIGHT_RECONSTRUCTION_SPECIFIC",
                "DYNAMIC_SCALE32_ACTIVATION_SPECIFIC",
                "W4_WEIGHT_AND_DYNAMIC_SCALE32_INTERACTION",
            },
            "pinned_model_capability_blocker": classification in {
                "PINNED_BF16_INTRINSIC_OR_REPRODUCTION_DRIFT",
                "UNRESOLVED_PINNED_MODEL_OR_QUANTIZATION_MECHANISM",
            },
        },
        "case_binding": {
            "case_id": "concise_summary",
            "input_token_count": len(case["input_ids"]),
            "source": case["source_record"],
            "v20": case["candidate_record"],
            "frozen_source_vs_v20_first_sequence_divergence": first_difference(case["source_tokens"], case["candidate_tokens"]),
        },
        "fresh_reproduction": {
            "bf16_generated_token_ids": bf16_tokens,
            "bf16_generated_token_ids_sha256": token_ids_sha256(bf16_tokens),
            "w4_only_generated_token_ids": w4_tokens,
            "w4_only_generated_token_ids_sha256": token_ids_sha256(w4_tokens),
            "w4a8_generated_token_ids": w4a8_tokens,
            "w4a8_generated_token_ids_sha256": token_ids_sha256(w4a8_tokens),
            "bf16_vs_w4_only_first_sequence_divergence": first_difference(bf16_tokens, w4_tokens),
            "bf16_vs_w4a8_first_sequence_divergence": first_difference(bf16_tokens, w4a8_tokens),
            "w4_only_vs_w4a8_first_sequence_divergence": first_difference(w4_tokens, w4a8_tokens),
        },
        "first_token_and_aligned_logit_trace": {
            "trace_step_count": len(aligned_trace),
            "first_step": aligned_trace[0],
            "first_w4_only_argmax_divergence_step": w4_aligned_divergence,
            "first_w4a8_argmax_divergence_step": w4a8_aligned_divergence,
            "first_dynamic_incremental_argmax_divergence_step": activation_aligned_divergence,
            "steps": aligned_trace,
        },
        "dynamic_greedy_common_prefix_trace": {
            "first_dynamic_incremental_argmax_divergence_step": greedy_dynamic_divergence,
            "steps": greedy_dynamic_trace,
        },
        "causal_localization": {
            "weight": weight_localization,
            "dynamic_scale32": dynamic_localization,
        },
        "source_independent_controls": controls,
        "v20_reconstruction": reconstruction,
        "bindings": {
            "runner": file_record(RUNNER),
            "source_baseline": file_record(SOURCE_BASELINE),
            "v20_raw_outputs": file_record(V20_RAW),
            "v20_rubric": file_record(V20_RUBRIC),
            "v20_candidate_manifest": file_record(V20_MANIFEST),
            "v20_fresh_review": file_record(V20_REVIEW),
            "base_scales": file_record(BASE_SCALES),
            "dynamic_policy_source": file_record(Path(campaign_backend.quality.v10.v7.v3.__file__).resolve()),
            "pipeline_state": file_record(PIPELINE_STATE),
            "immutable_v20_tree_before": immutable_before,
            "immutable_v20_tree_after": immutable_after,
        },
        "source_model": source_identity,
        "environment": environment,
        "scope_guards": {
            "v20_runner_imported_or_executed": False,
            "v20_campaign_replayed": False,
            "official_namespace_created": False,
            "attempt_or_authority_consumed": False,
            "model_weights_changed_beyond_ephemeral_in_memory_v20_reconstruction": False,
            "candidate_frozen_or_selected": False,
            "rtl_u280_simulation_formal_synthesis_timing_or_ppa_executed": False,
            "pipeline_stage_changed": False,
            "selected_policy_id": None,
        },
        "wall_seconds": time.monotonic() - started,
        "claim_boundary": "Bounded local diagnosis only. It distinguishes pinned BF16, V20 W4 reconstruction, and Dynamic Scale32 effects; it does not authorize or consume a successor, select a policy, replay V20, or make RTL, synthesis, PPA, U280, latency, or product-completion claims.",
    }
    raw = canonical_bytes(result)
    write_exclusive(output, raw)
    companion = output.with_suffix(output.suffix + ".sha256")
    write_exclusive(companion, f"{hashlib.sha256(raw).hexdigest()}  {output.relative_to(ROOT).as_posix()}\n".encode())
    print(
        "ACE2_POST_V20_DIAGNOSIS "
        f"mechanism={classification} "
        f"w4_divergence={w4_aligned_divergence} "
        f"w4a8_divergence={w4a8_aligned_divergence} "
        f"dynamic_divergence={activation_aligned_divergence} "
        f"output_sha256={hashlib.sha256(raw).hexdigest()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
