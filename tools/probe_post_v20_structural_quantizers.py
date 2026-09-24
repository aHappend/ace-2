#!/usr/bin/env python3
"""Compare structurally distinct post-V20 W4 layouts without text generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

import qwen_instruct_option_b as source_identity


ROOT = Path(__file__).resolve().parents[1]
V20_MANIFEST = ROOT / "build/stage1-option-b-post-v19-block16-affine-bf16-w4a8-v20/candidate_b_manifest.json"
EXPECTED_LINEAR_TENSORS = 169
EXPECTED_LINEAR_WEIGHT_COUNT = 493_961_216
ROW_CHUNK = 256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_block_affine(source: Tensor, block_width: int, metadata_dtype: torch.dtype) -> Tensor:
    rows, width = (int(value) for value in source.shape)
    require(width % block_width == 0, "block width does not divide source tensor")
    grouped = source.reshape(rows, width // block_width, block_width)
    minima = grouped.amin(dim=-1)
    offsets = minima.to(metadata_dtype).to(torch.float32)
    maxima = grouped.amax(dim=-1)
    steps = ((maxima - minima) / 15.0).to(metadata_dtype).to(torch.float32)
    nonzero = steps > 0
    denominator = torch.where(nonzero, steps, torch.ones_like(steps))
    indices = torch.round((grouped - offsets.unsqueeze(-1)) / denominator.unsqueeze(-1))
    indices = indices.clamp(0, 15)
    indices = torch.where(nonzero.unsqueeze(-1), indices, torch.zeros_like(indices))
    decoded = offsets.unsqueeze(-1) + indices * steps.unsqueeze(-1)
    return decoded.reshape(rows, width).to(torch.bfloat16)


def policy_record(block_width: int, metadata_dtype: str, metadata_bytes_per_scalar: int) -> dict[str, Any]:
    block_count = EXPECTED_LINEAR_WEIGHT_COUNT // block_width
    payload_bytes = EXPECTED_LINEAR_WEIGHT_COUNT // 2
    metadata_bytes = block_count * 2 * metadata_bytes_per_scalar
    total_bytes = payload_bytes + metadata_bytes
    return {
        "block_width": block_width,
        "payload": "one unsigned four-bit affine code per Linear weight",
        "metadata": f"one {metadata_dtype} offset and one {metadata_dtype} step per contiguous row-local block",
        "block_count": block_count,
        "four_bit_payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "total_linear_weight_bytes": total_bytes,
        "exact_payload_bits_per_weight": 4.0,
        "exact_total_bits_per_weight": total_bytes * 8 / EXPECTED_LINEAR_WEIGHT_COUNT,
        "linear_weight_byte_reduction_from_bf16_percent": 100.0
        * (1.0 - total_bytes / (EXPECTED_LINEAR_WEIGHT_COUNT * 2)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    require(output.is_relative_to((ROOT / "research/probes").resolve()), "output must be under research/probes")
    require(not output.exists(), "refusing to overwrite an existing probe")

    source_identity.verify_source_snapshot()
    v20 = json.loads(V20_MANIFEST.read_text(encoding="utf-8"))
    v20_sse = sum(float(item["literal_bf16_sse"]) for item in v20["weight_manifest"])
    v20_by_module = {item["module"]: float(item["literal_bf16_sse"]) for item in v20["weight_manifest"]}

    torch.set_num_threads(1)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        source_identity.SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == EXPECTED_LINEAR_TENSORS, "Linear tensor count differs")

    policies = {
        "block16_affine_bf16_v20_reference": (16, torch.bfloat16, "BF16", 2),
        "block8_affine_bf16": (8, torch.bfloat16, "BF16", 2),
        "block8_affine_fp16": (8, torch.float16, "FP16", 2),
        "block4_affine_bf16": (4, torch.bfloat16, "BF16", 2),
        "block4_affine_fp16": (4, torch.float16, "FP16", 2),
    }
    aggregate_sse = {name: 0.0 for name in policies}
    per_tensor_nonregression = {name: 0 for name in policies}
    per_tensor_strict_improvement = {name: 0 for name in policies}
    weight_count = 0

    with torch.no_grad():
        for ordinal, (module_name, module) in enumerate(modules, start=1):
            print(f"POST_V20_STRUCTURAL_WEIGHT_PROBE {ordinal}/{len(modules)} {module_name}", flush=True)
            source_weight = module.weight.detach().cpu()
            rows, width = (int(value) for value in source_weight.shape)
            weight_count += rows * width
            module_sse = {name: 0.0 for name in policies}
            for row_start in range(0, rows, ROW_CHUNK):
                row_stop = min(rows, row_start + ROW_CHUNK)
                source = source_weight[row_start:row_stop].to(torch.float32)
                for policy_name, (block_width, dtype, _label, _bytes) in policies.items():
                    reconstruction = quantize_block_affine(source, block_width, dtype).to(torch.float32)
                    difference = source - reconstruction
                    module_sse[policy_name] += float(torch.sum(difference * difference, dtype=torch.float64))
            require(
                abs(module_sse["block16_affine_bf16_v20_reference"] - v20_by_module[module_name]) <= 1e-12,
                f"V20 source-only reproduction differs: {module_name}",
            )
            for policy_name, value in module_sse.items():
                aggregate_sse[policy_name] += value
                if value <= v20_by_module[module_name]:
                    per_tensor_nonregression[policy_name] += 1
                if value < v20_by_module[module_name]:
                    per_tensor_strict_improvement[policy_name] += 1

    require(weight_count == EXPECTED_LINEAR_WEIGHT_COUNT, "whole-model Linear weight count differs")
    require(
        abs(aggregate_sse["block16_affine_bf16_v20_reference"] - v20_sse) <= 1e-9,
        "aggregate V20 reproduction differs",
    )

    results = {}
    for name, (block_width, _dtype, label, bytes_per_scalar) in policies.items():
        results[name] = {
            **policy_record(block_width, label, bytes_per_scalar),
            "aggregate_literal_bf16_sse": aggregate_sse[name],
            "relative_sse_vs_v20": aggregate_sse[name] / v20_sse,
            "per_tensor_nonregressing_vs_v20": per_tensor_nonregression[name],
            "per_tensor_strictly_improved_vs_v20": per_tensor_strict_improvement[name],
        }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "non_consuming_source_weight_only_structural_quantizer_comparison",
        "source": {
            "repository": source_identity.REPOSITORY,
            "revision": source_identity.REVISION,
            "snapshot": str(source_identity.SNAPSHOT),
        },
        "v20_manifest": {
            "path": V20_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(V20_MANIFEST),
            "aggregate_literal_bf16_sse": v20_sse,
        },
        "linear_tensor_count": len(modules),
        "linear_weight_count": weight_count,
        "policies": results,
        "selection_boundary": "This marker-free probe uses pinned source weights only. It does not inspect prompts or outputs, select a product policy, create an official namespace, authorize or consume an attempt, or make an RTL, quality, latency, U280, or stage-advance claim.",
        "wall_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(report))
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "sha256": sha256_file(output), "policies": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
