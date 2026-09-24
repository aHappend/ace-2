#!/usr/bin/env python3
"""Audit source geometry and FP16 metadata viability for the proposed V21 codec."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

import qwen_instruct_option_b as source_identity


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-post-v20-block4-affine-fp16-w4a8-v21"
EXPECTED_LINEAR_TENSORS = 169
EXPECTED_LINEAR_WEIGHT_COUNT = 493_961_216
BLOCK_WIDTH = 4
EXPECTED_BLOCK_COUNT = EXPECTED_LINEAR_WEIGHT_COUNT // BLOCK_WIDTH
ROW_CHUNK = 256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    require(output.is_relative_to((ROOT / "research/probes").resolve()), "output must be under research/probes")
    require(not output.exists(), "refusing to overwrite source audit")
    require(not OFFICIAL_ROOT.exists(), "official V21 namespace must remain absent")

    source_identity.verify_source_snapshot()
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

    tensor_records = []
    total_weights = 0
    total_blocks = 0
    constant_blocks = 0
    nonconstant_zero_step_blocks = 0
    nonfinite_metadata_blocks = 0
    for ordinal, (module_name, module) in enumerate(modules, start=1):
        print(f"POST_V20_BLOCK4_FP16_SOURCE_AUDIT {ordinal}/{len(modules)} {module_name}", flush=True)
        source_weight = module.weight.detach().cpu()
        rows, width = (int(value) for value in source_weight.shape)
        require(width % BLOCK_WIDTH == 0, f"block width does not divide {module_name}")
        module_blocks = rows * (width // BLOCK_WIDTH)
        module_constant = 0
        module_zero_step = 0
        module_nonfinite = 0
        for row_start in range(0, rows, ROW_CHUNK):
            row_stop = min(rows, row_start + ROW_CHUNK)
            grouped = source_weight[row_start:row_stop].to(torch.float32).reshape(
                row_stop - row_start, width // BLOCK_WIDTH, BLOCK_WIDTH
            )
            minima = grouped.amin(dim=-1)
            maxima = grouped.amax(dim=-1)
            constant = maxima == minima
            offset = minima.to(torch.float16)
            step = ((maxima - minima) / 15.0).to(torch.float16)
            module_constant += int(constant.sum())
            module_zero_step += int((~constant & (step <= 0)).sum())
            module_nonfinite += int((~torch.isfinite(offset) | ~torch.isfinite(step)).sum())
        tensor_records.append(
            {
                "ordinal": ordinal,
                "module": module_name,
                "shape": [rows, width],
                "weight_count": rows * width,
                "block_count": module_blocks,
                "constant_block_count": module_constant,
                "nonconstant_zero_fp16_step_block_count": module_zero_step,
                "nonfinite_fp16_metadata_block_count": module_nonfinite,
            }
        )
        total_weights += rows * width
        total_blocks += module_blocks
        constant_blocks += module_constant
        nonconstant_zero_step_blocks += module_zero_step
        nonfinite_metadata_blocks += module_nonfinite

    require(total_weights == EXPECTED_LINEAR_WEIGHT_COUNT, "whole-model Linear weight count differs")
    require(total_blocks == EXPECTED_BLOCK_COUNT, "whole-model block count differs")
    require(nonconstant_zero_step_blocks == 0, "source contains nonconstant blocks with zero FP16 step")
    require(nonfinite_metadata_blocks == 0, "source contains non-finite FP16 metadata")

    geometry_manifest = canonical_bytes(tensor_records)
    payload_bytes = EXPECTED_LINEAR_WEIGHT_COUNT // 2
    metadata_bytes = EXPECTED_BLOCK_COUNT * 4
    total_bytes = payload_bytes + metadata_bytes
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_NON_CONSUMING_BLOCK4_FP16_SOURCE_GEOMETRY",
        "candidate_id": "option_b_post_v20_block4_affine_fp16_offset_step_w4a8_v21",
        "claim_boundary": "Pinned source-weight geometry and FP16 metadata viability only; no tokenizer, prompts, model forwards, text, candidate construction, campaign authority, RTL, or product-quality claim.",
        "source": {
            "repository": source_identity.REPOSITORY,
            "revision": source_identity.REVISION,
            "snapshot": str(source_identity.SNAPSHOT),
        },
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("huggingface-hub", "safetensors", "tokenizers", "torch", "transformers")
        },
        "geometry": {
            "block_width": BLOCK_WIDTH,
            "linear_tensor_count": len(modules),
            "linear_weight_count": total_weights,
            "block_count": total_blocks,
            "constant_block_count": constant_blocks,
            "nonconstant_zero_fp16_step_block_count": nonconstant_zero_step_blocks,
            "nonfinite_fp16_metadata_block_count": nonfinite_metadata_blocks,
            "geometry_manifest_sha256": sha256_bytes(geometry_manifest),
            "tensor_records": tensor_records,
        },
        "storage": {
            "four_bit_payload_bytes": payload_bytes,
            "metadata_bytes": metadata_bytes,
            "total_linear_weight_bytes": total_bytes,
            "exact_payload_bits_per_weight": 4.0,
            "exact_total_bits_per_weight": total_bytes * 8 / total_weights,
        },
        "official_namespace_absent": not OFFICIAL_ROOT.exists(),
        "wall_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(report)
    output.write_bytes(raw)
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "sha256": sha256_bytes(raw), "status": report["status"], "storage": report["storage"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
