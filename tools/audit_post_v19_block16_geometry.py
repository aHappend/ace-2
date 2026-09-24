#!/usr/bin/env python3
"""Derive fresh block-16 storage geometry from the pinned production model.

This is a marker-free specification fixture.  It loads no tokenizer, generates
no tokens, constructs no quantized candidate, and never creates the proposed
successor's official namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

import qwen_instruct_option_b as source_identity


ROOT = Path(__file__).resolve().parents[1]
BLOCK_WEIGHTS = 16
PAYLOAD_BITS_PER_WEIGHT = 4
METADATA_BYTES_PER_RECORD = 4

EXPECTED_LINEAR_TENSOR_COUNT = 169
EXPECTED_LINEAR_WEIGHT_COUNT = 493_961_216
EXPECTED_RECORD_COUNT = 30_872_576
EXPECTED_PAYLOAD_BYTES = 246_980_608
EXPECTED_METADATA_BYTES = 123_490_304
EXPECTED_TOTAL_BYTES = 370_470_912

V19_ROOT = ROOT / "build/stage1-option-b-post-e248-block16-affine-bf16-w4a8-v1"
V20_ROOT = ROOT / "build/stage1-option-b-post-v19-block16-affine-bf16-w4a8-v20"
V19_REVIEW = ROOT / "research/raw/specification/v19-block16-affine-bf16-construction-preflight-no-go-review-replan-20260807T151344Z.json"


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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_fixture() -> dict[str, Any]:
    require(V19_ROOT.is_dir(), "sealed V19 namespace is missing")
    require((V19_ROOT / "construction_started.json").is_file(), "sealed V19 construction marker is missing")
    require((V19_ROOT / "failure.json").is_file(), "sealed V19 failure record is missing")
    require(V19_REVIEW.is_file(), "authoritative V19 Fresh Reviewer record is missing")
    require(not V20_ROOT.exists(), "fresh V20 official namespace already exists")

    source = source_identity.verify_source_snapshot()
    versions = source_identity.verify_versions()
    model = AutoModelForCausalLM.from_pretrained(
        source_identity.SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()

    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == EXPECTED_LINEAR_TENSOR_COUNT, "production Linear tensor count differs")
    require(len({name for name, _ in modules}) == len(modules), "production Linear module names are not unique")
    require(modules[-1][0] == "lm_head", "production Linear traversal does not end at lm_head")

    records: list[dict[str, Any]] = []
    totals = {
        "linear_tensor_count": 0,
        "linear_weight_count": 0,
        "metadata_record_count": 0,
        "payload_bytes": 0,
        "metadata_bytes": 0,
        "total_bytes": 0,
    }

    for ordinal, (name, module) in enumerate(modules, start=1):
        weight = module.weight
        require(weight.ndim == 2, f"production Linear weight is not rank-2: {name}")
        require(weight.dtype == torch.bfloat16, f"production Linear weight is not BF16: {name}")
        rows, width = (int(value) for value in weight.shape)
        require(width % BLOCK_WEIGHTS == 0, f"block width does not divide production tensor: {name}")
        weight_count = rows * width
        record_count = rows * (width // BLOCK_WEIGHTS)
        payload_bits = weight_count * PAYLOAD_BITS_PER_WEIGHT
        require(payload_bits % 8 == 0, f"payload is not byte aligned: {name}")
        payload_bytes = payload_bits // 8
        metadata_bytes = record_count * METADATA_BYTES_PER_RECORD
        record = {
            "ordinal": ordinal,
            "module": name,
            "shape": [rows, width],
            "block_weights": BLOCK_WEIGHTS,
            "blocks_per_row": width // BLOCK_WEIGHTS,
            "weight_count": weight_count,
            "metadata_record_count": record_count,
            "payload_bytes": payload_bytes,
            "metadata_bytes": metadata_bytes,
            "total_bytes": payload_bytes + metadata_bytes,
        }
        records.append(record)
        totals["linear_tensor_count"] += 1
        totals["linear_weight_count"] += weight_count
        totals["metadata_record_count"] += record_count
        totals["payload_bytes"] += payload_bytes
        totals["metadata_bytes"] += metadata_bytes
        totals["total_bytes"] += payload_bytes + metadata_bytes

    expected = {
        "linear_tensor_count": EXPECTED_LINEAR_TENSOR_COUNT,
        "linear_weight_count": EXPECTED_LINEAR_WEIGHT_COUNT,
        "metadata_record_count": EXPECTED_RECORD_COUNT,
        "payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "metadata_bytes": EXPECTED_METADATA_BYTES,
        "total_bytes": EXPECTED_TOTAL_BYTES,
    }
    require(totals == expected, f"production geometry differs: observed={totals} expected={expected}")
    require(EXPECTED_LINEAR_WEIGHT_COUNT % BLOCK_WEIGHTS == 0, "whole-model block division is not exact")
    require(
        EXPECTED_LINEAR_WEIGHT_COUNT // BLOCK_WEIGHTS == EXPECTED_RECORD_COUNT,
        "whole-model record-count arithmetic differs",
    )
    require(
        EXPECTED_RECORD_COUNT * METADATA_BYTES_PER_RECORD == EXPECTED_METADATA_BYTES,
        "whole-model metadata-byte arithmetic differs",
    )
    require(not V20_ROOT.exists(), "fixture created the fresh V20 official namespace")

    geometry_manifest_sha256 = hashlib.sha256(canonical_bytes(records)).hexdigest()
    return {
        "schema_version": 1,
        "status": "PASS_NON_CONSUMING_FULL_169_TENSOR_PRODUCTION_GEOMETRY",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "specification_stage_marker_free_geometry_only",
        "source": source,
        "package_versions": versions,
        "v19_terminal_review": file_record(V19_REVIEW),
        "fresh_successor_namespace": {
            "path": V20_ROOT.relative_to(ROOT).as_posix(),
            "absent_before_fixture": True,
            "absent_after_fixture": True,
            "construction_marker_created": False,
            "quality_marker_created": False,
        },
        "representation": {
            "block_weights": BLOCK_WEIGHTS,
            "payload_bits_per_weight": PAYLOAD_BITS_PER_WEIGHT,
            "metadata_record": "offset_u16_le,step_u16_le",
            "metadata_bytes_per_record": METADATA_BYTES_PER_RECORD,
        },
        "invariants": {
            "record_count": "sum(rows * (input_width / 16)) over all production Linear tensors",
            "whole_model_division": "493961216 / 16 = 30872576",
            "metadata_bytes": "30872576 * 4 = 123490304",
            "payload_bytes": "493961216 * 4 / 8 = 246980608",
            "total_bytes": "246980608 + 123490304 = 370470912",
        },
        "totals": totals,
        "expected": expected,
        "geometry_manifest_sha256": geometry_manifest_sha256,
        "tensors": records,
        "execution_accounting": {
            "tokenizer_loaded": False,
            "model_forward_count": 0,
            "candidate_text_or_token_output_generated": False,
            "quantized_candidate_constructed": False,
            "official_namespace_created": False,
            "construction_authority_consumed": 0,
            "quality_authority_consumed": 0,
            "rtl_simulation_formal_synthesis_ppa_or_u280_executed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    require(not output.exists(), f"refusing to overwrite geometry fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build_fixture()
    output.write_bytes(canonical_bytes(result))
    print(
        "PASS_NON_CONSUMING_FULL_169_TENSOR_PRODUCTION_GEOMETRY "
        f"records={result['totals']['metadata_record_count']} "
        f"metadata_bytes={result['totals']['metadata_bytes']} "
        f"output={output.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
