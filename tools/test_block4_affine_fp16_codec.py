#!/usr/bin/env python3
"""Source-independent regression for the proposed block-4 affine FP16 W4 codec."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-post-v20-block4-affine-fp16-w4a8-v21"
BLOCK_WIDTH = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fp16_round(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def scalar_encode(values: list[float]) -> dict[str, Any]:
    require(len(values) == BLOCK_WIDTH, "scalar block width differs")
    require(all(math.isfinite(value) for value in values), "non-finite scalar input")
    minimum = min(values)
    maximum = max(values)
    offset = fp16_round(minimum)
    step = fp16_round((maximum - minimum) / 15.0)
    nonconstant = maximum != minimum
    require(not nonconstant or step > 0.0, "nonconstant scalar block rounded to zero FP16 step")
    codes = []
    for value in values:
        code = 0 if not nonconstant else round((value - offset) / step)
        codes.append(max(0, min(15, int(code))))
    metadata = struct.pack("<ee", offset, step)
    payload = bytes((codes[0] | (codes[1] << 4), codes[2] | (codes[3] << 4)))
    reconstructed = (
        torch.tensor([offset + code * step for code in codes], dtype=torch.float32)
        .to(torch.bfloat16)
        .to(torch.float32)
        .tolist()
    )
    return {
        "codes": codes,
        "metadata": metadata,
        "payload": payload,
        "reconstructed": reconstructed,
    }


def tensor_encode(values: list[float]) -> dict[str, Any]:
    require(len(values) == BLOCK_WIDTH, "tensor block width differs")
    source = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).reshape(1, 1, BLOCK_WIDTH)
    require(bool(torch.isfinite(source).all()), "non-finite tensor input")
    minimum = source.amin(dim=-1)
    maximum = source.amax(dim=-1)
    offset_fp16 = minimum.to(torch.float16)
    step_fp16 = ((maximum - minimum) / 15.0).to(torch.float16)
    offset = offset_fp16.to(torch.float32)
    step = step_fp16.to(torch.float32)
    nonconstant = maximum != minimum
    require(not bool((nonconstant & (step <= 0)).any()), "nonconstant tensor block rounded to zero FP16 step")
    denominator = torch.where(nonconstant, step, torch.ones_like(step))
    codes = torch.round((source - offset.unsqueeze(-1)) / denominator.unsqueeze(-1)).clamp(0, 15).to(torch.uint8)
    codes = torch.where(nonconstant.unsqueeze(-1), codes, torch.zeros_like(codes))
    code_list = [int(value) for value in codes.reshape(-1)]
    offset_bits = offset_fp16.reshape(-1).view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    step_bits = step_fp16.reshape(-1).view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    metadata = offset_bits + step_bits
    payload = bytes((code_list[0] | (code_list[1] << 4), code_list[2] | (code_list[3] << 4)))
    reconstructed = (
        (offset.unsqueeze(-1) + codes.to(torch.float32) * step.unsqueeze(-1))
        .to(torch.bfloat16)
        .to(torch.float32)
        .reshape(-1)
        .tolist()
    )
    return {
        "source": source.reshape(-1).tolist(),
        "codes": code_list,
        "metadata": metadata,
        "payload": payload,
        "reconstructed": reconstructed,
    }


def compare_case(name: str, values: list[float]) -> dict[str, Any]:
    tensor = tensor_encode(values)
    scalar = scalar_encode(tensor["source"])
    require(tensor["codes"] == scalar["codes"], f"{name}: code mismatch")
    require(tensor["metadata"] == scalar["metadata"], f"{name}: metadata mismatch")
    require(tensor["payload"] == scalar["payload"], f"{name}: payload mismatch")
    require(tensor["reconstructed"] == scalar["reconstructed"], f"{name}: reconstruction mismatch")
    return {
        "name": name,
        "source": tensor["source"],
        "codes": tensor["codes"],
        "metadata_hex": tensor["metadata"].hex(),
        "payload_hex": tensor["payload"].hex(),
        "reconstruction": tensor["reconstructed"],
    }


def deterministic_blocks(count: int) -> list[list[float]]:
    state = 0x6D2B79F5
    blocks: list[list[float]] = []
    for index in range(count):
        values = []
        exponent = (index % 21) - 14
        scale = 2.0**exponent
        for _lane in range(BLOCK_WIDTH):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            signed = ((state >> 8) & 0xFFFF) - 32768
            values.append((signed / 32768.0) * scale)
        blocks.append(values)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    require(output.is_relative_to((ROOT / "build").resolve()), "output must be under build")
    require(not output.is_relative_to(OFFICIAL_ROOT.resolve()), "regression output must remain outside official root")
    require(not output.exists(), "refusing to overwrite regression output")
    require(not OFFICIAL_ROOT.exists(), "official V21 namespace must remain absent")

    directed = [
        compare_case("constant_zero", [0.0, 0.0, 0.0, 0.0]),
        compare_case("constant_negative", [-0.5, -0.5, -0.5, -0.5]),
        compare_case("mixed_full_range", [-1.0, -0.5, 0.5, 1.0]),
        compare_case("positive_only", [0.125, 0.25, 0.5, 1.0]),
        compare_case("negative_only", [-1.0, -0.5, -0.25, -0.125]),
        compare_case("small_safe", [0.0, 2.0**-12, 2.0**-11, 2.0**-10]),
        compare_case("large_finite", [-256.0, -64.0, 64.0, 256.0]),
    ]

    sweep_digest = hashlib.sha256()
    sweep_count = 4096
    for index, values in enumerate(deterministic_blocks(sweep_count)):
        result = compare_case(f"sweep_{index:04d}", values)
        sweep_digest.update(bytes.fromhex(result["metadata_hex"]))
        sweep_digest.update(bytes.fromhex(result["payload_hex"]))
        sweep_digest.update(struct.pack("<4f", *result["reconstruction"]))

    rejected = {}
    for name, values in {
        "nan": [0.0, 1.0, math.nan, 2.0],
        "positive_infinity": [0.0, 1.0, math.inf, 2.0],
        "negative_infinity": [0.0, 1.0, -math.inf, 2.0],
        "nonconstant_fp16_step_underflow": [0.0, 0.0, 0.0, 2.0**-24],
    }.items():
        try:
            tensor_encode(values)
        except RuntimeError:
            rejected[name] = True
        else:
            rejected[name] = False
    require(all(rejected.values()), "one or more illegal blocks were not rejected")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SOURCE_INDEPENDENT_QUANTIZER_REGRESSION",
        "candidate_id": "option_b_post_v20_block4_affine_fp16_offset_step_w4a8_v21",
        "claim_boundary": "Synthetic source-independent block-4 affine FP16 codec regression only; no Qwen weights, prompts, outputs, model forwards, candidate construction, campaign authority, RTL, or product-quality claim.",
        "codec": {
            "block_width": BLOCK_WIDTH,
            "payload": "four unsigned uint4 affine codes packed low nibble first",
            "metadata": "little-endian FP16 offset then little-endian FP16 step",
            "decode": "BF16(fp32(offset) + code * fp32(step))",
            "tie_rounding": "round to nearest, ties to even",
            "nonconstant_zero_step": "fail closed",
        },
        "directed_cases": directed,
        "deterministic_sweep": {
            "block_count": sweep_count,
            "record_digest_sha256": sweep_digest.hexdigest(),
            "scalar_tensor_identity": True,
        },
        "illegal_case_rejection": rejected,
        "official_namespace_absent": not OFFICIAL_ROOT.exists(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(report)
    output.write_bytes(raw)
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "sha256": sha256_bytes(raw), "status": report["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
