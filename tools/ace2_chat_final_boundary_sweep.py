#!/usr/bin/env python3
"""Isolate final-RMSNorm and LM-head quality for an ACE2RT2 chat prompt."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ace2_full_model_fixed_point as fixed_point
from tools.ace2_chat_demo import (
    TOKENIZER_SNAPSHOT,
    canonical_bytes,
    decode_generated,
    load_tokenizer,
    read_ace2rt2_package_metadata,
    sha256_bytes,
    sha256_file,
    write_atomic,
)


FROZEN_SCALES = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1"
    / "paired-smoke-20260801-v1/derived_scales.json"
)
ROPE_MECHANISM = "layer0_tile_bfp_score_attention_v1"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def top_k(tokenizer: Any, logits: torch.Tensor, count: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.detach().to(torch.float32), count)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "logit": float(value),
            "decoded_piece": tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }
        for rank, (token_id, value) in enumerate(
            zip(indices.tolist(), values.tolist(), strict=True), start=1
        )
    ]


def raw_stats(value: torch.Tensor) -> dict[str, Any]:
    raw = value.detach().to(torch.float64)
    contiguous = value.detach().cpu().contiguous()
    tensor_sha256 = (
        sha256_bytes(contiguous.view(torch.uint16).numpy().tobytes())
        if contiguous.dtype == torch.bfloat16
        else fixed_point.sha256_tensor(contiguous)
    )
    return {
        "min": float(raw.min()),
        "max": float(raw.max()),
        "zero_fraction": float((raw == 0).to(torch.float64).mean()),
        "saturated_fraction": float(
            ((raw <= -128) | (raw >= 127)).to(torch.float64).mean()
        ),
        "sha256": tensor_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--candidate-final-scales",
        default="1.0,0.5,0.25,0.125,0.0625",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 64:
        raise RuntimeError("--top-k must be in 1..64")

    package = read_ace2rt2_package_metadata(args.package.resolve())
    tokenizer = load_tokenizer()
    input_ids = torch.tensor([package["prompt_tokens"]], dtype=torch.long)
    prompt_manifest = json.loads(fixed_point.PROMPT_MANIFEST.read_text(encoding="utf-8"))
    calibration_spec = prompt_manifest["datasets"]["c4_calibration"]
    calibration_text = fixed_point.selected_texts(calibration_spec, limit=1)
    calibration_prompt = fixed_point.tokenize_prompts(
        tokenizer,
        calibration_text,
        calibration_spec["token_limit"],
    )[0][:, :128]

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    source_final_norm = model.model.norm
    source_lm_head = model.lm_head

    with torch.inference_mode():
        baseline_hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        baseline_logits = source_lm_head(baseline_hidden)[0, -1]

    ranges, operator_ranges = fixed_point.calibrate(model, [calibration_prompt])
    fixed_point.replace_linears(
        model,
        ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
    )
    fixed_point.replace_fixed_operators(
        model,
        operator_ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
    )
    scale_table = fixed_point.derived_scale_table(ranges, operator_ranges, model)
    scale_table_sha256 = sha256_bytes(canonical_bytes(scale_table))
    frozen_scale_sha256 = sha256_file(FROZEN_SCALES)

    captured: dict[str, torch.Tensor] = {}

    def capture_pre(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["pre"] = inputs[0].detach().clone()

    def capture_post(
        _module: Any,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["post"] = output.detach().clone()

    pre_hook = model.model.norm.register_forward_pre_hook(capture_pre)
    post_hook = model.model.norm.register_forward_hook(capture_post)
    try:
        with torch.inference_mode():
            fixed_logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
    finally:
        pre_hook.remove()
        post_hook.remove()

    fixed_norm = model.model.norm
    fixed_lm_head = model.lm_head
    fixed_pre = captured["pre"]
    fixed_post_raw = captured["post"]
    with torch.inference_mode():
        fixed_post_dequantized = fixed_post_raw * fixed_norm.output_scale
        fixed_hidden_bf16_lm_logits = source_lm_head(fixed_post_dequantized)[0, -1]
        upstream_bf16_final_hidden = source_final_norm(fixed_pre)
        upstream_bf16_final_logits = source_lm_head(upstream_bf16_final_hidden)[0, -1]
        baseline_hidden_q = fixed_point.quantize_int8(
            baseline_hidden,
            fixed_norm.output_scale,
        )
        baseline_hidden_fixed_lm_raw = fixed_lm_head.forward_quantized(
            baseline_hidden_q
        )
        baseline_hidden_fixed_lm_logits = (
            baseline_hidden_fixed_lm_raw.to(torch.float64)
            * fixed_lm_head.output_scale_per_channel.to(torch.float64)
        )[0, -1]

    gain_floor = (
        float(source_final_norm.weight.detach().to(torch.float64).abs().amax())
        * (1 << fixed_point.RMS_GAIN_FRAC)
        / 32767.0
    )
    requested_scales = [
        float(value) for value in args.candidate_final_scales.split(",") if value.strip()
    ]
    candidate_scales = []
    for value in [fixed_norm.output_scale, *requested_scales, gain_floor]:
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("candidate final scale must be finite and positive")
        if all(abs(value - previous) > 1e-15 for previous in candidate_scales):
            candidate_scales.append(value)

    sweep = []
    for output_scale in candidate_scales:
        if output_scale < gain_floor:
            sweep.append(
                {
                    "final_rmsnorm_output_scale": output_scale,
                    "final_rmsnorm_gain_floor": gain_floor,
                    "status": "INVALID_BELOW_Q7_8_GAIN_FLOOR",
                }
            )
            continue
        candidate_norm = fixed_point.FixedRMSNorm(
            source_final_norm,
            fixed_norm.input_scale,
            output_scale,
        )
        model.model.norm = candidate_norm
        fixed_lm_head.bind_hardware_input_scale(output_scale)
        candidate_capture: dict[str, torch.Tensor] = {}

        def capture_candidate(
            _module: Any,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            candidate_capture["post"] = output.detach().clone()

        hook = model.model.norm.register_forward_hook(capture_candidate)
        try:
            with torch.inference_mode():
                logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
        finally:
            hook.remove()
        records = top_k(tokenizer, logits, args.top_k)
        sweep.append(
            {
                "status": "EXECUTED",
                "final_rmsnorm_output_scale": output_scale,
                "final_rmsnorm_gain_floor": gain_floor,
                "final_rmsnorm_raw": raw_stats(candidate_capture["post"]),
                "selected_token_id": records[0]["token_id"],
                "selected_piece": records[0]["decoded_piece"],
                "readable_selected_piece": bool(records[0]["decoded_piece"].strip()),
                "top_k": records,
            }
        )

    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_final_boundary_calibration_and_provenance_diagnostic",
        "package": package,
        "calibration": {
            "dataset": "allenai/c4 validation calibration slice",
            "token_count": int(calibration_prompt.numel()),
            "rope_mechanism": ROPE_MECHANISM,
            "derived_scale_table_sha256": scale_table_sha256,
            "frozen_scale_table_sha256": frozen_scale_sha256,
            "exactly_reproduces_frozen_scale_table": scale_table_sha256
            == frozen_scale_sha256,
            "model_norm_input": scale_table["operators"]["model.norm.input"],
            "model_norm_output": scale_table["operators"]["model.norm.output"],
            "lm_head": {
                key: scale_table["linears"]["lm_head"][key]
                for key in (
                    "hardware_input_scale",
                    "input_absmax",
                    "input_scale",
                    "output_absmax",
                    "output_scale",
                )
            },
        },
        "boundaries": {
            "normal_bf16": {
                "top_k": top_k(tokenizer, baseline_logits, args.top_k),
            },
            "full_fixed_w4a8": {
                "top_k": top_k(tokenizer, fixed_logits, args.top_k),
                "final_rmsnorm_input": raw_stats(fixed_pre),
                "final_rmsnorm_output_raw": raw_stats(fixed_post_raw),
                "final_rmsnorm_output_scale": fixed_norm.output_scale,
            },
            "fixed_upstream_and_norm_with_bf16_lm_head": {
                "top_k": top_k(
                    tokenizer, fixed_hidden_bf16_lm_logits, args.top_k
                ),
            },
            "fixed_upstream_with_bf16_final_norm_and_lm_head": {
                "top_k": top_k(tokenizer, upstream_bf16_final_logits, args.top_k),
            },
            "bf16_hidden_with_fixed_lm_head": {
                "quantized_hidden": raw_stats(baseline_hidden_q),
                "top_k": top_k(
                    tokenizer, baseline_hidden_fixed_lm_logits, args.top_k
                ),
            },
        },
        "final_rmsnorm_scale_sweep": sweep,
        "wall_seconds": time.perf_counter() - started,
        "status": "PASS_DIAGNOSTIC_COMPLETE",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, canonical_bytes(result))
    print(
        "ACE2_CHAT_FINAL_BOUNDARY_SWEEP_RESULT "
        f"status={result['status']} "
        f"frozen_scale_reproduced={result['calibration']['exactly_reproduces_frozen_scale_table']} "
        f"fixed_token={result['boundaries']['full_fixed_w4a8']['top_k'][0]['token_id']} "
        f"sweep_tokens={[item.get('selected_token_id') for item in sweep]} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_CHAT_FINAL_BOUNDARY_SWEEP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
