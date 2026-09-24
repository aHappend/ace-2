#!/usr/bin/env python3
"""Sweep bounded W4A8 activation calibration policies on one chat package."""

from __future__ import annotations

import argparse
import gc
import json
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
    write_atomic,
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def top_k(tokenizer: Any, logits: torch.Tensor, count: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.detach().to(torch.float32), count)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "logit": float(value),
            "decoded_piece": token_piece(tokenizer, int(token_id)),
        }
        for rank, (token_id, value) in enumerate(
            zip(indices.tolist(), values.tolist(), strict=True), start=1
        )
    ]


def generate(
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    termination_token_ids: list[int],
    *,
    steps: int,
    top_k_count: int,
) -> dict[str, Any]:
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    generated: list[int] = []
    records = []
    with torch.inference_mode():
        for generation_index in range(steps):
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
            candidates = top_k(tokenizer, logits, top_k_count)
            selected = int(candidates[0]["token_id"])
            generated.append(selected)
            terminated = selected in termination_token_ids
            records.append(
                {
                    "generation_index": generation_index,
                    "selected_token_id": selected,
                    "selected_piece": token_piece(tokenizer, selected),
                    "terminated": terminated,
                    "top_k": candidates,
                }
            )
            input_ids = torch.cat(
                [input_ids, torch.tensor([[selected]], dtype=torch.long)], dim=1
            )
            if terminated:
                break
    text = decode_generated(tokenizer, generated)
    return {
        "generated_token_ids": generated,
        "decoded_text": text,
        "readable_text_observed": bool(text.strip()),
        "steps": records,
    }


def load_model() -> Any:
    return AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.steps <= 32 or not 1 <= args.top_k <= 64:
        raise RuntimeError("steps/top-k are outside the supported range")

    package = read_ace2rt2_package_metadata(args.package.resolve())
    tokenizer = load_tokenizer()
    prompt_manifest = json.loads(fixed_point.PROMPT_MANIFEST.read_text(encoding="utf-8"))
    calibration_spec = prompt_manifest["datasets"]["c4_calibration"]
    calibration_text = fixed_point.selected_texts(calibration_spec, limit=1)
    calibration_prompt = fixed_point.tokenize_prompts(
        tokenizer,
        calibration_text,
        calibration_spec["token_limit"],
    )[0][:, :128]
    policies = [
        {
            "name": "standard_static_absmax",
            "percentile": None,
            "scale_cap": None,
            "rope_mechanism": None,
        },
        {
            "name": "standard_static_percentile_0_9999",
            "percentile": 0.9999,
            "scale_cap": None,
            "rope_mechanism": None,
        },
        {
            "name": "standard_static_percentile_0_999",
            "percentile": 0.999,
            "scale_cap": None,
            "rope_mechanism": None,
        },
        {
            "name": "dynamic_scale_absmax",
            "percentile": None,
            "scale_cap": None,
            "rope_mechanism": "dynamic_rope_head_scale_v1",
        },
        {
            "name": "dynamic_scale_percentile_0_9999",
            "percentile": 0.9999,
            "scale_cap": None,
            "rope_mechanism": "dynamic_rope_head_scale_v1",
        },
        {
            "name": "dynamic_scale_percentile_0_999",
            "percentile": 0.999,
            "scale_cap": None,
            "rope_mechanism": "dynamic_rope_head_scale_v1",
        },
    ]

    started = time.perf_counter()
    baseline_model = load_model()
    baseline = generate(
        baseline_model,
        tokenizer,
        package["prompt_tokens"],
        package["termination_token_ids"],
        steps=args.steps,
        top_k_count=args.top_k,
    )
    del baseline_model
    gc.collect()

    records = []
    for policy in policies:
        policy_started = time.perf_counter()
        model = load_model()
        try:
            ranges, operator_ranges = fixed_point.calibrate(
                model,
                [calibration_prompt],
                activation_scale_percentile=policy["percentile"],
            )
            fixed_point.replace_linears(
                model,
                ranges,
                scale_cap=policy["scale_cap"],
                use_percentile_scale=policy["percentile"] is not None,
                rope_diagnostic_mechanism=policy["rope_mechanism"],
            )
            fixed_point.replace_fixed_operators(
                model,
                operator_ranges,
                scale_cap=policy["scale_cap"],
                use_percentile_scale=policy["percentile"] is not None,
                rope_diagnostic_mechanism=policy["rope_mechanism"],
            )
            scale_table = fixed_point.derived_scale_table(
                ranges, operator_ranges, model
            )
            generated = generate(
                model,
                tokenizer,
                package["prompt_tokens"],
                package["termination_token_ids"],
                steps=args.steps,
                top_k_count=args.top_k,
            )
            record = {
                **policy,
                "status": "EXECUTED",
                "generated": generated,
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
                "scale_table_sha256": sha256_bytes(canonical_bytes(scale_table)),
            }
        except Exception as error:
            record = {
                **policy,
                "status": "FAILED",
                "error": str(error),
            }
        record["wall_seconds"] = time.perf_counter() - policy_started
        records.append(record)
        print(
            "ACE2_CHAT_CALIBRATION_POLICY "
            f"name={policy['name']} status={record['status']} "
            f"tokens={record.get('generated', {}).get('generated_token_ids')} "
            f"text={record.get('generated', {}).get('decoded_text')!r}",
            flush=True,
        )
        del model
        gc.collect()

    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_bounded_whole_model_calibration_policy_sweep",
        "package": package,
        "calibration": {
            "dataset": "allenai/c4 validation calibration slice",
            "token_count": int(calibration_prompt.numel()),
            "mechanisms": [None, "dynamic_rope_head_scale_v1"],
        },
        "normal_bf16": baseline,
        "policies": records,
        "wall_seconds": time.perf_counter() - started,
        "status": "PASS_SWEEP_COMPLETE",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, canonical_bytes(result))
    print(
        "ACE2_CHAT_CALIBRATION_SWEEP_RESULT "
        f"status={result['status']} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_CHAT_CALIBRATION_SWEEP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
