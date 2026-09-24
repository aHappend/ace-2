#!/usr/bin/env python3
"""Run two-prompt all-layer W4A8 generation with one-round residual fusion."""

from __future__ import annotations

import argparse
import hashlib
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

from tools import ace2_full_model_fixed_point as fixed_point
from tools.ace2_chat_demo import (
    REVISION as TOKENIZER_REVISION,
    TOKENIZER_SNAPSHOT,
    canonical_bytes,
    decode_generated,
    load_tokenizer,
    sha256_file,
    tokenize_prompt,
    write_atomic,
)
from tools.ace2_chat_reference_repairs import readability_gate


FROZEN_SCALES = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1"
    / "paired-smoke-20260801-v1/derived_scales.json"
)
ROPE_MECHANISM = "layer0_tile_bfp_score_attention_v1"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_process_argv() -> list[str]:
    proc_cmdline = Path("/proc/self/cmdline")
    if proc_cmdline.is_file():
        fields = proc_cmdline.read_bytes().split(b"\0")
        return [
            field.decode("utf-8", errors="surrogateescape")
            for field in fields
            if field
        ]
    return [sys.executable, *sys.argv]


def frozen_ranges(
    scales: dict[str, Any],
) -> tuple[
    dict[str, fixed_point.CalibrationRange],
    dict[str, fixed_point.ObservedRange],
]:
    ranges: dict[str, fixed_point.CalibrationRange] = {}
    for name, record in scales["linears"].items():
        output_head_scales = [
            float(value) for value in record.get("output_head_scales", [])
        ]
        ranges[name] = fixed_point.CalibrationRange(
            input_absmax=float(record["input_absmax"]),
            output_absmax=float(record["output_absmax"]),
            output_head_absmax=(
                [value * 127.0 for value in output_head_scales]
                if output_head_scales
                else None
            ),
        )
    operator_ranges = {
        name: fixed_point.ObservedRange(absmax=float(record["absmax"]))
        for name, record in scales["operators"].items()
    }
    return ranges, operator_ranges


def validate_frozen_scale_binding(
    model: Any,
    scales: dict[str, Any],
) -> dict[str, Any]:
    linear_records = []
    for name, module in model.named_modules():
        if not isinstance(module, fixed_point.W4A8Linear):
            continue
        expected = scales["linears"][name]
        checks = {
            "hardware_input_scale": module.hardware_input_scale
            == float(expected["hardware_input_scale"]),
            "multiplier": module.multiplier.cpu().tolist()
            == expected["multiplier"],
            "right_shift": module.right_shift.cpu().tolist()
            == expected["right_shift"],
            "qweight_sha256": fixed_point.sha256_tensor(module.qweight)
            == expected["qweight_sha256"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"frozen scale binding differs for {name}: {checks}")
        linear_records.append({"name": name, "checks": checks})

    joins = []
    for layer_index, layer in enumerate(model.model.layers):
        families = (
            (
                "attention",
                layer.self_attn.o_proj,
                layer.input_scale,
                layer.post_attention_scale,
            ),
            (
                "mlp",
                layer.mlp.down_proj,
                layer.post_attention_scale,
                layer.post_mlp_scale,
            ),
        )
        for family, projection, residual_scale, destination_scale in families:
            accumulator_records = projection.native_scale32_records.to(torch.int64)
            if accumulator_records.shape != (fixed_point.RMS_HIDDEN_SIZE,):
                raise RuntimeError(
                    f"layer {layer_index} {family} accumulator scales do not cover 896 lanes"
                )
            for value in accumulator_records.cpu().tolist():
                fixed_point.unpack_scale32(int(value))
            residual_scale32 = fixed_point.ceil_scale32_from_float(residual_scale)
            destination_scale32 = fixed_point.ceil_scale32_from_float(
                destination_scale
            )
            fixed_point.unpack_scale32(residual_scale32)
            fixed_point.unpack_scale32(destination_scale32)
            joins.append(
                {
                    "family": family,
                    "layer_index": layer_index,
                    "projection": (
                        f"model.layers.{layer_index}.self_attn.o_proj"
                        if family == "attention"
                        else f"model.layers.{layer_index}.mlp.down_proj"
                    ),
                    "accumulator_scale32_records": fixed_point.RMS_HIDDEN_SIZE,
                    "accumulator_scale32_sha256": fixed_point.sha256_tensor(
                        accumulator_records
                    ),
                    "residual_scale_source": residual_scale,
                    "residual_scale32": residual_scale32,
                    "destination_scale_source": destination_scale,
                    "destination_scale32": destination_scale32,
                }
            )
    if len(linear_records) != 169:
        raise RuntimeError(
            f"expected 169 W4A8 linears including LM head, got {len(linear_records)}"
        )
    if len(joins) != 48:
        raise RuntimeError(f"expected 48 residual joins, got {len(joins)}")
    return {
        "all_linear_bindings_match": True,
        "linears_checked": len(linear_records),
        "joins_checked": len(joins),
        "attention_joins": sum(item["family"] == "attention" for item in joins),
        "mlp_joins": sum(item["family"] == "mlp" for item in joins),
        "joins": joins,
    }


def token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def top_k(
    tokenizer: Any,
    logits: torch.Tensor,
    count: int,
) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.detach().to(torch.float64), count)
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


def prompt_coherence(
    prompt_name: str,
    decoded: str,
    readability: dict[str, Any],
) -> dict[str, Any]:
    lowered = decoded.casefold()
    if prompt_name == "hello_world":
        semantic_match = any(
            word in lowered
            for word in ("hello", "hi", "welcome", "greeting")
        )
        criterion = "contains a visible greeting"
    elif prompt_name == "second_prompt":
        semantic_match = "blue" in lowered
        criterion = "contains the requested word 'blue'"
    else:
        raise ValueError(f"unknown prompt: {prompt_name}")
    return {
        "criterion": criterion,
        "semantic_match": semantic_match,
        "passed": bool(readability["passed"] and semantic_match),
    }


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt_name: str,
    prompt_tokens: list[int],
    termination_token_ids: set[int],
    *,
    max_new_tokens: int,
    top_k_count: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    generated: list[int] = []
    steps = []
    with torch.inference_mode():
        for generation_index in range(max_new_tokens):
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
            candidates = top_k(tokenizer, logits, top_k_count)
            selected = int(candidates[0]["token_id"])
            generated.append(selected)
            terminated = selected in termination_token_ids
            steps.append(
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
    decoded = decode_generated(tokenizer, generated)
    readability = readability_gate(decoded, generated)
    return {
        "prompt_token_count": len(prompt_tokens),
        "prompt_tokens_sha256": sha256_bytes(
            torch.tensor(prompt_tokens, dtype=torch.int32).numpy().tobytes()
        ),
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "readability_gate": readability,
        "coherence_gate": prompt_coherence(prompt_name, decoded, readability),
        "terminated": bool(steps and steps[-1]["terminated"]),
        "steps": steps,
        "wall_seconds": time.perf_counter() - started,
    }


def summarize_execution(
    runtime: fixed_point.AllProjectionResidualFusionRuntime,
) -> dict[str, Any]:
    summary = runtime.summary()
    passes = summary["passes"]
    coverage = {
        (int(join["layer_index"]), str(join["family"]))
        for item in passes
        for join in item["joins"]
    }
    expected = {
        (layer, family)
        for layer in range(24)
        for family in ("attention", "mlp")
    }
    if coverage != expected:
        raise RuntimeError("executed residual coverage differs from all 48 joins")
    if any(
        item["attention_joins"] != 24 or item["mlp_joins"] != 24
        for item in passes
    ):
        raise RuntimeError("a generation pass did not execute 24+24 residual joins")
    aggregate = {
        family: {
            "calls": 0,
            "lanes": 0,
            "positive_saturations": 0,
            "negative_saturations": 0,
        }
        for family in ("attention", "mlp")
    }
    for item in passes:
        for join in item["joins"]:
            target = aggregate[str(join["family"])]
            trace = join["independent_trace"]
            target["calls"] += 1
            target["lanes"] += int(trace["lanes"])
            target["positive_saturations"] += int(trace["positive_saturations"])
            target["negative_saturations"] += int(trace["negative_saturations"])
    return {
        "semantics": summary["semantics"],
        "completed_passes": summary["completed_passes"],
        "all_24_layers_both_families_executed": True,
        "aggregate": aggregate,
        "passes": passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--second-prompt", default="Write one word: blue")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 32:
        raise RuntimeError("--max-new-tokens must be in 1..32")
    if not 1 <= args.top_k <= 64:
        raise RuntimeError("--top-k must be in 1..64")

    started = time.perf_counter()
    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))
    ranges, operator_ranges = frozen_ranges(scales)
    tokenizer = load_tokenizer()
    tokenizations = {
        "hello_world": tokenize_prompt(tokenizer, "Hello world", ""),
        "second_prompt": tokenize_prompt(tokenizer, args.second_prompt, ""),
    }
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_point.replace_linears(
        model,
        ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
    )
    fixed_point.replace_fixed_operators(
        model,
        operator_ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
        all_projection_residual_fusion=True,
    )
    scale_binding = validate_frozen_scale_binding(model, scales)
    runtime = model.ace2_all_projection_residual_fusion_runtime
    if not isinstance(runtime, fixed_point.AllProjectionResidualFusionRuntime):
        raise RuntimeError("all-projection residual fusion runtime was not installed")

    termination_token_ids = {
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    generations = {}
    for prompt_name, tokenization in tokenizations.items():
        generations[prompt_name] = greedy_generate(
            model,
            tokenizer,
            prompt_name,
            tokenization["chat_template_token_ids"],
            termination_token_ids,
            max_new_tokens=args.max_new_tokens,
            top_k_count=args.top_k,
        )
        print(
            "ACE2_ALL_RESIDUAL_PROMPT "
            f"prompt={prompt_name} "
            f"decoded={generations[prompt_name]['decoded_text']!r} "
            f"coherent={generations[prompt_name]['coherence_gate']['passed']}",
            flush=True,
        )

    execution = summarize_execution(runtime)
    coherent = all(
        generation["coherence_gate"]["passed"]
        for generation in generations.values()
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_reference_only_all_projection_residual_fusion",
        "invocation": {
            "argv": exact_process_argv(),
            "cwd": str(Path.cwd().resolve()),
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": TOKENIZER_REVISION,
            "snapshot": str(TOKENIZER_SNAPSHOT.resolve()),
            "dtype_at_load": "bfloat16",
            "weight_quantization": "W4_symmetric_per_output_channel",
            "activation_arithmetic": "fixed_W4A8_all_24_layers",
        },
        "prompts": {
            "hello_world": "Hello world",
            "second_prompt": args.second_prompt,
        },
        "generations": generations,
        "scale_authentication": scale_binding,
        "residual_execution": execution,
        "artifacts": {
            "frozen_scales": {
                "path": str(FROZEN_SCALES.resolve()),
                "sha256": sha256_file(FROZEN_SCALES),
            },
            "fixed_point_source": {
                "path": str((ROOT / "tools/ace2_full_model_fixed_point.py").resolve()),
                "sha256": sha256_file(
                    ROOT / "tools/ace2_full_model_fixed_point.py"
                ),
            },
            "runner_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "two_prompt_coherence_passed": coherent,
        "status": (
            "PASS_COHERENT_TWO_PROMPT_W4A8"
            if coherent
            else "FAIL_W4A8_TWO_PROMPT_COHERENCE"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, canonical_bytes(result))
    print(
        "ACE2_ALL_RESIDUAL_REFERENCE_RESULT "
        f"status={result['status']} "
        f"passes={execution['completed_passes']} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_ALL_RESIDUAL_REFERENCE_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
