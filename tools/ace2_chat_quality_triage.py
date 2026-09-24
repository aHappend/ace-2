#!/usr/bin/env python3
"""Reference-only output-quality triage for an ACE2RT2 chat package."""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_attention_compose_reference import AttentionComposePhaseReplay
from tools.ace2_chat_demo import (
    DYNAMIC_SCALE32_FLAG,
    LM_HEAD_TILE_COUNT,
    PACKAGE_V2_DYNAMIC_RECORD,
    PINNED_EMBEDDING_OFFSET,
    TOKENIZER_SNAPSHOT,
    _PositionReferenceMemory,
    _dynamic_rms_output_sidecar,
    _pack_int8_bytes,
    _position_expected_payload,
    canonical_bytes,
    decode_generated,
    load_tokenizer,
    quantized_embedding,
    read_ace2rt2_package_metadata,
    reconstruct_ace2rt2_schedule,
    resolve_position_token,
    sha256_bytes,
    sha256_file,
    write_atomic,
)

FROZEN_SCALES = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1"
    / "paired-smoke-20260801-v1/derived_scales.json"
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def exact_process_argv() -> list[str]:
    proc_cmdline = Path("/proc/self/cmdline")
    if proc_cmdline.is_file():
        fields = proc_cmdline.read_bytes().split(b"\0")
        return [field.decode("utf-8", errors="surrogateescape") for field in fields if field]
    return [sys.executable, *sys.argv]


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def top_k_records(tokenizer: Any, logits: np.ndarray, count: int) -> list[dict[str, Any]]:
    token_ids = np.arange(logits.size, dtype=np.int64)
    order = np.lexsort((token_ids, -logits.astype(np.int16)))[:count]
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "logit_s8": int(logits[token_id]),
            "decoded_piece": token_text(tokenizer, int(token_id)),
        }
        for rank, token_id in enumerate(order.tolist(), start=1)
    ]


def top_k_float_records(
    tokenizer: Any,
    logits: np.ndarray,
    count: int,
) -> list[dict[str, Any]]:
    token_ids = np.arange(logits.size, dtype=np.int64)
    order = np.lexsort((token_ids, -logits.astype(np.float64)))[:count]
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "logit": float(logits[token_id]),
            "decoded_piece": token_text(tokenizer, int(token_id)),
        }
        for rank, token_id in enumerate(order.tolist(), start=1)
    ]


def int8_stats(values: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.int8)
    return {
        "count": int(raw.size),
        "min": int(raw.min()),
        "max": int(raw.max()),
        "zero_fraction": float(np.count_nonzero(raw == 0) / raw.size),
        "saturated_fraction": float(
            np.count_nonzero((raw == -128) | (raw == 127)) / raw.size
        ),
        "sha256": sha256_bytes(raw.tobytes()),
    }


def residual_domain_scales(
    scales: dict[str, Any],
    command: dict[str, Any],
) -> tuple[float, float, float]:
    layer = int(command["layer_id"])
    if command["operator"] == "attention_residual_add":
        lhs_name = (
            "model.layers.0.input_layernorm.input"
            if layer == 0
            else f"model.layers.{layer - 1}.post_mlp_residual"
        )
        lhs_scale = float(scales["operators"][lhs_name]["scale"])
        rhs_scale = float(
            scales["linears"][f"model.layers.{layer}.self_attn.o_proj"][
                "output_scale"
            ]
        )
        destination_scale = float(
            scales["operators"][f"model.layers.{layer}.post_attention_residual"][
                "scale"
            ]
        )
        return lhs_scale, rhs_scale, destination_scale
    if command["operator"] == "mlp_residual_add":
        lhs_scale = float(
            scales["linears"][f"model.layers.{layer}.mlp.down_proj"][
                "output_scale"
            ]
        )
        rhs_scale = float(
            scales["operators"][f"model.layers.{layer}.post_attention_residual"][
                "scale"
            ]
        )
        destination_scale = float(
            scales["operators"][f"model.layers.{layer}.post_mlp_residual"]["scale"]
        )
        return lhs_scale, rhs_scale, destination_scale
    raise ValueError(f"not a residual operator: {command['operator']}")


def residual_domain_diagnostic(
    scales: dict[str, Any],
    command: dict[str, Any],
    memory: _PositionReferenceMemory,
) -> tuple[dict[str, Any], bytes]:
    count = int(command["n"])
    lhs = np.frombuffer(
        memory.read(int(command["src0_addr"]), count), dtype=np.int8
    ).astype(np.int16)
    rhs = np.frombuffer(
        memory.read(int(command["src1_addr"]), count), dtype=np.int8
    ).astype(np.int16)
    lhs_scale, rhs_scale, destination_scale = residual_domain_scales(scales, command)
    direct = np.clip(lhs + rhs, -128, 127).astype(np.int8)
    lhs_destination = np.rint(
        lhs.astype(np.float64) * lhs_scale / destination_scale
    ).astype(np.int16)
    rhs_destination = np.rint(
        rhs.astype(np.float64) * rhs_scale / destination_scale
    ).astype(np.int16)
    scale_aware = np.clip(lhs_destination + rhs_destination, -128, 127).astype(
        np.int8
    )
    differing = direct != scale_aware
    return {
        "command_ordinal": int(command["ordinal"]),
        "position": int(command["token_step"]),
        "layer_id": int(command["layer_id"]),
        "operator": command["operator"],
        "lhs_scale": lhs_scale,
        "rhs_scale": rhs_scale,
        "destination_scale": destination_scale,
        "direct_add_lhs_overweight_factor": destination_scale / lhs_scale,
        "direct_add_rhs_overweight_factor": destination_scale / rhs_scale,
        "lanes_checked": count,
        "differing_lanes": int(np.count_nonzero(differing)),
        "maximum_absolute_s8_difference": int(
            np.abs(direct.astype(np.int16) - scale_aware.astype(np.int16)).max()
        ),
        "direct_add": int8_stats(direct),
        "scale_aware_add": int8_stats(scale_aware),
    }, scale_aware.tobytes()


def lm_head_global_domain(
    tokenizer: Any,
    logits: np.ndarray,
    *,
    output_scale: float,
    top_k: int,
) -> dict[str, Any]:
    raw = np.asarray(logits, dtype=np.int8)
    dequantized = raw.astype(np.float64) * output_scale
    raw_top_k = top_k_records(tokenizer, raw, top_k)
    dequantized_top_k = top_k_float_records(tokenizer, dequantized, top_k)
    raw_ids = [item["token_id"] for item in raw_top_k]
    dequantized_ids = [item["token_id"] for item in dequantized_top_k]
    if raw_ids != dequantized_ids:
        raise RuntimeError("global LM-head dequantization changed top-k ordering")
    per_tile = []
    for tile in range(LM_HEAD_TILE_COUNT):
        start = tile * 32
        stop = start + 32
        tile_raw = raw[start:stop]
        tile_dequantized = dequantized[start:stop]
        per_tile.append(
            {
                "vocab_tile": tile,
                "token_id_start": start,
                "token_id_stop_exclusive": stop,
                "raw": int8_stats(tile_raw),
                "dequantized_min": float(tile_dequantized.min()),
                "dequantized_max": float(tile_dequantized.max()),
                "dequantized_sha256": sha256_bytes(
                    tile_dequantized.astype("<f8", copy=False).tobytes()
                ),
            }
        )
    return {
        "classification": "all_tokens_and_tiles_dequantized_in_one_global_domain",
        "tokens_checked": int(raw.size),
        "tiles_checked": LM_HEAD_TILE_COUNT,
        "global_output_scale": output_scale,
        "unique_output_scale_count": 1,
        "raw": int8_stats(raw),
        "dequantized_min": float(dequantized.min()),
        "dequantized_max": float(dequantized.max()),
        "dequantized_zero_fraction": float(
            np.count_nonzero(dequantized == 0.0) / dequantized.size
        ),
        "dequantized_saturated_fraction": float(
            np.count_nonzero((raw == -128) | (raw == 127)) / raw.size
        ),
        "dequantized_sha256": sha256_bytes(
            dequantized.astype("<f8", copy=False).tobytes()
        ),
        "raw_and_dequantized_top_k_identical": True,
        "top_k": dequantized_top_k,
        "per_token_dequantized_logits": dequantized.tolist(),
        "per_tile": per_tile,
    }


def lm_head_candidate_metadata(
    memory: _PositionReferenceMemory,
    command: dict[str, Any],
    token_id: int,
) -> dict[str, Any]:
    lane = token_id % 32
    record_addr = int(command["scale_addr"]) + lane * 16
    record = memory.read(record_addr, 16)
    return {
        "token_id": token_id,
        "vocab_tile": token_id // 32,
        "lane": lane,
        "activation_addr": int(command["src0_addr"]),
        "packed_weight_tile_addr": int(command["src1_addr"]),
        "destination_addr": int(command["dst_addr"]),
        "requant_record_addr": record_addr,
        "requant_multiplier": struct.unpack_from("<i", record, 0)[0],
        "requant_right_shift": record[4] & 0x3F,
        "requant_zero_point": struct.unpack_from("<b", record, 5)[0],
        "bias_accumulator": struct.unpack_from("<i", record, 6)[0],
        "requant_record_sha256": sha256_bytes(record),
    }


def run_normal_baseline(
    package: dict[str, Any],
    tokenizer: Any,
    *,
    steps: int,
    top_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
    )
    model.eval()
    input_ids = torch.tensor([package["prompt_tokens"]], dtype=torch.long)
    past_key_values = None
    generated: list[int] = []
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for generation_index in range(steps):
            model_input = input_ids if past_key_values is None else input_ids[:, -1:]
            result = model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = result.logits[0, -1].float()
            values, indices = torch.topk(logits, top_k)
            candidates = [
                {
                    "rank": rank,
                    "token_id": int(token_id),
                    "logit_fp32": float(value),
                    "decoded_piece": token_text(tokenizer, int(token_id)),
                }
                for rank, (token_id, value) in enumerate(
                    zip(indices.tolist(), values.tolist(), strict=True), start=1
                )
            ]
            selected = int(indices[0])
            generated.append(selected)
            terminated = selected in package["termination_token_ids"]
            records.append(
                {
                    "generation_index": generation_index,
                    "model_position": len(package["prompt_tokens"]) - 1 + generation_index,
                    "selected_token_id": selected,
                    "selected_piece": token_text(tokenizer, selected),
                    "terminated": terminated,
                    "top_k": candidates,
                }
            )
            input_ids = torch.cat(
                [input_ids, torch.tensor([[selected]], dtype=torch.long)], dim=1
            )
            past_key_values = result.past_key_values
            if terminated:
                break
    del model
    return {
        "classification": "diagnostic_normal_qwen_fp32_baseline_not_accelerator_evidence",
        "generated_token_ids": generated,
        "decoded_text": decode_generated(tokenizer, generated),
        "steps": records,
        "wall_seconds": time.perf_counter() - started,
    }


def run_w4a8_reference(
    package: dict[str, Any],
    tokenizer: Any,
    *,
    top_k: int,
    max_new_tokens: int,
    scale_aware_residuals: bool,
) -> dict[str, Any]:
    reconstruction = reconstruct_ace2rt2_schedule(
        Path(package["path"]),
        package,
    )
    commands = reconstruction["commands"]
    reconstructed_schedule_sha256 = reconstruction["schedule_sha256"]

    started = time.perf_counter()
    generated: list[int] = []
    feedback: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    compose_states: dict[tuple[int, int, int], AttentionComposePhaseReplay] = {}
    loaded_position = -1
    completed_commands = 0
    terminated = False
    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))
    lm_head_scale_metadata = scales["linears"]["lm_head"]
    lm_head_output_scale = float(lm_head_scale_metadata["output_scale"])
    if lm_head_scale_metadata.get("output_head_scales"):
        raise RuntimeError("LM head unexpectedly has multiple output-scale domains")
    residual_domain_checks: list[dict[str, Any]] = []
    first_residual_domain_divergence: dict[str, Any] | None = None
    saturation_counts: dict[str, int] = {}
    package_raw = Path(package["path"]).read_bytes()
    sidecar_stop = int(package["command_records_offset"])
    sidecar_start = sidecar_stop - (
        int(package["dynamic_scale32_sidecar_count"])
        * PACKAGE_V2_DYNAMIC_RECORD.size
    )
    prompt_sidecars = [
        PACKAGE_V2_DYNAMIC_RECORD.unpack_from(package_raw, offset)
        for offset in range(
            sidecar_start,
            sidecar_stop,
            PACKAGE_V2_DYNAMIC_RECORD.size,
        )
    ]

    with accepted_runtime.IMAGE.open("rb") as image_handle:
        with mmap.mmap(image_handle.fileno(), 0, access=mmap.ACCESS_READ) as image_map:
            memory = _PositionReferenceMemory(image_map)
            first_command = commands[0]
            scale_record = memory.read(int(first_command["scale_addr"]), 1808)
            embedding_scale = np.float32(struct.unpack_from("<d", scale_record, 1800)[0])
            logits = np.empty(LM_HEAD_TILE_COUNT * 32, dtype=np.int8)
            lm_head_commands: dict[int, dict[str, Any]] = {}
            current_input: dict[str, Any] | None = None
            final_rmsnorm: dict[str, Any] | None = None

            for command in commands:
                position = int(command["token_step"])
                ordinal = int(command["ordinal"])
                if loaded_position != position:
                    token_id, source, generated_index = resolve_position_token(
                        package["prompt_tokens"], generated, position
                    )
                    embedding = quantized_embedding(
                        token_id,
                        PINNED_EMBEDDING_OFFSET,
                        embedding_scale,
                    )
                    embedding_raw = _pack_int8_bytes(embedding)
                    memory.write(int(command["src0_addr"]), embedding_raw)
                    if (
                        position < int(package["prompt_token_count"])
                        and int(command["flags"]) & DYNAMIC_SCALE32_FLAG
                    ):
                        payload_addr, host_plan_sidecar = prompt_sidecars[position]
                        if int(payload_addr) != int(command["src0_addr"]):
                            raise RuntimeError(
                                "package DynamicScale32 sidecar payload address differs"
                            )
                        memory.write(int(payload_addr) - 64, host_plan_sidecar)
                    current_input = {
                        "position": position,
                        "token_id": token_id,
                        "token_source": source,
                        "generated_index": generated_index,
                        "embedding_destination_addr": int(command["src0_addr"]),
                        "embedding_scale_record_addr": int(command["scale_addr"]),
                        "embedding_scale_fp32": float(embedding_scale),
                        "embedding_s8_sha256": sha256_bytes(embedding_raw),
                        "embedding_s8_min": min(embedding),
                        "embedding_s8_max": max(embedding),
                        "first_command_ordinal": ordinal,
                    }
                    if source == "generated_argmax":
                        feedback.append(dict(current_input))
                    loaded_position = position
                    lm_head_commands = {}
                    final_rmsnorm = None
                    print(
                        "ACE2_W4A8_REFERENCE_POSITION_START "
                        f"position={position} token={token_id} source={source}",
                        flush=True,
                    )

                residual_check = None
                scale_aware_residual_payload = None
                if command["operator"] in {
                    "attention_residual_add",
                    "mlp_residual_add",
                }:
                    (
                        residual_check,
                        scale_aware_residual_payload,
                    ) = residual_domain_diagnostic(
                        scales, command, memory
                    )
                    residual_domain_checks.append(residual_check)
                    if (
                        first_residual_domain_divergence is None
                        and residual_check["differing_lanes"] > 0
                    ):
                        first_residual_domain_divergence = dict(residual_check)

                payload, saturation, expected_base = _position_expected_payload(
                    command, memory, compose_states
                )
                if (
                    scale_aware_residuals
                    and scale_aware_residual_payload is not None
                ):
                    payload = scale_aware_residual_payload
                    repaired_raw = np.frombuffer(payload, dtype=np.int8)
                    saturation = bool(
                        np.any((repaired_raw == -128) | (repaired_raw == 127))
                    )
                if saturation:
                    operator = str(command["operator"])
                    saturation_counts[operator] = saturation_counts.get(operator, 0) + 1
                if payload:
                    memory.write(expected_base, payload)
                if (
                    command["operator"] == "input_rmsnorm"
                    and int(command["flags"]) & DYNAMIC_SCALE32_FLAG
                ):
                    dynamic_output_sidecar = _dynamic_rms_output_sidecar(
                        command,
                        memory.read(int(command["src0_addr"]) - 64, 64),
                    )
                    memory.write(expected_base - 64, dynamic_output_sidecar)
                completed_commands = ordinal + 1

                if command["operator"] == "final_rmsnorm":
                    final_rmsnorm = {
                        "command_ordinal": ordinal,
                        "source_addr": int(command["src0_addr"]),
                        "destination_addr": int(command["dst_addr"]),
                        "scale_record_addr": int(command["scale_addr"]),
                        "output_sha256": sha256_bytes(payload),
                        "output_s8_min": min(_signed for _signed in np.frombuffer(payload, dtype=np.int8).astype(int).tolist()),
                        "output_s8_max": max(_signed for _signed in np.frombuffer(payload, dtype=np.int8).astype(int).tolist()),
                    }
                elif command["operator"] == "lm_head_tile":
                    tile = int(command["vocab_tile"])
                    logits[tile * 32 : (tile + 1) * 32] = np.frombuffer(
                        payload, dtype=np.int8
                    )
                    lm_head_commands[tile] = command
                    if (
                        tile == LM_HEAD_TILE_COUNT - 1
                        and position >= int(package["prompt_token_count"]) - 1
                    ):
                        candidates = top_k_records(tokenizer, logits, top_k)
                        global_domain = lm_head_global_domain(
                            tokenizer,
                            logits,
                            output_scale=lm_head_output_scale,
                            top_k=top_k,
                        )
                        selected = int(candidates[0]["token_id"])
                        generated.append(selected)
                        terminated = selected in package["termination_token_ids"]
                        candidate_metadata = [
                            lm_head_candidate_metadata(
                                memory,
                                lm_head_commands[int(candidate["token_id"]) // 32],
                                int(candidate["token_id"]),
                            )
                            for candidate in candidates
                        ]
                        boundaries.append(
                            {
                                "generation_index": len(generated) - 1,
                                "model_position": position,
                                "input": current_input,
                                "selected_token_id": selected,
                                "selected_piece": token_text(tokenizer, selected),
                                "terminated": terminated,
                                "termination_policy": {
                                    "token_ids": list(package["termination_token_ids"]),
                                    "matched": terminated,
                                },
                                "top_k": candidates,
                                "lm_head_global_domain": global_domain,
                                "top_k_lm_head_metadata": candidate_metadata,
                                "final_rmsnorm": final_rmsnorm,
                                "lm_head_first_tile": {
                                    "command_ordinal": int(lm_head_commands[0]["ordinal"]),
                                    "activation_addr": int(lm_head_commands[0]["src0_addr"]),
                                    "packed_weight_addr": int(lm_head_commands[0]["src1_addr"]),
                                    "destination_addr": int(lm_head_commands[0]["dst_addr"]),
                                    "scale_addr": int(lm_head_commands[0]["scale_addr"]),
                                },
                                "lm_head_last_tile": {
                                    "command_ordinal": ordinal,
                                    "activation_addr": int(command["src0_addr"]),
                                    "packed_weight_addr": int(command["src1_addr"]),
                                    "destination_addr": int(command["dst_addr"]),
                                    "scale_addr": int(command["scale_addr"]),
                                },
                            }
                        )
                        print(
                            "ACE2_W4A8_REFERENCE_BOUNDARY "
                            f"position={position} generation_index={len(generated) - 1} "
                            f"token={selected} logit={candidates[0]['logit_s8']} "
                            f"decoded={token_text(tokenizer, selected)!r} terminated={terminated}",
                            flush=True,
                        )
                        if terminated or len(generated) >= max_new_tokens:
                            break

    decoded = decode_generated(tokenizer, generated)
    return {
        "classification": "independent_full_w4a8_fixed_point_model_reference_without_rtl_or_journal",
        "residual_execution_policy": (
            "scale_aware_fixed_semantics"
            if scale_aware_residuals
            else "accepted_direct_signed_int8_add"
        ),
        "status": "PASS_REFERENCE_COMPLETE",
        "reconstructed_schedule_sha256": reconstructed_schedule_sha256,
        "commands_executed": completed_commands,
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "readable_text_observed": bool(decoded.strip()),
        "terminated": terminated,
        "termination_token_ids": list(package["termination_token_ids"]),
        "embedding_feedback": feedback,
        "prompt_end_and_generated_lm_head_boundaries": boundaries,
        "residual_domain_diagnostic": {
            "classification": "accepted_raw_s8_add_versus_scale_aware_fixed_semantics",
            "checks": len(residual_domain_checks),
            "first_divergence": first_residual_domain_divergence,
            "final_prompt_position_checks": [
                item
                for item in residual_domain_checks
                if item["position"] == int(package["prompt_token_count"]) - 1
            ],
        },
        "saturation_counts": saturation_counts,
        "lm_head_scale_domain": {
            "frozen_scales_path": str(FROZEN_SCALES.resolve()),
            "frozen_scales_sha256": sha256_file(FROZEN_SCALES),
            "output_scale": lm_head_output_scale,
            "unique_output_scale_count": 1,
            "tokens_per_step": LM_HEAD_TILE_COUNT * 32,
        },
        "wall_seconds": time.perf_counter() - started,
    }


def run_bf16_on_w4a8_contexts(
    package: dict[str, Any],
    tokenizer: Any,
    w4a8_reference: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    records = []
    generated = list(w4a8_reference["generated_token_ids"])
    boundaries = w4a8_reference["prompt_end_and_generated_lm_head_boundaries"]
    with torch.inference_mode():
        for generation_index, boundary in enumerate(boundaries):
            context = [
                *package["prompt_tokens"],
                *generated[:generation_index],
            ]
            input_ids = torch.tensor([context], dtype=torch.long)
            bf16_logits = (
                model(input_ids=input_ids, use_cache=False).logits[0, -1]
                .to(torch.float64)
                .cpu()
                .numpy()
            )
            w4a8_logits = np.asarray(
                boundary["lm_head_global_domain"]["per_token_dequantized_logits"],
                dtype=np.float64,
            )[: bf16_logits.size]
            dot = float(np.dot(bf16_logits, w4a8_logits))
            bf16_norm = float(np.linalg.norm(bf16_logits))
            w4a8_norm = float(np.linalg.norm(w4a8_logits))
            difference_norm = float(np.linalg.norm(w4a8_logits - bf16_logits))
            bf16_top_k = top_k_float_records(tokenizer, bf16_logits, top_k)
            w4a8_top_k = boundary["lm_head_global_domain"]["top_k"]
            records.append(
                {
                    "generation_index": generation_index,
                    "context_token_count": len(context),
                    "context_sha256": sha256_bytes(
                        np.asarray(context, dtype="<u4").tobytes()
                    ),
                    "bf16_top_k": bf16_top_k,
                    "w4a8_top_k": w4a8_top_k,
                    "top1_match": (
                        bf16_top_k[0]["token_id"] == w4a8_top_k[0]["token_id"]
                    ),
                    "cosine_similarity": (
                        dot / (bf16_norm * w4a8_norm)
                        if bf16_norm and w4a8_norm
                        else 0.0
                    ),
                    "relative_l2_error": (
                        difference_norm / bf16_norm if bf16_norm else float("inf")
                    ),
                    "bf16_zero_fraction": float(
                        np.count_nonzero(bf16_logits == 0.0) / bf16_logits.size
                    ),
                    "w4a8_zero_fraction": boundary["lm_head_global_domain"][
                        "dequantized_zero_fraction"
                    ],
                    "w4a8_saturated_fraction": boundary["lm_head_global_domain"][
                        "dequantized_saturated_fraction"
                    ],
                }
            )
    del model
    return {
        "classification": "bf16_and_packed_w4a8_logits_on_identical_token_contexts",
        "steps_checked": len(records),
        "steps": records,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare normal Qwen output with the independent packed W4A8 reference"
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--normal-baseline-steps", type=int, default=8)
    parser.add_argument("--w4a8-steps", type=int, default=4)
    parser.add_argument("--scale-aware-residuals", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 64:
        raise RuntimeError("--top-k must be between 1 and 64")
    if args.normal_baseline_steps < 1 or args.normal_baseline_steps > 32:
        raise RuntimeError("--normal-baseline-steps must be between 1 and 32")
    if args.w4a8_steps < 1 or args.w4a8_steps > 32:
        raise RuntimeError("--w4a8-steps must be between 1 and 32")

    package_path = args.package.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    package = read_ace2rt2_package_metadata(package_path)
    tokenizer = load_tokenizer()
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_output_quality_root_cause_triage",
        "invocation": {
            "cwd": str(Path.cwd().resolve()),
            "argv": exact_process_argv(),
        },
        "package": package,
        "artifacts": {
            "packed_image": {
                "path": str(accepted_runtime.IMAGE.resolve()),
                "bytes": accepted_runtime.IMAGE.stat().st_size,
                "sha256": sha256_file(accepted_runtime.IMAGE),
            },
            "raw_model": {
                "path": str(accepted_runtime.MODEL.resolve()),
                "bytes": accepted_runtime.MODEL.stat().st_size,
                "sha256": sha256_file(accepted_runtime.MODEL),
            },
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "prompt": {
            "token_count": len(package["prompt_tokens"]),
            "token_ids": package["prompt_tokens"],
            "decoded_chat_template": tokenizer.decode(
                package["prompt_tokens"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        },
        "status": "RUNNING",
    }
    write_atomic(output_path, canonical_bytes(result))
    result["normal_qwen_baseline"] = run_normal_baseline(
        package,
        tokenizer,
        steps=args.normal_baseline_steps,
        top_k=args.top_k,
    )
    write_atomic(output_path, canonical_bytes(result))
    result["w4a8_reference"] = run_w4a8_reference(
        package,
        tokenizer,
        top_k=args.top_k,
        max_new_tokens=min(args.w4a8_steps, int(package["max_new_tokens"])),
        scale_aware_residuals=args.scale_aware_residuals,
    )
    write_atomic(output_path, canonical_bytes(result))
    result["bf16_on_w4a8_contexts"] = run_bf16_on_w4a8_contexts(
        package,
        tokenizer,
        result["w4a8_reference"],
        top_k=args.top_k,
    )
    result["wall_seconds"] = time.perf_counter() - started
    result["status"] = "PASS_TRIAGE_COMPLETE"
    write_atomic(output_path, canonical_bytes(result))
    print(
        "ACE2_CHAT_QUALITY_TRIAGE_RESULT "
        f"status={result['status']} "
        f"w4a8_tokens={result['w4a8_reference']['generated_token_ids']} "
        f"readable={result['w4a8_reference']['readable_text_observed']} "
        f"wall_seconds={result['wall_seconds']:.6f} "
        f"output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_CHAT_QUALITY_TRIAGE_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
