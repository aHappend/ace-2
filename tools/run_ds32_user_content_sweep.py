#!/usr/bin/env python3
"""Regenerate the frozen DS32 RMSNorm/Q package across real user-token activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ace2_chat_demo import (
    DYNAMIC_SCALE32_RMS_OUTPUT_DELTAS,
    PINNED_EMBEDDING_OFFSET,
    build_prompt_package,
    canonical_bytes,
    projection_reference,
    quantized_embedding,
    read_ace2rt2_package_metadata,
    read_image,
    sha256_file,
    utc_now,
    write_atomic,
)
from ace2_dynamic_scale32_reference import (
    dynamic_group,
    quantize_for_delta,
    round_shift_even_signed,
)
from ace2_quality_contracts import pack_scale32
from ace2_rmsnorm_reference import reference_rmsnorm


USER_PREFIX = [151644, 872, 198]
IM_END_TOKEN = 151645
RMSNORM_GAIN_BYTES = 1792
RMSNORM_SCALE_OFFSET = 1800
GROUP_LANES = 128
GROUP_COUNT = 7


def s8_bytes(values: list[int] | tuple[int, ...]) -> bytes:
    return bytes(value & 0xFF for value in values)


def user_content_slice(prompt_tokens: list[int], user_token_count: int) -> tuple[int, list[int]]:
    starts = [
        index + len(USER_PREFIX)
        for index in range(len(prompt_tokens) - len(USER_PREFIX) + 1)
        if prompt_tokens[index : index + len(USER_PREFIX)] == USER_PREFIX
    ]
    if len(starts) != 1:
        raise RuntimeError("authenticated package lacks one canonical user-content prefix")
    start = starts[0]
    stop = start + user_token_count
    if stop >= len(prompt_tokens) or prompt_tokens[stop] != IM_END_TOKEN:
        raise RuntimeError("authenticated package user-content boundary differs")
    return start, prompt_tokens[start:stop]


def restore_group_mantissa(value: int, delta: int) -> int:
    if delta >= 0:
        return value << delta
    return round_shift_even_signed(value, -delta)


def evaluate_token(
    *,
    output: Path,
    original_chat_position: int,
    token_id: int,
    embedding_scale: np.float32,
    gains: list[int],
) -> dict[str, Any]:
    case_dir = output / f"position-{original_chat_position:04d}-token-{token_id}"
    package_path = case_dir / "runtime_package.bin"
    package = build_prompt_package(package_path, [token_id], 1)
    commands = package.pop("_first_reference_commands")
    package.pop("_position0_reference_commands")
    package.pop("_position1_reference_commands")
    package.pop("_position2_reference_commands")
    package.pop("_position3_reference_commands")
    package.pop("_all_reference_commands")
    rms_command = commands[0]
    q_command = commands[1]

    embedding = quantized_embedding(
        token_id,
        PINNED_EMBEDDING_OFFSET,
        embedding_scale,
    )
    rms_result = reference_rmsnorm(embedding, gains)
    base_scale = pack_scale32(0x8000, 0)
    selected_deltas: list[int] = []
    selected_mantissas: list[int] = []
    frozen_mantissas: list[int] = []
    restored_rms: list[int] = []
    frozen_representable = True

    for group in range(GROUP_COUNT):
        values = rms_result.outputs[group * GROUP_LANES : (group + 1) * GROUP_LANES]
        selected = dynamic_group(values, base_scale)
        selected_deltas.append(selected.delta)
        selected_mantissas.extend(selected.mantissas)
        frozen_delta = DYNAMIC_SCALE32_RMS_OUTPUT_DELTAS[group]
        for value in values:
            mantissa = quantize_for_delta(value, frozen_delta)
            frozen_representable &= -127 <= mantissa <= 127
            frozen_mantissas.append(mantissa)
            restored_rms.append(restore_group_mantissa(mantissa, frozen_delta))

    rms_restored_exact = restored_rms == rms_result.outputs
    expected_q = projection_reference(q_command, rms_result.outputs)
    frozen_q = projection_reference(q_command, restored_rms)
    q_semantics_match = expected_q == frozen_q
    frozen_deltas = list(DYNAMIC_SCALE32_RMS_OUTPUT_DELTAS)

    mismatch_category = None
    if selected_deltas != frozen_deltas:
        mismatch_category = "exponent_delta_vector"
    elif not frozen_representable:
        mismatch_category = "frozen_mantissa_unrepresentable"
    elif not rms_restored_exact or not q_semantics_match:
        mismatch_category = "q_projection_semantics"

    result = {
        "schema_version": 1,
        "original_chat_position": original_chat_position,
        "token_id": token_id,
        "activation_mapping": (
            "actual user token regenerated as the sole prompt token so the frozen "
            "ordinal-0/1 hardware tuple remains legal"
        ),
        "absolute_position_semantics_claimed": False,
        "package": {
            "path": str(package_path.resolve()),
            "sha256": package["sha256"],
            "schedule_sha256": package["schedule_sha256"],
            "prompt_tokens_sha256": package["prompt_tokens_sha256"],
            "dynamic_scale32_sidecar_records_sha256": package[
                "dynamic_scale32_sidecar_records_sha256"
            ],
            "model_sha256": package["model_sha256"],
            "image_sha256": package["image_sha256"],
        },
        "reference": {
            "selection_rule": "smallest legal delta with all RNE mantissas in [-127,127]",
            "selected_deltas": selected_deltas,
            "frozen_deltas": frozen_deltas,
            "delta_vector_matches": selected_deltas == frozen_deltas,
            "frozen_representable": bool(frozen_representable),
            "rms_restored_exact": rms_restored_exact,
            "q_semantics_match": q_semantics_match,
            "embedding_s8_sha256": hashlib.sha256(s8_bytes(embedding)).hexdigest(),
            "rms_s8_sha256": hashlib.sha256(s8_bytes(rms_result.outputs)).hexdigest(),
            "selected_mantissa_sha256": hashlib.sha256(
                s8_bytes(selected_mantissas)
            ).hexdigest(),
            "frozen_mantissa_sha256": hashlib.sha256(
                s8_bytes(frozen_mantissas)
            ).hexdigest(),
            "q_expected_sha256": hashlib.sha256(s8_bytes(expected_q)).hexdigest(),
            "q_frozen_sha256": hashlib.sha256(s8_bytes(frozen_q)).hexdigest(),
        },
        "mismatch_category": mismatch_category,
        "status": "MISMATCH" if mismatch_category else "PASS",
    }
    write_atomic(case_dir / "reference.json", canonical_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 2:
        raise RuntimeError("--limit must cover at least two user-content positions")

    source_package = args.source_package.resolve()
    source_provenance = args.source_provenance.resolve()
    output = args.output.resolve()
    provenance = json.loads(source_provenance.read_text(encoding="utf-8"))
    metadata = read_ace2rt2_package_metadata(source_package)
    if metadata["sha256"] != provenance["runtime_package"]["sha256"]:
        raise RuntimeError("source package and provenance hashes differ")
    prompt_tokens = [int(token) for token in metadata["prompt_tokens"]]
    user_token_count = int(provenance["prompt"]["user_token_count"])
    user_start, user_tokens = user_content_slice(prompt_tokens, user_token_count)
    selected_tokens = user_tokens[: args.limit]
    if len(selected_tokens) < 2:
        raise RuntimeError("source package has fewer than two user-content tokens")

    scale_address = 0x0000000300000000
    embedding_scale = np.float32(
        struct.unpack("<d", read_image(scale_address + RMSNORM_SCALE_OFFSET, 8))[0]
    )
    gains = np.frombuffer(
        read_image(scale_address, RMSNORM_GAIN_BYTES), dtype="<i2"
    ).astype(int).tolist()

    cases = [
        evaluate_token(
            output=output,
            original_chat_position=user_start + index,
            token_id=token_id,
            embedding_scale=embedding_scale,
            gains=gains,
        )
        for index, token_id in enumerate(selected_tokens)
    ]
    earliest = next((case for case in cases if case["status"] == "MISMATCH"), None)
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "argv": sys.argv,
        "classification": "bounded_actual_user_content_ds32_reference_package_sweep",
        "source": {
            "package_path": str(source_package),
            "package_sha256": sha256_file(source_package),
            "provenance_path": str(source_provenance),
            "provenance_sha256": sha256_file(source_provenance),
            "prompt_sha256": provenance["prompt"]["prompt_sha256"],
            "prompt_text_persisted": False,
        },
        "scope": {
            "positions_checked": len(cases),
            "first_original_chat_position": cases[0]["original_chat_position"],
            "last_original_chat_position": cases[-1]["original_chat_position"],
            "absolute_position_semantics_claimed": False,
            "rtl_extended": False,
            "eda_or_ppa_run": False,
        },
        "frozen_deltas": list(DYNAMIC_SCALE32_RMS_OUTPUT_DELTAS),
        "cases": cases,
        "earliest_mismatch": (
            None
            if earliest is None
            else {
                "original_chat_position": earliest["original_chat_position"],
                "token_id": earliest["token_id"],
                "category": earliest["mismatch_category"],
                "selected_deltas": earliest["reference"]["selected_deltas"],
                "frozen_deltas": earliest["reference"]["frozen_deltas"],
                "frozen_representable": earliest["reference"]["frozen_representable"],
                "q_semantics_match": earliest["reference"]["q_semantics_match"],
            }
        ),
        "status": "MISMATCH_FOUND" if earliest is not None else "PASS",
    }
    write_atomic(output / "summary.json", canonical_bytes(summary))
    print(
        "ACE2_DS32_USER_SWEEP "
        f"status={summary['status']} positions={len(cases)} "
        f"earliest_position={None if earliest is None else earliest['original_chat_position']} "
        f"category={None if earliest is None else earliest['mismatch_category']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
