#!/usr/bin/env python3
"""Run a non-consuming BF16 baseline over the frozen V16 quality matrix.

This probe never writes under an official candidate namespace and does not
authorize, replay, or repair V16. It exists only to distinguish a source-model
or generation-contract ceiling from quantization-specific quality loss.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import v13_v10_transformer_v12_lm_head64_backend as quality
from qwen_instruct_option_b import (
    SNAPSHOT,
    file_record,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    ROOT
    / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v16"
    / "quality-campaign-0001/frozen_matrix.json"
)
MATRIX_SHA256 = "829d3fc355ecba7dd01aed6a06c165b6a664cbc144591a0c40b45ae4fa27c6a1"
V16_RESULT_PATH = MATRIX_PATH.parent / "RESULT.json"
V16_RESULT_SHA256 = "f59f6523b61c9d02077db46b13852fab83b9e2d1de9bb5b2da692a5c5a6a2f93"
PROBE_ID = "ace2-v16-matrix-pinned-bf16-baseline-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_probe(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"probe output already exists: {output}")
    require(quality.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "frozen V16 matrix hash differs")
    require(quality.sha256_file(V16_RESULT_PATH) == V16_RESULT_SHA256, "immutable V16 result hash differs")
    matrix = load_json(MATRIX_PATH)
    require(matrix.get("matrix_id") == "ace2-v13-product-quality-v1", "matrix identity differs")
    require(matrix.get("response_count") == 23, "matrix response count differs")

    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    torch.set_num_threads(32)

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for case in matrix["cases"]:
        messages: list[dict[str, str]] = [{"role": "system", "content": matrix["system_message"]}]
        for turn_index, user_text in enumerate(case["user_turns"], start=1):
            messages.append({"role": "user", "content": user_text})
            messages_before = [dict(item) for item in messages]
            generated = quality._generate_response(model, tokenizer, messages_before)
            record = {
                "schema_version": 1,
                "probe_id": PROBE_ID,
                "run_kind": "non_consuming_bf16_baseline",
                "case_id": case["id"],
                "category": case["category"],
                "turn_index": turn_index,
                "is_final_turn": turn_index == len(case["user_turns"]),
                "raw_user_prompt": user_text,
                "messages_before_generation": messages_before,
                "required_observable": case["required_observable"] if turn_index == len(case["user_turns"]) else None,
                "model_forward_count": len(generated["generated_token_ids"]),
                **generated,
            }
            records.append(record)
            messages.append({"role": "assistant", "content": generated["decoded_text"]})
            print(
                f"BF16_BASELINE {case['id']} turn={turn_index} "
                f"tokens={len(generated['generated_token_ids'])} "
                f"seconds={generated['wall_seconds']:.3f}",
                flush=True,
            )

    require(len(records) == matrix["response_count"], "BF16 response count differs")
    rubric = quality._build_rubric(records)
    rubric["candidate_id"] = None
    rubric["probe_id"] = PROBE_ID
    rubric["interpretation"] = (
        "Automated hard checks only. Fresh Reviewer scoring is not synthesized by this probe."
    )
    total_wall_seconds = time.monotonic() - started
    result = {
        "schema_version": 1,
        "probe_id": PROBE_ID,
        "status": "PASS_FOR_FRESH_REVIEW"
        if rubric["engineer_automated_preliminary_status"] == "PASS_FOR_FRESH_REVIEW"
        else "SOURCE_BASELINE_HARD_GATE_NO_GO",
        "created_at_utc": utc_now(),
        "claim_boundary": (
            "Non-consuming specification-stage BF16 source-model baseline only; no V16 replay, "
            "W4A8 successor selection, RTL, demo, PPA, U280, or project-completion claim."
        ),
        "source_contract_id": source_contract["contract_id"],
        "source_identity": source_identity,
        "tool_versions": versions,
        "frozen_matrix": file_record(MATRIX_PATH),
        "immutable_v16_result": file_record(V16_RESULT_PATH),
        "generation_contract": matrix["generation_contract"],
        "response_count": len(records),
        "model_forward_count": sum(int(item["model_forward_count"]) for item in records),
        "total_probe_wall_seconds": total_wall_seconds,
        "rubric": rubric,
        "responses": records,
    }
    write_exclusive_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "response_count": result["response_count"],
                "model_forward_count": result["model_forward_count"],
                "hard_check_response_pass_count": result["rubric"]["hard_check_response_pass_count"],
                "visible_single_turn_required_observable_count": result["rubric"]["visible_single_turn_required_observable_count"],
                "visible_multi_turn_required_observable_count": result["rubric"]["visible_multi_turn_required_observable_count"],
                "fresh_reviewer_holdout_required_observable_count": result["rubric"]["fresh_reviewer_holdout_required_observable_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
