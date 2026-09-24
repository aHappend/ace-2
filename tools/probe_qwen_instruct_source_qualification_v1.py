#!/usr/bin/env python3
"""Run the frozen non-consuming Qwen source-qualification probe."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
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
CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_SOURCE_QUALIFICATION_V1.json"
MATRIX_PATH = (
    ROOT
    / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v16"
    / "quality-campaign-0001/frozen_matrix.json"
)
V16_RAW_OUTPUTS_PATH = MATRIX_PATH.parent / "raw_outputs.jsonl"
PROBE_ID = "qwen2.5-0.5b-instruct-source-qualification-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def logic_observable(text: str) -> bool:
    value = re.sub(r"\s+", " ", text.strip().lower())
    if not value.startswith("no"):
        return False
    if not all(word in value for word in ("glib", "red", "transparent")):
        return False
    if any(phrase in value for phrase in ("can be transparent", "may be transparent", "transparent appearance")):
        return False
    return any(
        phrase in value
        for phrase in (
            "cannot be transparent",
            "can't be transparent",
            "not transparent",
            "no red object is transparent",
            "red objects are not transparent",
        )
    )


def simple_code_observable(text: str) -> bool:
    definition = re.search(r"(?m)^\s*def\s+is_even\s*\(", text)
    modulo = re.search(r"%\s*2", text)
    example = re.search(r"(?m)^\s*(?:print\s*\(\s*)?is_even\s*\(", text)
    return bool(definition and modulo and example)


def case_observable(case_id: str, text: str) -> bool:
    if case_id == "logic_inference":
        return logic_observable(text)
    if case_id == "simple_code":
        return simple_code_observable(text)
    return quality._case_observable(case_id, text)


def hard_checks(record: dict[str, Any]) -> dict[str, bool]:
    text_value = record["decoded_text"]
    controls_ok = all(ord(char) >= 32 or char in "\n\r\t" for char in text_value)
    words = re.findall(r"[\w'-]+", text_value.lower(), flags=re.UNICODE)
    grams = Counter(tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3)))
    return {
        "valid_utf8_no_replacement_or_disallowed_controls": "\ufffd" not in text_value and controls_ok,
        "at_least_two_visible_nonterminating_tokens": int(record["visible_nonterminating_generated_token_count"]) >= 2,
        "no_runtime_nan_empty_or_malformed_decode": bool(text_value.strip()) and record["all_logits_finite"] is True,
        "no_degenerate_repeated_nonpunctuation_four_gram": not grams or max(grams.values()) <= 2,
        "case_specific_required_observable": case_observable(record["case_id"], text_value) if record["is_final_turn"] else True,
    }


def build_rubric(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    responses = []
    for record in outputs:
        checks = hard_checks(record)
        responses.append(
            {
                "case_id": record["case_id"],
                "turn_index": record["turn_index"],
                "checks": checks,
                "all_hard_checks_pass": all(checks.values()),
            }
        )
    paired = list(zip(responses, outputs, strict=True))
    visible_single = [item for item, source in paired if source["is_final_turn"] and source["category"] == "visible_single_turn"]
    visible_multi = [item for item, source in paired if source["is_final_turn"] and source["category"] == "visible_multi_turn"]
    holdouts = [item for item, source in paired if source["is_final_turn"] and source["category"].startswith("fresh_reviewer_holdout")]
    observable = lambda items: sum(item["checks"]["case_specific_required_observable"] for item in items)
    hard_count = sum(item["all_hard_checks_pass"] for item in responses)
    passed = hard_count == len(responses) and observable(visible_single) >= 11 and observable(visible_multi) == 3 and observable(holdouts) == 4
    return {
        "response_count": len(responses),
        "hard_check_response_pass_count": hard_count,
        "hard_check_response_pass_rate": hard_count / len(responses),
        "visible_single_turn_required_observable_count": observable(visible_single),
        "visible_multi_turn_required_observable_count": observable(visible_multi),
        "public_holdout_required_observable_count": observable(holdouts),
        "automated_source_qualification_status": "PASS" if passed else "NO_GO",
        "responses": responses,
    }


def detector_self_test() -> dict[str, bool]:
    v16_records = [json.loads(line) for line in V16_RAW_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines()]
    by_case = {item["case_id"]: item for item in v16_records if item["case_id"] in {"logic_inference", "simple_code"}}
    checks = {
        "rejects_v16_logic_false_positive": not logic_observable(by_case["logic_inference"]["decoded_text"]),
        "accepts_correct_logic_example": logic_observable(
            "No. All glibs are red, and no red object is transparent, so glibs cannot be transparent."
        ),
        "rejects_v16_code_without_example_call": not simple_code_observable(by_case["simple_code"]["decoded_text"]),
        "accepts_code_with_example_call": simple_code_observable(
            "def is_even(value):\n    return value % 2 == 0\n\nprint(is_even(4))"
        ),
    }
    require(all(checks.values()), "detector self-test failed")
    return checks


def run_probe(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"probe output already exists: {output}")
    contract = load_json(CONTRACT_PATH)
    expected_matrix_sha256 = contract["source_qualification_contract"]["case_source_sha256"]
    require(quality.sha256_file(MATRIX_PATH) == expected_matrix_sha256, "source case matrix hash differs")
    require(
        quality.sha256_file(ROOT / contract["binding_evidence"]["v16_terminal_result"])
        == contract["binding_evidence"]["v16_terminal_result_sha256"],
        "V16 terminal result hash differs",
    )
    baseline_path = ROOT / contract["binding_evidence"]["exact_v16_contract_bf16_probe"]
    require(
        quality.sha256_file(baseline_path)
        == contract["binding_evidence"]["exact_v16_contract_bf16_probe_sha256"],
        "exact-contract BF16 baseline hash differs",
    )
    detector_checks = detector_self_test()
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    matrix = load_json(MATRIX_PATH)
    source_rules = contract["source_qualification_contract"]
    require(source_rules["maximum_new_tokens_per_response"] == 64, "source token cap differs")

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    torch.set_num_threads(32)
    previous_max = quality.MAX_NEW_TOKENS
    quality.MAX_NEW_TOKENS = 64
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for case in matrix["cases"]:
            messages: list[dict[str, str]] = [{"role": "system", "content": source_rules["system_message"]}]
            for turn_index, user_text in enumerate(case["user_turns"], start=1):
                messages.append({"role": "user", "content": user_text})
                messages_before = [dict(item) for item in messages]
                generated = quality._generate_response(model, tokenizer, messages_before)
                record = {
                    "schema_version": 1,
                    "probe_id": PROBE_ID,
                    "run_kind": "non_consuming_source_qualification",
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
                    f"SOURCE_QUALIFICATION {case['id']} turn={turn_index} "
                    f"tokens={len(generated['generated_token_ids'])} seconds={generated['wall_seconds']:.3f}",
                    flush=True,
                )
    finally:
        quality.MAX_NEW_TOKENS = previous_max

    require(len(records) == 23, "source qualification response count differs")
    rubric = build_rubric(records)
    result = {
        "schema_version": 1,
        "probe_id": PROBE_ID,
        "status": "SOURCE_QUALIFIED" if rubric["automated_source_qualification_status"] == "PASS" else "SOURCE_QUALIFICATION_NO_GO",
        "created_at_utc": utc_now(),
        "claim_boundary": (
            "Non-consuming specification-stage pinned-BF16 source qualification only; no W4A8 policy, "
            "Fresh Reviewer score, RTL, demo, synthesis/PPA, U280, stage advance, or completion claim."
        ),
        "contract": file_record(CONTRACT_PATH),
        "probe_tool": file_record(Path(__file__)),
        "source_contract_id": source_contract["contract_id"],
        "source_identity": source_identity,
        "tool_versions": versions,
        "case_matrix": file_record(MATRIX_PATH),
        "detector_self_test": detector_checks,
        "generation_contract": source_rules,
        "response_count": len(records),
        "model_forward_count": sum(int(item["model_forward_count"]) for item in records),
        "total_probe_wall_seconds": time.monotonic() - started,
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
                "public_holdout_required_observable_count": result["rubric"]["public_holdout_required_observable_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
