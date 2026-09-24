#!/usr/bin/env python3
"""Probe a bounded, same-model BF16 safety/structure self-review mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probe_qwen_instruct_source_qualification_v1 as source_probe
from qwen_instruct_option_b import (
    SNAPSHOT,
    file_record,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_BF16_SELF_REVIEW_V1.json"
MATRIX_PATH = source_probe.MATRIX_PATH
PROBE_ID = "qwen2.5-0.5b-instruct-bf16-self-review-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = text.encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_contract() -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    require(contract.get("contract_id") == PROBE_ID, "self-review contract id differs")
    require(contract.get("status") == "FROZEN_NON_CONSUMING_BF16_PROBE", "self-review contract is not frozen")
    tooling = contract.get("tooling", {})
    require(sha256_file(Path(__file__)) == tooling.get("probe_tool_sha256"), "self-review probe tool differs")
    require(
        sha256_file(ROOT / tooling["corrected_evaluator_path"]) == tooling.get("corrected_evaluator_sha256"),
        "corrected evaluator differs",
    )
    evidence = contract.get("binding_evidence", {})
    for path_key, hash_key in (
        ("case_matrix_path", "case_matrix_sha256"),
        ("prior_source_qualification_path", "prior_source_qualification_sha256"),
        ("v16_terminal_review_path", "v16_terminal_review_sha256"),
    ):
        require(sha256_file(ROOT / evidence[path_key]) == evidence[hash_key], f"binding evidence differs: {path_key}")
    mechanism = contract.get("mechanism", {})
    require(mechanism.get("draft_token_cap") == 20, "draft token cap differs")
    require(mechanism.get("total_generated_token_cap_per_response") == 64, "total token cap differs")
    require(mechanism.get("final_answer_source") == "same_pinned_bf16_model", "final answer source differs")
    require(mechanism.get("software_answer_replacement") is False, "software answer replacement is enabled")
    require(mechanism.get("case_or_answer_specific_rules") is False, "case-specific rules are enabled")
    require(mechanism.get("prompt_or_answer_specific_few_shot_examples") is False, "few-shot answers are enabled")
    return contract


def generate_with_cap(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    token_cap: int,
) -> dict[str, Any]:
    require(token_cap >= 1, "generation token cap must be positive")
    previous = source_probe.quality.MAX_NEW_TOKENS
    source_probe.quality.MAX_NEW_TOKENS = token_cap
    try:
        return source_probe.quality._generate_response(model, tokenizer, messages)
    finally:
        source_probe.quality.MAX_NEW_TOKENS = previous


def mechanism_self_test(contract: dict[str, Any]) -> dict[str, bool]:
    mechanism = contract["mechanism"]
    review_instruction = mechanism["review_instruction"]
    checks = {
        "total_cap_is_frozen_64": mechanism["total_generated_token_cap_per_response"] == 64,
        "draft_cap_leaves_final_budget": 0 < mechanism["draft_token_cap"] < mechanism["total_generated_token_cap_per_response"],
        "same_model_produces_visible_answer": mechanism["final_answer_source"] == "same_pinned_bf16_model",
        "software_answer_replacement_disabled": mechanism["software_answer_replacement"] is False,
        "case_specific_rules_disabled": mechanism["case_or_answer_specific_rules"] is False,
        "few_shot_answers_disabled": mechanism["prompt_or_answer_specific_few_shot_examples"] is False,
        "review_instruction_has_no_matrix_case_ids": not any(
            case_id in review_instruction for case_id in (
                "arithmetic_purchase",
                "format_bullets",
                "logic_inference",
                "safe_refusal",
                "holdout_json_cow",
            )
        ),
    }
    require(all(checks.values()), "mechanism self-test failed")
    return checks


def run_probe(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"probe output already exists: {output}")
    companion = output.with_suffix(".sha256")
    require(not companion.exists(), f"probe checksum already exists: {companion}")
    contract = verify_contract()
    detector_checks = source_probe.detector_self_test()
    mechanism_checks = mechanism_self_test(contract)
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    matrix = load_json(MATRIX_PATH)
    mechanism = contract["mechanism"]
    system_message = contract["generation_contract"]["system_message"]
    total_cap = int(mechanism["total_generated_token_cap_per_response"])
    draft_cap = int(mechanism["draft_token_cap"])
    review_instruction = str(mechanism["review_instruction"])

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
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
        messages: list[dict[str, str]] = [{"role": "system", "content": system_message}]
        for turn_index, user_text in enumerate(case["user_turns"], start=1):
            messages.append({"role": "user", "content": user_text})
            messages_before = [dict(item) for item in messages]
            draft = generate_with_cap(model, tokenizer, messages_before, draft_cap)
            draft_token_count = len(draft["generated_token_ids"])
            final_cap = total_cap - draft_token_count
            require(final_cap >= 1, "hidden draft exhausted total generation cap")
            review_messages = [
                *messages_before,
                {"role": "assistant", "content": draft["decoded_text"]},
                {"role": "user", "content": review_instruction},
            ]
            final = generate_with_cap(model, tokenizer, review_messages, final_cap)
            total_generated = draft_token_count + len(final["generated_token_ids"])
            require(total_generated <= total_cap, "self-review exceeded total generation cap")
            record = {
                "schema_version": 1,
                "probe_id": PROBE_ID,
                "run_kind": "non_consuming_bf16_self_review",
                "case_id": case["id"],
                "category": case["category"],
                "turn_index": turn_index,
                "is_final_turn": turn_index == len(case["user_turns"]),
                "raw_user_prompt": user_text,
                "messages_before_generation": messages_before,
                "required_observable": case["required_observable"] if turn_index == len(case["user_turns"]) else None,
                "model_forward_count": total_generated,
                "mechanism_trace": {
                    "hidden_draft": draft,
                    "review_messages": review_messages,
                    "final_generation_token_cap": final_cap,
                    "total_generated_token_count": total_generated,
                    "total_generated_token_cap": total_cap,
                    "visible_answer_is_exact_final_model_decode": True,
                    "software_answer_replacement": False,
                },
                **final,
            }
            require(
                record["decoded_text"] == tokenizer.decode(record["generated_token_ids"], skip_special_tokens=True),
                "visible answer differs from final model decode",
            )
            records.append(record)
            messages.append({"role": "assistant", "content": final["decoded_text"]})
            print(
                f"BF16_SELF_REVIEW {case['id']} turn={turn_index} "
                f"draft_tokens={draft_token_count} final_tokens={len(final['generated_token_ids'])} "
                f"total_tokens={total_generated}",
                flush=True,
            )

    require(len(records) == 23, "self-review response count differs")
    rubric = source_probe.build_rubric(records)
    status = "SOURCE_QUALIFIED" if rubric["automated_source_qualification_status"] == "PASS" else "SOURCE_QUALIFICATION_NO_GO"
    result = {
        "schema_version": 1,
        "probe_id": PROBE_ID,
        "status": status,
        "created_at_utc": source_probe.utc_now(),
        "claim_boundary": (
            "Non-consuming specification-stage pinned-BF16 safety/structure probe only; no W4A8 policy, "
            "Fresh Reviewer score, RTL, demo, synthesis/PPA, U280, stage advance, or completion claim."
        ),
        "contract": file_record(CONTRACT_PATH),
        "probe_tool": file_record(Path(__file__)),
        "corrected_evaluator": file_record(Path(source_probe.__file__)),
        "source_contract_id": source_contract["contract_id"],
        "source_identity": source_identity,
        "tool_versions": versions,
        "case_matrix": file_record(MATRIX_PATH),
        "detector_self_test": detector_checks,
        "mechanism_self_test": mechanism_checks,
        "generation_contract": contract["generation_contract"],
        "mechanism": mechanism,
        "response_count": len(records),
        "model_forward_count": sum(int(item["model_forward_count"]) for item in records),
        "total_probe_wall_seconds": time.monotonic() - started,
        "rubric": rubric,
        "responses": records,
    }
    source_probe.write_exclusive_json(output, result)
    write_exclusive_text(companion, f"{sha256_file(output)}  {output.name}\n")
    return result


def verify_probe(output: Path) -> dict[str, Any]:
    contract = verify_contract()
    companion = output.with_suffix(".sha256")
    require(output.is_file() and companion.is_file(), "probe artifact or checksum is missing")
    expected_line = f"{sha256_file(output)}  {output.name}"
    require(companion.read_text(encoding="utf-8").strip() == expected_line, "probe checksum differs")
    result = load_json(output)
    require(result.get("probe_id") == PROBE_ID, "probe id differs")
    require(result.get("contract", {}).get("sha256") == sha256_file(CONTRACT_PATH), "recorded contract hash differs")
    require(result.get("probe_tool", {}).get("sha256") == sha256_file(Path(__file__)), "recorded probe tool hash differs")
    require(
        result.get("corrected_evaluator", {}).get("sha256") == sha256_file(Path(source_probe.__file__)),
        "recorded evaluator hash differs",
    )
    require(result.get("case_matrix", {}).get("sha256") == sha256_file(MATRIX_PATH), "recorded matrix hash differs")
    records = result.get("responses", [])
    require(len(records) == 23 and result.get("response_count") == 23, "probe response count differs")
    matrix = load_json(MATRIX_PATH)
    system_message = contract["generation_contract"]["system_message"]
    review_instruction = contract["mechanism"]["review_instruction"]
    total_cap = contract["mechanism"]["total_generated_token_cap_per_response"]
    record_index = 0
    for case in matrix["cases"]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_message}]
        for turn_index, user_text in enumerate(case["user_turns"], start=1):
            messages.append({"role": "user", "content": user_text})
            record = records[record_index]
            record_index += 1
            require(record["case_id"] == case["id"] and record["turn_index"] == turn_index, "response ordering differs")
            require(record["messages_before_generation"] == messages, "production transcript differs")
            trace = record["mechanism_trace"]
            expected_review = [
                *messages,
                {"role": "assistant", "content": trace["hidden_draft"]["decoded_text"]},
                {"role": "user", "content": review_instruction},
            ]
            require(trace["review_messages"] == expected_review, "review transcript differs")
            require(trace["software_answer_replacement"] is False, "software answer replacement recorded")
            require(trace["visible_answer_is_exact_final_model_decode"] is True, "visible-answer identity not recorded")
            require(
                record["generated_token_ids_sha256"]
                == source_probe.quality._token_ids_sha256(record["generated_token_ids"]),
                "final token hash differs",
            )
            require(
                trace["hidden_draft"]["generated_token_ids_sha256"]
                == source_probe.quality._token_ids_sha256(trace["hidden_draft"]["generated_token_ids"]),
                "draft token hash differs",
            )
            require(trace["total_generated_token_count"] <= total_cap, "recorded response exceeds total token cap")
            require(
                trace["total_generated_token_count"]
                == len(trace["hidden_draft"]["generated_token_ids"]) + len(record["generated_token_ids"]),
                "recorded total token count differs",
            )
            messages.append({"role": "assistant", "content": record["decoded_text"]})
    require(record_index == len(records), "unverified response records remain")
    recomputed_rubric = source_probe.build_rubric(records)
    require(result.get("rubric") == recomputed_rubric, "stored rubric differs from corrected evaluator")
    expected_status = "SOURCE_QUALIFIED" if recomputed_rubric["automated_source_qualification_status"] == "PASS" else "SOURCE_QUALIFICATION_NO_GO"
    require(result.get("status") == expected_status, "probe status differs from rubric")
    return {
        "integrity_status": "VERIFIED",
        "source_gate_status": result["status"],
        "response_count": result["response_count"],
        "model_forward_count": result["model_forward_count"],
        "hard_check_response_pass_count": recomputed_rubric["hard_check_response_pass_count"],
        "visible_single_turn_required_observable_count": recomputed_rubric["visible_single_turn_required_observable_count"],
        "visible_multi_turn_required_observable_count": recomputed_rubric["visible_multi_turn_required_observable_count"],
        "public_holdout_required_observable_count": recomputed_rubric["public_holdout_required_observable_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    contract = verify_contract()
    if args.self_test_only:
        summary = {
            "contract": "VERIFIED",
            "detectors": source_probe.detector_self_test(),
            "mechanism": mechanism_self_test(contract),
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    require(args.output is not None, "--output is required")
    output = args.output.resolve()
    if args.verify:
        print(json.dumps(verify_probe(output), sort_keys=True))
        return 0
    result = run_probe(output)
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
