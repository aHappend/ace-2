#!/usr/bin/env python3
"""Frozen transparent semantic gate for Reviewer-selected Option-B nonce prompts."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any

from qwen_instruct_option_b import (
    IMAGE_DIR,
    TERMINATION_TOKEN_IDS,
    canonical_bytes,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_wrapped_contract,
)


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def invalid_character(value: str) -> str | None:
    for character in value:
        if character == "\ufffd" or unicodedata.category(character) in {"Cc", "Cs"}:
            return f"U+{ord(character):04X}"
    return None


def evaluate(result: dict[str, Any], expected_text: str) -> dict[str, Any]:
    decoded = result.get("decoded_text")
    token_ids = result.get("generated_token_ids")
    if not isinstance(decoded, str) or not isinstance(token_ids, list):
        raise RuntimeError("oracle/runtime result schema differs")
    normalized_actual = normalize(decoded)
    normalized_expected = normalize(expected_text)
    require(normalized_expected != "", "expected response is empty")
    bad = invalid_character(normalized_actual)
    nonterminating: list[int] = []
    for value in token_ids:
        token = int(value)
        if token in TERMINATION_TOKEN_IDS:
            break
        nonterminating.append(token)
    checks = {
        "exact_normalized_text": normalized_actual == normalized_expected,
        "minimum_two_nonterminating_tokens": len(nonterminating) >= 2,
        "nonempty_visible_text": bool(normalized_actual),
        "no_control_surrogate_or_replacement_character": bad is None,
    }
    return {
        "schema_version": 1,
        "classification": "qwen_instruct_option_b_exact_nonce_response_gate_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "generated_token_ids": [int(value) for value in token_ids],
        "nonterminating_generated_token_count": len(nonterminating),
        "decoded_text_sha256": sha256_bytes(decoded.encode()),
        "normalized_decoded_text_sha256": sha256_bytes(normalized_actual.encode()),
        "expected_text_sha256": sha256_bytes(expected_text.encode()),
        "normalized_expected_text_sha256": sha256_bytes(normalized_expected.encode()),
        "first_invalid_character": bad,
    }


def self_test() -> None:
    passing = evaluate({"decoded_text": " amber compass ", "generated_token_ids": [1, 2]}, "amber compass")
    require(passing["status"] == "PASS", "response-gate positive self-test failed")
    cases = [
        ({"decoded_text": "amber", "generated_token_ids": [1]}, "amber"),
        ({"decoded_text": "amber compass!", "generated_token_ids": [1, 2]}, "amber compass"),
        ({"decoded_text": "amber\u0000compass", "generated_token_ids": [1, 2]}, "amber compass"),
        ({"decoded_text": "amber compass", "generated_token_ids": [151645]}, "amber compass"),
    ]
    require(all(evaluate(result, expected)["status"] == "FAIL" for result, expected in cases), "response-gate negative self-test failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path)
    parser.add_argument("--expected-text")
    parser.add_argument("--contract", type=Path, default=IMAGE_DIR / "response_gate_contract.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ACE2_QWEN_INSTRUCT_RESPONSE_GATE_SELF_TEST_PASS")
        return
    require(args.result is not None and args.expected_text is not None and args.output is not None, "result, expected text, and output are required")
    contract = verify_wrapped_contract(args.contract.resolve())
    require(contract.get("contract_id") == "qwen-instruct-option-b-response-gate-v1", "response-gate contract differs")
    require(contract["source"]["sha256"] == sha256_file(Path(__file__)), "response-gate source differs")
    gate = evaluate(load_json(args.result.resolve()), args.expected_text)
    gate["contract_sha256"] = sha256_file(args.contract.resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_bytes(canonical_bytes(gate))
    print(f"ACE2_QWEN_INSTRUCT_RESPONSE_GATE_{gate['status']} output={args.output.resolve()}")
    if gate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
