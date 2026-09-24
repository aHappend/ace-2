#!/usr/bin/env python3
"""Artifact-only truncation audit for frozen BF16 and terminal V20 traces.

This tool never invokes a model, evaluator, simulator, or sealed campaign.  It
reads only already-published response metadata and rubric outcomes, and emits
no prompt text, decoded text, hidden harness content, or golden output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/probes/qwen-instruct-source-relative-quality-v2-reviewer-20260807T112030Z.json"
V20_DIR = ROOT / "build/stage1-option-b-post-v19-block16-affine-bf16-w4a8-v20/quality-campaign-0001"
V20_RAW = V20_DIR / "raw_outputs.jsonl"
V20_RUBRIC = V20_DIR / "source_relative_rubric.json"
V20_RESULT = V20_DIR / "RESULT.json"

EXPECTED_SHA256 = {
    SOURCE: "fdbba290ef91bd522c0a8e836eeeac7743fad5b5d836084a983b4db1c1cc489e",
    V20_RAW: "3b170cd8095d5505d861f264e6e250e64a83a30d15a79802afcc54f82028c03a",
    V20_RUBRIC: "03d44effdef239871341260f4f122e28c81909f28b48232fb2fa0bd0fd0556a4",
    V20_RESULT: "7f2d5d5afef2c3e5f932bf0e56e5eba40600c79bf3f0bdde9b20030557a47b14",
}

TERMINATION_IDS = {151643, 151645}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def key(row: dict[str, object]) -> tuple[str, int]:
    return str(row["case_id"]), int(row["turn_index"])


def cap_hit_without_eos(row: dict[str, object]) -> bool:
    token_ids = row.get("generated_token_ids")
    return bool(
        isinstance(token_ids, list)
        and len(token_ids) == 64
        and row.get("termination_reason") == "maximum_new_tokens"
        and row.get("terminating_token_id") is None
    )


def compact_case(case_key: tuple[str, int], classification: str) -> dict[str, object]:
    return {
        "case_id": case_key[0],
        "turn_index": case_key[1],
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA256.items():
        checks[f"input_hash_{path.name}"] = sha256(path) == expected
    if not all(checks.values()):
        raise RuntimeError("frozen input hash mismatch")

    source = load_json(SOURCE)
    source_rows = source.get("responses")
    source_rubric = source.get("rubric")
    if not isinstance(source_rows, list) or not isinstance(source_rubric, dict):
        raise RuntimeError("malformed BF16 source artifact")
    source_rubric_rows = source_rubric.get("responses")
    if not isinstance(source_rubric_rows, list):
        raise RuntimeError("malformed BF16 rubric")
    source_action = {key(row): bool(row.get("action_pass")) for row in source_rubric_rows if isinstance(row, dict)}

    v20_rows = load_jsonl(V20_RAW)
    v20_rubric = load_json(V20_RUBRIC)
    v20_result = load_json(V20_RESULT)
    v20_rubric_rows = v20_rubric.get("responses")
    if not isinstance(v20_rubric_rows, list):
        raise RuntimeError("malformed V20 rubric")
    v20_action = {
        key(row): (bool(row.get("source_action_pass")), bool(row.get("action_pass")))
        for row in v20_rubric_rows
        if isinstance(row, dict)
    }

    checks.update(
        {
            "source_response_count_23": len(source_rows) == 23,
            "v20_response_count_23": len(v20_rows) == 23,
            "source_historical_cap_64": source.get("generation_contract", {}).get(
                "maximum_new_tokens_per_delivered_response"
            )
            == 64,
            "v20_terminal_no_go_preserved": v20_result.get("status") == "SOURCE_RELATIVE_QUALITY_NO_GO"
            and v20_result.get("selected_policy_id") is None,
        }
    )

    source_by_key = {key(row): row for row in source_rows if isinstance(row, dict)}
    v20_by_key = {key(row): row for row in v20_rows}
    source_cap_keys = sorted(k for k, row in source_by_key.items() if cap_hit_without_eos(row))
    v20_cap_keys = sorted(k for k, row in v20_by_key.items() if cap_hit_without_eos(row))
    expected_source_cap = [("arithmetic_purchase", 1), ("simple_code", 1)]
    expected_v20_cap = [("arithmetic_purchase", 1)]
    checks["source_cap_hit_set_exact"] = source_cap_keys == expected_source_cap
    checks["v20_cap_hit_set_exact"] = v20_cap_keys == expected_v20_cap
    checks["all_cap_hits_are_rubric_failures"] = all(not source_action[k] for k in source_cap_keys) and all(
        not v20_action[k][1] for k in v20_cap_keys
    )

    # Exact suffix checks establish that each capped response stopped before its
    # required observable completed.  The suffixes are tested but never emitted.
    checks["source_arithmetic_incomplete_at_cap"] = str(
        source_by_key[("arithmetic_purchase", 1)].get("decoded_text", "")
    ).endswith("Since M")
    checks["source_code_incomplete_at_cap"] = str(
        source_by_key[("simple_code", 1)].get("decoded_text", "")
    ).endswith("# Example usage:\n")
    checks["v20_arithmetic_incomplete_at_cap"] = str(
        v20_by_key[("arithmetic_purchase", 1)].get("decoded_text", "")
    ).endswith("The equation")

    source_failure_keys = sorted(k for k, passed in source_action.items() if not passed)
    v20_failure_keys = sorted(k for k, (_source_pass, candidate_pass) in v20_action.items() if not candidate_pass)
    v20_regression_keys = sorted(
        k for k, (source_pass, candidate_pass) in v20_action.items() if source_pass and not candidate_pass
    )
    checks["source_failure_count_11"] = len(source_failure_keys) == 11
    checks["v20_failure_count_10"] = len(v20_failure_keys) == 10
    checks["v20_unique_regression_is_concise_summary"] = v20_regression_keys == [("concise_summary", 1)]

    decisive = v20_by_key[("concise_summary", 1)]
    decisive_tokens = decisive.get("generated_token_ids")
    checks["v20_decisive_regression_wrong_before_cap"] = bool(
        isinstance(decisive_tokens, list)
        and len(decisive_tokens) == 33
        and decisive.get("termination_reason") == "termination_token"
        and decisive.get("terminating_token_id") in TERMINATION_IDS
        and not v20_action[("concise_summary", 1)][1]
    )

    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("audit checks failed: " + ", ".join(failed))

    source_wrong_before_cap = sorted(set(source_failure_keys) - set(source_cap_keys))
    v20_wrong_before_cap = sorted(set(v20_failure_keys) - set(v20_cap_keys))
    result = {
        "schema_version": 1,
        "audit_id": "frozen-bf16-v20-generation-truncation-v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS_ARTIFACT_ONLY_NO_REPLAY",
        "inputs": {
            str(path.relative_to(ROOT)): {"sha256": expected}
            for path, expected in EXPECTED_SHA256.items()
        },
        "checks": dict(sorted(checks.items())),
        "historical_64_token_results": {
            "bf16_source": {
                "response_count": len(source_rows),
                "rubric_failure_count": len(source_failure_keys),
                "hit_64_without_eos_count": len(source_cap_keys),
                "plausibly_truncation_only": [
                    compact_case(k, "PLAUSIBLY_TRUNCATION_ONLY") for k in source_cap_keys
                ],
                "wrong_before_cap_or_other_nontruncation_failure_count": len(source_wrong_before_cap),
            },
            "terminal_v20": {
                "response_count": len(v20_rows),
                "rubric_failure_count": len(v20_failure_keys),
                "hit_64_without_eos_count": len(v20_cap_keys),
                "plausibly_truncation_only": [
                    compact_case(k, "PLAUSIBLY_TRUNCATION_ONLY") for k in v20_cap_keys
                ],
                "wrong_before_cap_or_other_nontruncation_failure_count": len(v20_wrong_before_cap),
                "decisive_source_passing_regression": compact_case(
                    ("concise_summary", 1), "WRONG_BEFORE_CAP_EOS_AT_33"
                ),
            },
        },
        "routing_decision": {
            "single_authorized_sft_pilot_remains_necessary": True,
            "reason": "A larger generation budget can plausibly repair only the historical cap-hit failures. V20's decisive source-passing concise-summary regression terminated with EOS at 33 tokens, and nine BF16 source rubric failures occurred without hitting the cap.",
            "additional_sft_scope_authorized_by_this_audit": False,
        },
        "forward_generation_budget": {
            "primary": "EOS-stopped greedy generation with max_new_tokens=128",
            "diagnostic": "A bounded max_new_tokens=256 rerun is permitted only for a fresh response that reaches 128 without EOS",
            "historical_64_results_overwritten": False,
        },
        "claim_boundary": "Artifact-only classification. No model or evaluator execution occurred, no sealed campaign was replayed, and no 128- or 256-token output result is claimed.",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "output": str(output.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
