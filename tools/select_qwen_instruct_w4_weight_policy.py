#!/usr/bin/env python3
"""Select one Option-B v2 W4 policy from a completed discrimination matrix."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any

from qwen_instruct_option_b import (
    ROOT,
    canonical_bytes,
    file_record,
    load_json,
    require,
    verify_versions,
)


EXPECTED_POLICY_IDS = {
    "layer2_gate_per_128",
    "layer2_gate_per_64",
    "layer2_gate_per_32",
    "layer23_down_per_64",
    "layer23_down_per_32",
    "layer23_gate_per_64",
    "layer23_gate_per_32",
}


def require_project_python() -> dict[str, Any]:
    expected = ROOT / ".venv/bin/python"
    observed = Path(sys.executable)
    require(expected.is_file(), f"project Python is missing: {expected}")
    require(os.path.samefile(observed, expected), f"wrong Python executable: {observed}")
    return {
        "bound_entrypoint": "./.venv/bin/python",
        "bound_entrypoint_absolute": str(expected),
        "resolved_executable": str(expected.resolve()),
        "sys_executable": str(observed),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": verify_versions(),
    }


def rank_key(record: dict[str, Any]) -> tuple[int, int, int, float, float, int]:
    aggregate = record["aggregate"]
    cost = record["metadata_and_compute_cost"]
    return (
        -int(aggregate["exact_bf16_sequence_match_count"]),
        -int(aggregate["response_gate_outcome_match_count"]),
        -int(aggregate["total_common_bf16_prefix_tokens"]),
        float(aggregate["worst_aligned_relative_l2_error"]),
        float(aggregate["mean_aligned_relative_l2_error"]),
        int(cost["additional_scale32_metadata_bytes"]),
    )


def public_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": record["policy_id"],
        "operator": record["operator"],
        "input_group_lanes": record["input_group_lanes"],
        "aggregate": record["aggregate"],
        "weight_error": record["weight_error"],
        "metadata_and_compute_cost": record["metadata_and_compute_cost"],
        "full_sequence_prompt_outcomes": [
            {
                "case_id": prompt["case_id"],
                "generated_token_ids": prompt["generated_token_ids"],
                "decoded_text": prompt["decoded_text"],
                "exact_bf16_sequence_match": prompt["exact_bf16_sequence_match"],
                "common_bf16_prefix_tokens": prompt["common_bf16_prefix_tokens"],
                "response_gate_status": prompt["response_gate"]["status"],
                "bf16_response_gate_status": prompt["bf16_response_gate_status"],
            }
            for prompt in record["prompts"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    environment = require_project_python()
    matrix_path = args.matrix.resolve()
    matrix = load_json(matrix_path)
    require(
        matrix.get("classification")
        == "qwen_instruct_option_b_w4_weight_policy_full_sequence_discrimination",
        "weight-policy matrix classification differs",
    )
    policies = matrix.get("candidate_policies")
    require(isinstance(policies, list) and len(policies) == 7, "matrix policy count differs")
    require(
        {record["policy_id"] for record in policies} == EXPECTED_POLICY_IDS,
        "matrix policy identities differ",
    )
    require(
        matrix.get("prompt_suite", {}).get("case_count") == 9,
        "matrix prompt count differs",
    )
    require(
        matrix.get("bf16_reference", {}).get("independent_response_gate_pass_count")
        == 4,
        "matrix BF16 response-gate screen differs",
    )
    require(
        matrix.get("scope_guards", {}).get("frozen_v1_artifacts_mutated") is False
        and matrix.get("scope_guards", {}).get("demo_retargeted") is False
        and matrix.get("scope_guards", {}).get("ppa_executed") is False,
        "matrix scope guards differ",
    )

    ranked = sorted(policies, key=rank_key)
    winner = ranked[0]
    unique = sum(rank_key(record) == rank_key(winner) for record in ranked) == 1
    require(unique, "weight-policy adjudication is not unique")
    status = "PASS_UNIQUE_SEQUENCE_FIRST_WEIGHT_POLICY_SELECTED"
    result = {
        "schema_version": 1,
        "classification": "qwen_instruct_option_b_v2_weight_policy_adjudication",
        "status": status,
        "environment": environment,
        "selection_rule": {
            "scope": "completed source-bound nine-prompt full-sequence matrix",
            "lexicographic_order": [
                "maximize exact complete BF16 sequence matches",
                "maximize response-gate outcome matches to BF16",
                "maximize total common BF16 prefix tokens",
                "minimize worst aligned full-vocabulary relative-L2 logit error",
                "minimize mean aligned full-vocabulary relative-L2 logit error",
                "minimize additional 32-bit weight-scale metadata bytes",
            ],
            "stable_name_tie_break_used": False,
        },
        "selected": public_summary(winner),
        "ranked_policy_ids": [record["policy_id"] for record in ranked],
        "ranked_summaries": [public_summary(record) for record in ranked],
        "artifacts": {
            "discrimination_matrix": file_record(matrix_path),
            "selector_source": file_record(Path(__file__)),
        },
        "scope_guards": {
            "model_executed": False,
            "frozen_v1_artifacts_mutated": False,
            "accelerator_executed": False,
            "demo_retargeted": False,
            "network_access_performed": False,
            "ppa_executed": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(result))
    print(
        "ACE2_QWEN_INSTRUCT_W4_POLICY_SELECTED "
        f"status={status} selected={winner['policy_id']} "
        f"output={output.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
