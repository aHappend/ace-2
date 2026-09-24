#!/usr/bin/env python3
"""Localize the frozen four-operator W4 error on BF16 teacher-forced paths.

This probe intentionally does not select a grouped-weight policy.  It evaluates
all 16 BF16/W4 restoration subsets of the already-localized four operators on
the unchanged nine-prompt suite, while every other weight and all activations
remain BF16.  Version 2 freezes greedy selection as torch.argmax so exact logit
ties use the same lowest-index rule as the existing generation path.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from discriminate_qwen_instruct_w4_weight_policies import CAUSAL_OPERATORS, PROMPTS
from localize_qwen_instruct_w4_weight_error import quantize_w4_into
from qwen_instruct_option_b import (
    ROOT,
    SNAPSHOT,
    canonical_bytes,
    file_record,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM


MATRIX = (
    ROOT
    / "build/option-b-v2-weight-policy-20260806/"
    "full_sequence_discrimination.json"
)
PROMPT_SOURCE = ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py"


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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(set(payload) == {"contract", "contract_sha256"}, "contract envelope differs")
    require(
        canonical_sha256(payload["contract"]) == payload["contract_sha256"],
        "predeclared contract digest differs",
    )
    contract = payload["contract"]
    require(contract["schema_version"] == 1, "unsupported contract schema")
    require(
        contract["contract_id"]
        == "qwen-instruct-four-operator-teacher-forced-causality-v2",
        "contract identity differs",
    )
    require(
        contract["test_source"]["sha256"] == sha256_file(Path(__file__)),
        "test source changed after predeclaration",
    )
    require(
        contract["matrix"]["sha256"] == sha256_file(MATRIX),
        "frozen discrimination matrix changed",
    )
    require(
        contract["prompt_binding"]["prompt_source"]["sha256"]
        == sha256_file(PROMPT_SOURCE),
        "prompt/policy source changed after predeclaration",
    )
    require(
        contract["prompt_binding"]["system_prompt_sha256"]
        == sha256_bytes(DEFAULT_SYSTEM.encode()),
        "system prompt changed after predeclaration",
    )
    require(
        contract["causal_operators"] == list(CAUSAL_OPERATORS),
        "causal operator order differs",
    )
    require(contract["subset_count"] == 16, "subset count differs")
    return payload


def load_frozen_matrix(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    declared_matrix = contract["contract"]["matrix"]
    require(matrix["status"] == declared_matrix["expected_status"], "matrix status differs")
    selection = matrix["selection"]
    require(
        selection["selected_policy_id"] == declared_matrix["expected_selected_policy_id"],
        "matrix unexpectedly selects a policy",
    )
    require(
        selection["eligible_policy_ids"] == declared_matrix["expected_eligible_policy_ids"],
        "matrix unexpectedly has an eligible policy",
    )
    require(not selection["unique_selected_policy"], "matrix unexpectedly reports uniqueness")

    reference_by_case = {
        record["case_id"]: record for record in matrix["bf16_reference"]["prompts"]
    }
    declared_cases = contract["contract"]["prompt_binding"]["cases"]
    require(len(declared_cases) == len(PROMPTS) == 9, "prompt count differs")
    require(set(reference_by_case) == {case.case_id for case in PROMPTS}, "case IDs differ")
    for prompt_case, declared in zip(PROMPTS, declared_cases, strict=True):
        require(declared["case_id"] == prompt_case.case_id, "predeclared case order differs")
        require(
            declared["prompt_sha256"] == sha256_bytes(prompt_case.prompt.encode()),
            f"prompt hash differs: {prompt_case.case_id}",
        )
        frozen_reference = reference_by_case[prompt_case.case_id]
        require(
            declared["bf16_generated_token_ids"] == frozen_reference["generated_token_ids"],
            f"BF16 token binding differs: {prompt_case.case_id}",
        )

    policy_disagreements: dict[str, Any] = {}
    for prompt_case in PROMPTS:
        outcomes = [
            next(
                prompt
                for prompt in policy["prompts"]
                if prompt["case_id"] == prompt_case.case_id
            )
            for policy in matrix["candidate_policies"]
        ]
        policy_disagreements[prompt_case.case_id] = {
            "exact_sequence_policy_count": sum(
                bool(outcome["exact_bf16_sequence_match"]) for outcome in outcomes
            ),
            "tested_policy_count": len(outcomes),
            "common_bf16_prefix_tokens": {
                policy["policy_id"]: outcome["common_bf16_prefix_tokens"]
                for policy, outcome in zip(
                    matrix["candidate_policies"], outcomes, strict=True
                )
            },
            "generated_token_ids": {
                policy["policy_id"]: outcome["generated_token_ids"]
                for policy, outcome in zip(
                    matrix["candidate_policies"], outcomes, strict=True
                )
            },
        }
    return matrix, policy_disagreements


def chat_input(tokenizer: Any, prompt: str) -> Tensor:
    value = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(value, Tensor) and value.ndim == 2, "tokenization shape differs")
    return value


def teacher_forced_input(prompt_ids: Tensor, generated_token_ids: list[int]) -> Tensor:
    require(generated_token_ids, "BF16 generated sequence is empty")
    continuation = torch.tensor(
        [generated_token_ids[:-1]], dtype=prompt_ids.dtype, device=prompt_ids.device
    )
    return torch.cat((prompt_ids, continuation), dim=1)


def prediction_logits(logits: Tensor, prompt_token_count: int, step_count: int) -> Tensor:
    start = prompt_token_count - 1
    stop = start + step_count
    require(stop <= logits.shape[1], "teacher-forced prediction positions exceed logits")
    return logits[0, start:stop].detach().cpu()


def linear_override(reference: nn.Linear) -> Callable[[nn.Module, tuple[Any, ...], Any], Tensor]:
    def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> Tensor:
        return F.linear(inputs[0], reference.weight, reference.bias)

    return hook


def evaluate_subset(
    candidate_model: nn.Module,
    reference_modules: dict[str, nn.Module],
    candidate_modules: dict[str, nn.Module],
    w4_operators: frozenset[str],
    input_ids: Tensor,
    prompt_token_count: int,
    step_count: int,
) -> Tensor:
    hooks: list[Any] = []
    for name in CAUSAL_OPERATORS:
        if name in w4_operators:
            continue
        reference = reference_modules[name]
        candidate = candidate_modules[name]
        require(
            isinstance(reference, nn.Linear) and isinstance(candidate, nn.Linear),
            f"causal operator is not linear: {name}",
        )
        hooks.append(candidate.register_forward_hook(linear_override(reference)))
    try:
        with torch.inference_mode():
            logits = candidate_model(input_ids=input_ids, use_cache=False).logits
        return prediction_logits(logits, prompt_token_count, step_count)
    finally:
        for hook in hooks:
            hook.remove()


def subset_id(operators: frozenset[str]) -> str:
    if not operators:
        return "bf16_all_four"
    return "w4__" + "__".join(name.replace("model.layers.", "l").replace(".mlp.", "_") for name in sorted(operators))


def step_comparison(reference: Tensor, candidate: Tensor, token_id: int) -> dict[str, Any]:
    left = reference.to(torch.float64)
    right = candidate.to(torch.float64)
    difference = right - left
    denominator = torch.linalg.vector_norm(left)
    top = torch.topk(right, k=2)
    candidate_token = int(right.argmax())
    return {
        "bf16_token_id": token_id,
        "candidate_token_id": candidate_token,
        "token_match": candidate_token == token_id,
        "candidate_top_one_margin": float(top.values[0] - top.values[1]),
        "candidate_top_one_tied": bool(top.values[0] == top.values[1]),
        "candidate_top_two_token_ids": [int(value) for value in top.indices],
        "relative_l2_error": (
            None
            if float(denominator) == 0.0
            else float(torch.linalg.vector_norm(difference) / denominator)
        ),
        "maximum_absolute_error": float(difference.abs().amax()),
    }


def proper_single_removals(operators: frozenset[str]) -> list[frozenset[str]]:
    return [operators - {name} for name in operators]


def attribute_prompt(
    prompt_records: list[dict[str, Any]], generated_token_ids: list[int]
) -> dict[str, Any]:
    by_set = {
        frozenset(record["w4_operators"]): record for record in prompt_records
    }
    require(len(by_set) == 16, "prompt subset records are incomplete")
    sequence_failures = [record for record in prompt_records if not record["all_tokens_match"]]
    minimum_failure_size = (
        None
        if not sequence_failures
        else min(record["w4_operator_count"] for record in sequence_failures)
    )
    exact_minimum_failure_sets = [
        record["w4_operators"]
        for record in sequence_failures
        if record["w4_operator_count"] == minimum_failure_size
    ]
    one_minimal_failure_sets = [
        record["w4_operators"]
        for record in sequence_failures
        if record["w4_operator_count"] > 0
        and all(by_set[reduced]["all_tokens_match"] for reduced in proper_single_removals(frozenset(record["w4_operators"])))
    ]

    step_attribution = []
    for step_index, token_id in enumerate(generated_token_ids):
        failing = [
            record
            for record in prompt_records
            if not record["steps"][step_index]["token_match"]
        ]
        minimum_size = (
            None if not failing else min(record["w4_operator_count"] for record in failing)
        )
        exact_minimum = [
            record["w4_operators"]
            for record in failing
            if record["w4_operator_count"] == minimum_size
        ]
        one_minimal = []
        for record in failing:
            operators = frozenset(record["w4_operators"])
            if not operators:
                continue
            if all(
                by_set[reduced]["steps"][step_index]["token_match"]
                for reduced in proper_single_removals(operators)
            ):
                one_minimal.append(record["w4_operators"])
        step_attribution.append(
            {
                "step_index": step_index,
                "bf16_token_id": token_id,
                "failing_subset_count": len(failing),
                "minimum_w4_operator_count_causing_mismatch": minimum_size,
                "exact_minimum_failure_sets": exact_minimum,
                "one_minimal_failure_sets": one_minimal,
            }
        )

    all_four = by_set[frozenset(CAUSAL_OPERATORS)]
    return {
        "all_four_w4": {
            "all_tokens_match": all_four["all_tokens_match"],
            "common_bf16_prefix_tokens": all_four["common_bf16_prefix_tokens"],
            "candidate_token_ids": [step["candidate_token_id"] for step in all_four["steps"]],
        },
        "sequence_failure_subset_count": len(sequence_failures),
        "minimum_w4_operator_count_causing_sequence_mismatch": minimum_failure_size,
        "exact_minimum_sequence_failure_sets": exact_minimum_failure_sets,
        "one_minimal_sequence_failure_sets": one_minimal_failure_sets,
        "steps": step_attribution,
    }


def all_subsets() -> list[frozenset[str]]:
    result: list[frozenset[str]] = []
    for size in range(len(CAUSAL_OPERATORS) + 1):
        result.extend(
            frozenset(values) for values in itertools.combinations(CAUSAL_OPERATORS, size)
        )
    require(len(result) == 16, "causal subset enumeration differs")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite first-attempt output: {output}")
    environment = require_project_python()
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    contract_envelope = load_contract(args.contract.resolve())
    matrix, matrix_disagreements = load_frozen_matrix(contract_envelope)

    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    candidate_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    reference_modules = dict(reference_model.named_modules())
    candidate_modules = dict(candidate_model.named_modules())
    for name in CAUSAL_OPERATORS:
        reference = reference_modules.get(name)
        candidate = candidate_modules.get(name)
        require(
            isinstance(reference, nn.Linear) and isinstance(candidate, nn.Linear),
            f"causal operator missing or not linear: {name}",
        )
        quantize_w4_into(candidate.weight, reference.weight)

    reference_by_case = {
        record["case_id"]: record for record in matrix["bf16_reference"]["prompts"]
    }
    prepared: dict[str, dict[str, Any]] = {}
    for prompt_case in PROMPTS:
        prompt_ids = chat_input(tokenizer, prompt_case.prompt)
        frozen_tokens = reference_by_case[prompt_case.case_id]["generated_token_ids"]
        teacher_ids = teacher_forced_input(prompt_ids, frozen_tokens)
        with torch.inference_mode():
            logits = reference_model(input_ids=teacher_ids, use_cache=False).logits
        prepared[prompt_case.case_id] = {
            "prompt_ids": prompt_ids,
            "teacher_ids": teacher_ids,
            "generated_token_ids": frozen_tokens,
            "reference_logits": prediction_logits(
                logits, int(prompt_ids.shape[1]), len(frozen_tokens)
            ),
        }

    prompt_subset_records: dict[str, list[dict[str, Any]]] = {
        prompt_case.case_id: [] for prompt_case in PROMPTS
    }
    subsets = all_subsets()
    for subset_index, operators in enumerate(subsets, start=1):
        print(
            "ACE2_FOUR_OPERATOR_CAUSALITY_PROGRESS "
            f"subset={subset_index}/{len(subsets)} id={subset_id(operators)}",
            flush=True,
        )
        for prompt_case in PROMPTS:
            item = prepared[prompt_case.case_id]
            candidate_logits = evaluate_subset(
                candidate_model,
                reference_modules,
                candidate_modules,
                operators,
                item["teacher_ids"],
                int(item["prompt_ids"].shape[1]),
                len(item["generated_token_ids"]),
            )
            steps = [
                {
                    "step_index": step_index,
                    **step_comparison(reference, candidate, token_id),
                }
                for step_index, (reference, candidate, token_id) in enumerate(
                    zip(
                        item["reference_logits"],
                        candidate_logits,
                        item["generated_token_ids"],
                        strict=True,
                    )
                )
            ]
            common_prefix = 0
            for step in steps:
                if not step["token_match"]:
                    break
                common_prefix += 1
            prompt_subset_records[prompt_case.case_id].append(
                {
                    "subset_id": subset_id(operators),
                    "w4_operator_count": len(operators),
                    "w4_operators": sorted(operators),
                    "all_tokens_match": all(step["token_match"] for step in steps),
                    "common_bf16_prefix_tokens": common_prefix,
                    "steps": steps,
                }
            )

    empty_exact = True
    prompt_results = []
    for prompt_case in PROMPTS:
        records = prompt_subset_records[prompt_case.case_id]
        empty = next(record for record in records if record["w4_operator_count"] == 0)
        empty_exact &= all(
            step["token_match"] and step["maximum_absolute_error"] == 0.0
            for step in empty["steps"]
        )
        item = prepared[prompt_case.case_id]
        prompt_results.append(
            {
                "case_id": prompt_case.case_id,
                "prompt_sha256": sha256_bytes(prompt_case.prompt.encode()),
                "chat_template_token_count": int(item["prompt_ids"].shape[1]),
                "bf16_generated_token_ids": item["generated_token_ids"],
                "matrix_policy_disagreements": matrix_disagreements[prompt_case.case_id],
                "attribution": attribute_prompt(records, item["generated_token_ids"]),
                "subsets": records,
            }
        )

    declared_result = next(
        record for record in prompt_results if record["case_id"] == "declared_cobalt_lantern"
    )
    declared_all_four = next(
        record
        for record in declared_result["subsets"]
        if set(record["w4_operators"]) == set(CAUSAL_OPERATORS)
    )
    checks = {
        "empty_subset_exact_bf16_logit_match": empty_exact,
        "all_nine_prompts_evaluated": len(prompt_results) == 9,
        "all_16_subsets_evaluated_per_prompt": all(
            len(record["subsets"]) == 16 for record in prompt_results
        ),
        "declared_all_four_tie_uses_greedy_argmax_token_1": (
            declared_all_four["steps"][0]["candidate_top_one_tied"]
            and declared_all_four["steps"][0]["candidate_token_id"] == 1
        ),
        "matrix_remains_zero_eligible": matrix["selection"]["eligible_policy_ids"] == [],
        "no_policy_selected": matrix["selection"]["selected_policy_id"] is None,
    }
    status = (
        "PASS_EXACT_FOUR_OPERATOR_TEACHER_FORCED_CAUSALITY_V2_RECORDED"
        if all(checks.values())
        else "FAIL_CAUSALITY_PROBE_CHECK"
    )
    result = {
        "schema_version": 1,
        "classification": "qwen_instruct_option_b_four_operator_teacher_forced_causality_v2",
        "status": status,
        "environment": environment,
        "model": source_identity,
        "predeclared_contract": {
            "path": str(args.contract.resolve().relative_to(ROOT)),
            "sha256": sha256_file(args.contract.resolve()),
            "contract_sha256": contract_envelope["contract_sha256"],
        },
        "quantization_scope": {
            "weight_bits": 4,
            "signed_integer_range": [-8, 7],
            "w4_rule": "one_absmax_over_7_scale_per_output_channel",
            "w4_operator_universe": list(CAUSAL_OPERATORS),
            "all_other_weights": "BF16",
            "activations_and_non_linear_operators": "BF16",
            "grouped_weight_policy_executed": False,
        },
        "method": {
            "teacher_forcing": "each candidate sees the exact frozen BF16 generated-token prefix at every measured step",
            "greedy_tie_rule": "torch.argmax over vocabulary logits; exact ties select the lowest token index",
            "subset_semantics": "operators in w4_operators use per-output-channel W4; the other members of the frozen four-operator universe are restored exactly to BF16",
            "subset_count_per_prompt": 16,
            "prompt_count": 9,
            "model_forward_count_excluding_reference": 144,
            "acceptance_boundary": "causal attribution only; no policy eligibility, full W4A8 oracle, demo, RTL, PPA, or Stage 2 conclusion",
        },
        "checks": checks,
        "prompts": prompt_results,
        "artifacts": {
            "source_contract": file_record(
                ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
            ),
            "frozen_discrimination_matrix": file_record(MATRIX),
            "prompt_and_policy_source": file_record(
                PROMPT_SOURCE
            ),
            "test_source": file_record(Path(__file__)),
        },
        "scope_guards": {
            "accelerator_executed": False,
            "candidate_frozen": False,
            "demo_retargeted": False,
            "frozen_v1_artifacts_mutated": False,
            "grouped_policy_selected": False,
            "network_access_performed": False,
            "ppa_executed": False,
            "response_gate_changed": False,
            "rtl_mutated": False,
            "stage2_entered": False,
            "source_contract_status": source_contract["status"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(result))
    print(
        "ACE2_FOUR_OPERATOR_TEACHER_FORCED_CAUSALITY "
        f"status={status} prompts={len(prompt_results)} subsets={len(subsets)} "
        f"output={output.relative_to(ROOT)}",
        flush=True,
    )
    return 0 if status.startswith("PASS_") else 4


if __name__ == "__main__":
    raise SystemExit(main())
