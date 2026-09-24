#!/usr/bin/env python3
"""Discriminate the recovering Option-B W4 weight policies on full sequences."""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from localize_qwen_instruct_w4_weight_error import quantize_w4_into
from qwen_instruct_option_b import (
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    TERMINATION_TOKEN_IDS,
    canonical_bytes,
    file_record,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from qwen_instruct_response_gate import evaluate as evaluate_response_gate
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM


CAUSAL_OPERATORS = (
    "model.layers.2.mlp.down_proj",
    "model.layers.2.mlp.gate_proj",
    "model.layers.23.mlp.down_proj",
    "model.layers.23.mlp.gate_proj",
)
LOCALIZATION_RESULT = (
    ROOT
    / "build/option-b-weight-localization-20260806/"
    "declared_nonce_w4_weight_localization_exact.json"
)
BOUNDARY_RESULT = (
    ROOT
    / "build/option-b-weight-localization-20260806/"
    "independent_prompt_boundary_separation.json"
)
V1_IDENTITY = IMAGE_DIR / "option_b_identity_manifest.json"
V1_SUMS = IMAGE_DIR / "OPTION_B_SHA256SUMS"


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class WeightPolicy:
    policy_id: str
    operator: str
    input_group_lanes: int


PROMPTS = (
    PromptCase(
        "declared_cobalt_lantern",
        "Reply with exactly these two words: cobalt lantern",
        "cobalt lantern",
    ),
    PromptCase(
        "independent_amber_compass",
        "Answer with exactly these two words and no punctuation: amber compass",
        "amber compass",
    ),
    PromptCase(
        "independent_silver_heron",
        "Answer with exactly these two words and no punctuation: silver heron",
        "silver heron",
    ),
    PromptCase(
        "independent_quiet_river",
        "Answer with exactly these two words and no punctuation: quiet river",
        "quiet river",
    ),
    PromptCase(
        "independent_blue_jay",
        "Answer with exactly these two words and no punctuation: blue jay",
        "blue jay",
    ),
    PromptCase(
        "independent_green_heron",
        "Answer with exactly these two words and no punctuation: green heron",
        "green heron",
    ),
    PromptCase(
        "independent_open_book",
        "Answer with exactly these two words and no punctuation: open book",
        "open book",
    ),
    PromptCase(
        "independent_calm_lake",
        "Answer with exactly these two words and no punctuation: calm lake",
        "calm lake",
    ),
    PromptCase(
        "independent_cedar_falcon",
        "Answer with exactly these two words and no punctuation: cedar falcon",
        "cedar falcon",
    ),
)


POLICIES = (
    WeightPolicy("layer2_gate_per_128", "model.layers.2.mlp.gate_proj", 128),
    WeightPolicy("layer2_gate_per_64", "model.layers.2.mlp.gate_proj", 64),
    WeightPolicy("layer2_gate_per_32", "model.layers.2.mlp.gate_proj", 32),
    WeightPolicy("layer23_down_per_64", "model.layers.23.mlp.down_proj", 64),
    WeightPolicy("layer23_down_per_32", "model.layers.23.mlp.down_proj", 32),
    WeightPolicy("layer23_gate_per_64", "model.layers.23.mlp.gate_proj", 64),
    WeightPolicy("layer23_gate_per_32", "model.layers.23.mlp.gate_proj", 32),
)


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


def verify_v1_identities() -> list[dict[str, Any]]:
    require(V1_SUMS.is_file(), "Option-B v1 checksum manifest is missing")
    records: list[dict[str, Any]] = []
    for line in V1_SUMS.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = IMAGE_DIR / name
        require(path.is_file(), f"Option-B v1 artifact is missing: {name}")
        require(sha256_file(path) == digest, f"Option-B v1 artifact differs: {name}")
        records.append(file_record(path))
    require(len(records) == 9, f"expected nine Option-B v1 identities, found {len(records)}")
    return records


def chat_input(tokenizer: Any, prompt: str, system_prompt: str) -> Tensor:
    value = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(value, Tensor) and value.ndim == 2, "tokenization shape differs")
    return value


def logit_comparison(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    left = reference.to(torch.float64)
    right = candidate.to(torch.float64)
    difference = right - left
    denominator = torch.linalg.vector_norm(left)
    top = torch.topk(right, k=2)
    return {
        "candidate_token_id": int(right.argmax()),
        "reference_token_id": int(left.argmax()),
        "candidate_top_one_margin": float(top.values[0] - top.values[1]),
        "relative_l2_error": (
            None
            if float(denominator) == 0.0
            else float(torch.linalg.vector_norm(difference) / denominator)
        ),
        "maximum_absolute_error": float(difference.abs().amax()),
    }


def generate_reference(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    expected: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prefix = input_ids
    generated: list[int] = []
    logits_by_step: list[Tensor] = []
    steps: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index in range(max_new_tokens):
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1].detach().cpu()
            token = int(logits.argmax())
            top = torch.topk(logits.to(torch.float64), k=2)
            generated.append(token)
            logits_by_step.append(logits)
            steps.append(
                {
                    "index": index,
                    "token_id": token,
                    "top_one_margin": float(top.values[0] - top.values[1]),
                }
            )
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat(
                [prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1
            )
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    gate = evaluate_response_gate(
        {"decoded_text": decoded, "generated_token_ids": generated}, expected
    )
    return {
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "response_gate": gate,
        "steps": steps,
        "_logits": logits_by_step,
    }


def generate_candidate(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    expected: str,
    reference: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    prefix = input_ids
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    aligned = True
    with torch.inference_mode():
        for index in range(max_new_tokens):
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1].detach().cpu()
            token = int(logits.argmax())
            generated.append(token)
            comparison = None
            if aligned and index < len(reference["_logits"]):
                comparison = logit_comparison(reference["_logits"][index], logits)
            steps.append(
                {
                    "index": index,
                    "token_id": token,
                    "prefix_aligned_with_bf16": aligned,
                    "aligned_logit_comparison": comparison,
                }
            )
            reference_ids = reference["generated_token_ids"]
            aligned = aligned and index < len(reference_ids) and token == reference_ids[index]
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat(
                [prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1
            )
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    gate = evaluate_response_gate(
        {"decoded_text": decoded, "generated_token_ids": generated}, expected
    )
    reference_ids = reference["generated_token_ids"]
    common_prefix = 0
    for left, right in zip(generated, reference_ids):
        if left != right:
            break
        common_prefix += 1
    return {
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "response_gate": gate,
        "exact_bf16_sequence_match": generated == reference_ids,
        "common_bf16_prefix_tokens": common_prefix,
        "steps": steps,
    }


def weight_error(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    left = reference.detach().to(torch.float64)
    right = candidate.detach().to(torch.float64)
    difference = right - left
    denominator = torch.linalg.vector_norm(left)
    return {
        "relative_l2_error": float(torch.linalg.vector_norm(difference) / denominator),
        "maximum_absolute_error": float(difference.abs().amax()),
    }


def metadata_cost(module: nn.Linear, group_lanes: int) -> dict[str, Any]:
    require(module.in_features % group_lanes == 0, "policy group size does not divide input width")
    groups = module.in_features // group_lanes
    baseline_scales = module.out_features
    grouped_scales = module.out_features * groups
    return {
        "input_features": module.in_features,
        "output_features": module.out_features,
        "input_group_lanes": group_lanes,
        "groups_per_output": groups,
        "w4_weight_values": module.in_features * module.out_features,
        "w4_weight_payload_bits_unchanged": True,
        "multiply_accumulate_count_ratio": 1.0,
        "baseline_weight_scale_count": baseline_scales,
        "candidate_weight_scale_count": grouped_scales,
        "additional_weight_scale_count": grouped_scales - baseline_scales,
        "candidate_scale32_metadata_bytes": grouped_scales * 4,
        "additional_scale32_metadata_bytes": (grouped_scales - baseline_scales) * 4,
        "group_partial_dot_products": grouped_scales,
        "additional_group_combines": module.out_features * (groups - 1),
    }


def aggregate_policy(record: dict[str, Any]) -> dict[str, Any]:
    prompt_records = record["prompts"]
    aligned_errors = [
        float(step["aligned_logit_comparison"]["relative_l2_error"])
        for prompt in prompt_records
        for step in prompt["steps"]
        if step["aligned_logit_comparison"] is not None
        and step["aligned_logit_comparison"]["relative_l2_error"] is not None
    ]
    return {
        "prompt_count": len(prompt_records),
        "response_gate_pass_count": sum(
            prompt["response_gate"]["status"] == "PASS" for prompt in prompt_records
        ),
        "response_gate_outcome_match_count": sum(
            prompt["response_gate"]["status"]
            == prompt["bf16_response_gate_status"]
            for prompt in prompt_records
        ),
        "exact_bf16_sequence_match_count": sum(
            prompt["exact_bf16_sequence_match"] for prompt in prompt_records
        ),
        "total_common_bf16_prefix_tokens": sum(
            prompt["common_bf16_prefix_tokens"] for prompt in prompt_records
        ),
        "aligned_logit_step_count": len(aligned_errors),
        "worst_aligned_relative_l2_error": max(aligned_errors),
        "mean_aligned_relative_l2_error": sum(aligned_errors) / len(aligned_errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(2 <= args.max_new_tokens <= 32, "max-new-tokens must be 2..32")

    environment = require_project_python()
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    v1_records = verify_v1_identities()
    for path in (LOCALIZATION_RESULT, BOUNDARY_RESULT, V1_IDENTITY):
        require(path.is_file(), f"required source-bound artifact is missing: {path}")

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
        require(
            isinstance(reference_modules.get(name), nn.Linear)
            and isinstance(candidate_modules.get(name), nn.Linear),
            f"causal linear is missing: {name}",
        )

    reference_prompts: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Tensor] = {}
    for prompt_case in PROMPTS:
        input_ids = chat_input(tokenizer, prompt_case.prompt, args.system_prompt)
        inputs[prompt_case.case_id] = input_ids
        reference_prompts[prompt_case.case_id] = generate_reference(
            reference_model,
            tokenizer,
            input_ids,
            prompt_case.expected,
            args.max_new_tokens,
        )
    reference_gate_passed = all(
        record["response_gate"]["status"] == "PASS"
        for record in reference_prompts.values()
    )
    independent_reference_gate_pass_count = sum(
        reference_prompts[prompt_case.case_id]["response_gate"]["status"] == "PASS"
        for prompt_case in PROMPTS[1:]
    )

    public_references = []
    for prompt_case in PROMPTS:
        record = dict(reference_prompts[prompt_case.case_id])
        record.pop("_logits")
        record["case_id"] = prompt_case.case_id
        record["prompt_utf8_bytes"] = len(prompt_case.prompt.encode())
        record["prompt_sha256"] = sha256_bytes(prompt_case.prompt.encode())
        record["expected_utf8_bytes"] = len(prompt_case.expected.encode())
        record["expected_sha256"] = sha256_bytes(prompt_case.expected.encode())
        public_references.append(record)
    if args.reference_only:
        screen_status = (
            "PASS_BF16_PROMPT_SUITE_SCREEN"
            if independent_reference_gate_pass_count >= 3
            else "BLOCKED_BF16_PROMPT_SUITE_INSUFFICIENT"
        )
        screen = {
            "schema_version": 1,
            "classification": "qwen_instruct_option_b_bf16_prompt_suite_screen",
            "status": screen_status,
            "environment": environment,
            "model": source_identity,
            "prompt_suite": {
                "case_count": len(PROMPTS),
                "declared_nonce_included": True,
                "independent_natural_language_case_count": len(PROMPTS) - 1,
                "independent_response_gate_pass_count": independent_reference_gate_pass_count,
                "required_independent_response_gate_pass_count": 3,
                "system_utf8_bytes": len(args.system_prompt.encode()),
                "system_sha256": sha256_bytes(args.system_prompt.encode()),
                "max_new_tokens": args.max_new_tokens,
                "prompt_plaintext_bound_by_discriminator_source": True,
            },
            "bf16_reference": {
                "all_response_gates_pass": reference_gate_passed,
                "prompts": public_references,
            },
            "artifacts": {
                "source_contract": file_record(
                    ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
                ),
                "discriminator_source": file_record(Path(__file__)),
                "response_gate_source": file_record(
                    ROOT / "tools/qwen_instruct_response_gate.py"
                ),
                "response_gate_contract": file_record(
                    IMAGE_DIR / "response_gate_contract.json"
                ),
                "v1_identity_manifest": file_record(V1_IDENTITY),
                "v1_checksum_manifest": file_record(V1_SUMS),
                "v1_verified_artifacts": v1_records,
            },
            "scope_guards": {
                "frozen_v1_artifacts_mutated": False,
                "candidate_policies_executed": False,
                "accelerator_executed": False,
                "demo_retargeted": False,
                "network_access_performed": False,
                "ppa_executed": False,
                "source_contract_status": source_contract["status"],
            },
        }
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_bytes(screen))
        print(
            "ACE2_QWEN_INSTRUCT_BF16_PROMPT_SCREEN "
            f"status={screen_status} independent_gate_passes="
            f"{independent_reference_gate_pass_count} output={output.relative_to(ROOT)}"
        )
        return 0 if screen_status == "PASS_BF16_PROMPT_SUITE_SCREEN" else 4

    policies: list[dict[str, Any]] = []
    for policy in POLICIES:
        for name in CAUSAL_OPERATORS:
            quantize_w4_into(
                candidate_modules[name].weight,
                reference_modules[name].weight,
            )
        quantize_w4_into(
            candidate_modules[policy.operator].weight,
            reference_modules[policy.operator].weight,
            policy.input_group_lanes,
        )
        selected_module = candidate_modules[policy.operator]
        reference_module = reference_modules[policy.operator]
        require(
            isinstance(selected_module, nn.Linear)
            and isinstance(reference_module, nn.Linear),
            "selected policy operator is not linear",
        )
        prompt_records: list[dict[str, Any]] = []
        for prompt_case in PROMPTS:
            candidate = generate_candidate(
                candidate_model,
                tokenizer,
                inputs[prompt_case.case_id],
                prompt_case.expected,
                reference_prompts[prompt_case.case_id],
                args.max_new_tokens,
            )
            candidate["case_id"] = prompt_case.case_id
            candidate["prompt_utf8_bytes"] = len(prompt_case.prompt.encode())
            candidate["prompt_sha256"] = sha256_bytes(prompt_case.prompt.encode())
            candidate["expected_utf8_bytes"] = len(prompt_case.expected.encode())
            candidate["expected_sha256"] = sha256_bytes(prompt_case.expected.encode())
            candidate["bf16_response_gate_status"] = reference_prompts[
                prompt_case.case_id
            ]["response_gate"]["status"]
            prompt_records.append(candidate)
        record = {
            "policy_id": policy.policy_id,
            "operator": policy.operator,
            "input_group_lanes": policy.input_group_lanes,
            "weight_error": weight_error(
                reference_module.weight, selected_module.weight
            ),
            "metadata_and_compute_cost": metadata_cost(
                selected_module, policy.input_group_lanes
            ),
            "prompts": prompt_records,
        }
        record["aggregate"] = aggregate_policy(record)
        policies.append(record)

    eligible = [
        record
        for record in policies
        if record["aggregate"]["response_gate_outcome_match_count"] == len(PROMPTS)
        and record["aggregate"]["exact_bf16_sequence_match_count"] == len(PROMPTS)
    ]
    ranking = sorted(
        eligible,
        key=lambda record: (
            record["aggregate"]["worst_aligned_relative_l2_error"],
            record["aggregate"]["mean_aligned_relative_l2_error"],
            record["metadata_and_compute_cost"]["additional_scale32_metadata_bytes"],
        ),
    )
    winner = ranking[0] if ranking else None
    winner_key = (
        None
        if winner is None
        else (
            winner["aggregate"]["worst_aligned_relative_l2_error"],
            winner["aggregate"]["mean_aligned_relative_l2_error"],
            winner["metadata_and_compute_cost"]["additional_scale32_metadata_bytes"],
        )
    )
    unique_winner = winner is not None and sum(
        (
            record["aggregate"]["worst_aligned_relative_l2_error"],
            record["aggregate"]["mean_aligned_relative_l2_error"],
            record["metadata_and_compute_cost"]["additional_scale32_metadata_bytes"],
        )
        == winner_key
        for record in ranking
    ) == 1
    status = (
        "BLOCKED_BF16_PROMPT_SUITE_INSUFFICIENT"
        if independent_reference_gate_pass_count < 3
        else (
            "PASS_UNIQUE_V2_WEIGHT_POLICY_SELECTED"
            if unique_winner
            else (
                "BLOCKED_NO_ELIGIBLE_WEIGHT_POLICY"
                if not eligible
                else "BLOCKED_WEIGHT_POLICY_SELECTION_NOT_UNIQUE"
            )
        )
    )

    result = {
        "schema_version": 1,
        "classification": "qwen_instruct_option_b_w4_weight_policy_full_sequence_discrimination",
        "status": status,
        "environment": environment,
        "model": source_identity,
        "prompt_suite": {
            "case_count": len(PROMPTS),
            "declared_nonce_included": True,
            "independent_natural_language_case_count": len(PROMPTS) - 1,
            "system_utf8_bytes": len(args.system_prompt.encode()),
            "system_sha256": sha256_bytes(args.system_prompt.encode()),
            "max_new_tokens": args.max_new_tokens,
            "prompt_plaintext_bound_by_discriminator_source": True,
        },
        "quantization_scope": {
            "weight_bits": 4,
            "signed_integer_range": [-8, 7],
            "baseline_causal_operator_rule": "one_absmax_over_7_scale_per_output_channel",
            "candidate_rule": "one_absmax_over_7_scale_per_output_and_input_lane_group at exactly one recovering causal operator",
            "causal_operators": list(CAUSAL_OPERATORS),
            "all_other_weights": "BF16",
            "activations_and_non_linear_operators": "BF16",
            "scope_matches_exact_causal_localizer": True,
        },
        "bf16_reference": {
            "all_response_gates_pass": reference_gate_passed,
            "independent_response_gate_pass_count": independent_reference_gate_pass_count,
            "prompts": public_references,
        },
        "candidate_policies": policies,
        "selection": {
            "predeclared_functional_eligibility": "every complete generated token sequence exactly matches BF16 and every response-gate outcome matches BF16; the independent BF16 suite must contain at least three passing gates",
            "predeclared_ranking": [
                "minimum worst aligned full-vocabulary relative-L2 logit error",
                "minimum mean aligned full-vocabulary relative-L2 logit error",
                "minimum additional 32-bit weight-scale metadata bytes",
            ],
            "recovering_policy_count_tested": len(POLICIES),
            "eligible_policy_ids": [record["policy_id"] for record in eligible],
            "ranked_eligible_policy_ids": [record["policy_id"] for record in ranking],
            "selected_policy_id": None if winner is None else winner["policy_id"],
            "unique_selected_policy": unique_winner,
        },
        "artifacts": {
            "source_contract": file_record(
                ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
            ),
            "discriminator_source": file_record(Path(__file__)),
            "causal_localizer_source": file_record(
                ROOT / "tools/localize_qwen_instruct_w4_weight_error.py"
            ),
            "causal_localization_result": file_record(LOCALIZATION_RESULT),
            "boundary_diagnostic_result": file_record(BOUNDARY_RESULT),
            "response_gate_source": file_record(
                ROOT / "tools/qwen_instruct_response_gate.py"
            ),
            "response_gate_contract": file_record(
                IMAGE_DIR / "response_gate_contract.json"
            ),
            "v1_identity_manifest": file_record(V1_IDENTITY),
            "v1_checksum_manifest": file_record(V1_SUMS),
            "v1_verified_artifacts": v1_records,
        },
        "scope_guards": {
            "frozen_v1_artifacts_mutated": False,
            "accelerator_executed": False,
            "demo_retargeted": False,
            "network_access_performed": False,
            "ppa_executed": False,
            "source_contract_status": source_contract["status"],
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(result))
    print(
        "ACE2_QWEN_INSTRUCT_W4_POLICY_DISCRIMINATION "
        f"status={status} selected={result['selection']['selected_policy_id']} "
        f"eligible={len(eligible)} output={output.relative_to(ROOT)}"
    )
    return 0 if status == "PASS_UNIQUE_V2_WEIGHT_POLICY_SELECTED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
