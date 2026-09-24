#!/usr/bin/env python3
"""Causally localize Option-B W4 weight-only first-token sensitivity."""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_instruct_option_b import (
    ROOT,
    SNAPSHOT,
    canonical_bytes,
    file_record,
    require,
    sha256_bytes,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM


EMBEDDING_OPERATOR = "model.embed_tokens"
LM_HEAD_OPERATOR = "lm_head"
ROW_CHUNK = 1024


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


def quantized_w4_chunk(weight: Tensor, group_lanes: int | None) -> Tensor:
    value = weight.to(torch.float64)
    if group_lanes is None:
        scale = value.abs().amax(dim=1, keepdim=True) / 7.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quantized = torch.round(value / scale).clamp(-8, 7)
        return (quantized * scale).to(weight.dtype)
    require(
        value.shape[1] % group_lanes == 0,
        f"input width {value.shape[1]} is not divisible by group size {group_lanes}",
    )
    grouped = value.reshape(value.shape[0], -1, group_lanes)
    scale = grouped.abs().amax(dim=2, keepdim=True) / 7.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(grouped / scale).clamp(-8, 7)
    return (quantized * scale).reshape_as(value).to(weight.dtype)


def quantize_w4_into(target: Tensor, source: Tensor, group_lanes: int | None = None) -> None:
    require(target.shape == source.shape, "quantization source and target shapes differ")
    with torch.no_grad():
        for start in range(0, source.shape[0], ROW_CHUNK):
            stop = min(start + ROW_CHUNK, source.shape[0])
            target[start:stop].copy_(quantized_w4_chunk(source[start:stop], group_lanes))


def head_logits(hidden: Tensor, weight: Tensor, group_lanes: int | None) -> Tensor:
    pieces: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, weight.shape[0], ROW_CHUNK):
            stop = min(start + ROW_CHUNK, weight.shape[0])
            quantized = quantized_w4_chunk(weight[start:stop], group_lanes)
            pieces.append(F.linear(hidden, quantized).to(torch.float64).cpu())
    return torch.cat(pieces)


def logit_stats(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    reference = reference.to(torch.float64)
    candidate = candidate.to(torch.float64)
    difference = candidate - reference
    denominator = torch.linalg.vector_norm(reference)
    top = torch.topk(candidate, k=2)
    return {
        "first_token_id": int(candidate.argmax()),
        "top_two_token_ids": [int(value) for value in top.indices],
        "top_two_logits": [float(value) for value in top.values],
        "top_one_margin": float(top.values[0] - top.values[1]),
        "relative_l2_error": (
            None
            if float(denominator) == 0.0
            else float(torch.linalg.vector_norm(difference) / denominator)
        ),
    }


def linear_override(reference: nn.Linear) -> Callable[[nn.Module, tuple[Any, ...], Any], Tensor]:
    def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> Tensor:
        return F.linear(inputs[0], reference.weight, reference.bias)

    return hook


def embedding_override(reference: nn.Embedding) -> Callable[[nn.Module, tuple[Any, ...], Any], Tensor]:
    def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> Tensor:
        return F.embedding(
            inputs[0],
            reference.weight,
            reference.padding_idx,
            reference.max_norm,
            reference.norm_type,
            reference.scale_grad_by_freq,
            reference.sparse,
        )

    return hook


def evaluate_operator_set(
    candidate: nn.Module,
    reference_modules: dict[str, nn.Module],
    candidate_modules: dict[str, nn.Module],
    operators: list[str],
    w4_operators: set[str],
    input_ids: Tensor,
) -> Tensor:
    hooks: list[Any] = []
    for name in operators:
        if name in w4_operators:
            continue
        reference = reference_modules[name]
        module = candidate_modules[name]
        if isinstance(reference, nn.Linear) and isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_override(reference)))
        elif isinstance(reference, nn.Embedding) and isinstance(module, nn.Embedding):
            hooks.append(module.register_forward_hook(embedding_override(reference)))
        else:
            raise TypeError(f"unsupported causal operator: {name}")
    try:
        with torch.inference_mode():
            return candidate(input_ids=input_ids, use_cache=False).logits[0, -1].detach().cpu().to(torch.float64)
    finally:
        for hook in hooks:
            hook.remove()


def intervention_record(
    name: str,
    logits: Tensor,
    reference_logits: Tensor,
    failure_token: int,
    operators: set[str],
) -> dict[str, Any]:
    stats = logit_stats(reference_logits, logits)
    return {
        "name": name,
        "w4_operator_count": len(operators),
        "w4_operators": sorted(operators),
        "failure_token_reproduced": stats["first_token_id"] == failure_token,
        **stats,
    }


def partitions(values: list[str], count: int) -> list[list[str]]:
    size = math.ceil(len(values) / count)
    return [values[start : start + size] for start in range(0, len(values), size)]


def minimize_failure_set(
    initial: list[str],
    reproduces_failure: Callable[[set[str]], bool],
    max_evaluations: int,
) -> tuple[list[str], bool]:
    current = list(initial)
    partitions_count = 2
    evaluations = 0
    while len(current) >= 2 and evaluations < max_evaluations:
        chunks = partitions(current, partitions_count)
        reduced = False
        for chunk in chunks:
            evaluations += 1
            if reproduces_failure(set(chunk)):
                current = chunk
                partitions_count = max(partitions_count - 1, 2)
                reduced = True
                break
            if evaluations >= max_evaluations:
                break
        if reduced or evaluations >= max_evaluations:
            continue
        for chunk in chunks:
            chunk_set = set(chunk)
            complement = [name for name in current if name not in chunk_set]
            evaluations += 1
            if complement and reproduces_failure(set(complement)):
                current = complement
                partitions_count = max(partitions_count - 1, 2)
                reduced = True
                break
            if evaluations >= max_evaluations:
                break
        if reduced or evaluations >= max_evaluations:
            continue
        if partitions_count >= len(current):
            break
        partitions_count = min(len(current), partitions_count * 2)

    changed = True
    while changed and len(current) >= 2 and evaluations < max_evaluations:
        changed = False
        for name in list(current):
            reduced = [candidate for candidate in current if candidate != name]
            evaluations += 1
            if reproduces_failure(set(reduced)):
                current = reduced
                changed = True
                break
            if evaluations >= max_evaluations:
                break
    budget_exhausted = evaluations >= max_evaluations
    return current, budget_exhausted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    parser.add_argument(
        "--candidate-operator",
        action="append",
        default=[],
        help="restrict exact minimization to a previously reduced causal frontier",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    environment = require_project_python()
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()

    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(input_ids, Tensor) and input_ids.ndim == 2, "tokenization differs")

    reference_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    with torch.inference_mode():
        reference_hidden = reference_model.model(
            input_ids=input_ids, use_cache=False
        ).last_hidden_state[0, -1]
        reference_logits = F.linear(
            reference_hidden, reference_model.lm_head.weight
        ).detach().cpu().to(torch.float64)

    per_channel_head_logits = head_logits(
        reference_hidden, reference_model.lm_head.weight, None
    )
    group_sizes = [128, 64, 32]
    grouped_head_logits = {
        size: head_logits(reference_hidden, reference_model.lm_head.weight, size)
        for size in group_sizes
    }

    candidate_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    require(candidate_model.config.tie_word_embeddings, "pinned model no longer ties embeddings")
    require(
        candidate_model.lm_head.weight.data_ptr()
        == candidate_model.model.embed_tokens.weight.data_ptr(),
        "pinned tied parameter alias is missing",
    )
    candidate_model.lm_head.weight = nn.Parameter(
        candidate_model.lm_head.weight.detach().clone(), requires_grad=False
    )
    require(
        candidate_model.lm_head.weight.data_ptr()
        != candidate_model.model.embed_tokens.weight.data_ptr(),
        "causal intervention failed to separate embedding and output-head uses",
    )

    reference_modules = dict(reference_model.named_modules())
    candidate_modules = dict(candidate_model.named_modules())
    linear_operators = [
        name for name, module in candidate_modules.items() if isinstance(module, nn.Linear)
    ]
    require(len(linear_operators) == 169, f"expected 169 linears, found {len(linear_operators)}")
    transformer_operators = [name for name in linear_operators if name != LM_HEAD_OPERATOR]
    require(len(transformer_operators) == 168, "transformer projection count differs")
    operators = sorted(transformer_operators + [EMBEDDING_OPERATOR, LM_HEAD_OPERATOR])

    for name in linear_operators:
        quantize_w4_into(
            candidate_modules[name].weight,
            reference_modules[name].weight,
        )
    quantize_w4_into(
        candidate_model.model.embed_tokens.weight,
        reference_model.model.embed_tokens.weight,
    )

    all_operator_set = set(operators)
    all_w4_logits = evaluate_operator_set(
        candidate_model,
        reference_modules,
        candidate_modules,
        operators,
        all_operator_set,
        input_ids,
    )
    failure_token = int(all_w4_logits.argmax())
    reference_token = int(reference_logits.argmax())
    require(reference_token != failure_token, "declared nonce does not reproduce W4 weight-only divergence")

    evaluation_cache: dict[frozenset[str], Tensor] = {
        frozenset(): reference_logits,
        frozenset({LM_HEAD_OPERATOR}): per_channel_head_logits,
        frozenset(all_operator_set): all_w4_logits,
    }
    evaluation_trace: list[dict[str, Any]] = []

    def evaluate_cached(operator_set: set[str]) -> Tensor:
        key = frozenset(operator_set)
        if key not in evaluation_cache:
            evaluation_cache[key] = evaluate_operator_set(
                candidate_model,
                reference_modules,
                candidate_modules,
                operators,
                operator_set,
                input_ids,
            )
        logits = evaluation_cache[key]
        evaluation_trace.append(
            {
                "w4_operator_count": len(operator_set),
                "first_token_id": int(logits.argmax()),
                "failure_token_reproduced": int(logits.argmax()) == failure_token,
                "w4_operators": sorted(operator_set),
            }
        )
        return logits

    def reproduces_failure(operator_set: set[str]) -> bool:
        return int(evaluate_cached(operator_set).argmax()) == failure_token

    transformer_set = set(transformer_operators)
    intervention_sets = [
        ("lm_head_only", {LM_HEAD_OPERATOR}),
        ("embedding_only", {EMBEDDING_OPERATOR}),
        ("tied_weight_uses_only", {EMBEDDING_OPERATOR, LM_HEAD_OPERATOR}),
        ("transformer_projections_only", transformer_set),
        ("all_except_embedding", all_operator_set - {EMBEDDING_OPERATOR}),
        ("all_except_lm_head", all_operator_set - {LM_HEAD_OPERATOR}),
        ("all_w4_weight_operators", all_operator_set),
    ]
    interventions: list[dict[str, Any]] = []
    for name, operator_set in intervention_sets:
        logits = evaluate_cached(operator_set)
        interventions.append(
            intervention_record(
                name, logits, reference_logits, failure_token, operator_set
            )
        )

    reproducing_singletons = [
        record["w4_operators"][0]
        for record in interventions
        if record["w4_operator_count"] == 1 and record["failure_token_reproduced"]
    ]
    if args.candidate_operator:
        unknown = sorted(set(args.candidate_operator) - set(operators))
        require(not unknown, f"unknown candidate operators: {unknown}")
        minimization_seed = [
            name for name in operators if name in set(args.candidate_operator)
        ]
        require(
            reproduces_failure(set(minimization_seed)),
            "provided candidate frontier does not reproduce the all-W4 failure token",
        )
    else:
        minimization_seed = list(operators)
        for removable in [EMBEDDING_OPERATOR, LM_HEAD_OPERATOR]:
            reduced = [name for name in minimization_seed if name != removable]
            if reproduces_failure(set(reduced)):
                minimization_seed = reduced
    minimized_set, minimization_budget_exhausted = minimize_failure_set(
        minimization_seed,
        reproduces_failure,
        max_evaluations=128,
    )
    exhaustive_frontier = list(minimized_set)
    exact_minimum_within_frontier: list[str] | None = None
    exhaustive_subset_tests = 0
    for subset_size in range(1, len(exhaustive_frontier) + 1):
        for subset in combinations(exhaustive_frontier, subset_size):
            exhaustive_subset_tests += 1
            if reproduces_failure(set(subset)):
                exact_minimum_within_frontier = list(subset)
                break
        if exact_minimum_within_frontier is not None:
            break
    require(
        exact_minimum_within_frontier is not None,
        "reduced causal frontier stopped reproducing the failure token",
    )
    minimized_set = exact_minimum_within_frontier
    removal_tests = []
    for name in minimized_set:
        reduced = set(minimized_set) - {name}
        logits = evaluate_cached(reduced)
        removal_tests.append(
            {
                "removed_operator": name,
                "remaining_first_token_id": int(logits.argmax()),
                "failure_token_reproduced": int(logits.argmax()) == failure_token,
            }
        )
    one_minimal = bool(minimized_set) and all(
        not record["failure_token_reproduced"] for record in removal_tests
    )
    localization_status = (
        "PASS_SINGLETON_CAUSAL_OPERATOR_LOCALIZED"
        if len(minimized_set) == 1 and one_minimal
        else (
            "PASS_EXACT_MINIMUM_WITHIN_CAUSAL_FRONTIER_LOCALIZED"
            if one_minimal
            else "BLOCKED_CAUSAL_MINIMIZATION_INCOMPLETE"
        )
    )

    granularity_interventions: list[dict[str, Any]] = []
    causal_operator_set = set(minimized_set)
    for name in minimized_set:
        candidate_module = candidate_modules[name]
        reference_module = reference_modules[name]
        require(
            isinstance(candidate_module, nn.Linear)
            and isinstance(reference_module, nn.Linear),
            f"granularity probe requires a linear operator: {name}",
        )
        for group_lanes in [128, 64, 32]:
            if reference_module.weight.shape[1] % group_lanes != 0:
                continue
            quantize_w4_into(
                candidate_module.weight,
                reference_module.weight,
                group_lanes,
            )
            logits = evaluate_operator_set(
                candidate_model,
                reference_modules,
                candidate_modules,
                operators,
                causal_operator_set,
                input_ids,
            )
            granularity_interventions.append(
                {
                    "operator": name,
                    "changed_operator_count": 1,
                    "input_group_lanes": group_lanes,
                    "reference_token_recovered": int(logits.argmax()) == reference_token,
                    **logit_stats(reference_logits, logits),
                }
            )
        quantize_w4_into(candidate_module.weight, reference_module.weight)

    jointly_grouped_operators = []
    for name in minimized_set:
        candidate_module = candidate_modules[name]
        reference_module = reference_modules[name]
        if reference_module.weight.shape[1] % 128 == 0:
            quantize_w4_into(candidate_module.weight, reference_module.weight, 128)
            jointly_grouped_operators.append(name)
    jointly_grouped_logits = evaluate_operator_set(
        candidate_model,
        reference_modules,
        candidate_modules,
        operators,
        causal_operator_set,
        input_ids,
    )
    for name in jointly_grouped_operators:
        quantize_w4_into(
            candidate_modules[name].weight,
            reference_modules[name].weight,
        )

    failure_row = failure_token
    reference_row = reference_token
    only_failure_row = reference_logits.clone()
    only_failure_row[failure_row] = per_channel_head_logits[failure_row]
    only_reference_row = reference_logits.clone()
    only_reference_row[reference_row] = per_channel_head_logits[reference_row]
    only_transition_rows = reference_logits.clone()
    only_transition_rows[failure_row] = per_channel_head_logits[failure_row]
    only_transition_rows[reference_row] = per_channel_head_logits[reference_row]
    restore_failure_row = per_channel_head_logits.clone()
    restore_failure_row[failure_row] = reference_logits[failure_row]
    restore_reference_row = per_channel_head_logits.clone()
    restore_reference_row[reference_row] = reference_logits[reference_row]

    granularity_probes = {
        "per_output_channel": logit_stats(reference_logits, per_channel_head_logits),
        **{
            f"per_{size}_input_lane_group": logit_stats(reference_logits, logits)
            for size, logits in grouped_head_logits.items()
        },
    }
    recovering_group_sizes = [
        size
        for size, logits in grouped_head_logits.items()
        if int(logits.argmax()) == reference_token
    ]

    result = {
        "schema_version": 1,
        "classification": "qwen_instruct_option_b_w4_weight_only_causal_localization",
        "status": localization_status,
        "environment": environment,
        "model": source_identity,
        "prompt": {
            "chat_template_token_count": int(input_ids.shape[1]),
            "system_sha256": sha256_bytes(args.system_prompt.encode()),
            "user_sha256": sha256_bytes(args.prompt.encode()),
        },
        "quantization_under_test": {
            "weight_bits": 4,
            "signed_integer_range": [-8, 7],
            "frozen_rule": "one_absmax_over_7_scale_per_output_channel",
            "activation_and_non_linear_operator_dtype": "bfloat16",
            "transformer_linear_operator_count": len(transformer_operators),
            "execution_operator_count_including_embedding_and_lm_head": len(operators),
        },
        "target_transition": {
            "bf16_first_token_id": reference_token,
            "all_w4_weight_first_token_id": failure_token,
        },
        "tied_parameter_confound": {
            "tie_word_embeddings": True,
            "prior_in_place_lm_head_quantization_also_changed_model_embed_tokens": True,
            "causal_intervention_untied_numerically_identical_uses_before_quantization": True,
        },
        "interventions": interventions,
        "localization": {
            "exact_minimum_within_causal_frontier": minimized_set,
            "minimality_definition": "all non-empty subsets of the reduced causal frontier were tested in increasing cardinality; this is the first subset reproducing the all-W4 first token while every other weight operator uses BF16 weights",
            "minimality_scope": "exact within the causally reduced frontier; no global uniqueness claim outside that frontier",
            "empty_set_first_token_id": reference_token,
            "singleton_failure_token_reproduced": bool(reproducing_singletons),
            "one_minimal": one_minimal,
            "minimization_budget_exhausted": minimization_budget_exhausted,
            "requested_candidate_frontier": args.candidate_operator,
            "reduced_frontier_before_exhaustive_search": exhaustive_frontier,
            "exhaustive_subset_tests": exhaustive_subset_tests,
            "member_removal_tests": removal_tests,
            "full_transformer_projection_set_tested_separately": True,
            "evaluation_trace": evaluation_trace,
        },
        "lm_head_row_causality": {
            "bf16_winner_row": reference_row,
            "w4_winner_row": failure_row,
            "only_w4_winner_row_quantized_first_token_id": int(only_failure_row.argmax()),
            "only_bf16_winner_row_quantized_first_token_id": int(only_reference_row.argmax()),
            "only_transition_rows_quantized_first_token_id": int(only_transition_rows.argmax()),
            "w4_head_with_w4_winner_row_restored_first_token_id": int(restore_failure_row.argmax()),
            "w4_head_with_bf16_winner_row_restored_first_token_id": int(restore_reference_row.argmax()),
            "bf16_logits": {
                str(reference_row): float(reference_logits[reference_row]),
                str(failure_row): float(reference_logits[failure_row]),
            },
            "per_output_channel_w4_logits": {
                str(reference_row): float(per_channel_head_logits[reference_row]),
                str(failure_row): float(per_channel_head_logits[failure_row]),
            },
        },
        "lm_head_granularity_probe": {
            "classification": "diagnostic_only_no_frozen_artifact_or_weight_format_changed",
            "probes": granularity_probes,
            "reference_token_recovered_group_sizes": recovering_group_sizes,
            "largest_recovering_input_group_lanes": (
                None if not recovering_group_sizes else max(recovering_group_sizes)
            ),
        },
        "causal_set_granularity_probe": {
            "classification": "diagnostic_only_no_frozen_artifact_or_weight_format_changed",
            "single_operator_interventions": granularity_interventions,
            "all_causal_operators_per_128_input_lane_group": {
                "operators": jointly_grouped_operators,
                "reference_token_recovered": int(jointly_grouped_logits.argmax())
                == reference_token,
                **logit_stats(reference_logits, jointly_grouped_logits),
            },
        },
        "artifacts": {
            "source_contract": file_record(
                ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
            ),
            "localizer_source": file_record(Path(__file__)),
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
        "ACE2_QWEN_INSTRUCT_W4_WEIGHT_LOCALIZATION "
        f"status={localization_status} bf16={reference_token} all_w4={failure_token} "
        f"causal={','.join(minimized_set) if minimized_set else 'unresolved'} "
        f"output={output.relative_to(ROOT)}"
    )
    return 0 if one_minimal else 4


if __name__ == "__main__":
    raise SystemExit(main())
