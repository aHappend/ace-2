#!/usr/bin/env python3
"""Run one predeclared four-operator Scale32 W4 Stage-1 repair attempt."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import ace2_full_model_fixed_point as fixed_point
from discriminate_qwen_instruct_w4_weight_policies import PROMPTS, chat_input
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM
from run_all_transformer_per32_stage1 import (
    GroupedScale32W4A8Linear,
    aggregate_prompts,
    generate_candidate,
    generate_reference,
    log_message,
    packed_w4,
    prompt_binding,
    require_project_python,
    scale32_bytes,
    scale32_self_test,
    sequence_disagreements,
    utc_now,
    write_json,
)


CANDIDATE_ID = "causal_four_per_32_v1"
MISSION_ID = "stage1-causal-four-w4-per32-repair-v1"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
CAUSALITY_PATH = ROOT / "build/option-b-four-operator-teacher-forced-causality-v2-20260806/results.json"
ROW_LANE_PATH = ROOT / "build/option-b-layer2-down-row-lane-causality-20260806/attempt-0001/results.json"
CONSUMED_RESULT_PATH = ROOT / "build/stage1-global-w4-per32-repair-v1/attempt-0001/results.json"
OUTPUT_DIR = ROOT / "build/stage1-causal-four-w4-per32-repair-v1"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
GROUP_LANES = 32
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16

TARGETS = (
    ("model.layers.2.mlp.gate_proj", 896, 4864),
    ("model.layers.2.mlp.down_proj", 4864, 896),
    ("model.layers.23.mlp.gate_proj", 896, 4864),
    ("model.layers.23.mlp.down_proj", 4864, 896),
)


def target_specs() -> list[dict[str, Any]]:
    records = []
    for module, input_features, output_features in TARGETS:
        groups = input_features // GROUP_LANES
        records.append(
            {
                "name": f"{module}.weight",
                "module": module,
                "shape": [output_features, input_features],
                "input_group_lanes": GROUP_LANES,
                "groups_per_output": groups,
                "scale32_record_count": output_features * groups,
            }
        )
    return records


def candidate_cost(specs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_scale_count = sum(record["shape"][0] for record in specs)
    candidate_scale_count = sum(record["scale32_record_count"] for record in specs)
    w4_values = sum(math.prod(record["shape"]) for record in specs)
    return {
        "changed_transformer_linear_tensor_count": len(specs),
        "unchanged_transformer_linear_tensor_count": 168 - len(specs),
        "input_group_lanes": GROUP_LANES,
        "baseline_weight_scale_count_in_changed_scope": baseline_scale_count,
        "candidate_weight_scale32_record_count": candidate_scale_count,
        "additional_weight_scale32_record_count": candidate_scale_count
        - baseline_scale_count,
        "baseline_weight_scale_metadata_bytes_in_changed_scope": baseline_scale_count * 4,
        "candidate_weight_scale32_metadata_bytes": candidate_scale_count * 4,
        "additional_weight_scale32_metadata_bytes": (candidate_scale_count - baseline_scale_count)
        * 4,
        "group_partial_dot_products_per_model_token_position": candidate_scale_count,
        "group_combines_per_model_token_position": candidate_scale_count
        - baseline_scale_count,
        "changed_scope_w4_weight_values": w4_values,
        "changed_scope_packed_w4_payload_bytes": (w4_values + 1) // 2,
        "full_model_w4_payload_size_unchanged": True,
        "multiply_accumulate_count_ratio": 1.0,
    }


def contract_artifacts() -> dict[str, Any]:
    paths = {
        "matrix": MATRIX_PATH,
        "four_operator_causality": CAUSALITY_PATH,
        "layer2_down_row_lane_taxonomy": ROW_LANE_PATH,
        "consumed_global_attempt_result": CONSUMED_RESULT_PATH,
        "source_contract": ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json",
        "derived_scales": CALIBRATION_DIR / "derived_scales.json",
        "full_model_image": IMAGE_DIR / "full_model_image.bin",
        "image_contract": IMAGE_DIR / "image_contract_v2.json",
        "image_manifest": IMAGE_DIR / "manifest.json",
        "option_b_identity_manifest": IMAGE_DIR / "option_b_identity_manifest.json",
        "oracle_contract": IMAGE_DIR / "oracle_contract.json",
        "response_gate_contract": IMAGE_DIR / "response_gate_contract.json",
        "runner_source": Path(__file__),
        "grouped_arithmetic_source": ROOT / "tools/run_all_transformer_per32_stage1.py",
        "fixed_point_source": ROOT / "tools/ace2_full_model_fixed_point.py",
        "prompt_and_matrix_evaluator_source": ROOT
        / "tools/discriminate_qwen_instruct_w4_weight_policies.py",
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
    }
    return {name: file_record(path) for name, path in paths.items()}


def prepare_contract() -> dict[str, Any]:
    require_project_python()
    verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "immutable matrix hash differs")
    matrix = load_json(MATRIX_PATH)
    prompts = prompt_binding(matrix)
    specs = target_specs()
    cost = candidate_cost(specs)
    consumed = load_json(CONSUMED_RESULT_PATH)
    require(
        consumed.get("candidate_id") == "all_transformer_per_32_v1"
        and consumed.get("eligibility", {}).get("eligible") is False,
        "consumed global attempt binding differs",
    )
    expected_targets = [record[0] for record in TARGETS]
    causality_text = CAUSALITY_PATH.read_text(encoding="utf-8")
    require(
        all(name in causality_text for name in expected_targets),
        "four-operator causality artifact no longer names the exact target set",
    )
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "hypothesis": (
            "The exact four-operator causal frontier is interacting, while the layer-2 "
            "down-projection row/lane error is distributed. Applying per-row, per-32-input-lane "
            "Scale32 W4 to all and only those four operators can reduce the interaction without "
            "the destructive global perturbation observed when all 168 transformer projections changed."
        ),
        "distinctness": {
            "consumed_candidate_id": "all_transformer_per_32_v1",
            "consumed_scope_tensor_count": 168,
            "new_scope_tensor_count": 4,
            "scope_identical": False,
            "replay": False,
            "consumed_attempt_result": file_record(CONSUMED_RESULT_PATH),
        },
        "matrix": {
            "path": MATRIX_PATH.relative_to(ROOT).as_posix(),
            "sha256": MATRIX_SHA256,
            "case_count": 9,
            "full_generated_sequences_bound": True,
        },
        "scope": {
            "exact_changed_tensor_count": 4,
            "tensors": specs,
            "all_and_only_exact_causal_frontier": True,
            "other_164_transformer_linears": (
                "unchanged authenticated per-output-row W4A8 oracle policy"
            ),
            "lm_head_policy": "unchanged authenticated per-output-row W4A8 oracle policy",
            "embedding_policy": "unchanged authenticated oracle policy",
            "nonlinear_and_residual_policy": "unchanged authenticated oracle policy",
        },
        "weight_policy": {
            "weight_bits": 4,
            "signed_integer_range": [-8, 7],
            "group_axis": "input_feature",
            "input_group_lanes": GROUP_LANES,
            "scale_scope": (
                "one normalized Scale32 record per output row and contiguous 32-input-lane "
                "group at exactly the four named tensors"
            ),
            "scale_selection": (
                "ceil-encode max(abs(BF16 group))/7 as the smallest representable Scale32 "
                "value not below the ratio; all-zero groups use the canonical all-zero record"
            ),
            "rounding": "torch.round round-to-nearest ties-to-even",
            "saturation": (
                "clamp signed W4 to [-8,7]; final projection output clamp signed A8 to [-128,127]"
            ),
            "packing": (
                "row-major output row then input lane; two's-complement signed-int4; even flat "
                "element in low nibble and odd flat element in high nibble"
            ),
            "group_accumulation": (
                "signed A8xW4 dot per 32-lane group; decode each Scale32 product scale, align "
                "and sum group contributions plus unchanged bias, divide by unchanged output "
                "scale, round ties-to-even once, then saturate"
            ),
            "hardware_realizable": True,
        },
        "activation_policy": {
            "contract": fixed_point.ACTIVE_ROPE_MECHANISM,
            "dynamic_scale32_semantics": "unchanged",
            "input_A8_quantization": "unchanged authenticated oracle scales and rounding",
            "projection_output_A8_scales": "unchanged authenticated oracle scales",
            "rope_and_attention_scale32": "unchanged authenticated oracle behavior",
            "activation_recalibration_permitted": False,
        },
        "generation_and_evaluator": {
            "system_prompt_utf8_bytes": len(DEFAULT_SYSTEM.encode()),
            "system_prompt_sha256": sha256_bytes(DEFAULT_SYSTEM.encode()),
            "chat_template_sha256": source_identity["chat_template_sha256"],
            "greedy_argmax": True,
            "sampling": False,
            "use_cache": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "termination_token_ids": [151643, 151645],
            "greedy_tie_rule": "torch.argmax; exact ties choose the lowest token id",
            "response_gate": "frozen qwen-instruct-option-b-response-gate-v1",
            "aligned_logit_error": (
                "full-vocabulary candidate versus recomputed BF16 logits while incoming "
                "generated prefix remains BF16-aligned, including the first mismatch step"
            ),
            "prompts": prompts,
        },
        "eligibility": {
            "rule": (
                "all nine complete generated token sequences exactly equal BF16 and all nine "
                "response-gate outcomes equal BF16"
            ),
            "score_policy": "binary only; 9/9 exact plus 9/9 gate-outcome agreement",
            "aggregate_improvement_is_insufficient": True,
            "local_arithmetic_agreement_is_insufficient": True,
            "selected_policy_id_before_execution": None,
            "fresh_reviewer_required": True,
        },
        "execution": {
            "exact_authorized_candidate_attempts": 1,
            "attempt_id": "attempt-0001",
            "torch_num_threads": TORCH_THREADS,
            "network_access_permitted": False,
            "alternate_policy_permitted": False,
            "tuning_ranking_bisection_sweep_permitted": False,
            "rtl_demo_ppa_stage2_mutation_permitted": False,
        },
        "model": source_identity,
        "image_binding": {
            "image_id": "build-qwen2.5-0.5b-instruct-w4a8-image-v1",
            "dynamic_activation_semantics_source": (
                "authenticated Option-B W4A8 oracle and derived scale table"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
        },
        "static_cost": cost,
        "artifacts": contract_artifacts(),
        "no_outcome_assumption": (
            "No token, response-gate, eligibility, demo, RTL, synthesis, timing, PPA, or FPGA "
            "outcome is assumed before the single execution."
        ),
    }
    wrapper = {"contract": contract, "contract_sha256": sha256_bytes(canonical_bytes(contract))}
    write_json(CONTRACT_PATH, wrapper)
    return wrapper


def load_verified_contract(*, require_unconsumed: bool) -> dict[str, Any]:
    require_project_python()
    require(CONTRACT_PATH.is_file(), "predeclared contract is missing")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "contract wrapper differs")
    contract = wrapper["contract"]
    require(
        sha256_bytes(canonical_bytes(contract)) == wrapper["contract_sha256"],
        "contract digest differs",
    )
    require(contract.get("candidate_id") == CANDIDATE_ID, "candidate id differs")
    require(contract.get("matrix", {}).get("sha256") == MATRIX_SHA256, "matrix binding differs")
    require(contract.get("scope", {}).get("tensors") == target_specs(), "target scope differs")
    require(
        contract.get("weight_policy", {}).get("input_group_lanes") == GROUP_LANES,
        "group size differs",
    )
    for record in contract.get("artifacts", {}).values():
        path = ROOT / record["path"]
        require(path.is_file(), f"contract artifact is missing: {record['path']}")
        require(path.stat().st_size == record["bytes"], f"artifact size differs: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"artifact hash differs: {record['path']}")
    verify_source_contract()
    verify_source_snapshot()
    verify_versions()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "immutable matrix hash differs")
    matrix = load_json(MATRIX_PATH)
    require(
        prompt_binding(matrix) == contract["generation_and_evaluator"]["prompts"],
        "prompt binding differs",
    )
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    if require_unconsumed:
        require(not attempts, f"exactly-once authorization is consumed: {[p.name for p in attempts]}")
    probe = OUTPUT_DIR / ".preflight-write-probe"
    probe.write_bytes(b"preflight\n")
    require(probe.read_bytes() == b"preflight\n", "output path readback differs")
    probe.unlink()
    scale32_self_test()
    return wrapper


def replace_candidate_linears(
    model: nn.Module, ranges: dict[str, fixed_point.CalibrationRange]
) -> list[str]:
    replacements = [
        (name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    require(len(replacements) == 169, f"expected 169 linears, found {len(replacements)}")
    target_names = [record[0] for record in TARGETS]
    observed_targets: list[str] = []
    for name, module in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        use_per_head_output_scales, emit_scale32 = fixed_point.rope_linear_contract(
            name, fixed_point.ACTIVE_ROPE_MECHANISM
        )
        frozen_output_head_absmax = fixed_point.qk_residual_projection_output_absmax(
            name, fixed_point.ACTIVE_ROPE_MECHANISM
        )
        common = {
            "input_is_quantized": False,
            "output_head_absmax": (
                frozen_output_head_absmax
                if frozen_output_head_absmax is not None
                else ranges[name].output_head_absmax
                if use_per_head_output_scales
                else None
            ),
            "output_head_size": fixed_point.ROPE_HEAD_DIM if use_per_head_output_scales else None,
            "scale_cap": None,
            "use_percentile_scale": False,
            "scale32_output": emit_scale32,
        }
        if name in target_names:
            replacement: nn.Module = GroupedScale32W4A8Linear(
                module, ranges[name], module_name=name, **common
            )
            observed_targets.append(name)
        else:
            replacement = fixed_point.W4A8Linear(module, ranges[name], **common)
        setattr(parent, child_name, replacement)
    require(observed_targets == target_names, "live target order/scope differs")
    return observed_targets


def reconstruct_candidate_model(scales: dict[str, Any]) -> tuple[nn.Module, list[str]]:
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    ranges: dict[str, fixed_point.CalibrationRange] = {}
    for name, metadata in scales["linears"].items():
        head_scales = metadata.get("output_head_scales") or []
        ranges[name] = fixed_point.CalibrationRange(
            input_absmax=float(metadata["input_absmax"]),
            output_absmax=float(metadata["output_absmax"]),
            output_head_absmax=[float(value) * 127.0 for value in head_scales] or None,
        )
    operators = {
        name: fixed_point.ObservedRange(absmax=float(metadata["absmax"]))
        for name, metadata in scales["operators"].items()
    }
    scope = replace_candidate_linears(model, ranges)
    fixed_point.replace_fixed_operators(
        model,
        operators,
        rope_diagnostic_mechanism=fixed_point.ACTIVE_ROPE_MECHANISM,
    )
    grouped = [
        name
        for name, module in model.named_modules()
        if isinstance(module, GroupedScale32W4A8Linear)
    ]
    require(grouped == scope, "post-replacement grouped scope differs")
    return model, scope


def candidate_tensor_manifest(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    modules = dict(model.named_modules())
    records: list[dict[str, Any]] = []
    aggregate_packed = hashlib.sha256()
    aggregate_scale32 = hashlib.sha256()
    for spec in target_specs():
        module = modules.get(spec["module"])
        require(
            isinstance(module, GroupedScale32W4A8Linear),
            f"grouped target is missing: {spec['module']}",
        )
        qweight_raw = (
            module.qweight.detach().cpu().contiguous().numpy().astype(np.int8, copy=False).tobytes()
        )
        packed_raw = packed_w4(module)
        metadata_raw = scale32_bytes(module)
        aggregate_packed.update(spec["name"].encode() + b"\0" + packed_raw)
        aggregate_scale32.update(spec["name"].encode() + b"\0" + metadata_raw)
        records.append(
            {
                **spec,
                "qweight_s8_container_sha256": hashlib.sha256(qweight_raw).hexdigest(),
                "packed_w4_bytes": len(packed_raw),
                "packed_w4_sha256": hashlib.sha256(packed_raw).hexdigest(),
                "weight_scale32_metadata_bytes": len(metadata_raw),
                "weight_scale32_sha256": hashlib.sha256(metadata_raw).hexdigest(),
                "weight_error": {
                    "relative_l2_error": module.grouped_record.weight_error_relative_l2,
                    "maximum_absolute_error": module.grouped_record.weight_error_maximum_absolute,
                },
            }
        )
    return records, {
        "packed_w4_scope_sha256": aggregate_packed.hexdigest(),
        "weight_scale32_scope_sha256": aggregate_scale32.hexdigest(),
        "tensor_manifest_sha256": sha256_bytes(canonical_bytes(records)),
    }


def execute_once() -> int:
    wrapper = load_verified_contract(require_unconsumed=True)
    ATTEMPT_DIR.mkdir(parents=True, exist_ok=False)
    run_log_path = ATTEMPT_DIR / "run.log"
    started = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "attempt_id": "attempt-0001",
        "started_at_utc": utc_now(),
        "contract_sha256": wrapper["contract_sha256"],
        "matrix_sha256": MATRIX_SHA256,
        "authorization_consumed": True,
    }
    write_json(ATTEMPT_DIR / "execution_started.json", started)
    started_monotonic = time.monotonic()
    with run_log_path.open("w", encoding="utf-8") as run_log:
        try:
            log_message(run_log, f"official_attempt_started candidate={CANDIDATE_ID}")
            torch.set_num_threads(TORCH_THREADS)
            torch.set_num_interop_threads(1)
            source_contract = verify_source_contract()
            source_identity = verify_source_snapshot()
            versions = verify_versions()
            matrix = load_json(MATRIX_PATH)
            bound_prompts = prompt_binding(matrix)
            prompt_specs = {record["case_id"]: record for record in bound_prompts}
            tokenizer = AutoTokenizer.from_pretrained(
                SNAPSHOT, local_files_only=True, trust_remote_code=False
            )
            inputs = {
                prompt_case.case_id: chat_input(tokenizer, prompt_case.prompt, DEFAULT_SYSTEM)
                for prompt_case in PROMPTS
            }

            log_message(run_log, "bf16_reference_recomputation_started prompts=9")
            reference_model = AutoModelForCausalLM.from_pretrained(
                SNAPSHOT,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval()
            references: dict[str, dict[str, Any]] = {}
            for prompt_case in PROMPTS:
                record = generate_reference(
                    reference_model,
                    tokenizer,
                    inputs[prompt_case.case_id],
                    prompt_case.expected,
                )
                expected = prompt_specs[prompt_case.case_id]
                require(
                    record["generated_token_ids"] == expected["bf16_generated_token_ids"],
                    f"BF16 token binding differs: {prompt_case.case_id}",
                )
                require(
                    record["response_gate"]["status"] == expected["bf16_response_gate_status"],
                    f"BF16 response gate differs: {prompt_case.case_id}",
                )
                references[prompt_case.case_id] = record
                log_message(
                    run_log,
                    f"bf16_prompt_complete case={prompt_case.case_id} "
                    f"tokens={len(record['generated_token_ids'])} gate={record['response_gate']['status']}",
                )
            del reference_model
            gc.collect()

            log_message(run_log, "candidate_model_construction_started tensors=4 group_lanes=32")
            scales = load_json(CALIBRATION_DIR / "derived_scales.json")
            candidate_model, scope = reconstruct_candidate_model(scales)
            require(len(scope) == 4, "candidate scope count differs")
            tensor_manifest, candidate_hashes = candidate_tensor_manifest(candidate_model)
            static_cost = candidate_cost(target_specs())
            require(
                sum(record["weight_scale32_metadata_bytes"] for record in tensor_manifest)
                == static_cost["candidate_weight_scale32_metadata_bytes"],
                "candidate metadata byte count differs",
            )
            require(
                sum(record["packed_w4_bytes"] for record in tensor_manifest)
                == static_cost["changed_scope_packed_w4_payload_bytes"],
                "candidate packed W4 byte count differs",
            )
            log_message(
                run_log,
                "candidate_model_construction_complete "
                f"packed_w4_sha256={candidate_hashes['packed_w4_scope_sha256']} "
                f"scale32_sha256={candidate_hashes['weight_scale32_scope_sha256']}",
            )

            prompt_results: list[dict[str, Any]] = []
            for prompt_case in PROMPTS:
                reference = references[prompt_case.case_id]
                candidate = generate_candidate(
                    candidate_model,
                    tokenizer,
                    inputs[prompt_case.case_id],
                    prompt_case.expected,
                    reference,
                    static_cost,
                )
                disagreements = sequence_disagreements(
                    reference["generated_token_ids"], candidate["generated_token_ids"]
                )
                gate_match = (
                    candidate["response_gate"]["status"] == reference["response_gate"]["status"]
                )
                public_reference = dict(reference)
                public_reference.pop("_logits")
                prompt_results.append(
                    {
                        "case_id": prompt_case.case_id,
                        "prompt_sha256": prompt_specs[prompt_case.case_id]["prompt_sha256"],
                        "expected_sha256": prompt_specs[prompt_case.case_id]["expected_sha256"],
                        "bf16": public_reference,
                        "candidate": candidate,
                        "token_disagreements": disagreements,
                        "response_gate_outcome_match": gate_match,
                    }
                )
                log_message(
                    run_log,
                    f"candidate_prompt_complete case={prompt_case.case_id} "
                    f"tokens={len(candidate['generated_token_ids'])} "
                    f"exact={candidate['exact_bf16_sequence_match']} "
                    f"gate={candidate['response_gate']['status']} gate_match={gate_match}",
                )

            aggregate = aggregate_prompts(prompt_results)
            disagreement_set = [
                {
                    "case_id": prompt["case_id"],
                    "token_disagreements": prompt["token_disagreements"],
                    "bf16_response_gate_status": prompt["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": prompt["candidate"]["response_gate"][
                        "status"
                    ],
                    "response_gate_outcome_match": prompt["response_gate_outcome_match"],
                }
                for prompt in prompt_results
                if prompt["token_disagreements"] or not prompt["response_gate_outcome_match"]
            ]
            eligible = (
                aggregate["exact_bf16_sequence_match_count"] == 9
                and aggregate["response_gate_outcome_match_count"] == 9
            )
            status = "PASS_ELIGIBLE_FOR_FRESH_REVIEW" if eligible else "BLOCKED_EXACT_BF16_DISAGREEMENT"
            result = {
                "schema_version": 1,
                "classification": "qwen_instruct_stage1_causal_four_per32_single_candidate",
                "status": status,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "attempt_id": "attempt-0001",
                "contract": {
                    "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(CONTRACT_PATH),
                    "contract_sha256": wrapper["contract_sha256"],
                },
                "matrix": file_record(MATRIX_PATH),
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                },
                "model": source_identity,
                "source_contract_status": source_contract["status"],
                "policy": wrapper["contract"]["weight_policy"],
                "activation_policy": wrapper["contract"]["activation_policy"],
                "static_cost": static_cost,
                "tensor_manifest": tensor_manifest,
                "candidate_hashes": candidate_hashes,
                "prompts": prompt_results,
                "aggregate": aggregate,
                "eligibility": {
                    "rule": wrapper["contract"]["eligibility"]["rule"],
                    "eligible": eligible,
                    "eligible_candidate_id": CANDIDATE_ID if eligible else None,
                    "selected_policy_id": None,
                    "fresh_reviewer_required": True,
                },
                "disagreement_set": disagreement_set,
                "disposition": "READY_FOR_FRESH_REVIEW" if eligible else "BLOCKED",
                "scope_guards": {
                    "candidate_attempt_count": 1,
                    "consumed_global_candidate_replayed": False,
                    "alternate_policy_executed": False,
                    "ranking_or_selection_executed": False,
                    "activation_policy_changed": False,
                    "rtl_mutated": False,
                    "demo_retargeted": False,
                    "ppa_executed": False,
                    "stage2_entered": False,
                    "network_access_performed": False,
                },
                "timing": {
                    "completed_at_utc": utc_now(),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                },
            }
            results_path = ATTEMPT_DIR / "results.json"
            write_json(results_path, result)
            log_message(
                run_log,
                f"official_attempt_complete status={status} "
                f"exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 "
                f"gate_matches={aggregate['response_gate_outcome_match_count']}/9",
            )
            sums_paths = [
                CONTRACT_PATH,
                ATTEMPT_DIR / "execution_started.json",
                run_log_path,
                results_path,
            ]
            sums = "".join(
                f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n"
                for path in sums_paths
            )
            (ATTEMPT_DIR / "SHA256SUMS").write_text(sums, encoding="utf-8")
            if eligible:
                reviewer_input = {
                    "schema_version": 1,
                    "mission_id": MISSION_ID,
                    "candidate_id": CANDIDATE_ID,
                    "status": "READY_FOR_FRESH_REVIEW",
                    "contract": file_record(CONTRACT_PATH),
                    "results": file_record(results_path),
                    "checksums": file_record(ATTEMPT_DIR / "SHA256SUMS"),
                    "independent_acceptance_required": True,
                }
                write_json(ATTEMPT_DIR / "fresh_reviewer_input.json", reviewer_input)
            return 0 if eligible else 4
        except Exception as exc:
            failure = {
                "schema_version": 1,
                "classification": "EVALUATOR_EXECUTION_FAILURE",
                "status": "BLOCKED_EVALUATOR_EXECUTION_FAILURE",
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "attempt_id": "attempt-0001",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "numerical_correctness_conclusion": None,
                "selected_policy_id": None,
                "authorization_consumed": True,
                "failed_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
            }
            write_json(ATTEMPT_DIR / "failure.json", failure)
            log_message(
                run_log,
                f"official_attempt_failed classification=EVALUATOR_EXECUTION_FAILURE "
                f"error_type={type(exc).__name__} error={exc}",
            )
            return 5


def verify_result() -> dict[str, Any]:
    wrapper = load_verified_contract(require_unconsumed=False)
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    require([path.name for path in attempts] == ["attempt-0001"], "attempt set differs")
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    results_path = ATTEMPT_DIR / "results.json"
    require(sums_path.is_file() and results_path.is_file(), "attempt result is incomplete")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"checksummed artifact is missing: {relative}")
        require(sha256_file(path) == digest, f"checksummed artifact differs: {relative}")
    result = load_json(results_path)
    require(result.get("candidate_id") == CANDIDATE_ID, "result candidate differs")
    require(
        result.get("contract", {}).get("contract_sha256") == wrapper["contract_sha256"],
        "result contract binding differs",
    )
    require(result.get("matrix", {}).get("sha256") == MATRIX_SHA256, "result matrix differs")
    require(len(result.get("prompts", [])) == 9, "result prompt count differs")
    recomputed = aggregate_prompts(result["prompts"])
    require(recomputed == result.get("aggregate"), "result aggregate differs")
    eligible = (
        recomputed["exact_bf16_sequence_match_count"] == 9
        and recomputed["response_gate_outcome_match_count"] == 9
    )
    require(result.get("eligibility", {}).get("eligible") is eligible, "eligibility differs")
    require(result.get("eligibility", {}).get("selected_policy_id") is None, "policy was selected")
    require(result.get("scope_guards", {}).get("candidate_attempt_count") == 1, "attempt count differs")
    return {
        "candidate_id": CANDIDATE_ID,
        "attempt_count": 1,
        "eligible": eligible,
        "exact_sequence_matches": recomputed["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": recomputed["response_gate_outcome_match_count"],
        "disagreement_case_ids": [entry["case_id"] for entry in result["disagreement_set"]],
        "results_sha256": sha256_file(results_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare-contract", "verify-contract", "execute", "verify-result")
    )
    args = parser.parse_args()
    if args.command == "prepare-contract":
        require(not list(OUTPUT_DIR.glob("attempt-*")), "cannot rewrite contract after execution")
        wrapper = prepare_contract()
        print(
            "ACE2_STAGE1_CAUSAL_FOUR_CONTRACT_PREPARED "
            f"candidate={CANDIDATE_ID} tensors=4 matrix={MATRIX_SHA256} "
            f"contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "verify-contract":
        wrapper = load_verified_contract(require_unconsumed=True)
        print(
            "ACE2_STAGE1_CAUSAL_FOUR_PREFLIGHT_PASS "
            f"candidate={CANDIDATE_ID} tensors=4 attempts=0 "
            f"contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "execute":
        return execute_once()
    summary = verify_result()
    print("ACE2_STAGE1_CAUSAL_FOUR_RESULT_VERIFIED " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
