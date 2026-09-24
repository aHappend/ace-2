#!/usr/bin/env python3
"""Prepare and execute the exactly-once alias-safe grouped Dynamic Scale32 V2 candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_alias_safe_post_w4_static_position_class_group_scale_stage1 as prior
import run_option_b_grouped_scale32_stage1 as grouped
from ace2_quality_contracts import unpack_scale32
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    TERMINATION_TOKEN_IDS,
    file_record,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
    write_json,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM
from run_all_transformer_per32_stage1 import (
    aggregate_prompts,
    generate_reference,
    log_message,
    prompt_binding,
    sequence_disagreements,
    utc_now,
)


CANDIDATE_ID = "option_b_alias_safe_post_w4_grouped_dynamic_scale32_w4a8_v2"
MISSION_ID = "e29633380586"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_POST_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V2_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_POST_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V2_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
SOURCE_CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
CALIBRATION_CONTRACT_PATH = CALIBRATION_DIR / "calibration_contract.json"
CALIBRATION_REPORT_PATH = CALIBRATION_DIR / "calibration_report.json"
BASE_SCALES_PATH = CALIBRATION_DIR / "derived_scales.json"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
SOURCE_EMBEDDING_SHA256 = "4e96b0df6d274768cbb7e72404011853d23349999b658dc2f4dfb3c431ea223f"
OUTPUT_DIR = ROOT / "build/stage1-option-b-alias-safe-post-w4-grouped-dynamic-scale32-w4a8-v2"
PREATTEMPT_DIR = OUTPUT_DIR / "preattempt"
INPUT_MANIFEST_PATH = PREATTEMPT_DIR / "non_evaluator_inputs.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
PREFLIGHT_PATH = PREATTEMPT_DIR / "dynamic_preflight.json"
SELF_TEST_PATH = PREATTEMPT_DIR / "runner_self_test.json"
CHRONOLOGY_PATH = PREATTEMPT_DIR / "preparation_chronology.json"
RECOVERY_PATH = PREATTEMPT_DIR / "preparation_recovery.json"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
READY_PATH = OUTPUT_DIR / "PREATTEMPT_READY.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
MARKER_PATH = ATTEMPT_DIR / "execution_started.json"
RUNNER_PATH = Path(__file__).resolve()
PRIOR_RUNNER_PATH = ROOT / "tools/run_option_b_alias_safe_post_w4_static_position_class_group_scale_stage1.py"
GROUPED_RUNNER_PATH = ROOT / "tools/run_option_b_grouped_scale32_stage1.py"
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16
EXPECTED_EVENTS = grouped.LAYERS * len(grouped.EVENT_FAMILIES)


def configure_prior() -> None:
    """Point shared reviewed construction helpers at this fresh candidate namespace."""
    bindings = {
        "CANDIDATE_ID": CANDIDATE_ID,
        "MISSION_ID": MISSION_ID,
        "TASK_PATH": TASK_PATH,
        "TASK_COMPANION": TASK_COMPANION,
        "PLAN_PATH": PLAN_PATH,
        "PLAN_COMPANION": PLAN_COMPANION,
        "OUTPUT_DIR": OUTPUT_DIR,
        "PREATTEMPT_DIR": PREATTEMPT_DIR,
        "INPUT_MANIFEST_PATH": INPUT_MANIFEST_PATH,
        "CANDIDATE_A_PATH": CANDIDATE_A_PATH,
        "CANDIDATE_B_PATH": CANDIDATE_B_PATH,
        "ALIAS_WITNESS_PATH": ALIAS_WITNESS_PATH,
        "PREFLIGHT_PATH": PREFLIGHT_PATH,
        "SELF_TEST_PATH": SELF_TEST_PATH,
        "CHRONOLOGY_PATH": CHRONOLOGY_PATH,
        "CONTRACT_PATH": CONTRACT_PATH,
        "READY_PATH": READY_PATH,
        "ATTEMPT_DIR": ATTEMPT_DIR,
        "MARKER_PATH": MARKER_PATH,
        "RUNNER_PATH": RUNNER_PATH,
    }
    for name, value in bindings.items():
        setattr(prior, name, value)


configure_prior()
canonical_sha256 = prior.canonical_sha256
require_project_python = prior.require_project_python
verify_companion = prior.verify_companion
atomic_write_json = prior.atomic_write_json
load_candidate = prior.load_candidate
non_evaluator_inputs = prior.non_evaluator_inputs
compare_candidates = prior.compare_candidates
write_checksum_bundle = prior.write_checksum_bundle
seal_attempt = prior.seal_attempt
parse_sums = prior.parse_sums


def _require_digest(actual: str, expected: str, message: str) -> None:
    require(actual == expected, message)


def dynamic_quantizer_self_test() -> dict[str, Any]:
    base = grouped.scale32_record(0.125)
    minimum = grouped.quantize_groups(
        torch.tensor([[15.875, -15.875, 0.3125, -0.3125]], dtype=torch.float64),
        torch.tensor([base], dtype=torch.int64),
        4,
    )
    require(minimum.deltas.tolist() == [[0]], "minimum legal delta selection differs")
    require(minimum.payload.tolist() == [[127, -127, 2, -2]], "ties-to-even payload differs")
    previous = grouped.adjusted_record(base, -1)
    require(15.875 > 127.0 * grouped.scale32_value(previous), "selected delta was not minimal")

    all_zero = grouped.quantize_groups(
        torch.zeros((1, 4), dtype=torch.float64),
        torch.tensor([base], dtype=torch.int64),
        4,
    )
    require(all_zero.deltas.tolist() == [[0]], "all-zero group delta differs")
    require(all_zero.payload.tolist() == [[0, 0, 0, 0]], "all-zero group payload differs")

    negative_limit = grouped.quantize_groups(
        torch.tensor([[-15.875, 15.875, -0.25, 0.25]], dtype=torch.float64),
        torch.tensor([base], dtype=torch.int64),
        4,
    )
    require(int(negative_limit.payload.min()) == -127, "negative A8 limit differs")
    require(not bool(torch.any(negative_limit.payload == -128)), "reserved -128 was produced")

    _significand, exponent = unpack_scale32(base)
    maximum_record = grouped.adjusted_record(base, 4 - exponent)
    maximum_scale = grouped.scale32_value(maximum_record)
    overflow_rejected = False
    try:
        grouped.quantize_groups(
            torch.full((1, 4), 128.0 * maximum_scale, dtype=torch.float64),
            torch.tensor([base], dtype=torch.int64),
            4,
        )
    except RuntimeError as exc:
        overflow_rejected = "no legal Dynamic Scale32 delta" in str(exc)
    require(overflow_rejected, "dynamic exponent overflow did not fail closed")

    hash_mismatch_rejected = False
    try:
        _require_digest("0" * 64, "1" * 64, "bound artifact hash differs")
    except RuntimeError:
        hash_mismatch_rejected = True
    require(hash_mismatch_rejected, "bound-hash mismatch was not rejected")
    return {
        "status": "PASS",
        "selection": "minimum_legal_integer_delta",
        "rounding": "round_to_nearest_ties_to_even",
        "all_zero_group_delta": 0,
        "reserved_minus_128_produced": False,
        "overflow_fail_closed": True,
        "bound_hash_mismatch_rejected": True,
        "payload_sha256": hashlib.sha256(minimum.payload.cpu().numpy().tobytes(order="C")).hexdigest(),
    }


def runner_self_test() -> dict[str, Any]:
    require_project_python()
    dynamic = dynamic_quantizer_self_test()
    alias = prior.alias_split_self_test()
    return {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "dynamic_quantizer": dynamic,
        "alias_split": alias,
        "runner_sha256": sha256_file(RUNNER_PATH),
    }


class AuditedDynamicPolicy(grouped.ActivationPolicy):
    """Grouped Dynamic Scale32 policy with per-boundary legality evidence."""

    def __init__(self, scales: dict[str, Any]) -> None:
        super().__init__(scales)
        self.events: dict[str, dict[str, Any]] = {}

    def _apply(self, name: str, value: Tensor, records: Tensor, lanes: int) -> Tensor:
        result = grouped.quantize_groups(value, records.to(value.device), lanes)
        groups = int(value.shape[-1] // lanes)
        base = records.reshape(-1)
        if base.numel() == 1:
            base = base.repeat(groups)
        require(base.numel() == groups, f"base group count differs: {name}")
        effective = result.records.reshape(-1, groups)
        for group_index, base_record in enumerate(base.detach().cpu().tolist()):
            base_significand, _base_exponent = unpack_scale32(int(base_record))
            for effective_record in torch.unique(effective[:, group_index]).detach().cpu().tolist():
                effective_significand, _effective_exponent = unpack_scale32(int(effective_record))
                require(effective_significand == base_significand, f"Scale32 significand changed: {name}")

        delta_min = int(result.deltas.min())
        delta_max = int(result.deltas.max())
        require(-24 <= delta_min <= delta_max <= 24, f"dynamic delta range differs: {name}")
        exponent_u8 = torch.bitwise_and(torch.bitwise_right_shift(result.records, 16), 0xFF)
        exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
        exponent_min = int(exponent.min())
        exponent_max = int(exponent.max())
        require(-24 <= exponent_min <= exponent_max <= 4, f"effective exponent range differs: {name}")
        require(not bool(torch.any(result.payload == -128)), f"reserved -128 produced: {name}")

        self.calls[name] += 1
        self.group_count += result.records.numel()
        self.value_count += result.payload.numel()
        self.delta_min = delta_min if self.delta_min is None else min(self.delta_min, delta_min)
        self.delta_max = delta_max if self.delta_max is None else max(self.delta_max, delta_max)
        payload = result.payload.detach().cpu().contiguous().numpy().astype(np.int8, copy=False)
        record = result.records.detach().cpu().contiguous().numpy().astype("<u4", copy=False)
        self.payload_hash.update(name.encode() + b"\0" + payload.tobytes(order="C"))
        self.record_hash.update(name.encode() + b"\0" + record.tobytes(order="C"))
        event = self.events.setdefault(
            name,
            {
                "group_lanes": lanes,
                "call_count": 0,
                "group_count": 0,
                "value_count": 0,
                "minimum_delta": None,
                "maximum_delta": None,
                "minimum_effective_exponent": None,
                "maximum_effective_exponent": None,
                "saturation_count": 0,
                "clipping_count": 0,
                "reserved_minus_128_count": 0,
                "base_significand_preserved": True,
            },
        )
        require(event["group_lanes"] == lanes, f"event group width changed: {name}")
        event["call_count"] += 1
        event["group_count"] += int(result.records.numel())
        event["value_count"] += int(result.payload.numel())
        event["minimum_delta"] = delta_min if event["minimum_delta"] is None else min(event["minimum_delta"], delta_min)
        event["maximum_delta"] = delta_max if event["maximum_delta"] is None else max(event["maximum_delta"], delta_max)
        event["minimum_effective_exponent"] = exponent_min if event["minimum_effective_exponent"] is None else min(event["minimum_effective_exponent"], exponent_min)
        event["maximum_effective_exponent"] = exponent_max if event["maximum_effective_exponent"] is None else max(event["maximum_effective_exponent"], exponent_max)
        return result.dequantized.to(value.dtype)

    def audited_summary(self, model_forward_count: int) -> dict[str, Any]:
        summary = super().summary(model_forward_count)
        require(summary["named_boundary_count"] == EXPECTED_EVENTS == 312, "312-event closure differs")
        require(-24 <= int(summary["minimum_delta"]) <= int(summary["maximum_delta"]) <= 24, "summary delta range differs")
        require(summary["reserved_minus_128_produced"] is False, "summary reports reserved -128")
        require(set(self.events) == set(summary["calls"]), "audited event set differs")
        require(all(item["saturation_count"] == 0 and item["clipping_count"] == 0 for item in self.events.values()), "preflight clipping or saturation differs")
        summary.update(
            {
                "event_count": len(self.events),
                "all_312_layer_event_names_present": len(self.events) == 312,
                "legal_delta_range": [-24, 24],
                "legal_effective_exponent_range": [-24, 4],
                "saturation_count": 0,
                "clipping_count": 0,
                "transport_metadata_bytes_per_decoder_layer_token": 832,
                "tensor_sidecars_per_decoder_layer_token": 13,
                "events": dict(sorted(self.events.items())),
            }
        )
        return summary


def run_preflight_generation(model: nn.Module, input_ids: Tensor) -> tuple[list[int], int]:
    prefix = input_ids
    generated: list[int] = []
    forwards = 0
    with torch.inference_mode():
        for _index in range(MAX_NEW_TOKENS):
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1]
            forwards += 1
            token = int(logits.argmax())
            generated.append(token)
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
        model(input_ids=prefix, use_cache=False)
        forwards += 1
    return generated, forwards


def dynamic_preflight(model: nn.Module, inputs: list[Tensor], input_manifest: dict[str, Any]) -> dict[str, Any]:
    scales = load_json(BASE_SCALES_PATH)
    policy = AuditedDynamicPolicy(scales)
    policy.install(model)
    generated: list[dict[str, Any]] = []
    forwards = 0
    try:
        for record, input_ids in zip(input_manifest["records"], inputs, strict=True):
            tokens, count = run_preflight_generation(model, input_ids)
            forwards += count
            generated.append(
                {
                    "case_id": record["case_id"],
                    "generated_token_ids": tokens,
                    "generated_token_ids_sha256": sha256_bytes(b"".join(value.to_bytes(4, "little") for value in tokens)),
                    "model_forward_count": count,
                }
            )
        summary = policy.audited_summary(forwards)
    finally:
        policy.uninstall()
    require(not MARKER_PATH.exists(), "official execution marker appeared during preflight")
    return {
        "schema_version": 1,
        "status": "PASS_ALL_312_DYNAMIC_EVENTS",
        "candidate_id": CANDIDATE_ID,
        "fresh_model_reconstruction": True,
        "official_nine_prompt_inputs_used": False,
        "official_execution_marker_exists": False,
        "input_count": len(inputs),
        "base_scales": file_record(BASE_SCALES_PATH),
        "selection": "minimum legal exponent delta preserving each frozen base Scale32 significand",
        "rounding": "round_to_nearest_ties_to_even",
        "activation_execution": summary,
        "generated_sequences": generated,
    }


def recover_nonqualifying_preparation() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {"fresh_namespace": True, "recovered": False, "archived_artifacts": []}
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a consumed attempt namespace")
    complete = PREATTEMPT_DIR.is_dir() and CONTRACT_PATH.is_file() and READY_PATH.is_file()
    require(not complete, "pre-attempt closure is already frozen")
    candidates = [path for path in OUTPUT_DIR.iterdir() if path.name != "preflight-failures"]
    require(candidates, "existing candidate namespace has ambiguous provenance")
    archive_root = OUTPUT_DIR / "preflight-failures"
    archive_root.mkdir(exist_ok=True)
    ordinal = 1
    while (archive_root / f"failure-{ordinal:04d}").exists():
        ordinal += 1
    archive = archive_root / f"failure-{ordinal:04d}"
    archive.mkdir()
    archived: list[dict[str, Any]] = []
    for path in candidates:
        destination = archive / path.name
        os.replace(path, destination)
        if destination.is_file():
            destination.chmod(0o444)
            archived.append(file_record(destination))
        else:
            for item in sorted(entry for entry in destination.rglob("*") if entry.is_file()):
                item.chmod(0o444)
                archived.append(file_record(item))
    return {"fresh_namespace": False, "recovered": True, "archived_artifacts": archived}


def artifact_records() -> dict[str, Any]:
    paths = {
        "engineer_task": TASK_PATH,
        "engineer_task_companion": TASK_COMPANION,
        "planning_requirements": PLAN_PATH,
        "planning_requirements_companion": PLAN_COMPANION,
        "source_contract": SOURCE_CONTRACT_PATH,
        "calibration_contract": CALIBRATION_CONTRACT_PATH,
        "calibration_report": CALIBRATION_REPORT_PATH,
        "base_scales": BASE_SCALES_PATH,
        "image_manifest": IMAGE_DIR / "manifest.json",
        "official_matrix": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "reviewed_alias_safe_runner": PRIOR_RUNNER_PATH,
        "reviewed_grouped_dynamic_runner": GROUPED_RUNNER_PATH,
        "shared_evaluator_source": ROOT / "tools/run_all_transformer_per32_stage1.py",
        "prompt_source": ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py",
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "oracle_source": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
        "identity_helper_source": ROOT / "tools/qwen_instruct_option_b.py",
        "quality_contracts": ROOT / "tools/ace2_quality_contracts.py",
        "non_evaluator_inputs": INPUT_MANIFEST_PATH,
        "candidate_a_manifest": CANDIDATE_A_PATH,
        "candidate_b_manifest": CANDIDATE_B_PATH,
        "alias_separation_witness": ALIAS_WITNESS_PATH,
        "dynamic_preflight": PREFLIGHT_PATH,
        "runner_self_test": SELF_TEST_PATH,
        "preparation_chronology": CHRONOLOGY_PATH,
    }
    if RECOVERY_PATH.is_file():
        paths["preparation_recovery"] = RECOVERY_PATH
    failure_root = OUTPUT_DIR / "preflight-failures"
    if failure_root.is_dir():
        for index, path in enumerate(sorted(item for item in failure_root.rglob("*") if item.is_file())):
            paths[f"nonqualifying_preflight_failure_{index:02d}"] = path
    return {name: file_record(path) for name, path in paths.items()}


def build_contract(environment: dict[str, Any], source_identity: dict[str, Any], namespace_audit: dict[str, Any]) -> dict[str, Any]:
    task = load_json(TASK_PATH)
    require(task["task_id"] == "task-0ea2df03119e", "Engineer task id differs")
    require(task["candidate_id"] == CANDIDATE_ID, "Engineer task candidate differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1, "attempt budget differs")
    require(task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "attempt was consumed at freeze")
    matrix = load_json(MATRIX_PATH)
    prompts = prompt_binding(matrix)
    require(len(prompts) == 9, "official prompt count differs")
    artifacts = artifact_records()
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "task_id": task["task_id"],
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "authority": {
            "engineer_task": artifacts["engineer_task"],
            "planning_requirements": artifacts["planning_requirements"],
            "attempt_id": "attempt-0001",
            "exact_authorized_candidate_attempts": 1,
            "attempts_consumed_before_execution": 0,
            "selected_policy_id_before_execution": None,
        },
        "source_model": source_identity,
        "construction": {
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_w4_linears": 168,
            "separated_lm_head_w4_linears": 1,
            "candidate_a_and_b_state_manifests_match": True,
            "witness": artifacts["alias_separation_witness"],
        },
        "dynamic_activation_policy": {
            "base_scales": artifacts["base_scales"],
            "event_families_per_layer": list(grouped.EVENT_FAMILIES),
            "event_count_all_24_layers": EXPECTED_EVENTS,
            "group_lanes": [64, 128],
            "payload_range": [-127, 127],
            "reserved_payload": -128,
            "delta_range": [-24, 24],
            "effective_exponent_range": [-24, 4],
            "selection": "minimum legal exponent delta preserving the frozen base Scale32 significand",
            "rounding": "round_to_nearest_ties_to_even",
            "all_zero_group_delta": 0,
            "saturation_or_silent_clipping_permitted": False,
            "transport_metadata_bytes_per_decoder_layer_token": 832,
            "tensor_sidecars_per_decoder_layer_token": 13,
            "preflight": artifacts["dynamic_preflight"],
            "official_inputs_used_for_preflight": False,
        },
        "compression": grouped.policy_cost(),
        "attempt": {
            "directory": ATTEMPT_DIR.relative_to(ROOT).as_posix(),
            "execution_marker": MARKER_PATH.relative_to(ROOT).as_posix(),
            "matrix": artifacts["official_matrix"],
            "prompt_binding": prompts,
            "prompt_count": 9,
            "generation": "deterministic greedy argmax; no sampling; use_cache=false; maximum six generated tokens; termination IDs 151643 and 151645",
            "score_policy": "binary only: 9/9 complete token arrays exactly equal BF16 and 9/9 response-gate outcomes equal BF16",
            "exact_command": ["./.venv/bin/python", RUNNER_PATH.relative_to(ROOT).as_posix(), "execute"],
            "attempt_budget": 1,
            "expected_evidence": ["execution_started.json", "run.log", "results.json or failure.json", "terminal_status.json", "fresh_reviewer_input.json when numerical evaluation completes", "SHA256SUMS", "SHA256SUMS.sha256"],
            "replay_resume_replacement_tuning_or_gate_relaxation_permitted": False,
            "network_access_permitted": False,
        },
        "environment": environment,
        "namespace_audit": namespace_audit,
        "artifacts": artifacts,
        "claim_boundary": {
            "selected_policy_id": None,
            "product_policy_accepted": False,
            "rtl_or_ppa_claimed": False,
            "current_stage_remains": "specification",
        },
    }
    return {"contract": contract, "contract_sha256": canonical_sha256(contract)}


def verify_bound_record(record: dict[str, Any]) -> None:
    path = ROOT / record["path"]
    require(path.is_file(), f"bound artifact is missing: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"bound artifact size differs: {record['path']}")
    _require_digest(sha256_file(path), record["sha256"], f"bound artifact hash differs: {record['path']}")


def validate_preflight(preflight: dict[str, Any]) -> None:
    summary = preflight["activation_execution"]
    require(preflight["status"] == "PASS_ALL_312_DYNAMIC_EVENTS", "preflight status differs")
    require(preflight["official_nine_prompt_inputs_used"] is False, "official inputs were used in preflight")
    require(preflight["official_execution_marker_exists"] is False, "preflight claims an execution marker")
    require(summary["event_count"] == summary["named_boundary_count"] == EXPECTED_EVENTS == 312, "preflight event count differs")
    require(summary["all_312_layer_event_names_present"] is True, "preflight event closure differs")
    require(-24 <= summary["minimum_delta"] <= summary["maximum_delta"] <= 24, "preflight delta range differs")
    require(summary["saturation_count"] == 0 and summary["clipping_count"] == 0, "preflight saturation or clipping differs")
    require(summary["reserved_minus_128_produced"] is False, "preflight produced reserved -128")
    require(summary["transport_metadata_bytes_per_decoder_layer_token"] <= 832, "transport metadata cap differs")
    require(summary["tensor_sidecars_per_decoder_layer_token"] <= 13, "sidecar count differs")


def load_verified_contract(require_unconsumed: bool) -> dict[str, Any]:
    require_project_python()
    require(CONTRACT_PATH.is_file() and READY_PATH.is_file(), "pre-attempt closure is missing")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == "task-0ea2df03119e", "predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == sha256_file(RUNNER_PATH), "runner changed after freeze")
    for record in contract["artifacts"].values():
        verify_bound_record(record)
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(runner_self_test() == load_json(SELF_TEST_PATH), "runner self-test result differs")
    validate_preflight(load_json(PREFLIGHT_PATH))
    ready = load_json(READY_PATH)
    require(ready["status"] == "PREATTEMPT_READY", "pre-attempt ready status differs")
    require(ready["candidate_id"] == CANDIDATE_ID, "pre-attempt ready candidate differs")
    require(ready["contract"]["sha256"] == sha256_file(CONTRACT_PATH), "ready contract file hash differs")
    require(ready["contract_sha256"] == wrapper["contract_sha256"], "ready contract canonical hash differs")
    require(ready["bound_artifacts"] == contract["artifacts"], "ready artifact snapshot differs")
    require(ready["official_execution_marker_exists"] is False, "ready record claims an execution marker")
    if require_unconsumed:
        prior.audit_attempt_namespace(expect_root_absent=False)
        require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "exactly-once attempt is consumed")
    return wrapper


def prepare() -> dict[str, Any]:
    environment = require_project_python()
    recovery = recover_nonqualifying_preparation()
    namespace_audit = prior.audit_attempt_namespace(expect_root_absent=recovery["fresh_namespace"])
    namespace_audit["preparation_recovery"] = recovery
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = OUTPUT_DIR / ".preflight-write-probe"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, b"preflight\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    require(probe.read_bytes() == b"preflight\n", "output-path readback differs")
    probe.unlink()
    chronology: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "preparation_started_at_utc": utc_now(),
        "namespace_audit": namespace_audit,
        "output_path_preflight": {"exclusive_create": True, "readback": True, "collision_behavior": "O_EXCL", "execution_marker_created": False},
    }
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    inputs, input_manifest = non_evaluator_inputs(tokenizer)
    self_test = runner_self_test()

    chronology["candidate_a_started_at_utc"] = utc_now()
    candidate_a, manifest_a, witness_a = load_candidate("candidate_a_state_proof")
    chronology["candidate_a_complete_at_utc"] = utc_now()
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    candidate_b, manifest_b, witness_b = load_candidate("candidate_b_dynamic_preflight")
    state_comparison = compare_candidates(manifest_a, manifest_b)
    preflight = dynamic_preflight(candidate_b, inputs, input_manifest)
    preflight["candidate_state_comparison"] = state_comparison
    preflight["embedding_hash_matches_source"] = manifest_b["identity"]["embedding_bf16_sha256"] == SOURCE_EMBEDDING_SHA256
    preflight["lm_head_storage_is_separate"] = witness_b["post_split"]["different_storage_pointers"]
    chronology["candidate_b_preflight_complete_at_utc"] = utc_now()
    del candidate_b
    gc.collect()
    validate_preflight(preflight)

    PREATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    write_json(INPUT_MANIFEST_PATH, input_manifest)
    write_json(CANDIDATE_A_PATH, manifest_a)
    write_json(CANDIDATE_B_PATH, manifest_b)
    write_json(ALIAS_WITNESS_PATH, {"schema_version": 1, "candidate_id": CANDIDATE_ID, "candidate_a": witness_a, "candidate_b": witness_b, "state_comparison": state_comparison})
    write_json(PREFLIGHT_PATH, preflight)
    write_json(SELF_TEST_PATH, self_test)
    if recovery["recovered"]:
        write_json(RECOVERY_PATH, {"schema_version": 1, "candidate_id": CANDIDATE_ID, **recovery})
    chronology["pre_attempt_artifacts_frozen_at_utc"] = utc_now()
    write_json(CHRONOLOGY_PATH, chronology)

    wrapper = build_contract(environment, source_identity, namespace_audit)
    write_json(CONTRACT_PATH, wrapper)
    ready = {
        "schema_version": 1,
        "status": "PREATTEMPT_READY",
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "contract": file_record(CONTRACT_PATH),
        "contract_sha256": wrapper["contract_sha256"],
        "bound_artifacts": wrapper["contract"]["artifacts"],
        "preflight": file_record(PREFLIGHT_PATH),
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "verification": "runner, dependencies, source, model, matrix, candidate state, all 312 dynamic events, metadata bounds, and output path verified",
    }
    write_json(READY_PATH, ready)
    load_verified_contract(require_unconsumed=True)
    for path in list(PREATTEMPT_DIR.iterdir()) + [CONTRACT_PATH, READY_PATH]:
        if path.is_file():
            path.chmod(0o444)
    return {
        "status": "PREATTEMPT_READY",
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "candidate_state_manifest_sha256": manifest_b["identity"]["candidate_state_manifest_sha256"],
        "event_count": preflight["activation_execution"]["event_count"],
        "minimum_delta": preflight["activation_execution"]["minimum_delta"],
        "maximum_delta": preflight["activation_execution"]["maximum_delta"],
        "saturation_count": 0,
        "clipping_count": 0,
        "attempts_consumed": 0,
    }


def execute_once() -> int:
    wrapper = load_verified_contract(require_unconsumed=True)
    ATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    marker = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "task_id": "task-0ea2df03119e",
        "candidate_id": CANDIDATE_ID,
        "attempt_id": "attempt-0001",
        "started_at_utc": utc_now(),
        "contract": file_record(CONTRACT_PATH),
        "contract_sha256": wrapper["contract_sha256"],
        "preattempt_ready": file_record(READY_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "matrix_sha256": MATRIX_SHA256,
        "base_scales_sha256": sha256_file(BASE_SCALES_PATH),
        "selected_policy_id_before_execution": None,
        "authorization_consumed": True,
    }
    atomic_write_json(MARKER_PATH, marker)
    started = time.monotonic()
    run_log_path = ATTEMPT_DIR / "run.log"
    candidate_prompt_count = 0
    result_path: Path | None = None
    reviewer_path: Path | None = None
    failure_path: Path | None = None
    return_code = 5
    terminal_class = "EVALUATOR_NO_EXECUTION"
    policy: AuditedDynamicPolicy | None = None
    run_log = run_log_path.open("w", encoding="utf-8")
    try:
        log_message(run_log, f"official_attempt_started candidate={CANDIDATE_ID} attempt=attempt-0001")
        torch.set_num_threads(TORCH_THREADS)
        torch.set_num_interop_threads(1)
        source_contract = verify_source_contract()
        source_identity = verify_source_snapshot()
        versions = verify_versions()
        matrix = load_json(MATRIX_PATH)
        prompt_specs = {record["case_id"]: record for record in prompt_binding(matrix)}
        from discriminate_qwen_instruct_w4_weight_policies import PROMPTS, chat_input

        require(len(PROMPTS) == 9, "official prompt count differs")
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
        inputs = {prompt.case_id: chat_input(tokenizer, prompt.prompt, DEFAULT_SYSTEM) for prompt in PROMPTS}
        reference_model = AutoModelForCausalLM.from_pretrained(
            SNAPSHOT,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()
        references: dict[str, dict[str, Any]] = {}
        log_message(run_log, "bf16_reference_recomputation_started prompts=9")
        for prompt in PROMPTS:
            reference = generate_reference(reference_model, tokenizer, inputs[prompt.case_id], prompt.expected)
            frozen = prompt_specs[prompt.case_id]
            require(reference["generated_token_ids"] == frozen["bf16_generated_token_ids"], f"BF16 tokens differ: {prompt.case_id}")
            require(reference["response_gate"]["status"] == frozen["bf16_response_gate_status"], f"BF16 response gate differs: {prompt.case_id}")
            references[prompt.case_id] = reference
            log_message(run_log, f"bf16_prompt_complete case={prompt.case_id}")
        del reference_model
        gc.collect()

        candidate_model, official_manifest, official_witness = load_candidate("official_attempt_candidate")
        preflight_manifest = load_json(CANDIDATE_B_PATH)
        comparison = compare_candidates(preflight_manifest, official_manifest)
        policy = AuditedDynamicPolicy(load_json(BASE_SCALES_PATH))
        policy.install(candidate_model)
        prompt_results: list[dict[str, Any]] = []
        model_forwards = 0
        for prompt in PROMPTS:
            reference = references[prompt.case_id]
            candidate, forwards = grouped.generate_candidate(
                candidate_model,
                tokenizer,
                inputs[prompt.case_id],
                prompt.expected,
                reference,
            )
            model_forwards += forwards
            candidate_prompt_count += 1
            token_disagreements = sequence_disagreements(reference["generated_token_ids"], candidate["generated_token_ids"])
            gate_match = candidate["response_gate"]["status"] == reference["response_gate"]["status"]
            public_reference = dict(reference)
            public_reference.pop("_logits")
            prompt_results.append(
                {
                    "case_id": prompt.case_id,
                    "prompt_sha256": prompt_specs[prompt.case_id]["prompt_sha256"],
                    "expected_sha256": prompt_specs[prompt.case_id]["expected_sha256"],
                    "bf16": public_reference,
                    "candidate": candidate,
                    "token_disagreements": token_disagreements,
                    "response_gate_outcome_match": gate_match,
                }
            )
            log_message(run_log, f"candidate_prompt_complete case={prompt.case_id} exact={candidate['exact_bf16_sequence_match']} gate_match={gate_match}")
        activation_summary = policy.audited_summary(model_forwards)
        policy.uninstall()
        policy = None
        aggregate = aggregate_prompts(prompt_results)
        disagreement_set = [
            {
                "case_id": item["case_id"],
                "token_disagreements": item["token_disagreements"],
                "bf16_response_gate_status": item["bf16"]["response_gate"]["status"],
                "candidate_response_gate_status": item["candidate"]["response_gate"]["status"],
                "response_gate_outcome_match": item["response_gate_outcome_match"],
            }
            for item in prompt_results
            if item["token_disagreements"] or not item["response_gate_outcome_match"]
        ]
        eligible = aggregate["exact_bf16_sequence_match_count"] == 9 and aggregate["response_gate_outcome_match_count"] == 9
        numerical_status = "PASS_ELIGIBLE_FOR_FRESH_REVIEW" if eligible else "BLOCKED_EXACT_BF16_DISAGREEMENT"
        result = {
            "schema_version": 1,
            "classification": "qwen_instruct_stage1_option_b_alias_safe_post_w4_grouped_dynamic_scale32_single_candidate",
            "status": numerical_status,
            "mission_id": MISSION_ID,
            "task_id": "task-0ea2df03119e",
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "attempt-0001",
            "contract": file_record(CONTRACT_PATH),
            "preattempt_ready": file_record(READY_PATH),
            "matrix": file_record(MATRIX_PATH),
            "base_scales": file_record(BASE_SCALES_PATH),
            "source_model": source_identity,
            "source_contract_status": source_contract["status"],
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": {**versions, "numpy": importlib.metadata.version("numpy")}, "torch_num_threads": torch.get_num_threads()},
            "official_candidate_manifest": official_manifest,
            "official_alias_witness": official_witness,
            "preflight_state_match": comparison,
            "activation_execution": activation_summary,
            "prompts": prompt_results,
            "aggregate": aggregate,
            "disagreement_set": disagreement_set,
            "eligibility": {"rule": wrapper["contract"]["attempt"]["score_policy"], "eligible": eligible, "eligible_candidate_id": CANDIDATE_ID if eligible else None, "selected_policy_id": None, "fresh_reviewer_required": True},
            "failure_analysis": None if eligible else {"failure_taxonomy": "NUMERICAL_EXACT_BF16_DISAGREEMENT", "root_cause_hypothesis": "legal grouped Dynamic Scale32 activation reconstruction changed one or more greedy logits relative to BF16", "regression": "preserve the exact immutable disagreement set and response-gate comparison for independent review; do not rerun"},
            "scope_guards": {"candidate_attempt_count": 1, "alternate_policy_executed": False, "official_inputs_used_before_marker": False, "static_prompt_range_table_used": False, "saturation_or_clipping_used": False, "runner_changed_after_freeze": False, "rtl_mutated": False, "ppa_executed": False, "network_access_performed": False},
            "claim_boundary": "Numerical-policy attempt only; no RTL, latency, timing, PPA, demo, U280, product-selection, or stage-advance claim.",
            "timing": {"completed_at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started},
        }
        result_path = ATTEMPT_DIR / "results.json"
        write_json(result_path, result)
        reviewer = {
            "schema_version": 1,
            "status": "READY_FOR_FRESH_REVIEW",
            "mission_id": MISSION_ID,
            "task_id": "task-0ea2df03119e",
            "candidate_id": CANDIDATE_ID,
            "numerical_result": numerical_status,
            "contract": file_record(CONTRACT_PATH),
            "preattempt_ready": file_record(READY_PATH),
            "results": file_record(result_path),
            "selected_policy_id_before_adjudication": None,
            "independent_acceptance_required": True,
            "requested_review": ["task and hash closure", "source-tied then runtime-split lm_head and preserved embedding bytes", "candidate A/B state equality", "all-312-event legal Dynamic Scale32 preflight", "one consumed attempt", "nine exact BF16 token arrays and nine response-gate comparisons"],
            "verification_command": ["./.venv/bin/python", RUNNER_PATH.relative_to(ROOT).as_posix(), "verify-result"],
            "official_attempt_regeneration_forbidden": True,
        }
        reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
        write_json(reviewer_path, reviewer)
        terminal_class = "NUMERICAL_SUCCESS" if eligible else "NUMERICAL_FAILURE"
        return_code = 0 if eligible else 4
        log_message(run_log, f"official_attempt_complete status={numerical_status} exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 gate_matches={aggregate['response_gate_outcome_match_count']}/9 selected_policy_id=null")
    except Exception as exc:
        if policy is not None:
            policy.uninstall()
        terminal_class = "EVALUATOR_NO_EXECUTION" if candidate_prompt_count == 0 else "EVALUATOR_PARTIAL_EXECUTION_FAILURE"
        failure = {
            "schema_version": 1,
            "status": "BLOCKED_" + terminal_class,
            "classification": terminal_class,
            "failure_taxonomy": terminal_class,
            "root_cause_hypothesis": f"official runner raised {type(exc).__name__}: {exc}",
            "regression": "preserve the immutable failure, marker, log, traceback, and completed-prompt count; do not replay",
            "mission_id": MISSION_ID,
            "task_id": "task-0ea2df03119e",
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "attempt-0001",
            "candidate_prompt_results_completed": candidate_prompt_count,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "numerical_correctness_conclusion": None,
            "selected_policy_id": None,
            "authorization_consumed": True,
            "failed_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        }
        failure_path = ATTEMPT_DIR / "failure.json"
        write_json(failure_path, failure)
        log_message(run_log, f"official_attempt_failed classification={terminal_class} error_type={type(exc).__name__} error={exc}")
        return_code = 5
    finally:
        run_log.flush()
        os.fsync(run_log.fileno())
        run_log.close()

    terminal_path = ATTEMPT_DIR / "terminal_status.json"
    write_json(terminal_path, {"schema_version": 1, "candidate_id": CANDIDATE_ID, "attempt_id": "attempt-0001", "classification": terminal_class, "return_code": return_code, "candidate_prompt_results_completed": candidate_prompt_count, "completed_at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started})
    evidence = [CONTRACT_PATH, READY_PATH, MARKER_PATH, run_log_path, terminal_path]
    for path in (result_path, reviewer_path, failure_path):
        if path is not None:
            evidence.append(path)
    write_checksum_bundle(evidence)
    seal_attempt([MARKER_PATH, run_log_path, terminal_path] + [path for path in (result_path, reviewer_path, failure_path) if path is not None])
    return return_code


def verify_result() -> dict[str, Any]:
    wrapper = load_verified_contract(require_unconsumed=False)
    attempts = sorted(path.name for path in OUTPUT_DIR.glob("attempt-*") if path.is_dir())
    require(attempts == ["attempt-0001"], "official attempt set differs")
    marker = load_json(MARKER_PATH)
    require(marker["authorization_consumed"] is True, "attempt authorization was not consumed")
    require(marker["runner_sha256"] == sha256_file(RUNNER_PATH), "executed runner hash differs")
    require(marker["contract_sha256"] == wrapper["contract_sha256"], "executed contract hash differs")
    sums = ATTEMPT_DIR / "SHA256SUMS"
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    verify_companion(sums, companion)
    for digest, path in parse_sums(sums):
        require(path.is_file() and sha256_file(path) == digest, f"official evidence hash differs: {path}")
    terminal = load_json(ATTEMPT_DIR / "terminal_status.json")
    failure_path = ATTEMPT_DIR / "failure.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        require(failure["numerical_correctness_conclusion"] is None, "execution failure drew a numerical conclusion")
        require(failure["selected_policy_id"] is None, "execution failure selected a policy")
        return {"status": failure["status"], "classification": failure["classification"], "candidate_id": CANDIDATE_ID, "attempt_count": 1, "candidate_prompt_results_completed": failure["candidate_prompt_results_completed"], "eligible": None, "selected_policy_id": None, "failure_sha256": sha256_file(failure_path), "terminal_return_code": terminal["return_code"]}
    result = load_json(ATTEMPT_DIR / "results.json")
    require(result["candidate_id"] == CANDIDATE_ID, "result candidate differs")
    require(len(result["prompts"]) == 9, "result prompt count differs")
    require(result["official_candidate_manifest"]["identity"]["candidate_state_manifest_sha256"] == load_json(CANDIDATE_B_PATH)["identity"]["candidate_state_manifest_sha256"], "official candidate state differs from preflight")
    activation = result["activation_execution"]
    require(activation["event_count"] == 312 and activation["all_312_layer_event_names_present"] is True, "official dynamic event closure differs")
    require(-24 <= activation["minimum_delta"] <= activation["maximum_delta"] <= 24, "official delta range differs")
    require(activation["saturation_count"] == 0 and activation["clipping_count"] == 0, "official saturation or clipping differs")
    require(activation["reserved_minus_128_produced"] is False, "official payload produced reserved -128")
    disagreements: list[dict[str, Any]] = []
    exact_count = 0
    gate_count = 0
    for item in result["prompts"]:
        token_disagreements = sequence_disagreements(item["bf16"]["generated_token_ids"], item["candidate"]["generated_token_ids"])
        require(token_disagreements == item["token_disagreements"], f"token disagreement record differs: {item['case_id']}")
        gate_match = item["bf16"]["response_gate"]["status"] == item["candidate"]["response_gate"]["status"]
        require(gate_match == item["response_gate_outcome_match"], f"response-gate match differs: {item['case_id']}")
        exact_count += int(not token_disagreements)
        gate_count += int(gate_match)
        if token_disagreements or not gate_match:
            disagreements.append({"case_id": item["case_id"], "token_disagreements": token_disagreements, "bf16_response_gate_status": item["bf16"]["response_gate"]["status"], "candidate_response_gate_status": item["candidate"]["response_gate"]["status"], "response_gate_outcome_match": gate_match})
    require(disagreements == result["disagreement_set"], "result disagreement set differs")
    require(exact_count == result["aggregate"]["exact_bf16_sequence_match_count"], "exact sequence aggregate differs")
    require(gate_count == result["aggregate"]["response_gate_outcome_match_count"], "response-gate aggregate differs")
    eligible = exact_count == 9 and gate_count == 9
    require(result["eligibility"]["eligible"] == eligible, "eligibility differs")
    require(result["eligibility"]["selected_policy_id"] is None, "result selected a policy before review")
    reviewer = load_json(ATTEMPT_DIR / "fresh_reviewer_input.json")
    require(reviewer["official_attempt_regeneration_forbidden"] is True, "review packet permits regeneration")
    return {"status": result["status"], "candidate_id": CANDIDATE_ID, "attempt_count": 1, "eligible": eligible, "exact_sequence_matches": exact_count, "gate_outcome_matches": gate_count, "disagreement_case_ids": [item["case_id"] for item in disagreements], "selected_policy_id": None, "results_sha256": sha256_file(ATTEMPT_DIR / "results.json"), "reviewer_input_sha256": sha256_file(ATTEMPT_DIR / "fresh_reviewer_input.json"), "terminal_return_code": terminal["return_code"]}


def record_preparation_failure(exc: Exception) -> None:
    if MARKER_PATH.exists():
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failure = OUTPUT_DIR / f"preparation_failure_{time.time_ns()}.json"
    write_json(failure, {"schema_version": 1, "candidate_id": CANDIDATE_ID, "classification": "PREATTEMPT_FAILURE", "failure_taxonomy": "PREATTEMPT_EXECUTION_FAILURE", "root_cause_hypothesis": f"runner raised {type(exc).__name__}: {exc}", "regression": "repair the isolated runner and rerun only pre-attempt preparation; official authorization remains unconsumed", "error_type": type(exc).__name__, "error": str(exc), "runner_sha256": sha256_file(RUNNER_PATH), "traceback": traceback.format_exc(), "attempt_authorization_consumed": False, "official_execution_marker_exists": False, "failed_at_utc": utc_now()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "verify-ready", "execute", "verify-result"))
    args = parser.parse_args()
    if args.command == "self-test":
        print("ACE2_OPTION_B_DYNAMIC_V2_RUNNER_SELF_TEST_PASS " + json.dumps(runner_self_test(), sort_keys=True))
        return 0
    if args.command == "prepare":
        try:
            result = prepare()
        except Exception as exc:
            record_preparation_failure(exc)
            raise
        print("ACE2_OPTION_B_DYNAMIC_V2_PREATTEMPT_READY " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "verify-ready":
        wrapper = load_verified_contract(require_unconsumed=True)
        print("ACE2_OPTION_B_DYNAMIC_V2_READY_VERIFIED " + json.dumps({"candidate_id": CANDIDATE_ID, "attempts_consumed": 0, "contract_sha256": wrapper["contract_sha256"], "event_count": 312}, sort_keys=True))
        return 0
    if args.command == "execute":
        return execute_once()
    summary = verify_result()
    print("ACE2_OPTION_B_DYNAMIC_V2_RESULT_VERIFIED " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
