#!/usr/bin/env python3
"""Run one frozen lossless element-Scale32 W4/A8 Stage-1 candidate."""

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
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from discriminate_qwen_instruct_w4_weight_policies import PROMPTS, chat_input
from qwen_instruct_option_b import (
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    TERMINATION_TOKEN_IDS,
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
from qwen_instruct_response_gate import evaluate as evaluate_response_gate
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM
from run_all_transformer_per32_stage1 import (
    aggregate_prompts,
    generate_reference,
    log_message,
    logit_comparison,
    prompt_binding,
    sequence_disagreements,
    utc_now,
    write_json,
)


CANDIDATE_ID = "lossless_element_scale32_w4a8_v1"
MISSION_ID = "f4af3b351500"
MISSION_CONTRACT_SHA256 = "d45137154180400bfa6ae18b24003b058b65d8b07ffa78f7c6e5f6a69605151e"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
OUTPUT_DIR = ROOT / "build/stage1-lossless-element-scale32-w4a8-v1"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
RUNNER_PATH = Path(__file__).resolve()
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16
CODEC_CHUNK_ELEMENTS = 1 << 20
CONSUMED_RESULTS = (
    ROOT / "build/stage1-global-w4-per32-repair-v1/attempt-0001/results.json",
    ROOT / "build/stage1-causal-four-w4-per32-repair-v1/attempt-0001/results.json",
    ROOT / "build/stage1-layer2-dyns32-layer23-down-per32-v1/attempt-0001/results.json",
)


def require_project_python() -> None:
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), f"project Python is missing: {expected}")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")


def scale32x_encode_bits(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode finite BF16 bits as q in {-1,0,+1} and a u32 exact scale record."""

    raw = np.asarray(bits, dtype=np.uint16)
    magnitude = raw & np.uint16(0x7FFF)
    exponent_bits = (magnitude >> np.uint16(7)).astype(np.int32)
    fraction = (magnitude & np.uint16(0x007F)).astype(np.int32)
    require(not bool(np.any(exponent_bits == 0xFF)), "Scale32-X16 source is non-finite")

    nonzero = magnitude != 0
    negative_zero = raw == np.uint16(0x8000)
    negative = (raw & np.uint16(0x8000)) != 0
    qvalue = np.zeros(raw.shape, dtype=np.int8)
    qvalue[nonzero & ~negative] = 1
    qvalue[nonzero & negative] = -1

    significand = np.zeros(raw.shape, dtype=np.uint32)
    exponent = np.zeros(raw.shape, dtype=np.int32)
    normal = exponent_bits != 0
    significand[normal] = ((128 + fraction[normal]) << 8).astype(np.uint32)
    exponent[normal] = exponent_bits[normal] - 127

    subnormal = nonzero & ~normal
    if bool(np.any(subnormal)):
        sub_fraction = fraction[subnormal]
        highest_bit = np.floor(np.log2(sub_fraction.astype(np.float64))).astype(np.int32)
        significand[subnormal] = (sub_fraction << (15 - highest_bit)).astype(np.uint32)
        exponent[subnormal] = highest_bit - 133

    require(bool(np.all((significand[nonzero] >= 0x8000) & (significand[nonzero] <= 0xFFFF))), "Scale32-X16 significand escaped normalized range")
    require(bool(np.all((exponent[nonzero] >= -133) & (exponent[nonzero] <= 127))), "Scale32-X16 exponent escaped BF16 range")
    record = significand | ((exponent.astype(np.int16).view(np.uint16).astype(np.uint32)) << 16)
    record[~nonzero] = 0
    record[negative_zero] = np.uint32(0x80000000)
    return qvalue, record.astype("<u4", copy=False)


def scale32x_decode(qvalue: np.ndarray, records: np.ndarray) -> Tensor:
    q = torch.from_numpy(np.asarray(qvalue, dtype=np.int8).copy()).to(torch.float64)
    record = np.asarray(records, dtype="<u4")
    significand = torch.from_numpy((record & np.uint32(0xFFFF)).astype(np.int64, copy=False).copy())
    exponent_u16 = (record >> np.uint32(16)).astype(np.uint16, copy=False)
    exponent = torch.from_numpy(exponent_u16.view(np.int16).astype(np.int64, copy=False).copy())
    decoded = q * torch.ldexp(significand.to(torch.float64), exponent.to(torch.int32) - 15)
    negative_zero = torch.from_numpy(
        ((record == np.uint32(0x80000000)) & (np.asarray(qvalue, dtype=np.int8) == 0)).copy()
    )
    decoded[negative_zero] = -0.0
    return decoded


def exhaustive_codec_self_test() -> dict[str, Any]:
    bits = np.arange(1 << 16, dtype=np.uint16)
    exponent_bits = (bits >> np.uint16(7)) & np.uint16(0x00FF)
    finite = exponent_bits != np.uint16(0x00FF)
    finite_bits = bits[finite]
    qvalue, records = scale32x_encode_bits(finite_bits)
    decoded = scale32x_decode(qvalue, records).to(torch.bfloat16)
    decoded_bits = decoded.view(torch.int16).to(torch.int32).numpy().astype(np.uint16, copy=False)
    require(np.array_equal(decoded_bits, finite_bits), "Scale32-X16 exhaustive BF16 round-trip differs")
    return {
        "status": "PASS",
        "finite_bf16_pattern_count": int(finite_bits.size),
        "nonfinite_patterns_rejected": int(bits.size - finite_bits.size),
        "qvalue_set": [-1, 0, 1],
        "record_bits": 32,
        "minimum_exponent": -133,
        "maximum_exponent": 127,
        "decoded_bits_sha256": hashlib.sha256(decoded_bits.astype("<u2", copy=False).tobytes()).hexdigest(),
    }


def pack_w4_chunk(qvalue: np.ndarray) -> bytes:
    q = np.asarray(qvalue, dtype=np.int8).reshape(-1)
    if q.size & 1:
        q = np.concatenate((q, np.zeros(1, dtype=np.int8)))
    nibble = q.astype(np.uint8, copy=False) & np.uint8(0x0F)
    return (nibble[0::2] | (nibble[1::2] << np.uint8(4))).tobytes(order="C")


def audit_model_parameters(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    aggregate_w4 = hashlib.sha256()
    aggregate_scale32x = hashlib.sha256()
    total_values = 0
    packed_bytes = 0
    scale_bytes = 0
    parameter_count = 0

    for name, parameter in model.named_parameters():
        require(parameter.dtype == torch.bfloat16, f"candidate parameter is not BF16: {name}")
        flat = parameter.detach().contiguous().view(torch.int16).reshape(-1)
        parameter_w4 = hashlib.sha256()
        parameter_scale = hashlib.sha256()
        parameter_values = int(flat.numel())
        parameter_packed_bytes = 0
        parameter_scale_bytes = 0
        for start in range(0, parameter_values, CODEC_CHUNK_ELEMENTS):
            raw = flat[start : start + CODEC_CHUNK_ELEMENTS].to(torch.int32).numpy()
            bits = (raw & 0xFFFF).astype(np.uint16, copy=False)
            qvalue, scale_records = scale32x_encode_bits(bits)
            packed = pack_w4_chunk(qvalue)
            scale_raw = scale_records.tobytes(order="C")
            parameter_w4.update(packed)
            parameter_scale.update(scale_raw)
            aggregate_w4.update(packed)
            aggregate_scale32x.update(scale_raw)
            parameter_packed_bytes += len(packed)
            parameter_scale_bytes += len(scale_raw)
        expected_packed_bytes = (parameter_values + 1) // 2
        require(parameter_packed_bytes == expected_packed_bytes, f"packed W4 length differs: {name}")
        require(parameter_scale_bytes == parameter_values * 4, f"Scale32-X16 length differs: {name}")
        records.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "value_count": parameter_values,
                "signed_w4_payload_bytes": parameter_packed_bytes,
                "signed_w4_payload_sha256": parameter_w4.hexdigest(),
                "scale32x_metadata_bytes": parameter_scale_bytes,
                "scale32x_metadata_sha256": parameter_scale.hexdigest(),
                "exact_bf16_round_trip": True,
            }
        )
        total_values += parameter_values
        packed_bytes += parameter_packed_bytes
        scale_bytes += parameter_scale_bytes
        parameter_count += 1

    require(parameter_count > 0 and total_values > 0, "candidate parameter audit is empty")
    return records, {
        "unique_parameter_tensor_count": parameter_count,
        "unique_parameter_value_count": total_values,
        "signed_w4_payload_bytes": packed_bytes,
        "scale32x_metadata_bytes": scale_bytes,
        "metadata_to_payload_ratio": scale_bytes / packed_bytes,
        "signed_w4_scope_sha256": aggregate_w4.hexdigest(),
        "scale32x_scope_sha256": aggregate_scale32x.hexdigest(),
        "parameter_manifest_sha256": sha256_bytes(canonical_bytes(records)),
    }


def candidate_cost() -> dict[str, Any]:
    return {
        "weight_mantissa_bits": 4,
        "activation_mantissa_bits": 8,
        "weight_mantissa_values": [-1, 0, 1],
        "activation_mantissa_values": [-1, 0, 1],
        "scale32x_metadata_bits_per_weight_scalar": 32,
        "scale32x_metadata_bits_per_live_activation_scalar": 32,
        "lossless_bf16_value_domain": True,
        "metadata_cost_is_unbounded_by_current_product_budget": True,
        "rtl_area_timing_or_bandwidth_feasibility_claimed": False,
    }


def contract_artifacts() -> dict[str, Any]:
    paths = {
        "matrix": MATRIX_PATH,
        "source_contract": ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json",
        "full_model_image": IMAGE_DIR / "full_model_image.bin",
        "image_contract": IMAGE_DIR / "image_contract_v2.json",
        "image_manifest": IMAGE_DIR / "manifest.json",
        "option_b_identity_manifest": IMAGE_DIR / "option_b_identity_manifest.json",
        "oracle_contract": IMAGE_DIR / "oracle_contract.json",
        "response_gate_contract": IMAGE_DIR / "response_gate_contract.json",
        "runner_source": RUNNER_PATH,
        "shared_evaluator_source": ROOT / "tools/run_all_transformer_per32_stage1.py",
        "prompt_and_matrix_source": ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py",
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "oracle_source": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
    }
    for index, path in enumerate(CONSUMED_RESULTS, start=1):
        paths[f"consumed_candidate_result_{index}"] = path
    return {name: file_record(path) for name, path in paths.items()}


def scan_existing_candidate_results() -> list[str]:
    matches: list[str] = []
    for path in sorted((ROOT / "build").glob("**/attempt-*/results.json")):
        try:
            record = load_json(path)
        except Exception:
            continue
        if record.get("candidate_id") == CANDIDATE_ID:
            matches.append(path.relative_to(ROOT).as_posix())
    return matches


def verify_consumed_candidates() -> list[dict[str, Any]]:
    expected = (
        "all_transformer_per_32_v1",
        "causal_four_per_32_v1",
        "layer2_dyns32_layer23_down_per32_v1",
    )
    records: list[dict[str, Any]] = []
    for path, candidate_id in zip(CONSUMED_RESULTS, expected, strict=True):
        result = load_json(path)
        require(result.get("candidate_id") == candidate_id, f"consumed candidate binding differs: {candidate_id}")
        require(result.get("eligibility", {}).get("eligible") is False, f"consumed candidate eligibility differs: {candidate_id}")
        records.append(
            {
                "candidate_id": candidate_id,
                "result": file_record(path),
                "attempt_count": 1,
                "replayed": False,
            }
        )
    return records


def prepare_contract() -> dict[str, Any]:
    require_project_python()
    require(not OUTPUT_DIR.exists(), "candidate output directory already exists")
    require(not scan_existing_candidate_results(), "candidate result already exists")
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "immutable matrix hash differs")
    verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    prompts = prompt_binding(load_json(MATRIX_PATH))
    consumed = verify_consumed_candidates()
    codec = exhaustive_codec_self_test()
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "mission_contract_sha256": MISSION_CONTRACT_SHA256,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "hypothesis": (
            "A structurally distinct signed-unit W4/A8 representation with one exact 32-bit "
                "Scale32-X16 sidecar per scalar removes quantization error by bijectively encoding "
            "every finite BF16 value; it should therefore match the BF16 sequence and gate oracle "
            "while exposing a deliberately severe metadata cost for later architecture review."
        ),
        "distinctness": {
            "mechanism": "lossless_per_scalar_signed_unit_mantissa_plus_scale32x_sidecar",
            "weight_grouping": "one scale record per scalar rather than per row, per-32 group, or selected tensor",
            "activation_scope": "every finite BF16 activation value is representable exactly",
            "replay": False,
            "scope_identical_to_consumed_candidate": False,
            "consumed_candidates": consumed,
        },
        "numerical_policy": {
            "weight_payload": "signed int4 q in {-1,0,+1}",
            "activation_payload": "signed int8 q in {-1,0,+1}",
            "scale_record": (
                "unsigned 16-bit normalized significand in bits[15:0] and signed 16-bit binary "
                "exponent in bits[31:16]; +0 uses q=0/record=0 and -0 uses the reserved "
                "q=0/record=0x80000000 marker"
            ),
            "decode": "q * significand * 2^(signed_exponent-15)",
            "finite_bf16_bijection": True,
            "nonfinite_values": "rejected",
            "operator_realization": (
                "decode the exact BF16 value before each frozen operator; the reference realization "
                "may execute the identical BF16 operator because exhaustive codec equivalence proves "
                "the decoded operand bits are unchanged"
            ),
            "rounding": "none at encode/decode because the finite BF16 mapping is exact",
            "saturation": "none for finite BF16 values",
            "dataset_prompt_or_token_control": False,
            "selector_or_gate_control": False,
            "hardware_feasibility_claimed": False,
        },
        "codec_preflight": codec,
        "matrix": {
            "path": MATRIX_PATH.relative_to(ROOT).as_posix(),
            "sha256": MATRIX_SHA256,
            "case_count": 9,
            "full_generated_sequences_bound": True,
        },
        "generation_and_evaluator": {
            "system_prompt_utf8_bytes": len(DEFAULT_SYSTEM.encode()),
            "system_prompt_sha256": sha256_bytes(DEFAULT_SYSTEM.encode()),
            "chat_template_sha256": source_identity["chat_template_sha256"],
            "greedy_argmax": True,
            "sampling": False,
            "use_cache": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
            "response_gate": "frozen qwen-instruct-option-b-response-gate-v1",
            "prompts": prompts,
        },
        "eligibility": {
            "rule": "all nine complete generated token sequences exactly equal BF16 and all nine response-gate outcomes equal BF16",
            "aggregate_improvement_is_insufficient": True,
            "selected_policy_id_before_execution": None,
            "fresh_reviewer_required": True,
        },
        "execution": {
            "exact_authorized_candidate_attempts": 1,
            "attempt_id": "attempt-0001",
            "exact_command": [
                "./.venv/bin/python",
                "tools/run_lossless_element_scale32_stage1.py",
                "execute",
            ],
            "working_directory": str(ROOT),
            "torch_num_threads": TORCH_THREADS,
            "network_access_permitted": False,
            "alternate_policy_permitted": False,
            "tuning_ranking_bisection_sweep_permitted": False,
            "selector_or_gate_relaxation_permitted": False,
            "rtl_demo_ppa_u280_stage2_mutation_permitted": False,
            "expected_outputs": [
                "build/stage1-lossless-element-scale32-w4a8-v1/attempt-0001/execution_started.json",
                "build/stage1-lossless-element-scale32-w4a8-v1/attempt-0001/run.log",
                "build/stage1-lossless-element-scale32-w4a8-v1/attempt-0001/results.json",
                "build/stage1-lossless-element-scale32-w4a8-v1/attempt-0001/fresh_reviewer_input.json",
                "build/stage1-lossless-element-scale32-w4a8-v1/attempt-0001/SHA256SUMS",
                "build/stage1-lossless-element-scale32-w4a8-v1/attempt-0001/SHA256SUMS.sha256",
            ],
        },
        "model": source_identity,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
        },
        "static_cost": candidate_cost(),
        "artifacts": contract_artifacts(),
        "no_outcome_assumption": (
            "No token, response-gate, eligibility, stage, RTL, PPA, FPGA, or product outcome "
            "is assumed before the single authorized execution."
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
    require(sha256_bytes(canonical_bytes(contract)) == wrapper["contract_sha256"], "contract digest differs")
    require(contract.get("mission_id") == MISSION_ID, "mission id differs")
    require(contract.get("mission_contract_sha256") == MISSION_CONTRACT_SHA256, "mission binding differs")
    require(contract.get("candidate_id") == CANDIDATE_ID, "candidate id differs")
    require(contract.get("matrix", {}).get("sha256") == MATRIX_SHA256, "matrix binding differs")
    require(contract.get("execution", {}).get("exact_authorized_candidate_attempts") == 1, "attempt budget differs")
    require(contract.get("execution", {}).get("alternate_policy_permitted") is False, "alternate policy guard differs")
    require(contract.get("numerical_policy", {}).get("finite_bf16_bijection") is True, "lossless policy differs")
    for record in contract.get("artifacts", {}).values():
        path = ROOT / record["path"]
        require(path.is_file(), f"contract artifact is missing: {record['path']}")
        require(path.stat().st_size == record["bytes"], f"contract artifact size differs: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"contract artifact hash differs: {record['path']}")
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "immutable matrix hash differs")
    require(prompt_binding(load_json(MATRIX_PATH)) == contract["generation_and_evaluator"]["prompts"], "prompt binding differs")
    require(exhaustive_codec_self_test() == contract["codec_preflight"], "codec preflight differs")
    verify_source_contract()
    verify_source_snapshot()
    verify_versions()
    verify_consumed_candidates()
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    if require_unconsumed:
        require(not attempts, f"exactly-once authorization is already consumed: {[path.name for path in attempts]}")
    return wrapper


def generate_candidate(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    expected: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    prefix = input_ids
    generated: list[int] = []
    prefix_lengths: list[int] = []
    steps: list[dict[str, Any]] = []
    aligned = True
    with torch.inference_mode():
        for index in range(MAX_NEW_TOKENS):
            prefix_lengths.append(int(prefix.shape[1]))
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1].detach().cpu()
            token = int(logits.argmax())
            comparison = None
            if aligned and index < len(reference["_logits"]):
                comparison = logit_comparison(reference["_logits"][index], logits)
            steps.append(
                {
                    "index": index,
                    "token_id": token,
                    "incoming_prefix_aligned_with_bf16": aligned,
                    "aligned_logit_comparison": comparison,
                }
            )
            generated.append(token)
            reference_ids = reference["generated_token_ids"]
            aligned = aligned and index < len(reference_ids) and token == reference_ids[index]
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
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
        "prefix_lengths": prefix_lengths,
        "model_token_positions_evaluated": sum(prefix_lengths),
        "reference_realization": "exact Scale32-X16 decode followed by the frozen BF16 operator",
        "steps": steps,
    }


def write_attempt_checksums(paths: list[Path]) -> None:
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    payload = "".join(
        f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths
    )
    sums_path.write_text(payload, encoding="utf-8")
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    companion.write_text(f"{sha256_file(sums_path)}  {sums_path.name}\n", encoding="utf-8")


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
        "exact_command": wrapper["contract"]["execution"]["exact_command"],
        "authorization_consumed": True,
    }
    started_path = ATTEMPT_DIR / "execution_started.json"
    write_json(started_path, started)
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
                require(record["generated_token_ids"] == expected["bf16_generated_token_ids"], f"BF16 token binding differs: {prompt_case.case_id}")
                require(record["response_gate"]["status"] == expected["bf16_response_gate_status"], f"BF16 response gate differs: {prompt_case.case_id}")
                references[prompt_case.case_id] = record
                log_message(run_log, f"bf16_prompt_complete case={prompt_case.case_id} tokens={len(record['generated_token_ids'])} gate={record['response_gate']['status']}")
            del reference_model
            gc.collect()

            log_message(run_log, "candidate_model_construction_started policy=lossless_element_scale32x")
            candidate_model = AutoModelForCausalLM.from_pretrained(
                SNAPSHOT,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval()
            codec = exhaustive_codec_self_test()
            parameter_manifest, parameter_scope = audit_model_parameters(candidate_model)
            log_message(
                run_log,
                "candidate_model_construction_complete "
                f"parameters={parameter_scope['unique_parameter_tensor_count']} "
                f"values={parameter_scope['unique_parameter_value_count']} "
                f"w4_sha256={parameter_scope['signed_w4_scope_sha256']} "
                f"scale32x_sha256={parameter_scope['scale32x_scope_sha256']}",
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
                )
                disagreements = sequence_disagreements(
                    reference["generated_token_ids"], candidate["generated_token_ids"]
                )
                gate_match = candidate["response_gate"]["status"] == reference["response_gate"]["status"]
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
                    f"candidate_prompt_complete case={prompt_case.case_id} tokens={len(candidate['generated_token_ids'])} "
                    f"exact={candidate['exact_bf16_sequence_match']} gate={candidate['response_gate']['status']} gate_match={gate_match}",
                )

            aggregate = aggregate_prompts(prompt_results)
            disagreement_set = [
                {
                    "case_id": prompt["case_id"],
                    "token_disagreements": prompt["token_disagreements"],
                    "bf16_response_gate_status": prompt["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": prompt["candidate"]["response_gate"]["status"],
                    "response_gate_outcome_match": prompt["response_gate_outcome_match"],
                }
                for prompt in prompt_results
                if prompt["token_disagreements"] or not prompt["response_gate_outcome_match"]
            ]
            eligible = aggregate["exact_bf16_sequence_match_count"] == 9 and aggregate["response_gate_outcome_match_count"] == 9
            status = "PASS_ELIGIBLE_FOR_FRESH_REVIEW" if eligible else "BLOCKED_EXACT_BF16_DISAGREEMENT"
            result = {
                "schema_version": 1,
                "classification": "qwen_instruct_stage1_lossless_element_scale32x_single_candidate",
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
                "numerical_policy": wrapper["contract"]["numerical_policy"],
                "codec_self_test": codec,
                "static_cost": wrapper["contract"]["static_cost"],
                "parameter_scope": parameter_scope,
                "parameter_manifest": parameter_manifest,
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
                    "consumed_candidate_replayed": False,
                    "alternate_policy_executed": False,
                    "ranking_or_selection_executed": False,
                    "prompt_dataset_or_token_control_used": False,
                    "selector_or_gate_control_used": False,
                    "gate_or_matrix_changed": False,
                    "rtl_mutated": False,
                    "demo_retargeted": False,
                    "ppa_executed": False,
                    "u280_or_stage2_entered": False,
                    "network_access_performed": False,
                },
                "claim_boundary": (
                    "This result establishes only the frozen nine-prompt numerical gate for a "
                    "lossless but metadata-heavy representation. It makes no area, timing, bandwidth, "
                    "RTL, FPGA, full-quality, demo, or product-feasibility claim."
                ),
                "timing": {
                    "completed_at_utc": utc_now(),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                },
            }
            results_path = ATTEMPT_DIR / "results.json"
            write_json(results_path, result)
            log_message(run_log, f"official_attempt_complete status={status} exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 gate_matches={aggregate['response_gate_outcome_match_count']}/9")
            reviewer_input = {
                "schema_version": 1,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "status": "READY_FOR_FRESH_INDEPENDENT_REVIEW",
                "contract": file_record(CONTRACT_PATH),
                "results": file_record(results_path),
                "truthful_eligibility_decision": eligible,
                "review_requirements": [
                    "verify the contract predates execution_started.json",
                    "verify exactly one candidate attempt exists",
                    "verify the candidate is structurally distinct from all three consumed candidates",
                    "verify the exhaustive finite-BF16 codec round-trip",
                    "recompute all nine complete-token and response-gate outcomes",
                    "confirm the severe metadata cost and narrow claim boundary",
                    "accept stage advancement only on 9/9 exact plus 9/9 gate agreement",
                ],
                "independent_acceptance_required": True,
            }
            reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
            write_json(reviewer_path, reviewer_input)
            write_attempt_checksums(
                [CONTRACT_PATH, started_path, run_log_path, results_path, reviewer_path]
            )
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
            failure_path = ATTEMPT_DIR / "failure.json"
            write_json(failure_path, failure)
            log_message(run_log, f"official_attempt_failed classification=EVALUATOR_EXECUTION_FAILURE error_type={type(exc).__name__} error={exc}")
            reviewer_input = {
                "schema_version": 1,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "status": "READY_FOR_FRESH_INDEPENDENT_REVIEW_OF_EXECUTION_FAILURE",
                "contract": file_record(CONTRACT_PATH),
                "failure": file_record(failure_path),
                "truthful_eligibility_decision": None,
                "numerical_correctness_conclusion": None,
                "independent_acceptance_required": True,
            }
            reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
            write_json(reviewer_path, reviewer_input)
            write_attempt_checksums(
                [CONTRACT_PATH, started_path, run_log_path, failure_path, reviewer_path]
            )
            return 5


def verify_checksum_file() -> None:
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    require(sums_path.is_file() and companion.is_file(), "attempt checksum seal is missing")
    digest, filename = companion.read_text(encoding="utf-8").strip().split("  ", 1)
    require(filename == sums_path.name, "checksum companion filename differs")
    require(digest == sha256_file(sums_path), "checksum companion digest differs")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"checksummed artifact is missing: {relative}")
        require(sha256_file(path) == expected, f"checksummed artifact differs: {relative}")


def verify_result() -> dict[str, Any]:
    wrapper = load_verified_contract(require_unconsumed=False)
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    require([path.name for path in attempts] == ["attempt-0001"], "attempt set differs")
    verify_checksum_file()
    failure_path = ATTEMPT_DIR / "failure.json"
    if failure_path.is_file():
        failure = load_json(failure_path)
        require(failure.get("numerical_correctness_conclusion") is None, "execution failure drew a numerical conclusion")
        return {
            "candidate_id": CANDIDATE_ID,
            "attempt_count": 1,
            "execution_status": failure["status"],
            "eligible": None,
            "numerical_correctness_conclusion": None,
        }

    results_path = ATTEMPT_DIR / "results.json"
    reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
    require(results_path.is_file() and reviewer_path.is_file(), "attempt result is incomplete")
    result = load_json(results_path)
    require(result.get("candidate_id") == CANDIDATE_ID, "result candidate differs")
    require(result.get("contract", {}).get("contract_sha256") == wrapper["contract_sha256"], "result contract binding differs")
    require(result.get("matrix", {}).get("sha256") == MATRIX_SHA256, "result matrix differs")
    require(result.get("codec_self_test") == exhaustive_codec_self_test(), "result codec test differs")
    require(len(result.get("prompts", [])) == 9, "result prompt count differs")
    recomputed = aggregate_prompts(result["prompts"])
    require(recomputed == result.get("aggregate"), "result aggregate differs")
    disagreements: list[dict[str, Any]] = []
    for prompt in result["prompts"]:
        tokens = sequence_disagreements(
            prompt["bf16"]["generated_token_ids"], prompt["candidate"]["generated_token_ids"]
        )
        require(tokens == prompt["token_disagreements"], f"token disagreement differs: {prompt['case_id']}")
        gate_match = prompt["bf16"]["response_gate"]["status"] == prompt["candidate"]["response_gate"]["status"]
        require(gate_match == prompt["response_gate_outcome_match"], f"gate agreement differs: {prompt['case_id']}")
        if tokens or not gate_match:
            disagreements.append(
                {
                    "case_id": prompt["case_id"],
                    "token_disagreements": tokens,
                    "bf16_response_gate_status": prompt["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": prompt["candidate"]["response_gate"]["status"],
                    "response_gate_outcome_match": gate_match,
                }
            )
    require(disagreements == result["disagreement_set"], "disagreement set differs")
    eligible = recomputed["exact_bf16_sequence_match_count"] == 9 and recomputed["response_gate_outcome_match_count"] == 9
    require(result["eligibility"]["eligible"] is eligible, "eligibility differs")
    require(result["eligibility"]["selected_policy_id"] is None, "policy was selected before review")
    require(result["scope_guards"]["candidate_attempt_count"] == 1, "attempt count guard differs")
    require(result["scope_guards"]["consumed_candidate_replayed"] is False, "consumed candidate replay guard differs")
    reviewer = load_json(reviewer_path)
    require(reviewer.get("truthful_eligibility_decision") is eligible, "reviewer input eligibility differs")
    return {
        "candidate_id": CANDIDATE_ID,
        "attempt_count": 1,
        "execution_status": result["status"],
        "eligible": eligible,
        "exact_sequence_matches": recomputed["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": recomputed["response_gate_outcome_match_count"],
        "disagreement_case_ids": [entry["case_id"] for entry in disagreements],
        "results_sha256": sha256_file(results_path),
        "fresh_reviewer_input_sha256": sha256_file(reviewer_path),
        "checksum_manifest_sha256": sha256_file(ATTEMPT_DIR / "SHA256SUMS"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("self-test", "prepare-contract", "verify-contract", "execute", "verify-result"),
    )
    args = parser.parse_args()
    if args.command == "self-test":
        print("ACE2_LOSSLESS_ELEMENT_SCALE32X_SELF_TEST " + json.dumps(exhaustive_codec_self_test(), sort_keys=True))
        return 0
    if args.command == "prepare-contract":
        wrapper = prepare_contract()
        print(
            "ACE2_STAGE1_LOSSLESS_ELEMENT_SCALE32X_CONTRACT_PREPARED "
            f"candidate={CANDIDATE_ID} matrix={MATRIX_SHA256} contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "verify-contract":
        wrapper = load_verified_contract(require_unconsumed=True)
        print(
            "ACE2_STAGE1_LOSSLESS_ELEMENT_SCALE32X_PREFLIGHT_PASS "
            f"candidate={CANDIDATE_ID} attempts=0 contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "execute":
        return execute_once()
    summary = verify_result()
    print("ACE2_STAGE1_LOSSLESS_ELEMENT_SCALE32X_RESULT_VERIFIED " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
