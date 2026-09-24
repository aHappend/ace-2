#!/usr/bin/env python3
"""V13 exact hybrid constructor, Dynamic Scale32 replay, and quality flow."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import re
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import v10_packed_scale24_per_row_selector_backend as v10
import v12_v10_preserving_mixed_bank_lm_head64_backend as v12
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    SNAPSHOT,
    file_record,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
    write_json,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v13"
MISSION_ID = "9f4ab32999cc"
TASK_ID = "task-6b8d1f4a2c73"

PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V13_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V13_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
RUNNER_PATH = ROOT / "tools/run_option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_stage1.py"
BACKEND_PATH = Path(__file__).resolve()

OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v13"
CONSTRUCTION_MARKER = OFFICIAL_ROOT / "construction_started.json"
CONSTRUCTION_RECOVERY_MARKER = OFFICIAL_ROOT / "construction_recovery_started.json"
CONSTRUCTOR_SELF_TEST_PATH = OFFICIAL_ROOT / "constructor_self_test.json"
CONSTRUCTOR_RECOVERY_SELF_TEST_PATH = OFFICIAL_ROOT / "constructor_self_test_recovery.json"
CANDIDATE_A_PATH = OFFICIAL_ROOT / "candidate_a_manifest.json"
CANDIDATE_B_PATH = OFFICIAL_ROOT / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = OFFICIAL_ROOT / "alias_separation_witness.json"
TRANSFORMER_REPRODUCTION_PATH = OFFICIAL_ROOT / "transformer_v10_exact_reproduction.json"
LM_HEAD_REPRODUCTION_PATH = OFFICIAL_ROOT / "lm_head_v12_exact_reproduction.json"
ACCOUNTING_PATH = OFFICIAL_ROOT / "codec_and_byte_accounting.json"
DYNAMIC_PATH = OFFICIAL_ROOT / "dynamic_preflight.json"
CONTRACT_PATH = OFFICIAL_ROOT / "predeclared_contract.json"
PREFLIGHT_READY_PATH = OFFICIAL_ROOT / "PRODUCT_PREFLIGHT_READY.json"
PREFLIGHT_NO_GO_PATH = OFFICIAL_ROOT / "PRODUCT_PREFLIGHT_NO_GO.json"
QUALITY_DIR = OFFICIAL_ROOT / "quality-campaign-0001"
FROZEN_MATRIX_PATH = QUALITY_DIR / "frozen_matrix.json"
QUALITY_MARKER = QUALITY_DIR / "execution_started.json"
RAW_OUTPUTS_PATH = QUALITY_DIR / "raw_outputs.jsonl"
TOKEN_LATENCY_PATH = QUALITY_DIR / "token_and_latency_results.json"
RUBRIC_PATH = QUALITY_DIR / "rubric_results.json"
QUALITY_RESULT_PATH = QUALITY_DIR / "RESULT.json"
QUALITY_SUMS_PATH = QUALITY_DIR / "SHA256SUMS"
FRESH_REVIEWER_SUBMISSION_PATH = OFFICIAL_ROOT / "fresh_reviewer_submission.json"

V10_RECONSTRUCTION_PATH = ROOT / "build/stage1-option-b-alias-safe-packed-scale24-per-row-selector-dual-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v10/preattempt/model_only_v9_relative_reconstruction.json"
V10_RECONSTRUCTION_SHA256 = "47a276395a2cf7e574f1d15ec61942ad7c60679ff0129c31ead3d75b944938de"
V12_RECONSTRUCTION_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-preserving-mixed-four-bank-lm-head64-grouped-dual-codebook-packed-scale24-w4-grouped-dynamic-scale32-w4a8-v12/preattempt/model_only_v10_relative_reconstruction.json"
V12_RECONSTRUCTION_SHA256 = "21f0b765848c0d8efaae0371730680f7fa7ae9919c287375e4f3ca4ceb008773"
V12_TERMINAL_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-preserving-mixed-four-bank-lm-head64-grouped-dual-codebook-packed-scale24-w4-grouped-dynamic-scale32-w4a8-v12/PREATTEMPT_NO_GO.json"
V12_TERMINAL_SHA256 = "1f8f9ddbba7c756811bf216ec1aadf474aee665c7a9677fe84bb888453ae3d5c"
V12_REVIEW_PATH = ROOT / "research/raw/specification/v10-preserving-mixed-bank-lm-head64-v12-review-done-20260807T051730Z.json"
V12_REVIEW_SHA256 = "2c136a7434639d8c158214be578b9b1963ad3ad293410cdcdb093c81c22676b4"
V10_CANDIDATE_MANIFEST_PATH = ROOT / "build/stage1-option-b-alias-safe-packed-scale24-per-row-selector-dual-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v10/preattempt/candidate_b_manifest.json"
V10_CANDIDATE_MANIFEST_SHA256 = "59bb66df93445ab03a13281775d36d041adbd9f2069f052558f4fe8582f2df10"
V12_CANDIDATE_MANIFEST_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-preserving-mixed-four-bank-lm-head64-grouped-dual-codebook-packed-scale24-w4-grouped-dynamic-scale32-w4a8-v12/preattempt/candidate_b_manifest.json"
V12_CANDIDATE_MANIFEST_SHA256 = "af3ae528c40b0c943c4203173f507e78ade30acb32d906dc40a03570a01dcac1"
V12_CONTRACT_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-preserving-mixed-four-bank-lm-head64-grouped-dual-codebook-packed-scale24-w4-grouped-dynamic-scale32-w4a8-v12/predeclared_contract.json"
V12_CONTRACT_SHA256 = "b008888c04d12c306cfde377f9a466b2357f1822ea7216c4a079c112822f174f"
LM_HEAD_V12_RECONSTRUCTION_SHA256 = "ff463c0a71035c4dc8305c9fe47c96fed7e06a96da0a04cc37b5477c6ab7df76"
SOURCE_EMBEDDING_SHA256 = "4e96b0df6d274768cbb7e72404011853d23349999b658dc2f4dfb3c431ea223f"
SOURCE_CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
BASE_SCALES_PATH = CALIBRATION_DIR / "derived_scales.json"
RTL_SHELL_PATH = ROOT / "rtl/ace2_shell.sv"

LM_HEAD_ROWS_PER_GROUP = 2374
EXPECTED_TRANSFORMER_TENSORS = 168
EXPECTED_NAMED_DYNAMIC_EVENTS = 312
MAX_NEW_TOKENS = 32
TERMINATION_TOKEN_IDS = (151643, 151645)
CONSTRUCTION_TORCH_THREADS = 1
QUALITY_TORCH_THREADS = 32


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()


def _verify_bound_file(path: Path, expected_sha256: str) -> None:
    require(path.is_file(), f"bound V13 input is missing: {path.relative_to(ROOT)}")
    require(sha256_file(path) == expected_sha256, f"bound V13 input hash differs: {path.relative_to(ROOT)}")


def _module_family(module_name: str) -> str:
    return v12.reference.module_family(module_name)


def _literal_bf16_sse(source: Tensor, reconstruction: Tensor) -> float:
    difference = source.to(torch.float64) - reconstruction.to(torch.float64)
    return float(torch.sum(difference * difference))


def construct_v13_state(
    source: Tensor,
    module_name: str,
    *,
    lm_head_group_size: int = LM_HEAD_ROWS_PER_GROUP,
) -> dict[str, Any]:
    """Construct one V13 tensor strictly from the frozen V10/V12 kernels."""

    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    seed = v10.construct_packed_scale24_state(source, module_name)
    if _module_family(module_name) == "lm_head":
        state = v12._construct_grouped_lm_head_from_seed(source, module_name, seed, lm_head_group_size)
        state["v13_source"] = "isolated_v12_grouped_lm_head"
        return state

    reconstruction = v10._decode(seed)
    state = dict(seed)
    state.update(
        {
            "mode": "transformer_v10_dual_bank_preserved",
            "v13_source": "sealed_v10_transformer",
            "bank_count": 2,
            "selector_bits": 1,
            "initial_v10_literal_bf16_sse": _literal_bf16_sse(source, reconstruction),
            "final_v13_literal_bf16_sse": _literal_bf16_sse(source, reconstruction),
        }
    )
    return state


def decode_v13_state(state: dict[str, Any]) -> Tensor:
    if state["mode"] == "transformer_v10_dual_bank_preserved":
        return v10._decode(state)
    require(state["mode"] == "lm_head_grouped_dual_bank", "V13 state mode differs")
    return v12._decode_state(state)


def state_hashes(state: dict[str, Any]) -> dict[str, str]:
    return {
        "payload_sha256": hashlib.sha256(state["payload"]).hexdigest(),
        "packed_scale24_stream_sha256": hashlib.sha256(state["packed_scale24"]).hexdigest(),
        "codebook_stream_sha256": hashlib.sha256(state["codebook_raw"]).hexdigest(),
        "selector_stream_sha256": hashlib.sha256(state["selector_raw"]).hexdigest(),
        "reconstruction_bf16_sha256": v10.v7.v3.tensor_sha256(decode_v13_state(state)),
        "iteration_trace_sha256": canonical_sha256(state["iteration_trace"]),
    }


def storage_closure() -> dict[str, Any]:
    standard_decoder_layer = {
        "weight_count": 14_909_440,
        "payload_bytes": 7_454_720,
        "metadata_bytes": 55_136,
        "total_weight_bytes": 7_509_856,
    }
    lm_head = {
        "weight_count": 136_134_656,
        "payload_bytes": 68_067_328,
        "metadata_bytes": 603_088,
        "total_weight_bytes": 68_670_416,
    }
    whole_model = {
        "weight_count": standard_decoder_layer["weight_count"] * 24 + lm_head["weight_count"],
        "payload_bytes": standard_decoder_layer["payload_bytes"] * 24 + lm_head["payload_bytes"],
        "metadata_bytes": standard_decoder_layer["metadata_bytes"] * 24 + lm_head["metadata_bytes"],
        "total_weight_bytes": standard_decoder_layer["total_weight_bytes"] * 24 + lm_head["total_weight_bytes"],
    }
    for record in (standard_decoder_layer, lm_head, whole_model):
        record["exact_bits_per_weight"] = record["total_weight_bytes"] * 8 / record["weight_count"]
    require(whole_model["weight_count"] == 493_961_216, "V13 whole-model weight count differs")
    require(whole_model["payload_bytes"] == 246_980_608, "V13 payload bytes differ")
    require(whole_model["metadata_bytes"] == 1_926_352, "V13 metadata bytes differ")
    require(whole_model["total_weight_bytes"] == 248_906_960, "V13 total weight bytes differ")
    require(whole_model["exact_bits_per_weight"] == 4.031198433198448, "V13 bits/weight differs")
    return {
        "standard_decoder_layer": standard_decoder_layer,
        "lm_head": lm_head,
        "whole_model": whole_model,
    }


def _verify_frozen_authority() -> dict[str, Any]:
    v12.reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    v12.reference.verify_companion(TASK_PATH, TASK_COMPANION)
    plan = load_json(PLAN_PATH)
    task = load_json(TASK_PATH)
    require(plan["candidate_id"] == CANDIDATE_ID and plan["status"] == "frozen_not_started", "frozen V13 plan differs")
    require(task["candidate_id"] == CANDIDATE_ID and task["task_id"] == TASK_ID, "frozen V13 task differs")
    require(task["required_constructor"]["determinism"].startswith("two clean independent constructions"), "V13 determinism contract differs")
    require(task["required_self_tests"][8].startswith("run all 312 named grouped Dynamic Scale32 events"), "V13 Dynamic Scale32 contract differs")
    _verify_bound_file(V10_RECONSTRUCTION_PATH, V10_RECONSTRUCTION_SHA256)
    _verify_bound_file(V12_RECONSTRUCTION_PATH, V12_RECONSTRUCTION_SHA256)
    _verify_bound_file(V12_TERMINAL_PATH, V12_TERMINAL_SHA256)
    _verify_bound_file(V12_REVIEW_PATH, V12_REVIEW_SHA256)

    v10_report = load_json(V10_RECONSTRUCTION_PATH)
    v12_report = load_json(V12_RECONSTRUCTION_PATH)
    require(len(v10_report["tensors"]) == EXPECTED_TRANSFORMER_TENSORS + 1, "sealed V10 tensor count differs")
    require(sum(item["module"] != "lm_head" for item in v10_report["tensors"]) == EXPECTED_TRANSFORMER_TENSORS, "sealed V10 transformer count differs")
    lm_head_records = [item for item in v12_report["tensors"] if item["module"] == "lm_head"]
    require(len(lm_head_records) == 1, "sealed V12 lm_head record count differs")
    require(lm_head_records[0]["reconstruction_bf16_sha256"] == LM_HEAD_V12_RECONSTRUCTION_SHA256, "sealed V12 lm_head reconstruction differs")
    terminal = load_json(V12_TERMINAL_PATH)
    require(terminal["status"] == "PREATTEMPT_NO_GO" and terminal["attempts_consumed"] == 0, "V12 terminal state differs")
    return {
        "plan_sha256": sha256_file(PLAN_PATH),
        "task_sha256": sha256_file(TASK_PATH),
        "sealed_v10_reconstruction_sha256": sha256_file(V10_RECONSTRUCTION_PATH),
        "sealed_v12_reconstruction_sha256": sha256_file(V12_RECONSTRUCTION_PATH),
        "sealed_v12_lm_head_reconstruction_bf16_sha256": lm_head_records[0]["reconstruction_bf16_sha256"],
        "v12_terminal_sha256": sha256_file(V12_TERMINAL_PATH),
        "v12_review_sha256": sha256_file(V12_REVIEW_PATH),
    }


def combined_self_test(*, allow_recovery: bool = False) -> dict[str, Any]:
    require(Path(v12.reference.require_project_python()["entrypoint"]).as_posix() == ".venv/bin/python", "project Python contract differs")
    if allow_recovery:
        require(OFFICIAL_ROOT.is_dir() and PREFLIGHT_NO_GO_PATH.is_file(), "V13 recovery self-test requires the preserved preflight failure")
        require(not QUALITY_MARKER.exists(), "V13 recovery self-test may not follow quality execution")
    else:
        require(not OFFICIAL_ROOT.exists(), "V13 official namespace must remain absent during kernel self-test")
    require(not QUALITY_MARKER.exists(), "V13 quality marker exists before kernel self-test")
    authority = _verify_frozen_authority()

    lanes = torch.arange(896, dtype=torch.int64)
    row0 = ((lanes % 47) - 23).to(torch.float64) / 256.0
    transformer_fixture = torch.stack(
        (
            row0,
            torch.where((lanes % 5) == 0, row0 * 0.25, row0 * 1.5),
            torch.where((lanes % 7) < 3, -row0 * 0.75, row0 * 0.125),
            torch.zeros_like(row0),
        )
    ).to(torch.bfloat16)
    transformer_a = construct_v13_state(transformer_fixture, "self_test.q_proj")
    transformer_b = construct_v13_state(transformer_fixture, "self_test.q_proj")
    transformer_geometry = v10._validate_state(transformer_a)
    require(transformer_a["v13_source"] == "sealed_v10_transformer", "V13 transformer source differs")
    require(state_hashes(transformer_a) == state_hashes(transformer_b), "independent V13 transformer hashes differ")
    require(transformer_a["iteration_trace"] == transformer_b["iteration_trace"], "independent V13 transformer traces differ")
    require(v10._state_hashes(transformer_a)["reconstruction_bf16_sha256"] == state_hashes(transformer_a)["reconstruction_bf16_sha256"], "V13 transformer is not exact V10")
    require(transformer_a["initial_v10_literal_bf16_sse"] == transformer_a["final_v13_literal_bf16_sse"], "V13 transformer changed V10 reconstruction")

    grouped_fixture = torch.stack(
        tuple(
            torch.where(
                (lanes % (5 + row)) < (2 + row % 3),
                row0 * (0.5 + row / 8.0),
                -row0 * (0.25 + row / 16.0),
            )
            for row in range(64)
        )
    ).to(torch.bfloat16)
    lm_head_a = construct_v13_state(grouped_fixture, "self_test.lm_head", lm_head_group_size=32)
    lm_head_b = construct_v13_state(grouped_fixture, "self_test.lm_head", lm_head_group_size=32)
    lm_head_geometry = {
        "rows": int(lm_head_a["shape"][0]),
        "input_features": int(lm_head_a["shape"][1]),
        "input_block_count": int(lm_head_a["input_block_count"]),
        "group_count": int(lm_head_a["group_count"]),
        "group_size": int(lm_head_a["group_size"]),
        "selector_bits": int(lm_head_a["selector_bits"]),
        "codebook_bytes": len(lm_head_a["codebook_raw"]),
        "selector_bytes": len(lm_head_a["selector_raw"]),
    }
    require(lm_head_a["v13_source"] == "isolated_v12_grouped_lm_head", "V13 lm_head source differs")
    require(lm_head_a["mode"] == "lm_head_grouped_dual_bank", "V13 lm_head mode differs")
    require(lm_head_a["group_count"] == 2 and lm_head_a["group_size"] == 32, "V13 synthetic lm_head group geometry differs")
    require(lm_head_a["group_selector_metadata_bits"] == 0, "V13 lm_head added group-selector metadata")
    require(state_hashes(lm_head_a) == state_hashes(lm_head_b), "independent V13 lm_head hashes differ")
    require(lm_head_a["iteration_trace"] == lm_head_b["iteration_trace"], "independent V13 lm_head traces differ")
    require(lm_head_a["final_v12_literal_bf16_sse"] <= lm_head_a["initial_v10_literal_bf16_sse"], "V13 lm_head regressed against V10 seed")
    require(len(lm_head_a["codebook_raw"]) == 2 * 7 * 2 * 16, "V13 synthetic lm_head codebook bytes differ")

    storage = storage_closure()
    require((allow_recovery or not OFFICIAL_ROOT.exists()) and not QUALITY_MARKER.exists(), "V13 kernel self-test changed official state")
    return {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "environment": {
            **v12.reference.require_project_python(),
            "platform": platform.platform(),
            "packages": {"torch": importlib.metadata.version("torch")},
            "backend": "exact V10 transformer plus isolated V12 grouped-lm_head constructor kernel",
        },
        "runner_sha256": sha256_file(RUNNER_PATH),
        "backend_sha256": sha256_file(BACKEND_PATH),
        "authority": authority,
        "checks": {
            "frozen_plan_task_and_predecessor_hashes_verified": True,
            "sealed_v10_tensor_count_169_with_168_transformers": True,
            "sealed_v12_lm_head_reconstruction_hash_verified": True,
            "transformer_constructor_is_exact_v10": True,
            "isolated_lm_head_constructor_is_v12_grouped_dual_bank": True,
            "two_independent_transformer_constructions_match": True,
            "two_independent_lm_head_constructions_match": True,
            "packed_scale24_codebook_selector_and_reconstruction_hashes_match": True,
            "source_embedding_and_lm_head_alias_test_scope": "not exercised by synthetic tensor kernel",
            "exact_v13_storage_closure": True,
            "official_v13_namespace_changed": False,
            "construction_recovery_mode": allow_recovery,
            "quality_campaign_consumed": False,
            "dynamic_scale32_312_event_closure": False,
        },
        "transformer_fixture": {
            "shape": list(transformer_fixture.shape),
            "geometry": transformer_geometry,
            "literal_bf16_sse": transformer_a["final_v13_literal_bf16_sse"],
            **state_hashes(transformer_a),
        },
        "lm_head_fixture": {
            "shape": list(grouped_fixture.shape),
            "geometry": lm_head_geometry,
            "initial_v10_literal_bf16_sse": lm_head_a["initial_v10_literal_bf16_sse"],
            "final_v12_literal_bf16_sse": lm_head_a["final_v12_literal_bf16_sse"],
            **state_hashes(lm_head_a),
        },
        "storage": storage,
        "remaining_required_closure": {
            "full_model_independent_constructions": 2,
            "transformer_exact_match_count": EXPECTED_TRANSFORMER_TENSORS,
            "dynamic_scale32_named_event_count": EXPECTED_NAMED_DYNAMIC_EVENTS,
            "quality_campaign": "quality-campaign-0001 remains unstarted",
        },
        "claim_boundary": "synthetic constructor-kernel and frozen-input self-test only; no full-model, 312-event, product-quality, accelerator, RTL, synthesis, timing, area, or Fresh Reviewer conclusion",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, value: Any) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_file_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_file(), f"bound artifact missing: {record['path']}")
    require(path.stat().st_size == int(record["bytes"]), f"bound artifact size differs: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"bound artifact hash differs: {record['path']}")


def _helper_paths() -> dict[str, Path]:
    return {
        "v13_runner": RUNNER_PATH,
        "v13_backend": BACKEND_PATH,
        "v10_runner": Path(v10.reference.__file__).resolve(),
        "v10_backend": Path(v10.__file__).resolve(),
        "v12_runner": Path(v12.reference.__file__).resolve(),
        "v12_backend": Path(v12.__file__).resolve(),
        "dynamic_policy_helper": Path(v10.v7.v3.__file__).resolve(),
        "grouped_scale32_helper": Path(v10.v7.grouped.__file__).resolve(),
        "source_identity_helper": ROOT / "tools/qwen_instruct_option_b.py",
        "generation_contract_helper": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
        "rtl_shell_unchanged_witness": RTL_SHELL_PATH,
    }


def _verify_full_authority_before_construction() -> dict[str, Any]:
    authority = _verify_frozen_authority()
    _verify_bound_file(V10_CANDIDATE_MANIFEST_PATH, V10_CANDIDATE_MANIFEST_SHA256)
    _verify_bound_file(V12_CANDIDATE_MANIFEST_PATH, V12_CANDIDATE_MANIFEST_SHA256)
    _verify_bound_file(V12_CONTRACT_PATH, V12_CONTRACT_SHA256)
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    require(BASE_SCALES_PATH.is_file(), "Dynamic Scale32 base scales are missing")
    v12_closure = v12.load_verified_closure(require_unconsumed=False)
    require(v12_closure["status"] == "PREATTEMPT_NO_GO", "sealed V12 terminal status differs")
    predecessor_contract = load_json(V12_CONTRACT_PATH)["contract"]
    for key in (
        "runner_source",
        "scalable_backend_source",
        "reviewed_v10_runner",
        "reviewed_v10_backend",
        "reviewed_quality_helper",
        "reviewed_grouped_dynamic_runner",
        "identity_helper_source",
    ):
        _verify_file_record(predecessor_contract["artifacts"][key])
    helpers = {name: file_record(path) for name, path in _helper_paths().items()}
    return {
        "authority": authority,
        "source_contract": file_record(SOURCE_CONTRACT_PATH),
        "source_model": source_identity,
        "base_scales": file_record(BASE_SCALES_PATH),
        "predecessor_contract": file_record(V12_CONTRACT_PATH),
        "predecessor_closure": v12_closure,
        "helpers": helpers,
    }


def _selector_padding_zero(raw: bytes, selector_count: int, bits_per_selector: int) -> bool:
    used_bits = selector_count * bits_per_selector
    logical_bytes = (used_bits + 7) // 8
    if logical_bytes:
        used_in_last = used_bits & 7
        if used_in_last:
            high_mask = 0xFF ^ ((1 << used_in_last) - 1)
            if raw[logical_bytes - 1] & high_mask:
                return False
    return all(value == 0 for value in raw[logical_bytes:])


def _decoded_chunks(state: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    rows, width = (int(value) for value in state["shape"])
    if state["mode"] == "lm_head_grouped_dual_bank":
        return tuple((start, start + int(state["group_size"])) for start in range(0, rows, int(state["group_size"])))
    return tuple(v10.v7.v4._row_chunks(rows, width))


def _decode_chunk(state: dict[str, Any], row_start: int, row_stop: int) -> Tensor:
    if state["mode"] == "lm_head_grouped_dual_bank":
        return v12._decoded_chunk(state, row_start, row_stop)
    require(state["mode"] == "transformer_v10_dual_bank_preserved", "V13 transformer mode differs")
    return v10.v9._decoded_chunk_v9(
        state["final_records"],
        state["final_assignments"],
        state["final_codebooks"],
        state["final_selectors"],
        1,
        row_start,
        row_stop,
    )


def _materialize_and_authenticate(
    module: nn.Linear,
    source: Tensor,
    state: dict[str, Any],
    reference_weight: dict[str, Any],
    reference_reconstruction: dict[str, Any],
    reference_trace: dict[str, Any],
) -> dict[str, Any]:
    name = reference_weight["module"]
    rows, width = (int(value) for value in source.shape)
    is_lm_head = name == "lm_head"
    source_sha256 = v10.v7.v3.tensor_sha256(source)
    geometry = v12._validate_full_state(state) if is_lm_head else v10._validate_state(state)
    require(v10.v7.v3.pack_codebook_indices(state["final_assignments"]) == state["payload"], f"payload codec differs: {name}")
    expected_codebook_raw = (
        v12._codebook_raw_grouped(state["final_codebooks"])
        if is_lm_head
        else v10.v9._dual_codebook_raw(state["final_codebooks"])
    )
    require(expected_codebook_raw == state["codebook_raw"], f"codebook codec differs: {name}")
    selector_count = rows * int(state["input_block_count"])
    require(
        _selector_padding_zero(state["selector_raw"], selector_count, int(state["selector_bits"])),
        f"selector padding differs: {name}",
    )

    reconstruction_digest = hashlib.sha256()
    reconstruction_sse = 0.0
    with torch.no_grad():
        for row_start, row_stop in _decoded_chunks(state):
            decoded = _decode_chunk(state, row_start, row_stop)
            difference = source[row_start:row_stop].to(torch.float64) - decoded.to(torch.float64)
            reconstruction_sse += float(torch.sum(difference * difference))
            reconstruction_digest.update(v10.v7.v3.raw_tensor_bytes(decoded))
            module.weight[row_start:row_stop].copy_(decoded)
    reconstruction_sha256 = reconstruction_digest.hexdigest()
    require(reconstruction_sha256 == v10.v7.v3.tensor_sha256(module.weight), f"materialized hash differs: {name}")

    component_key = "mixed_codebook" if is_lm_head else "dual_codebook"
    expected_components = reference_weight["components"]
    actual_hashes = {
        "payload": hashlib.sha256(state["payload"]).hexdigest(),
        "packed_scale24": hashlib.sha256(state["packed_scale24"]).hexdigest(),
        "codebook": hashlib.sha256(state["codebook_raw"]).hexdigest(),
        "selector": hashlib.sha256(state["selector_raw"]).hexdigest(),
        "reconstruction": reconstruction_sha256,
        "iteration_trace": canonical_sha256(state["iteration_trace"]),
        "source": source_sha256,
    }
    matches = {
        "source_bf16": source_sha256 == reference_trace["source_bf16_sha256"],
        "payload": actual_hashes["payload"] == expected_components["payload"]["sha256"],
        "packed_scale24": actual_hashes["packed_scale24"] == expected_components["packed_scale24"]["sha256"],
        "codebook": actual_hashes["codebook"] == expected_components[component_key]["sha256"],
        "selector": actual_hashes["selector"] == expected_components["selector"]["sha256"],
        "reconstruction": reconstruction_sha256 == reference_reconstruction["reconstruction_bf16_sha256"],
        "iteration_trace": actual_hashes["iteration_trace"] == reference_weight["iteration_trace_sha256"],
    }
    require(all(matches.values()), f"sealed component authentication differs: {name}: {matches}")
    require(actual_hashes["source"] == source_sha256, f"source authentication differs: {name}")
    if is_lm_head:
        require(reconstruction_sha256 == LM_HEAD_V12_RECONSTRUCTION_SHA256, "V13 lm_head reconstruction differs")
        require(int(state["group_count"]) == 64 and int(state["group_size"]) == LM_HEAD_ROWS_PER_GROUP, "V13 lm_head groups differ")

    return {
        "module": name,
        "name": f"{name}.weight",
        "shape": [rows, width],
        "scope": "separated_lm_head" if is_lm_head else "transformer",
        "source": "isolated_v12_grouped_lm_head" if is_lm_head else "sealed_v10_transformer",
        "geometry": geometry,
        "components": {
            "payload": {"bytes": len(state["payload"]), "sha256": actual_hashes["payload"]},
            "packed_scale24": {
                "bytes": len(state["packed_scale24"]),
                "sha256": actual_hashes["packed_scale24"],
                "record_count": rows,
                "bytes_per_record": 3,
                "reserved_high_byte_restored_zero": True,
            },
            "codebook": {"bytes": len(state["codebook_raw"]), "sha256": actual_hashes["codebook"]},
            "selector": {
                "bytes": len(state["selector_raw"]),
                "sha256": actual_hashes["selector"],
                "selector_bits": int(state["selector_bits"]),
                "padding_bits_and_alignment_bytes_zero": True,
            },
        },
        "reconstruction": {
            "bf16_sha256": reconstruction_sha256,
            "literal_bf16_sse": reconstruction_sse,
        },
        "source_bf16_sha256": source_sha256,
        "iteration_trace_sha256": actual_hashes["iteration_trace"],
        "authentication_matches": matches,
        "all_authentication_matches": True,
        "codec_round_trips": {
            "four_bit_payload": True,
            "packed_scale24": True,
            "codebooks": True,
            "selectors": True,
            "selector_padding": True,
            "group_transitions": True if is_lm_head else None,
        },
    }


def _construct_full_model() -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    embedding = model.model.embed_tokens.weight
    lm_head = model.lm_head.weight
    require(model.config.tie_word_embeddings is True, "source tie_word_embeddings differs")
    require(lm_head is embedding and lm_head.data_ptr() == embedding.data_ptr(), "source lm_head alias differs")
    source_object_id = id(embedding)
    source_pointer = embedding.data_ptr()
    source_shape = list(embedding.shape)
    source_dtype = str(embedding.dtype)
    source_embedding_sha256 = v10.v7.v3.tensor_sha256(embedding)
    require(source_embedding_sha256 == SOURCE_EMBEDDING_SHA256, "source embedding hash differs")
    detached_lm_head = lm_head.detach().clone()
    require(v10.v7.v3.tensor_sha256(detached_lm_head) == source_embedding_sha256, "detached lm_head bytes differ")
    model.lm_head.weight = nn.Parameter(detached_lm_head, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding, "lm_head parameter object remains aliased")
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head storage remains aliased")

    v10_manifest = load_json(V10_CANDIDATE_MANIFEST_PATH)
    v10_reconstruction = load_json(V10_RECONSTRUCTION_PATH)
    v12_manifest = load_json(V12_CANDIDATE_MANIFEST_PATH)
    v12_reconstruction = load_json(V12_RECONSTRUCTION_PATH)
    v10_weights = {item["module"]: item for item in v10_manifest["weight_manifest"]}
    v10_recon = {item["module"]: item for item in v10_reconstruction["tensors"]}
    v10_traces = {item["module"]: item for item in v10_manifest["iteration_trace_manifest"]}
    v12_weights = {item["module"]: item for item in v12_manifest["weight_manifest"]}
    v12_recon = {item["module"]: item for item in v12_reconstruction["tensors"]}
    v12_traces = {item["module"]: item for item in v12_manifest["iteration_trace_manifest"]}

    expected_transformers = [record["module"] for record in v10.v7.grouped.expected_transformer_tensors()]
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen Linear tensor count differs")
    require([name for name, _module in modules if name != "lm_head"] == expected_transformers, "transformer tensor order differs")

    records: list[dict[str, Any]] = []
    transformer_records: list[dict[str, Any]] = []
    lm_head_records: list[dict[str, Any]] = []
    streams = {name: hashlib.sha256() for name in ("payload", "packed_scale24", "codebook", "selector", "reconstruction")}
    totals = {"weight_count": 0, "payload": 0, "packed_scale24": 0, "codebook": 0, "selector": 0}
    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V13_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source = module.weight.detach()
        state = construct_v13_state(source, name)
        if name == "lm_head":
            record = _materialize_and_authenticate(module, source, state, v12_weights[name], v12_recon[name], v12_traces[name])
            lm_head_records.append(record)
        else:
            record = _materialize_and_authenticate(module, source, state, v10_weights[name], v10_recon[name], v10_traces[name])
            transformer_records.append(record)
        for component in ("payload", "packed_scale24", "codebook", "selector"):
            raw_key = {"payload": "payload", "packed_scale24": "packed_scale24", "codebook": "codebook_raw", "selector": "selector_raw"}[component]
            raw = state[raw_key]
            streams[component].update(name.encode() + b"\0" + raw)
            totals[component] += len(raw)
        streams["reconstruction"].update(name.encode() + b"\0" + bytes.fromhex(record["reconstruction"]["bf16_sha256"]))
        totals["weight_count"] += int(source.numel())
        records.append(record)
        print(f"V13_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic()-started:.3f}", flush=True)
        del state
        gc.collect()

    require(len(transformer_records) == 168 and len(lm_head_records) == 1, "V13 tensor-family counts differ")
    expected_totals = {
        "weight_count": 493_961_216,
        "payload": 246_980_608,
        "packed_scale24": 1_368_192,
        "codebook": 75_776,
        "selector": 482_384,
    }
    require(totals == expected_totals, f"V13 exact storage totals differ: {totals}")
    storage = storage_closure()
    require(totals["packed_scale24"] + totals["codebook"] + totals["selector"] == storage["whole_model"]["metadata_bytes"], "metadata byte closure differs")
    require(totals["payload"] + storage["whole_model"]["metadata_bytes"] == storage["whole_model"]["total_weight_bytes"], "total weight byte closure differs")

    require(id(model.model.embed_tokens.weight) == source_object_id, "embedding parameter object changed")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding format changed")
    final_embedding_sha256 = v10.v7.v3.tensor_sha256(embedding)
    require(final_embedding_sha256 == source_embedding_sha256 == SOURCE_EMBEDDING_SHA256, "embedding bytes changed")
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head storage was re-aliased")
    state_manifest = v10.v7.v3.state_manifest(model)
    record_hash = canonical_sha256(records)
    identity = {
        "embedding_bf16_sha256": final_embedding_sha256,
        "weight_manifest_sha256": record_hash,
        "candidate_state_manifest_sha256": state_manifest["candidate_state_manifest_sha256"],
        **{f"{name}_stream_sha256": digest.hexdigest() for name, digest in streams.items()},
    }
    core_manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "source_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "config_tie_word_embeddings_after_split": False,
        "module_counts": {"transformer_v10": 168, "lm_head_v12_grouped": 1, "total_linear_tensors": 169},
        "identity": identity,
        "storage": storage,
        "weight_manifest": records,
        "state_manifest": state_manifest,
    }
    witness = {
        "source_embedding_bf16_sha256": source_embedding_sha256,
        "post_construction_embedding_bf16_sha256": final_embedding_sha256,
        "source_lm_head_and_embedding_same_parameter_object": True,
        "source_lm_head_and_embedding_same_storage": True,
        "post_split_different_parameter_objects": model.lm_head.weight is not embedding,
        "post_split_different_storage": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
        "embedding_parameter_object_preserved": id(model.model.embed_tokens.weight) == source_object_id,
        "embedding_storage_preserved": model.model.embed_tokens.weight.data_ptr() == source_pointer,
        "embedding_bytes_preserved": final_embedding_sha256 == source_embedding_sha256,
        "lm_head_storage_separate_after_full_construction": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
    }
    transformer_report = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_168_OF_168_EXACT_V10_REPRODUCTION",
        "reference": file_record(V10_CANDIDATE_MANIFEST_PATH),
        "tensor_count": len(transformer_records),
        "exact_match_count": sum(item["all_authentication_matches"] for item in transformer_records),
        "all_168_payload_metadata_reconstruction_and_trace_hashes_match": all(item["all_authentication_matches"] for item in transformer_records),
        "tensors": transformer_records,
    }
    lm_head_report = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_EXACT_V12_LM_HEAD64_REPRODUCTION",
        "reference": file_record(V12_CANDIDATE_MANIFEST_PATH),
        "tensor_count": 1,
        "reconstruction_bf16_sha256": lm_head_records[0]["reconstruction"]["bf16_sha256"],
        "exact_payload_metadata_reconstruction_and_trace_hash_match": lm_head_records[0]["all_authentication_matches"],
        "tensor": lm_head_records[0],
    }
    accounting = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_EXACT_CODEC_AND_BYTE_CLOSURE",
        "component_bytes": totals,
        "metadata_bytes": totals["packed_scale24"] + totals["codebook"] + totals["selector"],
        "total_weight_bytes": totals["payload"] + totals["packed_scale24"] + totals["codebook"] + totals["selector"],
        "storage_closure": storage,
        "codec_tensor_count": len(records),
        "packed_scale24_round_trip_count": len(records),
        "selector_round_trip_and_zero_padding_count": len(records),
        "codebook_round_trip_count": len(records),
        "authentication_record_count": len(records),
        "lm_head_group_count": 64,
        "maximum_activation_transport_metadata_bytes_per_decoder_layer_token": 832,
        "maximum_incremental_sram_bytes": 2368,
        "maximum_live_weight_metadata_cache_bytes": 192,
        "abstract_streaming_memory_boundary_bits": 128,
        "public_ace2_shell_parameter_count": 12,
        "public_ace2_shell_port_count": 64,
        "public_interface_changed": False,
        "rtl_shell_source": file_record(RTL_SHELL_PATH),
    }
    return model, core_manifest, witness, transformer_report, lm_head_report, accounting


def _candidate_worker(output_dir: str) -> None:
    torch.set_num_threads(CONSTRUCTION_TORCH_THREADS)
    torch.set_num_interop_threads(1)
    directory = Path(output_dir)
    print("V13_CANDIDATE_A_WORKER_START", flush=True)
    model, manifest, witness, transformer, lm_head, accounting = _construct_full_model()
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "witness.json", witness)
    write_json(directory / "transformer.json", transformer)
    write_json(directory / "lm_head.json", lm_head)
    write_json(directory / "accounting.json", accounting)
    del model
    gc.collect()
    print("V13_CANDIDATE_A_WORKER_DONE", flush=True)


def _token_ids_sha256(token_ids: list[int]) -> str:
    raw = b"".join(int(value).to_bytes(4, "little", signed=False) for value in token_ids)
    return hashlib.sha256(raw).hexdigest()


def _tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> Tensor:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(input_ids, Tensor) and input_ids.ndim == 2 and input_ids.shape[0] == 1, "chat tokenization shape differs")
    return input_ids


def _replay_dynamic_scale32(model: nn.Module, tokenizer: Any) -> dict[str, Any]:
    require(not QUALITY_MARKER.exists(), "quality campaign started before Dynamic Scale32 replay")
    calibration = load_json(CALIBRATION_DIR / "calibration_contract.json")
    messages = calibration["chat"]["prompts"][0]["messages"]
    input_ids = _tokenize_messages(tokenizer, messages)
    policy = v10.v7.v3.AuditedDynamicPolicy(load_json(BASE_SCALES_PATH))
    policy.install(model)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=False)
        require(bool(torch.isfinite(output.logits).all()), "Dynamic Scale32 replay produced non-finite logits")
        summary = policy.audited_summary(1)
    finally:
        policy.uninstall()
    require(summary["event_count"] == EXPECTED_NAMED_DYNAMIC_EVENTS, "Dynamic Scale32 event count differs")
    require(summary["all_312_layer_event_names_present"] is True, "Dynamic Scale32 named-event closure differs")
    require(summary["saturation_count"] == 0 and summary["clipping_count"] == 0, "Dynamic Scale32 clipping differs")
    require(summary["reserved_minus_128_produced"] is False, "Dynamic Scale32 produced reserved -128")
    require(summary["transport_metadata_bytes_per_decoder_layer_token"] <= 832, "Dynamic Scale32 metadata cap differs")
    token_ids = [int(value) for value in input_ids[0].tolist()]
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_ALL_312_DYNAMIC_SCALE32_EVENTS",
        "quality_campaign_started": False,
        "input_source": "frozen calibration contract prompt 0",
        "input_token_count": len(token_ids),
        "input_token_ids_sha256": _token_ids_sha256(token_ids),
        "base_scales": file_record(BASE_SCALES_PATH),
        "rounding": "round_to_nearest_ties_to_even",
        "activation_execution": summary,
        "candidate_text_or_token_output_recorded": False,
    }


def _holdout_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "holdout_red_planet",
            "category": "fresh_reviewer_holdout_factual_instruction",
            "user_turns": ["Which planet is known as the Red Planet? Answer in one short sentence."],
            "required_observable": "states Mars in one short sentence",
        },
        {
            "id": "holdout_json_cow",
            "category": "fresh_reviewer_holdout_structured_output",
            "user_turns": ["Return valid JSON only with exactly two keys, animal and sound, for a cow and the sound it makes."],
            "required_observable": "valid JSON only with animal equal to cow and sound equal to moo",
        },
        {
            "id": "holdout_locker_memory",
            "category": "fresh_reviewer_holdout_two_turn",
            "user_turns": [
                "Please remember that the locker label is Cedar-8.",
                "What is the locker label? Answer in one short sentence.",
            ],
            "required_observable": "final answer states Cedar-8 in one short sentence",
        },
        {
            "id": "holdout_color_sort",
            "category": "fresh_reviewer_holdout_structured_instruction",
            "user_turns": ["Sort these color words alphabetically and output only a comma-separated list: red, blue, green."],
            "required_observable": "outputs blue, green, red with no additional items",
        },
    ]


def _freeze_quality_matrix(tokenizer: Any, core_manifest_sha256: str) -> dict[str, Any]:
    require(not QUALITY_MARKER.exists(), "quality marker exists before matrix freeze")
    plan = load_json(PLAN_PATH)
    frozen = plan["frozen_product_quality_matrix"]
    cases: list[dict[str, Any]] = []
    for item in frozen["single_turn_cases"]:
        cases.append(
            {
                "id": item["id"],
                "category": "visible_single_turn",
                "user_turns": [item["prompt"]],
                "required_observable": item["required_observable"],
            }
        )
    for item in frozen["multi_turn_cases"]:
        cases.append(
            {
                "id": item["id"],
                "category": "visible_multi_turn",
                "user_turns": item["user_turns"],
                "required_observable": item["required_observable"],
            }
        )
    holdouts = _holdout_cases()
    cases.extend(holdouts)
    require(len(cases) == 19 and sum(len(item["user_turns"]) for item in cases) == 23, "quality matrix geometry differs")
    for item in cases:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": item["user_turns"][0]}]
        token_ids = [int(value) for value in _tokenize_messages(tokenizer, messages)[0].tolist()]
        item["initial_messages"] = messages
        item["initial_input_token_ids"] = token_ids
        item["initial_input_token_ids_sha256"] = _token_ids_sha256(token_ids)
        item["prompt_text_sha256"] = canonical_sha256(item["user_turns"])
        item["required_observable_sha256"] = hashlib.sha256(item["required_observable"].encode()).hexdigest()
    body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "matrix_id": "ace2-v13-product-quality-v1",
        "selected_after_two_constructions_and_dynamic_replay": True,
        "candidate_core_manifest_sha256": core_manifest_sha256,
        "system_message": DEFAULT_SYSTEM,
        "generation_contract": {
            "chat_template": "pinned Qwen2.5-0.5B-Instruct with add_generation_prompt=true",
            "decode": "deterministic greedy argmax with no sampling",
            "use_cache": True,
            "maximum_new_tokens_per_response": MAX_NEW_TOKENS,
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
            "visible_terminator_policy": "record terminator token and omit it from decoded visible text",
        },
        "visible_case_count": 15,
        "fresh_reviewer_holdout_count": 4,
        "response_count": 23,
        "holdout_selection": {
            "selected_at_utc": utc_now(),
            "selection_phase": "after candidate A/B manifest equality and 312-event closure; before any candidate text generation",
            "hidden_harness_or_golden_output_used": False,
            "requirements_satisfied": ["factual/instruction", "structured output", "two-turn dialogue"],
        },
        "tokenizer": {
            "tokenizer_json": file_record(SNAPSHOT / "tokenizer.json"),
            "tokenizer_config_json": file_record(SNAPSHOT / "tokenizer_config.json"),
            "chat_template_sha256": verify_source_snapshot()["chat_template_sha256"],
        },
        "cases": cases,
    }
    return {**body, "matrix_content_sha256": canonical_sha256(body)}


def _generate_response(model: nn.Module, tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    input_ids = _tokenize_messages(tokenizer, messages)
    prompt_token_ids = [int(value) for value in input_ids[0].tolist()]
    attention_mask = torch.ones_like(input_ids)
    current_ids = input_ids
    past_key_values: Any | None = None
    generated: list[int] = []
    all_logits_finite = True
    cache_returned_every_step = True
    started = time.monotonic()
    with torch.inference_mode():
        for _index in range(MAX_NEW_TOKENS):
            output = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = output.logits[0, -1]
            finite = bool(torch.isfinite(logits).all())
            all_logits_finite = all_logits_finite and finite
            require(finite, "quality generation produced non-finite logits")
            token = int(torch.argmax(logits))
            generated.append(token)
            past_key_values = output.past_key_values
            cache_returned_every_step = cache_returned_every_step and past_key_values is not None
            if token in TERMINATION_TOKEN_IDS:
                break
            current_ids = torch.tensor([[token]], dtype=input_ids.dtype)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1)
    wall_seconds = time.monotonic() - started
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    termination_reason = "termination_token" if generated and generated[-1] in TERMINATION_TOKEN_IDS else "maximum_new_tokens"
    terminating_token_id = generated[-1] if termination_reason == "termination_token" else None
    nonterminating = generated[:-1] if terminating_token_id is not None else generated
    return {
        "input_token_ids": prompt_token_ids,
        "input_token_ids_sha256": _token_ids_sha256(prompt_token_ids),
        "generated_token_ids": generated,
        "generated_token_ids_sha256": _token_ids_sha256(generated),
        "visible_nonterminating_generated_token_count": len(nonterminating),
        "terminating_token_id": terminating_token_id,
        "termination_reason": termination_reason,
        "decoded_text": decoded,
        "decoded_text_sha256": hashlib.sha256(decoded.encode()).hexdigest(),
        "wall_seconds": wall_seconds,
        "all_logits_finite": all_logits_finite,
        "use_cache": True,
        "cache_returned_every_step": cache_returned_every_step,
    }


def _run_quality_matrix(model: nn.Module, tokenizer: Any, matrix: dict[str, Any], run_kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(run_kind in ("campaign", "deterministic_replay"), "quality run kind differs")
    policy = v10.v7.v3.AuditedDynamicPolicy(load_json(BASE_SCALES_PATH))
    policy.install(model)
    outputs: list[dict[str, Any]] = []
    try:
        for case in matrix["cases"]:
            messages: list[dict[str, str]] = [{"role": "system", "content": matrix["system_message"]}]
            for turn_index, user_text in enumerate(case["user_turns"], start=1):
                messages.append({"role": "user", "content": user_text})
                messages_before = [dict(item) for item in messages]
                result = _generate_response(model, tokenizer, messages_before)
                record = {
                    "schema_version": 1,
                    "candidate_id": CANDIDATE_ID,
                    "run_kind": run_kind,
                    "case_id": case["id"],
                    "category": case["category"],
                    "turn_index": turn_index,
                    "is_final_turn": turn_index == len(case["user_turns"]),
                    "raw_user_prompt": user_text,
                    "messages_before_generation": messages_before,
                    "required_observable": case["required_observable"] if turn_index == len(case["user_turns"]) else None,
                    **result,
                }
                outputs.append(record)
                messages.append({"role": "assistant", "content": result["decoded_text"]})
                print(
                    f"V13_QUALITY_{run_kind.upper()} {case['id']} turn={turn_index} tokens={len(result['generated_token_ids'])} seconds={result['wall_seconds']:.3f}",
                    flush=True,
                )
        activation = policy.audited_summary(len(outputs))
    finally:
        policy.uninstall()
    require(len(outputs) == int(matrix["response_count"]), "quality response count differs")
    return outputs, activation


def _sentence_count(text: str) -> int:
    chunks = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item.strip()]
    return len(chunks) if chunks else 0


def _case_observable(case_id: str, text: str) -> bool:
    value = text.strip()
    lower = value.lower()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if case_id == "fact_capital":
        return "paris" in lower and _sentence_count(value) == 1
    if case_id == "arithmetic_purchase":
        return "31" in value and all(token in value for token in ("12", "3", "5")) and ("=" in value or "-" in value)
    if case_id == "format_bullets":
        return len(lines) == 3 and all(re.match(r"^[-*•]\s+[A-Za-z]", line) for line in lines)
    if case_id == "child_explanation":
        return _sentence_count(value) == 3 and "=" not in value and ("scatter" in lower or "wavelength" in lower) and "blue" in lower
    if case_id == "polite_rewrite":
        return "report" in lower and any(word in lower for word in ("please", "could you", "would you"))
    if case_id == "structured_extraction":
        compact = re.sub(r"\s+", " ", value).strip(" .")
        return compact == "Ana, Luis, Zara"
    if case_id == "concise_summary":
        return _sentence_count(value) == 1 and "tuesday" in lower and ("two hour" in lower or "two-hour" in lower or "2 hour" in lower) and "evening" in lower
    if case_id == "logic_inference":
        return lower.startswith("no") and "red" in lower and "transparent" in lower
    if case_id == "translation":
        return "nine" in lower and "ten minutes" in lower and ("early" in lower or "before" in lower)
    if case_id == "safe_refusal":
        return any(term in lower for term in ("can't help", "cannot help", "won't help", "not help")) and any(term in lower for term in ("neighbor", "owner", "locksmith", "emergency"))
    if case_id == "simple_code":
        return "def is_even" in value and "% 2" in value and "is_even(" in value[value.find("def is_even") + 1 :]
    if case_id == "constrained_creative":
        return len(lines) == 2 and all("window" in line.lower() for line in lines)
    if case_id == "context_memory":
        return "juniper" in lower and "friday" in lower and _sentence_count(value) == 1
    if case_id == "iterative_refinement":
        words = re.findall(r"[A-Za-z0-9'-]+", value)
        return len(words) < 6 and any(term in lower for term in ("herb", "balcony", "garden", "plant"))
    if case_id == "correction_recovery":
        return "monday" in lower and "thursday" in lower and "wednesday" not in lower
    if case_id == "holdout_red_planet":
        return "mars" in lower and _sentence_count(value) == 1
    if case_id == "holdout_json_cow":
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict) and set(parsed) == {"animal", "sound"} and str(parsed["animal"]).lower() == "cow" and str(parsed["sound"]).lower() == "moo"
    if case_id == "holdout_locker_memory":
        return "cedar-8" in lower and _sentence_count(value) == 1
    if case_id == "holdout_color_sort":
        return re.sub(r"\s+", " ", lower).strip(" .") == "blue, green, red"
    raise RuntimeError(f"unknown quality case: {case_id}")


def _hard_checks(record: dict[str, Any]) -> dict[str, bool]:
    text_value = record["decoded_text"]
    controls_ok = all(ord(char) >= 32 or char in "\n\r\t" for char in text_value)
    words = re.findall(r"[\w'-]+", text_value.lower(), flags=re.UNICODE)
    grams = Counter(tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3)))
    return {
        "valid_utf8_no_replacement_or_disallowed_controls": "\ufffd" not in text_value and controls_ok,
        "at_least_two_visible_nonterminating_tokens": int(record["visible_nonterminating_generated_token_count"]) >= 2,
        "no_runtime_nan_empty_or_malformed_decode": bool(text_value.strip()) and record["all_logits_finite"] is True,
        "no_degenerate_repeated_nonpunctuation_four_gram": not grams or max(grams.values()) <= 2,
        "case_specific_required_observable": _case_observable(record["case_id"], text_value) if record["is_final_turn"] else True,
    }


def _build_rubric(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    responses = []
    for record in outputs:
        checks = _hard_checks(record)
        responses.append({"case_id": record["case_id"], "turn_index": record["turn_index"], "checks": checks, "all_hard_checks_pass": all(checks.values())})
    final = [item for item, source in zip(responses, outputs, strict=True) if source["is_final_turn"]]
    visible_single = [item for item, source in zip(responses, outputs, strict=True) if source["is_final_turn"] and source["category"] == "visible_single_turn"]
    visible_multi = [item for item, source in zip(responses, outputs, strict=True) if source["is_final_turn"] and source["category"] == "visible_multi_turn"]
    holdouts = [item for item, source in zip(responses, outputs, strict=True) if source["is_final_turn"] and source["category"].startswith("fresh_reviewer_holdout")]
    observable = lambda items: sum(item["checks"]["case_specific_required_observable"] for item in items)
    hard_count = sum(item["all_hard_checks_pass"] for item in responses)
    preliminary_pass = hard_count == len(responses) and observable(visible_single) >= 11 and observable(visible_multi) == 3 and observable(holdouts) == 4
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "engineer_automated_preliminary_status": "PASS_FOR_FRESH_REVIEW" if preliminary_pass else "PRODUCT_QUALITY_NO_GO",
        "response_count": len(responses),
        "hard_check_response_pass_count": hard_count,
        "hard_check_response_pass_rate": hard_count / len(responses),
        "visible_single_turn_required_observable_count": observable(visible_single),
        "visible_multi_turn_required_observable_count": observable(visible_multi),
        "fresh_reviewer_holdout_required_observable_count": observable(holdouts),
        "responses": responses,
        "fresh_reviewer_dimensions": {
            "status": "PENDING_INDEPENDENT_FRESH_REVIEW",
            "dimensions": ["correctness", "instruction_following", "relevance", "coherence_and_readability", "multi_turn_context_when_applicable"],
            "scores": None,
            "overall_mean": None,
        },
        "local_final_product_quality_claim": None,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        for record in records:
            raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_quality_sums() -> None:
    paths = [FROZEN_MATRIX_PATH, QUALITY_MARKER, RAW_OUTPUTS_PATH, TOKEN_LATENCY_PATH, RUBRIC_PATH, QUALITY_RESULT_PATH]
    QUALITY_SUMS_PATH.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in paths), encoding="utf-8")


def _verify_preflight_artifacts() -> dict[str, Any]:
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    require(candidate_a == candidate_b, "candidate A/B full manifests differ")
    transformer = load_json(TRANSFORMER_REPRODUCTION_PATH)
    lm_head = load_json(LM_HEAD_REPRODUCTION_PATH)
    accounting = load_json(ACCOUNTING_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    require(transformer["tensor_count"] == transformer["exact_match_count"] == 168, "transformer exact reproduction differs")
    require(lm_head["reconstruction_bf16_sha256"] == LM_HEAD_V12_RECONSTRUCTION_SHA256, "lm_head reproduction differs")
    require(accounting["metadata_bytes"] == 1_926_352 and accounting["total_weight_bytes"] == 248_906_960, "accounting closure differs")
    require(dynamic["activation_execution"]["event_count"] == 312, "dynamic event count differs")
    return {
        "candidate_manifest_sha256": sha256_file(CANDIDATE_A_PATH),
        "transformer_exact_match_count": transformer["exact_match_count"],
        "lm_head_reconstruction_bf16_sha256": lm_head["reconstruction_bf16_sha256"],
        "metadata_bytes": accounting["metadata_bytes"],
        "total_weight_bytes": accounting["total_weight_bytes"],
        "dynamic_event_count": dynamic["activation_execution"]["event_count"],
    }


def run_all(*, recover: bool = False) -> int:
    require(Path(v12.reference.require_project_python()["entrypoint"]).as_posix() == ".venv/bin/python", "project Python contract differs")
    if recover:
        require(OFFICIAL_ROOT.is_dir(), "V13 recovery requires the existing official namespace")
        require(PREFLIGHT_NO_GO_PATH.is_file(), "V13 recovery requires the preserved preflight failure")
        require(not QUALITY_MARKER.exists(), "V13 recovery is forbidden after quality execution")
        require(not any(path.exists() for path in (CANDIDATE_A_PATH, CANDIDATE_B_PATH, PREFLIGHT_READY_PATH)), "V13 recovery found completed preflight artifacts")
    else:
        require(not OFFICIAL_ROOT.exists(), "fresh V13 namespace must be absent")
    self_test = combined_self_test(allow_recovery=recover)
    bindings = _verify_full_authority_before_construction()
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {**verify_versions(), "numpy": importlib.metadata.version("numpy")},
        "construction_torch_num_threads": CONSTRUCTION_TORCH_THREADS,
        "quality_torch_num_threads": QUALITY_TORCH_THREADS,
    }
    if not recover:
        OFFICIAL_ROOT.mkdir(parents=True, exist_ok=False)
    marker = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "candidate_id": CANDIDATE_ID,
        "started_at_utc": utc_now(),
        "root_absent_before_creation": not recover,
        "recovery_of_nonqualifying_preflight_failure": recover,
        "quality_campaign_started": False,
        "bindings": bindings,
        "environment": environment,
    }
    if recover:
        marker["original_construction_marker"] = file_record(CONSTRUCTION_MARKER)
        marker["preserved_failure"] = file_record(PREFLIGHT_NO_GO_PATH)
        atomic_write_json(CONSTRUCTION_RECOVERY_MARKER, marker)
        write_json(CONSTRUCTOR_RECOVERY_SELF_TEST_PATH, self_test)
    else:
        atomic_write_json(CONSTRUCTION_MARKER, marker)
        write_json(CONSTRUCTOR_SELF_TEST_PATH, self_test)
    temporary = OFFICIAL_ROOT / ".candidate-a"
    temporary.mkdir()
    candidate_a_process: multiprocessing.Process | None = None
    candidate_b_model: nn.Module | None = None
    try:
        torch.set_num_threads(CONSTRUCTION_TORCH_THREADS)
        torch.set_num_interop_threads(1)
        context = multiprocessing.get_context("spawn")
        candidate_a_process = context.Process(target=_candidate_worker, args=(str(temporary),), name="v13-candidate-a")
        candidate_a_process.start()
        print("V13_CANDIDATE_B_PARENT_START", flush=True)
        candidate_b_model, core_b, witness_b, transformer_b, lm_head_b, accounting_b = _construct_full_model()
        candidate_a_process.join()
        require(candidate_a_process.exitcode == 0, f"candidate A worker failed: exitcode={candidate_a_process.exitcode}")
        core_a = load_json(temporary / "manifest.json")
        witness_a = load_json(temporary / "witness.json")
        transformer_a = load_json(temporary / "transformer.json")
        lm_head_a = load_json(temporary / "lm_head.json")
        accounting_a = load_json(temporary / "accounting.json")
        require(core_a == core_b, "two clean V13 core manifests differ")
        require(transformer_a == transformer_b and lm_head_a == lm_head_b and accounting_a == accounting_b, "two clean V13 proof reports differ")
        require(witness_a == witness_b and all(witness_b.values()), "two clean V13 alias witnesses differ or fail")
        core_sha256 = canonical_sha256(core_b)

        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
        torch.set_num_threads(QUALITY_TORCH_THREADS)
        dynamic = _replay_dynamic_scale32(candidate_b_model, tokenizer)
        QUALITY_DIR.mkdir(parents=False, exist_ok=False)
        matrix = _freeze_quality_matrix(tokenizer, core_sha256)
        write_json(FROZEN_MATRIX_PATH, matrix)
        matrix_record = file_record(FROZEN_MATRIX_PATH)
        final_manifest = {
            **core_b,
            "construction_bindings_sha256": canonical_sha256(bindings),
            "tokenizer_and_prompt_matrix": {
                "tokenizer_json_sha256": bindings["source_model"]["files"]["tokenizer.json"]["sha256"],
                "tokenizer_config_json_sha256": bindings["source_model"]["files"]["tokenizer_config.json"]["sha256"],
                "chat_template_sha256": bindings["source_model"]["chat_template_sha256"],
                "visible_matrix_sha256": canonical_sha256(load_json(PLAN_PATH)["frozen_product_quality_matrix"]),
                "frozen_matrix": matrix_record,
            },
        }
        write_json(CANDIDATE_A_PATH, final_manifest)
        write_json(CANDIDATE_B_PATH, final_manifest)
        write_json(ALIAS_WITNESS_PATH, {"schema_version": 1, "candidate_id": CANDIDATE_ID, "candidate_a": witness_a, "candidate_b": witness_b, "all_alias_proofs_pass": True})
        transformer_b["candidate_a_and_b_reports_identical"] = True
        lm_head_b["candidate_a_and_b_reports_identical"] = True
        accounting_b["candidate_a_and_b_reports_identical"] = True
        write_json(TRANSFORMER_REPRODUCTION_PATH, transformer_b)
        write_json(LM_HEAD_REPRODUCTION_PATH, lm_head_b)
        write_json(ACCOUNTING_PATH, accounting_b)
        write_json(DYNAMIC_PATH, dynamic)
        preflight = _verify_preflight_artifacts()
        artifacts = {
            name: file_record(path)
            for name, path in {
                "construction_marker": CONSTRUCTION_RECOVERY_MARKER if recover else CONSTRUCTION_MARKER,
                "constructor_self_test": CONSTRUCTOR_RECOVERY_SELF_TEST_PATH if recover else CONSTRUCTOR_SELF_TEST_PATH,
                "candidate_a_manifest": CANDIDATE_A_PATH,
                "candidate_b_manifest": CANDIDATE_B_PATH,
                "alias_separation_witness": ALIAS_WITNESS_PATH,
                "transformer_v10_exact_reproduction": TRANSFORMER_REPRODUCTION_PATH,
                "lm_head_v12_exact_reproduction": LM_HEAD_REPRODUCTION_PATH,
                "codec_and_byte_accounting": ACCOUNTING_PATH,
                "dynamic_preflight": DYNAMIC_PATH,
                "frozen_matrix": FROZEN_MATRIX_PATH,
            }.items()
        }
        contract_body = {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "task_id": TASK_ID,
            "candidate_id": CANDIDATE_ID,
            "created_at_utc": utc_now(),
            "status": "PRODUCT_PREFLIGHT_READY",
            "bindings": bindings,
            "environment": environment,
            "artifacts": artifacts,
            "preflight": preflight,
            "quality_campaign": {"id": "quality-campaign-0001", "attempt_budget": 1, "attempts_consumed": 0},
            "selected_policy_id": None,
        }
        write_json(CONTRACT_PATH, {"contract": contract_body, "contract_sha256": canonical_sha256(contract_body)})
        write_json(PREFLIGHT_READY_PATH, {"schema_version": 1, "status": "PRODUCT_PREFLIGHT_READY", "candidate_id": CANDIDATE_ID, "contract": file_record(CONTRACT_PATH), "preflight": preflight, "quality_campaign_started": False})

        quality_marker = {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "task_id": TASK_ID,
            "candidate_id": CANDIDATE_ID,
            "campaign_id": "quality-campaign-0001",
            "started_at_utc": utc_now(),
            "contract": file_record(CONTRACT_PATH),
            "preattempt_ready": file_record(PREFLIGHT_READY_PATH),
            "frozen_matrix": matrix_record,
            "authorization_consumed": True,
        }
        atomic_write_json(QUALITY_MARKER, quality_marker)
        campaign, campaign_activation = _run_quality_matrix(candidate_b_model, tokenizer, matrix, "campaign")
        replay, replay_activation = _run_quality_matrix(candidate_b_model, tokenizer, matrix, "deterministic_replay")
        require(len(campaign) == len(replay), "quality replay response count differs")
        replay_checks = []
        for first, second in zip(campaign, replay, strict=True):
            match = first["case_id"] == second["case_id"] and first["turn_index"] == second["turn_index"] and first["generated_token_ids"] == second["generated_token_ids"] and first["decoded_text"] == second["decoded_text"]
            require(match, f"deterministic replay differs: {first['case_id']} turn {first['turn_index']}")
            replay_checks.append({"case_id": first["case_id"], "turn_index": first["turn_index"], "token_ids_match": True, "decoded_text_match": True, "campaign_wall_seconds": first["wall_seconds"], "replay_wall_seconds": second["wall_seconds"]})
        _write_jsonl(RAW_OUTPUTS_PATH, campaign)
        token_latency = {"schema_version": 1, "candidate_id": CANDIDATE_ID, "response_count": len(campaign), "all_deterministic_replays_match": True, "responses": replay_checks, "campaign_activation_execution": campaign_activation, "replay_activation_execution": replay_activation}
        write_json(TOKEN_LATENCY_PATH, token_latency)
        rubric = _build_rubric(campaign)
        write_json(RUBRIC_PATH, rubric)
        result_status = "READY_FOR_FRESH_REVIEW" if rubric["engineer_automated_preliminary_status"] == "PASS_FOR_FRESH_REVIEW" else "PRODUCT_QUALITY_NO_GO"
        result = {"schema_version": 1, "status": result_status, "candidate_id": CANDIDATE_ID, "campaign_id": "quality-campaign-0001", "response_count": len(campaign), "deterministic_replay_passed": True, "engineer_preliminary_rubric_status": rubric["engineer_automated_preliminary_status"], "fresh_reviewer_status": "PENDING", "selected_policy_id": None, "claim_boundary": "Specification-stage quantized software reference quality evidence only; no RTL, demo, PPA, U280, policy-selection, or stage-advance claim."}
        write_json(QUALITY_RESULT_PATH, result)
        _write_quality_sums()
        submission = {"schema_version": 1, "status": "READY_FOR_FRESH_REVIEW", "mission_id": MISSION_ID, "task_id": TASK_ID, "candidate_id": CANDIDATE_ID, "requested_review": "independently score the frozen raw outputs under the V13 rubric and adjudicate product quality", "contract": file_record(CONTRACT_PATH), "preattempt_ready": file_record(PREFLIGHT_READY_PATH), "quality_result": file_record(QUALITY_RESULT_PATH), "quality_sha256s": file_record(QUALITY_SUMS_PATH), "raw_outputs": file_record(RAW_OUTPUTS_PATH), "token_and_latency_results": file_record(TOKEN_LATENCY_PATH), "rubric_results": file_record(RUBRIC_PATH), "frozen_matrix": file_record(FROZEN_MATRIX_PATH), "verification_command": ["./.venv/bin/python", RUNNER_PATH.relative_to(ROOT).as_posix(), "verify-result"], "official_campaign_regeneration_forbidden": True, "selected_policy_id_before_review": None}
        write_json(FRESH_REVIEWER_SUBMISSION_PATH, submission)
        return 0 if result_status == "READY_FOR_FRESH_REVIEW" else 4
    except Exception as exc:
        if candidate_a_process is not None and candidate_a_process.is_alive():
            candidate_a_process.terminate()
            candidate_a_process.join()
        failure = {"schema_version": 1, "status": "PRODUCT_PREFLIGHT_NO_GO" if not QUALITY_MARKER.exists() else "EVALUATOR_FAILURE_AFTER_CAMPAIGN_CONSUMPTION", "candidate_id": CANDIDATE_ID, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "quality_campaign_started": QUALITY_MARKER.exists(), "failed_at_utc": utc_now(), "selected_policy_id": None}
        target = (OFFICIAL_ROOT / "PRODUCT_PREFLIGHT_RECOVERY_NO_GO.json") if recover and not QUALITY_MARKER.exists() else (PREFLIGHT_NO_GO_PATH if not QUALITY_MARKER.exists() else QUALITY_DIR / "failure.json")
        if not target.exists():
            write_json(target, failure)
        print(f"V13_RUN_FAILED {type(exc).__name__}: {exc}", flush=True)
        return 2 if not QUALITY_MARKER.exists() else 5
    finally:
        if candidate_b_model is not None:
            del candidate_b_model
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
        gc.collect()


def verify_result() -> dict[str, Any]:
    require(CONSTRUCTION_MARKER.is_file(), "V13 construction marker is missing")
    require(CONTRACT_PATH.is_file(), "V13 predeclared contract is missing")
    preflight = _verify_preflight_artifacts()
    wrapper = load_json(CONTRACT_PATH)
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V13 contract hash differs")
    for record in wrapper["contract"]["artifacts"].values():
        _verify_file_record(record)
    require(CANDIDATE_A_PATH.read_bytes() == CANDIDATE_B_PATH.read_bytes(), "candidate manifest bytes differ")
    require(QUALITY_MARKER.is_file(), "quality campaign marker is missing")
    require(FRESH_REVIEWER_SUBMISSION_PATH.is_file(), "Fresh Reviewer submission is missing")
    sums = {}
    for line in QUALITY_SUMS_PATH.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    for path in (FROZEN_MATRIX_PATH, QUALITY_MARKER, RAW_OUTPUTS_PATH, TOKEN_LATENCY_PATH, RUBRIC_PATH, QUALITY_RESULT_PATH):
        require(sums[path.name] == sha256_file(path), f"quality checksum differs: {path.name}")
    raw_count = sum(1 for line in RAW_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines() if line)
    token_latency = load_json(TOKEN_LATENCY_PATH)
    require(raw_count == token_latency["response_count"] == 23, "quality response evidence count differs")
    require(token_latency["all_deterministic_replays_match"] is True, "deterministic replay did not pass")
    result = load_json(QUALITY_RESULT_PATH)
    return {"status": "VERIFIED", "candidate_id": CANDIDATE_ID, "preflight": preflight, "quality_status": result["status"], "response_count": raw_count, "deterministic_replay_passed": True, "fresh_reviewer_submission_sha256": sha256_file(FRESH_REVIEWER_SUBMISSION_PATH), "selected_policy_id": None}
