#!/usr/bin/env python3
"""Fresh V16 authority adapter with evaluator-count and publication closure.

V13, V14, and V15 sources and official namespaces are immutable.  This module
rebinds the frozen V13 constructor to a new V16 namespace, inherits the exact
V13 product-quality matrix, counts actual model forwards during generation,
and durably publishes campaign records before post-generation activation
assertions.  Its self-test is non-consuming and never enters the V16 official
namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

import v13_v10_transformer_v12_lm_head64_backend as base


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v16"
TASK_ID = "task-7c95f6e2a841"
MISSION_ID = "successor-of-c4cc05efa75a-v16"

PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V16_PLAN.json"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V16_ENGINEER_TASK.json"
RUNNER_PATH = ROOT / "tools/run_option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v16_stage1.py"
ADAPTER_PATH = Path(__file__).resolve()
ALGORITHM_PATH = ROOT / "tools/v13_v10_transformer_v12_lm_head64_backend.py"

V13_PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V13_PLAN.json"
V15_PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V15_PLAN.json"
V15_PLAN_SHA256 = "0bd404fa215f429ce5e112459f5f17b7fbf31d9ff062e3577ae01bbfdc4a196a"
V15_TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V15_ENGINEER_TASK.json"
V15_TASK_SHA256 = "324c12fe62f1db5be9e06ecaa7b486f62de1b86937517298ac2aa5f5dbbc647d"
V15_RUNNER_PATH = ROOT / "tools/run_option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v15_stage1.py"
V15_RUNNER_SHA256 = "ae4c389db3a404cec98a94bd09a4fb8c405ea4f19580d06e4f2d058bf910e628"
V15_ADAPTER_PATH = ROOT / "tools/v15_v10_transformer_v12_lm_head64_backend.py"
V15_ADAPTER_SHA256 = "ac475da4958f26a3a04995026272e986165e3d4436f7adba335346e63d2682c7"
V15_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v15"
V15_CONSTRUCTION_MARKER_PATH = V15_ROOT / "construction_started.json"
V15_CONSTRUCTION_MARKER_SHA256 = "08aeb5a9e5621a137aeb2c7ba39d54c61e33ee8ab230b37567a56f2ba02ed508"
V15_QUALITY_MARKER_PATH = V15_ROOT / "quality-campaign-0001/execution_started.json"
V15_QUALITY_MARKER_SHA256 = "4b10a51df8678821bf7fa61e2898a0805b92fd53fe45fa18b12e32ade0b27d54"
V15_FAILURE_PATH = V15_ROOT / "quality-campaign-0001/failure.json"
V15_FAILURE_SHA256 = "371c4217c5f7b2a1b823b5536d9523033fb494624427a5197cdc1b428a845446"
V15_SUBMISSION_PATH = V15_ROOT / "fresh_reviewer_submission.json"
V15_SUBMISSION_SHA256 = "b1784394460c36285fec04721a1ead21d1d2b46f38c1096adb45306638726eb8"
V15_REVIEW_PATH = ROOT / "research/raw/specification/v15-evaluator-counter-mismatch-review-done-20260807T084955Z.json"
V15_REVIEW_SHA256 = "4a5db7140876c3b652c410b36ccda0d9a11003b16c88a7cf33a33b3cb35edca9"

OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v16"
QUALITY_DIR = OFFICIAL_ROOT / "quality-campaign-0001"

_ORIGINAL_VERIFY_FROZEN_AUTHORITY = base._verify_frozen_authority
_ORIGINAL_CANDIDATE_WORKER = base._candidate_worker
_ORIGINAL_FREEZE_QUALITY_MATRIX = base._freeze_quality_matrix
_ORIGINAL_LOAD_JSON = base.load_json
_ORIGINAL_COMBINED_SELF_TEST = base.combined_self_test
_ORIGINAL_WRITE_JSONL = base._write_jsonl


def _load_json_v16(path: Path) -> Any:
    """Expose the inherited V13 matrix through the restored V16 plan view."""

    value = _ORIGINAL_LOAD_JSON(path)
    if Path(path).resolve() != PLAN_PATH.resolve():
        return value
    inherited_matrix = _ORIGINAL_LOAD_JSON(V13_PLAN_PATH)["frozen_product_quality_matrix"]
    inherited_hash = base.canonical_sha256(inherited_matrix)
    expected_hash = value["inherited_v15_acceptance_contract"]["sections"]["frozen_product_quality_matrix"]
    base.require(inherited_hash == expected_hash, "V16 inherited matrix hash differs")
    base.require(
        inherited_hash == value["frozen_product_quality_matrix_source"]["canonical_sha256"],
        "V16 matrix source hash differs",
    )
    resolved = dict(value)
    resolved["frozen_product_quality_matrix"] = inherited_matrix
    return resolved


def _helper_paths_v16() -> dict[str, Path]:
    return {
        "v16_runner": RUNNER_PATH,
        "v16_authority_adapter": ADAPTER_PATH,
        "frozen_v13_algorithm_backend": ALGORITHM_PATH,
        "frozen_v15_runner": V15_RUNNER_PATH,
        "frozen_v15_authority_adapter": V15_ADAPTER_PATH,
        "v10_runner": Path(base.v10.reference.__file__).resolve(),
        "v10_backend": Path(base.v10.__file__).resolve(),
        "v12_runner": Path(base.v12.reference.__file__).resolve(),
        "v12_backend": Path(base.v12.__file__).resolve(),
        "dynamic_policy_helper": Path(base.v10.v7.v3.__file__).resolve(),
        "grouped_scale32_helper": Path(base.v10.v7.grouped.__file__).resolve(),
        "source_identity_helper": ROOT / "tools/qwen_instruct_option_b.py",
        "generation_contract_helper": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
        "rtl_shell_unchanged_witness": ROOT / "rtl/ace2_shell.sv",
    }


def _verify_frozen_authority_v16() -> dict[str, Any]:
    authority = _ORIGINAL_VERIFY_FROZEN_AUTHORITY()
    for path, digest in (
        (V15_PLAN_PATH, V15_PLAN_SHA256),
        (V15_TASK_PATH, V15_TASK_SHA256),
        (V15_RUNNER_PATH, V15_RUNNER_SHA256),
        (V15_ADAPTER_PATH, V15_ADAPTER_SHA256),
        (V15_CONSTRUCTION_MARKER_PATH, V15_CONSTRUCTION_MARKER_SHA256),
        (V15_QUALITY_MARKER_PATH, V15_QUALITY_MARKER_SHA256),
        (V15_FAILURE_PATH, V15_FAILURE_SHA256),
        (V15_SUBMISSION_PATH, V15_SUBMISSION_SHA256),
        (V15_REVIEW_PATH, V15_REVIEW_SHA256),
    ):
        base._verify_bound_file(path, digest)

    plan = base.load_json(PLAN_PATH)
    task = base.load_json(TASK_PATH)
    for record in plan["corrected_execution_binding"].values():
        base._verify_file_record(record)
    inherited = plan["inherited_v15_acceptance_contract"]
    base.require(inherited["path"] == str(V15_PLAN_PATH.relative_to(ROOT)), "V16 inherited plan path differs")
    base.require(inherited["sha256"] == V15_PLAN_SHA256, "V16 inherited plan hash differs")
    base.require(inherited["acceptance_changes"] == [], "V16 changed the product-quality acceptance contract")
    base.require(task["predecessor"]["terminal_status"] == "EVALUATOR_FAILURE_AFTER_CAMPAIGN_CONSUMPTION", "V16 predecessor status differs")
    base.require(task["predecessor"]["quality_conclusion"] is None, "V16 predecessor quality conclusion differs")

    failure = _ORIGINAL_LOAD_JSON(V15_FAILURE_PATH)
    submission = _ORIGINAL_LOAD_JSON(V15_SUBMISSION_PATH)
    review = _ORIGINAL_LOAD_JSON(V15_REVIEW_PATH)
    base.require(failure["error"] == "activation event call counts differ across boundaries", "V15 failure differs")
    base.require(submission["terminal"]["quality_conclusion"] is None, "V15 submission quality conclusion differs")
    base.require(review["review_status"] == "done", "V15 Fresh Review status differs")
    base.require(review["quality_execution"]["actual_model_forward_count"] == 461, "V15 forward-count evidence differs")
    base.require(not OFFICIAL_ROOT.exists(), "fresh V16 namespace must remain absent before construction")

    authority.update(
        {
            "terminal_v15_plan_sha256": V15_PLAN_SHA256,
            "terminal_v15_task_sha256": V15_TASK_SHA256,
            "terminal_v15_construction_marker_sha256": V15_CONSTRUCTION_MARKER_SHA256,
            "terminal_v15_quality_marker_sha256": V15_QUALITY_MARKER_SHA256,
            "terminal_v15_failure_sha256": V15_FAILURE_SHA256,
            "terminal_v15_submission_sha256": V15_SUBMISSION_SHA256,
            "v15_fresh_review_sha256": V15_REVIEW_SHA256,
        }
    )
    return authority


def _candidate_worker_v16(output_dir: str) -> None:
    _configure()
    _ORIGINAL_CANDIDATE_WORKER(output_dir)


def _freeze_quality_matrix_v16(tokenizer: Any, core_manifest_sha256: str) -> dict[str, Any]:
    configured_plan = base.PLAN_PATH
    try:
        base.PLAN_PATH = V13_PLAN_PATH
        return _ORIGINAL_FREEZE_QUALITY_MATRIX(tokenizer, core_manifest_sha256)
    finally:
        base.PLAN_PATH = configured_plan


def _adapter_restoration_regression() -> dict[str, Any]:
    base.require(not OFFICIAL_ROOT.exists(), "V16 regression found an official namespace")
    base.require(not (QUALITY_DIR / "execution_started.json").exists(), "V16 regression found a quality marker")
    configured_plan = base.PLAN_PATH
    tokenizer = base.AutoTokenizer.from_pretrained(base.SNAPSHOT, local_files_only=True, trust_remote_code=False)
    matrix = base._freeze_quality_matrix(tokenizer, "0" * 64)
    base.require(base.PLAN_PATH == configured_plan == PLAN_PATH, "V16 adapter did not restore the successor plan")
    visible_matrix = base.load_json(base.PLAN_PATH)["frozen_product_quality_matrix"]
    visible_matrix_sha256 = base.canonical_sha256(visible_matrix)
    expected_sha256 = base.load_json(PLAN_PATH)["frozen_product_quality_matrix_source"]["canonical_sha256"]
    base.require(visible_matrix_sha256 == expected_sha256, "V16 restored visible matrix hash differs")
    base.require(matrix["matrix_id"] == "ace2-v13-product-quality-v1", "V16 frozen matrix identity differs")
    base.require(not OFFICIAL_ROOT.exists(), "V16 regression created an official namespace")
    base.require(not (QUALITY_DIR / "execution_started.json").exists(), "V16 regression consumed quality authority")
    return {
        "status": "PASS",
        "successor_plan_restored": True,
        "exact_visible_matrix_lookup_exercised": True,
        "visible_matrix_sha256": visible_matrix_sha256,
        "frozen_matrix_id": matrix["matrix_id"],
        "response_count": matrix["response_count"],
        "official_namespace_created": False,
        "quality_marker_created": False,
        "claim_boundary": "non-consuming inherited-matrix adapter-restoration control-path regression only",
    }


def _generate_response_v16(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    input_ids = base._tokenize_messages(tokenizer, messages)
    prompt_token_ids = [int(value) for value in input_ids[0].tolist()]
    attention_mask = torch.ones_like(input_ids)
    current_ids = input_ids
    past_key_values: Any | None = None
    generated: list[int] = []
    all_logits_finite = True
    cache_returned_every_step = True
    model_forward_count = 0
    started = time.monotonic()
    with torch.inference_mode():
        for _index in range(base.MAX_NEW_TOKENS):
            output = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            model_forward_count += 1
            logits = output.logits[0, -1]
            finite = bool(torch.isfinite(logits).all())
            all_logits_finite = all_logits_finite and finite
            base.require(finite, "quality generation produced non-finite logits")
            token = int(torch.argmax(logits))
            generated.append(token)
            past_key_values = output.past_key_values
            cache_returned_every_step = cache_returned_every_step and past_key_values is not None
            if token in base.TERMINATION_TOKEN_IDS:
                break
            current_ids = torch.tensor([[token]], dtype=input_ids.dtype)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1)
    wall_seconds = time.monotonic() - started
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    termination_reason = "termination_token" if generated and generated[-1] in base.TERMINATION_TOKEN_IDS else "maximum_new_tokens"
    terminating_token_id = generated[-1] if termination_reason == "termination_token" else None
    nonterminating = generated[:-1] if terminating_token_id is not None else generated
    return {
        "input_token_ids": prompt_token_ids,
        "input_token_ids_sha256": base._token_ids_sha256(prompt_token_ids),
        "generated_token_ids": generated,
        "generated_token_ids_sha256": base._token_ids_sha256(generated),
        "visible_nonterminating_generated_token_count": len(nonterminating),
        "terminating_token_id": terminating_token_id,
        "termination_reason": termination_reason,
        "decoded_text": decoded,
        "decoded_text_sha256": hashlib.sha256(decoded.encode()).hexdigest(),
        "wall_seconds": wall_seconds,
        "model_forward_count": model_forward_count,
        "all_logits_finite": all_logits_finite,
        "use_cache": True,
        "cache_returned_every_step": cache_returned_every_step,
    }


def _new_audited_policy() -> Any:
    return base.v10.v7.v3.AuditedDynamicPolicy(base.load_json(base.BASE_SCALES_PATH))


def _write_jsonl_record(descriptor: int, record: dict[str, Any]) -> None:
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]
    os.fsync(descriptor)


def _run_quality_matrix_v16(model: Any, tokenizer: Any, matrix: dict[str, Any], run_kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base.require(run_kind in ("campaign", "deterministic_replay"), "quality run kind differs")
    policy = _new_audited_policy()
    policy.install(model)
    outputs: list[dict[str, Any]] = []
    model_forward_count = 0
    descriptor: int | None = None
    try:
        if run_kind == "campaign":
            descriptor = os.open(base.RAW_OUTPUTS_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        for case in matrix["cases"]:
            messages: list[dict[str, str]] = [{"role": "system", "content": matrix["system_message"]}]
            for turn_index, user_text in enumerate(case["user_turns"], start=1):
                messages.append({"role": "user", "content": user_text})
                messages_before = [dict(item) for item in messages]
                result = _generate_response_v16(model, tokenizer, messages_before)
                model_forward_count += int(result["model_forward_count"])
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
                if descriptor is not None:
                    _write_jsonl_record(descriptor, record)
                messages.append({"role": "assistant", "content": result["decoded_text"]})
                print(
                    f"V16_QUALITY_{run_kind.upper()} {case['id']} turn={turn_index} "
                    f"tokens={len(result['generated_token_ids'])} forwards={result['model_forward_count']} "
                    f"seconds={result['wall_seconds']:.3f}",
                    flush=True,
                )
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        activation = policy.audited_summary(model_forward_count)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        policy.uninstall()
    base.require(len(outputs) == int(matrix["response_count"]), "quality response count differs")
    base.require(model_forward_count == sum(int(item["model_forward_count"]) for item in outputs), "quality forward aggregation differs")
    activation = dict(activation)
    activation["response_count"] = len(outputs)
    activation["model_forward_count_source"] = "sum_per_response_generation_steps"
    return outputs, activation


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode() for record in records)


def _write_jsonl_v16(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        base.require(path.read_bytes() == _jsonl_bytes(records), "failure-safe raw outputs differ from completed campaign records")
        return
    _ORIGINAL_WRITE_JSONL(path, records)


def _verify_quality_bundle_v16(expected_response_count: int) -> dict[str, Any]:
    base.require(base.QUALITY_SUMS_PATH.is_file(), "V16 quality checksum manifest is missing")
    sums: dict[str, str] = {}
    for line in base.QUALITY_SUMS_PATH.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    paths = (
        base.FROZEN_MATRIX_PATH,
        base.QUALITY_MARKER,
        base.RAW_OUTPUTS_PATH,
        base.TOKEN_LATENCY_PATH,
        base.RUBRIC_PATH,
        base.QUALITY_RESULT_PATH,
    )
    for path in paths:
        base.require(path.is_file(), f"V16 quality artifact is missing: {path.name}")
        base.require(sums.get(path.name) == base.sha256_file(path), f"V16 quality checksum differs: {path.name}")
    raw_records = [json.loads(line) for line in base.RAW_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines() if line]
    token_latency = base.load_json(base.TOKEN_LATENCY_PATH)
    base.require(len(raw_records) == token_latency["response_count"] == expected_response_count, "V16 quality response evidence count differs")
    base.require(token_latency["all_deterministic_replays_match"] is True, "V16 deterministic replay did not pass")
    campaign_activation = token_latency["campaign_activation_execution"]
    base.require(campaign_activation["model_forward_count"] == sum(item["model_forward_count"] for item in raw_records), "V16 campaign forward count differs")
    base.require(campaign_activation["model_forward_count"] > expected_response_count, "V16 fixture did not distinguish forwards from responses")
    result = base.load_json(base.QUALITY_RESULT_PATH)
    return {
        "status": "VERIFIED",
        "candidate_id": CANDIDATE_ID,
        "quality_status": result["status"],
        "response_count": len(raw_records),
        "model_forward_count": campaign_activation["model_forward_count"],
        "deterministic_replay_passed": True,
        "quality_sums_sha256": base.sha256_file(base.QUALITY_SUMS_PATH),
        "raw_outputs_sha256": base.sha256_file(base.RAW_OUTPUTS_PATH),
    }


def verify_result_v16() -> dict[str, Any]:
    base.require(base.CONSTRUCTION_MARKER.is_file(), "V16 construction marker is missing")
    base.require(base.CONTRACT_PATH.is_file(), "V16 predeclared contract is missing")
    preflight = base._verify_preflight_artifacts()
    wrapper = base.load_json(base.CONTRACT_PATH)
    base.require(wrapper["contract_sha256"] == base.canonical_sha256(wrapper["contract"]), "V16 contract hash differs")
    for record in wrapper["contract"]["artifacts"].values():
        base._verify_file_record(record)
    base.require(base.CANDIDATE_A_PATH.read_bytes() == base.CANDIDATE_B_PATH.read_bytes(), "V16 candidate manifest bytes differ")
    base.require(base.QUALITY_MARKER.is_file(), "V16 quality campaign marker is missing")
    base.require(base.FRESH_REVIEWER_SUBMISSION_PATH.is_file(), "V16 Fresh Reviewer submission is missing")
    quality = _verify_quality_bundle_v16(23)
    quality.update(
        {
            "preflight": preflight,
            "fresh_reviewer_submission_sha256": base.sha256_file(base.FRESH_REVIEWER_SUBMISSION_PATH),
            "selected_policy_id": None,
        }
    )
    return quality


def _evaluator_production_path_fixture() -> dict[str, Any]:
    """Exercise the repaired production functions without official authority."""

    base.require(not OFFICIAL_ROOT.exists(), "V16 evaluator fixture found the official namespace")
    terminator = int(base.TERMINATION_TOKEN_IDS[0])

    class FakeTokenizer:
        prompts = {
            "Which city is the capital of France?": 11,
            "Please remember Juniper Friday.": 22,
            "What are the remembered word and day?": 33,
        }
        decoded = {
            1001: "Paris is the capital.",
            2001: "I will remember Juniper Friday.",
            3001: "Juniper is Friday.",
        }

        def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: Any) -> torch.Tensor:
            user = [item["content"] for item in messages if item["role"] == "user"][-1]
            return torch.tensor([[self.prompts[user]]], dtype=torch.int64)

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            base.require(skip_special_tokens, "fixture decode policy differs")
            visible = [value for value in token_ids if value not in base.TERMINATION_TOKEN_IDS]
            return self.decoded[visible[0]]

    class FakeModel:
        sequences = {
            11: [1001, 1002, terminator],
            22: [2001, 2002, 2003, terminator],
            33: [3001, 3002, 3003, 3004, terminator],
        }

        def __init__(self) -> None:
            self.forward_count = 0

        def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, past_key_values: Any, use_cache: bool) -> Any:
            base.require(use_cache and attention_mask.ndim == 2, "fixture cache contract differs")
            if past_key_values is None:
                seed = int(input_ids[0, 0])
                index = 0
            else:
                seed, index = past_key_values
            token = self.sequences[seed][index]
            logits = torch.full((1, 1, terminator + 1), -1000.0, dtype=torch.float32)
            logits[0, 0, token] = 1000.0
            self.forward_count += 1
            return SimpleNamespace(logits=logits, past_key_values=(seed, index + 1))

    class FakePolicy:
        def __init__(self, *, fail_summary: bool = False) -> None:
            self.model: FakeModel | None = None
            self.start_count = 0
            self.fail_summary = fail_summary

        def install(self, model: FakeModel) -> None:
            self.model = model
            self.start_count = model.forward_count

        def uninstall(self) -> None:
            self.model = None

        def audited_summary(self, model_forward_count: int) -> dict[str, Any]:
            base.require(self.model is not None, "fixture policy model is missing")
            observed = self.model.forward_count - self.start_count
            base.require(observed == model_forward_count, "fixture actual forward count differs")
            if self.fail_summary:
                raise RuntimeError("fixture post-generation activation assertion")
            return {
                "model_forward_count": model_forward_count,
                "named_boundary_count": 312,
                "events_per_model_forward": 312,
                "event_call_count": 312 * model_forward_count,
                "reserved_minus_128_produced": False,
            }

    matrix = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "matrix_id": "v16-non-consuming-production-path-fixture",
        "system_message": "You are a deterministic fixture.",
        "response_count": 3,
        "cases": [
            {
                "id": "fact_capital",
                "category": "visible_single_turn",
                "user_turns": ["Which city is the capital of France?"],
                "required_observable": "states Paris in one sentence",
            },
            {
                "id": "context_memory",
                "category": "visible_multi_turn",
                "user_turns": ["Please remember Juniper Friday.", "What are the remembered word and day?"],
                "required_observable": "states Juniper and Friday in one sentence",
            },
        ],
    }

    saved_paths = {
        name: getattr(base, name)
        for name in (
            "FROZEN_MATRIX_PATH",
            "QUALITY_MARKER",
            "RAW_OUTPUTS_PATH",
            "TOKEN_LATENCY_PATH",
            "RUBRIC_PATH",
            "QUALITY_RESULT_PATH",
            "QUALITY_SUMS_PATH",
        )
    }
    original_policy_factory: Callable[[], Any] = globals()["_new_audited_policy"]
    build_dir = ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v16-evaluator-production-path-", dir=build_dir) as temporary:
        fixture_root = Path(temporary)
        success = fixture_root / "success"
        success.mkdir()
        base.FROZEN_MATRIX_PATH = success / "frozen_matrix.json"
        base.QUALITY_MARKER = success / "execution_started.json"
        base.RAW_OUTPUTS_PATH = success / "raw_outputs.jsonl"
        base.TOKEN_LATENCY_PATH = success / "token_and_latency_results.json"
        base.RUBRIC_PATH = success / "rubric_results.json"
        base.QUALITY_RESULT_PATH = success / "RESULT.json"
        base.QUALITY_SUMS_PATH = success / "SHA256SUMS"
        try:
            base.write_json(base.FROZEN_MATRIX_PATH, matrix)
            base.write_json(base.QUALITY_MARKER, {"schema_version": 1, "authorization_consumed": False, "fixture": True})
            globals()["_new_audited_policy"] = lambda: FakePolicy()
            model = FakeModel()
            tokenizer = FakeTokenizer()
            campaign, campaign_activation = _run_quality_matrix_v16(model, tokenizer, matrix, "campaign")
            replay, replay_activation = _run_quality_matrix_v16(model, tokenizer, matrix, "deterministic_replay")
            base.require(len(campaign) == len(replay) == 3, "fixture response count differs")
            replay_checks: list[dict[str, Any]] = []
            for first, second in zip(campaign, replay, strict=True):
                base.require(first["generated_token_ids"] == second["generated_token_ids"], "fixture token replay differs")
                base.require(first["decoded_text"] == second["decoded_text"], "fixture decoded replay differs")
                replay_checks.append(
                    {
                        "case_id": first["case_id"],
                        "turn_index": first["turn_index"],
                        "token_ids_match": True,
                        "decoded_text_match": True,
                        "campaign_wall_seconds": first["wall_seconds"],
                        "replay_wall_seconds": second["wall_seconds"],
                    }
                )
            _write_jsonl_v16(base.RAW_OUTPUTS_PATH, campaign)
            token_latency = {
                "schema_version": 1,
                "candidate_id": CANDIDATE_ID,
                "response_count": len(campaign),
                "all_deterministic_replays_match": True,
                "responses": replay_checks,
                "campaign_activation_execution": campaign_activation,
                "replay_activation_execution": replay_activation,
            }
            base.write_json(base.TOKEN_LATENCY_PATH, token_latency)
            rubric = base._build_rubric(campaign)
            base.write_json(base.RUBRIC_PATH, rubric)
            base.write_json(
                base.QUALITY_RESULT_PATH,
                {
                    "schema_version": 1,
                    "status": "PRODUCT_QUALITY_NO_GO",
                    "candidate_id": CANDIDATE_ID,
                    "campaign_id": "non-consuming-fixture",
                    "response_count": len(campaign),
                    "deterministic_replay_passed": True,
                    "engineer_preliminary_rubric_status": rubric["engineer_automated_preliminary_status"],
                    "fresh_reviewer_status": "NOT_APPLICABLE_NON_CONSUMING_FIXTURE",
                    "selected_policy_id": None,
                },
            )
            base._write_quality_sums()
            verified = _verify_quality_bundle_v16(3)

            failure = fixture_root / "failure-safe"
            failure.mkdir()
            base.RAW_OUTPUTS_PATH = failure / "raw_outputs.jsonl"
            globals()["_new_audited_policy"] = lambda: FakePolicy(fail_summary=True)
            failed_model = FakeModel()
            try:
                _run_quality_matrix_v16(failed_model, tokenizer, matrix, "campaign")
            except RuntimeError as exc:
                base.require(str(exc) == "fixture post-generation activation assertion", "fixture failure boundary differs")
            else:
                raise RuntimeError("fixture failure-safe assertion did not fire")
            failure_safe_records = [line for line in base.RAW_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines() if line]
            base.require(len(failure_safe_records) == 3, "fixture failure-safe raw output count differs")
        finally:
            globals()["_new_audited_policy"] = original_policy_factory
            for name, path in saved_paths.items():
                setattr(base, name, path)

    base.require(not OFFICIAL_ROOT.exists(), "V16 evaluator fixture created an official namespace")
    base.require(not (QUALITY_DIR / "execution_started.json").exists(), "V16 evaluator fixture created a quality marker")
    return {
        "status": "PASS",
        "response_count": verified["response_count"],
        "actual_model_forward_count": verified["model_forward_count"],
        "response_count_is_not_forward_count": verified["model_forward_count"] != verified["response_count"],
        "variable_length_model_forward_counts": [item["model_forward_count"] for item in campaign],
        "single_turn_response_count": 1,
        "multi_turn_response_count": 2,
        "failure_safe_raw_output_count": 3,
        "terminal_quality_bundle_verification": verified["status"],
        "official_namespace_created": False,
        "quality_marker_created": False,
        "claim_boundary": "non-consuming synthetic evaluator and publication-path regression only; no candidate text, product-quality, RTL, accelerator, or PPA conclusion",
    }


def _combined_self_test_v16(*, allow_recovery: bool = False) -> dict[str, Any]:
    base.require(not allow_recovery, "V16 recovery is forbidden")
    result = _ORIGINAL_COMBINED_SELF_TEST(allow_recovery=False)
    restoration = _adapter_restoration_regression()
    evaluator = _evaluator_production_path_fixture()
    result["checks"]["inherited_matrix_adapter_restoration_regression"] = True
    result["checks"]["actual_model_forward_count_regression"] = True
    result["checks"]["failure_safe_raw_output_publication_regression"] = True
    result["checks"]["quality_bundle_terminal_verification_regression"] = True
    result["adapter_restoration_regression"] = restoration
    result["evaluator_production_path_regression"] = evaluator
    result["claim_boundary"] = (
        "synthetic constructor-kernel, frozen-input, inherited-matrix restoration, and non-consuming evaluator/publication "
        "regressions only; no full-model V16 construction, official quality campaign, product-quality, accelerator, RTL, "
        "synthesis, timing, area, or Fresh Reviewer conclusion"
    )
    return result


def _configure() -> None:
    base.CANDIDATE_ID = CANDIDATE_ID
    base.MISSION_ID = MISSION_ID
    base.TASK_ID = TASK_ID
    base.PLAN_PATH = PLAN_PATH
    base.PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
    base.TASK_PATH = TASK_PATH
    base.TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
    base.RUNNER_PATH = RUNNER_PATH
    base.BACKEND_PATH = ADAPTER_PATH
    base.OFFICIAL_ROOT = OFFICIAL_ROOT
    base.CONSTRUCTION_MARKER = OFFICIAL_ROOT / "construction_started.json"
    base.CONSTRUCTION_RECOVERY_MARKER = OFFICIAL_ROOT / "construction_recovery_started.json"
    base.CONSTRUCTOR_SELF_TEST_PATH = OFFICIAL_ROOT / "constructor_self_test.json"
    base.CONSTRUCTOR_RECOVERY_SELF_TEST_PATH = OFFICIAL_ROOT / "constructor_self_test_recovery.json"
    base.CANDIDATE_A_PATH = OFFICIAL_ROOT / "candidate_a_manifest.json"
    base.CANDIDATE_B_PATH = OFFICIAL_ROOT / "candidate_b_manifest.json"
    base.ALIAS_WITNESS_PATH = OFFICIAL_ROOT / "alias_separation_witness.json"
    base.TRANSFORMER_REPRODUCTION_PATH = OFFICIAL_ROOT / "transformer_v10_exact_reproduction.json"
    base.LM_HEAD_REPRODUCTION_PATH = OFFICIAL_ROOT / "lm_head_v12_exact_reproduction.json"
    base.ACCOUNTING_PATH = OFFICIAL_ROOT / "codec_and_byte_accounting.json"
    base.DYNAMIC_PATH = OFFICIAL_ROOT / "dynamic_preflight.json"
    base.CONTRACT_PATH = OFFICIAL_ROOT / "predeclared_contract.json"
    base.PREFLIGHT_READY_PATH = OFFICIAL_ROOT / "PRODUCT_PREFLIGHT_READY.json"
    base.PREFLIGHT_NO_GO_PATH = OFFICIAL_ROOT / "PRODUCT_PREFLIGHT_NO_GO.json"
    base.QUALITY_DIR = QUALITY_DIR
    base.FROZEN_MATRIX_PATH = QUALITY_DIR / "frozen_matrix.json"
    base.QUALITY_MARKER = QUALITY_DIR / "execution_started.json"
    base.RAW_OUTPUTS_PATH = QUALITY_DIR / "raw_outputs.jsonl"
    base.TOKEN_LATENCY_PATH = QUALITY_DIR / "token_and_latency_results.json"
    base.RUBRIC_PATH = QUALITY_DIR / "rubric_results.json"
    base.QUALITY_RESULT_PATH = QUALITY_DIR / "RESULT.json"
    base.QUALITY_SUMS_PATH = QUALITY_DIR / "SHA256SUMS"
    base.FRESH_REVIEWER_SUBMISSION_PATH = OFFICIAL_ROOT / "fresh_reviewer_submission.json"
    base.load_json = _load_json_v16
    base._helper_paths = _helper_paths_v16
    base._verify_frozen_authority = _verify_frozen_authority_v16
    base._candidate_worker = _candidate_worker_v16
    base._freeze_quality_matrix = _freeze_quality_matrix_v16
    base._generate_response = _generate_response_v16
    base._run_quality_matrix = _run_quality_matrix_v16
    base._write_jsonl = _write_jsonl_v16
    base.combined_self_test = _combined_self_test_v16
    base.verify_result = verify_result_v16


_configure()

require = base.require
combined_self_test = base.combined_self_test
verify_result = verify_result_v16


def run_all() -> int:
    """Consume the fresh V16 construction and quality authorities exactly once."""

    _configure()
    return base.run_all(recover=False)
