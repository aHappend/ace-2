#!/usr/bin/env python3
"""Additively repair and verify the frozen W4A8 BF16 v2 control identities."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import run_w4a8_full_model_software_contract as frozen


ROOT = frozen.ROOT
MODEL_ALIASES = frozen.MODEL_ALIASES
SOURCE_STATUS = frozen.SEMANTIC_PASS_STATUS
REPAIR_NAMESPACE = "bf16-control-v2-repair-v1"
REPAIR_ID = "w4a8-bf16-control-v2-identity-oracle-repair-v1"
REPAIR_STATUS = "PASS_REPAIRED_IDENTITY_ORACLE"
SELF_TEST_NAME = "identity-oracle-self-test.json"
TEST_PATH = ROOT / "verification/test_w4a8_full_model_software_contract_identity.py"


def exact_runtime_environment(contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["tool_environment"]
    observed = {
        "network_access": False,
        "packages": {
            name: (
                frozen.torch.__version__
                if name == "torch"
                else importlib.metadata.version(name)
            )
            for name in expected["packages"]
        },
        "python": platform.python_version(),
    }
    frozen.require(
        observed == expected,
        "runtime environment differs from frozen contract: "
        + json.dumps({"expected": expected, "observed": observed}, sort_keys=True),
    )
    frozen.require(os.environ["HF_HUB_OFFLINE"] == "1", "HF offline mode is not enforced")
    frozen.require(
        os.environ["TRANSFORMERS_OFFLINE"] == "1",
        "Transformers offline mode is not enforced",
    )
    return observed


def termination_reason_from_raw_steps(
    selected_tokens: list[int],
    generation: dict[str, Any],
) -> str:
    termination_ids = set(generation["termination_token_ids"])
    frozen.require(
        all(token not in termination_ids for token in selected_tokens[:-1]),
        "raw replay contains a termination token before the final step",
    )
    if selected_tokens and selected_tokens[-1] in termination_ids:
        return f"termination_token_id:{selected_tokens[-1]}"
    frozen.require(
        len(selected_tokens) == generation["max_new_tokens"],
        "raw replay ended early without a termination token",
    )
    return "max_new_tokens"


def identity_from_raw_prompt(
    prompt: dict[str, Any],
    identity_fields: list[str],
) -> dict[str, Any]:
    raw = prompt["raw_identity_inputs"]
    frozen.require(
        set(raw)
        == set(identity_fields) - {"per_step_full_bf16_logit_tensor_sha256"},
        "raw prompt identity fields differ",
    )
    steps = prompt["cache_equivalence"]
    frozen.require(steps, "raw prompt cache-equivalence steps are empty")
    frozen.require(
        [step["step_index"] for step in steps] == list(range(len(steps))),
        "raw prompt step indices differ",
    )
    selected_tokens = [step["full_prefix_selected_token"] for step in steps]
    frozen.require(
        raw["generated_token_ids"] == selected_tokens,
        "raw generated tokens differ from full-prefix selected tokens",
    )
    frozen.require(
        raw["per_step_selected_token"] == selected_tokens,
        "raw selected tokens differ from full-prefix selected tokens",
    )
    full_prefix_hashes = [
        step["full_prefix_bf16_logit_tensor_sha256"] for step in steps
    ]
    frozen.require(
        all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in full_prefix_hashes
        ),
        "raw full-prefix BF16 logit hash differs",
    )
    identity = {
        "chat_template_input_ids": raw["chat_template_input_ids"],
        "generated_token_ids": raw["generated_token_ids"],
        "per_step_full_bf16_logit_tensor_sha256": full_prefix_hashes,
        "per_step_selected_token": raw["per_step_selected_token"],
        "termination_reason": raw["termination_reason"],
    }
    frozen.require(set(identity) == set(identity_fields), "reconstructed identity fields differ")
    return identity


def verify_prompt_identity(prompt: dict[str, Any], identity_fields: list[str]) -> None:
    frozen.require(
        prompt["identity"] == identity_from_raw_prompt(prompt, identity_fields),
        f"prompt identity does not match raw records: {prompt['prompt_id']}",
    )


def repair_prompt_record(
    source_prompt: dict[str, Any],
    generation: dict[str, Any],
    identity_fields: list[str],
) -> dict[str, Any]:
    steps = source_prompt["cache_equivalence"]
    frozen.require(steps, "source v2 prompt has no cache-equivalence steps")
    selected_tokens = [step["full_prefix_selected_token"] for step in steps]
    source_identity = source_prompt["identity"]
    termination_reason = termination_reason_from_raw_steps(selected_tokens, generation)
    frozen.require(
        source_identity["generated_token_ids"] == selected_tokens,
        "source v2 generated tokens differ from raw full-prefix selections",
    )
    frozen.require(
        source_identity["per_step_selected_token"] == selected_tokens,
        "source v2 selected tokens differ from raw full-prefix selections",
    )
    frozen.require(
        source_identity["termination_reason"] == termination_reason,
        "source v2 termination reason differs from raw selected tokens",
    )
    repaired = {
        "cache_equivalence": steps,
        "decoded_text": source_prompt["decoded_text"],
        "identity": {},
        "prompt_id": source_prompt["prompt_id"],
        "raw_identity_inputs": {
            "chat_template_input_ids": source_identity["chat_template_input_ids"],
            "generated_token_ids": selected_tokens,
            "per_step_selected_token": selected_tokens,
            "termination_reason": termination_reason,
        },
        "source_prompt_sha256": frozen.canonical_sha256(source_prompt),
    }
    repaired["identity"] = identity_from_raw_prompt(repaired, identity_fields)
    verify_prompt_identity(repaired, identity_fields)
    return repaired


def identity_oracle_self_test() -> dict[str, Any]:
    fields = [
        "chat_template_input_ids",
        "generated_token_ids",
        "per_step_selected_token",
        "per_step_full_bf16_logit_tensor_sha256",
        "termination_reason",
    ]
    cache_hash = "1" * 64
    full_prefix_hash = "2" * 64
    source_prompt = {
        "cache_equivalence": [
            {
                "cache_bf16_logit_tensor_sha256": cache_hash,
                "full_prefix_bf16_logit_tensor_sha256": full_prefix_hash,
                "full_prefix_selected_token": 7,
                "logit_exact": False,
                "logit_max_abs_difference": 0.125,
                "logit_mismatch_count": 1,
                "selected_token_exact": True,
                "step_index": 0,
            }
        ],
        "decoded_text": "fixture",
        "identity": {
            "chat_template_input_ids": [1, 2, 3],
            "generated_token_ids": [7],
            "per_step_full_bf16_logit_tensor_sha256": [cache_hash],
            "per_step_selected_token": [7],
            "termination_reason": "max_new_tokens",
        },
        "prompt_id": "identity-oracle-fixture",
    }
    repaired = repair_prompt_record(
        source_prompt,
        {"max_new_tokens": 1, "termination_token_ids": [99]},
        fields,
    )
    frozen.require(
        repaired["identity"]["per_step_full_bf16_logit_tensor_sha256"]
        == [full_prefix_hash],
        "identity repair did not select the full-prefix hash",
    )
    substituted = json.loads(json.dumps(repaired))
    substituted["identity"]["per_step_full_bf16_logit_tensor_sha256"] = [cache_hash]
    substitution_rejected = False
    try:
        verify_prompt_identity(substituted, fields)
    except RuntimeError:
        substitution_rejected = True
    frozen.require(substitution_rejected, "cache/full-prefix identity substitution was accepted")
    return {
        "cache_bf16_logit_tensor_sha256": cache_hash,
        "cache_full_prefix_substitution_rejected": True,
        "correct_full_prefix_hash_recorded": True,
        "full_prefix_bf16_logit_tensor_sha256": full_prefix_hash,
        "status": "PASS",
    }


def repair_source_hashes(contract_sha256: str) -> dict[str, str]:
    repair_path = Path(__file__).resolve()
    verifier_path = ROOT / "tools/verify_w4a8_full_model_software_contract.py"
    frozen.require(TEST_PATH.is_file(), "identity regression test is missing")
    return {
        "contract_sha256": contract_sha256,
        "frozen_contract_runner_sha256": frozen.sha256_file(Path(frozen.__file__)),
        "identity_regression_test_sha256": frozen.sha256_file(TEST_PATH),
        "repair_runner_sha256": frozen.sha256_file(repair_path),
        "static_contract_verifier_sha256": frozen.sha256_file(verifier_path),
    }


def repair_replay_record(
    contract: dict[str, Any],
    contract_sha256: str,
    source_replay: dict[str, Any],
    source_record: dict[str, str],
    tool_environment: dict[str, Any],
) -> dict[str, Any]:
    control = contract["evaluation_contract"]["bf16_control"]
    generation = contract["shared_w4a8_contract"]["generation"]
    identity_fields = control["identity_fields"]
    frozen.require(source_replay["contract_sha256"] == contract_sha256, "source replay contract differs")
    frozen.require(source_replay["identity_fields"] == identity_fields, "source replay identity fields differ")
    frozen.require(
        source_replay["identity_sha256"]
        == frozen.canonical_sha256(
            [prompt["identity"] for prompt in source_replay["prompts"]]
        ),
        "source replay self-reported identity digest differs",
    )
    repaired_prompts = [
        repair_prompt_record(prompt, generation, identity_fields)
        for prompt in source_replay["prompts"]
    ]
    mismatch_count = sum(
        step["cache_bf16_logit_tensor_sha256"]
        != step["full_prefix_bf16_logit_tensor_sha256"]
        for prompt in repaired_prompts
        for step in prompt["cache_equivalence"]
    )
    return {
        "cache_equivalence_passed": source_replay["cache_equivalence_passed"],
        "cache_full_prefix_hash_difference_count": mismatch_count,
        "contract_sha256": contract_sha256,
        "created_at_utc": frozen.utc_now(),
        "identity_fields": identity_fields,
        "identity_sha256": frozen.canonical_sha256(
            [prompt["identity"] for prompt in repaired_prompts]
        ),
        "model": source_replay["model"],
        "numerical_diagnostic": source_replay["numerical_diagnostic"],
        "prompts": repaired_prompts,
        "repair_id": REPAIR_ID,
        "replay_index": source_replay["replay_index"],
        "schema_version": 2,
        "seeds": source_replay["seeds"],
        "semantic_equivalence_passed": source_replay["semantic_equivalence_passed"],
        "source_identity_sha256": source_replay["identity_sha256"],
        "source_replay": source_record,
        "status": REPAIR_STATUS,
        "tool_environment": tool_environment,
        "tool_environment_sha256": frozen.canonical_sha256(tool_environment),
    }


def execute_repair(
    contract: dict[str, Any],
    contract_sha256: str,
    alias: str,
    tool_environment: dict[str, Any],
) -> Path:
    spec = frozen.model_spec(contract, alias)
    source_target = (
        ROOT
        / spec["artifact_root"]
        / contract["artifact_policy"]["bf16_control_namespace"]
    )
    target = ROOT / spec["artifact_root"] / REPAIR_NAMESPACE
    frozen.require(source_target.is_dir(), f"immutable v2 BF16 control is missing for {alias}")
    frozen.require(not target.exists(), f"immutable BF16 repair already exists: {target.relative_to(ROOT)}")
    source_summary_path = source_target / "summary.json"
    source_summary = frozen.load_json(source_summary_path)
    frozen.require(source_summary["status"] == SOURCE_STATUS, f"source v2 control did not pass for {alias}")
    frozen.require(source_summary["exact_replay_count"] == 2, f"source v2 replay count differs for {alias}")
    frozen.require(
        source_summary["source_hashes"] == frozen.source_hashes(contract_sha256),
        f"source v2 source hashes differ for {alias}",
    )
    frozen.require(
        source_summary["contract_control"] == contract["evaluation_contract"]["bf16_control"],
        f"source v2 contract binding differs for {alias}",
    )
    temporary = Path(tempfile.mkdtemp(prefix=".bf16-control-repair-", dir=target.parent))
    try:
        replay_records = []
        identities = []
        classifications = []
        mismatch_counts = []
        for source_local_record in source_summary["replays"]:
            source_path = source_target / source_local_record["path"]
            frozen.require(source_path.is_file(), f"source v2 replay is missing for {alias}")
            frozen.require(
                frozen.sha256_file(source_path) == source_local_record["sha256"],
                f"source v2 replay hash differs for {alias}",
            )
            source_record = {
                "path": source_path.relative_to(ROOT).as_posix(),
                "sha256": source_local_record["sha256"],
            }
            repaired = repair_replay_record(
                contract,
                contract_sha256,
                frozen.load_json(source_path),
                source_record,
                tool_environment,
            )
            repaired_path = temporary / source_local_record["path"]
            frozen.write_json(repaired_path, repaired)
            replay_records.append(
                {
                    "path": repaired_path.name,
                    "sha256": frozen.sha256_file(repaired_path),
                    "source": source_record,
                }
            )
            identities.append(repaired["identity_sha256"])
            classifications.append(repaired["numerical_diagnostic"]["classification"])
            mismatch_counts.append(repaired["cache_full_prefix_hash_difference_count"])
        frozen.require(len(set(identities)) == 1, f"repaired duplicate identities differ for {alias}")
        frozen.require(len(set(classifications)) == 1, f"repaired numerical classifications differ for {alias}")
        frozen.require(len(set(mismatch_counts)) == 1, f"repaired mismatch counts differ for {alias}")
        self_test_path = temporary / SELF_TEST_NAME
        frozen.write_json(self_test_path, identity_oracle_self_test())
        summary = {
            "cache_equivalence_passed": source_summary["cache_equivalence_passed"],
            "cache_full_prefix_hash_difference_count_per_replay": mismatch_counts[0],
            "contract_control": contract["evaluation_contract"]["bf16_control"],
            "created_at_utc": frozen.utc_now(),
            "exact_replay_count": len(replay_records),
            "identity_oracle_self_test": {
                "path": self_test_path.name,
                "sha256": frozen.sha256_file(self_test_path),
            },
            "identity_sha256": identities[0],
            "model": source_summary["model"],
            "numerical_diagnostic_classification": classifications[0],
            "repair_id": REPAIR_ID,
            "repaired_replay_records_sha256": frozen.canonical_sha256(replay_records),
            "replays": replay_records,
            "schema_version": 2,
            "source_control": {
                "path": source_summary_path.relative_to(ROOT).as_posix(),
                "sha256": frozen.sha256_file(source_summary_path),
            },
            "source_hashes": repair_source_hashes(contract_sha256),
            "source_replay_records_sha256": frozen.canonical_sha256(source_summary["replays"]),
            "status": REPAIR_STATUS,
            "tool_environment": tool_environment,
            "tool_environment_sha256": frozen.canonical_sha256(tool_environment),
        }
        frozen.write_json(temporary / "summary.json", summary)
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def verify_repair(
    contract: dict[str, Any],
    contract_sha256: str,
    alias: str,
    tool_environment: dict[str, Any],
) -> dict[str, Any]:
    spec = frozen.model_spec(contract, alias)
    target = ROOT / spec["artifact_root"] / REPAIR_NAMESPACE
    summary_path = target / "summary.json"
    frozen.require(summary_path.is_file(), f"BF16 control repair is missing for {alias}")
    summary = frozen.load_json(summary_path)
    frozen.require(summary["status"] == REPAIR_STATUS, f"BF16 repair did not pass for {alias}")
    frozen.require(summary["repair_id"] == REPAIR_ID, f"BF16 repair identity differs for {alias}")
    frozen.require(summary["source_hashes"] == repair_source_hashes(contract_sha256), f"BF16 repair source hashes differ for {alias}")
    frozen.require(summary["tool_environment"] == tool_environment, f"BF16 repair runtime differs for {alias}")
    frozen.require(
        summary["tool_environment_sha256"] == frozen.canonical_sha256(tool_environment),
        f"BF16 repair runtime hash differs for {alias}",
    )
    self_test_record = summary["identity_oracle_self_test"]
    self_test_path = target / self_test_record["path"]
    frozen.require(self_test_path.is_file(), f"identity self-test evidence is missing for {alias}")
    frozen.require(frozen.sha256_file(self_test_path) == self_test_record["sha256"], f"identity self-test hash differs for {alias}")
    frozen.require(frozen.load_json(self_test_path) == identity_oracle_self_test(), f"identity self-test result differs for {alias}")
    source_control = summary["source_control"]
    source_summary_path = ROOT / source_control["path"]
    frozen.require(source_summary_path.is_file(), f"source v2 summary is missing for {alias}")
    frozen.require(frozen.sha256_file(source_summary_path) == source_control["sha256"], f"source v2 summary hash differs for {alias}")
    source_summary = frozen.load_json(source_summary_path)
    frozen.require(
        summary["source_replay_records_sha256"]
        == frozen.canonical_sha256(source_summary["replays"]),
        f"source v2 replay-record binding differs for {alias}",
    )
    frozen.require(
        summary["repaired_replay_records_sha256"]
        == frozen.canonical_sha256(summary["replays"]),
        f"repaired replay-record binding differs for {alias}",
    )
    frozen.require(summary["model"] == source_summary["model"], f"BF16 repair model differs for {alias}")
    identities = []
    mismatch_counts = []
    generation = contract["shared_w4a8_contract"]["generation"]
    identity_fields = contract["evaluation_contract"]["bf16_control"]["identity_fields"]
    frozen.require(len(summary["replays"]) == len(source_summary["replays"]), f"repaired replay count differs for {alias}")
    for record, source_local_record in zip(summary["replays"], source_summary["replays"]):
        path = target / record["path"]
        frozen.require(path.is_file(), f"repaired replay is missing for {alias}")
        frozen.require(frozen.sha256_file(path) == record["sha256"], f"repaired replay hash differs for {alias}")
        source_record = record["source"]
        source_path = ROOT / source_record["path"]
        frozen.require(source_path.is_file(), f"source v2 replay is missing for {alias}")
        frozen.require(frozen.sha256_file(source_path) == source_record["sha256"], f"source v2 replay hash differs for {alias}")
        frozen.require(
            source_record["sha256"] == source_local_record["sha256"]
            and source_path.name == source_local_record["path"],
            f"source v2 replay record differs for {alias}",
        )
        replay = frozen.load_json(path)
        source_replay = frozen.load_json(source_path)
        frozen.require(replay["source_replay"] == source_record, f"source replay binding differs for {alias}")
        frozen.require(replay["tool_environment"] == tool_environment, f"repaired replay runtime differs for {alias}")
        frozen.require(replay["tool_environment_sha256"] == frozen.canonical_sha256(tool_environment), f"repaired runtime hash differs for {alias}")
        frozen.require(replay["contract_sha256"] == contract_sha256, f"repaired contract differs for {alias}")
        frozen.require(replay["identity_fields"] == identity_fields, f"repaired identity fields differ for {alias}")
        frozen.require(replay["status"] == REPAIR_STATUS, f"repaired replay did not pass for {alias}")
        frozen.require(len(replay["prompts"]) == len(source_replay["prompts"]), f"repaired prompt count differs for {alias}")
        for prompt, source_prompt in zip(replay["prompts"], source_replay["prompts"]):
            frozen.require(
                prompt == repair_prompt_record(source_prompt, generation, identity_fields),
                f"repaired raw prompt differs for {alias}: {prompt['prompt_id']}",
            )
            verify_prompt_identity(prompt, identity_fields)
        observed_identity = frozen.canonical_sha256(
            [prompt["identity"] for prompt in replay["prompts"]]
        )
        frozen.require(replay["identity_sha256"] == observed_identity, f"repaired identity digest differs for {alias}")
        observed_numerics = frozen.classify_cache_numerics(contract, alias, replay["prompts"])
        frozen.require(replay["numerical_diagnostic"] == observed_numerics, f"repaired numerical diagnostic differs for {alias}")
        mismatch_count = sum(
            step["cache_bf16_logit_tensor_sha256"]
            != step["full_prefix_bf16_logit_tensor_sha256"]
            for prompt in replay["prompts"]
            for step in prompt["cache_equivalence"]
        )
        frozen.require(replay["cache_full_prefix_hash_difference_count"] == mismatch_count, f"repaired mismatch count differs for {alias}")
        identities.append(observed_identity)
        mismatch_counts.append(mismatch_count)
    frozen.require(len(identities) == summary["exact_replay_count"] == 2, f"repaired replay count differs for {alias}")
    frozen.require(len(set(identities)) == 1 and identities[0] == summary["identity_sha256"], f"repaired duplicate identities differ for {alias}")
    frozen.require(
        len(set(mismatch_counts)) == 1
        and mismatch_counts[0] == summary["cache_full_prefix_hash_difference_count_per_replay"],
        f"repaired mismatch summary differs for {alias}",
    )
    return {
        "alias": alias,
        "cache_full_prefix_hash_difference_count_per_replay": mismatch_counts[0],
        "identity_sha256": summary["identity_sha256"],
        "numerical_diagnostic_classification": summary[
            "numerical_diagnostic_classification"
        ],
        "path": target.relative_to(ROOT).as_posix(),
        "status": REPAIR_STATUS,
    }


def selected_aliases(value: str) -> list[str]:
    return list(MODEL_ALIASES) if value == "all" else [value]


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("repair-controls", "self-test", "verify-controls", "ptq-preflight"),
        required=True,
    )
    parser.add_argument("--model", choices=(*MODEL_ALIASES, "all"), required=True)
    args = parser.parse_args()
    contract, contract_sha256 = frozen.load_contract()
    environment = exact_runtime_environment(contract)
    aliases = selected_aliases(args.model)
    if args.action == "self-test":
        frozen.require(args.model == "all", "identity self-test requires --model all")
        result: Any = {
            "environment": environment,
            "identity_oracle_self_test": identity_oracle_self_test(),
            "status": "PASS",
        }
    elif args.action == "repair-controls":
        result = [
            {
                "alias": alias,
                "path": execute_repair(contract, contract_sha256, alias, environment)
                .relative_to(ROOT)
                .as_posix(),
                "status": REPAIR_STATUS,
            }
            for alias in aliases
        ]
    else:
        controls = [
            verify_repair(contract, contract_sha256, alias, environment)
            for alias in aliases
        ]
        result = {
            "action": args.action,
            "controls": controls,
            "environment": environment,
            "ptq_authorized": False,
            "status": (
                "PASS"
                if args.action == "verify-controls"
                else "BLOCKED_PENDING_FRESH_L2_ACCEPTANCE"
            ),
        }
        if args.action == "ptq-preflight":
            frozen.require(args.model == "all", "PTQ preflight requires both models")
            frozen.require(
                contract["scope_guards"]["ptq_candidate_execution_authorized"] is False,
                "contract unexpectedly authorizes PTQ candidate execution",
            )
            result["blocker"] = (
                "fresh_l2_acceptance_of_non_overwriting_v2_identity_repair_required"
            )
            result["candidate_order"] = [
                candidate["candidate_id"] for candidate in contract["candidate_matrix"]
            ]
            result["maximum_genuine_candidate_model_attempts"] = contract[
                "stopping_rules"
            ]["maximum_genuine_candidate_model_attempts"]
    print(
        "ACE2_W4A8_CONTROL_REPAIR "
        + json.dumps(
            {
                "action": args.action,
                "contract_sha256": contract_sha256,
                "model": args.model,
                "result": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
