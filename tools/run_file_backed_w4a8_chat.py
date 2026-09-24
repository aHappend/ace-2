#!/usr/bin/env python3
"""Offline host command for file-backed ACE2W4M1 generation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from . import grouped_w4a8_payload_conformance as payload_format
    from . import model_hardware_contract as hardware
    from . import qualify_stage1_w4a8_software as runtime
    from . import w4a8_full_model_evaluator as evaluator
except ImportError:
    import grouped_w4a8_payload_conformance as payload_format
    import model_hardware_contract as hardware
    import qualify_stage1_w4a8_software as runtime
    import w4a8_full_model_evaluator as evaluator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "design/QWEN25_05B_W4A8_FULL_MODEL_SOFTWARE_CONTRACT_V1.json"
)
CONTRACT_SHA_PATH = CONTRACT_PATH.with_suffix(CONTRACT_PATH.suffix + ".sha256")
RuntimeFactory = Callable[
    [dict[str, Any], bytes, dict[str, Any]],
    tuple[Any, Any],
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract() -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    require(
        isinstance(contract, dict)
        and CONTRACT_PATH.read_bytes() == canonical_bytes(contract),
        "W4A8 software contract is not canonical",
    )
    fields = CONTRACT_SHA_PATH.read_text(encoding="utf-8").strip().split()
    digest = sha256_file(CONTRACT_PATH)
    require(
        len(fields) == 2
        and fields[0] == digest
        and fields[1] == CONTRACT_PATH.name,
        "W4A8 software contract SHA256 binding differs",
    )
    return contract, digest


def base_model_spec(contract: dict[str, Any]) -> dict[str, Any]:
    identity = contract["model_identities"]["qwen2.5-0.5b-instruct"]
    return {
        "alias": "base",
        "repository": identity["repository"],
        "revision": identity["revision"],
    }


def verify_local_model(contract: dict[str, Any], spec: dict[str, Any]) -> None:
    identity = contract["model_identities"]["qwen2.5-0.5b-instruct"]
    snapshot = evaluator.snapshot_path(spec["repository"], spec["revision"])
    require(snapshot.is_dir(), "pinned local model snapshot is missing")
    for filename, expected_sha256 in identity["files"].items():
        path = snapshot / filename
        require(path.is_file(), f"pinned model file is missing: {filename}")
        require(
            sha256_file(path) == expected_sha256,
            f"pinned model file differs: {filename}",
        )


def set_determinism() -> None:
    config = evaluator.load_json(evaluator.QUALITY_CONFIG_PATH)
    seeds = config["determinism"]
    os.environ["PYTHONHASHSEED"] = str(seeds["python_seed"])
    random.seed(seeds["python_seed"])
    np.random.seed(seeds["numpy_seed"])
    torch.manual_seed(seeds["torch_seed"])
    torch.use_deterministic_algorithms(
        seeds["torch_deterministic_algorithms"]
    )


def read_prompt(path: Path | None) -> str:
    data = sys.stdin.buffer.read() if path is None else path.read_bytes()
    require(bool(data), "prompt input is empty")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("prompt input is not valid UTF-8") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is invalid: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        require(key not in value, f"duplicate JSON object member: {key}")
        value[key] = member
    return value


def validate_prompt_batch(value: Any) -> list[dict[str, str]]:
    require(isinstance(value, list), "prompt batch must be a JSON array")
    require(bool(value), "prompt batch is empty")
    prompts: list[dict[str, str]] = []
    prompt_ids: set[str] = set()
    for index, prompt in enumerate(value):
        require(
            isinstance(prompt, dict),
            f"prompt batch record {index} is not an object",
        )
        require(
            set(prompt) == {"prompt_id", "user"},
            f"prompt batch record {index} fields differ",
        )
        prompt_id = prompt["prompt_id"]
        user = prompt["user"]
        require(
            isinstance(prompt_id, str) and bool(prompt_id),
            f"prompt batch record {index} prompt_id is invalid",
        )
        require(
            isinstance(user, str) and bool(user),
            f"prompt batch record {index} user is invalid",
        )
        require(
            prompt_id not in prompt_ids,
            f"duplicate prompt_id in prompt batch: {prompt_id}",
        )
        prompt_ids.add(prompt_id)
        prompts.append({"prompt_id": prompt_id, "user": user})
    return prompts


def read_prompt_batch(path: Path) -> list[dict[str, str]]:
    data = path.read_bytes()
    require(bool(data), "prompt batch input is empty")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("prompt batch input is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError("prompt batch input is not valid JSON") from exc
    return validate_prompt_batch(value)


def validate_conversation(value: Any) -> list[dict[str, str]]:
    require(isinstance(value, list), "conversation must be a JSON array")
    require(bool(value), "conversation is empty")
    turns: list[dict[str, str]] = []
    turn_ids: set[str] = set()
    for index, turn in enumerate(value):
        require(
            isinstance(turn, dict),
            f"conversation turn {index} is not an object",
        )
        require(
            set(turn) == {"turn_id", "user"},
            f"conversation turn {index} fields differ",
        )
        turn_id = turn["turn_id"]
        user = turn["user"]
        require(
            isinstance(turn_id, str) and bool(turn_id),
            f"conversation turn {index} turn_id is invalid",
        )
        require(
            isinstance(user, str) and bool(user),
            f"conversation turn {index} user is invalid",
        )
        require(
            turn_id not in turn_ids,
            f"duplicate turn_id in conversation: {turn_id}",
        )
        turn_ids.add(turn_id)
        turns.append({"turn_id": turn_id, "user": user})
    return turns


def read_conversation(path: Path) -> list[dict[str, str]]:
    data = path.read_bytes()
    require(bool(data), "conversation input is empty")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("conversation input is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError("conversation input is not valid JSON") from exc
    return validate_conversation(value)


def expected_projection_outputs(
    descriptor: dict[str, Any],
) -> dict[str, int]:
    dimensions = descriptor["dimensions"]
    hidden_size = dimensions["hidden_size"]
    attention_heads = dimensions["num_attention_heads"]
    key_value_heads = dimensions["num_key_value_heads"]
    require(
        hidden_size % attention_heads == 0,
        "model descriptor attention geometry differs",
    )
    key_value_width = key_value_heads * (hidden_size // attention_heads)
    return {
        "q_proj": hidden_size,
        "k_proj": key_value_width,
        "v_proj": key_value_width,
        "o_proj": hidden_size,
        "gate_proj": dimensions["intermediate_size"],
        "up_proj": dimensions["intermediate_size"],
        "down_proj": hidden_size,
        "lm_head": dimensions["vocab_size"],
    }


def validate_package(
    package_path: Path,
    descriptor: dict[str, Any],
) -> bytes:
    require(package_path.is_file(), "ACE2W4M1 package is not a regular file")
    package = package_path.read_bytes()
    names = runtime.full_qwen_payload_names()
    payloads = payload_format.decode_payload_package(package, names)
    expected_outputs = expected_projection_outputs(descriptor)
    for name in names:
        projection = name.rsplit(".", 1)[-1]
        payload_format.decode_payload(
            payloads[name],
            descriptor,
            expected_input_channels=payload_format.projection_input_channels(
                descriptor,
                projection,
            ),
            expected_output_channels=expected_outputs[projection],
        )
    return package


def build_fresh_runtime(
    descriptor: dict[str, Any],
    package: bytes,
    contract: dict[str, Any],
) -> tuple[Any, Any]:
    spec = base_model_spec(contract)
    verify_local_model(contract, spec)
    set_determinism()
    tokenizer = evaluator.load_tokenizer(spec)
    template = tokenizer.chat_template
    require(isinstance(template, str), "pinned tokenizer chat template is missing")
    require(
        hashlib.sha256(template.encode("utf-8")).hexdigest()
        == contract["model_identities"]["qwen2.5-0.5b-instruct"][
            "chat_template_sha256"
        ],
        "pinned tokenizer chat template differs",
    )
    model = evaluator.load_model(spec)
    manifest = evaluator.load_json(evaluator.PROMPT_MANIFEST_PATH)
    calibration_spec = manifest["datasets"]["c4_calibration"]
    calibration_texts = evaluator.selected_local_texts(calibration_spec)
    generation = contract["shared_w4a8_contract"]["generation"]
    calibration_inputs = evaluator._tokenize_calibration(
        tokenizer,
        calibration_texts,
        generation,
        contract["evaluation_contract"]["calibration_set"]["token_limit"],
    )
    ranges, operator_ranges = evaluator.fixed.calibrate(
        model,
        calibration_inputs,
    )
    candidate = evaluator.candidate_spec(contract, "c01-mse-clip-grid")
    evaluator.replace_linears_for_candidate(
        model,
        ranges,
        candidate,
        rope_diagnostic_mechanism=evaluator.fixed.ACTIVE_ROPE_MECHANISM,
    )
    evaluator.replace_fixed_operators_for_candidate(
        model,
        operator_ranges,
        candidate,
        rope_diagnostic_mechanism=evaluator.fixed.ACTIVE_ROPE_MECHANISM,
    )
    runtime.install_full_qwen_payload_package(model, package, descriptor)
    evaluator.enable_w4a8_kv_cache(model)
    return model, tokenizer


def validate_generation(result: dict[str, Any]) -> None:
    token_count = len(result["generated_token_ids"])
    require(token_count > 0, "W4A8 generation returned no tokens")
    require(
        len(result["per_step_integer_logit_tensor_sha256"]) == token_count,
        "integer-logit record count differs",
    )
    caches = result["per_step_kv_cache"]
    require(len(caches) == token_count, "K/V cache step count differs")
    for step, cache in enumerate(caches):
        layers = cache["layers"]
        require(len(layers) == 24, f"K/V cache layer count differs at step {step}")
        require(
            [layer["layer_index"] for layer in layers] == list(range(24)),
            f"K/V cache layer ordering differs at step {step}",
        )
        for layer in layers:
            require(
                set(layer)
                == {
                    "key_scale32_sha256",
                    "key_sha256",
                    "layer_index",
                    "sequence_length",
                    "value_sha256",
                },
                f"K/V/Scale32 cache record fields differ at step {step}",
            )


def _execute_prompts(
    package_path: Path,
    prompts: list[dict[str, str]],
    max_new_tokens: int,
    *,
    descriptor: dict[str, Any] | None = None,
    runtime_factory: RuntimeFactory = build_fresh_runtime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(max_new_tokens > 0, "max-new-tokens must be positive")
    selected_prompts = validate_prompt_batch(prompts)
    selected_descriptor = (
        hardware.load_descriptor("qwen2.5-0.5b")
        if descriptor is None
        else descriptor
    )
    package = validate_package(package_path, selected_descriptor)
    contract, contract_sha256 = load_contract()
    model, tokenizer = runtime_factory(
        selected_descriptor,
        package,
        contract,
    )
    generation = copy.deepcopy(contract["shared_w4a8_contract"]["generation"])
    generation["max_new_tokens"] = max_new_tokens
    results = evaluator.generate_w4a8(
        model,
        tokenizer,
        selected_prompts,
        generation,
    )
    require(len(results) == len(selected_prompts), "generation count differs")
    for prompt, result in zip(selected_prompts, results, strict=True):
        require(
            result["prompt_id"] == prompt["prompt_id"],
            "generation prompt ordering differs",
        )
        validate_generation(result)
    metadata = {
        "contract_sha256": contract_sha256,
        "model_id": "qwen2.5-0.5b-instruct",
        "package_sha256": hashlib.sha256(package).hexdigest(),
        "runtime": "fresh-pinned-cpu-fixed-w4a8",
        "schema_version": 1,
        "status": "PASS",
    }
    return results, metadata


def execute(
    package_path: Path,
    prompt: str,
    max_new_tokens: int,
    *,
    descriptor: dict[str, Any] | None = None,
    runtime_factory: RuntimeFactory = build_fresh_runtime,
) -> dict[str, Any]:
    results, metadata = _execute_prompts(
        package_path,
        [{"prompt_id": "stdin-or-file", "user": prompt}],
        max_new_tokens,
        descriptor=descriptor,
        runtime_factory=runtime_factory,
    )
    return {**metadata, "generation": results[0]}


def execute_batch(
    package_path: Path,
    prompts: list[dict[str, str]],
    max_new_tokens: int,
    *,
    descriptor: dict[str, Any] | None = None,
    runtime_factory: RuntimeFactory = build_fresh_runtime,
) -> dict[str, Any]:
    results, metadata = _execute_prompts(
        package_path,
        prompts,
        max_new_tokens,
        descriptor=descriptor,
        runtime_factory=runtime_factory,
    )
    return {
        **metadata,
        "generations": results,
        "prompt_count": len(results),
    }


def generate_conversation(
    model: Any,
    tokenizer: Any,
    turns: list[dict[str, str]],
    generation: dict[str, Any],
) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": generation["canonical_system_message"],
        }
    ]
    state: evaluator.W4A8ContinuationState | None = None
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for turn in turns:
            messages.append({"role": "user", "content": turn["user"]})
            input_ids = evaluator.render_conversation_ids(tokenizer, messages)
            result, state = evaluator.generate_w4a8_continuation(
                model,
                tokenizer,
                turn["turn_id"],
                input_ids,
                generation,
                state,
            )
            validate_generation(result)
            results.append(result)
            messages.append(
                {"role": "assistant", "content": result["decoded_text"]}
            )
    return results


def execute_conversation(
    package_path: Path,
    turns: list[dict[str, str]],
    max_new_tokens: int,
    *,
    descriptor: dict[str, Any] | None = None,
    runtime_factory: RuntimeFactory = build_fresh_runtime,
) -> dict[str, Any]:
    require(max_new_tokens > 0, "max-new-tokens must be positive")
    selected_turns = validate_conversation(turns)
    selected_descriptor = (
        hardware.load_descriptor("qwen2.5-0.5b")
        if descriptor is None
        else descriptor
    )
    package = validate_package(package_path, selected_descriptor)
    contract, contract_sha256 = load_contract()
    model, tokenizer = runtime_factory(
        selected_descriptor,
        package,
        contract,
    )
    generation = copy.deepcopy(contract["shared_w4a8_contract"]["generation"])
    generation["max_new_tokens"] = max_new_tokens
    results = generate_conversation(
        model,
        tokenizer,
        selected_turns,
        generation,
    )
    require(len(results) == len(selected_turns), "conversation turn count differs")
    require(
        [result["prompt_id"] for result in results]
        == [turn["turn_id"] for turn in selected_turns],
        "conversation turn ordering differs",
    )
    return {
        "contract_sha256": contract_sha256,
        "model_id": "qwen2.5-0.5b-instruct",
        "package_sha256": hashlib.sha256(package).hexdigest(),
        "runtime": "persistent-pinned-cpu-fixed-w4a8",
        "schema_version": 1,
        "status": "PASS",
        "turn_count": len(results),
        "turns": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one ACE2W4M1 package and generate from UTF-8 stdin, "
            "--prompt-file, a reset-between-prompts JSON --prompt-batch-file, "
            "or a persistent JSON --conversation-file without network access."
        )
    )
    parser.add_argument("--package", type=Path, required=True)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt-file", type=Path)
    prompt_source.add_argument("--prompt-batch-file", type=Path)
    prompt_source.add_argument("--conversation-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.conversation_file is not None:
            turns = read_conversation(args.conversation_file)
            output = execute_conversation(
                args.package,
                turns,
                args.max_new_tokens,
            )
        elif args.prompt_batch_file is None:
            prompt = read_prompt(args.prompt_file)
            output = execute(args.package, prompt, args.max_new_tokens)
        else:
            prompts = read_prompt_batch(args.prompt_batch_file)
            output = execute_batch(
                args.package,
                prompts,
                args.max_new_tokens,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ACE2_W4A8_CHAT_ERROR: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
