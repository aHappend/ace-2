#!/usr/bin/env python3
"""Qualify persistent grouped-W4A8 turns against fresh full prefixes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

try:
    from . import model_hardware_contract as hardware
    from . import run_file_backed_w4a8_chat as chat
except ImportError:
    import model_hardware_contract as hardware
    import run_file_backed_w4a8_chat as chat


TURNS = [
    {"turn_id": "definition", "user": "Define a cache in one short sentence."},
    {"turn_id": "follow-up", "user": "Why is that useful?"},
]
EXACT_FIELDS = (
    "generated_token_ids",
    "per_step_integer_logit_tensor_sha256",
    "per_step_kv_cache",
)
GENERATION_FIELDS = {
    "decoded_text",
    "decoded_text_sha256",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "input_token_ids",
    "per_step_integer_logit_tensor_sha256",
    "per_step_kv_cache",
    "per_step_kv_cache_sha256",
    "prompt_id",
    "termination_reason",
    "use_cache",
}
LAYER_FIELDS = {
    "key_scale32_sha256",
    "key_sha256",
    "layer_index",
    "sequence_length",
    "value_sha256",
}
T = TypeVar("T")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def peak_rss_kib() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    require(value > 0, "host process peak RSS is invalid")
    return value


def measure_host_phase(
    operation: Callable[[], T],
    *,
    clock: Callable[[], float] = time.perf_counter,
    rss_reader: Callable[[], int] = peak_rss_kib,
) -> tuple[T, dict[str, Any]]:
    started = clock()
    value = operation()
    wall_seconds = clock() - started
    rss_kib = rss_reader()
    require(
        isinstance(wall_seconds, (int, float))
        and not isinstance(wall_seconds, bool)
        and wall_seconds >= 0,
        "host phase wall time is invalid",
    )
    require(
        isinstance(rss_kib, int) and not isinstance(rss_kib, bool) and rss_kib > 0,
        "host phase peak RSS is invalid",
    )
    return value, {
        "process_peak_rss_kib_after_phase": rss_kib,
        "wall_seconds": wall_seconds,
    }


def validate_generation_evidence(
    result: Any,
    turn_id: str,
) -> dict[str, Any]:
    require(isinstance(result, dict), f"generation evidence is invalid: {turn_id}")
    require(
        set(result) == GENERATION_FIELDS,
        f"generation evidence fields differ: {turn_id}",
    )
    require(
        result["prompt_id"] == turn_id,
        f"generation turn identity differs: {turn_id}",
    )
    require(result["use_cache"] is True, f"K/V cache is disabled: {turn_id}")
    input_ids = result["input_token_ids"]
    generated_ids = result["generated_token_ids"]
    require(
        isinstance(input_ids, list)
        and bool(input_ids)
        and all(isinstance(token, int) and not isinstance(token, bool) for token in input_ids),
        f"input token evidence is invalid: {turn_id}",
    )
    require(
        isinstance(generated_ids, list)
        and bool(generated_ids)
        and all(
            isinstance(token, int) and not isinstance(token, bool)
            for token in generated_ids
        ),
        f"generated token evidence is invalid: {turn_id}",
    )
    require(
        result["generated_token_ids_sha256"]
        == chat.evaluator.canonical_sha256(generated_ids),
        f"generated token hash differs: {turn_id}",
    )
    decoded_text = result["decoded_text"]
    require(
        isinstance(decoded_text, str)
        and result["decoded_text_sha256"]
        == hashlib.sha256(decoded_text.encode("utf-8")).hexdigest(),
        f"decoded text hash differs: {turn_id}",
    )
    require(
        isinstance(result["termination_reason"], str)
        and bool(result["termination_reason"]),
        f"termination evidence is invalid: {turn_id}",
    )

    token_count = len(generated_ids)
    logit_hashes = result["per_step_integer_logit_tensor_sha256"]
    caches = result["per_step_kv_cache"]
    cache_hashes = result["per_step_kv_cache_sha256"]
    require(
        isinstance(logit_hashes, list)
        and len(logit_hashes) == token_count
        and all(is_sha256(value) for value in logit_hashes),
        f"integer-logit hash evidence is invalid: {turn_id}",
    )
    require(
        isinstance(caches, list) and len(caches) == token_count,
        f"K/V cache step evidence is invalid: {turn_id}",
    )
    require(
        isinstance(cache_hashes, list)
        and len(cache_hashes) == token_count
        and all(is_sha256(value) for value in cache_hashes),
        f"K/V cache hash evidence is invalid: {turn_id}",
    )
    for step, (cache, expected_cache_hash) in enumerate(
        zip(caches, cache_hashes, strict=True)
    ):
        require(
            isinstance(cache, dict) and set(cache) == {"layers", "sha256"},
            f"K/V cache record fields differ: {turn_id}: step {step}",
        )
        layers = cache["layers"]
        require(
            isinstance(layers, list) and len(layers) == 24,
            f"K/V cache layer count differs: {turn_id}: step {step}",
        )
        require(
            cache["sha256"] == expected_cache_hash
            == chat.evaluator.canonical_sha256(layers),
            f"K/V cache hash differs: {turn_id}: step {step}",
        )
        for layer_index, layer in enumerate(layers):
            require(
                isinstance(layer, dict) and set(layer) == LAYER_FIELDS,
                f"K/V/Scale32 record fields differ: {turn_id}: step {step}",
            )
            require(
                layer["layer_index"] == layer_index,
                f"K/V cache layer ordering differs: {turn_id}: step {step}",
            )
            require(
                isinstance(layer["sequence_length"], int)
                and not isinstance(layer["sequence_length"], bool)
                and layer["sequence_length"] > 0,
                f"K/V cache sequence length is invalid: {turn_id}: step {step}",
            )
            for field in (
                "key_scale32_sha256",
                "key_sha256",
                "value_sha256",
            ):
                require(
                    is_sha256(layer[field]),
                    f"K/V/Scale32 hash is invalid: {turn_id}: step {step}: {field}",
                )
    return result


def scale32_record_hash(result: dict[str, Any]) -> str:
    records = [
        [
            layer["key_scale32_sha256"]
            for layer in cache["layers"]
        ]
        for cache in result["per_step_kv_cache"]
    ]
    return chat.evaluator.canonical_sha256(records)


def compare_turn(
    incremental: dict[str, Any],
    fresh: dict[str, Any],
    turn_id: str,
) -> dict[str, Any]:
    validate_generation_evidence(incremental, turn_id)
    validate_generation_evidence(fresh, turn_id)
    require(
        incremental["input_token_ids"] == fresh["input_token_ids"],
        f"incremental and fresh full-prefix inputs differ: {turn_id}",
    )
    for field in EXACT_FIELDS:
        require(
            incremental[field] == fresh[field],
            f"incremental and fresh full-prefix records differ: {turn_id}: {field}",
        )
    for step, cache in enumerate(incremental["per_step_kv_cache"]):
        expected_length = len(incremental["input_token_ids"]) + step
        require(
            len(cache["layers"]) == 24
            and all(
                layer["sequence_length"] == expected_length
                for layer in cache["layers"]
            ),
            f"incremental cache growth differs: {turn_id}: step {step}",
        )
    return {
        "generated_token_ids_sha256": incremental[
            "generated_token_ids_sha256"
        ],
        "integer_logit_hash_record_sha256": chat.evaluator.canonical_sha256(
            incremental["per_step_integer_logit_tensor_sha256"]
        ),
        "scale32_record_sha256": scale32_record_hash(incremental),
        "turn_id": turn_id,
    }


def qualify(
    package_path: Path,
    output: Path,
    expected_package_sha256: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    require(package_path.is_file(), "pinned grouped-W4A8 package is missing")
    require(
        len(expected_package_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_package_sha256
        ),
        "expected package SHA256 is invalid",
    )
    observed_package_sha256 = sha256_file(package_path)
    require(
        observed_package_sha256 == expected_package_sha256,
        "pinned grouped-W4A8 package SHA256 differs",
    )
    require(max_new_tokens > 0, "max-new-tokens must be positive")
    output.mkdir(parents=True, exist_ok=False)

    turns = chat.validate_conversation(TURNS)
    descriptor = hardware.load_descriptor("qwen2.5-0.5b")
    package = chat.validate_package(package_path, descriptor)
    contract, contract_sha256 = chat.load_contract()
    generation = copy.deepcopy(
        contract["shared_w4a8_contract"]["generation"]
    )
    generation["max_new_tokens"] = max_new_tokens

    profile_started = time.perf_counter()
    (
        (incremental_model, incremental_tokenizer),
        incremental_build_metrics,
    ) = measure_host_phase(
        lambda: chat.build_fresh_runtime(
            descriptor,
            package,
            contract,
        )
    )
    incremental, incremental_generation_metrics = measure_host_phase(
        lambda: chat.generate_conversation(
            incremental_model,
            incremental_tokenizer,
            turns,
            generation,
        )
    )

    messages = [
        {
            "role": "system",
            "content": generation["canonical_system_message"],
        }
    ]
    fresh_records: list[dict[str, Any]] = []
    fresh_turn_metrics: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for turn, incremental_turn in zip(turns, incremental, strict=True):
            messages.append({"role": "user", "content": turn["user"]})
            (
                (fresh_model, fresh_tokenizer),
                fresh_build_metrics,
            ) = measure_host_phase(
                lambda: chat.build_fresh_runtime(
                    descriptor,
                    package,
                    contract,
                )
            )
            input_ids = chat.evaluator.render_conversation_ids(
                fresh_tokenizer,
                messages,
            )
            (fresh, _state), fresh_generation_metrics = measure_host_phase(
                lambda: chat.evaluator.generate_w4a8_continuation(
                    fresh_model,
                    fresh_tokenizer,
                    turn["turn_id"],
                    input_ids,
                    generation,
                )
            )
            comparison_records.append(
                compare_turn(incremental_turn, fresh, turn["turn_id"])
            )
            fresh_records.append(fresh)
            fresh_turn_metrics.append(
                {
                    "generation": fresh_generation_metrics,
                    "runtime_build": fresh_build_metrics,
                    "turn_id": turn["turn_id"],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": incremental_turn["decoded_text"],
                }
            )
    profile_wall_seconds = time.perf_counter() - profile_started
    require(profile_wall_seconds >= 0, "profiler wall time is invalid")
    write_json(output / "incremental.json", incremental)
    write_json(output / "fresh-full-prefix.json", fresh_records)

    result = {
        "checks": {
            "cache_lengths_grow_from_full_prefix": True,
            "complete_kv_scale32_records_match": True,
            "generated_token_ids_match": True,
            "integer_logit_hashes_match": True,
            "malformed_evidence_rejected": True,
            "one_incremental_runtime": True,
        },
        "comparison_records": comparison_records,
        "contract_sha256": contract_sha256,
        "max_new_tokens": max_new_tokens,
        "measurements": {
            "fresh_prefix_recomputation": {
                "turns": fresh_turn_metrics,
            },
            "measurement_kind": "computer-local host measurement",
            "not_hardware_latency": True,
            "persistent_kv": {
                "generation": incremental_generation_metrics,
                "runtime_build": incremental_build_metrics,
            },
            "process_peak_rss_semantics": (
                "Linux host-process ru_maxrss cumulative high-water mark sampled "
                "after each phase; fresh-prefix phases run after persistent-K/V "
                "phases in the same process"
            ),
            "profiler_total_wall_seconds": profile_wall_seconds,
            "timing_clock": "time.perf_counter",
        },
        "package_sha256": observed_package_sha256,
        "schema_version": 2,
        "status": "PASS",
        "turn_ids": [turn["turn_id"] for turn in turns],
    }
    write_json(output / "qualification.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one persistent grouped-W4A8 conversation runtime with "
            "fresh runtimes over each complete conversation prefix."
        )
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = qualify(
            args.package,
            args.output,
            args.expected_package_sha256,
            args.max_new_tokens,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"ACE2_W4A8_CONVERSATION_QUALIFICATION_ERROR: {exc}",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
