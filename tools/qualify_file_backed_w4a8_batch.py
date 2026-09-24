#!/usr/bin/env python3
"""Qualify ordered multi-prompt generation against isolated CLI processes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
CHAT_COMMAND = ROOT / "tools/run_file_backed_w4a8_chat.py"
PROMPTS = [
    {"prompt_id": "definition", "user": "Define a cache in one short sentence."},
    {"prompt_id": "arithmetic", "user": "What is two plus three?"},
]
Runner = Callable[..., subprocess.CompletedProcess[str]]


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


def run_cli(
    arguments: Sequence[str],
    *,
    timeout_seconds: int,
    runner: Runner,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "20260729",
            "PYTHONPATH": f"{ROOT}:{ROOT / 'tools'}",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return runner(
        [sys.executable, str(CHAT_COMMAND), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def parse_success(
    process: subprocess.CompletedProcess[str],
    identity: str,
) -> dict[str, Any]:
    require(
        process.returncode == 0,
        f"{identity} failed ({process.returncode}): {process.stderr.strip()}",
    )
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{identity} stdout is not JSON") from exc
    require(isinstance(value, dict), f"{identity} stdout is not a JSON object")
    require(value.get("status") == "PASS", f"{identity} did not report PASS")
    return value


def validate_fresh_cache(generation: dict[str, Any], identity: str) -> None:
    input_ids = generation.get("input_token_ids")
    token_ids = generation.get("generated_token_ids")
    caches = generation.get("per_step_kv_cache")
    require(
        isinstance(input_ids, list) and bool(input_ids),
        f"{identity} input token record is invalid",
    )
    require(
        isinstance(token_ids, list) and bool(token_ids),
        f"{identity} generated token record is invalid",
    )
    require(
        isinstance(caches, list) and len(caches) == len(token_ids),
        f"{identity} cache step count differs",
    )
    for step_index, cache in enumerate(caches):
        layers = cache.get("layers") if isinstance(cache, dict) else None
        require(
            isinstance(layers, list) and len(layers) == 24,
            f"{identity} cache layer count differs at step {step_index}",
        )
        expected_length = len(input_ids) + step_index
        require(
            [layer.get("layer_index") for layer in layers] == list(range(24)),
            f"{identity} cache layer ordering differs at step {step_index}",
        )
        require(
            all(layer.get("sequence_length") == expected_length for layer in layers),
            f"{identity} cache did not start from fresh prompt state",
        )


def normalized_generation(
    generation: dict[str, Any],
    prompt_id: str,
) -> dict[str, Any]:
    normalized = copy.deepcopy(generation)
    normalized["prompt_id"] = prompt_id
    return normalized


def qualify(
    package: Path,
    output: Path,
    expected_package_sha256: str,
    max_new_tokens: int,
    timeout_seconds: int,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    require(package.is_file(), "pinned grouped-W4A8 package is missing")
    require(
        len(expected_package_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_package_sha256),
        "expected package SHA256 is invalid",
    )
    observed_package_sha256 = sha256_file(package)
    require(
        observed_package_sha256 == expected_package_sha256,
        "pinned grouped-W4A8 package SHA256 differs",
    )
    require(max_new_tokens >= 2, "qualification requires at least two generated tokens")
    require(timeout_seconds > 0, "timeout must be positive")
    output.mkdir(parents=True, exist_ok=False)

    batch_input = output / "prompts.json"
    write_json(batch_input, PROMPTS)
    batch_process = run_cli(
        [
            "--package",
            str(package),
            "--prompt-batch-file",
            str(batch_input),
            "--max-new-tokens",
            str(max_new_tokens),
        ],
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    (output / "batch.stderr").write_text(batch_process.stderr, encoding="utf-8")
    (output / "batch.json").write_text(batch_process.stdout, encoding="utf-8")
    batch = parse_success(batch_process, "batch CLI")
    require(batch.get("prompt_count") == len(PROMPTS), "batch prompt count differs")
    generations = batch.get("generations")
    require(
        isinstance(generations, list) and len(generations) == len(PROMPTS),
        "batch generation count differs",
    )
    require(
        [generation.get("prompt_id") for generation in generations]
        == [prompt["prompt_id"] for prompt in PROMPTS],
        "batch generation ordering differs",
    )

    for index, prompt in enumerate(PROMPTS):
        prompt_path = output / f"prompt-{index}.txt"
        prompt_path.write_text(prompt["user"], encoding="utf-8")
        process = run_cli(
            [
                "--package",
                str(package),
                "--prompt-file",
                str(prompt_path),
                "--max-new-tokens",
                str(max_new_tokens),
            ],
            timeout_seconds=timeout_seconds,
            runner=runner,
        )
        (output / f"isolated-{index}.stderr").write_text(
            process.stderr,
            encoding="utf-8",
        )
        (output / f"isolated-{index}.json").write_text(
            process.stdout,
            encoding="utf-8",
        )
        isolated = parse_success(process, f"isolated CLI {index}")
        for field in (
            "contract_sha256",
            "model_id",
            "package_sha256",
            "runtime",
            "schema_version",
            "status",
        ):
            require(
                isolated.get(field) == batch.get(field),
                f"isolated CLI {index} metadata differs: {field}",
            )
        isolated_generation = isolated.get("generation")
        require(
            isinstance(isolated_generation, dict),
            f"isolated CLI {index} generation is invalid",
        )
        require(
            isolated_generation.get("prompt_id") == "stdin-or-file",
            f"isolated CLI {index} prompt identity differs",
        )
        normalized = normalized_generation(
            isolated_generation,
            prompt["prompt_id"],
        )
        require(
            generations[index] == normalized,
            f"batch and isolated CLI records differ for {prompt['prompt_id']}",
        )
        validate_fresh_cache(generations[index], prompt["prompt_id"])

    malformed_path = output / "malformed-duplicate-id.json"
    write_json(
        malformed_path,
        [
            {"prompt_id": "duplicate", "user": "first"},
            {"prompt_id": "duplicate", "user": "second"},
        ],
    )
    missing_package = output / "must-not-be-opened.ace2w4m1"
    malformed_process = run_cli(
        [
            "--package",
            str(missing_package),
            "--prompt-batch-file",
            str(malformed_path),
            "--max-new-tokens",
            str(max_new_tokens),
        ],
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    (output / "malformed.stderr").write_text(
        malformed_process.stderr,
        encoding="utf-8",
    )
    require(malformed_process.returncode == 2, "malformed batch was not rejected")
    require(
        "duplicate prompt_id" in malformed_process.stderr,
        "malformed batch rejection reason differs",
    )
    require(
        "package" not in malformed_process.stderr.lower(),
        "malformed batch reached package validation",
    )
    require(
        not missing_package.exists(),
        "malformed batch unexpectedly materialized the package sentinel",
    )

    result = {
        "checks": {
            "batch_matches_isolated_records": True,
            "fresh_per_prompt_kv_state": True,
            "malformed_input_rejected_before_package_and_runtime": True,
            "ordered_prompt_records": True,
        },
        "max_new_tokens": max_new_tokens,
        "package_sha256": observed_package_sha256,
        "prompt_ids": [prompt["prompt_id"] for prompt in PROMPTS],
        "schema_version": 1,
        "status": "PASS",
    }
    write_json(output / "qualification.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one file-backed grouped-W4A8 batch process with fresh "
            "single-prompt CLI processes."
        )
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = qualify(
            args.package,
            args.output,
            args.expected_package_sha256,
            args.max_new_tokens,
            args.timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"ACE2_W4A8_BATCH_QUALIFICATION_ERROR: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
