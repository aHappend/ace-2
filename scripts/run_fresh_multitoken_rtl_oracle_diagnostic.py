#!/usr/bin/env python3
"""Run one fresh multi-token RTL trajectory and its independent oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_stage1_chat_product as product
from tools import ace2_v73_stage1_chat_attempt as base_attempt
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation


EXPECTED_LAYERS = 24
EXPECTED_MODEL_OUTPUTS = 151_936
MAX_CONTEXT_TOKENS = 40
MIN_GENERATED_TOKENS = 4
MAX_GENERATED_TOKENS = 8
DEFAULT_GENERATED_TOKENS = 8


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def validate_generated_tokens(value: object) -> int:
    require(
        type(value) is int
        and MIN_GENERATED_TOKENS <= value <= MAX_GENERATED_TOKENS,
        "generated-token count must be an integer from 4 through 8",
    )
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT), f"artifact is outside project: {path}")
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_diagnostic_generation_bounds(
    prompt_token_count: int,
    max_new_tokens: int,
) -> dict[str, int]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    require(prompt_token_count > 0, "prompt tokenizer output must be non-empty")
    total_context_tokens = prompt_token_count + max_new_tokens
    require(
        total_context_tokens <= MAX_CONTEXT_TOKENS,
        (
            f"prompt plus generation requires {total_context_tokens} total tokens; "
            f"diagnostic context bound is {MAX_CONTEXT_TOKENS}"
        ),
    )
    return {
        "prompt_token_count": prompt_token_count,
        "max_prompt_tokens": MAX_CONTEXT_TOKENS - max_new_tokens,
        "max_new_tokens": max_new_tokens,
        "total_context_tokens": total_context_tokens,
        "processed_positions": total_context_tokens - 1,
        "rtl_context_bound": MAX_CONTEXT_TOKENS,
    }


def diagnostic_tokenizer_record(
    prompt: str,
    snapshot: Path,
    max_new_tokens: int,
) -> tuple[dict[str, Any], Any]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    record, tokenizer = generation.tokenizer_record(
        prompt,
        MAX_CONTEXT_TOKENS - generation.MAX_NEW_TOKENS,
        generation.MAX_NEW_TOKENS,
        snapshot,
    )
    bounds = validate_diagnostic_generation_bounds(
        len(record["prompt_token_ids"]),
        max_new_tokens,
    )
    record["max_prompt_tokens"] = bounds["max_prompt_tokens"]
    record["generation_bounds"] = bounds
    record["domain"]["accepted_prompt_token_count"] = [
        1,
        bounds["max_prompt_tokens"],
    ]
    record["domain"]["sha256"] = generation.identity_sha256(record["domain"])
    record["identity_sha256"] = generation.identity_sha256(
        {key: value for key, value in record.items() if key != "identity_sha256"}
    )
    return record, tokenizer


def diagnostic_source_binding() -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        Path(product.__file__).resolve(),
        Path(backend.__file__).resolve(),
        Path(generation.__file__).resolve(),
    ]
    paths.extend(sorted((ROOT / "rtl").rglob("*.sv")))
    paths.extend(sorted((ROOT / "rtl").rglob("*.svh")))
    records = [base_attempt.file_record(path) for path in sorted(set(paths))]
    return {
        "algorithm": "sha256(path-nul-file-sha256-newline)-v1",
        "files": records,
        "sha256": base_attempt.sha256_bytes(
            b"".join(
                record["path"].encode("utf-8")
                + b"\0"
                + record["sha256"].encode("ascii")
                + b"\n"
                for record in records
            )
        ),
    }


def build_attempt_worker_command(
    prompt_path: Path,
    runtime_output: Path,
    worker_result: Path,
    prompt_token_count: int,
    max_new_tokens: int,
) -> list[str]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    validate_diagnostic_generation_bounds(prompt_token_count, max_new_tokens)
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        str(Path(__file__).resolve()),
        "--attempt-worker",
        "--worker-prompt-file",
        str(prompt_path),
        "--worker-runtime-output",
        str(runtime_output),
        "--worker-result",
        str(worker_result),
        "--max-new-tokens",
        str(max_new_tokens),
    ]


def build_independent_oracle_command(
    attempt: Path,
    output: Path,
    max_new_tokens: int,
) -> list[str]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        str(ROOT / "scripts/verify_full_chain_independent_oracle.py"),
        "--attempt",
        str(attempt),
        "--output",
        str(output),
        "--expected-generated-tokens",
        str(max_new_tokens),
    ]


def run_diagnostic_product(
    prompt: str,
    output: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    require(bool(prompt.strip()), "prompt must contain non-whitespace UTF-8 text")
    require(not output.exists(), "diagnostic product output must be a fresh path")
    position01 = product.verify_accepted_position01()
    snapshot = generation.resolve_snapshot()
    tokenization, tokenizer = diagnostic_tokenizer_record(
        prompt,
        snapshot,
        max_new_tokens,
    )
    validate_diagnostic_generation_bounds(
        len(tokenization["prompt_token_ids"]),
        max_new_tokens,
    )
    with backend.process_tree_rss_tracking(interval_seconds=0.01) as tracker:
        summary = backend.run_generation(
            output,
            snapshot,
            tokenizer,
            tokenization["prompt_token_ids"],
            max_new_tokens,
            execution_mode="stage1_product",
        )
    process_tree = tracker.summary(require_complete=True)
    readability = product.validate_backend_summary(summary, prompt)
    backend.write_json(output / "run_summary.json", summary)
    result = {
        "schema_version": 1,
        "status": (
            "PASS_STAGE1_RTL_CHAT_PRODUCT_COHERENT"
            if readability["accepted"]
            else "BLOCKED_INCOHERENT_STAGE1_RTL_CHAT_OUTPUT"
        ),
        "prompt": {
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "utf8_bytes": len(prompt.encode("utf-8")),
            "text_persisted": False,
        },
        "tokenization": tokenization,
        "generated_token_ids": summary["generated_token_ids"],
        "decoded_text": summary["decoded_text"],
        "readability": readability,
        "single_session": True,
        "kv_continuity": True,
        "rtl_corrected_v_provenance": summary["rtl_corrected_v_provenance"],
        "software_transformer_or_logits_fallback": False,
        "latency": summary["latency"],
        "process_tree_rss": process_tree,
        "accepted_position01": position01,
        "run_summary": {
            "path": "run_summary.json",
            "sha256": sha256_file(output / "run_summary.json"),
        },
    }
    product.write_terminal_result(output, result, summary, readability)
    return result


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_attempt_worker(
    prompt_path: Path,
    runtime_output: Path,
    worker_result: Path,
    max_new_tokens: int,
) -> int:
    prompt = prompt_path.read_text(encoding="utf-8")
    result = run_diagnostic_product(prompt, runtime_output, max_new_tokens)
    base_attempt.write_exclusive(worker_result, result)
    return 0 if result["readability"]["accepted"] else 5


def run_multitoken_attempt(
    prompt: str,
    output: Path,
    timeout_seconds: int,
    max_new_tokens: int,
) -> int:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    output = output.resolve()
    require(not output.exists(), f"attempt output already exists: {output}")
    require(timeout_seconds > 0, "--timeout-seconds must be positive")
    position01 = product.verify_accepted_position01()
    snapshot = generation.resolve_snapshot()
    tokenization, _tokenizer = diagnostic_tokenizer_record(
        prompt,
        snapshot,
        max_new_tokens,
    )

    output.mkdir(parents=True)
    prompt_path = output / "prompt.utf8"
    base_attempt.write_exclusive(prompt_path, prompt.encode("utf-8"))
    runtime_output = output / "runtime-output"
    worker_result = output / "worker-result.json"
    stdout_path = output / "stdout.raw.log"
    stderr_path = output / "stderr.raw.log"
    command = build_attempt_worker_command(
        prompt_path,
        runtime_output,
        worker_result,
        len(tokenization["prompt_token_ids"]),
        max_new_tokens,
    )
    manifest = {
        "schema": "ace2-multitoken-rtl-diagnostic-attempt-manifest-v1",
        "manager": "fresh-multitoken-diagnostic",
        "status": "FROZEN_BEFORE_EXECUTION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "timeout_seconds": timeout_seconds,
        "pass_policy": (
            f"PASS only after exactly {max_new_tokens} RTL-selected tokens, "
            f"{max_new_tokens - 1} generated-token feedback transitions, continuous "
            "24-layer persistent K/V, full-vocabulary head agreement, and "
            "prompt-compatible readable detokenization"
        ),
        "accepted_position01": position01,
        "prompt": base_attempt.file_record(prompt_path),
        "tokenization": tokenization,
        "model": base_attempt.file_record(snapshot / "model.safetensors"),
        "model_config": base_attempt.file_record(snapshot / "config.json"),
        "adapter": base_attempt.file_record(backend.canonical.ADAPTER.resolve()),
        "source_and_rtl": diagnostic_source_binding(),
        "command": {
            "argv": command,
            "argv_sha256": base_attempt.sha256_bytes(
                base_attempt.canonical_bytes(command)
            ),
            "cwd": str(ROOT),
        },
        "tools": {
            "python": base_attempt.tool_record(
                [str(Path(sys.executable).resolve()), "--version"]
            ),
            "iverilog": base_attempt.tool_record(["/usr/bin/iverilog", "-V"]),
            "vvp": base_attempt.tool_record(["/usr/bin/vvp", "-V"]),
        },
        "environment": {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    manifest["environment_sha256"] = base_attempt.sha256_bytes(
        base_attempt.canonical_bytes(manifest["environment"])
    )
    base_attempt.write_exclusive(output / "attempt-manifest.json", manifest)
    base_attempt.write_exclusive(output / "command.json", manifest["command"])

    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            exit_code = process.returncode
    elapsed = time.monotonic() - started

    comparison: dict[str, Any] | None = None
    product_result: dict[str, Any] | None = None
    if worker_result.is_file():
        product_result = load_json(worker_result)
    if (runtime_output / "run_summary.json").is_file():
        comparison = base_attempt.independent_comparison(runtime_output)
        base_attempt.write_exclusive(
            output / "independent-reference-comparison.json",
            comparison,
        )
    if timed_out:
        status = "TIMEOUT"
    elif (
        exit_code == 0
        and product_result is not None
        and product_result.get("readability", {}).get("accepted") is True
        and comparison is not None
        and comparison["status"] == "PASS"
    ):
        status = "PASS"
    else:
        status = "FAIL"
    result = {
        "schema": "ace2-multitoken-rtl-diagnostic-attempt-result-v1",
        "status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_wall_seconds": elapsed,
        "generated_token_ids": (
            product_result.get("generated_token_ids", []) if product_result else []
        ),
        "decoded_text": (
            product_result.get("decoded_text", "") if product_result else ""
        ),
        "independent_comparison_status": (
            comparison.get("status") if comparison is not None else "NOT_REACHED"
        ),
    }
    base_attempt.write_exclusive(output / "attempt-result.json", result)
    base_attempt.write_exclusive(
        output / "timing.json",
        {
            "supervisor_total_wall_seconds": elapsed,
            "backend_latency": (
                product_result.get("latency") if product_result is not None else None
            ),
        },
    )
    base_attempt.seal_attempt(output)
    print(
        f"ACE2_MULTITOKEN_DIAGNOSTIC_ATTEMPT status={status} "
        f"tokens={len(result['generated_token_ids'])} "
        f"total_seconds={elapsed:.6f} output={output.relative_to(ROOT)}"
    )
    return 0 if status == "PASS" else (124 if status == "TIMEOUT" else 1)


def read_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt_file is not None:
        try:
            value = prompt_file.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise DiagnosticError("--prompt-file must contain valid UTF-8") from error
    elif prompt is not None:
        value = prompt
    elif not sys.stdin.isatty():
        try:
            value = sys.stdin.buffer.read().decode("utf-8")
        except UnicodeDecodeError as error:
            raise DiagnosticError("stdin must contain valid UTF-8") from error
    else:
        raise DiagnosticError("provide --prompt, --prompt-file, or UTF-8 stdin")
    require(bool(value.strip()), "prompt must contain non-whitespace UTF-8 text")
    return value


def validate_success(
    prompt: str,
    oracle_result: dict[str, Any],
    summary: dict[str, Any],
    product_result: dict[str, Any],
    expected_generated_tokens: int,
) -> dict[str, Any]:
    expected_generated_tokens = validate_generated_tokens(
        expected_generated_tokens
    )
    require(oracle_result.get("status") == "PASS", "independent oracle did not pass")
    require(
        oracle_result.get("numerical_status") == "PASS",
        "independent numerical status did not pass",
    )
    agreement = oracle_result.get("agreement")
    require(isinstance(agreement, dict), "oracle agreement is absent")
    prompt_ids = summary.get("prompt_token_ids")
    generated = summary.get("generated_token_ids")
    executions = summary.get("token_executions")
    head_steps = summary.get("head_steps")
    context = summary.get("context_contract")
    require(
        isinstance(prompt_ids, list) and prompt_ids,
        "RTL summary has no prompt token sequence",
    )
    require(
        isinstance(generated, list)
        and len(generated) == expected_generated_tokens,
        (
            "RTL trajectory must contain exactly "
            f"{expected_generated_tokens} generated tokens"
        ),
    )
    require(
        agreement.get("generated_token_ids") == generated,
        "oracle and RTL selected-token sequences differ",
    )
    require(
        summary.get("software_transformer_or_logits_fallback") is False,
        "software transformer/logits fallback was not explicitly false",
    )
    require(
        isinstance(executions, list)
        and len(executions) == len(prompt_ids) + len(generated) - 1,
        "RTL trajectory position count is not causal",
    )
    require(
        isinstance(head_steps, list) and len(head_steps) == len(generated),
        "RTL trajectory does not contain one head step per generated token",
    )
    require(
        isinstance(context, dict)
        and context.get("prompt_token_count") == len(prompt_ids)
        and context.get("max_new_tokens") == expected_generated_tokens
        and context.get("total_context_tokens") == len(prompt_ids) + len(generated)
        and context.get("processed_positions") == len(executions)
        and isinstance(context.get("backend_context_bound"), int)
        and context["backend_context_bound"] >= MAX_CONTEXT_TOKENS
        and context["total_context_tokens"] <= MAX_CONTEXT_TOKENS,
        "RTL trajectory does not satisfy the 40-token context contract",
    )
    require(
        product_result.get("status") == "PASS_STAGE1_RTL_CHAT_PRODUCT_COHERENT"
        and product_result.get("readability", {}).get("accepted") is True,
        "RTL trajectory did not pass prompt-compatible readability",
    )
    require(
        product_result.get("generated_token_ids") == generated
        and product_result.get("decoded_text") == agreement.get("decoded_text"),
        "readable product result differs from the RTL/oracle trajectory",
    )
    require(
        product_result.get("software_transformer_or_logits_fallback") is False,
        "product result did not explicitly reject software transformer/logits fallback",
    )

    layer_growth: list[list[dict[str, int]]] = [
        [] for _ in range(EXPECTED_LAYERS)
    ]
    phase_counts = {"prefill": 0, "decode": 0}
    for position, execution in enumerate(executions):
        expected_phase = "prefill" if position < len(prompt_ids) else "decode"
        require(
            execution.get("absolute_position") == position,
            "RTL trajectory positions are not contiguous",
        )
        require(
            execution.get("phase") == expected_phase,
            "RTL trajectory prefill/decode phase differs",
        )
        layers = execution.get("layers")
        require(
            isinstance(layers, list) and len(layers) == EXPECTED_LAYERS,
            "RTL trajectory position does not cover all 24 layers",
        )
        phase_counts[expected_phase] += 1
        for layer_id, layer in enumerate(layers):
            require(layer.get("layer_id") == layer_id, "RTL layer order differs")
            before = layer.get("cache_length_before")
            after = layer.get("cache_length_after")
            require(
                before == position and after == position + 1,
                f"layer {layer_id} cache length is discontinuous at position {position}",
            )
            layer_growth[layer_id].append(
                {
                    "absolute_position": position,
                    "cache_length_before": before,
                    "cache_length_after": after,
                }
            )

    require(
        phase_counts["decode"] == expected_generated_tokens - 1,
        (
            "RTL trajectory must contain exactly "
            f"{expected_generated_tokens - 1} decode transitions"
        ),
    )
    require(
        agreement.get("positions_checked") == len(executions)
        and agreement.get("layers_checked") == len(executions) * EXPECTED_LAYERS,
        "independent oracle did not check every position and layer",
    )
    require(
        agreement.get("integer_byte_mismatches") == 0,
        "independent oracle reported integer-byte mismatches",
    )
    require(
        agreement.get("selected_token_mismatches") == 0,
        "independent oracle reported selected-token mismatches",
    )
    required_byte_counts = (
        "kv_append_bytes_compared",
        "full_cache_bytes_compared",
        "quantized_layer_output_bytes_compared",
        "quantized_layer_output_scale_bytes_compared",
        "final_rmsnorm_bytes_compared",
        "final_rmsnorm_scale_bytes_compared",
        "full_vocabulary_logit_bytes_compared",
        "full_vocabulary_logit_scale_bytes_compared",
        "retained_payload_records_compared",
        "retained_payload_bytes_compared",
    )
    for field in required_byte_counts:
        require(
            isinstance(agreement.get(field), int) and agreement[field] > 0,
            f"independent oracle did not compare {field}",
        )
    retained_byte_fields = (
        "quantized_layer_output_bytes_compared",
        "quantized_layer_output_scale_bytes_compared",
        "final_rmsnorm_bytes_compared",
        "final_rmsnorm_scale_bytes_compared",
        "full_vocabulary_logit_bytes_compared",
        "full_vocabulary_logit_scale_bytes_compared",
    )
    require(
        agreement["retained_payload_bytes_compared"]
        == sum(agreement[field] for field in retained_byte_fields),
        "independent oracle retained-byte comparison is incomplete",
    )
    require(
        agreement.get("full_vocabulary_head_steps_checked") == len(generated),
        "independent oracle did not check every full-vocabulary head step",
    )
    require(
        oracle_result.get("constraints", {}).get("model_output_domain")
        == EXPECTED_MODEL_OUTPUTS,
        "oracle output domain is not the full model vocabulary",
    )
    require(
        oracle_result.get("reused_inputs", {}).get("prompt", {}).get("sha256")
        == hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "oracle prompt binding differs from the requested prompt",
    )
    return {
        "prefill_positions": phase_counts["prefill"],
        "decode_transitions": phase_counts["decode"],
        "positions_checked": len(executions),
        "layers_per_position": EXPECTED_LAYERS,
        "layer_cache_growth": [
            {"layer_id": layer_id, "positions": positions}
            for layer_id, positions in enumerate(layer_growth)
        ],
        "generated_token_ids": generated,
        "decoded_text": agreement.get("decoded_text"),
        "kv_append_bytes_compared": agreement["kv_append_bytes_compared"],
        "full_cache_bytes_compared": agreement["full_cache_bytes_compared"],
        "layer_output_bytes_compared": agreement[
            "quantized_layer_output_bytes_compared"
        ],
        "full_vocabulary_head_steps_checked": agreement[
            "full_vocabulary_head_steps_checked"
        ],
        "full_vocabulary_logit_bytes_compared": agreement[
            "full_vocabulary_logit_bytes_compared"
        ],
        "retained_payload_records_compared": agreement[
            "retained_payload_records_compared"
        ],
        "retained_payload_bytes_compared": agreement[
            "retained_payload_bytes_compared"
        ],
        "prompt_compatible_readability": True,
        "integer_byte_mismatches": 0,
        "selected_token_mismatches": 0,
        "software_transformer_or_logits_fallback": False,
    }


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        ).returncode


def write_report(output: Path, result: dict[str, Any]) -> None:
    coverage = result.get("coverage")
    lines = [
        "# Fresh multi-token RTL/oracle diagnostic",
        "",
        f"**Status:** {result['status']}",
        "",
        "This is a computer-local RTL-simulation and host-oracle diagnostic. It is not "
        "an FPGA, synthesis, PPA, deployed-hardware, or silicon result.",
        "",
        "## Reproduction",
        "",
        "```bash",
        result["reproduction_command"],
        "```",
        "",
    ]
    if coverage is not None:
        lines.extend(
            [
                "## Coverage",
                "",
                f"- Prefill positions: {coverage['prefill_positions']}",
                f"- Decode transitions: {coverage['decode_transitions']}",
                f"- Positions / layers per position: {coverage['positions_checked']} / 24",
                f"- Full-vocabulary head steps: {coverage['full_vocabulary_head_steps_checked']}",
                f"- K/V append / full-cache bytes compared: {coverage['kv_append_bytes_compared']} / {coverage['full_cache_bytes_compared']}",
                f"- Layer-output bytes compared: {coverage['layer_output_bytes_compared']}",
                f"- Full-vocabulary logit bytes compared: {coverage['full_vocabulary_logit_bytes_compared']}",
                f"- Retained payload records / bytes compared: {coverage['retained_payload_records_compared']} / {coverage['retained_payload_bytes_compared']}",
                "- Prompt-compatible readability: true",
                "- Integer-byte / selected-token mismatches: 0 / 0",
                "- Software transformer/logits fallback: false",
                "",
            ]
        )
    else:
        failure = result["failure"]
        lines.extend(
            [
                "## Failure",
                "",
                f"- Stage: `{failure['stage']}`",
                f"- Return code: {failure['returncode']}",
                "",
            ]
        )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(output: Path, result: dict[str, Any]) -> None:
    (output / "result.json").write_bytes(canonical_bytes(result))
    write_report(output, result)
    members = (output / "prompt.utf8", output / "result.json", output / "REPORT.md")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in members),
        encoding="ascii",
    )


def run(
    prompt: str,
    output: Path,
    timeout_seconds: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    output = output.resolve()
    require(
        output.is_relative_to(ROOT / "reports" / "verification"),
        "--output must be under reports/verification",
    )
    require(not output.exists(), f"diagnostic output already exists: {output}")
    require(timeout_seconds > 0, "--timeout-seconds must be positive")
    output.mkdir(parents=True)
    prompt_path = output / "prompt.utf8"
    prompt_path.write_text(prompt, encoding="utf-8")
    attempt = output / "rtl-attempt"
    oracle_output = output / "independent-oracle"
    python = str(Path(sys.executable).resolve())
    attempt_command = [
        python,
        "-B",
        str(Path(__file__).resolve()),
        "--attempt",
        "--prompt-file",
        str(prompt_path),
        "--attempt-output",
        str(attempt),
        "--max-new-tokens",
        str(max_new_tokens),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    oracle_command = build_independent_oracle_command(
        attempt,
        oracle_output,
        max_new_tokens,
    )
    reproduction = (
        "python3 -B scripts/run_fresh_multitoken_rtl_oracle_diagnostic.py "
        f"--prompt-file <utf8-file> --max-new-tokens {max_new_tokens} "
        f"--output {output.relative_to(ROOT).as_posix()}"
    )
    for stage, command in (
        ("rtl_trajectory", attempt_command),
        ("independent_oracle", oracle_command),
    ):
        stdout_path = output / f"{stage}.stdout.log"
        stderr_path = output / f"{stage}.stderr.log"
        returncode = run_logged(command, stdout_path, stderr_path)
        if returncode != 0:
            result = {
                "schema": "ace2-fresh-multitoken-rtl-oracle-diagnostic-v1",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "status": "FAIL",
                "reproduction_command": reproduction,
                "failure": {
                    "stage": stage,
                    "returncode": returncode,
                    "stdout": file_record(stdout_path),
                    "stderr": file_record(stderr_path),
                },
                "software_transformer_or_logits_fallback": None,
            }
            finalize(output, result)
            return result

    oracle_result_path = oracle_output / "result.json"
    summary_path = attempt / "runtime-output/run_summary.json"
    product_result_path = attempt / "worker-result.json"
    oracle_result = load_json(oracle_result_path)
    summary = load_json(summary_path)
    product_result = load_json(product_result_path)
    coverage = validate_success(
        prompt,
        oracle_result,
        summary,
        product_result,
        max_new_tokens,
    )
    result = {
        "schema": "ace2-fresh-multitoken-rtl-oracle-diagnostic-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "reproduction_command": reproduction,
        "prompt": file_record(prompt_path),
        "rtl_attempt": {
            "path": attempt.relative_to(ROOT).as_posix(),
            "tree_root": file_record(attempt / "TREE_ROOT.sha256"),
            "run_summary": file_record(summary_path),
            "product_result": file_record(product_result_path),
        },
        "independent_oracle": {
            "path": oracle_output.relative_to(ROOT).as_posix(),
            "result": file_record(oracle_result_path),
            "checksums": file_record(oracle_output / "SHA256SUMS"),
        },
        "coverage": coverage,
        "software_transformer_or_logits_fallback": False,
        "measurement_scope": "computer-local RTL simulation and host oracle only",
    }
    finalize(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=72_000)
    parser.add_argument("--attempt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-prompt-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-runtime-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_GENERATED_TOKENS,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.attempt_worker:
        require(
            all(
                (
                    args.worker_prompt_file,
                    args.worker_runtime_output,
                    args.worker_result,
                    args.max_new_tokens,
                )
            ),
            "attempt worker paths and generation count are required",
        )
        return run_attempt_worker(
            args.worker_prompt_file,
            args.worker_runtime_output,
            args.worker_result,
            args.max_new_tokens,
        )
    prompt = read_prompt(args.prompt, args.prompt_file)
    if args.attempt:
        require(args.attempt_output is not None, "--attempt-output is required")
        return run_multitoken_attempt(
            prompt,
            args.attempt_output,
            args.timeout_seconds,
            args.max_new_tokens,
        )
    require(args.output is not None, "--output is required")
    result = run(
        prompt,
        args.output,
        args.timeout_seconds,
        args.max_new_tokens,
    )
    coverage = result.get("coverage", {})
    print(
        "FRESH_MULTITOKEN_RTL_ORACLE_DIAGNOSTIC_"
        f"{result['status']} decode_transitions={coverage.get('decode_transitions', 0)} "
        f"integer_mismatches={coverage.get('integer_byte_mismatches', 'unknown')} "
        f"token_mismatches={coverage.get('selected_token_mismatches', 'unknown')} "
        f"software_fallback={result['software_transformer_or_logits_fallback']} "
        f"output={args.output}"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"FRESH_MULTITOKEN_RTL_ORACLE_DIAGNOSTIC_FAIL detail={error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
