#!/usr/bin/env python3
"""Execute and seal the V74 arbitrary-text multi-token Stage-1 chat attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_stage1_chat_product as product
from tools import ace2_v73_stage1_chat_attempt as base
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation


ATTEMPT001 = ROOT / "reports/ace2-stage1-chat-attempt-0001"
ATTEMPT001_TERMINAL = ROOT / "reports/ace2-stage1-chat-attempt-0001-v74-terminal.json"
DEFAULT_OUTPUT = ROOT / "reports/ace2-stage1-chat-attempt-0002"
DEFAULT_TIMEOUT_SECONDS = 64_800
MAX_NEW_TOKENS = 2
MIN_FREE_BYTES = 13_257_845_760
EXPECTED_PROMPT_SHA256 = "017ff7058fb8400edad6f6f2026627c0ffd176edcc476434084b97b9535e74c2"
EXPECTED_COMPLETE_FRONTIER = tuple(
    [(position, layer) for position in range(3) for layer in range(backend.LAYERS)]
    + [(3, layer) for layer in range(8)]
)
LAYER_COMPLETION_MARKERS = tuple(
    f"logs/{name}.vvp.result.json"
    for name in (
        "rmsnorm",
        "projection",
        "rope",
        "score",
        "softmax",
        "compose",
        "silu",
        "residual",
    )
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def attempt_tree_snapshot() -> dict[str, Any]:
    require(ATTEMPT001.is_dir(), "attempt001 is absent")
    members = sorted(path for path in ATTEMPT001.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in members:
        relative = path.relative_to(ATTEMPT001).as_posix()
        member_sha256 = base.sha256_file(path)
        total_bytes += path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(member_sha256.encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256(path-nul-file-sha256-newline)-v1",
        "file_count": len(members),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def completed_attempt001_frontier() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    token_root = ATTEMPT001 / "runtime-output/tokens"
    complete: list[tuple[int, int]] = []
    incomplete: list[tuple[int, int]] = []
    for position_dir in sorted(token_root.glob("position-*")):
        position = int(position_dir.name.removeprefix("position-"))
        for layer_dir in sorted(position_dir.glob("layer-*")):
            layer = int(layer_dir.name.removeprefix("layer-"))
            target = complete if all(
                (layer_dir / marker).is_file() for marker in LAYER_COMPLETION_MARKERS
            ) else incomplete
            target.append((position, layer))
    return complete, incomplete


def terminalize_attempt001() -> dict[str, Any]:
    snapshot = attempt_tree_snapshot()
    complete, incomplete = completed_attempt001_frontier()
    require(
        tuple(complete) == EXPECTED_COMPLETE_FRONTIER,
        f"attempt001 complete frontier changed: {complete}",
    )
    require(
        incomplete == [(3, 8)],
        f"attempt001 partial frontier changed: {incomplete}",
    )
    if ATTEMPT001_TERMINAL.is_file():
        receipt = json.loads(ATTEMPT001_TERMINAL.read_text(encoding="utf-8"))
        require(
            receipt.get("attempt001_tree_snapshot") == snapshot,
            "attempt001 changed after its V74 terminal receipt",
        )
        require(
            receipt.get("status") == "ABORTED_RESOURCE_BUDGET",
            "attempt001 V74 terminal status changed",
        )
        return receipt
    receipt = {
        "schema": "ace2-v74-stage1-chat-attempt001-terminal-v1",
        "manager": "V74",
        "status": "ABORTED_RESOURCE_BUDGET",
        "pass": False,
        "timeout": False,
        "process_exit_code": 143,
        "terminating_signal": "SIGTERM",
        "externally_observed_elapsed_wall_seconds": 4869.9,
        "elapsed_observation_source": "active_manager_v74_operator_directive",
        "configured_timeout_seconds": 21_600,
        "timeout_feasibility": {
            "complete_positions_observed": 3,
            "planned_rtl_execution_positions": 35,
            "planned_context_tokens": 36,
            "extrapolated_35_position_seconds": 56_815.5,
            "extrapolated_36_context_token_seconds": 58_438.8,
            "configured_timeout_infeasible": True,
        },
        "completed_rtl_layer_executions": len(complete),
        "complete_positions": [0, 1, 2],
        "partial_position": {
            "absolute_position": 3,
            "complete_layers": list(range(8)),
            "incomplete_layer_directory": 8,
        },
        "not_reached": [
            "generated_token_ids",
            "official_tokenizer_detokenization",
            "complete_timing",
            "independent_comparison",
        ],
        "attempt001_mutated": False,
        "attempt001_replayed": False,
        "terminal_receipt_location": "adjacent_to_immutable_attempt001_tree",
        "attempt001_tree_snapshot": snapshot,
    }
    base.write_exclusive(ATTEMPT001_TERMINAL, receipt)
    ATTEMPT001_TERMINAL.chmod(
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    )
    return receipt


def source_binding(baseline_manifest: dict[str, Any]) -> dict[str, Any]:
    baseline_records = {
        record["path"]: record
        for record in baseline_manifest["source_and_rtl"]["files"]
        if record["path"] != "tools/ace2_v73_stage1_chat_attempt.py"
    }
    records = []
    for relative, expected in sorted(baseline_records.items()):
        path = ROOT / relative
        require(path.is_file(), f"frozen baseline source is absent: {relative}")
        current = base.file_record(path)
        require(current == expected, f"frozen baseline source changed: {relative}")
        records.append(current)
    records.append(base.file_record(Path(__file__).resolve()))
    records.append(base.file_record(Path(base.__file__).resolve()))
    records.sort(key=lambda record: record["path"])
    return {
        "algorithm": "sha256(path-nul-file-sha256-newline)-v1",
        "files": records,
        "sha256": base.sha256_bytes(
            b"".join(
                record["path"].encode("utf-8")
                + b"\0"
                + record["sha256"].encode("ascii")
                + b"\n"
                for record in records
            )
        ),
        "unchanged_v73_execution_baseline_sha256": baseline_manifest[
            "source_and_rtl"
        ]["sha256"],
    }


def public_interface_record() -> dict[str, Any]:
    shell = ROOT / "rtl/ace2_shell.sv"
    source = shell.read_text(encoding="utf-8")
    start = source.index("module ace2_shell #(")
    end = source.index("\n);", start)
    header = source[start:end]
    parameters = re.findall(
        r"^\s*parameter\s+integer\s+([A-Za-z_][A-Za-z0-9_]*)\s*=",
        header,
        flags=re.MULTILINE,
    )
    ports = re.findall(
        r"^\s*(?:input|output)\s+(?:wire|reg)\s+"
        r"(?:\[[^\n]+\]\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*,?\s*$",
        header,
        flags=re.MULTILINE,
    )
    require(len(parameters) == 14, f"ace2_shell parameter count is {len(parameters)}, not 14")
    require(len(ports) == 64, f"ace2_shell port count is {len(ports)}, not 64")

    rtl_sources = sorted((ROOT / "rtl").glob("*.sv"))
    package = ROOT / "rtl/ace2_pkg.sv"
    rtl_sources = [package] + [path for path in rtl_sources if path != package]
    command = [
        "/usr/bin/iverilog",
        "-g2012",
        "-I",
        str(ROOT / "rtl"),
        "-I",
        str(ROOT / "rtl/generated"),
        "-s",
        "ace2_shell",
        "-tnull",
        *(str(path) for path in rtl_sources),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    require(
        completed.returncode == 0,
        f"ace2_shell public-contract compile failed: {completed.stderr}",
    )
    interface_manifest = ROOT / "design/BENCHMARK_INTERFACE.json"
    provenance = ROOT / "design/RTL_ACTIVE_CLOSURE_PROVENANCE.json"
    require(interface_manifest.is_file(), "public interface manifest is absent")
    require(provenance.is_file(), "active RTL closure provenance is absent")
    return {
        "top": "ace2_shell",
        "parameter_count": len(parameters),
        "parameters": parameters,
        "port_count": len(ports),
        "ports": ports,
        "compatibility_aliases": False,
        "shell": base.file_record(shell),
        "interface_manifest": base.file_record(interface_manifest),
        "active_closure_provenance": base.file_record(provenance),
        "compile": {
            "argv": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    }


def active_attempt_workers() -> list[dict[str, Any]]:
    workers = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        decoded = [argument.decode("utf-8", errors="replace") for argument in argv if argument]
        if "--worker-runtime-output" not in decoded:
            continue
        if any(
            "reports/ace2-stage1-chat-attempt-000" in argument
            for argument in decoded
        ):
            workers.append({"pid": int(proc.name), "argv": decoded})
    return workers


def readiness() -> dict[str, Any]:
    require(not DEFAULT_OUTPUT.exists(), f"attempt002 output already exists: {DEFAULT_OUTPUT}")
    workers = active_attempt_workers()
    require(not workers, f"an earlier Stage-1 attempt worker is active: {workers}")
    terminal = terminalize_attempt001()
    baseline_manifest = json.loads(
        (ATTEMPT001 / "attempt-manifest.json").read_text(encoding="utf-8")
    )
    require(
        baseline_manifest.get("manager") == "V73",
        "attempt001 baseline manifest is not V73",
    )
    prompt_path = ATTEMPT001 / "prompt.utf8"
    require(
        base.sha256_file(prompt_path) == EXPECTED_PROMPT_SHA256,
        "official attempt001 prompt changed",
    )
    source = source_binding(baseline_manifest)
    interface = public_interface_record()
    position01 = product.verify_accepted_position01()
    tools = {
        "python": base.tool_record([str(Path(sys.executable).resolve()), "--version"]),
        "iverilog": base.tool_record(["/usr/bin/iverilog", "-V"]),
        "vvp": base.tool_record(["/usr/bin/vvp", "-V"]),
    }
    require(
        all(record["exit_code"] == 0 for record in tools.values()),
        "a frozen execution tool is unavailable",
    )
    free_bytes = shutil.disk_usage(ROOT).free
    require(
        free_bytes >= MIN_FREE_BYTES,
        f"insufficient free storage: {free_bytes} < {MIN_FREE_BYTES}",
    )
    return {
        "status": "READY",
        "attempt001_terminal": {
            "path": ATTEMPT001_TERMINAL.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(ATTEMPT001_TERMINAL),
            "status": terminal["status"],
        },
        "accepted_position01": position01,
        "source_and_rtl": source,
        "public_interface": interface,
        "tools": tools,
        "storage": {
            "free_bytes": free_bytes,
            "required_free_bytes": MIN_FREE_BYTES,
        },
        "active_prior_workers": workers,
    }


def independent_comparison(runtime_output: Path) -> dict[str, Any]:
    comparison = base.independent_comparison(runtime_output)
    summary = json.loads((runtime_output / "run_summary.json").read_text(encoding="utf-8"))
    executions = summary["token_executions"]
    generated = summary["generated_token_ids"]
    prompt_ids = summary["prompt_token_ids"]
    layer_count = sum(len(execution["layers"]) for execution in executions)
    continuity = all(
        int(execution["absolute_position"]) == position
        and all(
            int(layer["cache_length_before"]) == position
            and int(layer["cache_length_after"]) == position + 1
            for layer in execution["layers"]
        )
        for position, execution in enumerate(executions)
    )
    token_feedback = (
        len(generated) == MAX_NEW_TOKENS
        and len(executions) == len(prompt_ids) + MAX_NEW_TOKENS - 1
        and executions[len(prompt_ids)]["phase"] == "decode"
        and int(executions[len(prompt_ids)]["input_token_id"]) == int(generated[0])
    )
    head_agreement = (
        len(summary["head_steps"]) == MAX_NEW_TOKENS
        and all(
            step.get("rtl_selected_token_agreement") is True
            for step in summary["head_steps"]
        )
    )
    checks = {
        "prompt_token_count_34": len(prompt_ids) == 34,
        "generated_token_count_2": len(generated) == MAX_NEW_TOKENS,
        "rtl_execution_position_count_35": len(executions) == 35,
        "rtl_layer_execution_count_840": layer_count == 840,
        "cache_length_continuity": continuity,
        "token0_feedback_into_token1": token_feedback,
        "full_head_agreement": head_agreement,
        "zero_integer_mismatches": comparison["total_integer_mismatches"] == 0,
        "software_transformer_or_logits_fallback_absent": (
            summary.get("software_transformer_or_logits_fallback") is False
        ),
    }
    comparison.update(
        {
            "schema": "ace2-v74-stage1-chat-independent-comparison-v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "rtl_execution_position_count": len(executions),
            "rtl_layer_execution_count": layer_count,
        }
    )
    return comparison


def failure_record(
    *,
    timed_out: bool,
    caught_signal: int | None,
    exit_code: int,
    product_result: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    execution_error: str | None,
) -> dict[str, str] | None:
    if timed_out:
        return {
            "failure_taxonomy": "resource_timeout",
            "root_cause_hypothesis": "The complete RTL session exceeded the frozen 64800-second attempt budget.",
            "regression": "Preserve this attempt and diagnose measured phase timing before any fresh repair attempt.",
        }
    if caught_signal is not None:
        return {
            "failure_taxonomy": "external_termination",
            "root_cause_hypothesis": f"The V74 supervisor received signal {caught_signal}.",
            "regression": "Preserve this attempt and verify durable-runner lifetime before any fresh repair attempt.",
        }
    if execution_error is not None or exit_code != 0:
        return {
            "failure_taxonomy": "rtl_execution_failure",
            "root_cause_hypothesis": execution_error or f"The RTL worker exited {exit_code}.",
            "regression": "Preserve this attempt and repair only the first concrete worker failure in a fresh attempt.",
        }
    if comparison is None or comparison.get("status") != "PASS":
        return {
            "failure_taxonomy": "independent_comparison_failure",
            "root_cause_hypothesis": "The preserved RTL observations did not satisfy the frozen independent comparison.",
            "regression": "Preserve this attempt and repair the first mismatching boundary in a fresh attempt.",
        }
    if product_result is None or not product_result.get("readability", {}).get("accepted"):
        return {
            "failure_taxonomy": "detokenized_output_readability",
            "root_cause_hypothesis": "The official-tokenizer two-token decode failed the frozen readability policy.",
            "regression": "Preserve this attempt and diagnose the selected full-head token sequence without relaxing readability.",
        }
    return None


def run_attempt(timeout_seconds: int) -> int:
    require(
        timeout_seconds >= DEFAULT_TIMEOUT_SECONDS,
        f"V74 timeout must be at least {DEFAULT_TIMEOUT_SECONDS} seconds",
    )
    preflight = readiness()
    prompt = (ATTEMPT001 / "prompt.utf8").read_text(encoding="utf-8")
    model_snapshot = generation.resolve_snapshot()
    tokenization, _tokenizer = generation.tokenizer_record(
        prompt,
        backend.MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS,
        MAX_NEW_TOKENS,
        model_snapshot,
    )
    baseline_manifest = json.loads(
        (ATTEMPT001 / "attempt-manifest.json").read_text(encoding="utf-8")
    )
    require(
        tokenization["prompt_token_ids"]
        == baseline_manifest["tokenization"]["prompt_token_ids"],
        "official 34-token prompt binding changed",
    )

    DEFAULT_OUTPUT.mkdir(parents=True)
    prompt_path = DEFAULT_OUTPUT / "prompt.utf8"
    base.write_exclusive(prompt_path, prompt.encode("utf-8"))
    runtime_output = DEFAULT_OUTPUT / "runtime-output"
    worker_result = DEFAULT_OUTPUT / "worker-result.json"
    stdout_path = DEFAULT_OUTPUT / "stdout.raw.log"
    stderr_path = DEFAULT_OUTPUT / "stderr.raw.log"
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-prompt-file",
        str(prompt_path),
        "--worker-runtime-output",
        str(runtime_output),
        "--worker-result",
        str(worker_result),
    ]
    environment_record = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    manifest = {
        "schema": "ace2-v74-stage1-chat-attempt-manifest-v1",
        "manager": "V74",
        "status": "FROZEN_BEFORE_EXECUTION",
        "created_at_utc": base.utc_now(),
        "timeout_seconds": timeout_seconds,
        "max_new_tokens": MAX_NEW_TOKENS,
        "expected_prompt_token_count": 34,
        "expected_rtl_execution_positions": 35,
        "expected_rtl_layer_executions": 840,
        "pass_policy": (
            "PASS only after exactly two RTL-selected tokens, token-0 feedback into "
            "token-1 decode, 840 zero-mismatch 24-layer RTL executions across 35 "
            "positions with persistent RTL-published K/V, two full-head agreements, "
            "complete timing, and readable official-tokenizer detokenization"
        ),
        "preflight": preflight,
        "prompt": base.file_record(prompt_path),
        "tokenization": tokenization,
        "model": base.file_record(model_snapshot / "model.safetensors"),
        "model_config": base.file_record(model_snapshot / "config.json"),
        "adapter": base.file_record(backend.canonical.ADAPTER.resolve()),
        "command": {
            "argv": command,
            "argv_sha256": base.sha256_bytes(base.canonical_bytes(command)),
            "cwd": str(ROOT),
        },
        "environment": environment_record,
        "environment_sha256": base.sha256_bytes(
            base.canonical_bytes(environment_record)
        ),
    }
    base.write_exclusive(DEFAULT_OUTPUT / "attempt-manifest.json", manifest)
    base.write_exclusive(DEFAULT_OUTPUT / "command.json", manifest["command"])

    environment = dict(os.environ)
    environment.update(environment_record)
    process: subprocess.Popen[bytes] | None = None
    caught_signal: int | None = None
    timed_out = False
    execution_error: str | None = None
    exit_code = 126

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal caught_signal
        caught_signal = signum
        if process is not None and process.poll() is None:
            base.terminate_process_group(process)

    previous_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    started = time.monotonic()
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            try:
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
                    base.terminate_process_group(process)
                    exit_code = process.returncode
            except (OSError, subprocess.SubprocessError) as error:
                execution_error = f"{type(error).__name__}: {error}"
                if process is not None and process.poll() is None:
                    base.terminate_process_group(process)
                exit_code = process.returncode if process is not None else 126
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    elapsed = time.monotonic() - started

    product_result: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    if worker_result.is_file():
        try:
            product_result = json.loads(worker_result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            execution_error = f"invalid worker result: {error}"
    if (runtime_output / "run_summary.json").is_file():
        try:
            comparison = independent_comparison(runtime_output)
            base.write_exclusive(
                DEFAULT_OUTPUT / "independent-reference-comparison.json",
                comparison,
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            execution_error = f"independent comparison failed: {error}"

    failure = failure_record(
        timed_out=timed_out,
        caught_signal=caught_signal,
        exit_code=exit_code,
        product_result=product_result,
        comparison=comparison,
        execution_error=execution_error,
    )
    if timed_out:
        status = "TIMEOUT"
    elif failure is None:
        status = "PASS"
    else:
        status = "FAIL"
    result = {
        "schema": "ace2-v74-stage1-chat-attempt-result-v1",
        "status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "caught_signal": caught_signal,
        "elapsed_wall_seconds": elapsed,
        "generated_token_ids": (
            product_result.get("generated_token_ids", []) if product_result else []
        ),
        "decoded_text": product_result.get("decoded_text", "") if product_result else "",
        "readability": (
            product_result.get("readability") if product_result is not None else None
        ),
        "independent_comparison_status": (
            comparison.get("status") if comparison is not None else "NOT_REACHED"
        ),
        "failure": failure,
        "next_action": (
            "independent Reviewer acceptance"
            if status == "PASS"
            else "preserve attempt002 and diagnose only this terminal cause before any fresh attempt"
        ),
    }
    base.write_exclusive(DEFAULT_OUTPUT / "attempt-result.json", result)
    base.write_exclusive(
        DEFAULT_OUTPUT / "timing.json",
        {
            "supervisor_total_wall_seconds": elapsed,
            "backend_latency": (
                product_result.get("latency") if product_result is not None else None
            ),
        },
    )
    base.seal_attempt(DEFAULT_OUTPUT)
    if status == "PASS":
        print(f"Assistant: {result['decoded_text']}")
    print(
        f"ACE2_STAGE1_CHAT_ATTEMPT status={status} "
        f"tokens={len(result['generated_token_ids'])} "
        f"total_seconds={elapsed:.6f} output={DEFAULT_OUTPUT.relative_to(ROOT)}"
    )
    return 0 if status == "PASS" else (124 if status == "TIMEOUT" else 1)


def run_worker(args: argparse.Namespace) -> int:
    prompt = args.worker_prompt_file.read_text(encoding="utf-8")
    require(
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() == EXPECTED_PROMPT_SHA256,
        "worker prompt differs from the frozen official prompt",
    )
    result = product.run_chat(
        prompt,
        args.worker_runtime_output,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    base.write_exclusive(args.worker_result, result)
    return 0 if result["readability"]["accepted"] else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V74 official-tokenizer W4A8 RTL Stage-1 chat attempt002"
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--terminalize-attempt001-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-prompt-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-runtime-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        require(
            all((args.worker_prompt_file, args.worker_runtime_output, args.worker_result)),
            "worker paths are required",
        )
        return run_worker(args)
    require(
        not (args.preflight_only and args.terminalize_attempt001_only),
        "select at most one non-executing action",
    )
    if args.terminalize_attempt001_only:
        print(json.dumps(terminalize_attempt001(), sort_keys=True))
        return 0
    if args.preflight_only:
        print(json.dumps(readiness(), sort_keys=True))
        return 0
    return run_attempt(args.timeout_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_STAGE1_CHAT_ATTEMPT_SETUP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
