#!/usr/bin/env python3
"""Execute and seal one V73 arbitrary-text multi-token Stage-1 chat attempt."""

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
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation


DEFAULT_OUTPUT = ROOT / "reports/ace2-stage1-chat-attempt-0001"
DEFAULT_TIMEOUT_SECONDS = 21_600


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_exclusive(path: Path, value: Any) -> None:
    raw = value if isinstance(value, bytes) else canonical_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        raw = args.prompt_file.read_bytes()
        try:
            prompt = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("--prompt-file must contain valid UTF-8") from error
    elif args.prompt is not None:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        raw = sys.stdin.buffer.read()
        try:
            prompt = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("stdin must contain valid UTF-8") from error
    else:
        raise RuntimeError("provide --prompt, --prompt-file, or UTF-8 stdin")
    if not prompt.strip():
        raise RuntimeError("prompt must contain non-whitespace UTF-8 text")
    return prompt


def source_binding() -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        Path(product.__file__).resolve(),
        Path(backend.__file__).resolve(),
        Path(generation.__file__).resolve(),
    ]
    paths.extend(sorted((ROOT / "rtl").rglob("*.sv")))
    paths.extend(sorted((ROOT / "rtl").rglob("*.svh")))
    records = [file_record(path) for path in sorted(set(paths))]
    return {
        "algorithm": "sha256(path-nul-file-sha256-newline)-v1",
        "files": records,
        "sha256": sha256_bytes(
            b"".join(
                record["path"].encode("utf-8")
                + b"\0"
                + record["sha256"].encode("ascii")
                + b"\n"
                for record in records
            )
        ),
    }


def tool_record(argv: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def seal_attempt(output: Path) -> None:
    members = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    )
    sums = b"".join(
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n".encode("utf-8")
        for path in members
    )
    write_exclusive(output / "SHA256SUMS", sums)
    root = sha256_bytes(sums)
    write_exclusive(output / "TREE_ROOT.sha256", f"{root}  SHA256SUMS\n".encode("ascii"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for path in sorted(
        (path for path in output.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(
            stat.S_IRUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
    output.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def independent_comparison(runtime_output: Path) -> dict[str, Any]:
    summary = json.loads((runtime_output / "run_summary.json").read_text(encoding="utf-8"))
    positions = []
    total_mismatches = 0
    for execution in summary["token_executions"]:
        layers = []
        for layer in execution["layers"]:
            mismatches = {
                key: int(value)
                for key, value in layer["integer_boundary_mismatches"].items()
            }
            total_mismatches += sum(mismatches.values())
            layers.append(
                {
                    "layer_id": layer["layer_id"],
                    "cache_length_before": layer["cache_length_before"],
                    "cache_length_after": layer["cache_length_after"],
                    "integer_boundary_mismatches": mismatches,
                    "rtl_cache_append": layer["rtl_cache_append"],
                }
            )
        positions.append(
            {
                "absolute_position": execution["absolute_position"],
                "phase": execution["phase"],
                "input_token_id": execution["input_token_id"],
                "layers": layers,
            }
        )
    return {
        "schema": "ace2-v73-stage1-chat-independent-comparison-v1",
        "status": "PASS" if total_mismatches == 0 else "FAIL",
        "oracle_independence": {
            "reference_derived_before_rtl_execution": True,
            "rtl_outputs_used_only_as_comparison_observations": True,
            "host_reference_consumed_rtl_as_input": False,
        },
        "positions": positions,
        "head_steps": summary["head_steps"],
        "generated_token_ids": summary["generated_token_ids"],
        "total_integer_mismatches": total_mismatches,
    }


def run_worker(args: argparse.Namespace) -> int:
    prompt = args.worker_prompt_file.read_text(encoding="utf-8")
    result = product.run_chat(
        prompt,
        args.worker_runtime_output,
        max_new_tokens=args.max_new_tokens,
    )
    write_exclusive(args.worker_result, result)
    return 0 if result["readability"]["accepted"] else 5


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_attempt(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"attempt output already exists: {output}")
    prompt = read_prompt(args)
    generation.validate_max_new_tokens(args.max_new_tokens)
    position01 = product.verify_accepted_position01()
    model_snapshot = generation.resolve_snapshot()
    tokenization, _tokenizer = generation.tokenizer_record(
        prompt,
        backend.MAX_CONTEXT_TOKENS - args.max_new_tokens,
        args.max_new_tokens,
        model_snapshot,
    )

    output.mkdir(parents=True)
    prompt_path = output / "prompt.utf8"
    write_exclusive(prompt_path, prompt.encode("utf-8"))
    runtime_output = output / "runtime-output"
    worker_result = output / "worker-result.json"
    stdout_path = output / "stdout.raw.log"
    stderr_path = output / "stderr.raw.log"
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
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    manifest = {
        "schema": "ace2-v73-stage1-chat-attempt-manifest-v1",
        "manager": "V73",
        "status": "FROZEN_BEFORE_EXECUTION",
        "created_at_utc": utc_now(),
        "timeout_seconds": args.timeout_seconds,
        "pass_policy": (
            "PASS only after the requested RTL-selected tokens, generated-token feedback "
            "across every decode transition, 24-layer RTL integer agreement at every "
            "executed position, persistent RTL-published K/V, full-head agreement, and "
            "readable official tokenizer detokenization; otherwise FAIL or TIMEOUT"
        ),
        "accepted_position01": position01,
        "prompt": file_record(prompt_path),
        "tokenization": tokenization,
        "model": file_record(model_snapshot / "model.safetensors"),
        "model_config": file_record(model_snapshot / "config.json"),
        "adapter": file_record(backend.canonical.ADAPTER.resolve()),
        "source_and_rtl": source_binding(),
        "command": {
            "argv": command,
            "argv_sha256": sha256_bytes(canonical_bytes(command)),
            "cwd": str(ROOT),
        },
        "tools": {
            "python": tool_record([str(Path(sys.executable).resolve()), "--version"]),
            "iverilog": tool_record([str(Path("/usr/bin/iverilog")), "-V"]),
            "vvp": tool_record([str(Path("/usr/bin/vvp")), "-V"]),
        },
        "environment": {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    manifest["environment_sha256"] = sha256_bytes(
        canonical_bytes(manifest["environment"])
    )
    write_exclusive(output / "attempt-manifest.json", manifest)
    write_exclusive(output / "command.json", manifest["command"])

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
            exit_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            exit_code = process.returncode
    elapsed = time.monotonic() - started

    comparison: dict[str, Any] | None = None
    product_result: dict[str, Any] | None = None
    if worker_result.is_file():
        product_result = json.loads(worker_result.read_text(encoding="utf-8"))
    if (runtime_output / "run_summary.json").is_file():
        comparison = independent_comparison(runtime_output)
        write_exclusive(output / "independent-reference-comparison.json", comparison)
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
        "schema": "ace2-v73-stage1-chat-attempt-result-v1",
        "status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_wall_seconds": elapsed,
        "generated_token_ids": (
            product_result.get("generated_token_ids", []) if product_result else []
        ),
        "decoded_text": product_result.get("decoded_text", "") if product_result else "",
        "independent_comparison_status": (
            comparison.get("status") if comparison is not None else "NOT_REACHED"
        ),
        "next_action": (
            "independent Reviewer acceptance"
            if status == "PASS"
            else "inspect the preserved last completed position and repair the first concrete defect in a fresh attempt"
        ),
    }
    write_exclusive(output / "attempt-result.json", result)
    write_exclusive(
        output / "timing.json",
        {
            "supervisor_total_wall_seconds": elapsed,
            "backend_latency": (
                product_result.get("latency") if product_result is not None else None
            ),
        },
    )
    seal_attempt(output)
    if status == "PASS":
        print(f"Assistant: {result['decoded_text']}")
    print(
        f"ACE2_STAGE1_CHAT_ATTEMPT status={status} "
        f"tokens={len(result['generated_token_ids'])} "
        f"total_seconds={elapsed:.6f} output={output.relative_to(ROOT)}"
    )
    return 0 if status == "PASS" else (124 if status == "TIMEOUT" else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V73 official-tokenizer W4A8 RTL Stage-1 chat attempt"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-prompt-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-runtime-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if not all(
            (args.worker_prompt_file, args.worker_runtime_output, args.worker_result)
        ):
            raise RuntimeError("worker paths are required")
        return run_worker(args)
    if args.timeout_seconds < 1:
        raise RuntimeError("--timeout-seconds must be positive")
    return run_attempt(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_STAGE1_CHAT_ATTEMPT_SETUP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
