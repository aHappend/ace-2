#!/usr/bin/env python3
"""Shared V21 production/synthetic evaluator invocation adapter."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, NamedTuple


MAX_PREVIEW_CHARS = 160
FAILURE_CLASSES = frozenset({
    "SPAWN", "ABI", "RETURN_CODE", "STDOUT", "STDERR", "DECODE",
    "RESULT_SCHEMA", "RESULT_RECORD", "EXCEPTION", "PUBLICATION",
})


class InvocationSpec(NamedTuple):
    argv: tuple[str, ...]
    cwd: str
    environment: tuple[tuple[str, str], ...]
    max_capture_bytes: int


class AdapterResult(NamedTuple):
    diagnostic: dict[str, Any]
    raw: bytes
    record: dict[str, Any]


class EvaluatorInvocationError(RuntimeError):
    def __init__(self, diagnostic: dict[str, Any]):
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic['failure_class']}:{diagnostic['diagnostic_sha256']}")


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _preview(raw: bytes) -> tuple[str, bool]:
    pieces: list[str] = []
    used = 0
    truncated = False
    for value in raw:
        piece = chr(value) if 0x20 <= value <= 0x7e else f"\\x{value:02x}"
        if used + len(piece) > MAX_PREVIEW_CHARS:
            truncated = True
            break
        pieces.append(piece)
        used += len(piece)
    if len(raw) > sum(1 if len(piece) == 1 else 1 for piece in pieces):
        truncated = True
    return "".join(pieces), truncated


def _stream_record(raw: bytes) -> dict[str, Any]:
    preview, truncated = _preview(raw)
    return {
        "byte_count": len(raw),
        "preview_ascii": preview,
        "preview_truncated": truncated,
        "sha256": sha256_bytes(raw),
    }


def _exception_record(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    raw = str(error).encode("utf-8", "replace")
    preview, truncated = _preview(raw)
    return {
        "message_preview_ascii": preview,
        "message_preview_truncated": truncated,
        "message_sha256": sha256_bytes(raw),
        "type": type(error).__name__,
    }


def _invocation_record(spec: InvocationSpec) -> dict[str, Any]:
    argv_bytes = compact_bytes(list(spec.argv))
    environment_bytes = compact_bytes(dict(spec.environment))
    return {
        "argv_sha256": sha256_bytes(argv_bytes),
        "cwd_sha256": sha256_bytes(spec.cwd.encode("utf-8")),
        "environment_sha256": sha256_bytes(environment_bytes),
        "input_transport": "STDIN_BYTES",
        "output_transport": "STDOUT_BYTES",
        "stderr_transport": "STDERR_BYTES",
        "shell": False,
    }


def _diagnostic(
    spec: InvocationSpec,
    input_bytes: bytes,
    *,
    outcome: str,
    failure_class: str | None,
    process_started: bool,
    return_code: int | None,
    stdout: bytes,
    stderr: bytes,
    error: BaseException | None,
) -> dict[str, Any]:
    record = {
        "adapter_kind": "qk_gbfp8_head64_granularity_sweep_evaluator_invocation_adapter_v21",
        "exception": _exception_record(error),
        "failure_class": failure_class,
        "input": {"byte_count": len(input_bytes), "sha256": sha256_bytes(input_bytes)},
        "invocation": _invocation_record(spec),
        "outcome": outcome,
        "process": {"return_code": return_code, "started": process_started},
        "schema_version": 1,
        "stderr": _stream_record(stderr),
        "stdout": _stream_record(stdout),
    }
    record["diagnostic_sha256"] = sha256_bytes(compact_bytes(record))
    return record


def _fail(
    spec: InvocationSpec,
    input_bytes: bytes,
    failure_class: str,
    *,
    process_started: bool,
    return_code: int | None,
    stdout: bytes = b"",
    stderr: bytes = b"",
    error: BaseException | None = None,
) -> None:
    if failure_class not in FAILURE_CLASSES:
        raise AssertionError(f"unknown failure class: {failure_class}")
    raise EvaluatorInvocationError(_diagnostic(
        spec,
        input_bytes,
        outcome="FAILED",
        failure_class=failure_class,
        process_started=process_started,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        error=error,
    ))


def decode_canonical_json(raw: bytes) -> dict[str, Any]:
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    value = json.loads(
        raw.decode("ascii", "strict"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if duplicate or type(value) is not dict or compact_bytes(value) != raw:
        raise ValueError("noncanonical evaluator stdout JSON object")
    return value


def invoke_evaluator(
    spec: InvocationSpec,
    input_bytes: bytes,
    schema_validate: Callable[[dict[str, Any]], None],
    record_validate: Callable[[dict[str, Any], bytes], None],
    publish_result: Callable[[bytes], None],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> AdapterResult:
    if type(input_bytes) is not bytes:
        raise TypeError("adapter input must be bytes")
    if not spec.argv or any(type(item) is not str or not item for item in spec.argv):
        raise ValueError("adapter argv")
    if not Path(spec.cwd).is_absolute() or spec.max_capture_bytes <= 0:
        raise ValueError("adapter cwd or capture bound")
    try:
        completed = runner(
            list(spec.argv),
            cwd=spec.cwd,
            env=dict(spec.environment),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
    except OSError as error:
        _fail(spec, input_bytes, "SPAWN", process_started=False, return_code=None, error=error)
    except Exception as error:
        _fail(spec, input_bytes, "EXCEPTION", process_started=False, return_code=None, error=error)
    if (
        not isinstance(completed, subprocess.CompletedProcess)
        or type(completed.returncode) is not int
        or type(completed.stdout) is not bytes
        or type(completed.stderr) is not bytes
    ):
        _fail(spec, input_bytes, "ABI", process_started=True, return_code=None, error=TypeError("invalid CompletedProcess ABI"))
    stdout = completed.stdout
    stderr = completed.stderr
    if len(stdout) > spec.max_capture_bytes:
        _fail(spec, input_bytes, "STDOUT", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=ValueError("stdout capture limit"))
    if len(stderr) > spec.max_capture_bytes:
        _fail(spec, input_bytes, "STDERR", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=ValueError("stderr capture limit"))
    if completed.returncode != 0:
        _fail(spec, input_bytes, "RETURN_CODE", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr)
    if stderr:
        _fail(spec, input_bytes, "STDERR", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr)
    if not stdout:
        _fail(spec, input_bytes, "STDOUT", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=ValueError("no evaluator stdout"))
    try:
        record = decode_canonical_json(stdout)
    except Exception as error:
        _fail(spec, input_bytes, "DECODE", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=error)
    try:
        schema_validate(record)
    except Exception as error:
        _fail(spec, input_bytes, "RESULT_SCHEMA", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=error)
    try:
        record_validate(record, stdout)
    except Exception as error:
        _fail(spec, input_bytes, "RESULT_RECORD", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=error)
    try:
        publish_result(stdout)
    except Exception as error:
        _fail(spec, input_bytes, "PUBLICATION", process_started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=error)
    diagnostic = _diagnostic(
        spec,
        input_bytes,
        outcome="SUCCEEDED",
        failure_class=None,
        process_started=True,
        return_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        error=None,
    )
    return AdapterResult(diagnostic=diagnostic, raw=stdout, record=record)
