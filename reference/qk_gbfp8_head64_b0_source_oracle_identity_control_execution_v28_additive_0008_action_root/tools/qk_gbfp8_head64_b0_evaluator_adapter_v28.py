#!/home/argustest/miniconda3/bin/python3.13
"""Shell-free V21-derived evaluator adapter with digest-only diagnostics."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, NamedTuple

from qk_gbfp8_head64_b0_execution_common_v28 import compact_bytes, sha256_bytes


FAILURE_CLASSES = frozenset({"SPAWN", "ABI", "RETURN_CODE", "STDOUT", "STDERR", "DECODE", "RESULT_SCHEMA", "RESULT_RECORD", "PUBLICATION", "EXCEPTION"})


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
    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic['failure_class']}:{diagnostic['diagnostic_sha256']}")


def stream_identity(raw: bytes) -> dict[str, Any]:
    return {"byte_count": len(raw), "sha256": sha256_bytes(raw)}


def invocation_identity(spec: InvocationSpec) -> dict[str, Any]:
    return {
        "argv_sha256": sha256_bytes(compact_bytes(list(spec.argv))),
        "cwd_sha256": sha256_bytes(spec.cwd.encode("utf-8")),
        "environment_sha256": sha256_bytes(compact_bytes(dict(spec.environment))),
        "input_transport": "STDIN_BYTES", "output_transport": "STDOUT_BYTES",
        "stderr_transport": "STDERR_BYTES", "shell": False,
    }


def diagnostic(spec: InvocationSpec, input_bytes: bytes, *, outcome: str, failure_class: str | None,
               started: bool, return_code: int | None, stdout: bytes, stderr: bytes,
               error: BaseException | None) -> dict[str, Any]:
    record = {
        "adapter_kind": "qk_gbfp8_head64_b0_source_oracle_v28_evaluator_adapter",
        "exception_type": None if error is None else type(error).__name__,
        "failure_class": failure_class,
        "input": stream_identity(input_bytes),
        "invocation": invocation_identity(spec),
        "outcome": outcome,
        "process": {"return_code": return_code, "started": started},
        "schema_version": 1,
        "stderr": stream_identity(stderr),
        "stdout": stream_identity(stdout),
    }
    record["diagnostic_sha256"] = sha256_bytes(compact_bytes(record))
    return record


def fail(spec: InvocationSpec, input_bytes: bytes, failure_class: str, *, started: bool,
         return_code: int | None, stdout: bytes = b"", stderr: bytes = b"",
         error: BaseException | None = None) -> None:
    if failure_class not in FAILURE_CLASSES:
        raise AssertionError(failure_class)
    raise EvaluatorInvocationError(diagnostic(
        spec, input_bytes, outcome="FAILED", failure_class=failure_class, started=started,
        return_code=return_code, stdout=stdout, stderr=stderr, error=error,
    ))


def decode_canonical_json(raw: bytes) -> dict[str, Any]:
    import json
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            duplicate |= key in result
            result[key] = value
        return result

    value = json.loads(raw.decode("ascii", "strict"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if duplicate or type(value) is not dict or compact_bytes(value) != raw:
        raise ValueError("noncanonical evaluator stdout")
    return value


def invoke_evaluator(spec: InvocationSpec, input_bytes: bytes, schema_validate: Callable[[dict[str, Any]], None],
                     record_validate: Callable[[dict[str, Any], bytes], None], publish: Callable[[bytes], None],
                     *, runner: Callable[..., Any] = subprocess.run) -> AdapterResult:
    if type(input_bytes) is not bytes or not spec.argv or not Path(spec.cwd).is_absolute() or spec.max_capture_bytes <= 0:
        raise TypeError("adapter ABI")
    try:
        completed = runner(list(spec.argv), cwd=spec.cwd, env=dict(spec.environment), input=input_bytes,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, check=False)
    except OSError as error:
        fail(spec, input_bytes, "SPAWN", started=False, return_code=None, error=error)
    except Exception as error:
        fail(spec, input_bytes, "EXCEPTION", started=False, return_code=None, error=error)
    if not isinstance(completed, subprocess.CompletedProcess) or type(completed.returncode) is not int or type(completed.stdout) is not bytes or type(completed.stderr) is not bytes:
        fail(spec, input_bytes, "ABI", started=True, return_code=None, error=TypeError("CompletedProcess ABI"))
    stdout, stderr = completed.stdout, completed.stderr
    if len(stdout) > spec.max_capture_bytes:
        fail(spec, input_bytes, "STDOUT", started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=ValueError("stdout bound"))
    if len(stderr) > spec.max_capture_bytes:
        fail(spec, input_bytes, "STDERR", started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr, error=ValueError("stderr bound"))
    if completed.returncode != 0:
        fail(spec, input_bytes, "RETURN_CODE", started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr)
    if stderr:
        fail(spec, input_bytes, "STDERR", started=True, return_code=completed.returncode, stdout=stdout, stderr=stderr)
    if not stdout:
        fail(spec, input_bytes, "STDOUT", started=True, return_code=completed.returncode, error=ValueError("missing output"))
    try:
        record = decode_canonical_json(stdout)
    except Exception as error:
        fail(spec, input_bytes, "DECODE", started=True, return_code=completed.returncode, stdout=stdout, error=error)
    try:
        schema_validate(record)
    except Exception as error:
        fail(spec, input_bytes, "RESULT_SCHEMA", started=True, return_code=completed.returncode, stdout=stdout, error=error)
    try:
        record_validate(record, stdout)
    except Exception as error:
        fail(spec, input_bytes, "RESULT_RECORD", started=True, return_code=completed.returncode, stdout=stdout, error=error)
    try:
        publish(stdout)
    except Exception as error:
        fail(spec, input_bytes, "PUBLICATION", started=True, return_code=completed.returncode, stdout=stdout, error=error)
    return AdapterResult(diagnostic(spec, input_bytes, outcome="SUCCEEDED", failure_class=None, started=True,
                                    return_code=completed.returncode, stdout=stdout, stderr=b"", error=None), stdout, record)
