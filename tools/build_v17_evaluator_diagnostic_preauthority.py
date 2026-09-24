#!/usr/bin/env python3
"""Build the additive V17 evaluator-diagnostic preauthority package.

This builder is static-only.  It never opens the official tensor bundle,
starts the production evaluator worker, creates authority, or writes into any
V15/V16 package or runtime namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
V15_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_action_root"
REJECTED_V16_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_action_root"
V16_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_shellfree_action_root"
V16_RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_e4c05735"
V16_ACTION_ID = "ace2:qk-gbfp8-base-v16:execute-once:e4c05735:20260814T093500Z"
FAILED_V17_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_action_root"
FAILED_V17_RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_9b1e6d42"
ACTION_ID = "ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31"
BUILD_ROOT = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001"
MANIFEST_NAME = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json"
MANIFEST = ACTION_ROOT / MANIFEST_NAME
ACCEPTANCE = BUILD_ROOT / "FRESH_L2_STATIC_ACCEPTANCE.json"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_EVALUATOR_ADAPTER_FIXTURE_REPORT.json"
CAUSE_REPORT = ACTION_ROOT / "evidence/V16_EVALUATOR_CALL_STATIC_DIAGNOSIS.json"
BUILD_REPORT = BUILD_ROOT / "build-report.json"

V16_RESULT_SCHEMA = V16_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"
V16_ACCEPTED_PACKAGE = V16_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
V16_RESULT_VALIDATION = V16_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_result_validation_v16.py"
V16_NOMINAL_FIXTURE = V16_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_marker_free_fixture_v16.py"
V16_RETURN_PATH = V16_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16.py"
V16_LAUNCHER = V16_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v16.py"
V16_TERMINAL = V16_RUNTIME_ROOT / "primary/authority/base/first-terminal.json"
OFFICIAL_EVALUATOR = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
OFFICIAL_PARSER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"

EXACT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    value = json.loads(raw.decode("ascii", "strict"), object_pairs_hook=pairs)
    require(type(value) is dict and not duplicate, f"non-object or duplicate key: {path}")
    require(compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def tree_inventory(root: Path) -> dict[str, dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"missing preserved root: {root}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink in preserved root: {path}")
        if stat.S_ISREG(info.st_mode):
            records[relative] = {
                "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                "sha256": sha256_file(path),
                "size": info.st_size,
            }
    return records


def write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing overwrite: {path}")
    path.write_bytes(raw)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, compact_bytes(value))


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            os.chmod(child, 0o700 if child.is_dir() else 0o600)
        except FileNotFoundError:
            pass
    os.chmod(path, 0o700)
    shutil.rmtree(path)


def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def adapter_source() -> bytes:
    return r'''#!/usr/bin/env python3
"""Shared V17 production/synthetic evaluator invocation adapter."""

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
        "adapter_kind": "qk_gbfp8_head64_granularity_sweep_evaluator_invocation_adapter_v17",
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
'''.encode("ascii")


def worker_source() -> bytes:
    return f'''#!/usr/bin/env python3
"""V17 evaluator worker with production and explicitly synthetic modes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ACTION_ID = {ACTION_ID!r}
PACKAGE = Path({str(MANIFEST)!r})
ACCEPTANCE = Path({str(ACCEPTANCE)!r})
OFFICIAL_EVALUATOR = Path({str(OFFICIAL_EVALUATOR)!r})
OFFICIAL_PARSER = Path({str(OFFICIAL_PARSER)!r})
EXACT_ENVIRONMENT = {EXACT_ENVIRONMENT!r}


class WorkerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\\n").encode("ascii")


def canonical_object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, "worker canonical input")
    return value


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {{name}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(scenario: str, raw: bytes) -> int:
    if scenario == "nominal":
        sys.stdout.buffer.write(raw)
        return 0
    if scenario == "nonzero":
        sys.stderr.buffer.write(b"synthetic evaluator nonzero\\n")
        return 7
    if scenario == "no-output":
        return 0
    if scenario == "malformed":
        sys.stdout.buffer.write(b"{{malformed\\n")
        return 0
    if scenario == "stderr":
        sys.stdout.buffer.write(raw)
        sys.stderr.buffer.write(b"synthetic evaluator stderr\\n")
        return 0
    raise WorkerError("unknown synthetic scenario")


def production(args: argparse.Namespace, raw: bytes) -> int:
    require(Path.cwd() == PACKAGE.parent, "production cwd")
    require(dict(os.environ) == EXACT_ENVIRONMENT, "production environment")
    require(args.package == str(PACKAGE), "production package")
    require(args.acceptance == str(ACCEPTANCE), "production acceptance")
    require(args.irreversible_action_id == ACTION_ID, "production action")
    package = canonical_object(PACKAGE.read_bytes())
    acceptance = canonical_object(ACCEPTANCE.read_bytes())
    require(package["action_identity"]["future_action_id"] == ACTION_ID, "bound package action")
    require(acceptance.get("accepted") is True, "Fresh-L2 acceptance")
    require(acceptance.get("static_acceptance_grants_execution_authority") is False, "acceptance authority boundary")
    envelope = canonical_object(raw)
    require(set(envelope) == {{"bindings", "context", "payload_base64"}}, "production envelope keys")
    payload = base64.b64decode(envelope["payload_base64"], validate=True)
    evaluator = load_module("v17_bound_official_evaluator", OFFICIAL_EVALUATOR)
    parser = load_module("v17_bound_official_parser", OFFICIAL_PARSER)
    result = evaluator.evaluate_bundle_once(lambda: payload, envelope["bindings"], parser, envelope["context"])
    require(type(result) is bytes, "official evaluator return ABI")
    sys.stdout.buffer.write(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("production", "synthetic"))
    parser.add_argument("--scenario", choices=("nominal", "nonzero", "no-output", "malformed", "stderr"))
    parser.add_argument("--package")
    parser.add_argument("--acceptance")
    parser.add_argument("--irreversible-action-id")
    args = parser.parse_args()
    raw = sys.stdin.buffer.read()
    try:
        if args.mode == "synthetic":
            require(args.scenario is not None, "synthetic scenario")
            return synthetic(args.scenario, raw)
        require(args.scenario is None, "production scenario prohibited")
        return production(args, raw)
    except Exception as error:
        detail = f"{{type(error).__name__}}:{{hashlib.sha256(str(error).encode('utf-8', 'replace')).hexdigest()}}\\n"
        sys.stderr.write(detail)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
'''.encode("ascii")


def bridge_source() -> bytes:
    return f'''#!/usr/bin/env python3
"""Frozen V17 production binding for the shared evaluator adapter."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Callable

from qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17 import (
    AdapterResult,
    EvaluatorInvocationError,
    InvocationSpec,
    compact_bytes,
    invoke_evaluator,
)


ACTION_ID = {ACTION_ID!r}
ROOT = {str(ACTION_ROOT)!r}
WORKER = ROOT + "/tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py"
PACKAGE = ROOT + "/{MANIFEST_NAME}"
ACCEPTANCE = {str(ACCEPTANCE)!r}
EXACT_ENVIRONMENT = {tuple(sorted(EXACT_ENVIRONMENT.items()))!r}
PRODUCTION_SPEC = InvocationSpec(
    argv=(
        {str(INTERPRETER)!r},
        WORKER,
        "--mode",
        "production",
        "--package",
        PACKAGE,
        "--acceptance",
        ACCEPTANCE,
        "--irreversible-action-id",
        ACTION_ID,
    ),
    cwd=ROOT,
    environment=EXACT_ENVIRONMENT,
    max_capture_bytes=16 * 1024 * 1024,
)


def pack_production_input(payload: bytes, bindings: dict[str, Any], context: dict[str, Any]) -> bytes:
    if type(payload) is not bytes or type(bindings) is not dict or type(context) is not dict:
        raise TypeError("production adapter envelope ABI")
    return compact_bytes({{
        "bindings": bindings,
        "context": context,
        "payload_base64": base64.b64encode(payload).decode("ascii"),
    }})


def invoke_production_evaluator(
    payload: bytes,
    bindings: dict[str, Any],
    context: dict[str, Any],
    schema_validate: Callable[[dict[str, Any]], None],
    record_validate: Callable[[dict[str, Any], bytes], None],
    publish_result: Callable[[bytes], None],
) -> AdapterResult:
    envelope = pack_production_input(payload, bindings, context)
    return invoke_evaluator(PRODUCTION_SPEC, envelope, schema_validate, record_validate, publish_result)


def terminal_diagnostic_fields(error: EvaluatorInvocationError) -> dict[str, Any]:
    diagnostic = error.diagnostic
    failure_stage = "RESULT_PUBLICATION" if diagnostic["failure_class"] == "PUBLICATION" else "EVALUATOR_CALL"
    return {{
        "evaluator_diagnostic": diagnostic,
        "failure_detail_sha256": hashlib.sha256(compact_bytes(diagnostic)).hexdigest(),
        "failure_stage": failure_stage,
    }}


def main() -> int:
    raise RuntimeError("PREAUTHORITY_ONLY: V17 has no execution authority")


if __name__ == "__main__":
    raise SystemExit(main())
'''.encode("ascii")


def terminal_schema() -> dict[str, Any]:
    sha = {"pattern": "^[0-9a-f]{64}$", "type": "string"}
    stream = {
        "additionalProperties": False,
        "properties": {
            "byte_count": {"minimum": 0, "type": "integer"},
            "preview_ascii": {"maxLength": 160, "type": "string"},
            "preview_truncated": {"type": "boolean"},
            "sha256": sha,
        },
        "required": ["byte_count", "preview_ascii", "preview_truncated", "sha256"],
        "type": "object",
    }
    diagnostic = {
        "additionalProperties": False,
        "properties": {
            "adapter_kind": {"const": "qk_gbfp8_head64_granularity_sweep_evaluator_invocation_adapter_v17"},
            "diagnostic_sha256": sha,
            "exception": {
                "oneOf": [
                    {"type": "null"},
                    {
                        "additionalProperties": False,
                        "properties": {
                            "message_preview_ascii": {"maxLength": 160, "type": "string"},
                            "message_preview_truncated": {"type": "boolean"},
                            "message_sha256": sha,
                            "type": {"maxLength": 96, "pattern": "^[A-Za-z_][A-Za-z0-9_.]*$", "type": "string"},
                        },
                        "required": ["message_preview_ascii", "message_preview_truncated", "message_sha256", "type"],
                        "type": "object",
                    },
                ]
            },
            "failure_class": {
                "oneOf": [
                    {"type": "null"},
                    {"enum": ["SPAWN", "ABI", "RETURN_CODE", "STDOUT", "STDERR", "DECODE", "RESULT_SCHEMA", "RESULT_RECORD", "EXCEPTION", "PUBLICATION"], "type": "string"},
                ]
            },
            "input": {
                "additionalProperties": False,
                "properties": {"byte_count": {"minimum": 0, "type": "integer"}, "sha256": sha},
                "required": ["byte_count", "sha256"],
                "type": "object",
            },
            "invocation": {
                "additionalProperties": False,
                "properties": {
                    "argv_sha256": sha,
                    "cwd_sha256": sha,
                    "environment_sha256": sha,
                    "input_transport": {"const": "STDIN_BYTES"},
                    "output_transport": {"const": "STDOUT_BYTES"},
                    "shell": {"const": False},
                    "stderr_transport": {"const": "STDERR_BYTES"},
                },
                "required": ["argv_sha256", "cwd_sha256", "environment_sha256", "input_transport", "output_transport", "shell", "stderr_transport"],
                "type": "object",
            },
            "outcome": {"enum": ["FAILED", "SUCCEEDED"]},
            "process": {
                "additionalProperties": False,
                "properties": {"return_code": {"oneOf": [{"type": "integer"}, {"type": "null"}]}, "started": {"type": "boolean"}},
                "required": ["return_code", "started"],
                "type": "object",
            },
            "schema_version": {"const": 1},
            "stderr": stream,
            "stdout": stream,
        },
        "required": ["adapter_kind", "diagnostic_sha256", "exception", "failure_class", "input", "invocation", "outcome", "process", "schema_version", "stderr", "stdout"],
        "type": "object",
    }
    return {
        "$defs": {"diagnostic": diagnostic, "sha256": sha},
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_FIRST_TERMINAL_SCHEMA",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "CONSUMED_ORPHAN"}}, "required": ["status"]},
                "then": {
                    "properties": {
                        "evaluator_diagnostic": {"$ref": "#/$defs/diagnostic"},
                        "failure_stage": {"enum": ["EVALUATOR_CALL", "RESULT_PUBLICATION"]},
                        "invocation_count_performed": {"const": 1},
                        "reason_code": {"const": "POST_CONSUMPTION_AMBIGUITY"},
                    }
                },
            },
            {
                "if": {"properties": {"status": {"enum": ["SUCCEEDED_TERMINAL", "FAILED_TERMINAL"]}}, "required": ["status"]},
                "then": {
                    "properties": {
                        "failure_stage": {"type": "null"},
                        "invocation_count_performed": {"const": 1},
                    }
                },
            },
        ],
        "properties": {
            "action_id": {"const": ACTION_ID},
            "action_retired": {"const": True},
            "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_first_terminal"},
            "evaluator_diagnostic": {"oneOf": [{"$ref": "#/$defs/diagnostic"}, {"type": "null"}]},
            "failure_detail_sha256": {"$ref": "#/$defs/sha256"},
            "failure_stage": {"oneOf": [{"enum": ["EVALUATOR_CALL", "RESULT_PUBLICATION", "PREFLIGHT", "TRANSPORT"], "type": "string"}, {"type": "null"}]},
            "first_record_immutable": {"const": True},
            "invocation_count_performed": {"maximum": 1, "minimum": 0, "type": "integer"},
            "official_target_process_starts": {"maximum": 1, "minimum": 0, "type": "integer"},
            "payload_open_count": {"maximum": 1, "minimum": 0, "type": "integer"},
            "reason_code": {"enum": ["HARD_THRESHOLD_FAILED", "HARD_THRESHOLDS_PASSED", "POST_CONSUMPTION_AMBIGUITY", "READ_ONLY_PREFLIGHT_FAILED", "TRANSPORT_ATTESTATION_FAILED"]},
            "result_file_sha256": {"oneOf": [{"$ref": "#/$defs/sha256"}, {"type": "null"}]},
            "retry_replay_resume_repair_replacement_permitted": {"const": False},
            "status": {"enum": ["CONSUMED_ORPHAN", "FAILED_TERMINAL", "PREFLIGHT_FAILED_TERMINAL", "SUCCEEDED_TERMINAL"]},
        },
        "required": ["action_id", "action_retired", "artifact_kind", "evaluator_diagnostic", "failure_detail_sha256", "failure_stage", "first_record_immutable", "invocation_count_performed", "official_target_process_starts", "payload_open_count", "reason_code", "result_file_sha256", "retry_replay_resume_repair_replacement_permitted", "status"],
        "type": "object",
    }


def fixture_source() -> bytes:
    return f'''#!/usr/bin/env python3
"""Synthetic-only regression fixture for the shared V17 adapter."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import jsonschema

from qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17 import (
    EvaluatorInvocationError,
    InvocationSpec,
    compact_bytes,
    invoke_evaluator,
)
from qk_gbfp8_head64_granularity_sweep_nominal_records_v17 import RESULT_VALIDATION, make_result


ACTION_ID = {ACTION_ID!r}
ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py"
TERMINAL_SCHEMA = ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_FIRST_TERMINAL_SCHEMA.json"
INTERPRETER = {str(INTERPRETER)!r}
EXACT_ENVIRONMENT = {tuple(sorted(EXACT_ENVIRONMENT.items()))!r}


def spec(scenario: str) -> InvocationSpec:
    return InvocationSpec(
        argv=(INTERPRETER, str(WORKER), "--mode", "synthetic", "--scenario", scenario),
        cwd=str(ROOT),
        environment=EXACT_ENVIRONMENT,
        max_capture_bytes=16 * 1024 * 1024,
    )


def terminal(diagnostic: dict[str, Any], failure_stage: str) -> dict[str, Any]:
    return {{
        "action_id": ACTION_ID,
        "action_retired": True,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_first_terminal",
        "evaluator_diagnostic": diagnostic,
        "failure_detail_sha256": hashlib.sha256(compact_bytes(diagnostic)).hexdigest(),
        "failure_stage": failure_stage,
        "first_record_immutable": True,
        "invocation_count_performed": 1,
        "official_target_process_starts": 1 if diagnostic["process"]["started"] else 0,
        "payload_open_count": 1,
        "reason_code": "POST_CONSUMPTION_AMBIGUITY",
        "result_file_sha256": None,
        "retry_replay_resume_repair_replacement_permitted": False,
        "status": "CONSUMED_ORPHAN",
    }}


def observe(
    name: str,
    expected: str,
    operation: Callable[[], Any],
    terminal_validator: Any,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] | None = None
    try:
        operation()
    except EvaluatorInvocationError as error:
        diagnostic = error.diagnostic
    passed = diagnostic is not None and diagnostic["failure_class"] == expected
    terminal_valid = False
    if diagnostic is not None:
        stage = "RESULT_PUBLICATION" if expected == "PUBLICATION" else "EVALUATOR_CALL"
        terminal_validator.validate(terminal(diagnostic, stage))
        terminal_valid = True
    return {{
        "diagnostic": diagnostic,
        "expected_failure_class": expected,
        "name": name,
        "passed": passed and terminal_valid,
        "terminal_schema_valid": terminal_valid,
    }}


def run_fixture() -> dict[str, Any]:
    terminal_schema = json.loads(TERMINAL_SCHEMA.read_text(encoding="ascii"))
    jsonschema.Draft202012Validator.check_schema(terminal_schema)
    terminal_validator = jsonschema.Draft202012Validator(terminal_schema)
    nominal_results = (make_result(None), make_result("G4"))
    nominal_valid = 0
    success_diagnostics: list[dict[str, Any]] = []
    for result in nominal_results:
        raw = compact_bytes(result)
        RESULT_VALIDATION.validate_result_schema(result)
        RESULT_VALIDATION.validate_result_record(result, raw)
        published: list[bytes] = []
        observed = invoke_evaluator(
            spec("nominal"),
            raw,
            RESULT_VALIDATION.validate_result_schema,
            RESULT_VALIDATION.validate_result_record,
            published.append,
        )
        if observed.raw == raw and published == [raw] and observed.diagnostic["outcome"] == "SUCCEEDED":
            nominal_valid += 1
            success_diagnostics.append(observed.diagnostic)

    all_fail_raw = compact_bytes(nominal_results[0])
    cases = [
        observe("evaluator_nonzero_exit", "RETURN_CODE", lambda: invoke_evaluator(spec("nonzero"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_no_output", "STDOUT", lambda: invoke_evaluator(spec("no-output"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_malformed_output", "DECODE", lambda: invoke_evaluator(spec("malformed"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_stderr", "STDERR", lambda: invoke_evaluator(spec("stderr"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_spawn_failure", "SPAWN", lambda: invoke_evaluator(InvocationSpec(argv=("/definitely/absent/v17-evaluator",), cwd=str(ROOT), environment=EXACT_ENVIRONMENT, max_capture_bytes=1024), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_abi_failure", "ABI", lambda: invoke_evaluator(spec("nominal"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None, runner=lambda *args, **kwargs: object()), terminal_validator),
        observe("evaluator_exception", "EXCEPTION", lambda: invoke_evaluator(spec("nominal"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None, runner=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic runner exception"))), terminal_validator),
        observe("result_publication_failure", "PUBLICATION", lambda: invoke_evaluator(spec("nominal"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: (_ for _ in ()).throw(OSError("synthetic publication failure"))), terminal_validator),
    ]
    required = {{"evaluator_nonzero_exit", "evaluator_no_output", "evaluator_malformed_output", "evaluator_exception", "result_publication_failure"}}
    names = {{case["name"] for case in cases if case["passed"]}}
    status = "PASS_V17_EVALUATOR_ADAPTER_SYNTHETIC_FIXTURE" if nominal_valid == 2 and required <= names and all(case["passed"] for case in cases) else "FAIL_V17_EVALUATOR_ADAPTER_SYNTHETIC_FIXTURE"
    return {{
        "adapter_case_count": len(cases),
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_adapter_fixture_report",
        "cases": cases,
        "exact_result_record_validator": "qk_gbfp8_head64_granularity_sweep_result_validation_v17.validate_result_record_exact",
        "exact_result_schema_validator": "qk_gbfp8_head64_granularity_sweep_result_validation_v17.validate_result_schema_exact",
        "nominal_frozen_schema_record_count": nominal_valid,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "production_mode_exercised": False,
        "required_fail_closed_cases": sorted(required),
        "status": status,
        "synthetic_bytes_only": True,
    }}


def main() -> int:
    report = run_fixture()
    print(json.dumps(report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''.encode("ascii")


def diagnosis_report() -> dict[str, Any]:
    terminal, _ = canonical_object(V16_TERMINAL)
    require(terminal["action_id"] == V16_ACTION_ID, "V16 terminal identity")
    require(terminal["status"] == "CONSUMED_ORPHAN", "V16 terminal status")
    require(terminal["failure_stage"] == "EVALUATOR_CALL", "V16 terminal failure stage")
    require(terminal["invocation_count_performed"] == 1 and terminal["payload_open_count"] == 1, "V16 terminal counters")
    launcher_source = V16_LAUNCHER.read_text(encoding="utf-8")
    evaluator_source = OFFICIAL_EVALUATOR.read_text(encoding="utf-8")
    require("return evaluate_bundle_once(open_bundle, tensor_bindings, parser, result_context)" in launcher_source, "V16 launcher call shape")
    require("def evaluate_bundle_once(\n    open_bundle: Any,\n    bindings: dict[str, dict[str, Any]],\n    parser_module: Any,\n    context: dict[str, Any],\n) -> bytes:" in evaluator_source, "official evaluator ABI")
    return {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_v16_evaluator_call_static_diagnosis",
        "confirmed_adapter_cause_class": "EVALUATOR_CALL_OBSERVABILITY_COLLAPSE",
        "evidence_bounded_underlying_cause_class": "UNKNOWN_IN_PROCESS_EVALUATOR_EXCEPTION",
        "excluded_by_static_evidence": [
            "TOP_LEVEL_PYTHON_CALLABLE_ARITY_MISMATCH",
            "EVALUATOR_SUBPROCESS_SPAWN_FAILURE",
            "EVALUATOR_SUBPROCESS_RETURN_CODE_FAILURE",
            "EVALUATOR_SUBPROCESS_STDOUT_FAILURE",
            "EVALUATOR_SUBPROCESS_STDERR_FAILURE",
        ],
        "exclusion_basis": {
            "callable_abi": "V16 launcher supplies open_bundle, bindings, parser_module, and context in the exact frozen four-positional-argument order.",
            "subprocess_classes": "V16 invokes the evaluator as an in-process Python callable, so evaluator spawn/return-code/stdout/stderr evidence does not exist.",
        },
        "failure_envelope": [
            "payload opener",
            "official payload digest check",
            "tensor binding checks",
            "producer parser",
            "accepted-reader parser",
            "cross-parser comparison",
            "candidate evaluation",
            "result construction",
        ],
        "retained_v16_evidence": {
            "failure_detail_sha256": terminal["failure_detail_sha256"],
            "failure_stage": terminal["failure_stage"],
            "invocation_count_performed": terminal["invocation_count_performed"],
            "payload_open_count": terminal["payload_open_count"],
            "status": terminal["status"],
        },
        "source_bindings": {
            "evaluator_sha256": sha256_file(OFFICIAL_EVALUATOR),
            "launcher_sha256": sha256_file(V16_LAUNCHER),
            "terminal_sha256": sha256_file(V16_TERMINAL),
        },
        "v17_required_repair": "Create an explicit shared invocation adapter that records bounded sanitized spawn/ABI/return-code/stdout/stderr/decode/exception/publication diagnostics and is exercised only with synthetic bytes before authority.",
    }


def transformed_nominal_fixture() -> bytes:
    source = V16_NOMINAL_FIXTURE.read_text(encoding="utf-8")
    source = source.replace(V16_ACTION_ID, ACTION_ID)
    source = source.replace("qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16", "qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v17")
    source = source.replace("qk_gbfp8_head64_granularity_sweep_result_validation_v16", "qk_gbfp8_head64_granularity_sweep_result_validation_v17")
    source = source.replace("execution_v16_marker_free_fixture_report", "execution_v17_nominal_records_fixture_report")
    return source.encode("ascii")


def build() -> dict[str, Any]:
    require(ACTION_ID != V16_ACTION_ID, "fresh action identity")
    require(not ACTION_ROOT.exists(), f"V17 action root already exists: {ACTION_ROOT}")
    require(not RUNTIME_ROOT.exists(), f"V17 runtime root already exists: {RUNTIME_ROOT}")
    require(not BUILD_ROOT.exists(), f"V17 build root already exists: {BUILD_ROOT}")
    require(not ACCEPTANCE.exists(), "V17 acceptance unexpectedly exists")
    require(OFFICIAL_PAYLOAD.is_file(), "official payload path absent")
    official_payload_lstat = os.lstat(OFFICIAL_PAYLOAD)
    require(stat.S_ISREG(official_payload_lstat.st_mode), "official payload lstat")

    preservation = {
        "failed_v17_candidate_root": {"root": str(FAILED_V17_ROOT), "files": tree_inventory(FAILED_V17_ROOT)},
        "failed_v17_candidate_runtime": {"root": str(FAILED_V17_RUNTIME_ROOT), "files": tree_inventory(FAILED_V17_RUNTIME_ROOT)},
        "retired_v15_root": {"root": str(V15_ROOT), "files": tree_inventory(V15_ROOT)},
        "rejected_v16_root": {"root": str(REJECTED_V16_ROOT), "files": tree_inventory(REJECTED_V16_ROOT)},
        "sealed_v16_root": {"root": str(V16_ROOT), "files": tree_inventory(V16_ROOT)},
        "sealed_v16_runtime": {"root": str(V16_RUNTIME_ROOT), "files": tree_inventory(V16_RUNTIME_ROOT)},
    }

    stage_parent = PROJECT_ROOT / "build"
    stage = Path(tempfile.mkdtemp(prefix=".v17-evaldiag-stage-", dir=stage_parent))
    package_stage = stage / ROOT_ID
    runtime_stage = stage / RUNTIME_ROOT.name
    build_stage = stage / "build"
    try:
        write_bytes(package_stage / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json", V16_ACCEPTED_PACKAGE.read_bytes())
        write_bytes(package_stage / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json", V16_RESULT_SCHEMA.read_bytes())
        write_bytes(package_stage / "tools/qk_gbfp8_head64_granularity_sweep_result_validation_v17.py", V16_RESULT_VALIDATION.read_bytes())
        return_path = V16_RETURN_PATH.read_text(encoding="utf-8").replace("V16", "V17")
        write_bytes(package_stage / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v17.py", return_path.encode("ascii"))
        write_bytes(package_stage / "tools/qk_gbfp8_head64_granularity_sweep_nominal_records_v17.py", transformed_nominal_fixture())
        write_bytes(package_stage / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py", adapter_source())
        write_bytes(package_stage / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py", worker_source())
        write_bytes(package_stage / "tools/qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17.py", bridge_source())
        terminal_schema_path = package_stage / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_FIRST_TERMINAL_SCHEMA.json"
        write_json(terminal_schema_path, terminal_schema())
        write_bytes(package_stage / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_fixture_v17.py", fixture_source())
        write_json(package_stage / "evidence/V16_EVALUATOR_CALL_STATIC_DIAGNOSIS.json", diagnosis_report())

        for relative in (
            "primary",
            "primary/authority",
            "primary/authority/base",
            "primary/result",
            "primary/result/base",
            "fallback",
        ):
            (runtime_stage / relative).mkdir(parents=True, exist_ok=True)
        build_stage.mkdir(parents=True, exist_ok=True)

        ACTION_ROOT.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME_ROOT.parent.mkdir(parents=True, exist_ok=True)
        BUILD_ROOT.parent.mkdir(parents=True, exist_ok=True)
        os.replace(package_stage, ACTION_ROOT)
        os.replace(runtime_stage, RUNTIME_ROOT)
        os.replace(build_stage, BUILD_ROOT)
        package_stage = ACTION_ROOT
        runtime_stage = RUNTIME_ROOT
        build_stage = BUILD_ROOT

        fixture_path = package_stage / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_fixture_v17.py"
        fixture_run = subprocess.run(
            [str(INTERPRETER), str(fixture_path)],
            cwd=package_stage,
            env=EXACT_ENVIRONMENT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(fixture_run.returncode == 0, f"V17 synthetic fixture failed: {fixture_run.stderr.decode('utf-8', 'replace')}")
        fixture_report = json.loads(fixture_run.stdout.decode("ascii", "strict"))
        require(fixture_report.get("status") == "PASS_V17_EVALUATOR_ADAPTER_SYNTHETIC_FIXTURE", "fixture status")
        require(fixture_report.get("official_payload_open_count") == 0 and fixture_report.get("official_target_process_starts") == 0, "fixture execution boundary")
        write_json(package_stage / "evidence/SYNTHETIC_EVALUATOR_ADAPTER_FIXTURE_REPORT.json", fixture_report)

        for relative in (
            "primary",
            "primary/authority",
            "primary/authority/base",
            "primary/result",
            "primary/result/base",
            "fallback",
        ):
            (runtime_stage / relative).mkdir(parents=True, exist_ok=True)

        generated_files: dict[str, dict[str, Any]] = {}
        for path in sorted(package_stage.rglob("*")):
            if path.is_file():
                relative = path.relative_to(package_stage).as_posix()
                generated_files[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}

        production_argv = [
            str(INTERPRETER),
            str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py"),
            "--mode",
            "production",
            "--package",
            str(MANIFEST),
            "--acceptance",
            str(ACCEPTANCE),
            "--irreversible-action-id",
            ACTION_ID,
        ]
        manifest = sealed(
            {
                "action_identity": {
                    "future_action_id": ACTION_ID,
                    "prior_action_id": V16_ACTION_ID,
                    "replacement_is_distinct": True,
                    "retry_or_resume": False,
                    "reuse_permitted": False,
                },
                "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_package",
                "claim_boundary": {
                    "acceptance_materialized": False,
                    "authority_materialized": False,
                    "credential_materialized": False,
                    "execution_authorized": False,
                    "official_payload_open_count": 0,
                    "official_target_process_starts": 0,
                    "runtime_namespace_file_count": 0,
                    "v17_executed": False,
                    "v16_replayed": False,
                },
                "diagnosis": {
                    "confirmed_adapter_cause_class": "EVALUATOR_CALL_OBSERVABILITY_COLLAPSE",
                    "report_path": str(CAUSE_REPORT),
                    "report_sha256": generated_files["evidence/V16_EVALUATOR_CALL_STATIC_DIAGNOSIS.json"]["sha256"],
                    "underlying_evaluator_cause_class": "UNKNOWN_IN_PROCESS_EVALUATOR_EXCEPTION",
                },
                "frozen_result_validation": {
                    "accepted_package_sha256": generated_files["accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"]["sha256"],
                    "record_validator": "qk_gbfp8_head64_granularity_sweep_result_validation_v17.validate_result_record_exact",
                    "result_schema_sha256": generated_files["accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"]["sha256"],
                    "schema_validator": "qk_gbfp8_head64_granularity_sweep_result_validation_v17.validate_result_schema_exact",
                    "validator_module_sha256": generated_files["tools/qk_gbfp8_head64_granularity_sweep_result_validation_v17.py"]["sha256"],
                },
                "generated_files": generated_files,
                "preattempt_synthetic_fixture": {
                    "fixture_path": str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_fixture_v17.py"),
                    "official_payload_open_count": 0,
                    "official_target_process_starts": 0,
                    "production_mode_exercised": False,
                    "report_path": str(FIXTURE_REPORT),
                    "report_sha256": generated_files["evidence/SYNTHETIC_EVALUATOR_ADAPTER_FIXTURE_REPORT.json"]["sha256"],
                    "synthetic_bytes_only": True,
                },
                "preservation": preservation,
                "production_evaluator_adapter": {
                    "adapter_callable": "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.invoke_evaluator",
                    "adapter_path": str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py"),
                    "argv": production_argv,
                    "cwd": str(ACTION_ROOT),
                    "environment": EXACT_ENVIRONMENT,
                    "input_transport": "STDIN_BYTES",
                    "output_transport": "STDOUT_BYTES",
                    "return_code_handling": "EXACT_INTEGER_ZERO_REQUIRED",
                    "shell": False,
                    "stderr_handling": "EMPTY_ON_SUCCESS_REQUIRED",
                    "worker_path": str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py"),
                },
                "regression_contract": {
                    "required_fail_closed_cases": [
                        "evaluator_nonzero_exit",
                        "evaluator_no_output",
                        "evaluator_malformed_output",
                        "evaluator_exception",
                        "result_publication_failure",
                    ],
                    "structured_failure_classes": [
                        "SPAWN",
                        "ABI",
                        "RETURN_CODE",
                        "STDOUT",
                        "STDERR",
                        "DECODE",
                        "RESULT_SCHEMA",
                        "RESULT_RECORD",
                        "EXCEPTION",
                        "PUBLICATION",
                    ],
                },
                "required_disposition": "V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY",
                "root_id": ROOT_ID,
                "runtime_namespace": {
                    "file_count": 0,
                    "precreated_directories": [
                        str(RUNTIME_ROOT),
                        str(RUNTIME_ROOT / "primary"),
                        str(RUNTIME_ROOT / "primary/authority"),
                        str(RUNTIME_ROOT / "primary/authority/base"),
                        str(RUNTIME_ROOT / "primary/result"),
                        str(RUNTIME_ROOT / "primary/result/base"),
                        str(RUNTIME_ROOT / "fallback"),
                    ],
                    "runtime_root": str(RUNTIME_ROOT),
                },
                "terminal_diagnostics": {
                    "bounded_preview_chars": 160,
                    "schema_path": str(ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_FIRST_TERMINAL_SCHEMA.json"),
                    "schema_sha256": generated_files["reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_FIRST_TERMINAL_SCHEMA.json"]["sha256"],
                    "terminal_factory": "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17.terminal_diagnostic_fields",
                },
            },
            "package_content_sha256",
        )
        write_json(package_stage / MANIFEST_NAME, manifest)

        review_request = {
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_fresh_l2_static_review_request",
            "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
            "manifest_path": str(MANIFEST),
            "manifest_sha256": sha256_file(package_stage / MANIFEST_NAME),
            "required_disposition": manifest["required_disposition"],
            "reviewer_must_rerun": str(PROJECT_ROOT / "tools/v17_evaluator_diagnostic_verify_independent.py"),
        }
        write_json(build_stage / "fresh-l2-review-request.json", review_request)

        for path in sorted(ACTION_ROOT.rglob("*"), reverse=True):
            if path.is_file():
                os.chmod(path, 0o444)
            elif path.is_dir():
                os.chmod(path, 0o555)
        os.chmod(ACTION_ROOT, 0o555)
        for path in sorted(RUNTIME_ROOT.rglob("*")):
            if path.is_dir():
                os.chmod(path, 0o700)
        os.chmod(RUNTIME_ROOT, 0o700)

        report = sealed(
            {
                "action_id": ACTION_ID,
                "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_build_report",
                "fixture_report_sha256": sha256_file(FIXTURE_REPORT),
                "manifest_sha256": sha256_file(MANIFEST),
                "official_payload_open_count": 0,
                "official_target_process_starts": 0,
                "runtime_namespace_file_count": 0,
                "status": "PASS_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_BUILD",
            },
            "report_sha256",
        )
        write_json(BUILD_REPORT, report)
        return report
    finally:
        remove_tree(stage)


def main() -> int:
    report = build()
    print(json.dumps(report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
