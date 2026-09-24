#!/usr/bin/env python3
"""Frozen shell-free transport for the future Base V13 execute-once action.

This helper is inert unless invoked.  It never opens the sealed tensor and
never imports the controller or evaluator.  Its only successful process
creation primitive is os.posix_spawn with explicit argv and environment
objects; it remains the launcher's immediate parent until waitpid completes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_action_root")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
PACKAGE = ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V13_SHELLFREE_PACKAGE.json"
ACCEPTANCE = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
TRANSPORT = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v13.py"
LAUNCHER = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v13.py"
FIRST_TERMINAL = ROOT / "live/authority/base/first-terminal.json"
ACTION_ID = "ace2:qk-gbfp8-base-v13:execute-once:81a8edc1:20260814T075838Z"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXACT_TRANSPORT_ARGV = [
    str(INTERPRETER),
    str(TRANSPORT),
    "--package",
    str(PACKAGE),
    "--acceptance",
    str(ACCEPTANCE),
    "--irreversible-action-id",
    ACTION_ID,
]
EXACT_LAUNCHER_ARGV = [
    str(INTERPRETER),
    str(LAUNCHER),
    "--package",
    str(PACKAGE),
    "--acceptance",
    str(ACCEPTANCE),
    "--irreversible-action-id",
    ACTION_ID,
]
SHELL = False
DIRECT_EXEC_API = "os.posix_spawn"
COMMAND_REPRESENTATION = "ARGV_VECTOR_ONLY"
FORBIDDEN_EXECUTABLES = ("/bin/bash", "/bin/sh")
FORBIDDEN_ARGV_TOKENS = ("-c",)


class TransportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransportError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    data = read_bytes(path)
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("ascii", "strict"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(TransportError(token)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TransportError("invalid package JSON") from error
    require(not duplicate and type(value) is dict, "duplicate package key or non-object")
    require(compact_bytes(value) == data, "noncanonical package bytes")
    return value, data


def verify_self_checksum(package: dict[str, Any]) -> None:
    observed = package.get("package_content_sha256")
    require(type(observed) is str and len(observed) == 64, "package checksum syntax")
    payload = dict(package)
    payload.pop("package_content_sha256")
    require(sha256_bytes(compact_bytes(payload)) == observed, "package checksum mismatch")


def invocation_record(argv: list[str]) -> dict[str, Any]:
    return {
        "argv": argv,
        "command_representation": COMMAND_REPRESENTATION,
        "cwd": str(ROOT),
        "environment": EXACT_ENVIRONMENT,
        "shell": SHELL,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    if path == ROOT:
        return
    if path.exists():
        require(path.is_dir() and not path.is_symlink(), f"unsafe live directory: {path}")
        return
    _ensure_directory(path.parent)
    os.mkdir(path, 0o700)
    _fsync_directory(path.parent)


def _durable_create(path: Path, data: bytes) -> None:
    _ensure_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            require(written > 0, "short terminal write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed = os.lstat(path)
    require(stat.S_ISREG(observed.st_mode) and stat.S_IMODE(observed.st_mode) == 0o400, "terminal publication mode")
    _fsync_directory(path.parent)


def _terminal_record(error: BaseException) -> dict[str, Any]:
    record = {
        "action_id": ACTION_ID,
        "action_retired": True,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_first_terminal",
        "authority_sha256": None,
        "consumed_ledger_sha256": None,
        "credential_consumed": False,
        "failure_detail_sha256": sha256_bytes(f"{type(error).__name__}:{error}".encode("utf-8")),
        "first_record_immutable": True,
        "invocation_count_performed": 0,
        "orphaned_after_consumption": False,
        "payload_open_count": 0,
        "reason_code": "TRANSPORT_ATTESTATION_FAILED",
        "result_file_sha256": None,
        "retry_replay_resume_repair_replacement_permitted": False,
        "status": "PREFLIGHT_FAILED_TERMINAL",
    }
    record["first_terminal_sha256"] = sha256_bytes(compact_bytes(record))
    return record


def _retire_transport_failure(error: BaseException) -> int:
    try:
        _durable_create(FIRST_TERMINAL, compact_bytes(_terminal_record(error)))
    except Exception:
        return 4
    return 3


def _transport_preflight() -> None:
    require(Path.cwd() == ROOT, "transport cwd mismatch")
    require([sys.executable, *sys.argv] == EXACT_TRANSPORT_ARGV, "transport argv mismatch")
    require(dict(os.environ) == EXACT_ENVIRONMENT, "transport environment mismatch")
    require(type(EXACT_TRANSPORT_ARGV) is list and type(EXACT_LAUNCHER_ARGV) is list, "argv vectors required")
    require(SHELL is False and DIRECT_EXEC_API == "os.posix_spawn", "direct-exec contract")
    require(not any(item in FORBIDDEN_EXECUTABLES or item in FORBIDDEN_ARGV_TOKENS for item in EXACT_TRANSPORT_ARGV + EXACT_LAUNCHER_ARGV), "shell token in frozen argv")
    package, _ = read_canonical_json(PACKAGE)
    verify_self_checksum(package)
    require(package["action_identity"]["future_action_id"] == ACTION_ID, "package action id")
    require(package["future_invocation"] == invocation_record(EXACT_TRANSPORT_ARGV) | {"invocation_sha256": sha256_bytes(compact_bytes(invocation_record(EXACT_TRANSPORT_ARGV)))}, "transport invocation binding")
    require(package["launcher_invocation"] == invocation_record(EXACT_LAUNCHER_ARGV) | {"invocation_sha256": sha256_bytes(compact_bytes(invocation_record(EXACT_LAUNCHER_ARGV)))}, "launcher invocation binding")
    local = {item["id"]: item for item in package["local_artifact_bindings"]}
    require(local["transport"]["path"] == str(TRANSPORT), "transport binding path")
    require(local["transport"]["sha256"] == sha256_bytes(read_bytes(TRANSPORT)), "transport binding bytes")
    require(local["launcher"]["path"] == str(LAUNCHER), "launcher binding path")
    require(local["launcher"]["sha256"] == sha256_bytes(read_bytes(LAUNCHER)), "launcher binding bytes")
    require(package["transport_contract"]["direct_exec_api"] == DIRECT_EXEC_API, "transport API binding")
    require(package["transport_contract"]["shell"] is False, "transport shell binding")
    require(package["transport_contract"]["environment_inheritance_permitted"] is False, "environment inheritance prohibited")
    require(not os.path.lexists(ROOT / "live"), "new action already has live state")


def main() -> int:
    try:
        _transport_preflight()
    except Exception as error:
        return _retire_transport_failure(error)
    try:
        child_pid = os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)
    except Exception as error:
        return _retire_transport_failure(error)
    waited_pid, wait_status = os.waitpid(child_pid, 0)
    if waited_pid != child_pid:
        return 5
    return os.waitstatus_to_exitcode(wait_status)


if __name__ == "__main__":
    raise SystemExit(main())
