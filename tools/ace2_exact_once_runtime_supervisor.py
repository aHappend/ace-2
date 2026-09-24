#!/usr/bin/env python3
"""Hash-bound, durable exactly-once supervisor for externally authorized commands.

This tool does not create execution authority.  It consumes a separately supplied
authority record that byte/hash-binds an immutable execution package.  The public
schemas and source-tree digest algorithm are intentionally small and strict so a
future package can be reviewed without depending on project-specific runtime code.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import secrets
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn


PACKAGE_SCHEMA = "ace2-exact-once-execution-package-v1"
AUTHORITY_SCHEMA = "ace2-exact-once-external-authority-v1"
INTENT_SCHEMA = "ace2-exact-once-pre-start-intent-v1"
STARTED_SCHEMA = "ace2-exact-once-process-started-v1"
TERMINAL_SCHEMA = "ace2-exact-once-terminal-record-v1"
SOURCE_TREE_ALGORITHM = "sha256-canonical-file-manifest-v1"

PACKAGE_KEYS = {"schema", "package_id", "bindings"}
BINDING_KEYS = {
    "bound_files",
    "source_tree",
    "cwd",
    "argv",
    "argv_sha256",
    "environment",
    "required_absent_outputs",
}
OPTIONAL_BINDING_KEYS = {"phase_binding"}
PHASE_BINDING_KEYS = {
    "certificate",
    "binding_core_sha256",
    "execution_envelope",
    "execution_envelope_sha256",
    "phase_transition_regression",
}
FILE_BINDING_KEYS = {"path", "byte_count", "sha256"}
TREE_BINDING_KEYS = {
    "path",
    "algorithm",
    "file_count",
    "byte_count",
    "sha256",
}
AUTHORITY_KEYS = {
    "schema",
    "authority_id",
    "decision",
    "package",
    "bindings",
    "runtime",
}
AUTHORITY_PACKAGE_KEYS = {"path", "byte_count", "sha256", "package_id"}
RUNTIME_KEYS = {"state_dir", "stdout_path", "stderr_path", "terminal_record_path"}
SHA256_HEX_LENGTH = 64


class SupervisorError(RuntimeError):
    """Fail-closed supervisor rejection before a child is started."""

    category = "preflight_rejected"


class DuplicateExecution(SupervisorError):
    """The durable one-start reservation already exists."""

    category = "duplicate_or_restart_rejected"


@dataclass(frozen=True)
class FileMeasurement:
    path: Path
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class SourceTreeMeasurement:
    path: Path
    file_count: int
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class ValidatedRun:
    package_path: Path
    package_measurement: FileMeasurement
    package: dict[str, Any]
    authority_path: Path
    authority_measurement: FileMeasurement
    authority: dict[str, Any]
    state_dir: Path
    stdout_path: Path
    stderr_path: Path
    terminal_record_path: Path


def _fail(message: str) -> NoReturn:
    raise SupervisorError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Serialize only standards-compliant JSON; NaN/Infinity are rejected."""

    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
        encoded = json.dumps(value, **options) + "\n"
    else:
        options["separators"] = (",", ":")
        encoded = json.dumps(value, **options)
    return encoded.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_value_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant rejected: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key rejected: {key}")
        value[key] = child
    return value


def _load_json_measurement(path: Path, label: str) -> tuple[dict[str, Any], FileMeasurement]:
    measurement, content = _read_regular_file(path, label, capture_content=True)
    _require(content is not None, f"internal error reading {label}")
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SupervisorError(f"cannot parse strict JSON {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, measurement


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} keys must be exactly {sorted(expected)}")


def _strict_string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    _require("\x00" not in value, f"{label} contains NUL")
    return value


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _sha256_string(value: Any, label: str) -> str:
    text = _strict_string(value, label)
    _require(
        len(text) == SHA256_HEX_LENGTH and all(character in "0123456789abcdef" for character in text),
        f"{label} must be lowercase SHA-256 hex",
    )
    return text


def _absolute_path(value: Any, label: str) -> Path:
    text = _strict_string(value, label)
    _require(os.path.isabs(text), f"{label} must be absolute")
    _require(os.path.normpath(text) == text, f"{label} must be lexically normalized")
    return Path(text)


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _read_regular_file(
    path: Path,
    label: str,
    *,
    capture_content: bool,
) -> tuple[FileMeasurement, bytes | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SupervisorError(f"cannot open bound {label} {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"bound {label} is not a regular file: {path}")
        chunks: list[bytes] | None = [] if capture_content else None
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            if chunks is not None:
                chunks.append(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"bound {label} changed while hashing: {path}",
        )
        _require(byte_count == before.st_size, f"bound {label} short read: {path}")
        content = b"".join(chunks) if chunks is not None else None
        return FileMeasurement(path, byte_count, digest.hexdigest()), content
    finally:
        os.close(descriptor)


def measure_file(path: Path, label: str = "file") -> FileMeasurement:
    measurement, _ = _read_regular_file(path, label, capture_content=False)
    return measurement


def measure_source_tree(path: Path) -> SourceTreeMeasurement:
    """Hash a symlink-free tree as canonical (relative path, bytes, SHA-256) rows."""

    try:
        root_stat = os.lstat(path)
    except OSError as exc:
        raise SupervisorError(f"cannot stat source tree {path}: {exc}") from exc
    _require(stat.S_ISDIR(root_stat.st_mode), f"source tree is not a real directory: {path}")
    rows: list[dict[str, Any]] = []
    for directory, names, filenames in os.walk(path, topdown=True, followlinks=False):
        names.sort()
        filenames.sort()
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            child_stat = os.lstat(child)
            _require(stat.S_ISDIR(child_stat.st_mode), f"source tree contains non-directory entry: {child}")
        for name in filenames:
            child = directory_path / name
            child_stat = os.lstat(child)
            _require(stat.S_ISREG(child_stat.st_mode), f"source tree contains non-regular file: {child}")
            measurement = measure_file(child, "source-tree file")
            rows.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "byte_count": measurement.byte_count,
                    "sha256": measurement.sha256,
                }
            )
    rows.sort(key=lambda row: row["path"])
    return SourceTreeMeasurement(
        path=path,
        file_count=len(rows),
        byte_count=sum(row["byte_count"] for row in rows),
        sha256=canonical_value_sha256(rows),
    )


def source_tree_binding(path: Path) -> dict[str, Any]:
    measurement = measure_source_tree(path)
    return {
        "path": str(path),
        "algorithm": SOURCE_TREE_ALGORITHM,
        "file_count": measurement.file_count,
        "byte_count": measurement.byte_count,
        "sha256": measurement.sha256,
    }


def file_binding(path: Path) -> dict[str, Any]:
    measurement = measure_file(path)
    return {
        "path": str(path),
        "byte_count": measurement.byte_count,
        "sha256": measurement.sha256,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        offset += os.write(descriptor, content[offset:])


def _atomic_create_bytes(path: Path, content: bytes) -> None:
    """Atomically publish a complete new file without replacing an existing path."""

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        _write_all(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_create_bytes(path, canonical_json_bytes(value, pretty=True))


def _atomic_copy_file(source: Path, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb", closefd=False) as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1 << 20)
            output_stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
            _fsync_directory(destination.parent)
        except FileNotFoundError:
            pass


def _validate_file_binding(binding: Any, label: str) -> Path:
    _require(isinstance(binding, dict), f"{label} binding must be an object")
    _exact_keys(binding, FILE_BINDING_KEYS, f"{label} binding")
    path = _absolute_path(binding["path"], f"{label}.path")
    expected_bytes = _strict_int(binding["byte_count"], f"{label}.byte_count")
    expected_sha256 = _sha256_string(binding["sha256"], f"{label}.sha256")
    measurement = measure_file(path, label)
    _require(measurement.byte_count == expected_bytes, f"{label} byte count mismatch: {path}")
    _require(measurement.sha256 == expected_sha256, f"{label} SHA-256 mismatch: {path}")
    return path


def _validate_tree_binding(binding: Any) -> Path:
    _require(isinstance(binding, dict), "source_tree binding must be an object")
    _exact_keys(binding, TREE_BINDING_KEYS, "source_tree binding")
    path = _absolute_path(binding["path"], "source_tree.path")
    _require(
        binding["algorithm"] == SOURCE_TREE_ALGORITHM,
        "unsupported source_tree digest algorithm",
    )
    expected_files = _strict_int(binding["file_count"], "source_tree.file_count")
    expected_bytes = _strict_int(binding["byte_count"], "source_tree.byte_count")
    expected_sha256 = _sha256_string(binding["sha256"], "source_tree.sha256")
    measurement = measure_source_tree(path)
    _require(measurement.file_count == expected_files, "source_tree file count mismatch")
    _require(measurement.byte_count == expected_bytes, "source_tree byte count mismatch")
    _require(measurement.sha256 == expected_sha256, "source_tree SHA-256 mismatch")
    return path


def _validate_phase_binding(binding: Any, bound_paths: set[Path], label: str) -> None:
    _require(isinstance(binding, dict), f"{label} must be an object")
    _exact_keys(binding, PHASE_BINDING_KEYS, label)
    certificate = _validate_file_binding(binding["certificate"], f"{label}.certificate")
    regression = _validate_file_binding(
        binding["phase_transition_regression"],
        f"{label}.phase_transition_regression",
    )
    _require(
        certificate in bound_paths and regression in bound_paths,
        f"{label} artifacts must also be present in bound_files",
    )
    _sha256_string(binding["binding_core_sha256"], f"{label}.binding_core_sha256")
    envelope = binding["execution_envelope"]
    _require(isinstance(envelope, dict) and envelope, f"{label}.execution_envelope must be an object")
    envelope_sha256 = _sha256_string(
        binding["execution_envelope_sha256"],
        f"{label}.execution_envelope_sha256",
    )
    _require(
        canonical_value_sha256(envelope) == envelope_sha256,
        f"{label} execution envelope digest mismatch",
    )


def _validate_bindings(bindings: Any, label: str) -> tuple[Path, list[Path]]:
    _require(isinstance(bindings, dict), f"{label} must be an object")
    keys = set(bindings)
    _require(
        keys in (BINDING_KEYS, BINDING_KEYS | OPTIONAL_BINDING_KEYS),
        f"{label} keys mismatch: expected {sorted(BINDING_KEYS)} with optional "
        f"{sorted(OPTIONAL_BINDING_KEYS)}, got {sorted(keys)}",
    )

    bound_files = bindings["bound_files"]
    _require(isinstance(bound_files, list) and bound_files, f"{label}.bound_files must be non-empty")
    bound_paths = [
        _validate_file_binding(binding, f"{label}.bound_files[{index}]")
        for index, binding in enumerate(bound_files)
    ]
    _require(len(bound_paths) == len(set(bound_paths)), f"{label}.bound_files contains duplicates")
    if "phase_binding" in bindings:
        _validate_phase_binding(bindings["phase_binding"], set(bound_paths), f"{label}.phase_binding")

    source_root = _validate_tree_binding(bindings["source_tree"])
    cwd = _absolute_path(bindings["cwd"], f"{label}.cwd")
    try:
        cwd_stat = os.lstat(cwd)
    except OSError as exc:
        raise SupervisorError(f"cannot stat bound cwd {cwd}: {exc}") from exc
    _require(stat.S_ISDIR(cwd_stat.st_mode), f"bound cwd is not a real directory: {cwd}")

    argv = bindings["argv"]
    _require(isinstance(argv, list) and argv, f"{label}.argv must be a non-empty list")
    for index, argument in enumerate(argv):
        _strict_string(argument, f"{label}.argv[{index}]")
    _require(os.path.isabs(argv[0]), f"{label}.argv[0] executable must be absolute")
    _require(Path(argv[0]) in set(bound_paths), f"{label}.argv[0] executable is not a bound file")
    argv_sha256 = _sha256_string(bindings["argv_sha256"], f"{label}.argv_sha256")
    _require(canonical_value_sha256(argv) == argv_sha256, f"{label} argv digest mismatch")

    environment = bindings["environment"]
    _require(isinstance(environment, dict), f"{label}.environment must be an object")
    for name, value in environment.items():
        _strict_string(name, f"{label}.environment name")
        _require("=" not in name, f"{label}.environment name contains '='")
        _require(isinstance(value, str) and "\x00" not in value, f"{label}.environment[{name}] invalid")

    outputs = bindings["required_absent_outputs"]
    _require(
        isinstance(outputs, list) and outputs,
        f"{label}.required_absent_outputs must be non-empty",
    )
    output_paths = [
        _absolute_path(path, f"{label}.required_absent_outputs[{index}]")
        for index, path in enumerate(outputs)
    ]
    _require(len(output_paths) == len(set(output_paths)), f"{label} output paths contain duplicates")
    return source_root, output_paths


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_runtime(runtime: Any, outputs: list[Path], source_root: Path) -> tuple[Path, Path, Path, Path]:
    _require(isinstance(runtime, dict), "authority.runtime must be an object")
    _exact_keys(runtime, RUNTIME_KEYS, "authority.runtime")
    state_dir = _absolute_path(runtime["state_dir"], "authority.runtime.state_dir")
    stdout_path = _absolute_path(runtime["stdout_path"], "authority.runtime.stdout_path")
    stderr_path = _absolute_path(runtime["stderr_path"], "authority.runtime.stderr_path")
    terminal_path = _absolute_path(
        runtime["terminal_record_path"], "authority.runtime.terminal_record_path"
    )
    runtime_outputs = {stdout_path, stderr_path, terminal_path}
    _require(runtime_outputs.issubset(set(outputs)), "runtime output paths must be required absent")
    _require(not _is_within(state_dir, source_root), "state_dir must be outside source_tree")
    for output in outputs:
        _require(not _is_within(output, source_root), f"output must be outside source_tree: {output}")
        _require(not _is_within(output, state_dir), f"required output must be outside state_dir: {output}")
        parent = output.parent
        try:
            parent_stat = os.lstat(parent)
        except OSError as exc:
            raise SupervisorError(f"cannot stat output parent {parent}: {exc}") from exc
        _require(stat.S_ISDIR(parent_stat.st_mode), f"output parent is not a real directory: {parent}")
    state_parent = state_dir.parent
    try:
        parent_stat = os.lstat(state_parent)
    except OSError as exc:
        raise SupervisorError(f"cannot stat state parent {state_parent}: {exc}") from exc
    _require(stat.S_ISDIR(parent_stat.st_mode), f"state parent is not a real directory: {state_parent}")
    return state_dir, stdout_path, stderr_path, terminal_path


def _check_state_available(state_dir: Path) -> None:
    if not _path_exists(state_dir):
        return
    state_stat = os.lstat(state_dir)
    _require(stat.S_ISDIR(state_stat.st_mode), f"state_dir is not a real directory: {state_dir}")
    intent_path = state_dir / "pre_start_intent.json"
    if _path_exists(intent_path):
        raise DuplicateExecution(f"durable pre-start intent already exists: {intent_path}")
    entries = list(state_dir.iterdir())
    _require(not entries, f"state_dir is not pristine: {state_dir}")


def _check_outputs_absent(outputs: list[Path]) -> None:
    existing = [str(path) for path in outputs if _path_exists(path)]
    _require(not existing, f"required output already exists: {existing}")


def prepare_runtime_bindings(
    *,
    source_root: Path,
    state_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    terminal_record_path: Path,
    additional_outputs: tuple[Path, ...] = (),
) -> tuple[list[str], dict[str, str]]:
    """Build and non-consumingly validate external runtime path bindings."""

    source_root = _absolute_path(str(source_root), "source_root")
    outputs = [
        _absolute_path(str(path), f"additional_outputs[{index}]")
        for index, path in enumerate(additional_outputs)
    ]
    outputs.extend(
        [
            _absolute_path(str(stdout_path), "stdout_path"),
            _absolute_path(str(stderr_path), "stderr_path"),
            _absolute_path(str(terminal_record_path), "terminal_record_path"),
        ]
    )
    _require(len(outputs) == len(set(outputs)), "runtime output paths contain duplicates")
    runtime = {
        "state_dir": str(_absolute_path(str(state_dir), "state_dir")),
        "stdout_path": str(outputs[-3]),
        "stderr_path": str(outputs[-2]),
        "terminal_record_path": str(outputs[-1]),
    }
    validated_state, _stdout, _stderr, _terminal = _validate_runtime(
        runtime, outputs, source_root
    )
    _check_state_available(validated_state)
    _check_outputs_absent(outputs)
    return [str(path) for path in outputs], runtime


def validate_run(package_path: Path, authority_path: Path) -> ValidatedRun:
    package_path = package_path.resolve(strict=True)
    authority_path = authority_path.resolve(strict=True)
    package, package_measurement = _load_json_measurement(package_path, "execution package")
    authority, authority_measurement = _load_json_measurement(authority_path, "external authority")

    _exact_keys(package, PACKAGE_KEYS, "execution package")
    _require(package["schema"] == PACKAGE_SCHEMA, "unsupported execution package schema")
    package_id = _strict_string(package["package_id"], "execution package.package_id")
    source_root, outputs = _validate_bindings(package["bindings"], "execution package.bindings")

    _exact_keys(authority, AUTHORITY_KEYS, "external authority")
    _require(authority["schema"] == AUTHORITY_SCHEMA, "unsupported external authority schema")
    _strict_string(authority["authority_id"], "external authority.authority_id")
    _require(authority["decision"] == "execute_once", "external authority decision is not execute_once")

    authority_package = authority["package"]
    _require(isinstance(authority_package, dict), "external authority.package must be an object")
    _exact_keys(authority_package, AUTHORITY_PACKAGE_KEYS, "external authority.package")
    _require(
        _absolute_path(authority_package["path"], "external authority.package.path") == package_path,
        "external authority package path mismatch",
    )
    _require(authority_package["package_id"] == package_id, "external authority package_id mismatch")
    _require(
        _strict_int(authority_package["byte_count"], "external authority.package.byte_count")
        == package_measurement.byte_count,
        "external authority package byte count mismatch",
    )
    _require(
        _sha256_string(authority_package["sha256"], "external authority.package.sha256")
        == package_measurement.sha256,
        "external authority package SHA-256 mismatch",
    )

    _require(isinstance(authority["bindings"], dict), "external authority.bindings must be an object")
    _require(authority["bindings"] == package["bindings"], "external authority bindings mismatch package")

    state_dir, stdout_path, stderr_path, terminal_path = _validate_runtime(
        authority["runtime"], outputs, source_root
    )
    _check_state_available(state_dir)
    _check_outputs_absent(outputs)
    return ValidatedRun(
        package_path=package_path,
        package_measurement=package_measurement,
        package=package,
        authority_path=authority_path,
        authority_measurement=authority_measurement,
        authority=authority,
        state_dir=state_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        terminal_record_path=terminal_path,
    )


def _provenance(validated: ValidatedRun) -> dict[str, Any]:
    return {
        "package": {
            "path": str(validated.package_path),
            "byte_count": validated.package_measurement.byte_count,
            "sha256": validated.package_measurement.sha256,
            "package_id": validated.package["package_id"],
        },
        "authority": {
            "path": str(validated.authority_path),
            "byte_count": validated.authority_measurement.byte_count,
            "sha256": validated.authority_measurement.sha256,
            "authority_id": validated.authority["authority_id"],
        },
    }


def _reserve_pre_start_intent(validated: ValidatedRun) -> Path:
    try:
        validated.state_dir.mkdir(mode=0o700)
        _fsync_directory(validated.state_dir.parent)
    except FileExistsError:
        pass
    _check_state_available(validated.state_dir)
    intent_path = validated.state_dir / "pre_start_intent.json"
    intent = {
        "schema": INTENT_SCHEMA,
        "created_at_utc": _utc_now(),
        "process_start_count": 0,
        "provenance": _provenance(validated),
        "bindings": validated.package["bindings"],
        "runtime": validated.authority["runtime"],
    }
    try:
        _atomic_create_json(intent_path, intent)
    except FileExistsError as exc:
        raise DuplicateExecution(f"durable pre-start intent already exists: {intent_path}") from exc
    return intent_path


def _revalidate_after_reservation(validated: ValidatedRun) -> None:
    package, package_measurement = _load_json_measurement(validated.package_path, "execution package")
    authority, authority_measurement = _load_json_measurement(validated.authority_path, "external authority")
    _require(package == validated.package, "execution package changed after preflight")
    _require(authority == validated.authority, "external authority changed after preflight")
    _require(package_measurement == validated.package_measurement, "execution package bytes changed")
    _require(authority_measurement == validated.authority_measurement, "external authority bytes changed")
    _validate_bindings(package["bindings"], "execution package.bindings")
    _require(authority["bindings"] == package["bindings"], "external authority bindings changed")
    outputs = [Path(path) for path in package["bindings"]["required_absent_outputs"]]
    _check_outputs_absent(outputs)


def _error_record(stage: str, exc: BaseException) -> dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def _write_started_metadata(path: Path, record: dict[str, Any]) -> None:
    _atomic_create_json(path, record)


def _wait4_without_abandoning(process: subprocess.Popen[Any]) -> tuple[int, Any, list[dict[str, str]]]:
    interruptions: list[dict[str, str]] = []
    while True:
        try:
            waited_pid, status, usage = os.wait4(process.pid, 0)
            _require(waited_pid == process.pid, "wait4 returned an unexpected PID")
            process.returncode = os.waitstatus_to_exitcode(status)
            return status, usage, interruptions
        except InterruptedError:
            continue
        except (KeyboardInterrupt, SystemExit) as exc:
            interruptions.append(_error_record("synchronous_wait_interrupted", exc))
            continue
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            raise


def _decode_wait_status(status: int) -> tuple[str, int | None, int | None, int]:
    returncode = os.waitstatus_to_exitcode(status)
    if os.WIFEXITED(status):
        exit_code = os.WEXITSTATUS(status)
        return ("success" if exit_code == 0 else "nonzero_exit", exit_code, None, returncode)
    if os.WIFSIGNALED(status):
        terminating_signal = os.WTERMSIG(status)
        return "signal", None, terminating_signal, returncode
    return "unknown_wait_status", None, None, returncode


def _peak_rss_kib(usage: Any) -> int:
    raw = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return math.ceil(raw / 1024)
    return raw


def _capture_record(spool_path: Path, requested_path: Path) -> dict[str, Any]:
    measurement = measure_file(spool_path, "captured stream")
    return {
        "spool_path": str(spool_path),
        "requested_path": str(requested_path),
        "byte_count": measurement.byte_count,
        "sha256": measurement.sha256,
        "published": False,
    }


def _publish_capture(record: dict[str, Any]) -> None:
    _atomic_copy_file(Path(record["spool_path"]), Path(record["requested_path"]))
    published = measure_file(Path(record["requested_path"]), "published capture")
    _require(published.byte_count == record["byte_count"], "published capture byte count mismatch")
    _require(published.sha256 == record["sha256"], "published capture SHA-256 mismatch")
    record["published"] = True


def _terminal_paths(validated: ValidatedRun) -> dict[str, str]:
    return {
        "authority_bound": str(validated.terminal_record_path),
        "durable_state": str(validated.state_dir / "terminal_record.json"),
    }


def _seal_terminal(validated: ValidatedRun, record: dict[str, Any]) -> dict[str, Any]:
    state_path = Path(record["terminal_record_paths"]["durable_state"])
    primary_error: BaseException | None = None
    try:
        _atomic_create_json(validated.terminal_record_path, record)
    except BaseException as exc:  # child is already reaped; preserve a durable fallback record
        primary_error = exc
        record["post_start_errors"].append(_error_record("terminal_record_publication", exc))
        record["outcome"] = "supervisor_post_start_error"
    try:
        _atomic_create_json(state_path, record)
    except BaseException:
        if primary_error is not None:
            raise
    return record


def _start_failure_record(
    validated: ValidatedRun,
    *,
    started_at_utc: str,
    wall_time_seconds: float,
    exc: BaseException,
) -> dict[str, Any]:
    record = {
        "schema": TERMINAL_SCHEMA,
        "sealed_at_utc": _utc_now(),
        "outcome": "process_start_failed",
        "process_started": False,
        "process_start_count": 0,
        "provenance": _provenance(validated),
        "process": {
            "pid": None,
            "cwd": validated.package["bindings"]["cwd"],
            "argv": validated.package["bindings"]["argv"],
            "argv_sha256": validated.package["bindings"]["argv_sha256"],
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_now(),
            "child_outcome": "not_started",
            "exit_code": None,
            "signal": None,
            "returncode": None,
            "wall_time_seconds": wall_time_seconds,
            "peak_rss_kib": None,
        },
        "captures": None,
        "post_start_errors": [_error_record("popen", exc)],
        "terminal_record_paths": _terminal_paths(validated),
    }
    return _seal_terminal(validated, record)


def run_once(package_path: Path, authority_path: Path) -> dict[str, Any]:
    """Validate, reserve, start at most one child, synchronously reap, and seal."""

    validated = validate_run(package_path, authority_path)
    _reserve_pre_start_intent(validated)
    try:
        _revalidate_after_reservation(validated)
    except BaseException as exc:
        rejection = {
            "schema": "ace2-exact-once-pre-start-rejection-v1",
            "created_at_utc": _utc_now(),
            "process_start_count": 0,
            "error": _error_record("post_reservation_revalidation", exc),
        }
        _atomic_create_json(validated.state_dir / "pre_start_rejection.json", rejection)
        if isinstance(exc, SupervisorError):
            raise
        raise SupervisorError(f"post-reservation revalidation failed: {exc}") from exc

    stdout_spool = validated.state_dir / "stdout.spool"
    stderr_spool = validated.state_dir / "stderr.spool"
    stdout_stream = stdout_spool.open("xb", buffering=0)
    stderr_stream = stderr_spool.open("xb", buffering=0)
    started_at_utc = _utc_now()
    monotonic_start = time.monotonic_ns()
    try:
        try:
            process = subprocess.Popen(
                validated.package["bindings"]["argv"],
                cwd=validated.package["bindings"]["cwd"],
                env=validated.package["bindings"]["environment"],
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                shell=False,
                close_fds=True,
            )
        except BaseException as exc:
            wall_time = (time.monotonic_ns() - monotonic_start) / 1_000_000_000
            return _start_failure_record(
                validated,
                started_at_utc=started_at_utc,
                wall_time_seconds=wall_time,
                exc=exc,
            )

        pid = process.pid  # capture immediately after the one and only Popen
        post_start_errors: list[dict[str, str]] = []
        started_record = {
            "schema": STARTED_SCHEMA,
            "recorded_at_utc": _utc_now(),
            "process_start_count": 1,
            "pid": pid,
            "provenance": _provenance(validated),
        }
        try:
            _write_started_metadata(validated.state_dir / "process_started.json", started_record)
        except BaseException as exc:
            post_start_errors.append(_error_record("process_started_metadata", exc))

        status, usage, wait_interruptions = _wait4_without_abandoning(process)
        post_start_errors.extend(wait_interruptions)
    finally:
        stdout_stream.flush()
        stderr_stream.flush()
        os.fsync(stdout_stream.fileno())
        os.fsync(stderr_stream.fileno())
        stdout_stream.close()
        stderr_stream.close()

    finished_at_utc = _utc_now()
    wall_time_seconds = (time.monotonic_ns() - monotonic_start) / 1_000_000_000
    child_outcome, exit_code, terminating_signal, returncode = _decode_wait_status(status)
    captures = {
        "stdout": _capture_record(stdout_spool, validated.stdout_path),
        "stderr": _capture_record(stderr_spool, validated.stderr_path),
    }
    for name, capture in captures.items():
        try:
            _publish_capture(capture)
        except BaseException as exc:
            post_start_errors.append(_error_record(f"{name}_publication", exc))

    if post_start_errors:
        outcome = "supervisor_post_start_error"
    elif child_outcome == "success":
        outcome = "success"
    elif child_outcome == "nonzero_exit":
        outcome = "child_nonzero_exit"
    elif child_outcome == "signal":
        outcome = "child_signaled"
    else:
        outcome = "unknown_child_outcome"

    terminal = {
        "schema": TERMINAL_SCHEMA,
        "sealed_at_utc": _utc_now(),
        "outcome": outcome,
        "process_started": True,
        "process_start_count": 1,
        "provenance": _provenance(validated),
        "process": {
            "pid": pid,
            "cwd": validated.package["bindings"]["cwd"],
            "argv": validated.package["bindings"]["argv"],
            "argv_sha256": validated.package["bindings"]["argv_sha256"],
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "child_outcome": child_outcome,
            "exit_code": exit_code,
            "signal": terminating_signal,
            "returncode": returncode,
            "wall_time_seconds": wall_time_seconds,
            "peak_rss_kib": _peak_rss_kib(usage),
        },
        "captures": captures,
        "post_start_errors": post_start_errors,
        "terminal_record_paths": _terminal_paths(validated),
    }
    return _seal_terminal(validated, terminal)


def _exit_code(record: dict[str, Any]) -> int:
    return {
        "success": 0,
        "child_nonzero_exit": 10,
        "child_signaled": 10,
        "supervisor_post_start_error": 11,
        "process_start_failed": 12,
    }.get(record["outcome"], 13)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        record = run_once(arguments.package, arguments.authority)
    except DuplicateExecution as exc:
        print(
            canonical_json_bytes(
                {"status": DuplicateExecution.category, "process_start_count": 0, "error": str(exc)}
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 21
    except SupervisorError as exc:
        print(
            canonical_json_bytes(
                {"status": SupervisorError.category, "process_start_count": 0, "error": str(exc)}
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 20
    print(
        canonical_json_bytes(
            {
                "status": record["outcome"],
                "process_start_count": record["process_start_count"],
                "terminal_record": record["terminal_record_paths"]["authority_bound"],
            }
        ).decode("utf-8")
    )
    return _exit_code(record)


if __name__ == "__main__":
    raise SystemExit(main())
