#!/usr/bin/env python3
"""Atomic authority consumer for the repair-0006 production continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
IDENTITY = (
    "stage1-official-rtl-backed-chat-demo-20260831T134355Z-"
    "attempt-0014-preparation-r5-continuation-repair-0006-production-executor"
)
CANONICAL_CONSUMPTION_NAMESPACE = Path(
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260831T134355Z-"
    "attempt-0014-r5-continuation-repair-0006-production-executor-"
    "consumption-state"
)
CANONICAL_RUNTIME_NAMESPACE = Path(
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260831T134355Z-"
    "runtime-output-0014-r5-continuation-repair-0006-production-executor"
)
PERMITTED_UNITS = [
    f"position-00/layer-{index:02d}" for index in range(11, 24)
]
SEAL_FILES = {"SHA256SUMS", "TREE_ROOT.sha256"}


class LaunchError(RuntimeError):
    pass


class DuplicateConsumptionError(LaunchError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        require(
            separator == "  "
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and relative
            and relative not in records,
            f"invalid checksum record: {line!r}",
        )
        records[relative] = digest
    return records


def validate_package(package: Path = PACKAGE) -> tuple[dict[str, Any], str]:
    require(package.is_dir() and not package.is_symlink(), "package is absent")
    sums = parse_sums(package / "SHA256SUMS")
    actual: set[str] = set()
    executable: list[str] = []
    for directory, names, filenames in os.walk(package, followlinks=False):
        root = Path(directory)
        for name in names:
            child = root / name
            require(not child.is_symlink(), f"package directory symlink: {child}")
            require(name != "__pycache__", f"bytecode cache in package: {child}")
        for name in filenames:
            child = root / name
            relative = child.relative_to(package).as_posix()
            require(
                child.is_file() and not child.is_symlink(),
                f"non-regular package member: {child}",
            )
            if relative not in SEAL_FILES:
                actual.add(relative)
            if stat.S_IMODE(child.stat().st_mode) & 0o111:
                executable.append(relative)
    require(actual == set(sums), "sealed package member set changed")
    require(
        executable == ["atomic_launcher.py"],
        "package executable selection changed",
    )
    for relative, expected in sums.items():
        require(
            sha256_file(package / relative) == expected,
            f"sealed package member changed: {relative}",
        )
    root = sha256_file(package / "SHA256SUMS")
    require(
        (package / "TREE_ROOT.sha256").read_text(encoding="ascii").strip()
        == f"{root}  SHA256SUMS",
        "package tree root changed",
    )
    manifest = load(package / "successor-manifest.json")
    require(
        manifest["identity"] == IDENTITY
        and manifest["permitted_unit_keys"] == PERMITTED_UNITS
        and manifest["first_incomplete_unit"] == PERMITTED_UNITS[0],
        "package identity or execution boundary changed",
    )
    return manifest, root


def validate_external_sources(manifest: dict[str, Any]) -> None:
    binding = manifest["production_source_closure"]
    index = PACKAGE / binding["path"]
    require(
        sha256_file(index) == binding["sha256"],
        "production source-binding index changed",
    )
    records = parse_sums(index)
    required = set(binding["required_members"])
    require(required <= set(records), "required production source is unbound")
    for relative in sorted(required):
        path = ROOT / relative
        require(
            path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == records[relative],
            f"bound production source changed: {relative}",
        )


def load_bound_source(
    manifest: dict[str, Any], member: str, module_name: str
) -> dict[str, Any]:
    binding = manifest["execution_members"][member]
    path = PACKAGE / binding["path"]
    require(
        path.is_file()
        and not path.is_symlink()
        and stat.S_IMODE(path.stat().st_mode) & 0o111 == 0
        and sha256_file(path) == binding["sha256"],
        f"bound {member} source changed",
    )
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(path),
        "__name__": module_name,
        "__package__": None,
    }
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), namespace)
    return namespace


def validate_pending_candidate(manifest: dict[str, Any]) -> None:
    authority = load(PACKAGE / "authority.json")
    checkpoint = load(PACKAGE / "checkpoint-manifest.json")
    require(
        authority["status"] == "PENDING_FRESH_REVIEW"
        and authority["consumable"] is False
        and authority["unit_execution_authority"] == "NOT_GRANTED"
        and set(authority["activity"].values()) == {0},
        "candidate no-execution authority gate changed",
    )
    require(
        checkpoint["completed_unit_keys"]
        == [f"position-00/layer-{index:02d}" for index in range(11)]
        and checkpoint["next_incomplete_unit"] == PERMITTED_UNITS[0]
        and manifest["authenticated_completed_unit_keys"]
        == checkpoint["completed_unit_keys"],
        "authenticated layer-00 through layer-10 boundary changed",
    )


def validate_review_and_authority(
    review_path: Path,
    authority_path: Path,
    manifest: dict[str, Any],
    package_root: str,
) -> tuple[dict[str, Any], str]:
    require(
        review_path.is_file()
        and not review_path.is_symlink()
        and authority_path.is_file()
        and not authority_path.is_symlink(),
        "separate Fresh Review and authority are required",
    )
    review = load(review_path)
    authority = load(authority_path)
    authority_sha256 = sha256_file(authority_path)
    binding = {
        "package_identity": IDENTITY,
        "package_tree_root_sha256": package_root,
        "successor_manifest_sha256": sha256_file(
            PACKAGE / "successor-manifest.json"
        ),
        "production_unit_executor_sha256": manifest["execution_members"][
            "production_unit_executor"
        ]["sha256"],
        "atomic_launcher_sha256": manifest["execution_members"][
            "atomic_launcher"
        ]["sha256"],
        "continuation_runner_sha256": manifest["execution_members"][
            "continuation_runner"
        ]["sha256"],
        "consumption_namespace": CANONICAL_CONSUMPTION_NAMESPACE.as_posix(),
        "runtime_namespace": CANONICAL_RUNTIME_NAMESPACE.as_posix(),
    }
    require(
        review.get("decision") == "PASS"
        and review.get("independent") is True
        and review.get("execution_performed") is False
        and review.get("binding") == binding,
        "Fresh Review does not approve this exact no-execution package",
    )
    require(
        authority.get("status") == "SEALED_GRANTED_UNCONSUMED"
        and authority.get("consumable") is True
        and authority.get("binding") == binding
        and authority.get("permitted_unit_keys") == PERMITTED_UNITS
        and authority.get("authenticated_completed_unit_keys")
        == [f"position-00/layer-{index:02d}" for index in range(11)],
        "separate execution authority is absent or has the wrong scope",
    )
    return authority, authority_sha256


def validate_execution_namespaces(
    manifest: dict[str, Any],
    consumption: Path,
    runtime: Path,
) -> tuple[Path, Path]:
    namespaces = manifest.get("future_namespaces")
    require(
        isinstance(namespaces, dict)
        and namespaces.get("consumption")
        == CANONICAL_CONSUMPTION_NAMESPACE.as_posix()
        and namespaces.get("runtime") == CANONICAL_RUNTIME_NAMESPACE.as_posix(),
        "sealed package does not bind the canonical execution namespaces",
    )
    canonical_consumption = ROOT / CANONICAL_CONSUMPTION_NAMESPACE
    canonical_runtime = ROOT / CANONICAL_RUNTIME_NAMESPACE
    supplied_consumption = Path(os.path.abspath(consumption))
    supplied_runtime = Path(os.path.abspath(runtime))
    require(
        supplied_consumption == canonical_consumption,
        "alternate consumption namespace is forbidden",
    )
    require(
        supplied_runtime == canonical_runtime,
        "alternate runtime namespace is forbidden",
    )
    require(
        not os.path.lexists(canonical_consumption)
        or (
            canonical_consumption.is_dir()
            and not canonical_consumption.is_symlink()
        ),
        "canonical consumption namespace is not a real directory",
    )
    require(
        not os.path.lexists(canonical_runtime)
        or (canonical_runtime.is_dir() and not canonical_runtime.is_symlink()),
        "canonical runtime namespace is not a real directory",
    )
    return canonical_consumption, canonical_runtime


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_consume(
    consumption: Path,
    record: dict[str, Any],
    *,
    recover: bool,
) -> dict[str, Any]:
    consumption.mkdir(parents=True, exist_ok=True)
    final = consumption / "consumed.json"
    raw = canonical_bytes(record)
    if final.exists():
        existing = load(final)
        if recover and canonical_bytes(existing) == raw:
            return existing
        raise DuplicateConsumptionError("execution authority was already consumed")
    require(not recover, "cannot recover an authority that was not consumed")
    temporary = consumption / ".consumed.json.prepared"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
        )
    except FileExistsError as error:
        raise DuplicateConsumptionError("another consumption is being prepared") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, final, follow_symlinks=False)
        except FileExistsError as error:
            raise DuplicateConsumptionError("execution authority was already consumed") from error
        _fsync_directory(consumption)
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(consumption)
    return record


def _prepare_runtime(runtime: Path, *, recover: bool) -> Path:
    checkpoint = runtime / "checkpoint-manifest.json"
    if runtime.exists():
        require(recover and checkpoint.is_file(), "runtime namespace is not fresh")
        return checkpoint
    require(not recover, "recovery runtime namespace is absent")
    temporary = runtime.with_name(runtime.name + ".preparing")
    require(not os.path.lexists(temporary), "stale runtime preparation exists")
    temporary.mkdir(parents=True)
    shutil.copyfile(PACKAGE / "checkpoint-manifest.json", temporary / checkpoint.name)
    shutil.copytree(PACKAGE / "salvage-state", temporary / "salvage-state")
    os.replace(temporary, runtime)
    _fsync_directory(runtime.parent)
    return checkpoint


def execute(
    review_path: Path,
    authority_path: Path,
    consumption: Path,
    runtime: Path,
    *,
    recover: bool,
) -> dict[str, Any]:
    manifest, package_root = validate_package()
    validate_external_sources(manifest)
    validate_pending_candidate(manifest)
    authority, authority_sha256 = validate_review_and_authority(
        review_path, authority_path, manifest, package_root
    )
    consumption, runtime = validate_execution_namespaces(
        manifest, consumption, runtime
    )
    consumption_record = {
        "schema": "ace2-stage1-repair-0006-authority-consumption-v2",
        "authority_identity": authority["authority_identity"],
        "authority_sha256": authority_sha256,
        "package_identity": IDENTITY,
        "package_tree_root_sha256": package_root,
        "consumption_namespace": CANONICAL_CONSUMPTION_NAMESPACE.as_posix(),
        "runtime_namespace": CANONICAL_RUNTIME_NAMESPACE.as_posix(),
    }
    _atomic_consume(consumption, consumption_record, recover=recover)
    checkpoint = _prepare_runtime(runtime, recover=recover)
    runner = load_bound_source(manifest, "continuation_runner", "ace2_r6_runner")
    executor_module = load_bound_source(
        manifest, "production_unit_executor", "ace2_r6_executor"
    )
    executor = executor_module["make_production_unit_executor"](runtime)
    result = runner["run_continuation"](checkpoint, executor)
    terminal = {
        "schema": "ace2-stage1-repair-0006-terminal-v1",
        "status": "PASS_CONTINUATION_COMPLETE",
        "authority_consumption": consumption_record,
        "completed_unit_keys": result["completed_unit_keys"],
        "next_incomplete_unit": result["next_incomplete_unit"],
    }
    terminal_path = consumption / "terminal.json"
    require(not terminal_path.exists(), "terminal record already exists")
    temporary = consumption / ".terminal.json.prepared"
    temporary.write_bytes(canonical_bytes(terminal))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, terminal_path)
    _fsync_directory(consumption)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--consumption", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    require(args.verify_only != args.execute, "select exactly one launch mode")
    if args.verify_only:
        manifest, package_root = validate_package()
        validate_external_sources(manifest)
        validate_pending_candidate(manifest)
        print(
            "ACE2_REPAIR_0006_PRODUCTION_EXECUTOR_CLOSURE_PASS "
            f"authority_root={package_root} first_incomplete=position-00/layer-11 "
            "authority=0 consumption=0 submission=0 model=0 rtl=0 output=0 "
            "consumable=false review=PENDING_FRESH_REVIEW"
        )
        return
    require(
        all(
            path is not None
            for path in (
                args.review,
                args.authority,
                args.consumption,
                args.runtime,
            )
        ),
        "execution requires review, authority, consumption, and runtime paths",
    )
    terminal = execute(
        args.review,
        args.authority,
        args.consumption,
        args.runtime,
        recover=args.recover,
    )
    print(
        "ACE2_REPAIR_0006_CONTINUATION_PASS "
        f"completed={len(terminal['completed_unit_keys'])}"
    )


if __name__ == "__main__":
    main()
