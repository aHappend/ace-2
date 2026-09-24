#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
IDENTITY = "stage1-w4a8-r6-generation-probe-terminal-accountability-cf17-0002"
STATE = PACKAGE.parent / f"{IDENTITY}-authority-state-0001"
CF16_PACKAGE = (
    PACKAGE.parent
    / "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0002"
)
CF16_STATE = (
    PACKAGE.parent
    / "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0002-authority-state-0001"
)


class ValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root is not an object: {path}")
    return value


def file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"expected regular file: {path}")
    observed = path.stat()
    return {
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": sha256_file(path),
        "size": observed.st_size,
    }


def verify_digest(path: Path, target_name: str) -> None:
    parts = path.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[1] != target_name:
        raise ValidationError(f"malformed digest sidecar: {path.name}")
    if sha256_file(path.parent / target_name) != parts[0]:
        raise ValidationError(f"digest mismatch: {target_name}")


def verify_package_manifest(package: Path = PACKAGE) -> None:
    verify_digest(package / "package-manifest.sha256", "package-manifest.json")
    manifest = load_object(package / "package-manifest.json")
    if (
        manifest.get("schema") != "ace2-r6-cf17-package-manifest-v1"
        or manifest.get("identity") != IDENTITY
    ):
        raise ValidationError("CF17 package manifest identity changed")
    expected = manifest.get("files")
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    actual = {
        path.relative_to(package).as_posix(): file_record(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.relative_to(package).as_posix() not in excluded
    }
    if actual != expected:
        raise ValidationError("CF17 immutable package closure changed")


def resolve_record_path(relative: str) -> Path:
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else ROOT / candidate


def verify_inventory(name: str, root: Path) -> None:
    inventory = load_object(PACKAGE / name)
    entries = inventory.get("entries")
    if not isinstance(entries, dict):
        raise ValidationError(f"{name} entries are malformed")
    actual = {
        path.relative_to(ROOT).as_posix(): file_record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if actual != entries:
        raise ValidationError(f"{name} exact read-only closure changed")


def verify_sources() -> None:
    verify_digest(
        PACKAGE / "source-constraint-manifest.sha256",
        "source-constraint-manifest.json",
    )
    manifest = load_object(PACKAGE / "source-constraint-manifest.json")
    if (
        manifest.get("rtl_change") != "NONE"
        or manifest.get("constraint_change") != "NONE"
        or manifest.get("public_rtl_contract")
        != "UNCHANGED_14_PARAMETERS_64_PORTS"
        or manifest.get("streaming_memory_boundary") != "ABSTRACT_UNCHANGED"
    ):
        raise ValidationError("CF17 source and constraint policy changed")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValidationError("CF17 source manifest files are malformed")
    for relative, expected in files.items():
        path = resolve_record_path(relative)
        observed = file_record(path)
        if observed != {
            key: expected[key] for key in ("mode", "sha256", "size")
        }:
            raise ValidationError(f"CF17 source or evidence binding changed: {relative}")
    path_bindings = manifest.get("path_bindings")
    if not isinstance(path_bindings, dict) or not path_bindings:
        raise ValidationError("CF17 canonical path bindings are absent")
    for source, target in path_bindings.items():
        path = Path(source)
        if not path.is_symlink() or path.resolve(strict=True) != Path(target):
            raise ValidationError(f"CF17 canonical path binding changed: {source}")


def verify_contract() -> None:
    authority = load_object(PACKAGE / "authority.json")
    contract = load_object(PACKAGE / "contract.json")
    endpoint = authority.get("rtl_endpoint_invocation", {})
    argv = endpoint.get("argv", [])
    if (
        authority.get("identity") != IDENTITY
        or authority.get("authority_cardinality") != 1
        or authority.get("execution_limit") != 1
        or authority.get("status")
        != "GRANTED_PENDING_INDEPENDENT_REVIEWER_DONE"
        or authority.get("software_fallback") != "FORBIDDEN"
        or authority.get("cf16_retry_replay_resume_relaunch")
        != "PERMANENTLY_FORBIDDEN"
        or authority.get("stage2") != "FORBIDDEN"
        or endpoint.get("software_fallback") != "FORBIDDEN"
        or endpoint.get("path_bindings")
        != load_object(PACKAGE / "source-constraint-manifest.json").get(
            "path_bindings"
        )
        or not isinstance(argv, list)
        or len(argv) < 3
        or any("cf16-0002/authority_runner.py" in value for value in argv)
        or any("product_probe.py" in value for value in argv)
        or "--output" not in argv
        or contract.get("terminal_publication")
        != "PRIMARY_OR_DISTINCT_FSYNCED_FALLBACK_REQUIRED"
    ):
        raise ValidationError("CF17 authority or terminal contract changed")


def exact_process_matches(authority: dict[str, Any]) -> list[int]:
    expected = {
        tuple(authority["launch_invocation"]["argv"]),
        tuple(authority["rtl_endpoint_invocation"]["argv"]),
    }
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = tuple(
            item.decode("utf-8", errors="surrogateescape")
            for item in raw.split(b"\0")
            if item
        )
        if argv in expected:
            matches.append(int(entry.name))
    return sorted(matches)


def verify_zero_state() -> None:
    authority = load_object(PACKAGE / "authority.json")
    zero = load_object(PACKAGE / "preconstruction-zero-state.json")
    if (
        os.path.lexists(STATE)
        or exact_process_matches(authority)
        or zero.get("status")
        != "PASS_FRESH_CF17_STATE_CF16_CONSUMED_STATE_READ_ONLY"
        or zero.get("cf16_terminal_seal_exists") is not False
    ):
        raise ValidationError("CF17 authority is not in its frozen zero state")


def validate_package(*, require_zero_state: bool = True) -> None:
    verify_package_manifest()
    verify_digest(PACKAGE / "review-request.sha256", "review-request.json")
    verify_sources()
    verify_inventory("predecessor-cf16-package-inventory.json", CF16_PACKAGE)
    verify_inventory("predecessor-cf16-state-inventory.json", CF16_STATE)
    verify_contract()
    if require_zero_state:
        verify_zero_state()


def main() -> None:
    validate_package()
    print("CF17_PACKAGE_VALID")


if __name__ == "__main__":
    main()
