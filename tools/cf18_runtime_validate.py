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


class ValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root is not an object: {path}")
    return value


def record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"expected regular file: {path}")
    observed = path.stat()
    return {
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": sha256_file(path),
        "size": observed.st_size,
    }


def verify_digest(sidecar: str, target: str) -> None:
    parts = (PACKAGE / sidecar).read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[1] != target:
        raise ValidationError(f"malformed digest sidecar: {sidecar}")
    if parts[0] != sha256_file(PACKAGE / target):
        raise ValidationError(f"digest mismatch: {target}")


def verify_manifest() -> None:
    verify_digest("package-manifest.sha256", "package-manifest.json")
    verify_digest("review-request.sha256", "review-request.json")
    manifest = load_object(PACKAGE / "package-manifest.json")
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    actual = {
        path.relative_to(PACKAGE).as_posix(): record(path)
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.relative_to(PACKAGE).as_posix() not in excluded
    }
    if (
        manifest.get("schema") != "ace2-cf18-package-manifest-v1"
        or manifest.get("files") != actual
    ):
        raise ValidationError("CF18 immutable package closure changed")


def verify_bound_sources() -> None:
    manifest = load_object(PACKAGE / "source-constraint-manifest.json")
    for relative, expected in manifest["files"].items():
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        if record(path) != expected:
            raise ValidationError(f"CF18 bound source changed: {relative}")


def verify_predecessors() -> None:
    inventory = load_object(PACKAGE / "predecessor-cf16-cf17-inventory.json")
    for relative, expected in inventory["files"].items():
        if record(ROOT / relative) != expected:
            raise ValidationError(f"immutable predecessor changed: {relative}")


def verify_contract() -> None:
    candidate = load_object(PACKAGE / "authority-candidate.json")
    contract = load_object(PACKAGE / "contract.json")
    estimate = load_object(PACKAGE / "finite-runtime-estimate.json")
    endpoint_argv = candidate["exact_endpoint_invocation"]["argv"]
    forbidden = {
        "--resume",
        "--stop-after",
        "--prefill-skip-intermediate-lm-head",
        "--focus-ds32-token-step",
        "--focus-layer0-through-token-step",
        "--focus-layer-prefix-through-token-step",
    }
    if (
        candidate.get("status") != "CANDIDATE_ONLY_SEPARATE_AUTHORITY_REQUIRED"
        or candidate.get("authority_cardinality") != 1
        or candidate.get("execution_limit") != 1
        or candidate.get("nonce") != contract.get("nonce")
        or candidate.get("identity") != contract.get("identity")
        or any(value in endpoint_argv for value in forbidden)
        or endpoint_argv.count("--persistence-batch-commands") != 1
        or endpoint_argv[
            endpoint_argv.index("--persistence-batch-commands") + 1
        ]
        != "1024"
        or contract.get("complete_frozen_schedule_commands") != 1_306_104
        or contract.get("schedule_sha256")
        != "0fc65c947ac2a34ee096815304e1ee68ec65c279160db08d4901e072930c560b"
        or estimate.get("admission_timeout_seconds") != 144_000
        or estimate.get("status") != "FINITE_ESTIMATE_WITH_MARGIN"
    ):
        raise ValidationError("CF18 execution or finite-bound contract changed")


def verify_zero_state() -> None:
    candidate = load_object(PACKAGE / "authority-candidate.json")
    authority_path = PACKAGE.parent / f"{PACKAGE.name}-execution-authority.json"
    if (
        os.path.lexists(Path(candidate["state_namespace"]))
        or authority_path.exists()
    ):
        raise ValidationError("CF18 candidate is not in zero state")


def validate_package() -> None:
    verify_manifest()
    verify_bound_sources()
    verify_predecessors()
    verify_contract()
    verify_zero_state()


if __name__ == "__main__":
    validate_package()
    print("CF18_PACKAGE_VALID_ZERO_STATE")
