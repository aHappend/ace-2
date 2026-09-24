#!/usr/bin/env python3
"""Construct and adversarially validate a fresh position-01 package."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("position01_builder", ROOT / "tools/build_position01_causal_state_package.py")
validator = load_module("position01_validator", ROOT / "tools/validate_position01_causal_state_package.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if set(value) == {"bytes", "path", "sha256"}:
            records.append(value)
        else:
            for child in value.values():
                records.extend(direct_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(direct_records(child))
    return records


def snapshot(records: list[dict[str, Any]]) -> dict[str, str]:
    return {record["path"]: sha256_file(ROOT / record["path"]) for record in records}


def copy_dependencies(manifest: dict[str, Any], destination: Path) -> None:
    for record in direct_records(manifest):
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / record["path"], target)
    for name in ("grant0005_package", "layer23_package"):
        relative = manifest["sources"][name]["path"]
        shutil.copytree(ROOT / relative, destination / relative, dirs_exist_ok=True)


def expect_rejection(package: Path, fixture: Path, label: str) -> str:
    try:
        validator.validate(package, fixture)
    except validator.ValidationError:
        return f"PASS_REJECTED_{label}"
    raise RuntimeError(f"validator accepted hostile case: {label}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-package", required=True, type=Path)
    parser.add_argument("--fresh-package", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    canonical = args.canonical_package.resolve()
    fresh = args.fresh_package.resolve()
    result_path = args.result.resolve()
    if os.path.lexists(fresh) or os.path.lexists(result_path):
        raise RuntimeError("fresh package and validation result must be absent")

    canonical_manifest, canonical_root = validator.validate_package_tree(canonical)
    records = direct_records(canonical_manifest)
    before = snapshot(records)
    construction = builder.build(fresh)
    validation = validator.validate(fresh, ROOT)
    fresh_manifest, fresh_root = validator.validate_package_tree(fresh)
    if fresh_manifest != canonical_manifest or fresh_root != canonical_root:
        raise RuntimeError("fresh construction is not byte-deterministic")

    with tempfile.TemporaryDirectory(prefix="ace2-position01-causal-state-") as temporary:
        fixture = Path(temporary) / "repository"
        fixture.mkdir()
        copy_dependencies(fresh_manifest, fixture)
        validator.validate(fresh, fixture)
        missing = fresh_manifest["causal_kv_publications"][0]["k_s8"]["path"]
        (fixture / missing).unlink()
        missing_status = expect_rejection(fresh, fixture, "MISSING_CAUSAL_KV")
        shutil.copyfile(ROOT / missing, fixture / missing)
        substituted = fresh_manifest["causal_kv_publications"][23]["v_s8"]["path"]
        original = (fixture / substituted).read_bytes()
        (fixture / substituted).write_bytes(bytes([original[0] ^ 1]) + original[1:])
        substituted_status = expect_rejection(fresh, fixture, "SUBSTITUTED_CAUSAL_KV")

    if snapshot(records) != before:
        raise RuntimeError("protected source state changed during construction validation")
    result = {
        "activity": validator.ZERO_ACTIVITY,
        "canonical_package_tree_root_sha256": canonical_root,
        "checks": {
            "fresh_construction": construction["status"],
            "fresh_package_native_validation": validation["status"],
            "missing_state": missing_status,
            "protected_state_unchanged": "PASS",
            "substituted_state": substituted_status,
        },
        "fresh_package": fresh.as_posix(),
        "fresh_package_tree_root_sha256": fresh_root,
        "schema": "ace2-position01-causal-state-construction-validation-v1",
        "stage_status": fresh_manifest["stage_status"],
        "status": "PASS_POSITION01_CAUSAL_STATE_CONSTRUCTION_ONLY",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(builder.canonical_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
