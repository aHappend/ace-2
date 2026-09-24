#!/usr/bin/env python3
"""Recursively validate DPRF evidence records with explicit history scoping."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SCOPES = {
    "historical_observation_not_live_binding",
    "historical_mutable_path_observation",
}
CURRENT_SCOPES = {
    None,
    "current_live_binding",
    "immutable_snapshot",
    "sealed_whole_file",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ValidationSummary:
    current_records: int = 0
    historical_records: int = 0
    recursive_documents: int = 0
    canonical_digests: int = 0


def validate_artifact_tree(
    value: dict[str, Any],
    *,
    label: str,
    root: Path = ROOT,
) -> ValidationSummary:
    """Validate every nested path/hash record and marked recursive JSON document.

    Historical mutable-path observations are accepted only when each record is
    explicitly scoped and explains why it is not a live binding. Current and
    immutable records are always checked against bytes on disk.
    """

    summary = ValidationSummary()
    visited: set[Path] = set()

    def validate_document(document: dict[str, Any], document_label: str) -> None:
        integrity = document.get("integrity")
        if isinstance(integrity, dict) and integrity.get("algorithm") == "sha256-canonical-json-v1":
            expected = integrity.get("canonical_sha256")
            require(isinstance(expected, str) and expected, f"{document_label}: canonical digest missing")
            require(expected == canonical_sha256(document), f"{document_label}: canonical digest differs")
            summary.canonical_digests += 1
        walk(document, document_label, None)

    def walk(node: Any, trail: str, inherited_scope: str | None) -> None:
        if isinstance(node, dict):
            scope = node.get("binding_scope", inherited_scope)
            if "path" in node and "sha256" in node:
                relative = node.get("path")
                expected_sha = node.get("sha256")
                require(isinstance(relative, str) and relative, f"{trail}: artifact path missing")
                require(isinstance(expected_sha, str) and len(expected_sha) == 64, f"{trail}: artifact hash invalid")
                require(scope in CURRENT_SCOPES or scope in HISTORICAL_SCOPES, f"{trail}: unknown binding_scope {scope!r}")
                if scope in HISTORICAL_SCOPES:
                    require(
                        isinstance(node.get("historical_reason"), str) and node["historical_reason"],
                        f"{trail}: historical artifact lacks historical_reason",
                    )
                    summary.historical_records += 1
                else:
                    path = root / relative
                    require(not Path(relative).is_absolute(), f"{trail}: absolute artifact path is not allowed")
                    require(path.is_file(), f"{trail}: artifact missing: {relative}")
                    if "bytes" in node:
                        require(
                            isinstance(node["bytes"], int) and node["bytes"] == path.stat().st_size,
                            f"{trail}: artifact bytes differ: {relative}",
                        )
                    require(sha256_file(path) == expected_sha, f"{trail}: artifact hash differs: {relative}")
                    summary.current_records += 1
                    content_validation = node.get("content_validation", "whole_file_only")
                    require(
                        content_validation in {"whole_file_only", "recursive_json"},
                        f"{trail}: invalid content_validation {content_validation!r}",
                    )
                    if content_validation == "recursive_json":
                        resolved = path.resolve()
                        if resolved not in visited:
                            visited.add(resolved)
                            nested = json.loads(path.read_text(encoding="utf-8"))
                            require(isinstance(nested, dict), f"{trail}: recursive JSON is not an object")
                            summary.recursive_documents += 1
                            validate_document(nested, f"{trail}->{relative}")
            for key, child in node.items():
                walk(child, f"{trail}.{key}", scope)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{trail}[{index}]", inherited_scope)

    validate_document(value, label)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="repository-relative JSON evidence roots")
    args = parser.parse_args()
    totals = ValidationSummary()
    for relative in args.paths:
        path = ROOT / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(value, dict), f"expected JSON object: {relative}")
        result = validate_artifact_tree(value, label=relative)
        totals.current_records += result.current_records
        totals.historical_records += result.historical_records
        totals.recursive_documents += result.recursive_documents
        totals.canonical_digests += result.canonical_digests
    print(
        "ACE2_DPRF_RECURSIVE_EVIDENCE_PASS "
        f"roots={len(args.paths)} current_records={totals.current_records} "
        f"historical_records={totals.historical_records} "
        f"recursive_documents={totals.recursive_documents} "
        f"canonical_digests={totals.canonical_digests}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
