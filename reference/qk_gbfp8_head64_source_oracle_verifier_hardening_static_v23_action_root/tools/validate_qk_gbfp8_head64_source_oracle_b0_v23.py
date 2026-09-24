#!/usr/bin/env python3
"""Draft 2020-12 plus semantic validator for synthetic-only V23 B0 fixtures."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROW_CLASSES = (
    "ALL_CAUSAL_ROWS",
    "SINGLETON_ROWS",
    "ORACLE_TIED_TOP_ROWS",
    "ORACLE_UNIQUE_POSITIVE_MARGIN_ROWS",
)
RESIDUAL_LABELS = ("B0", "G1", "G2", "G4", "G8")


class B0ValidationError(RuntimeError):
    pass


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def self_hash(value: dict[str, Any]) -> str:
    candidate = dict(value)
    candidate.pop("sidecar_sha256", None)
    return hashlib.sha256(compact_bytes(candidate)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if type(value) is not dict:
        raise B0ValidationError(f"JSON object required: {path}")
    return value


def _exact_once(observed: list[str], expected: tuple[str, ...], name: str) -> None:
    counts = Counter(observed)
    if len(observed) != len(expected) or set(counts) != set(expected) or any(counts[label] != 1 for label in expected):
        raise B0ValidationError(f"{name} exact-once membership: observed={observed}")


def validate_document(document: dict[str, Any], schema: dict[str, Any], expected_frozen: dict[str, Any]) -> dict[str, Any]:
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda item: tuple(str(part) for part in item.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise B0ValidationError(f"schema:{location}:{first.message}")
    rows = [item["row_class"] for item in document["row_class_aggregates"]]
    residuals = [item["label"] for item in document["diagnostic_residuals"]]
    _exact_once(rows, ROW_CLASSES, "row classes")
    _exact_once(residuals, RESIDUAL_LABELS, "residual labels")
    if document["frozen_contract"] != expected_frozen:
        raise B0ValidationError("frozen contract drift")
    if document["sidecar_sha256"] != self_hash(document):
        raise B0ValidationError("sidecar self hash")
    return {
        "row_class_count": len(rows),
        "row_classes": rows,
        "residual_label_count": len(residuals),
        "residual_labels": residuals,
        "status": "PASS_V23_B0_SYNTHETIC_EXACT_ONCE",
    }
