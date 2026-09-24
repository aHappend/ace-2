#!/usr/bin/env python3
"""Order-independent V21 production binding resolution primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CANONICAL_TENSOR_NAMES = (
    "bf16.k_rope",
    "bf16.q_rope",
    "bf16.qk_scaled_scores",
)
EXPECTED_BINDING_RECORD_COUNT = 25


class BindingResolutionError(RuntimeError):
    """Deterministic V18 binding failure suitable for wrapper diagnostics."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code
        self.failure_stage = "V18_BINDING"


@dataclass(frozen=True)
class ResolvedSelection:
    canonical_names: tuple[str, ...]
    source_keys_by_name: dict[str, str]
    records_by_name: dict[str, dict[str, Any]]
    binding_records: dict[str, dict[str, Any]]


def require(condition: bool, message: str, code: str) -> None:
    if not condition:
        raise BindingResolutionError(message, code)


def _canonical_names(expected_names: tuple[str, ...]) -> tuple[str, ...]:
    require(type(expected_names) is tuple, "canonical tensor names type", "EXPECTED_NAMES_MALFORMED")
    require(len(expected_names) > 0, "canonical tensor names empty", "EXPECTED_NAMES_MALFORMED")
    require(
        all(type(name) is str and name for name in expected_names),
        "canonical tensor name value",
        "EXPECTED_NAMES_MALFORMED",
    )
    require(len(set(expected_names)) == len(expected_names), "canonical tensor names duplicate", "EXPECTED_NAMES_MALFORMED")
    return expected_names


def resolve_core_tensor_records(
    tensor_records: dict[str, dict[str, Any]],
    expected_names: tuple[str, ...] = CANONICAL_TENSOR_NAMES,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Resolve frozen core records by ``tensor_name``, never mapping order."""

    canonical_names = _canonical_names(expected_names)
    expected_set = set(canonical_names)
    require(type(tensor_records) is dict, "frozen tensor records type", "TENSOR_RECORDS_MALFORMED")

    records_by_name: dict[str, dict[str, Any]] = {}
    source_keys_by_name: dict[str, str] = {}
    for source_key, record in tensor_records.items():
        require(type(source_key) is str and source_key, "frozen tensor record source key", "TENSOR_RECORDS_MALFORMED")
        require(type(record) is dict, f"frozen tensor record type: {source_key}", "TENSOR_RECORD_MALFORMED")
        tensor_name = record.get("tensor_name")
        require(
            type(tensor_name) is str and tensor_name,
            f"frozen tensor_name: {source_key}",
            "TENSOR_RECORD_MALFORMED",
        )
        require(tensor_name in expected_set, f"unexpected frozen tensor_name: {tensor_name}", "UNEXPECTED_TENSOR_NAME")
        require(tensor_name not in records_by_name, f"duplicate frozen tensor_name: {tensor_name}", "DUPLICATE_TENSOR_NAME")
        records_by_name[tensor_name] = record
        source_keys_by_name[tensor_name] = source_key

    missing = [name for name in canonical_names if name not in records_by_name]
    require(not missing, f"missing frozen tensor_name: {','.join(missing)}", "MISSING_TENSOR_NAME")
    require(len(records_by_name) == len(canonical_names), "frozen tensor name cardinality", "TENSOR_NAME_CARDINALITY")

    return (
        {name: records_by_name[name] for name in canonical_names},
        {name: source_keys_by_name[name] for name in canonical_names},
    )


def resolve_production_selection(
    tensor_records: dict[str, dict[str, Any]],
    binding_table: dict[str, Any],
    expected_names: tuple[str, ...] = CANONICAL_TENSOR_NAMES,
) -> ResolvedSelection:
    """Resolve core records first, then validate the frozen V18 table."""

    canonical_names = _canonical_names(expected_names)
    records_by_name, source_keys_by_name = resolve_core_tensor_records(tensor_records, canonical_names)

    require(type(binding_table) is dict, "V18 binding table type", "BINDING_TABLE_MALFORMED")
    require(
        binding_table.get("record_count") == EXPECTED_BINDING_RECORD_COUNT,
        "V18 binding record_count",
        "BINDING_TABLE_RECORD_COUNT",
    )
    binding_records = binding_table.get("records")
    require(type(binding_records) is dict, "V18 binding records type", "BINDING_TABLE_MALFORMED")
    require(
        len(binding_records) == EXPECTED_BINDING_RECORD_COUNT,
        "V18 binding records cardinality",
        "BINDING_TABLE_RECORD_COUNT",
    )
    selected_names = binding_table.get("selected_tensor_names")
    require(type(selected_names) is list, "V18 selected names type", "BINDING_TABLE_SELECTED_NAMES")
    require(
        tuple(selected_names) == canonical_names,
        "V18 selected names",
        "BINDING_TABLE_SELECTED_NAMES",
    )

    return ResolvedSelection(
        canonical_names=canonical_names,
        source_keys_by_name=source_keys_by_name,
        records_by_name=records_by_name,
        binding_records=binding_records,
    )
