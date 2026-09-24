#!/usr/bin/env python3
"""Pure aggregate-only B0 row identity classifier for V28 synthetic/future use."""

from __future__ import annotations

import hashlib
import json
from typing import Any

ROW_COUNT = 574
HEAD_COUNT = 14
SEQUENCE_LENGTH = 41
MATCH = "SOURCE_ORACLE_MATCH"
MISMATCH = "SOURCE_ORACLE_MISMATCH"


class IdentityControlError(RuntimeError):
    pass


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(compact_bytes(value)).hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise IdentityControlError(detail)


def row_ids() -> list[list[int]]:
    return [[head, query] for head in range(HEAD_COUNT) for query in range(SEQUENCE_LENGTH)]


def classify_rows(source_rows: list[list[int]], oracle_rows: list[list[int]]) -> dict[str, Any]:
    require(type(source_rows) is list and type(oracle_rows) is list, "row arrays")
    require(len(source_rows) == len(oracle_rows) == ROW_COUNT, "exact 574-row population")
    matches: list[list[int]] = []
    mismatches: list[list[int]] = []
    signed = absolute = squared = maximum = 0
    ids = row_ids()
    for index, (source, oracle) in enumerate(zip(source_rows, oracle_rows, strict=True)):
        query = ids[index][1]
        require(type(source) is list and type(oracle) is list, "row type")
        require(len(source) == len(oracle) == query + 1, "causal row width")
        require(all(type(value) is int for value in source + oracle), "integer Q12.20 rows")
        equal = source == oracle
        (matches if equal else mismatches).append(ids[index])
        for lhs, rhs in zip(source, oracle, strict=True):
            delta = lhs - rhs
            signed += delta
            absolute += abs(delta)
            squared += delta * delta
            maximum = max(maximum, abs(delta))
    classification = MATCH if not mismatches else MISMATCH
    return {
        "classification": classification,
        "match_membership_sha256": digest(matches),
        "maximum_absolute_error_q12_20_lsb": maximum,
        "mismatch_membership_sha256": digest(mismatches),
        "row_count": ROW_COUNT,
        "row_membership_sha256": digest(ids),
        "source_oracle_match_count": len(matches),
        "source_oracle_mismatch_count": len(mismatches),
        "sum_absolute_error_q12_20_lsb": absolute,
        "sum_signed_error_q12_20_lsb": signed,
        "sum_squared_error_q40_40_lsb2": squared,
    }


def synthetic_rows() -> list[list[int]]:
    rows: list[list[int]] = []
    for head in range(HEAD_COUNT):
        for query in range(SEQUENCE_LENGTH):
            rows.append([((head + 1) * 1000003) - (query * 4099) + (key * 17) for key in range(query + 1)])
    return rows
