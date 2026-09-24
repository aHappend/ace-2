#!/home/argustest/miniconda3/bin/python3.13
"""Aggregate-only B0 source/oracle worker; raw tensor values never leave stdout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from qk_gbfp8_head64_b0_execution_common_v28 import (
    ACTION_ROOT, EXACT_ENVIRONMENT, OFFICIAL_EVALUATOR, OFFICIAL_PARSER, SOURCE_STATIC_ACCEPTANCE,
    STATIC_ACCEPTANCE, STATIC_ACCEPTANCE_FILE_SHA256, STATIC_ACCEPTANCE_SELF_SHA256,
    STATIC_PACKAGE, STATIC_PACKAGE_CONTENT_SHA256, STATIC_PACKAGE_FILE_SHA256, STATIC_SIDECAR_SCHEMA,
    V21_BINDING_TABLE, compact_bytes, require, sealed, sha256_file, verify_self_hash,
)


BINDING_TABLE_SHA256 = "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655"
PARSER_SHA256 = "ff9216d58bcd1584361e17d8d1854f3bcb96923c17927b1f309861546807bdb1"
EVALUATOR_SHA256 = "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63"
ROW_COUNT = 574
HEAD_COUNT = 14
SEQUENCE_LENGTH = 41


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module {name}", "EVALUATOR", "IMPORT")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value: Any) -> str:
    import hashlib
    return hashlib.sha256(compact_bytes(value)).hexdigest()


def sign(value: int) -> int:
    return 0 if value == 0 else 1 if value > 0 else -1


def top_and_margin(row: list[int]) -> tuple[int, int, int]:
    top_value = max(row)
    top = row.index(top_value)
    ties = sum(value == top_value for value in row)
    if len(row) == 1:
        return top, 0, ties
    other = max(value for index, value in enumerate(row) if index != top)
    return top, top_value - other, ties


def aggregate(row_class: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["id"] for row in rows]
    score_mismatches = [row["id"] for row in rows if not row["score_agreement"]]
    top_mismatches = [row["id"] for row in rows if not row["top_key_agreement"]]
    margin_mismatches = [row["id"] for row in rows if not row["margin_sign_agreement"]]
    deltas = [delta for row in rows for delta in row["deltas"]]
    return {
        "margin_sign_agreement_count": len(rows) - len(margin_mismatches),
        "margin_sign_mismatch_count": len(margin_mismatches),
        "margin_sign_mismatch_membership_sha256": digest(margin_mismatches),
        "maximum_absolute_error_q12_20_lsb": max((abs(delta) for delta in deltas), default=0),
        "row_class": row_class,
        "row_count": len(rows),
        "row_membership_sha256": digest(ids),
        "score_agreement_count": len(rows) - len(score_mismatches),
        "score_mismatch_count": len(score_mismatches),
        "score_mismatch_membership_sha256": digest(score_mismatches),
        "sum_absolute_error_q12_20_lsb": sum(abs(delta) for delta in deltas),
        "sum_signed_error_q12_20_lsb": sum(deltas),
        "sum_squared_error_q40_40_lsb2": sum(delta * delta for delta in deltas),
        "top_key_agreement_count": len(rows) - len(top_mismatches),
        "top_key_mismatch_count": len(top_mismatches),
        "top_key_mismatch_membership_sha256": digest(top_mismatches),
    }


def classify_rows(source_rows: list[list[int]], oracle_rows: list[list[int]]) -> tuple[str, list[dict[str, Any]]]:
    require(type(source_rows) is list and type(oracle_rows) is list and len(source_rows) == len(oracle_rows) == ROW_COUNT,
            "exact 574 rows", "EVALUATOR", "ROW_ABI")
    annotated: list[dict[str, Any]] = []
    for index, (source, oracle) in enumerate(zip(source_rows, oracle_rows, strict=True)):
        head, query = divmod(index, SEQUENCE_LENGTH)
        require(type(source) is list and type(oracle) is list and len(source) == len(oracle) == query + 1,
                "causal row shape", "EVALUATOR", "ROW_ABI")
        require(all(type(value) is int for value in source + oracle), "Q12.20 row type", "EVALUATOR", "ROW_ABI")
        source_top, source_margin, _ = top_and_margin(source)
        oracle_top, oracle_margin, oracle_ties = top_and_margin(oracle)
        row_class = "SINGLETON_ROWS" if query == 0 else "ORACLE_TIED_TOP_ROWS" if oracle_ties > 1 else "ORACLE_UNIQUE_POSITIVE_MARGIN_ROWS"
        annotated.append({
            "deltas": [lhs - rhs for lhs, rhs in zip(source, oracle, strict=True)],
            "id": [head, query],
            "margin_sign_agreement": sign(source_margin) == sign(oracle_margin),
            "row_class": row_class,
            "score_agreement": source == oracle,
            "top_key_agreement": source_top == oracle_top,
        })
    matched = all(row["score_agreement"] and row["top_key_agreement"] and row["margin_sign_agreement"] for row in annotated)
    return ("SOURCE_ORACLE_MATCH" if matched else "SOURCE_ORACLE_MISMATCH"), annotated


def residual_controls(schema: dict[str, Any], outcome: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(schema["properties"]["residual_control_aggregates"]["prefixItems"]):
        props = item["properties"]
        value = {key: definition["const"] for key, definition in props.items() if "const" in definition}
        if index == 0:
            value["outcome"] = outcome
        result.append(value)
    return result


def build_sidecar(source_rows: list[list[int]], oracle_rows: list[list[int]], authority_envelope_sha256: str) -> dict[str, Any]:
    schema = json.loads(STATIC_SIDECAR_SCHEMA.read_text(encoding="ascii"))
    outcome, rows = classify_rows(source_rows, oracle_rows)
    classes = [
        ("ALL_CAUSAL_ROWS", rows),
        ("SINGLETON_ROWS", [row for row in rows if row["row_class"] == "SINGLETON_ROWS"]),
        ("ORACLE_TIED_TOP_ROWS", [row for row in rows if row["row_class"] == "ORACLE_TIED_TOP_ROWS"]),
        ("ORACLE_UNIQUE_POSITIVE_MARGIN_ROWS", [row for row in rows if row["row_class"] == "ORACLE_UNIQUE_POSITIVE_MARGIN_ROWS"]),
    ]
    require([len(items) for _, items in classes] == [574, 14, 214, 346], "frozen row partitions", "EVALUATOR", "PARTITION")
    properties = schema["properties"]
    sidecar = {
        "artifact_kind": properties["artifact_kind"]["const"],
        "authority_envelope_sha256": authority_envelope_sha256,
        "claim_boundary": properties["claim_boundary"]["const"],
        "control": properties["control"]["const"],
        "identity_bindings": properties["identity_bindings"]["const"],
        "outcome": outcome,
        "partition_accounting": properties["partition_accounting"]["const"],
        "residual_control_aggregates": residual_controls(schema, outcome),
        "row_class_aggregates": [aggregate(name, items) for name, items in classes],
        "schema_version": properties["schema_version"]["const"],
        "v28_action_id": properties["v28_action_id"]["const"],
    }
    sidecar = sealed(sidecar, "sidecar_sha256")
    Draft202012Validator(schema).validate(sidecar)
    verify_self_hash(sidecar, "sidecar_sha256")
    return sidecar


def sum_dyadic_products(words_a: tuple[int, ...], words_b: tuple[int, ...], evaluator: Any) -> tuple[int, int]:
    terms: list[tuple[int, int]] = []
    for lhs, rhs in zip(words_a, words_b, strict=True):
        lhs_coefficient, lhs_exponent = evaluator._signed_bf16_dyadic(lhs)
        rhs_coefficient, rhs_exponent = evaluator._signed_bf16_dyadic(rhs)
        coefficient = lhs_coefficient * rhs_coefficient
        if coefficient:
            terms.append((coefficient, lhs_exponent + rhs_exponent - 3))
    if not terms:
        return 0, 0
    common = min(exponent for _, exponent in terms)
    coefficient = sum(value << (exponent - common) for value, exponent in terms)
    if coefficient == 0:
        return 0, 0
    while coefficient % 2 == 0:
        coefficient //= 2
        common += 1
    return coefficient, common


def realize_source_row(score_pairs: list[tuple[int, int]], evaluator: Any) -> list[int]:
    common = min((exponent for coefficient, exponent in score_pairs if coefficient), default=0)
    aligned = [0 if coefficient == 0 else coefficient << (exponent - common) for coefficient, exponent in score_pairs]
    maximum = max(aligned)
    return [evaluator._realize_oracle_q12_20(value - maximum, common) for value in aligned]


def records_to_rows(records: dict[str, dict[str, Any]], evaluator: Any) -> tuple[list[list[int]], list[list[int]]]:
    query = records["bf16.q_rope"]
    key = records["bf16.k_rope"]
    oracle = records["bf16.qk_scaled_scores"]
    source_rows: list[list[int]] = []
    oracle_rows: list[list[int]] = []
    for query_head in range(HEAD_COUNT):
        kv_head = evaluator.kv_head_for_query(query_head)
        for query_index in range(SEQUENCE_LENGTH):
            q_words = tuple(evaluator.bf16_word(query, (0, query_head, query_index, lane)) for lane in range(64))
            pairs = []
            for key_index in range(query_index + 1):
                k_words = tuple(evaluator.bf16_word(key, (0, kv_head, key_index, lane)) for lane in range(64))
                pairs.append(sum_dyadic_products(q_words, k_words, evaluator))
            source_rows.append(realize_source_row(pairs, evaluator))
            oracle_words = tuple(evaluator.bf16_word(oracle, (0, query_head, query_index, key_index)) for key_index in range(query_index + 1))
            oracle_values, _, _ = evaluator.exact_oracle_row(oracle_words)
            oracle_rows.append(list(oracle_values))
    return source_rows, oracle_rows


def production(authority_envelope_sha256: str, payload: bytes) -> dict[str, Any]:
    require(Path.cwd() == ACTION_ROOT, "worker cwd", "EVALUATOR", "INVOCATION")
    require(dict(os.environ) == EXACT_ENVIRONMENT, "worker environment", "EVALUATOR", "INVOCATION")
    require(len(authority_envelope_sha256) == 64, "authority envelope hash", "EVALUATOR", "INVOCATION")
    require(sha256_file(STATIC_PACKAGE) == STATIC_PACKAGE_FILE_SHA256, "static package hash", "EVALUATOR", "STATIC_BINDING")
    require(sha256_file(STATIC_ACCEPTANCE) == STATIC_ACCEPTANCE_FILE_SHA256, "static acceptance hash", "EVALUATOR", "STATIC_BINDING")
    static_package = json.loads(STATIC_PACKAGE.read_text(encoding="ascii"))
    static_acceptance = json.loads(STATIC_ACCEPTANCE.read_text(encoding="ascii"))
    require(static_package["package_content_sha256"] == STATIC_PACKAGE_CONTENT_SHA256, "static package content", "EVALUATOR", "STATIC_BINDING")
    require(static_acceptance["acceptance_sha256"] == STATIC_ACCEPTANCE_SELF_SHA256, "static acceptance self", "EVALUATOR", "STATIC_BINDING")
    require(static_acceptance["static_acceptance_grants_execution_authority"] is False, "acceptance authority boundary", "EVALUATOR", "STATIC_BINDING")
    require(sha256_file(V21_BINDING_TABLE) == BINDING_TABLE_SHA256, "V21 binding table", "EVALUATOR", "STATIC_BINDING")
    require(sha256_file(OFFICIAL_PARSER) == PARSER_SHA256 and sha256_file(OFFICIAL_EVALUATOR) == EVALUATOR_SHA256,
            "V21 evaluator lineage", "EVALUATOR", "STATIC_BINDING")
    binding_table = json.loads(V21_BINDING_TABLE.read_text(encoding="ascii"))
    bindings = binding_table["records"]
    parser = load_module("qk_gbfp8_head64_b0_bound_parser_v28", OFFICIAL_PARSER)
    evaluator = load_module("qk_gbfp8_head64_b0_bound_evaluator_v28", OFFICIAL_EVALUATOR)
    producer = parser.parse_c02_producer_bytes(payload, bindings)
    accepted = parser.parse_c02_accepted_reader(payload, bindings)
    require(producer == accepted, "cross-parser agreement", "EVALUATOR", "PARSER")
    source_rows, oracle_rows = records_to_rows(producer, evaluator)
    return build_sidecar(source_rows, oracle_rows, authority_envelope_sha256)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("production",), required=True)
    parser.add_argument("--static-package", required=True)
    parser.add_argument("--static-acceptance", required=True)
    parser.add_argument("--authority-envelope-sha256", required=True)
    arguments = parser.parse_args()
    try:
        require(arguments.static_package == str(STATIC_PACKAGE), "worker static package", "EVALUATOR", "INVOCATION")
        require(arguments.static_acceptance == str(STATIC_ACCEPTANCE), "worker static acceptance", "EVALUATOR", "INVOCATION")
        result = production(arguments.authority_envelope_sha256, sys.stdin.buffer.read())
    except Exception as error:
        import hashlib
        sys.stderr.write(f"B0_WORKER_FAILED:{type(error).__name__}:{hashlib.sha256(type(error).__name__.encode('ascii')).hexdigest()}\n")
        return 70
    sys.stdout.buffer.write(compact_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
