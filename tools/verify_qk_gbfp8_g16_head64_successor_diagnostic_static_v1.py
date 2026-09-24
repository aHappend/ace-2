#!/usr/bin/env python3
"""Decisive synthetic-only verifier for the static G16 diagnostic package."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import random
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "design/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_CONTRACT.json"
EVALUATOR_PATH = ROOT / "reference/qk_gbfp8_g16_head64_successor_diagnostic_static_v1.py"
SCHEMA_PATH = ROOT / "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_SCHEMA.json"
PROOF_PATH = ROOT / "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_NO_EXECUTION_PROOF.json"
PACKAGE_PATH = ROOT / "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE.json"
PACKAGE_SHA_PATH = Path(str(PACKAGE_PATH) + ".sha256")
ACCEPTED_G16_PACKAGE_PATH = ROOT / "reference/QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1_PACKAGE.json"
ACCEPTED_REVIEW_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/46d03751e890/round-0002.json"
)
OUTPUT_ROOT = ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1"
AUTHORITY_ROOT = ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority"
EXPECTED_G16_PACKAGE_SHA256 = "d6e4969660de746f7123b29e80f49febcf18fdeca8ffab234f4849a8dd8feea8"
EXPECTED_REVIEW_SHA256 = "fe36c4c7dee34bfd7a77647e8f2737353ae280d955a93df019baf5312a3e6cc0"
FORBIDDEN_PAYLOAD_SUFFIX = "attention-substage-tensors.bin"
EXPECTED_NONFINITE_SCAN_ORDER = (
    "Before masking, indexing, encoding, metric accumulation, or output construction, scan every BF16 word "
    "in all three selected records and reject the lane if any word is NaN or infinity."
)
EXPECTED_RANK_MARGIN_EMPTY_SET = (
    "If unique_oracle_top_row_count is zero, minimum_realized_margin_q12_20_lsb and "
    "preserved_positive_margin_fraction are null and the fraction threshold fails."
)
EXPECTED_TOP_KEY_EMPTY_SET = (
    "If row_count is zero, matching_fraction is null and the fraction threshold fails."
)
EXPECTED_REPAIR_POLICY = (
    "No inference, defaulting, truncation, compatibility alias, partial publication, retry, replay, resume, "
    "repair, recalibration, regeneration, or result substitution is permitted."
)
EXPECTED_FAILED_TERMINAL = (
    "Exactly one authorized invocation occurred after a valid cardinality-one authority was consumed. "
    "Pre-metric failures raised after authority consumption publish null metrics and thresholds and use only "
    "INPUT_BINDING_REJECTED, INPUT_SCHEMA_REJECTED, NON_FINITE_INPUT_REJECTED, NORMALIZATION_REJECTED, "
    "REPRESENTATION_REJECTED, or "
    "RESULT_SCHEMA_REJECTED; hard-threshold failure publishes complete metrics and threshold evaluation "
    "with all_hard_thresholds_pass=false and reason HARD_THRESHOLD_FAILED."
)
EXPECTED_CLI_REJECTION = (
    "Failures before authority consumption, including CROSS_LANE_REJECTED, publish no result and exit nonzero. "
    "OUTPUT_PUBLICATION_REJECTED is an unpublishable create-only publication failure: it publishes no result, "
    "exits nonzero, and permits no retry, fallback, substitution, or overwrite."
)
EXPECTED_RESULT_REASON_CODES = [
    "HARD_THRESHOLDS_PASSED",
    "HARD_THRESHOLD_FAILED",
    "INPUT_BINDING_REJECTED",
    "INPUT_SCHEMA_REJECTED",
    "NON_FINITE_INPUT_REJECTED",
    "NO_EXECUTION_AUTHORITY",
    "NORMALIZATION_REJECTED",
    "REPRESENTATION_REJECTED",
    "RESULT_SCHEMA_REJECTED",
]
EXPECTED_INVOCATION_ARGV = {
    "Base": [
        "/home/argustest/miniconda3/bin/python3.13",
        "-I",
        "-B",
        "reference/qk_gbfp8_g16_head64_successor_diagnostic_static_v1.py",
        "--package",
        "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE.json",
        "--authority",
        "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority.json",
        "--ledger",
        "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority-ledger.json",
        "--lane",
        "Base",
        "--metadata",
        "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json",
        "--tensor-bundle",
        "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin",
        "--output",
        "build/qk-gbfp8-g16-head64-successor-diagnostic-v1/base/result.json",
    ],
    "checkpoint-176": [
        "/home/argustest/miniconda3/bin/python3.13",
        "-I",
        "-B",
        "reference/qk_gbfp8_g16_head64_successor_diagnostic_static_v1.py",
        "--package",
        "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE.json",
        "--authority",
        "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/checkpoint-176/authority.json",
        "--ledger",
        "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/checkpoint-176/authority-ledger.json",
        "--lane",
        "checkpoint-176",
        "--metadata",
        "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/checkpoint-176/result.json",
        "--tensor-bundle",
        "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/checkpoint-176/attention-substage-tensors.bin",
        "--output",
        "build/qk-gbfp8-g16-head64-successor-diagnostic-v1/checkpoint-176/result.json",
    ],
}

sys.dont_write_bytecode = True


def fail(message: str) -> None:
    raise AssertionError(message)


def sha256_static(path: Path) -> str:
    if path.name == FORBIDDEN_PAYLOAD_SUFFIX:
        fail("static verifier attempted to open a sealed tensor payload")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_object_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def exact_keys(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        fail(f"{context} exact-key mismatch")
    return value


def valid_sha(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_binding(value: Any, context: str) -> dict[str, Any]:
    binding = exact_keys(value, {"byte_count", "path", "sha256"}, context)
    if type(binding["byte_count"]) is not int or binding["byte_count"] <= 0:
        fail(f"{context} byte_count")
    if type(binding["path"]) is not str or not binding["path"]:
        fail(f"{context} path")
    if not valid_sha(binding["sha256"]):
        fail(f"{context} sha256")
    return binding


def validate_package(package: Any) -> dict[str, Any]:
    package = exact_keys(
        package,
        {
            "accepted_g16_package_sha256",
            "accepted_g16_review_mission_id",
            "artifact_kind",
            "artifacts",
            "authority",
            "claim_boundary",
            "contract_id",
            "evaluator_id",
            "package_id",
            "result_schema_id",
            "schema_version",
        },
        "package",
    )
    if package["artifact_kind"] != "qk_gbfp8_g16_head64_successor_diagnostic_static_pre_execution_package":
        fail("package artifact kind")
    if package["authority"] != "NO_EXECUTION_AUTHORITY" or package["schema_version"] != 1:
        fail("package authority/version")
    if package["contract_id"] != "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1":
        fail("package contract id")
    if package["package_id"] != "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE":
        fail("package id")
    if package["evaluator_id"] != "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_EVALUATOR_V1":
        fail("evaluator id")
    if package["result_schema_id"] != "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_RESULT_V1":
        fail("result schema id")
    if package["accepted_g16_package_sha256"] != EXPECTED_G16_PACKAGE_SHA256:
        fail("accepted G16 package identity")
    if package["accepted_g16_review_mission_id"] != "46d03751e890":
        fail("accepted review mission")
    artifacts = exact_keys(
        package["artifacts"],
        {"contract", "no_execution_proof", "reference_evaluator", "result_schema", "static_verifier"},
        "package artifacts",
    )
    for name, binding in artifacts.items():
        validate_binding(binding, f"package artifacts.{name}")
    claims = exact_keys(
        package["claim_boundary"],
        {
            "base_or_checkpoint_176_evaluated",
            "c02_invoked",
            "execution_authority_created",
            "model_or_sealed_tensor_accessed",
            "new_model_scores_computed",
            "numerical_improvement_claimed",
            "result_namespace_created",
            "rtl_or_stage_modified",
        },
        "package claim boundary",
    )
    if any(type(value) is not bool or value for value in claims.values()):
        fail("package claim boundary must be exact false booleans")
    return package


def invocation_digest(invocation: dict[str, Any]) -> str:
    descriptor = {
        "argv": invocation["argv"],
        "environment": invocation["environment"],
        "interpreter": invocation["interpreter"],
        "lane_label": invocation["lane_label"],
    }
    return canonical_object_sha256(descriptor)


def validate_contract(contract: Any) -> dict[str, Any]:
    if type(contract) is not dict:
        fail("contract type")
    if contract.get("contract_id") != "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1":
        fail("contract id")
    if contract.get("authority") != "NO_EXECUTION_AUTHORITY" or contract.get("schema_version") != 1:
        fail("contract authority/version")
    accepted = contract.get("accepted_g16", {})
    if accepted.get("alternative_id") != "QK_GBFP8_G16_E16_HEAD64_V1":
        fail("accepted alternative")
    expected_bindings = {
        "contract": "64a7d1efce2336721a81be7e8319b1740cebad2449ac44ea29ca1007d2e5b549",
        "package": EXPECTED_G16_PACKAGE_SHA256,
        "pure_reference": "f3a5ebcff18403c048f2b279b5af2e56acaa7f551e68bfb2c475b418ce27140f",
        "schema": "6dcda9692926962d5888fcb1ff99bf4f695d2cc8740d06a61b4b19c93cde2ef3",
        "static_verifier": "18378125fcc96e38c15be30dca75843caaa384f2beb01f917646df9ea89ac4e6",
    }
    for name, digest in expected_bindings.items():
        if validate_binding(accepted.get(name), f"accepted_g16.{name}")["sha256"] != digest:
            fail(f"accepted G16 {name} hash")
    review = accepted.get("fresh_l2_handoff", {})
    if review.get("mission_id") != "46d03751e890" or review.get("status") != "done":
        fail("accepted Fresh-L2 handoff")
    if review.get("byte_count") != 891 or review.get("sha256") != EXPECTED_REVIEW_SHA256:
        fail("accepted Fresh-L2 handoff binding")

    claims = contract.get("claim_boundary")
    if type(claims) is not dict or any(type(value) is not bool or value for value in claims.values()):
        fail("contract claim boundary")
    authority = contract.get("authority_interface", {})
    if authority.get("cardinality") != 1 or authority.get("required_state_before_execution") != "READY_UNCONSUMED":
        fail("future authority cardinality/state")
    if authority.get("planned_authority_root") != AUTHORITY_ROOT.relative_to(ROOT).as_posix():
        fail("future authority root")

    evaluation = contract.get("deterministic_evaluation", {})
    g16 = evaluation.get("g16_realization", {})
    if (g16.get("group_count"), g16.get("group_size_lanes"), g16.get("head_record_bytes")) != (4, 16, 72):
        fail("G16 geometry")
    if g16.get("query_head_to_kv_head") != "kv_head = floor(query_head/7) for query_head 0 through 13":
        fail("Q to KV mapping")
    oracle = evaluation.get("oracle", {})
    if oracle.get("nonfinite_scan_order") != EXPECTED_NONFINITE_SCAN_ORDER:
        fail("complete-record nonfinite scan ordering")
    selectors = evaluation.get("record_selectors", {})
    if selectors.get("batch_index") != 0 or selectors.get("sequence_length") != 41:
        fail("record selectors")
    expected_roles = {
        "bf16_oracle_scores": ("bf16.qk_scaled_scores", [1, 14, 41, 41]),
        "realized_key_source": ("bf16.k_rope", [1, 2, 41, 64]),
        "realized_query_source": ("bf16.q_rope", [1, 14, 41, 64]),
    }
    if set(selectors.get("roles", {})) != set(expected_roles):
        fail("record role closure")
    for role, (name, shape) in expected_roles.items():
        value = selectors["roles"][role]
        if value != {"dtype": "torch.bfloat16", "shape": shape, "tensor_name": name}:
            fail(f"record selector {role}")
    if "bf16.qk_centered_scores" not in selectors.get("excluded_tensor_names", []):
        fail("excluded oracle closure")

    lanes = contract.get("input_set", {}).get("lanes")
    if type(lanes) is not list or [lane.get("label") for lane in lanes] != ["Base", "checkpoint-176"]:
        fail("lane ordering")
    expected_lane_records = {
        "Base": {
            "bf16_oracle_scores": "49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765",
            "realized_key_source": "401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a",
            "realized_query_source": "285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429",
        },
        "checkpoint-176": {
            "bf16_oracle_scores": "776be2e58246c8641fa483e331e0d63896f96f14b3756548e15af3870d89c456",
            "realized_key_source": "2c4b99f5bc97d0f95b886dfebf62242148940d7cc9021e197dfdee0aed9e9902",
            "realized_query_source": "18671805abe3835d8f2e23bffb041ced6a05f41b8eac4f698f41d302def73b18",
        },
    }
    for lane in lanes:
        if set(lane.get("authoritative_tensors", {})) != set(expected_roles):
            fail("lane role set")
        for role, (name, shape) in expected_roles.items():
            record = lane["authoritative_tensors"][role]
            if record.get("tensor_name") != name or record.get("shape") != shape or record.get("dtype") != "torch.bfloat16":
                fail("lane record selector mismatch")
            if record.get("sha256") != expected_lane_records[lane["label"]][role]:
                fail("lane record hash")
        if lane["output_path"].startswith("build/qk-bfp8-e16-head64-v1-c02"):
            fail("c02 output namespace reused")

    invocations = contract.get("invocations")
    if type(invocations) is not list or [item.get("lane_label") for item in invocations] != ["Base", "checkpoint-176"]:
        fail("invocation ordering")
    for invocation in invocations:
        exact_keys(
            invocation,
            {"argv", "environment", "interpreter", "invocation_sha256", "lane_label"},
            f"invocation {invocation.get('lane_label')}",
        )
        lane_label = invocation["lane_label"]
        if invocation.get("argv") != EXPECTED_INVOCATION_ARGV[lane_label]:
            fail(f"exact invocation argv: {lane_label}")
        if invocation.get("environment") != {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}:
            fail("exact invocation environment")
        interpreter = invocation.get("interpreter", {})
        if interpreter != {
            "implementation": "CPython",
            "path": "/home/argustest/miniconda3/bin/python3.13",
            "sha256": "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad",
            "version": "3.13.5",
        }:
            fail("interpreter identity")
        if invocation.get("invocation_sha256") != invocation_digest(invocation):
            fail("invocation digest")

    limits = contract.get("thresholds", {})
    if limits.get("top_key_matching_fraction_minimum") != {"denominator": 1, "numerator": 1}:
        fail("top-key threshold")
    if limits.get("unique_oracle_positive_margin_preserved_fraction_minimum") != {"denominator": 1, "numerator": 1}:
        fail("margin threshold")
    if any(
        limits.get(name) != 0
        for name in (
            "cross_lane_record_count_maximum",
            "invalid_or_non_finite_value_count_maximum",
            "normalization_rejection_count_maximum",
            "positive_centered_realized_score_count_maximum",
            "rank_margin_violation_count_maximum",
            "saturation_event_count_maximum",
            "top_key_mismatch_count_maximum",
        )
    ):
        fail("hard zero thresholds")
    metrics = contract.get("metrics", {})
    if metrics.get("rank_margin", {}).get("empty_set") != EXPECTED_RANK_MARGIN_EMPTY_SET:
        fail("rank-margin empty-set semantics")
    if metrics.get("top_key", {}).get("empty_set") != EXPECTED_TOP_KEY_EMPTY_SET:
        fail("top-key empty-set semantics")
    if metrics.get("score_error", {}).get("gate_mode") != "TRACKING":
        fail("tracking-only score error")
    malformed = contract.get("malformed_input_policy", {})
    if malformed.get("repair_policy") != EXPECTED_REPAIR_POLICY:
        fail("retry/replay/resume/repair prohibition")
    output = contract.get("output", {})
    if output.get("root") != OUTPUT_ROOT.relative_to(ROOT).as_posix():
        fail("output root")
    if output.get("create_only") is not True:
        fail("create-only output")
    if output.get("cli_rejection_semantics") != EXPECTED_CLI_REJECTION:
        fail("CLI rejection/publication semantics")
    if output.get("terminal_semantics") != {
        "FAILED_TERMINAL": EXPECTED_FAILED_TERMINAL,
        "NO_EXECUTION_TERMINAL": (
            "No authority was consumed, invocation_count_performed is zero, and metrics and thresholds are null. "
            "This package creates no such result file."
        ),
        "SUCCEEDED_TERMINAL": "Exactly one invocation occurred and complete metrics satisfy every hard threshold.",
        "first_record_immutable": True,
        "retry_replay_resume_repair_permitted": False,
    }:
        fail("terminal semantics")
    return contract


def validate_nested_object_schema_closure(value: Any, context: str = "schema") -> None:
    if type(value) is dict:
        if value.get("type") == "object":
            properties = value.get("properties")
            required = value.get("required")
            if value.get("additionalProperties") is not False:
                fail(f"{context} object additionalProperties closure")
            if type(properties) is not dict or type(required) is not list or set(required) != set(properties):
                fail(f"{context} object required-field closure")
        for key, child in value.items():
            validate_nested_object_schema_closure(child, f"{context}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            validate_nested_object_schema_closure(child, f"{context}[{index}]")


def status_selector(status: str) -> dict[str, Any]:
    return {
        "properties": {
            "terminal": {
                "properties": {"status": {"const": status}},
                "required": ["status"],
            }
        },
        "required": ["terminal"],
    }


def terminal_constraints(
    invocation_count: int, metrics_published: bool, reason: dict[str, Any], thresholds_evaluated: bool
) -> dict[str, Any]:
    return {
        "properties": {
            "invocation_count_performed": {"const": invocation_count},
            "metrics_published": {"const": metrics_published},
            "reason_code": reason,
            "thresholds_evaluated": {"const": thresholds_evaluated},
        },
        "required": [
            "invocation_count_performed",
            "metrics_published",
            "reason_code",
            "thresholds_evaluated",
        ],
    }


def threshold_payload_schema(passed: bool) -> dict[str, Any]:
    return {
        "allOf": [
            {"$ref": "#/$defs/threshold_evaluation"},
            {
                "properties": {"all_hard_thresholds_pass": {"const": passed}},
                "required": ["all_hard_thresholds_pass"],
            },
        ]
    }


def expected_terminal_conditionals() -> list[dict[str, Any]]:
    pre_metric_reasons = [
        "INPUT_BINDING_REJECTED",
        "INPUT_SCHEMA_REJECTED",
        "NON_FINITE_INPUT_REJECTED",
        "NORMALIZATION_REJECTED",
        "REPRESENTATION_REJECTED",
        "RESULT_SCHEMA_REJECTED",
    ]
    return [
        {
            "if": status_selector("SUCCEEDED_TERMINAL"),
            "then": {
                "properties": {
                    "metrics": {"$ref": "#/$defs/metrics"},
                    "terminal": terminal_constraints(1, True, {"const": "HARD_THRESHOLDS_PASSED"}, True),
                    "threshold_evaluation": threshold_payload_schema(True),
                }
            },
        },
        {
            "if": status_selector("NO_EXECUTION_TERMINAL"),
            "then": {
                "properties": {
                    "metrics": {"type": "null"},
                    "terminal": terminal_constraints(0, False, {"const": "NO_EXECUTION_AUTHORITY"}, False),
                    "threshold_evaluation": {"type": "null"},
                }
            },
        },
        {
            "if": status_selector("FAILED_TERMINAL"),
            "then": {
                "oneOf": [
                    {
                        "properties": {
                            "metrics": {"type": "null"},
                            "terminal": terminal_constraints(1, False, {"enum": pre_metric_reasons}, False),
                            "threshold_evaluation": {"type": "null"},
                        }
                    },
                    {
                        "properties": {
                            "metrics": {"$ref": "#/$defs/metrics"},
                            "terminal": terminal_constraints(1, True, {"const": "HARD_THRESHOLD_FAILED"}, True),
                            "threshold_evaluation": threshold_payload_schema(False),
                        }
                    },
                ]
            },
        },
    ]


def validate_schema(schema: Any) -> dict[str, Any]:
    if type(schema) is not dict or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("schema identity")
    if schema.get("additionalProperties") is not False:
        fail("schema top-level closure")
    required = schema.get("required")
    properties = schema.get("properties")
    if type(required) is not list or type(properties) is not dict or set(required) != set(properties):
        fail("schema required-field closure")
    if properties.get("schema_id", {}).get("const") != "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_RESULT_V1":
        fail("schema id")
    definitions = schema.get("$defs", {})
    if set(definitions) != {
        "artifact_binding",
        "fraction",
        "fraction_check",
        "integer_check",
        "metrics",
        "tensor_record",
        "threshold_evaluation",
    }:
        fail("schema definition closure")
    validate_nested_object_schema_closure(schema)
    if schema.get("allOf") != expected_terminal_conditionals():
        fail("terminal status/schema semantic closure")
    threshold_properties = definitions.get("threshold_evaluation", {}).get("properties", {})
    if "score_error" in threshold_properties:
        fail("tracking score error entered hard thresholds")
    metric_properties = definitions.get("metrics", {}).get("properties", {})
    if set(metric_properties) != {
        "invalid_accounting",
        "rank_margin",
        "score_error",
        "top_key",
    }:
        fail("schema metric closure")
    expected_metric_fields = {
        "invalid_accounting": {
            "cross_lane_record_count",
            "invalid_or_non_finite_value_count",
            "normalization_rejection_count",
            "positive_centered_realized_score_count",
            "saturation_event_count",
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb",
            "oracle_tied_row_count",
            "preserved_positive_margin_fraction",
            "preserved_positive_margin_row_count",
            "singleton_valid_key_row_count",
            "unique_oracle_top_row_count",
            "violation_count",
        },
        "score_error": {
            "maximum_absolute_error_q12_20_lsb",
            "sum_absolute_error_q12_20_lsb",
            "sum_signed_error_q12_20_lsb",
            "sum_squared_error_q40_40_lsb2",
            "valid_value_count",
        },
        "top_key": {"matching_fraction", "matching_row_count", "mismatch_count", "row_count"},
    }
    for name, fields in expected_metric_fields.items():
        if set(metric_properties.get(name, {}).get("properties", {})) != fields:
            fail(f"schema nested metric closure: {name}")
    terminal = properties.get("terminal", {})
    if terminal.get("properties", {}).get("status", {}).get("enum") != [
        "SUCCEEDED_TERMINAL",
        "FAILED_TERMINAL",
        "NO_EXECUTION_TERMINAL",
    ]:
        fail("terminal status enumeration")
    if terminal.get("properties", {}).get("reason_code", {}).get("enum") != EXPECTED_RESULT_REASON_CODES:
        fail("terminal reason enumeration")
    if terminal.get("properties", {}).get("retry_replay_resume_repair_permitted") != {"const": False}:
        fail("terminal retry/replay prohibition")
    return schema


def validate_proof(proof: Any) -> dict[str, Any]:
    if type(proof) is not dict or proof.get("authority") != "NO_EXECUTION_AUTHORITY":
        fail("proof authority")
    counts = proof.get("event_counts")
    if type(counts) is not dict or any(type(value) is not int or value != 0 for value in counts.values()):
        fail("proof event counts")
    if proof.get("checked_absent_paths") != [
        OUTPUT_ROOT.relative_to(ROOT).as_posix(),
        AUTHORITY_ROOT.relative_to(ROOT).as_posix(),
    ]:
        fail("proof absent-path set")
    bindings = proof.get("immutable_bindings", {})
    if bindings.get("accepted_g16_package_sha256") != EXPECTED_G16_PACKAGE_SHA256:
        fail("proof accepted package")
    if bindings.get("accepted_g16_review_handoff_sha256") != EXPECTED_REVIEW_SHA256:
        fail("proof review")
    return proof


def verify_artifact_bindings(package: dict[str, Any]) -> None:
    for binding in package["artifacts"].values():
        path = ROOT / binding["path"]
        if not path.is_file() or path.stat().st_size != binding["byte_count"]:
            fail(f"artifact size binding: {binding['path']}")
        if sha256_static(path) != binding["sha256"]:
            fail(f"artifact hash binding: {binding['path']}")
    sidecar = PACKAGE_SHA_PATH.read_text(encoding="ascii").strip().split()
    if len(sidecar) != 2 or sidecar[0] != sha256_static(PACKAGE_PATH) or sidecar[1] != PACKAGE_PATH.name:
        fail("package sidecar")


def verify_accepted_static_bytes(contract: dict[str, Any]) -> int:
    checks = 0
    for name in ("contract", "package", "pure_reference", "schema", "static_verifier"):
        binding = contract["accepted_g16"][name]
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        if path.stat().st_size != binding["byte_count"] or sha256_static(path) != binding["sha256"]:
            fail(f"accepted G16 drift: {name}")
        checks += 1
    if sha256_static(ACCEPTED_G16_PACKAGE_PATH) != EXPECTED_G16_PACKAGE_SHA256:
        fail("accepted G16 package drift")
    accepted_package = load_json(ACCEPTED_G16_PACKAGE_PATH)
    if accepted_package.get("artifacts", {}).get("contract", {}).get("sha256") != contract["accepted_g16"]["contract"]["sha256"]:
        fail("accepted package indirect contract binding")
    if accepted_package.get("artifacts", {}).get("pure_reference", {}).get("sha256") != contract["accepted_g16"]["pure_reference"]["sha256"]:
        fail("accepted package indirect reference binding")
    if accepted_package.get("artifacts", {}).get("schema", {}).get("sha256") != contract["accepted_g16"]["schema"]["sha256"]:
        fail("accepted package indirect schema binding")
    checks += 4
    review_binding = contract["accepted_g16"]["fresh_l2_handoff"]
    if ACCEPTED_REVIEW_PATH.stat().st_size != review_binding["byte_count"] or sha256_static(ACCEPTED_REVIEW_PATH) != review_binding["sha256"]:
        fail("Fresh-L2 handoff drift")
    review = load_json(ACCEPTED_REVIEW_PATH)
    if review.get("review", {}).get("status") != "done" or "d6e496" not in review.get("review", {}).get("reason", ""):
        fail("Fresh-L2 acceptance content")
    checks += 2
    for binding in contract["preserved_c02_static_bindings"].values():
        path = ROOT / binding["path"]
        if path.stat().st_size != binding["byte_count"] or sha256_static(path) != binding["sha256"]:
            fail(f"accepted c02 static drift: {binding['path']}")
        checks += 1
    return checks


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction_rne(value: Fraction) -> int:
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude.numerator, magnitude.denominator)
    doubled = remainder * 2
    if doubled > magnitude.denominator or (doubled == magnitude.denominator and (quotient & 1)):
        quotient += 1
    return -quotient if sign < 0 else quotient


def bf16_fraction(bits: int) -> Fraction:
    sign = -1 if bits & 0x8000 else 1
    exponent_field = (bits >> 7) & 0xFF
    fraction = bits & 0x7F
    if exponent_field == 0xFF:
        raise ValueError("nonfinite")
    if exponent_field == 0:
        return Fraction(sign * fraction, 1 << 133)
    significand = sign * (128 + fraction)
    exponent = exponent_field - 134
    return Fraction(significand << exponent, 1) if exponent >= 0 else Fraction(significand, 1 << (-exponent))


def dyadic_fraction(coefficient: int, exponent: int) -> Fraction:
    return Fraction(coefficient << exponent, 1) if exponent >= 0 else Fraction(coefficient, 1 << (-exponent))


def independent_normalize(query: Any, keys: tuple[Any, ...], mask: tuple[bool, ...]) -> tuple[Any, int, int, int]:
    q_m, q_e = query
    terms = []
    nonzero_exponents = []
    for key in keys:
        k_m, k_e = key
        row_terms = []
        for group in range(4):
            partial = sum(q_m[lane] * k_m[lane] for lane in range(group * 16, group * 16 + 16))
            exponent = q_e[group] + k_e[group]
            row_terms.append((partial, exponent))
        terms.append(tuple(row_terms))
    for key_index, row_terms in enumerate(terms):
        if mask[key_index]:
            nonzero_exponents.extend(exponent for partial, exponent in row_terms if partial != 0)
    common = min(nonzero_exponents) if nonzero_exponents else 0
    scores = []
    for key_index, row_terms in enumerate(terms):
        if not mask[key_index]:
            scores.append(0)
        else:
            scores.append(sum(partial << (exponent - common) for partial, exponent in row_terms if partial))
    valid = [index for index, bit in enumerate(mask) if bit]
    top = max(valid, key=lambda index: (scores[index], -index))
    maximum = scores[top]
    output = []
    saturations = 0
    for index, bit in enumerate(mask):
        if not bit:
            output.append(0)
            continue
        value = fraction_rne(Fraction(scores[index] - maximum) * dyadic_fraction(1, common + 17))
        if value < -(1 << 63):
            value = -(1 << 63)
            saturations += 1
        output.append(value)
    return tuple(output), common, top, saturations


def canonical_encoded(rng: random.Random) -> tuple[tuple[int, ...], tuple[int, ...]]:
    mantissas: list[int] = []
    exponents: list[int] = []
    for _ in range(4):
        group = [rng.randint(-127, 127) for _ in range(16)]
        group[0] = rng.choice((-1, 1)) * rng.randint(64, 127)
        mantissas.extend(group)
        exponents.append(rng.randint(-8, 8))
    return tuple(mantissas), tuple(exponents)


def exact_arithmetic_cross_checks(evaluator: Any, g16: Any) -> tuple[int, int]:
    bf16_checks = 0
    for bits in range(1 << 16):
        if ((bits >> 7) & 0xFF) == 0xFF:
            continue
        coefficient, exponent = evaluator.signed_bf16_dyadic(bits)
        if dyadic_fraction(coefficient, exponent) != bf16_fraction(bits):
            fail("exhaustive BF16 exact decode")
        bf16_checks += 1
    rng = random.Random(0xA2E2_0016)
    row_checks = 0
    for row_index in range(240):
        query = canonical_encoded(rng)
        row_length = 1 + (row_index % 9)
        keys = tuple(canonical_encoded(rng) for _ in range(row_length))
        mask_list = [bool(rng.getrandbits(1)) for _ in range(row_length)]
        mask_list[row_index % row_length] = True
        mask = tuple(mask_list)
        actual = g16.normalize_score_row(query, keys, mask, "QK_GBFP8_G16_E16_HEAD64_V1")
        expected = independent_normalize(query, keys, mask)
        if actual != expected:
            fail("independent G16 arithmetic mismatch")
        row_checks += 1
    return bf16_checks, row_checks


def flat_index(shape: tuple[int, ...], indices: tuple[int, ...]) -> int:
    value = 0
    for index, dimension in zip(indices, shape):
        value = value * dimension + index
    return value


def set_word(payload: bytearray, shape: tuple[int, ...], indices: tuple[int, ...], word: int) -> None:
    offset = flat_index(shape, indices) * 2
    payload[offset : offset + 2] = word.to_bytes(2, "little")


def synthetic_records() -> dict[str, dict[str, Any]]:
    oracle_shape = (1, 14, 41, 41)
    key_shape = (1, 2, 41, 64)
    query_shape = (1, 14, 41, 64)
    oracle_payload = bytearray(2 * math.prod(oracle_shape))
    key_payload = bytearray(2 * math.prod(key_shape))
    query_payload = bytearray(2 * math.prod(query_shape))
    for kv_head in range(2):
        set_word(key_payload, key_shape, (0, kv_head, 0, 0), 0x3F80)
    for query_head in range(14):
        for query_index in range(41):
            set_word(query_payload, query_shape, (0, query_head, query_index, 0), 0x3F80)
            set_word(oracle_payload, oracle_shape, (0, query_head, query_index, 0), 0x3E00)
    return {
        "bf16_oracle_scores": {"dtype": "torch.bfloat16", "payload": bytes(oracle_payload), "shape": oracle_shape},
        "realized_key_source": {"dtype": "torch.bfloat16", "payload": bytes(key_payload), "shape": key_shape},
        "realized_query_source": {"dtype": "torch.bfloat16", "payload": bytes(query_payload), "shape": query_shape},
    }


def synthetic_evaluation_checks(
    evaluator: Any, g16: Any, contract: dict[str, Any]
) -> tuple[int, int, int, int]:
    selected = synthetic_records()
    metrics = evaluator.evaluate_selected_records(selected, g16)
    top = metrics["top_key"]
    rank = metrics["rank_margin"]
    score = metrics["score_error"]
    if top != {
        "matching_fraction": {"denominator": 1, "numerator": 1},
        "matching_row_count": 574,
        "mismatch_count": 0,
        "row_count": 574,
    }:
        fail("synthetic top-key metrics")
    if (
        rank["singleton_valid_key_row_count"],
        rank["unique_oracle_top_row_count"],
        rank["preserved_positive_margin_row_count"],
        rank["violation_count"],
        rank["oracle_tied_row_count"],
    ) != (14, 560, 560, 0, 0):
        fail("synthetic rank-margin populations")
    if rank["preserved_positive_margin_fraction"] != {"denominator": 1, "numerator": 1}:
        fail("synthetic margin fraction")
    if score["valid_value_count"] != 12054 or score["maximum_absolute_error_q12_20_lsb"] != 0:
        fail("synthetic score-error metrics")
    thresholds = evaluator.evaluate_thresholds(metrics, contract["thresholds"])
    if thresholds["all_hard_thresholds_pass"] is not True:
        fail("synthetic hard thresholds")

    reference_imports = 0
    original_import = evaluator.import_g16_reference

    def synthetic_import(expected_sha256: str) -> Any:
        nonlocal reference_imports
        if expected_sha256 != "0" * 64:
            fail("synthetic G16 import identity")
        reference_imports += 1
        return g16

    try:
        evaluator.import_g16_reference = synthetic_import
        imported_metrics = evaluator.evaluate_selected_records_from_reference(selected, "0" * 64)
    finally:
        evaluator.import_g16_reference = original_import
    if reference_imports != 1 or imported_metrics != metrics:
        fail("finite scan/import/evaluation sequence")

    nonfinite_checks = 0
    preimport_nonfinite_checks = 0

    class PoisonG16:
        def __getattr__(self, _name: str) -> Any:
            fail("numeric work began before complete selected-record finite scan")

    def poison_import(_expected_sha256: str) -> Any:
        fail("G16 import began before complete selected-record finite scan")

    nonfinite_locations = {
        "bf16_oracle_scores": (0, 0, 0, 40),
        "realized_key_source": (0, 1, 40, 63),
        "realized_query_source": (0, 13, 40, 63),
    }
    masked_oracle_checks = 0
    try:
        evaluator.import_g16_reference = poison_import
        for role in evaluator.SELECTED_ROLES:
            for _classification, bits in (
                ("nan", 0x7FC1),
                ("positive_infinity", 0x7F80),
                ("negative_infinity", 0xFF80),
            ):
                mutated = copy.deepcopy(selected)
                payload = bytearray(mutated[role]["payload"])
                set_word(payload, mutated[role]["shape"], nonfinite_locations[role], bits)
                mutated[role]["payload"] = bytes(payload)
                try:
                    evaluator.evaluate_selected_records(mutated, PoisonG16())
                except evaluator.EvaluationError as exc:
                    if exc.reason_code != "NON_FINITE_INPUT_REJECTED":
                        fail("nonfinite rejection reason")
                else:
                    fail("nonfinite selected record accepted")
                nonfinite_checks += 1
                try:
                    evaluator.evaluate_selected_records_from_reference(mutated, "0" * 64)
                except evaluator.EvaluationError as exc:
                    if exc.reason_code != "NON_FINITE_INPUT_REJECTED":
                        fail("pre-import nonfinite rejection reason")
                else:
                    fail("nonfinite selected record reached G16 import")
                preimport_nonfinite_checks += 1
                if role == "bf16_oracle_scores":
                    masked_oracle_checks += 1
    finally:
        evaluator.import_g16_reference = original_import
    return 3, nonfinite_checks, masked_oracle_checks, preimport_nonfinite_checks


def synthetic_argv_path_checks(evaluator: Any) -> int:
    base_paths = {
        "package": evaluator.PACKAGE_REL,
        "authority": "authority/authority.json",
        "ledger": "authority/authority-ledger.json",
        "metadata": "inputs/metadata.json",
        "tensor_bundle": "inputs/tensors.bin",
        "output": "outputs/result.json",
    }

    def argv(paths: dict[str, str]) -> list[str]:
        return [
            "reference/qk_gbfp8_g16_head64_successor_diagnostic_static_v1.py",
            "--package",
            paths["package"],
            "--authority",
            paths["authority"],
            "--ledger",
            paths["ledger"],
            "--lane",
            "Base",
            "--metadata",
            paths["metadata"],
            "--tensor-bundle",
            paths["tensor_bundle"],
            "--output",
            paths["output"],
        ]

    def expect_pre_read_rejection(temporary_root: Path, paths: dict[str, str]) -> None:
        old_root = evaluator.ROOT
        old_argv = evaluator.sys.argv
        old_load_json = evaluator.load_json
        old_cwd = Path.cwd()
        old_environment = dict(evaluator.os.environ)

        def poison_load_json(_path: Path) -> Any:
            fail("argv-bound artifact read began before plain-path validation")

        try:
            evaluator.ROOT = temporary_root
            evaluator.sys.argv = argv(paths)
            evaluator.load_json = poison_load_json
            evaluator.os.chdir(temporary_root)
            evaluator.os.environ.clear()
            evaluator.os.environ.update(evaluator.EXACT_ENVIRONMENT)
            try:
                evaluator._main()
            except evaluator.EvaluationError as exc:
                if exc.reason_code != "INPUT_BINDING_REJECTED":
                    fail("argv path rejection reason")
            else:
                fail("malformed or symlinked argv path accepted")
        finally:
            evaluator.os.chdir(old_cwd)
            evaluator.os.environ.clear()
            evaluator.os.environ.update(old_environment)
            evaluator.load_json = old_load_json
            evaluator.sys.argv = old_argv
            evaluator.ROOT = old_root

    checks = 0
    for name in base_paths:
        with tempfile.TemporaryDirectory(prefix=f"ace2-g16-{name}-symlink-") as temporary:
            temporary_root = Path(temporary)
            link = temporary_root / base_paths[name]
            link.parent.mkdir(parents=True, exist_ok=True)
            target = temporary_root / "symlink-target"
            target.write_bytes(b"synthetic\n")
            link.symlink_to(target)
            expect_pre_read_rejection(temporary_root, base_paths)
            checks += 1

    with tempfile.TemporaryDirectory(prefix="ace2-g16-component-symlink-") as temporary:
        temporary_root = Path(temporary)
        paths = dict(base_paths)
        paths["metadata"] = "linked-inputs/metadata.json"
        target_directory = temporary_root / "real-inputs"
        target_directory.mkdir()
        (temporary_root / "linked-inputs").symlink_to(target_directory, target_is_directory=True)
        expect_pre_read_rejection(temporary_root, paths)
        checks += 1

    lexical_mutations = {
        "package": "reference/./QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE.json",
        "authority": "authority/../authority.json",
        "ledger": "authority//authority-ledger.json",
        "metadata": "./inputs/metadata.json",
        "tensor_bundle": "inputs/tensors.bin/",
        "output": "outputs\\result.json",
    }
    for name, mutation in lexical_mutations.items():
        with tempfile.TemporaryDirectory(prefix=f"ace2-g16-{name}-alias-") as temporary:
            paths = dict(base_paths)
            paths[name] = mutation
            expect_pre_read_rejection(Path(temporary), paths)
            checks += 1
    return checks


def fault_injection_persistence_checks(evaluator: Any, lane: dict[str, Any], result: dict[str, Any]) -> int:
    checks = 0

    def fragmented_write(original: Any) -> Any:
        def injected(descriptor: int, payload: bytes) -> int:
            return original(descriptor, payload[: max(1, len(payload) // 3)])

        return injected

    def short_write_then_error(original: Any) -> Any:
        calls = 0

        def injected(descriptor: int, payload: bytes) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                return original(descriptor, payload[: max(1, len(payload) // 2)])
            raise OSError("synthetic failure after partial write")

        return injected

    def fsync_failure_at(call_index: int) -> Any:
        def factory(original: Any) -> Any:
            calls = 0

            def injected(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == call_index:
                    raise OSError(f"synthetic fsync failure at call {call_index}")
                original(descriptor)

            return injected

        return factory

    def result_case(
        *,
        write_factory: Any | None = None,
        fsync_factory: Any | None = None,
        succeeds: bool,
    ) -> None:
        nonlocal checks
        with tempfile.TemporaryDirectory(prefix="ace2-g16-result-fault-") as temporary:
            old_root = evaluator.ROOT
            old_write = evaluator.os.write
            old_fsync = evaluator.os.fsync
            temporary_root = Path(temporary)
            temporary_lane = copy.deepcopy(lane)
            temporary_lane["output_path"] = "base/result.json"
            output = temporary_root / temporary_lane["output_path"]
            try:
                evaluator.ROOT = temporary_root
                evaluator.prepare_output(output, temporary_lane)
                if write_factory is not None:
                    evaluator.os.write = write_factory(old_write)
                if fsync_factory is not None:
                    evaluator.os.fsync = fsync_factory(old_fsync)
                try:
                    evaluator.exclusive_publish(output, result)
                except evaluator.PublicationError as exc:
                    if succeeds or exc.reason_code != "OUTPUT_PUBLICATION_REJECTED":
                        fail("result fault-injection rejection")
                else:
                    if not succeeds:
                        fail("result fault injection falsely succeeded")
            finally:
                evaluator.os.write = old_write
                evaluator.os.fsync = old_fsync
                evaluator.ROOT = old_root
            expected_paths = ["base", "base/result.json"] if succeeds else ["base"]
            observed_paths = sorted(
                path.relative_to(temporary_root).as_posix() for path in temporary_root.rglob("*")
            )
            if observed_paths != expected_paths:
                fail("result fault injection left partial or staging output")
            if succeeds and output.read_bytes() != evaluator.canonical_json_bytes(result):
                fail("fragmented result write was not complete canonical JSON")
            if not succeeds and (output.exists() or output.is_symlink()):
                fail("failed result publication left a final output")
            checks += 1

    authority = {
        "authority_id": "synthetic-authority",
        "invocation_sha256": "4" * 64,
        "lane_label": lane["label"],
    }

    def ledger_case(
        *,
        write_factory: Any | None = None,
        fsync_factory: Any | None = None,
        succeeds: bool,
    ) -> None:
        nonlocal checks
        with tempfile.TemporaryDirectory(prefix="ace2-g16-ledger-fault-") as temporary:
            temporary_root = Path(temporary)
            authority_path = temporary_root / "authority.json"
            ledger_path = temporary_root / "authority-ledger.json"
            authority_path.write_bytes(evaluator.canonical_json_bytes(authority))
            authority_sha256 = evaluator.sha256_file(authority_path)
            expected_ledger = {
                "authority_id": authority["authority_id"],
                "authority_sha256": authority_sha256,
                "invocation_sha256": authority["invocation_sha256"],
                "lane_label": authority["lane_label"],
                "schema_id": "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_AUTHORITY_LEDGER_V1",
                "state": "CONSUMED",
            }
            old_write = evaluator.os.write
            old_fsync = evaluator.os.fsync
            try:
                if write_factory is not None:
                    evaluator.os.write = write_factory(old_write)
                if fsync_factory is not None:
                    evaluator.os.fsync = fsync_factory(old_fsync)
                try:
                    evaluator.consume_authority(
                        authority_path,
                        ledger_path,
                        authority,
                        authority_sha256,
                    )
                except OSError:
                    if succeeds:
                        fail("ledger fault-injection rejection")
                else:
                    if not succeeds:
                        fail("ledger fault injection falsely succeeded")
            finally:
                evaluator.os.write = old_write
                evaluator.os.fsync = old_fsync
            expected_paths = (
                ["authority-ledger.json", "authority.json"] if succeeds else ["authority.json"]
            )
            observed_paths = sorted(path.name for path in temporary_root.iterdir())
            if observed_paths != expected_paths:
                fail("ledger fault injection left malformed or staging ledger")
            if succeeds and ledger_path.read_bytes() != evaluator.canonical_json_bytes(expected_ledger):
                fail("fragmented ledger write was not complete canonical JSON")
            if not succeeds and (ledger_path.exists() or ledger_path.is_symlink()):
                fail("failed authority consumption left a final ledger")
            checks += 1

    result_case(write_factory=fragmented_write, succeeds=True)
    result_case(write_factory=short_write_then_error, succeeds=False)
    result_case(fsync_factory=fsync_failure_at(1), succeeds=False)
    result_case(fsync_factory=fsync_failure_at(2), succeeds=False)
    ledger_case(write_factory=fragmented_write, succeeds=True)
    ledger_case(write_factory=short_write_then_error, succeeds=False)
    ledger_case(fsync_factory=fsync_failure_at(1), succeeds=False)
    ledger_case(fsync_factory=fsync_failure_at(2), succeeds=False)
    return checks


def synthetic_control_flow_checks(evaluator: Any, g16: Any, contract: dict[str, Any]) -> int:
    lane = contract["input_set"]["lanes"][0]
    invocation_sha256 = contract["invocations"][0]["invocation_sha256"]
    hashes = ("1" * 64, "2" * 64, "3" * 64)
    metrics = evaluator.evaluate_selected_records(synthetic_records(), g16)
    thresholds = evaluator.evaluate_thresholds(metrics, contract["thresholds"])
    checks = 0

    success = evaluator.authorized_result(
        lane,
        contract,
        *hashes,
        invocation_sha256,
        lambda: (metrics, thresholds, "HARD_THRESHOLDS_PASSED"),
    )
    if success["terminal"]["status"] != "SUCCEEDED_TERMINAL":
        fail("authorized success control flow")
    checks += 1

    failed_metrics = copy.deepcopy(metrics)
    failed_metrics["top_key"].update(
        matching_fraction={"denominator": 574, "numerator": 573},
        matching_row_count=573,
        mismatch_count=1,
    )
    failed_thresholds = evaluator.evaluate_thresholds(failed_metrics, contract["thresholds"])
    threshold_failure = evaluator.authorized_result(
        lane,
        contract,
        *hashes,
        invocation_sha256,
        lambda: (failed_metrics, failed_thresholds, "HARD_THRESHOLD_FAILED"),
    )
    if threshold_failure["terminal"]["reason_code"] != "HARD_THRESHOLD_FAILED":
        fail("authorized hard-threshold control flow")
    checks += 1

    def reject(reason_code: str) -> Any:
        raise evaluator.EvaluationError(reason_code, "synthetic path rejection")

    for reason_code in sorted(evaluator.PRE_METRIC_FAILURE_REASON_CODES):
        failure = evaluator.authorized_result(
            lane,
            contract,
            *hashes,
            invocation_sha256,
            lambda reason_code=reason_code: reject(reason_code),
        )
        if failure["terminal"] != {
            "first_record_immutable": True,
            "invocation_count_performed": 1,
            "metrics_published": False,
            "reason_code": reason_code,
            "retry_replay_resume_repair_permitted": False,
            "status": "FAILED_TERMINAL",
            "thresholds_evaluated": False,
        }:
            fail(f"authorized terminal path: {reason_code}")
        checks += 1

    malformed_metrics = copy.deepcopy(metrics)
    malformed_metrics["top_key"].pop("mismatch_count")
    schema_failure = evaluator.authorized_result(
        lane,
        contract,
        *hashes,
        invocation_sha256,
        lambda: (malformed_metrics, thresholds, "HARD_THRESHOLDS_PASSED"),
    )
    if schema_failure["terminal"]["reason_code"] != "RESULT_SCHEMA_REJECTED":
        fail("result-schema translation path")
    checks += 1

    for reason_code in ("CROSS_LANE_REJECTED", "OUTPUT_PUBLICATION_REJECTED"):
        try:
            evaluator.authorized_result(
                lane,
                contract,
                *hashes,
                invocation_sha256,
                lambda reason_code=reason_code: reject(reason_code),
            )
        except evaluator.EvaluationError as exc:
            if exc.reason_code != reason_code:
                fail("CLI rejection reason translation")
        else:
            fail(f"CLI rejection was falsely published: {reason_code}")
        checks += 1

    with tempfile.TemporaryDirectory(prefix="ace2-g16-diagnostic-") as temporary:
        old_root = evaluator.ROOT
        temporary_root = Path(temporary)
        temporary_lane = copy.deepcopy(lane)
        temporary_lane["output_path"] = "base/result.json"
        output = temporary_root / temporary_lane["output_path"]
        try:
            evaluator.ROOT = temporary_root
            evaluator.prepare_output(output, temporary_lane)
            evaluator.exclusive_publish(output, success)
            first_bytes = output.read_bytes()
            checks += 1
            try:
                evaluator.exclusive_publish(output, threshold_failure)
            except evaluator.PublicationError as exc:
                if exc.reason_code != "OUTPUT_PUBLICATION_REJECTED":
                    fail("publication rejection reason")
            else:
                fail("create-only publication overwrite accepted")
            if output.read_bytes() != first_bytes:
                fail("publication failure changed immutable first record")
            if sorted(path.relative_to(temporary_root).as_posix() for path in temporary_root.rglob("*")) != [
                "base",
                "base/result.json",
            ]:
                fail("publication failure created fallback or substitution")
            checks += 1
            try:
                evaluator.prepare_output(output, temporary_lane)
            except evaluator.PublicationError as exc:
                if exc.reason_code != "OUTPUT_PUBLICATION_REJECTED":
                    fail("publication preflight rejection reason")
            else:
                fail("publication preflight collision accepted")
            checks += 1
        finally:
            evaluator.ROOT = old_root

    checks += fault_injection_persistence_checks(evaluator, lane, success)

    class SyntheticStderr:
        def __init__(self) -> None:
            self.buffer = io.BytesIO()

        def flush(self) -> None:
            return None

        def write(self, value: str) -> int:
            return len(value)

    old_main = evaluator._main
    old_stderr = evaluator.sys.stderr
    try:
        for error, expected_exit in (
            (evaluator.EvaluationError("CROSS_LANE_REJECTED", "synthetic pre-authority rejection"), 2),
            (evaluator.PublicationError("synthetic publication rejection"), 3),
        ):
            synthetic_stderr = SyntheticStderr()

            def reject_cli(error: Exception = error) -> int:
                raise error

            evaluator._main = reject_cli
            evaluator.sys.stderr = synthetic_stderr
            if evaluator.main() != expected_exit:
                fail("CLI rejection exit code")
            rejection = json.loads(synthetic_stderr.buffer.getvalue().decode("ascii"))
            if rejection != {
                "message": str(error),
                "reason_code": error.reason_code,
                "result_published": False,
                "schema_id": "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_CLI_REJECTION_V1",
            }:
                fail("CLI rejection record")
            checks += 1
    finally:
        evaluator._main = old_main
        evaluator.sys.stderr = old_stderr
    return checks


def expect_reject(function: Any, *args: Any) -> None:
    try:
        function(*args)
    except (AssertionError, ValueError, TypeError, RuntimeError):
        return
    fail("semantic mutation unexpectedly accepted")


def reseal_result(evaluator: Any, result: dict[str, Any]) -> None:
    payload = dict(result)
    payload.pop("result_sha256")
    result["result_sha256"] = evaluator.object_sha256(payload)


def mutation_closure(
    package: dict[str, Any], contract: dict[str, Any], schema: dict[str, Any], proof: dict[str, Any], evaluator: Any
) -> tuple[int, int]:
    checks = 0
    for field in tuple(package):
        mutation = copy.deepcopy(package)
        mutation.pop(field)
        expect_reject(validate_package, mutation)
        checks += 1
    mutation = copy.deepcopy(package)
    mutation["extra"] = 0
    expect_reject(validate_package, mutation)
    checks += 1
    for field in package["claim_boundary"]:
        mutation = copy.deepcopy(package)
        mutation["claim_boundary"][field] = True
        expect_reject(validate_package, mutation)
        checks += 1

    contract_mutations = []
    mutation = copy.deepcopy(contract)
    mutation["accepted_g16"]["alternative_id"] = "QK_GBFP8_G8_E16_HEAD64_V1"
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["deterministic_evaluation"]["record_selectors"]["roles"]["bf16_oracle_scores"]["tensor_name"] = "bf16.qk_centered_scores"
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["deterministic_evaluation"]["g16_realization"]["query_head_to_kv_head"] = "kv_head=query_head"
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["thresholds"]["top_key_matching_fraction_minimum"] = {"denominator": 10, "numerator": 9}
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["thresholds"]["unique_oracle_positive_margin_preserved_fraction_minimum"] = {"denominator": 10, "numerator": 9}
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["metrics"]["score_error"]["gate_mode"] = "HARD"
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["authority_interface"]["cardinality"] = 2
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["claim_boundary"]["sealed_tensor_payload_opened_or_deserialized"] = True
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["input_set"]["lanes"][0]["authoritative_tensors"]["realized_query_source"]["sha256"] = "0" * 64
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["deterministic_evaluation"]["oracle"].pop("nonfinite_scan_order")
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["metrics"]["rank_margin"].pop("empty_set")
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["metrics"]["top_key"].pop("empty_set")
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["malformed_input_policy"]["repair_policy"] = "Retry and replay are permitted."
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["output"]["terminal_semantics"]["retry_replay_resume_repair_permitted"] = True
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["output"]["cli_rejection_semantics"] = "Publication failures publish a fallback result."
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    base_invocation = mutation["invocations"][0]
    base_invocation["argv"][13] = "evidence/diagnostics/alternate/base/result.json"
    base_invocation["invocation_sha256"] = invocation_digest(base_invocation)
    contract_mutations.append(mutation)
    for mutation in contract_mutations:
        expect_reject(validate_contract, mutation)
        checks += 1

    mutation = copy.deepcopy(schema)
    mutation["additionalProperties"] = True
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(schema)
    mutation["$defs"]["threshold_evaluation"]["properties"]["score_error"] = {}
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(schema)
    mutation["allOf"] = []
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(schema)
    mutation["allOf"].pop(2)
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(schema)
    mutation["$defs"]["metrics"]["properties"]["rank_margin"]["required"].remove("violation_count")
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(schema)
    rank_margin = mutation["$defs"]["metrics"]["properties"]["rank_margin"]
    rank_margin["properties"].pop("minimum_realized_margin_q12_20_lsb")
    rank_margin["required"].remove("minimum_realized_margin_q12_20_lsb")
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(schema)
    mutation["allOf"][2]["then"]["oneOf"][0]["properties"]["terminal"]["required"].remove("reason_code")
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(schema)
    mutation["allOf"][2]["then"]["oneOf"][1]["properties"]["threshold_evaluation"]["allOf"][1]["properties"]["all_hard_thresholds_pass"]["const"] = True
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(schema)
    mutation["properties"]["terminal"]["properties"]["reason_code"]["enum"].append("OUTPUT_PUBLICATION_REJECTED")
    expect_reject(validate_schema, mutation)
    checks += 1
    mutation = copy.deepcopy(proof)
    mutation["event_counts"]["sealed_tensor_payload_opens"] = 1
    expect_reject(validate_proof, mutation)
    checks += 1

    lane = contract["input_set"]["lanes"][0]
    invocation = contract["invocations"][0]
    metrics = evaluator.evaluate_selected_records(synthetic_records(), import_module(ROOT / contract["accepted_g16"]["pure_reference"]["path"], "mutation_g16"))
    thresholds = evaluator.evaluate_thresholds(metrics, contract["thresholds"])
    result = evaluator.result_object(
        lane,
        contract,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        invocation["invocation_sha256"],
        metrics,
        thresholds,
        "HARD_THRESHOLDS_PASSED",
    )
    evaluator.validate_result(result)
    failed_terminal_checks = 0

    pre_metric_failure = evaluator.result_object(
        lane,
        contract,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        invocation["invocation_sha256"],
        None,
        None,
        "INPUT_BINDING_REJECTED",
    )
    evaluator.validate_result(pre_metric_failure)
    failed_terminal_checks += 1

    failed_metrics = copy.deepcopy(metrics)
    failed_metrics["top_key"].update(
        matching_fraction={"denominator": 574, "numerator": 573},
        matching_row_count=573,
        mismatch_count=1,
    )
    failed_thresholds = evaluator.evaluate_thresholds(failed_metrics, contract["thresholds"])
    if failed_thresholds["all_hard_thresholds_pass"] is not False:
        fail("synthetic threshold failure did not fail")
    threshold_failure = evaluator.result_object(
        lane,
        contract,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        invocation["invocation_sha256"],
        failed_metrics,
        failed_thresholds,
        "HARD_THRESHOLD_FAILED",
    )
    evaluator.validate_result(threshold_failure)
    failed_terminal_checks += 1

    for source, mutate in (
        (pre_metric_failure, lambda value: value["terminal"].update(invocation_count_performed=0)),
        (pre_metric_failure, lambda value: value["terminal"].update(metrics_published=True)),
        (pre_metric_failure, lambda value: value["terminal"].update(reason_code="HARD_THRESHOLD_FAILED")),
        (threshold_failure, lambda value: value["terminal"].update(reason_code="INPUT_BINDING_REJECTED")),
        (threshold_failure, lambda value: value["terminal"].update(thresholds_evaluated=False)),
        (threshold_failure, lambda value: value["terminal"].update(status="SUCCEEDED_TERMINAL")),
    ):
        mutation = copy.deepcopy(source)
        mutate(mutation)
        reseal_result(evaluator, mutation)
        expect_reject(evaluator.validate_result, mutation)
        failed_terminal_checks += 1

    for mutate in (
        lambda value: value.pop("authority_sha256"),
        lambda value: value.update(extra=0),
        lambda value: value["metrics"]["top_key"].update(mismatch_count=1),
        lambda value: value["terminal"].update(status="NO_EXECUTION_TERMINAL"),
        lambda value: value.update(result_sha256="0" * 64),
    ):
        mutation = copy.deepcopy(result)
        mutate(mutation)
        expect_reject(evaluator.validate_result, mutation)
        checks += 1
    return checks, failed_terminal_checks


def verify_non_execution(proof: dict[str, Any]) -> int:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        fail("result namespace exists")
    if AUTHORITY_ROOT.exists() or AUTHORITY_ROOT.is_symlink():
        fail("authority namespace exists")
    source = EVALUATOR_PATH.read_text(encoding="utf-8")
    forbidden_source = (
        "import torch",
        "from torch",
        "import transformers",
        "from transformers",
        "qk_bfp8_e16_head64_v1_c02_score_path_evaluator_v2",
    )
    if any(token in source for token in forbidden_source):
        fail("evaluator contains forbidden model/c02 dependency")
    if proof["event_counts"]["sealed_tensor_payload_opens"] != 0:
        fail("sealed payload proof")
    return len(proof["checked_absent_paths"]) + len(forbidden_source) + 1


def main() -> int:
    package = validate_package(load_json(PACKAGE_PATH))
    contract = validate_contract(load_json(CONTRACT_PATH))
    schema = validate_schema(load_json(SCHEMA_PATH))
    proof = validate_proof(load_json(PROOF_PATH))
    verify_artifact_bindings(package)
    accepted_checks = verify_accepted_static_bytes(contract)
    evaluator = import_module(EVALUATOR_PATH, "g16_diagnostic_evaluator")
    g16 = import_module(ROOT / contract["accepted_g16"]["pure_reference"]["path"], "accepted_g16_for_diagnostic")
    bf16_checks, arithmetic_rows = exact_arithmetic_cross_checks(evaluator, g16)
    (
        synthetic_checks,
        nonfinite_checks,
        masked_oracle_nonfinite_checks,
        preimport_nonfinite_checks,
    ) = synthetic_evaluation_checks(evaluator, g16, contract)
    argv_path_checks = synthetic_argv_path_checks(evaluator)
    control_flow_checks = synthetic_control_flow_checks(evaluator, g16, contract)
    mutation_checks, failed_terminal_checks = mutation_closure(package, contract, schema, proof, evaluator)
    non_execution_checks = verify_non_execution(proof)
    print("QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1=PASS")
    print("AUTHORITY=NO_EXECUTION_AUTHORITY")
    print("ACCEPTED_ALTERNATIVE=QK_GBFP8_G16_E16_HEAD64_V1")
    print(f"EXHAUSTIVE_FINITE_BF16_CHECKS={bf16_checks}")
    print(f"INDEPENDENT_G16_ARITHMETIC_ROWS={arithmetic_rows}")
    print(f"SYNTHETIC_EVALUATION_CHECKS={synthetic_checks}")
    print(f"COMPLETE_RECORD_NONFINITE_CHECKS={nonfinite_checks}")
    print(f"MASKED_ORACLE_NONFINITE_CHECKS={masked_oracle_nonfinite_checks}")
    print(f"PREIMPORT_NONFINITE_CHECKS={preimport_nonfinite_checks}")
    print(f"ARGV_PATH_NEGATIVE_CHECKS={argv_path_checks}")
    print(f"SYNTHETIC_CONTROL_FLOW_CHECKS={control_flow_checks}")
    print(f"FAILED_TERMINAL_CHECKS={failed_terminal_checks}")
    print(f"SEMANTIC_MUTATION_CHECKS={mutation_checks}")
    print(f"ACCEPTED_STATIC_BYTE_CHECKS={accepted_checks}")
    print(f"NO_EXECUTION_CHECKS={non_execution_checks}")
    print("SEALED_TENSOR_PAYLOAD_OPENS=0")
    print("MODEL_LOADS=0")
    print("C02_INVOCATIONS=0")
    print("NEW_MODEL_SCORES_COMPUTED=0")
    print("RESULT_OR_AUTHORITY_NAMESPACES_CREATED=0")
    print("RTL_OR_STAGE_EDITS=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
