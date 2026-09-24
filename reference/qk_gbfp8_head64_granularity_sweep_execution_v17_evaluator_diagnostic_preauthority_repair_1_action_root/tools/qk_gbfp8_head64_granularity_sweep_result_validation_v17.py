#!/usr/bin/env python3
"""Exact frozen result schema and record validators shared by production and fixture."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from typing import Any, Callable, NamedTuple

import jsonschema


FROZEN_RESULT_SCHEMA_BYTES = b'{"$defs":{"artifact_binding":{"additionalProperties":false,"properties":{"sealed_set_id":{"const":"w4a8-c02-attention-substage-trace-v2"},"tensor_bundle_sha256":{"$ref":"#/$defs/sha256"},"tensor_record_count":{"const":25}},"required":["sealed_set_id","tensor_bundle_sha256","tensor_record_count"],"type":"object"},"candidate_result":{"additionalProperties":false,"properties":{"bytes_per_head":{"minimum":1,"type":"integer"},"exponent_bytes_per_head":{"minimum":1,"type":"integer"},"group_count":{"minimum":1,"type":"integer"},"group_size":{"minimum":1,"type":"integer"},"label":{"enum":["G8","G4","G2","G1"]},"mantissa_bytes_per_head":{"const":64},"metrics":{"$ref":"#/$defs/metrics"},"threshold_evaluation":{"$ref":"#/$defs/threshold_evaluation"}},"required":["bytes_per_head","exponent_bytes_per_head","group_count","group_size","label","mantissa_bytes_per_head","metrics","threshold_evaluation"],"type":"object"},"count_gate":{"additionalProperties":false,"oneOf":[{"properties":{"actual":{"const":0},"pass":{"const":true}}},{"properties":{"actual":{"minimum":1},"pass":{"const":false}}}],"properties":{"actual":{"$ref":"#/$defs/nonnegative_integer"},"limit":{"const":0},"pass":{"type":"boolean"}},"required":["actual","limit","pass"],"type":"object"},"fraction":{"additionalProperties":false,"properties":{"denominator":{"minimum":1,"type":"integer"},"numerator":{"$ref":"#/$defs/nonnegative_integer"}},"required":["denominator","numerator"],"type":"object"},"fraction_gate":{"additionalProperties":false,"properties":{"actual":{"$ref":"#/$defs/fraction"},"limit":{"const":{"denominator":1,"numerator":1}},"pass":{"type":"boolean"}},"required":["actual","limit","pass"],"type":"object"},"invalid_accounting":{"additionalProperties":false,"properties":{"cross_lane_record_count":{"$ref":"#/$defs/nonnegative_integer"},"invalid_or_non_finite_value_count":{"$ref":"#/$defs/nonnegative_integer"},"normalization_rejection_count":{"$ref":"#/$defs/nonnegative_integer"},"positive_centered_realized_score_count":{"$ref":"#/$defs/nonnegative_integer"},"saturation_event_count":{"$ref":"#/$defs/nonnegative_integer"}},"required":["cross_lane_record_count","invalid_or_non_finite_value_count","normalization_rejection_count","positive_centered_realized_score_count","saturation_event_count"],"type":"object"},"metrics":{"additionalProperties":false,"properties":{"invalid_accounting":{"$ref":"#/$defs/invalid_accounting"},"rank_margin":{"$ref":"#/$defs/rank_margin"},"score_error":{"$ref":"#/$defs/score_error"},"top_key":{"$ref":"#/$defs/top_key"}},"required":["invalid_accounting","rank_margin","score_error","top_key"],"type":"object"},"nonnegative_integer":{"minimum":0,"type":"integer"},"passing_candidates":{"enum":[[],["G8"],["G4"],["G2"],["G1"],["G8","G4"],["G8","G2"],["G8","G1"],["G4","G2"],["G4","G1"],["G2","G1"],["G8","G4","G2"],["G8","G4","G1"],["G8","G2","G1"],["G4","G2","G1"],["G8","G4","G2","G1"]],"type":"array"},"rank_fraction":{"additionalProperties":false,"properties":{"denominator":{"const":346},"numerator":{"maximum":346,"minimum":0,"type":"integer"}},"required":["denominator","numerator"],"type":"object"},"rank_fraction_gate":{"additionalProperties":false,"oneOf":[{"properties":{"actual":{"properties":{"numerator":{"const":346}}},"pass":{"const":true}}},{"properties":{"actual":{"properties":{"numerator":{"maximum":345}}},"pass":{"const":false}}}],"properties":{"actual":{"$ref":"#/$defs/rank_fraction"},"limit":{"const":{"denominator":1,"numerator":1}},"pass":{"type":"boolean"}},"required":["actual","limit","pass"],"type":"object"},"rank_margin":{"additionalProperties":false,"allOf":[{"else":{"properties":{"minimum_realized_margin_q12_20_lsb":{"maximum":0},"preserved_positive_margin_fraction":{"properties":{"numerator":{"maximum":345}}},"preserved_positive_margin_row_count":{"maximum":345}}},"if":{"properties":{"violation_count":{"const":0}}},"then":{"properties":{"minimum_realized_margin_q12_20_lsb":{"minimum":1},"preserved_positive_margin_fraction":{"const":{"denominator":346,"numerator":346}},"preserved_positive_margin_row_count":{"const":346}}}}],"properties":{"minimum_realized_margin_q12_20_lsb":{"type":"integer"},"preserved_positive_margin_fraction":{"$ref":"#/$defs/rank_fraction"},"preserved_positive_margin_row_count":{"maximum":346,"minimum":0,"type":"integer"},"unique_oracle_top_row_count":{"const":346},"violation_count":{"maximum":346,"minimum":0,"type":"integer"}},"required":["minimum_realized_margin_q12_20_lsb","preserved_positive_margin_fraction","preserved_positive_margin_row_count","unique_oracle_top_row_count","violation_count"],"type":"object"},"score_error":{"additionalProperties":false,"properties":{"maximum_absolute_error_q12_20_lsb":{"$ref":"#/$defs/nonnegative_integer"},"sum_absolute_error_q12_20_lsb":{"$ref":"#/$defs/nonnegative_integer"},"sum_signed_error_q12_20_lsb":{"type":"integer"},"sum_squared_error_q40_40_lsb2":{"$ref":"#/$defs/nonnegative_integer"},"valid_value_count":{"const":12054}},"required":["maximum_absolute_error_q12_20_lsb","sum_absolute_error_q12_20_lsb","sum_signed_error_q12_20_lsb","sum_squared_error_q40_40_lsb2","valid_value_count"],"type":"object"},"selection":{"additionalProperties":false,"properties":{"passing_candidates":{"$ref":"#/$defs/passing_candidates"},"policy":{"const":"FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER"},"selected_candidate":{"enum":["G8","G4","G2","G1",null]}},"required":["passing_candidates","policy","selected_candidate"],"type":"object"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"terminal":{"additionalProperties":false,"properties":{"first_record_immutable":{"const":true},"invocation_count_performed":{"const":1},"metrics_published":{"const":true},"reason_code":{"enum":["HARD_THRESHOLD_FAILED","HARD_THRESHOLDS_PASSED"]},"retry_replay_resume_repair_permitted":{"const":false},"status":{"enum":["FAILED_TERMINAL","SUCCEEDED_TERMINAL"]},"tensor_open_count":{"const":1},"thresholds_evaluated":{"const":true}},"required":["first_record_immutable","invocation_count_performed","metrics_published","reason_code","retry_replay_resume_repair_permitted","status","tensor_open_count","thresholds_evaluated"],"type":"object"},"threshold_evaluation":{"additionalProperties":false,"allOf":[{"else":{"properties":{"all_hard_gates_pass":{"const":false}}},"if":{"properties":{"cross_lane_record_count_maximum":{"properties":{"pass":{"const":true}}},"invalid_or_non_finite_value_count_maximum":{"properties":{"pass":{"const":true}}},"normalization_rejection_count_maximum":{"properties":{"pass":{"const":true}}},"positive_centered_realized_score_count_maximum":{"properties":{"pass":{"const":true}}},"rank_margin_violation_count_maximum":{"properties":{"pass":{"const":true}}},"saturation_event_count_maximum":{"properties":{"pass":{"const":true}}},"top_key_matching_fraction_minimum":{"properties":{"pass":{"const":true}}},"top_key_mismatch_count_maximum":{"properties":{"pass":{"const":true}}},"unique_oracle_positive_margin_preserved_fraction_minimum":{"properties":{"pass":{"const":true}}}},"required":["cross_lane_record_count_maximum","invalid_or_non_finite_value_count_maximum","normalization_rejection_count_maximum","positive_centered_realized_score_count_maximum","rank_margin_violation_count_maximum","saturation_event_count_maximum","top_key_matching_fraction_minimum","top_key_mismatch_count_maximum","unique_oracle_positive_margin_preserved_fraction_minimum"]},"then":{"properties":{"all_hard_gates_pass":{"const":true}}}}],"properties":{"all_hard_gates_pass":{"type":"boolean"},"cross_lane_record_count_maximum":{"$ref":"#/$defs/count_gate"},"invalid_or_non_finite_value_count_maximum":{"$ref":"#/$defs/count_gate"},"normalization_rejection_count_maximum":{"$ref":"#/$defs/count_gate"},"positive_centered_realized_score_count_maximum":{"$ref":"#/$defs/count_gate"},"rank_margin_violation_count_maximum":{"$ref":"#/$defs/count_gate"},"saturation_event_count_maximum":{"$ref":"#/$defs/count_gate"},"top_key_matching_fraction_minimum":{"$ref":"#/$defs/top_fraction_gate"},"top_key_mismatch_count_maximum":{"$ref":"#/$defs/count_gate"},"unique_oracle_positive_margin_preserved_fraction_minimum":{"$ref":"#/$defs/rank_fraction_gate"}},"required":["all_hard_gates_pass","cross_lane_record_count_maximum","invalid_or_non_finite_value_count_maximum","normalization_rejection_count_maximum","positive_centered_realized_score_count_maximum","rank_margin_violation_count_maximum","saturation_event_count_maximum","top_key_matching_fraction_minimum","top_key_mismatch_count_maximum","unique_oracle_positive_margin_preserved_fraction_minimum"],"type":"object"},"top_fraction":{"additionalProperties":false,"properties":{"denominator":{"const":574},"numerator":{"maximum":574,"minimum":0,"type":"integer"}},"required":["denominator","numerator"],"type":"object"},"top_fraction_gate":{"additionalProperties":false,"oneOf":[{"properties":{"actual":{"properties":{"numerator":{"const":574}}},"pass":{"const":true}}},{"properties":{"actual":{"properties":{"numerator":{"maximum":573}}},"pass":{"const":false}}}],"properties":{"actual":{"$ref":"#/$defs/top_fraction"},"limit":{"const":{"denominator":1,"numerator":1}},"pass":{"type":"boolean"}},"required":["actual","limit","pass"],"type":"object"},"top_key":{"additionalProperties":false,"allOf":[{"else":{"properties":{"matching_fraction":{"properties":{"numerator":{"maximum":573}}},"matching_row_count":{"maximum":573}}},"if":{"properties":{"mismatch_count":{"const":0}}},"then":{"properties":{"matching_fraction":{"const":{"denominator":574,"numerator":574}},"matching_row_count":{"const":574}}}}],"properties":{"matching_fraction":{"$ref":"#/$defs/top_fraction"},"matching_row_count":{"maximum":574,"minimum":0,"type":"integer"},"mismatch_count":{"maximum":574,"minimum":0,"type":"integer"},"row_count":{"const":574}},"required":["matching_fraction","matching_row_count","mismatch_count","row_count"],"type":"object"}},"$id":"QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1_SCHEMA","$schema":"https://json-schema.org/draft/2020-12/schema","additionalProperties":false,"allOf":[{"else":{"properties":{"selection":{"properties":{"passing_candidates":{"not":{"contains":{"const":"G8"}}}}}}},"if":{"properties":{"candidate_results":{"prefixItems":[{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":true}}}}}]}}},"then":{"properties":{"selection":{"properties":{"passing_candidates":{"contains":{"const":"G8"}}}}}}},{"else":{"properties":{"selection":{"properties":{"passing_candidates":{"not":{"contains":{"const":"G4"}}}}}}},"if":{"properties":{"candidate_results":{"prefixItems":[{},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":true}}}}}]}}},"then":{"properties":{"selection":{"properties":{"passing_candidates":{"contains":{"const":"G4"}}}}}}},{"else":{"properties":{"selection":{"properties":{"passing_candidates":{"not":{"contains":{"const":"G2"}}}}}}},"if":{"properties":{"candidate_results":{"prefixItems":[{},{},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":true}}}}}]}}},"then":{"properties":{"selection":{"properties":{"passing_candidates":{"contains":{"const":"G2"}}}}}}},{"else":{"properties":{"selection":{"properties":{"passing_candidates":{"not":{"contains":{"const":"G1"}}}}}}},"if":{"properties":{"candidate_results":{"prefixItems":[{},{},{},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":true}}}}}]}}},"then":{"properties":{"selection":{"properties":{"passing_candidates":{"contains":{"const":"G1"}}}}}}}],"oneOf":[{"properties":{"candidate_results":{"prefixItems":[{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":true}}}}}]},"selected_candidate":{"const":"G8"},"selection":{"properties":{"selected_candidate":{"const":"G8"}}},"terminal":{"properties":{"reason_code":{"const":"HARD_THRESHOLDS_PASSED"},"status":{"const":"SUCCEEDED_TERMINAL"}}}}},{"properties":{"candidate_results":{"prefixItems":[{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":true}}}}}]},"selected_candidate":{"const":"G4"},"selection":{"properties":{"selected_candidate":{"const":"G4"}}},"terminal":{"properties":{"reason_code":{"const":"HARD_THRESHOLDS_PASSED"},"status":{"const":"SUCCEEDED_TERMINAL"}}}}},{"properties":{"candidate_results":{"prefixItems":[{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":true}}}}}]},"selected_candidate":{"const":"G2"},"selection":{"properties":{"selected_candidate":{"const":"G2"}}},"terminal":{"properties":{"reason_code":{"const":"HARD_THRESHOLDS_PASSED"},"status":{"const":"SUCCEEDED_TERMINAL"}}}}},{"properties":{"candidate_results":{"prefixItems":[{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":true}}}}}]},"selected_candidate":{"const":"G1"},"selection":{"properties":{"selected_candidate":{"const":"G1"}}},"terminal":{"properties":{"reason_code":{"const":"HARD_THRESHOLDS_PASSED"},"status":{"const":"SUCCEEDED_TERMINAL"}}}}},{"properties":{"candidate_results":{"prefixItems":[{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}},{"properties":{"threshold_evaluation":{"properties":{"all_hard_gates_pass":{"const":false}}}}}]},"selected_candidate":{"const":null},"selection":{"properties":{"selected_candidate":{"const":null}}},"terminal":{"properties":{"reason_code":{"const":"HARD_THRESHOLD_FAILED"},"status":{"const":"FAILED_TERMINAL"}}}}}],"properties":{"authority_sha256":{"$ref":"#/$defs/sha256"},"candidate_results":{"items":false,"maxItems":4,"minItems":4,"prefixItems":[{"allOf":[{"$ref":"#/$defs/candidate_result"},{"properties":{"bytes_per_head":{"const":80},"exponent_bytes_per_head":{"const":16},"group_count":{"const":8},"group_size":{"const":8},"label":{"const":"G8"}}}]},{"allOf":[{"$ref":"#/$defs/candidate_result"},{"properties":{"bytes_per_head":{"const":96},"exponent_bytes_per_head":{"const":32},"group_count":{"const":16},"group_size":{"const":4},"label":{"const":"G4"}}}]},{"allOf":[{"$ref":"#/$defs/candidate_result"},{"properties":{"bytes_per_head":{"const":128},"exponent_bytes_per_head":{"const":64},"group_count":{"const":32},"group_size":{"const":2},"label":{"const":"G2"}}}]},{"allOf":[{"$ref":"#/$defs/candidate_result"},{"properties":{"bytes_per_head":{"const":192},"exponent_bytes_per_head":{"const":128},"group_count":{"const":64},"group_size":{"const":1},"label":{"const":"G1"}}}]}],"type":"array"},"consumed_ledger_sha256":{"$ref":"#/$defs/sha256"},"evaluator_sha256":{"$ref":"#/$defs/sha256"},"fresh_l2_acceptance_sha256":{"$ref":"#/$defs/sha256"},"generation_id":{"const":"qk-gbfp8-head64-granularity-sweep-v8-base"},"input_bindings":{"$ref":"#/$defs/artifact_binding"},"invocation_sha256":{"$ref":"#/$defs/sha256"},"irreversible_action_id":{"pattern":"^[A-Za-z0-9][A-Za-z0-9._:-]{15,255}$","type":"string"},"lane_label":{"const":"Base"},"model_identity_sha256":{"$ref":"#/$defs/sha256"},"namespace_label":{"const":"base"},"package_id":{"const":"QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE"},"package_sha256":{"$ref":"#/$defs/sha256"},"result_sha256":{"$ref":"#/$defs/sha256"},"schema_id":{"const":"QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1"},"selected_candidate":{"enum":["G8","G4","G2","G1",null]},"selection":{"$ref":"#/$defs/selection"},"terminal":{"$ref":"#/$defs/terminal"}},"required":["authority_sha256","candidate_results","consumed_ledger_sha256","evaluator_sha256","fresh_l2_acceptance_sha256","generation_id","input_bindings","invocation_sha256","irreversible_action_id","lane_label","model_identity_sha256","namespace_label","package_id","package_sha256","result_sha256","schema_id","selected_candidate","selection","terminal"],"type":"object"}\n'
RESULT_SCHEMA_SHA256 = "07f818190e7f97032a6bc3724914f00f36070f429f1cd549c67e4e7b026ae720"
RESULT_KEYS = frozenset({
    "authority_sha256", "candidate_results", "consumed_ledger_sha256", "evaluator_sha256",
    "fresh_l2_acceptance_sha256", "generation_id", "input_bindings", "invocation_sha256",
    "irreversible_action_id", "lane_label", "model_identity_sha256", "namespace_label", "package_id",
    "package_sha256", "result_sha256", "schema_id", "selected_candidate", "selection", "terminal",
})
CANDIDATE_RESULT_KEYS = frozenset({
    "bytes_per_head", "exponent_bytes_per_head", "group_count", "group_size", "label",
    "mantissa_bytes_per_head", "metrics", "threshold_evaluation",
})
SELECTION_KEYS = frozenset({"passing_candidates", "policy", "selected_candidate"})
TERMINAL_KEYS = frozenset({
    "first_record_immutable", "invocation_count_performed", "metrics_published", "reason_code",
    "retry_replay_resume_repair_permitted", "status", "tensor_open_count", "thresholds_evaluated",
})
CANDIDATE_SPECS = (
    ("G8", 8, 8, 80, 16),
    ("G4", 4, 16, 96, 32),
    ("G2", 2, 32, 128, 64),
    ("G1", 1, 64, 192, 128),
)


class ResultValidationError(RuntimeError):
    pass


class ResultBindings(NamedTuple):
    acceptance_sha256: str
    authority_sha256: str
    evaluator_sha256: str
    invocation_sha256: str
    ledger_sha256: str
    model_identity_sha256: str
    package_sha256: str
    tensor_bundle_sha256: str


class ResultValidationBindings(NamedTuple):
    validate_result_record: Callable[[dict[str, Any], bytes], None]
    validate_result_schema: Callable[[dict[str, Any]], None]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultValidationError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_result_schema() -> dict[str, Any]:
    require(sha256_bytes(FROZEN_RESULT_SCHEMA_BYTES) == RESULT_SCHEMA_SHA256, "embedded result schema checksum")
    value = json.loads(FROZEN_RESULT_SCHEMA_BYTES.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == FROZEN_RESULT_SCHEMA_BYTES, "embedded canonical result schema")
    return value


def construct_result_schema_validator(result_schema: dict[str, Any]) -> Any:
    require(compact_bytes(result_schema) == FROZEN_RESULT_SCHEMA_BYTES, "production result schema bytes")
    validator_class = jsonschema.Draft202012Validator
    validator_class.check_schema(result_schema)
    return validator_class(result_schema)


def validate_result_schema_exact(result_validator: Any, result: dict[str, Any]) -> None:
    result_validator.validate(result)


def _valid_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_result_record_exact(
    result: dict[str, Any],
    result_bytes: bytes,
    *,
    bindings: ResultBindings,
    action_id: str,
) -> None:
    require(compact_bytes(result) == result_bytes, "result canonical bytes")
    require(set(result) == RESULT_KEYS, "result exact keys")
    require(result["package_id"] == "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE", "result package id")
    require(result["schema_id"] == "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1", "result schema id")
    require(result["generation_id"] == "qk-gbfp8-head64-granularity-sweep-v8-base", "result generation")
    require(result["lane_label"] == "Base" and result["namespace_label"] == "base", "result Base lane")
    require(result["package_sha256"] == bindings.package_sha256, "result package checksum")
    require(result["evaluator_sha256"] == bindings.evaluator_sha256, "result evaluator checksum")
    require(result["authority_sha256"] == bindings.authority_sha256, "result authority checksum")
    require(result["consumed_ledger_sha256"] == bindings.ledger_sha256, "result ledger checksum")
    require(result["fresh_l2_acceptance_sha256"] == bindings.acceptance_sha256, "result acceptance checksum")
    require(result["invocation_sha256"] == bindings.invocation_sha256, "result invocation checksum")
    require(result["irreversible_action_id"] == action_id, "result action identity")
    require(result["model_identity_sha256"] == bindings.model_identity_sha256, "result model checksum")
    for key in (
        "authority_sha256", "consumed_ledger_sha256", "evaluator_sha256", "fresh_l2_acceptance_sha256",
        "invocation_sha256", "model_identity_sha256", "package_sha256", "result_sha256",
    ):
        require(_valid_sha256(result[key]), f"result checksum syntax: {key}")
    require(result["input_bindings"] == {
        "sealed_set_id": "w4a8-c02-attention-substage-trace-v2",
        "tensor_bundle_sha256": bindings.tensor_bundle_sha256,
        "tensor_record_count": 25,
    }, "result input bindings")
    candidates = result["candidate_results"]
    require(type(candidates) is list and len(candidates) == len(CANDIDATE_SPECS), "result candidate cardinality")
    passing: list[str] = []
    for candidate, (label, group_size, group_count, bytes_per_head, exponent_bytes_per_head) in zip(candidates, CANDIDATE_SPECS):
        require(type(candidate) is dict and set(candidate) == CANDIDATE_RESULT_KEYS, f"result candidate keys: {label}")
        require(candidate["label"] == label, f"result candidate label: {label}")
        require(candidate["group_size"] == group_size and candidate["group_count"] == group_count, f"result candidate groups: {label}")
        require(candidate["bytes_per_head"] == bytes_per_head, f"result candidate bytes: {label}")
        require(candidate["exponent_bytes_per_head"] == exponent_bytes_per_head, f"result candidate exponent bytes: {label}")
        require(candidate["mantissa_bytes_per_head"] == 64, f"result candidate mantissa bytes: {label}")
        if candidate["threshold_evaluation"]["all_hard_gates_pass"]:
            passing.append(label)
    selected = passing[0] if passing else None
    require(result["selected_candidate"] == selected, "result selected candidate")
    selection = result["selection"]
    require(type(selection) is dict and set(selection) == SELECTION_KEYS, "result selection keys")
    require(selection == {
        "passing_candidates": passing,
        "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER",
        "selected_candidate": selected,
    }, "result selection")
    terminal = result["terminal"]
    require(type(terminal) is dict and set(terminal) == TERMINAL_KEYS, "result terminal keys")
    require(terminal["first_record_immutable"] is True, "result terminal immutability")
    require(terminal["invocation_count_performed"] == 1 and terminal["tensor_open_count"] == 1, "result terminal counters")
    require(terminal["metrics_published"] is True and terminal["thresholds_evaluated"] is True, "result terminal publication")
    require(terminal["retry_replay_resume_repair_permitted"] is False, "result terminal replay")
    require(terminal["status"] == ("SUCCEEDED_TERMINAL" if selected is not None else "FAILED_TERMINAL"), "result terminal status")
    require(terminal["reason_code"] == ("HARD_THRESHOLDS_PASSED" if selected is not None else "HARD_THRESHOLD_FAILED"), "result terminal reason")
    payload = dict(result)
    observed = payload.pop("result_sha256")
    require(sha256_bytes(compact_bytes(payload)) == observed, "result self-checksum")


def bind_result_validators(result_validator: Any, bindings: ResultBindings, action_id: str) -> ResultValidationBindings:
    return ResultValidationBindings(
        validate_result_record=partial(validate_result_record_exact, bindings=bindings, action_id=action_id),
        validate_result_schema=partial(validate_result_schema_exact, result_validator),
    )
