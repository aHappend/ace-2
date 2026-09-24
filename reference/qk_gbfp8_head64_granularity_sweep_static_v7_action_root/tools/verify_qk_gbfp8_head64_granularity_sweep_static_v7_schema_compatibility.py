#!/usr/bin/env python3
"""Decisive static-only V7 package/result compatibility verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import struct
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

from jsonschema import Draft202012Validator


ACTION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ACTION_ROOT.parents[1]
PACKAGE_PATH = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_PACKAGE.json"
SCHEMA_PATH = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_RESULT_SCHEMA.json"
CONTROLLER_PATH = ACTION_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v7.py"
EVALUATOR_PATH = ACTION_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v7.py"
VERIFIER_PATH = Path(__file__).resolve()
RUNTIME_PACKAGE_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_RUNTIME_PACKAGE.json"
MANIFEST_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_MANIFEST.json"
V6_ROOT = REPOSITORY_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root"
V6_HANDOFF = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/c9e0569b083e")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == expected, f"{context} exact keys")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text("utf-8"), object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(VerificationError(f"nonfinite JSON: {value}")))
    require(type(value) is dict, f"JSON object: {path}")
    return value


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, f"module spec: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def reseal(evaluator: ModuleType, result: dict[str, Any]) -> dict[str, Any]:
    return evaluator.seal_result(result)


def expect_reject(callable_value: Any, *args: Any, **kwargs: Any) -> None:
    try:
        callable_value(*args, **kwargs)
    except Exception:
        return
    raise VerificationError("mutation unexpectedly accepted")


def schema_errors(validator: Draft202012Validator, value: Any) -> list[Any]:
    return sorted(validator.iter_errors(value), key=lambda error: tuple(error.absolute_path))


def require_schema_accepts(validator: Draft202012Validator, value: Any, context: str) -> None:
    errors = schema_errors(validator, value)
    require(not errors, f"schema rejected {context}: {errors[0].message if errors else ''}")


def require_schema_rejects(validator: Draft202012Validator, value: Any, context: str) -> None:
    require(bool(schema_errors(validator, value)), f"schema accepted invalid {context}")


def verify_schema_contract(schema: dict[str, Any], controller: ModuleType) -> None:
    Draft202012Validator.check_schema(schema)
    require(schema["$id"] == controller.RESULT_SCHEMA_DOCUMENT_ID, "schema document id")
    require(schema["properties"]["package_id"]["const"] == controller.PACKAGE_ID, "schema package_id")
    require(schema["properties"]["schema_id"]["const"] == controller.RESULT_SCHEMA_ID, "schema result id")
    require(set(schema["required"]) == controller.RESULT_KEYS, "schema exact top-level fields")
    definitions = schema["$defs"]
    candidate_definition = definitions["candidate_result"]
    require(set(candidate_definition["required"]) == controller.CANDIDATE_RESULT_KEYS, "schema exact candidate fields")
    require(set(candidate_definition["properties"]) == controller.CANDIDATE_RESULT_KEYS, "schema candidate properties")
    metrics_definition = definitions["metrics"]
    require(set(metrics_definition["required"]) == controller.METRICS_KEYS, "schema exact metrics fields")
    require(set(metrics_definition["properties"]) == controller.METRICS_KEYS, "schema metrics properties")
    require(set(definitions["invalid_accounting"]["required"]) == controller.INVALID_KEYS, "schema invalid fields")
    require(set(definitions["rank_margin"]["required"]) == controller.RANK_KEYS, "schema rank fields")
    require(set(definitions["score_error"]["required"]) == controller.SCORE_ERROR_KEYS, "schema score fields")
    require(set(definitions["top_key"]["required"]) == controller.TOP_KEY_KEYS, "schema top-key fields")
    require(set(definitions["threshold_evaluation"]["required"]) == controller.THRESHOLD_KEYS, "schema threshold fields")
    require(set(definitions["selection"]["required"]) == controller.SELECTION_KEYS, "schema selection fields")
    require(set(definitions["terminal"]["required"]) == controller.TERMINAL_KEYS, "schema terminal fields")
    candidates = schema["properties"]["candidate_results"]
    require(candidates["minItems"] == 4 and candidates["maxItems"] == 4, "schema candidate cardinality")
    prefix = candidates["prefixItems"]
    require(len(prefix) == 4, "schema candidate prefix count")
    for index, expected in enumerate(controller.CANDIDATES):
        properties = prefix[index]["allOf"][1]["properties"]
        for key in ("label", "group_size", "group_count", "exponent_bytes_per_head", "bytes_per_head"):
            require(properties[key]["const"] == expected[key], f"schema candidate[{index}] {key}")


def verify_static_action_contract(
    manifest: dict[str, Any],
    runtime_package: dict[str, Any],
    package: dict[str, Any],
    controller: ModuleType,
) -> None:
    bindings = package["static_bindings"]
    package_sha256 = sha256_file(PACKAGE_PATH)
    runtime_files = [
        {"path": bindings["controller"]["path"], "sha256": sha256_file(CONTROLLER_PATH)},
        {"path": bindings["evaluator"]["path"], "sha256": sha256_file(EVALUATOR_PATH)},
        {"path": "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_PACKAGE.json", "sha256": package_sha256},
        {"path": bindings["result_schema"]["path"], "sha256": sha256_file(SCHEMA_PATH)},
    ]
    exact_keys(
        runtime_package,
        {
            "artifact_kind",
            "claim_boundary",
            "files",
            "future_invocation",
            "mission_id",
            "package_id",
            "package_sha256",
            "runtime_package_schema_id",
            "schema_version",
        },
        "runtime package",
    )
    require(runtime_package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_static_v7_runtime_package", "runtime artifact kind")
    require(runtime_package["runtime_package_schema_id"] == "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_RUNTIME_PACKAGE_V1", "runtime schema")
    require(runtime_package["schema_version"] == 1, "runtime schema version")
    require(runtime_package["mission_id"] == controller.MISSION_ID, "runtime mission")
    require(runtime_package["package_id"] == controller.PACKAGE_ID, "runtime package id")
    require(runtime_package["package_sha256"] == package_sha256, "runtime package checksum")
    require(runtime_package["files"] == runtime_files, "runtime exact ordered files")
    require(
        runtime_package["claim_boundary"]
        == {
            "execution_authorized": False,
            "payload_included": False,
            "runtime_complete": False,
            "status": "STATIC_DESCRIPTOR_ONLY",
        },
        "runtime claim boundary",
    )
    require(
        runtime_package["future_invocation"]
        == {
            "controller_argv": bindings["controller_argv"],
            "environment": bindings["environment"],
            "evaluator_argv": bindings["evaluator_argv"],
            "interpreter": bindings["interpreter"],
            "reviewer": bindings["reviewer"],
        },
        "runtime future invocation",
    )
    require(all(not item["path"].startswith(("build/", "evidence/")) for item in runtime_files), "runtime payload/live path guard")

    runtime_sha256 = sha256_file(RUNTIME_PACKAGE_PATH)
    manifest_artifacts = [
        {"path": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_RUNTIME_PACKAGE.json", "sha256": runtime_sha256},
        {"path": bindings["controller"]["path"], "sha256": sha256_file(CONTROLLER_PATH)},
        {"path": bindings["evaluator"]["path"], "sha256": sha256_file(EVALUATOR_PATH)},
        {"path": "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_PACKAGE.json", "sha256": package_sha256},
        {"path": bindings["result_schema"]["path"], "sha256": sha256_file(SCHEMA_PATH)},
        {"path": bindings["verifier"]["path"], "sha256": sha256_file(VERIFIER_PATH)},
    ]
    exact_keys(
        manifest,
        {
            "artifact_bindings",
            "artifact_kind",
            "claim_boundary",
            "manifest_self_hash_policy",
            "mission_id",
            "pending_fresh_l2_review",
            "root_id",
            "schema_version",
        },
        "manifest",
    )
    require(manifest["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_static_v7_manifest", "manifest artifact kind")
    require(manifest["schema_version"] == 1 and manifest["mission_id"] == controller.MISSION_ID, "manifest identity")
    require(manifest["root_id"] == "qk-gbfp8-head64-granularity-sweep-static-v7-action-root", "manifest root")
    require(manifest["artifact_bindings"] == manifest_artifacts, "manifest exact ordered artifacts")
    require(manifest["manifest_self_hash_policy"] == "EXTERNAL_SHA256_IDENTITY", "manifest self hash policy")
    require(manifest["claim_boundary"] == package["claim_boundary"], "manifest claim boundary")
    require(
        manifest["pending_fresh_l2_review"]
        == {
            "artifact_path": bindings["reviewer"]["acceptance_artifact"],
            "required": True,
            "role": bindings["reviewer"]["role"],
            "status": bindings["reviewer"]["status"],
        },
        "manifest reviewer binding",
    )


def verify_v6_immutability_without_tensor_open(package: dict[str, Any]) -> None:
    evidence = package["v6_immutable_evidence"]
    files = {
        V6_HANDOFF / "round-0001.json": evidence["fresh_l2_terminal_handoff_sha256"],
        V6_HANDOFF / "v6ns-r2-aftermath-recompute.json": evidence["aftermath_recomputation_sha256"],
        V6_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v6/base/result.json": evidence["result_file_sha256"],
        V6_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v6-authority/base/first-terminal.json": evidence["first_terminal_file_sha256"],
    }
    for path, expected in files.items():
        require(path.is_file() and not path.is_symlink(), f"V6 immutable file: {path}")
        require(sha256_file(path) == expected, f"V6 immutable checksum: {path}")
    result = load_json(V6_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v6/base/result.json")
    result_payload = dict(result)
    observed = result_payload.pop("result_sha256")
    require(observed == evidence["result_self_checksum"], "V6 result self-checksum identity")
    require(hashlib.sha256(controller_compact_bytes(result_payload)).hexdigest() == observed, "V6 result self-checksum")
    require(result["terminal"] == {
        "first_record_immutable": True,
        "invocation_count_performed": 1,
        "metrics_published": True,
        "reason_code": "HARD_THRESHOLD_FAILED",
        "retry_replay_resume_repair_permitted": False,
        "status": "FAILED_TERMINAL",
        "thresholds_evaluated": True,
    }, "V6 honest failure terminal")
    terminal = load_json(V6_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v6-authority/base/first-terminal.json")
    require(terminal["authority_consumed"] is True and terminal["status"] == "CONSUMED_ORPHAN", "V6 consumed state")
    require(terminal["retry_replay_resume_repair_permitted"] is False, "V6 no replay")
    tensor = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
    tensor_stat = tensor.lstat()
    require(stat.S_ISREG(tensor_stat.st_mode) and tensor_stat.st_size == 1305797, "V6 sealed tensor lstat binding")
    credential = V6_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v6-authority/base/credential.json"
    require(not credential.exists(), "V6 credential consumed")


def verify_pure_c02_and_numerical_contract(evaluator: ModuleType, controller: ModuleType) -> tuple[int, int]:
    words = (
        0x3F80,
        0x3C00,
        0x3CC0,
        0xBCC0,
        0,
        0,
        0,
        0,
        0x3B80,
        *([0] * 55),
    )
    require(len(words) == 64, "synthetic BF16 head geometry")
    payload = struct.pack("<64H", *words)
    records = [
        {"dtype": "torch.bfloat16", "name": "query", "payload": payload, "shape": (1, 1, 64)},
        {"dtype": "torch.bfloat16", "name": "key", "payload": payload, "shape": (1, 1, 64)},
    ]
    bundle = evaluator.produce_c02_bundle(records)
    producer_records = evaluator.parse_c02_producer_bytes(bundle)
    accepted_records = evaluator.parse_c02_accepted_reader(bundle)
    require(producer_records == accepted_records, "independent c02 cross-parser agreement")
    query_record = producer_records["query"]
    evaluator.validate_record_binding(
        query_record,
        {"dtype": "torch.bfloat16", "shape": [1, 1, 64], "sha256": query_record["sha256"]},
    )
    require(evaluator.finite_bf16_record_words(query_record) == words, "complete BF16 record scan")

    grammar_mutations = 0
    malformed_bundles = [
        bytes((bundle[0] ^ 1,)) + bundle[1:],
        bundle[:-1],
        bundle + b"\0",
        evaluator.C02_MAGIC + b"\0\0\0\0",
    ]
    for malformed in malformed_bundles:
        expect_reject(evaluator.parse_c02_producer_bytes, malformed)
        expect_reject(evaluator.parse_c02_accepted_reader, malformed)
        grammar_mutations += 2
    duplicate = [records[0], dict(records[0])]
    expect_reject(evaluator.produce_c02_bundle, duplicate)
    grammar_mutations += 1
    expect_reject(
        evaluator.validate_record_binding,
        query_record,
        {"dtype": "torch.bfloat16", "shape": [1, 1, 64], "sha256": "0" * 64},
    )
    grammar_mutations += 1
    short_payload_bundle = evaluator.produce_c02_bundle(
        [{"dtype": "torch.bfloat16", "name": "short", "payload": payload[:-2], "shape": (1, 1, 64)}]
    )
    short_record = evaluator.parse_c02_accepted_reader(short_payload_bundle)["short"]
    expect_reject(evaluator.finite_bf16_record_words, short_record)
    grammar_mutations += 1
    nonfinite_words = words + tuple([0] * 63) + (0x7F80,)
    nonfinite_payload = struct.pack("<128H", *nonfinite_words)
    nonfinite_bundle = evaluator.produce_c02_bundle(
        [{"dtype": "torch.bfloat16", "name": "nonfinite", "payload": nonfinite_payload, "shape": (1, 2, 64)}]
    )
    nonfinite_record = evaluator.parse_c02_producer_bytes(nonfinite_bundle)["nonfinite"]
    expect_reject(evaluator.encode_grouped_head_from_record, nonfinite_record, 0, 8)
    grammar_mutations += 1

    arithmetic_mutations = 0
    encoded_by_label: dict[str, dict[str, Any]] = {}
    for candidate in controller.CANDIDATES:
        encoded = evaluator.encode_grouped_head_from_record(query_record, 0, candidate["group_size"])
        packed = evaluator.pack_grouped_head(encoded)
        require(len(packed) == candidate["bytes_per_head"], f"{candidate['label']} packed byte count")
        require(evaluator.unpack_grouped_head(packed, candidate["group_size"]) == encoded, f"{candidate['label']} packed round trip")
        encoded_by_label[candidate["label"]] = encoded
    g8 = encoded_by_label["G8"]
    require(g8["exponents"][:2] == (-6, -14), "G8 independent canonical group exponents")
    require(g8["mantissas"][:4] == (64, 0, 2, -2), "ties-to-even grouped mantissas")
    bad_head = deepcopy(g8)
    bad_mantissas = list(bad_head["mantissas"])
    bad_mantissas[0] = -128
    bad_head["mantissas"] = tuple(bad_mantissas)
    expect_reject(evaluator.pack_grouped_head, bad_head)
    arithmetic_mutations += 1
    bad_head = deepcopy(g8)
    bad_exponents = list(bad_head["exponents"])
    bad_exponents[-1] = 1
    bad_head["exponents"] = tuple(bad_exponents)
    expect_reject(evaluator.pack_grouped_head, bad_head)
    arithmetic_mutations += 1

    one_words = (0x3F80,) * 64
    for candidate in controller.CANDIDATES:
        encoded = evaluator.encode_grouped_head(one_words, candidate["group_size"])
        score_pair = evaluator.grouped_dot_score_pair(encoded, encoded)
        require(score_pair == (1, 6), f"{candidate['label']} exact grouped dot")
        require(evaluator.realize_score_pair_q12_20(score_pair) == (1 << 23), f"{candidate['label']} Q12.20 score")
    normalized, common_exponent, top = evaluator.normalize_score_row_q12_20(
        ((1, 6), (1, 6), (0, 0)),
        (True, True, True),
    )
    require(normalized == (0, 0, -(1 << 23)) and common_exponent == 6 and top == 0, "Q12.20 row normalization and lowest-index tie")
    expect_reject(
        evaluator.normalize_score_row_q12_20,
        ((-(1 << 80), 0), (0, 0)),
        (True, True),
    )
    arithmetic_mutations += 1
    require(tuple(evaluator.kv_head_for_query(index) for index in (0, 6, 7, 13)) == (0, 0, 1, 1), "Q to KV head mapping")
    expect_reject(evaluator.kv_head_for_query, 14)
    arithmetic_mutations += 1
    return grammar_mutations, arithmetic_mutations


def controller_compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def main() -> int:
    require(not (ACTION_ROOT / "build").exists(), "V7 live namespace must not exist")
    package = load_json(PACKAGE_PATH)
    schema = load_json(SCHEMA_PATH)
    runtime_package = load_json(RUNTIME_PACKAGE_PATH)
    manifest = load_json(MANIFEST_PATH)
    controller = load_module("v7_controller", CONTROLLER_PATH)
    evaluator = load_module("v7_evaluator", EVALUATOR_PATH)
    controller.validate_package(package)
    require(controller.PACKAGE_ID == evaluator.PACKAGE_ID, "controller/evaluator package_id")
    require(controller.RESULT_SCHEMA_ID == evaluator.RESULT_SCHEMA_ID, "controller/evaluator schema_id")
    require(tuple(controller.CANDIDATES) == tuple(evaluator.CANDIDATES), "controller/evaluator candidates")
    require(controller.HARD_GATES == evaluator.HARD_GATES, "controller/evaluator hard gates")
    verify_schema_contract(schema, controller)
    schema_validator = Draft202012Validator(schema)
    bindings = package["static_bindings"]
    require(sha256_file(CONTROLLER_PATH) == bindings["controller"]["sha256"], "controller binding")
    require(sha256_file(EVALUATOR_PATH) == bindings["evaluator"]["sha256"], "evaluator binding")
    require(sha256_file(SCHEMA_PATH) == bindings["result_schema"]["sha256"], "schema binding")
    require(sha256_file(VERIFIER_PATH) == bindings["verifier"]["sha256"], "verifier binding")
    require(package["result_contract"]["result_schema"]["sha256"] == bindings["result_schema"]["sha256"], "schema binding agreement")
    verify_static_action_contract(manifest, runtime_package, package, controller)
    grammar_mutations, arithmetic_mutations = verify_pure_c02_and_numerical_contract(evaluator, controller)
    package_sha256 = sha256_file(PACKAGE_PATH)
    evaluator_sha256 = sha256_file(EVALUATOR_PATH)
    failure = evaluator.build_synthetic_result(
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        first_passing_candidate=None,
    )
    success = evaluator.build_synthetic_result(
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        first_passing_candidate="G4",
    )
    require_schema_accepts(schema_validator, failure, "canonical failure")
    require_schema_accepts(schema_validator, success, "canonical success")
    controller.validate_result_record(package, failure, package_sha256=package_sha256, evaluator_sha256=evaluator_sha256)
    controller.validate_result_record(package, success, package_sha256=package_sha256, evaluator_sha256=evaluator_sha256)
    require(controller.classify_terminal(1, "a" * 64, failure)["status"] == "FAILED_TERMINAL", "failure classification")
    require(controller.classify_terminal(0, "a" * 64, success)["status"] == "SUCCEEDED_TERMINAL", "success classification")
    require(success["selected_candidate"] == "G4" and success["selection"]["passing_candidates"] == ["G4", "G2", "G1"], "deterministic selection")

    mutations: list[tuple[str, dict[str, Any], bool]] = []
    mutation = deepcopy(failure); mutation["package_id"] = "WRONG"; mutations.append(("package_id", reseal(evaluator, mutation), True))
    mutation = deepcopy(failure); mutation["schema_id"] = "WRONG"; mutations.append(("schema_id", reseal(evaluator, mutation), True))
    mutation = deepcopy(failure); mutation["candidate_results"].pop(); mutations.append(("candidate cardinality", reseal(evaluator, mutation), True))
    mutation = deepcopy(failure); mutation["candidate_results"][0], mutation["candidate_results"][1] = mutation["candidate_results"][1], mutation["candidate_results"][0]; mutations.append(("candidate order", reseal(evaluator, mutation), True))
    mutation = deepcopy(failure); mutation["candidate_results"][0]["label"] = "G4"; mutations.append(("candidate label", reseal(evaluator, mutation), True))
    mutation = deepcopy(failure); mutation["candidate_results"][0]["bytes_per_head"] = 81; mutations.append(("candidate byte count", reseal(evaluator, mutation), True))
    mutation = deepcopy(success); mutation["selected_candidate"] = "G1"; mutation["selection"]["selected_candidate"] = "G1"; mutations.append(("selected candidate", reseal(evaluator, mutation), True))
    mutation = deepcopy(failure); mutation["candidate_results"][0]["threshold_evaluation"]["top_key_mismatch_count_maximum"]["actual"] = 0; mutations.append(("threshold derivation", reseal(evaluator, mutation), True))
    mutation = deepcopy(failure); mutation["result_sha256"] = "0" * 64; mutations.append(("self-checksum", mutation, False))
    mutation = deepcopy(success); mutation["terminal"]["status"] = "FAILED_TERMINAL"; mutations.append(("terminal status", reseal(evaluator, mutation), True))
    mutation = deepcopy(success); mutation["terminal"]["reason_code"] = "HARD_THRESHOLD_FAILED"; mutations.append(("terminal reason", reseal(evaluator, mutation), True))
    mutation = deepcopy(success); mutation["terminal"]["tensor_open_count"] = 2; mutations.append(("terminal tensor count", reseal(evaluator, mutation), True))
    schema_rejections = 0
    for name, mutation, schema_must_reject in mutations:
        expect_reject(
            controller.validate_result_record,
            package,
            mutation,
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
        )
        if schema_must_reject:
            require_schema_rejects(schema_validator, mutation, name)
            schema_rejections += 1
        else:
            require_schema_accepts(schema_validator, mutation, f"structurally valid {name} mutation")
    package_mutation = deepcopy(package); package_mutation["unexpected"] = True
    expect_reject(controller.validate_package, package_mutation)
    package_mutation = deepcopy(package); package_mutation["candidates"][0]["bytes_per_head"] = 81
    expect_reject(controller.validate_package, package_mutation)
    package_mutation = deepcopy(package); package_mutation["static_bindings"]["controller_argv"][-1] = "REUSED_ACTION_ID"
    expect_reject(controller.validate_package, package_mutation)
    package_mutation = deepcopy(package); package_mutation["static_bindings"]["environment"]["TZ"] = "US/Pacific"
    expect_reject(controller.validate_package, package_mutation)
    package_mutation = deepcopy(package); package_mutation["static_bindings"]["reviewer"]["role"] = "engineer"
    expect_reject(controller.validate_package, package_mutation)
    static_mutations = 0
    runtime_mutation = deepcopy(runtime_package); runtime_mutation["package_sha256"] = "0" * 64
    expect_reject(verify_static_action_contract, manifest, runtime_mutation, package, controller); static_mutations += 1
    runtime_mutation = deepcopy(runtime_package); runtime_mutation["files"][0]["sha256"] = "0" * 64
    expect_reject(verify_static_action_contract, manifest, runtime_mutation, package, controller); static_mutations += 1
    runtime_mutation = deepcopy(runtime_package); runtime_mutation["future_invocation"]["evaluator_argv"][-1] = "REUSED_ACTION_ID"
    expect_reject(verify_static_action_contract, manifest, runtime_mutation, package, controller); static_mutations += 1
    manifest_mutation = deepcopy(manifest); manifest_mutation["artifact_bindings"].reverse()
    expect_reject(verify_static_action_contract, manifest_mutation, runtime_package, package, controller); static_mutations += 1
    manifest_mutation = deepcopy(manifest); manifest_mutation["pending_fresh_l2_review"]["role"] = "engineer"
    expect_reject(verify_static_action_contract, manifest_mutation, runtime_package, package, controller); static_mutations += 1
    expect_reject(controller.classify_terminal, 0, "a" * 64, failure)
    verify_v6_immutability_without_tensor_open(package)
    require(os.environ.get("ACE2_V7_EXECUTION_AUTHORITY") is None, "unexpected V7 execution authority environment")
    print("V7_SCHEMA_COMPATIBILITY=PASS")
    print("V7_STATIC_ACTION_CONTRACT=PASS")
    print("CANONICAL_CONTROLLER_RESULTS_ACCEPTED=2")
    print(f"SEMANTIC_MUTATIONS_REJECTED={len(mutations) + 6}")
    print(f"RESULT_SCHEMA_MUTATIONS_REJECTED={schema_rejections}")
    print(f"STATIC_ACTION_MUTATIONS_REJECTED={static_mutations}")
    print(f"C02_GRAMMAR_GUARD_MUTATIONS_REJECTED={grammar_mutations}")
    print(f"GROUPED_BFP8_ARITHMETIC_MUTATIONS_REJECTED={arithmetic_mutations}")
    print("C02_INDEPENDENT_CROSS_PARSER=PASS")
    print("GROUPED_BFP8_Q12_20_PRIMITIVES=PASS")
    print("V6_IMMUTABLE_EVIDENCE=PASS")
    print("SEALED_TENSOR_ACCESS=LSTAT_ONLY")
    print("LIVE_V7_ARTIFACTS_CREATED=0")
    print(f"MANIFEST_SHA256={sha256_file(MANIFEST_PATH)}")
    print(f"RUNTIME_PACKAGE_SHA256={sha256_file(RUNTIME_PACKAGE_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
