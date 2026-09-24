#!/usr/bin/env python3
"""Decisive static-only verifier for the complete V8 Base surface."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import platform
import stat
import struct
import sys
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from types import ModuleType
from typing import Any

from jsonschema import Draft202012Validator


sys.dont_write_bytecode = True
ACTION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ACTION_ROOT.parents[1]
REFERENCE_ROOT = ACTION_ROOT / "reference"
C02_PATH = REFERENCE_ROOT / "qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
EVALUATOR_PATH = REFERENCE_ROOT / "qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
CONTROLLER_PATH = REFERENCE_ROOT / "qk_gbfp8_head64_granularity_sweep_controller_static_v8.py"
PACKAGE_PATH = REFERENCE_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json"
SCHEMA_PATH = REFERENCE_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json"
FOCUSED_VERIFIER_PATH = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
INCREMENT_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_INCREMENT.json"
RUNTIME_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json"
MANIFEST_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_MANIFEST.json"
SEALED_TENSOR = (
    REPOSITORY_ROOT
    / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root"
    / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def expect_reject(callable_value: Any, *args: Any, **kwargs: Any) -> None:
    try:
        callable_value(*args, **kwargs)
    except Exception:
        return
    raise VerificationError("mutation unexpectedly accepted")


def sha256_file(path: Path) -> str:
    require(os.path.abspath(path) != os.path.abspath(SEALED_TENSOR), "sealed tensor hash prohibited")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def pretty_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(VerificationError(f"nonfinite JSON: {token}")),
    )
    require(type(value) is dict, f"JSON object: {path}")
    return value


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, f"module spec: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def oracle_record_sha256(dtype: str, shape: tuple[int, ...], payload: bytes) -> str:
    prefix = dtype.encode("ascii") + b"\0" + struct.pack(">I", len(shape))
    dimensions = b"".join(struct.pack(">Q", dimension) for dimension in shape)
    return hashlib.sha256(prefix + dimensions + payload).hexdigest()


def binding(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dtype": record["dtype"],
        "sha256": oracle_record_sha256(record["dtype"], record["shape"], record["payload"]),
        "shape": list(record["shape"]),
    }


def fixture_records(*, passing: bool) -> list[dict[str, Any]]:
    query_words = [0] * (14 * 41 * 64)
    key_words = [0] * (2 * 41 * 64)
    oracle_words = [0] * (14 * 41 * 41)

    def query_offset(head: int, position: int, lane: int) -> int:
        return ((head * 41 + position) * 64) + lane

    def key_offset(head: int, position: int, lane: int) -> int:
        return ((head * 41 + position) * 64) + lane

    def oracle_offset(head: int, query_index: int, key_index: int) -> int:
        return ((head * 41 + query_index) * 41) + key_index

    for kv_head in range(2):
        for key_index in range(41):
            key_words[key_offset(kv_head, key_index, key_index)] = 0x3F80
    for query_head in range(14):
        for query_index in range(41):
            rank_row = query_head * 40 + query_index - 1
            oracle_tied = query_index > 0 and rank_row < 214
            realized_top = (0 if oracle_tied else query_index) if passing else query_index
            query_words[query_offset(query_head, query_index, realized_top)] = 0x3F80
            oracle_tops = ({0, 1} if oracle_tied else {query_index}) if passing else ({0, 1} if oracle_tied else {0})
            for key_index in range(query_index + 1):
                oracle_words[oracle_offset(query_head, query_index, key_index)] = (
                    0x0000 if key_index in oracle_tops else 0xBE00
                )

    records = [
        {
            "dtype": "torch.bfloat16",
            "name": "bf16.k_rope",
            "payload": struct.pack(f"<{len(key_words)}H", *key_words),
            "shape": (1, 2, 41, 64),
        },
        {
            "dtype": "torch.bfloat16",
            "name": "bf16.q_rope",
            "payload": struct.pack(f"<{len(query_words)}H", *query_words),
            "shape": (1, 14, 41, 64),
        },
        {
            "dtype": "torch.bfloat16",
            "name": "bf16.qk_scaled_scores",
            "payload": struct.pack(f"<{len(oracle_words)}H", *oracle_words),
            "shape": (1, 14, 41, 41),
        },
    ]
    for index in range(22):
        records.append(
            {
                "dtype": "torch.bfloat16",
                "name": f"fixture.aux.{index:02d}",
                "payload": b"\0\0",
                "shape": (1,),
            }
        )
    require(len(records) == 25, "fixture record count")
    return records


def fixture_input_bindings(
    official: dict[str, Any],
    bindings: dict[str, dict[str, Any]],
    bundle: bytes,
) -> dict[str, Any]:
    value = deepcopy(official)
    value["tensor_bundle"]["byte_count"] = len(bundle)
    value["tensor_bundle"]["sha256"] = hashlib.sha256(bundle).hexdigest()
    for record in value["tensor_records"].values():
        observed = bindings[record["tensor_name"]]
        record["dtype"] = observed["dtype"]
        record["sha256"] = observed["sha256"]
        record["shape"] = observed["shape"]
    return value


def selected_tensor_bindings(input_bindings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        record["tensor_name"]: {
            "dtype": record["dtype"],
            "sha256": record["sha256"],
            "shape": record["shape"],
        }
        for record in input_bindings["tensor_records"].values()
    }


def official_identity_mutations() -> tuple[tuple[str, Any], ...]:
    return (
        ("model", lambda value: value.__setitem__("model_identity_sha256", "8" * 64)),
        ("input token", lambda value: value["input_bindings"].__setitem__("input_token_ids_sha256", "8" * 64)),
        ("lane metadata", lambda value: value["input_bindings"]["lane_metadata"].__setitem__("sha256", "8" * 64)),
        ("tensor bundle", lambda value: value["input_bindings"]["tensor_bundle"].__setitem__("sha256", "8" * 64)),
        ("oracle tensor record", lambda value: value["input_bindings"]["tensor_records"]["bf16_oracle_scores"].__setitem__("sha256", "8" * 64)),
        ("key tensor record", lambda value: value["input_bindings"]["tensor_records"]["realized_key_source"].__setitem__("sha256", "8" * 64)),
        ("query tensor record", lambda value: value["input_bindings"]["tensor_records"]["realized_query_source"].__setitem__("sha256", "8" * 64)),
    )


def result_context(
    *,
    package_sha256: str,
    evaluator_sha256: str,
    action_id: str,
    model_identity_sha256: str,
    input_bindings: dict[str, Any],
    authority_sha256: str = "1" * 64,
    ledger_sha256: str = "2" * 64,
) -> dict[str, Any]:
    return {
        "authority_sha256": authority_sha256,
        "consumed_ledger_sha256": ledger_sha256,
        "evaluator_sha256": evaluator_sha256,
        "fresh_l2_acceptance_sha256": "3" * 64,
        "input_bindings": deepcopy(input_bindings),
        "invocation_sha256": "4" * 64,
        "irreversible_action_id": action_id,
        "model_identity_sha256": model_identity_sha256,
        "package_sha256": package_sha256,
    }


def schema_errors(validator: Draft202012Validator, value: Any) -> list[Any]:
    return sorted(validator.iter_errors(value), key=lambda error: tuple(error.absolute_path))


def verify_schema_validation_stack(
    package: dict[str, Any],
    runtime: dict[str, Any],
    controller: ModuleType,
) -> None:
    expected = controller.SCHEMA_VALIDATION_STACK
    require(package["static_bindings"]["schema_validation_stack"] == expected, "package schema validation stack")
    require(runtime["future_invocation"]["schema_validation_stack"] == expected, "runtime schema validation stack")
    for distribution, expected_version in expected.items():
        try:
            actual_version = distribution_version(distribution)
        except PackageNotFoundError as error:
            raise VerificationError(f"missing schema validation distribution: {distribution}") from error
        require(actual_version == expected_version, f"schema validation distribution version: {distribution}")


def verify_schema_contract(schema: dict[str, Any], controller: ModuleType) -> None:
    Draft202012Validator.check_schema(schema)
    require(schema["$id"] == controller.RESULT_SCHEMA_DOCUMENT_ID, "schema document id")
    require(set(schema["required"]) == controller.RESULT_KEYS, "schema result exact keys")
    definitions = schema["$defs"]
    require(set(definitions["candidate_result"]["required"]) == controller.CANDIDATE_RESULT_KEYS, "schema candidate keys")
    require(set(definitions["metrics"]["required"]) == controller.METRICS_KEYS, "schema metric keys")
    require(set(definitions["invalid_accounting"]["required"]) == controller.INVALID_KEYS, "schema invalid keys")
    require(set(definitions["rank_margin"]["required"]) == controller.RANK_KEYS, "schema rank keys")
    require(set(definitions["score_error"]["required"]) == controller.SCORE_ERROR_KEYS, "schema score keys")
    require(set(definitions["top_key"]["required"]) == controller.TOP_KEY_KEYS, "schema top keys")
    require(set(definitions["threshold_evaluation"]["required"]) == controller.THRESHOLD_KEYS, "schema threshold keys")
    populations = controller.FIXED_METRIC_POPULATIONS
    require(
        definitions["rank_margin"]["properties"]["unique_oracle_top_row_count"]["const"]
        == populations["rank_margin_unique_oracle_top_row_count"],
        "schema rank population",
    )
    require(
        definitions["score_error"]["properties"]["valid_value_count"]["const"]
        == populations["score_error_valid_value_count"],
        "schema score population",
    )
    require(
        definitions["top_key"]["properties"]["row_count"]["const"]
        == populations["top_key_row_count"],
        "schema top population",
    )
    require(
        definitions["rank_fraction"]["properties"]["denominator"]["const"]
        == populations["rank_margin_unique_oracle_top_row_count"],
        "schema rank fraction population",
    )
    require(
        definitions["top_fraction"]["properties"]["denominator"]["const"]
        == populations["top_key_row_count"],
        "schema top fraction population",
    )
    prefix = schema["properties"]["candidate_results"]["prefixItems"]
    require(len(prefix) == 4, "schema candidate cardinality")
    for index, expected in enumerate(controller.CANDIDATES):
        properties = prefix[index]["allOf"][1]["properties"]
        for key in ("bytes_per_head", "exponent_bytes_per_head", "group_count", "group_size", "label"):
            require(properties[key]["const"] == expected[key], f"schema candidate[{index}] {key}")


def verify_static_descriptors(
    package: dict[str, Any],
    runtime: dict[str, Any],
    manifest: dict[str, Any],
    controller: ModuleType,
    *,
    package_sha256: str | None = None,
    runtime_sha256: str | None = None,
    validate_package_binding: bool = True,
) -> None:
    bindings = package["static_bindings"]
    package_sha256 = sha256_file(PACKAGE_PATH) if package_sha256 is None else package_sha256
    runtime_sha256 = sha256_file(RUNTIME_PATH) if runtime_sha256 is None else runtime_sha256
    runtime_files = [
        {"path": controller.C02_PARSER_PATH, "sha256": sha256_file(C02_PATH)},
        {"path": controller.CONTROLLER_PATH, "sha256": sha256_file(CONTROLLER_PATH)},
        {"path": controller.EVALUATOR_PATH, "sha256": sha256_file(EVALUATOR_PATH)},
        {"path": controller.PACKAGE_PATH, "sha256": package_sha256},
        {"path": controller.RESULT_SCHEMA_PATH, "sha256": sha256_file(SCHEMA_PATH)},
    ]
    require(
        set(runtime)
        == {
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
        "runtime exact keys",
    )
    require(runtime["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_static_v8_runtime_package", "runtime kind")
    require(runtime["mission_id"] == controller.MISSION_ID and runtime["schema_version"] == 1, "runtime identity")
    require(runtime["package_id"] == controller.PACKAGE_ID and runtime["package_sha256"] == package_sha256, "runtime package binding")
    require(runtime["files"] == runtime_files, "runtime ordered files")
    require(
        runtime["claim_boundary"]
        == {
            "execution_authorized": False,
            "payload_included": False,
            "runtime_complete": True,
            "status": "STATIC_EXECUTABLE_BUT_INERT",
        },
        "runtime claim boundary",
    )
    require(
        runtime["future_invocation"]
        == {
            "controller_argv": bindings["controller_argv"],
            "environment": bindings["environment"],
            "evaluator_argv": bindings["evaluator_argv"],
            "interpreter": bindings["interpreter"],
            "reviewer": bindings["reviewer"],
            "schema_validation_stack": bindings["schema_validation_stack"],
        },
        "runtime invocation",
    )
    artifacts = [
        {"path": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_INCREMENT.json", "sha256": sha256_file(INCREMENT_PATH)},
        {"path": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json", "sha256": runtime_sha256},
        {"path": controller.C02_PARSER_PATH, "sha256": sha256_file(C02_PATH)},
        {"path": controller.CONTROLLER_PATH, "sha256": sha256_file(CONTROLLER_PATH)},
        {"path": controller.EVALUATOR_PATH, "sha256": sha256_file(EVALUATOR_PATH)},
        {"path": controller.PACKAGE_PATH, "sha256": package_sha256},
        {"path": controller.RESULT_SCHEMA_PATH, "sha256": sha256_file(SCHEMA_PATH)},
        {"path": "tools/verify_qk_gbfp8_head64_granularity_sweep_c02_static_v8.py", "sha256": sha256_file(FOCUSED_VERIFIER_PATH)},
        {"path": controller.VERIFIER_PATH, "sha256": sha256_file(Path(__file__))},
    ]
    require(
        set(manifest)
        == {
            "artifact_bindings",
            "artifact_kind",
            "claim_boundary",
            "frozen_inputs",
            "manifest_self_hash_policy",
            "mission_id",
            "pending_fresh_l2_review",
            "root_id",
            "schema_version",
        },
        "manifest exact keys",
    )
    require(manifest["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_static_v8_manifest", "manifest kind")
    require(manifest["mission_id"] == controller.MISSION_ID and manifest["schema_version"] == 1, "manifest identity")
    require(manifest["root_id"] == ACTION_ROOT.name, "manifest root")
    require(manifest["artifact_bindings"] == artifacts, "manifest ordered artifacts")
    require(manifest["manifest_self_hash_policy"] == "EXTERNAL_SHA256_IDENTITY", "manifest self hash policy")
    require(manifest["claim_boundary"] == package["claim_boundary"], "manifest claim boundary")
    require(
        manifest["pending_fresh_l2_review"]
        == {
            "artifact_path": bindings["reviewer"]["acceptance_artifact"],
            "required": True,
            "role": "Fresh-L2",
            "status": "PENDING_INDEPENDENT_REVIEW",
        },
        "manifest reviewer",
    )
    require(manifest["frozen_inputs"] == load_json(INCREMENT_PATH)["frozen_inputs"], "manifest frozen inputs")
    if validate_package_binding:
        controller.validate_package(package)


def _replace_artifact_sha256(records: list[dict[str, str]], path: str, digest: str) -> None:
    matching = [record for record in records if record["path"] == path]
    require(len(matching) == 1, f"reseal artifact binding: {path}")
    matching[0]["sha256"] = digest


def verify_resealed_static_binding_mutations(
    package: dict[str, Any],
    runtime: dict[str, Any],
    manifest: dict[str, Any],
    controller: ModuleType,
) -> int:
    mutations: list[tuple[str, Any]] = []
    wrong_sha256 = "0" * 64
    for name in controller.STATIC_ARTIFACT_PATHS:
        mutations.append(
            (
                f"static {name} path",
                lambda value, binding_name=name: value["static_bindings"][binding_name].__setitem__(
                    "path", f"forbidden/{binding_name}"
                ),
            )
        )
        mutations.append(
            (
                f"static {name} sha256",
                lambda value, binding_name=name: value["static_bindings"][binding_name].__setitem__(
                    "sha256", wrong_sha256
                ),
            )
        )
    mutations.extend(
        (
            (
                "result contract schema path",
                lambda value: value["result_contract"]["result_schema"].__setitem__("path", "forbidden/result-schema"),
            ),
            (
                "result contract schema sha256",
                lambda value: value["result_contract"]["result_schema"].__setitem__("sha256", wrong_sha256),
            ),
            (
                "coordinated duplicate schema path",
                lambda value: (
                    value["result_contract"]["result_schema"].__setitem__("path", "forbidden/result-schema"),
                    value["static_bindings"]["result_schema"].__setitem__("path", "forbidden/result-schema"),
                ),
            ),
            (
                "coordinated duplicate schema sha256",
                lambda value: (
                    value["result_contract"]["result_schema"].__setitem__("sha256", wrong_sha256),
                    value["static_bindings"]["result_schema"].__setitem__("sha256", wrong_sha256),
                ),
            ),
        )
    )
    for label, mutation in mutations:
        changed_package = deepcopy(package)
        changed_runtime = deepcopy(runtime)
        changed_manifest = deepcopy(manifest)
        mutation(changed_package)
        changed_package_sha256 = sha256_bytes(pretty_json_bytes(changed_package))
        changed_runtime["package_sha256"] = changed_package_sha256
        _replace_artifact_sha256(changed_runtime["files"], controller.PACKAGE_PATH, changed_package_sha256)
        _replace_artifact_sha256(changed_manifest["artifact_bindings"], controller.PACKAGE_PATH, changed_package_sha256)
        changed_runtime_sha256 = sha256_bytes(pretty_json_bytes(changed_runtime))
        _replace_artifact_sha256(
            changed_manifest["artifact_bindings"],
            "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json",
            changed_runtime_sha256,
        )
        verify_static_descriptors(
            changed_package,
            changed_runtime,
            changed_manifest,
            controller,
            package_sha256=changed_package_sha256,
            runtime_sha256=changed_runtime_sha256,
            validate_package_binding=False,
        )
        expect_reject(controller.validate_package, changed_package)
        expect_reject(
            verify_static_descriptors,
            changed_package,
            changed_runtime,
            changed_manifest,
            controller,
            package_sha256=changed_package_sha256,
            runtime_sha256=changed_runtime_sha256,
        )
    return len(mutations)


def verify_import_safety() -> None:
    evaluator_tree = ast.parse(EVALUATOR_PATH.read_text("utf-8"))
    controller_tree = ast.parse(CONTROLLER_PATH.read_text("utf-8"))
    evaluator_imports = {
        alias.name
        for node in ast.walk(evaluator_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    controller_imports = {
        alias.name
        for node in ast.walk(controller_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    require(not evaluator_imports.intersection({"argparse", "subprocess", "torch"}), "evaluator unsafe import")
    require("subprocess" not in controller_imports, "controller process import")
    evaluator_functions = {node.name: node for node in evaluator_tree.body if isinstance(node, ast.FunctionDef)}
    opener_calls = [
        node
        for node in ast.walk(evaluator_functions["evaluate_bundle_once"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open_bundle"
    ]
    require(len(opener_calls) == 1, "evaluator opener cardinality source")


def verify_arithmetic(evaluator: ModuleType, controller: ModuleType) -> int:
    mutations = 0
    words = (0x3F80, 0x3C00, 0x3CC0, 0xBCC0, 0, 0, 0, 0, 0x3B80, *([0] * 55))
    require(len(words) == 64, "arithmetic head geometry")
    encoded_by_label: dict[str, dict[str, Any]] = {}
    for candidate in controller.CANDIDATES:
        encoded = evaluator.encode_grouped_head(words, candidate["group_size"])
        packed = evaluator.pack_grouped_head(encoded)
        require(len(packed) == candidate["bytes_per_head"], f"{candidate['label']} byte geometry")
        require(evaluator.unpack_grouped_head(packed, candidate["group_size"]) == encoded, f"{candidate['label']} round trip")
        encoded_by_label[candidate["label"]] = encoded
    g8 = encoded_by_label["G8"]
    require(g8["exponents"][:2] == (-6, -14), "canonical exponent choice")
    require(g8["mantissas"][:4] == (64, 0, 2, -2), "ties-to-even mantissas")

    mutation = deepcopy(g8)
    mantissas = list(mutation["mantissas"])
    mantissas[0] = -128
    mutation["mantissas"] = tuple(mantissas)
    expect_reject(evaluator.pack_grouped_head, mutation)
    mutations += 1

    mutation = deepcopy(g8)
    exponents = list(mutation["exponents"])
    exponents[0] = 1 << 15
    mutation["exponents"] = tuple(exponents)
    expect_reject(evaluator.pack_grouped_head, mutation)
    mutations += 1

    one_words = (0x3F80,) * 64
    for candidate in controller.CANDIDATES:
        encoded = evaluator.encode_grouped_head(one_words, candidate["group_size"])
        pair = evaluator.grouped_dot_score_pair(encoded, encoded)
        require(pair == (1, 6), f"{candidate['label']} grouped dot")
        require(evaluator.realize_score_pair_q12_20(pair) == (1 << 23), f"{candidate['label']} Q12.20")

    normalized, exponent, top = evaluator.normalize_score_row_q12_20(
        ((1, 6), (1, 6), (0, 0)),
        (True, True, True),
    )
    require(normalized == (0, 0, -(1 << 23)) and exponent == 6 and top == 0, "ranking tie and normalization")
    expect_reject(evaluator.normalize_score_row_q12_20, ((-(1 << 80), 0), (0, 0)), (True, True))
    mutations += 1
    require(tuple(evaluator.kv_head_for_query(index) for index in (0, 6, 7, 13)) == (0, 0, 1, 1), "Q-to-KV mapping")
    expect_reject(evaluator.kv_head_for_query, 14)
    mutations += 1
    return mutations


class MemoryOps:
    def __init__(
        self,
        initial: dict[str, bytes],
        *,
        max_write: int | None = None,
        zero_write_once: bool = False,
        fail_fsync_at: int | None = None,
        fail_unlink: bool = False,
    ) -> None:
        self.files = {path: bytearray(value) for path, value in initial.items()}
        self.max_write = max_write
        self.zero_write_once = zero_write_once
        self.fail_fsync_at = fail_fsync_at
        self.fail_unlink = fail_unlink
        self.events: list[tuple[Any, ...]] = []
        self.handles: dict[int, tuple[str, str]] = {}
        self.next_descriptor = 10
        self.fsync_count = 0

    def exists(self, path: str) -> bool:
        return path in self.files

    def open(self, path: str, flags: int, mode: int) -> int:
        directory = bool(flags & os.O_DIRECTORY)
        if not directory:
            if flags & os.O_EXCL and path in self.files:
                raise FileExistsError(path)
            if flags & os.O_CREAT:
                self.files[path] = bytearray()
        descriptor = self.next_descriptor
        self.next_descriptor += 1
        kind = "dir" if directory else "file"
        self.handles[descriptor] = (kind, path)
        self.events.append(("open", kind, path, flags, mode))
        return descriptor

    def write(self, descriptor: int, payload: bytes) -> int:
        kind, path = self.handles[descriptor]
        require(kind == "file", "write to directory")
        if self.zero_write_once:
            self.zero_write_once = False
            self.events.append(("write", path, 0))
            return 0
        count = len(payload) if self.max_write is None else min(len(payload), self.max_write)
        self.files[path].extend(payload[:count])
        self.events.append(("write", path, count))
        return count

    def fsync(self, descriptor: int) -> None:
        self.fsync_count += 1
        kind, path = self.handles[descriptor]
        self.events.append(("fsync", kind, path, self.fsync_count))
        if self.fail_fsync_at == self.fsync_count:
            raise OSError("injected fsync failure")

    def close(self, descriptor: int) -> None:
        kind, path = self.handles.pop(descriptor)
        self.events.append(("close", kind, path))

    def unlink(self, path: str) -> None:
        self.events.append(("unlink", path))
        if self.fail_unlink:
            raise OSError("injected unlink failure")
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]


def authority_and_credential(
    controller: ModuleType,
    *,
    package_sha256: str,
    evaluator_sha256: str,
    action_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = controller.seal_record(
        {
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_v8_authority",
            "evaluator_sha256": evaluator_sha256,
            "fresh_l2_acceptance_sha256": "3" * 64,
            "input_bindings": deepcopy(controller.OFFICIAL_INPUT_BINDINGS),
            "invocation_sha256": "4" * 64,
            "irreversible_action_id": action_id,
            "model_identity_sha256": controller.OFFICIAL_MODEL_IDENTITY_SHA256,
            "package_sha256": package_sha256,
        },
        "authority_sha256",
    )
    credential = controller.seal_record(
        {
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_v8_credential",
            "authority_sha256": authority["authority_sha256"],
            "credential_nonce_sha256": "7" * 64,
            "irreversible_action_id": action_id,
        },
        "credential_sha256",
    )
    return authority, credential


def verify_official_identity_substitutions(
    package: dict[str, Any],
    controller: ModuleType,
    evaluator: ModuleType,
    candidate_results: list[dict[str, Any]],
    package_sha256: str,
    evaluator_sha256: str,
) -> int:
    cases = 0
    action_id = "ace2:v8:official-identity-static-fixture"
    authority, _credential = authority_and_credential(
        controller,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        action_id=action_id,
    )
    controller.validate_authority_record(
        authority,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
    )
    context = result_context(
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        action_id=action_id,
        model_identity_sha256=evaluator.OFFICIAL_MODEL_IDENTITY_SHA256,
        input_bindings=evaluator.OFFICIAL_INPUT_BINDINGS,
    )
    evaluator.build_result(deepcopy(candidate_results), context)

    for label, mutate in official_identity_mutations():
        substituted_authority = deepcopy(authority)
        mutate(substituted_authority)
        substituted_authority = controller.seal_record(substituted_authority, "authority_sha256")
        expect_reject(
            controller.validate_authority_record,
            substituted_authority,
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
        )
        cases += 1

        substituted_context = deepcopy(context)
        mutate(substituted_context)
        expect_reject(evaluator.build_result, deepcopy(candidate_results), substituted_context)
        cases += 1

    require(package["official_benchmark"] == controller.OFFICIAL_BENCHMARK, "package official benchmark fixture")
    return cases


def lifecycle_ops(
    package: dict[str, Any],
    controller: ModuleType,
    authority: dict[str, Any],
    credential: dict[str, Any],
    **kwargs: Any,
) -> MemoryOps:
    namespaces = package["future_namespaces"]
    return MemoryOps(
        {
            namespaces["authority"]: controller.compact_bytes(authority),
            namespaces["credential"]: controller.compact_bytes(credential),
        },
        **kwargs,
    )


def verify_lifecycle(
    package: dict[str, Any],
    controller: ModuleType,
    evaluator: ModuleType,
    success_result: dict[str, Any],
    failure_result: dict[str, Any],
    package_sha256: str,
    evaluator_sha256: str,
) -> int:
    cases = 0
    action_id = "ace2:v8:lifecycle-static-fixture:0001"
    authority, credential = authority_and_credential(
        controller,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        action_id=action_id,
    )
    authority_bytes = controller.compact_bytes(authority)
    credential_bytes = controller.compact_bytes(credential)
    namespaces = package["future_namespaces"]

    def result_for(ledger: dict[str, Any], source: dict[str, Any]) -> bytes:
        context = result_context(
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
            action_id=action_id,
            model_identity_sha256=evaluator.OFFICIAL_MODEL_IDENTITY_SHA256,
            input_bindings=evaluator.OFFICIAL_INPUT_BINDINGS,
            authority_sha256=authority["authority_sha256"],
            ledger_sha256=ledger["consumed_ledger_sha256"],
        )
        return evaluator.compact_bytes(evaluator.build_result(deepcopy(source["candidate_results"]), context))

    def substituted_result_for(ledger: dict[str, Any], field: str) -> bytes:
        result = controller.load_canonical_json_bytes(result_for(ledger, success_result), "lifecycle substitution fixture")
        if field == "tensor_bundle_sha256":
            result["input_bindings"][field] = "8" * 64
        else:
            result[field] = "8" * 64
        return evaluator.compact_bytes(evaluator.seal_result(result))

    invocations = 0
    ops = lifecycle_ops(package, controller, authority, credential, max_write=7)

    def invoke_success(ledger: dict[str, Any]) -> tuple[int, bytes]:
        nonlocal invocations
        invocations += 1
        ops.events.append(("invoke",))
        return 0, result_for(ledger, success_result)

    terminal = controller.run_inert_lifecycle(
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=ops,
        evaluator_invoke=invoke_success,
    )
    require(terminal["status"] == "SUCCEEDED_TERMINAL" and invocations == 1, "successful lifecycle")
    require(namespaces["credential"] not in ops.files, "credential consumed")
    require(all(path in ops.files for path in (namespaces["ledger"], namespaces["result"], namespaces["first_terminal"])), "lifecycle publications")
    invoke_index = ops.events.index(("invoke",))
    credential_unlink = max(index for index, event in enumerate(ops.events[:invoke_index]) if event == ("unlink", namespaces["credential"]))
    ledger_dir_fsync = max(
        index
        for index, event in enumerate(ops.events[:credential_unlink])
        if event[:3] == ("fsync", "dir", os.path.dirname(namespaces["ledger"]))
    )
    require(ledger_dir_fsync < credential_unlink < invoke_index, "consume-before-payload event order")
    require(sum(1 for event in ops.events if event[0] == "write" and event[1] == namespaces["ledger"]) > 1, "short-write completion")
    expect_reject(
        controller.run_inert_lifecycle,
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=ops,
        evaluator_invoke=invoke_success,
    )
    require(invocations == 1, "permanent no replay after success")
    cases += 2

    failure_ops = lifecycle_ops(package, controller, authority, credential)
    failure_calls = 0

    def invoke_failure(ledger: dict[str, Any]) -> tuple[int, bytes]:
        nonlocal failure_calls
        failure_calls += 1
        return 1, result_for(ledger, failure_result)

    terminal = controller.run_inert_lifecycle(
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=failure_ops,
        evaluator_invoke=invoke_failure,
    )
    require(terminal["status"] == "FAILED_TERMINAL" and failure_calls == 1, "honest failed lifecycle")
    cases += 1

    orphan_ops = lifecycle_ops(package, controller, authority, credential)
    terminal = controller.run_inert_lifecycle(
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=orphan_ops,
        evaluator_invoke=lambda _ledger: (9, None),
    )
    require(terminal["status"] == "CONSUMED_ORPHAN", "no-result orphan")
    require(namespaces["result"] not in orphan_ops.files and namespaces["ledger"] in orphan_ops.files, "orphan artifacts")
    cases += 1

    invalid_ops = lifecycle_ops(package, controller, authority, credential)
    terminal = controller.run_inert_lifecycle(
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=invalid_ops,
        evaluator_invoke=lambda _ledger: (0, b"{}\n"),
    )
    require(terminal["status"] == "CONSUMED_ORPHAN" and terminal["reason_code"] == "INVALID_EVALUATOR_RESULT", "invalid-result orphan")
    cases += 1

    for field in (
        "fresh_l2_acceptance_sha256",
        "invocation_sha256",
        "model_identity_sha256",
        "tensor_bundle_sha256",
    ):
        substitution_ops = lifecycle_ops(package, controller, authority, credential)
        terminal = controller.run_inert_lifecycle(
            package,
            authority_bytes,
            credential_bytes,
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
            ops=substitution_ops,
            evaluator_invoke=lambda ledger, substituted_field=field: (0, substituted_result_for(ledger, substituted_field)),
        )
        require(
            terminal["status"] == "CONSUMED_ORPHAN" and terminal["reason_code"] == "INVALID_EVALUATOR_RESULT",
            f"resealed {field} substitution terminal",
        )
        require(
            namespaces["result"] not in substitution_ops.files
            and namespaces["ledger"] in substitution_ops.files
            and namespaces["first_terminal"] in substitution_ops.files,
            f"resealed {field} substitution publication",
        )
        cases += 1

    zero_ops = lifecycle_ops(package, controller, authority, credential, zero_write_once=True)
    zero_calls = 0

    def zero_invoke(_ledger: dict[str, Any]) -> tuple[int, None]:
        nonlocal zero_calls
        zero_calls += 1
        return 9, None

    expect_reject(
        controller.run_inert_lifecycle,
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=zero_ops,
        evaluator_invoke=zero_invoke,
    )
    require(zero_calls == 0 and namespaces["ledger"] in zero_ops.files, "zero-write fail closed")
    expect_reject(
        controller.run_inert_lifecycle,
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=zero_ops,
        evaluator_invoke=zero_invoke,
    )
    require(zero_calls == 0, "zero-write no replay")
    cases += 2

    fsync_ops = lifecycle_ops(package, controller, authority, credential, fail_fsync_at=1)
    fsync_calls = 0

    def fsync_invoke(_ledger: dict[str, Any]) -> tuple[int, None]:
        nonlocal fsync_calls
        fsync_calls += 1
        return 9, None

    expect_reject(
        controller.run_inert_lifecycle,
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=fsync_ops,
        evaluator_invoke=fsync_invoke,
    )
    require(fsync_calls == 0 and namespaces["ledger"] in fsync_ops.files, "ledger fsync fail closed")
    cases += 1

    unlink_ops = lifecycle_ops(package, controller, authority, credential, fail_unlink=True)
    unlink_calls = 0

    def unlink_invoke(_ledger: dict[str, Any]) -> tuple[int, None]:
        nonlocal unlink_calls
        unlink_calls += 1
        return 9, None

    expect_reject(
        controller.run_inert_lifecycle,
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=unlink_ops,
        evaluator_invoke=unlink_invoke,
    )
    require(unlink_calls == 0 and namespaces["ledger"] in unlink_ops.files, "unlink fault fail closed")
    cases += 1

    result_fsync_ops = lifecycle_ops(package, controller, authority, credential, fail_fsync_at=4)
    terminal = controller.run_inert_lifecycle(
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=result_fsync_ops,
        evaluator_invoke=lambda ledger: (0, result_for(ledger, success_result)),
    )
    require(terminal["status"] == "CONSUMED_ORPHAN", "result publication fault orphan")
    expect_reject(
        controller.run_inert_lifecycle,
        package,
        authority_bytes,
        credential_bytes,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
        ops=result_fsync_ops,
        evaluator_invoke=lambda _ledger: (9, None),
    )
    cases += 2

    return cases


def verify_result_mutations(
    package: dict[str, Any],
    schema_validator: Draft202012Validator,
    controller: ModuleType,
    evaluator: ModuleType,
    success: dict[str, Any],
    failure: dict[str, Any],
    package_sha256: str,
    evaluator_sha256: str,
) -> tuple[int, int]:
    mutations: list[tuple[str, dict[str, Any], bool]] = []
    mutation = deepcopy(success); mutation["unexpected"] = True; mutations.append(("top exact keys", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["package_id"] = "WRONG"; mutations.append(("package id", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["candidate_results"].pop(); mutations.append(("candidate count", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["candidate_results"][0], mutation["candidate_results"][1] = mutation["candidate_results"][1], mutation["candidate_results"][0]; mutations.append(("candidate order", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["candidate_results"][0]["bytes_per_head"] = 81; mutations.append(("candidate geometry", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["selected_candidate"] = "G1"; mutation["selection"]["selected_candidate"] = "G1"; mutations.append(("selection order", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["terminal"]["status"] = "FAILED_TERMINAL"; mutations.append(("terminal status", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["terminal"]["tensor_open_count"] = 2; mutations.append(("tensor open count", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["candidate_results"][0]["threshold_evaluation"]["top_key_mismatch_count_maximum"]["actual"] = 1; mutations.append(("threshold derivation", evaluator.seal_result(mutation), True))
    mutation = deepcopy(success); mutation["result_sha256"] = "0" * 64; mutations.append(("self checksum", mutation, False))
    mutation = deepcopy(success); mutation["authority_sha256"] = "8" * 64; mutations.append(("authority binding", evaluator.seal_result(mutation), False))
    mutation = deepcopy(success); mutation["fresh_l2_acceptance_sha256"] = "8" * 64; mutations.append(("Fresh-L2 acceptance binding", evaluator.seal_result(mutation), False))
    mutation = deepcopy(success); mutation["invocation_sha256"] = "8" * 64; mutations.append(("invocation binding", evaluator.seal_result(mutation), False))
    mutation = deepcopy(success); mutation["model_identity_sha256"] = "8" * 64; mutations.append(("model identity binding", evaluator.seal_result(mutation), False))
    mutation = deepcopy(success); mutation["input_bindings"]["tensor_bundle_sha256"] = "8" * 64; mutations.append(("tensor bundle binding", evaluator.seal_result(mutation), False))

    coupled = deepcopy(success)
    coupled_metrics = coupled["candidate_results"][0]["metrics"]
    coupled_metrics["score_error"]["valid_value_count"] = 1
    coupled_metrics["top_key"] = {
        "matching_fraction": {"denominator": 1, "numerator": 1},
        "matching_row_count": 1,
        "mismatch_count": 0,
        "row_count": 1,
    }
    coupled_thresholds = coupled["candidate_results"][0]["threshold_evaluation"]
    coupled_thresholds["top_key_matching_fraction_minimum"]["actual"] = {"denominator": 1, "numerator": 1}
    coupled = evaluator.seal_result(coupled)
    coupled_payload = deepcopy(coupled)
    coupled_checksum = coupled_payload.pop("result_sha256")
    require(
        hashlib.sha256(evaluator.compact_bytes(coupled_payload)).hexdigest() == coupled_checksum,
        "coupled population mutation resealed",
    )
    expect_reject(
        evaluator.build_result,
        deepcopy(coupled["candidate_results"]),
        result_context(
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
            action_id=success["irreversible_action_id"],
            model_identity_sha256=evaluator.OFFICIAL_MODEL_IDENTITY_SHA256,
            input_bindings=evaluator.OFFICIAL_INPUT_BINDINGS,
            authority_sha256=success["authority_sha256"],
            ledger_sha256=success["consumed_ledger_sha256"],
        ),
    )
    mutations.append(("coupled fixed populations thresholds and checksum", coupled, True))

    schema_rejections = 0
    for name, value, schema_must_reject in mutations:
        expect_reject(
            controller.validate_result_record,
            package,
            value,
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
            expected_authority_sha256="1" * 64,
            expected_consumed_ledger_sha256="2" * 64,
            expected_fresh_l2_acceptance_sha256="3" * 64,
            expected_invocation_sha256="4" * 64,
            expected_irreversible_action_id="ace2:v8:evaluator-fixture:success",
            expected_model_identity_sha256=controller.OFFICIAL_MODEL_IDENTITY_SHA256,
            expected_tensor_bundle_sha256=controller.OFFICIAL_INPUT_BINDINGS["tensor_bundle"]["sha256"],
        )
        if schema_must_reject:
            require(bool(schema_errors(schema_validator, value)), f"schema accepted {name}")
            schema_rejections += 1

    tracking = deepcopy(success)
    tracking["candidate_results"][0]["metrics"]["score_error"] = {
        "maximum_absolute_error_q12_20_lsb": 1,
        "sum_absolute_error_q12_20_lsb": 1,
        "sum_signed_error_q12_20_lsb": 1,
        "sum_squared_error_q40_40_lsb2": 1,
        "valid_value_count": controller.FIXED_METRIC_POPULATIONS["score_error_valid_value_count"],
    }
    tracking = evaluator.seal_result(tracking)
    controller.validate_result_record(
        package,
        tracking,
        package_sha256=package_sha256,
        evaluator_sha256=evaluator_sha256,
    )
    require(tracking["selected_candidate"] == "G8", "score error affected selection")

    expect_reject(controller.load_canonical_json_bytes, evaluator.compact_bytes(success) + b" ", "trailing result bytes")
    expect_reject(controller.load_canonical_json_bytes, b'{"x":NaN}\n', "nonfinite result bytes")
    return len(mutations) + 3, schema_rejections


def main() -> int:
    sealed_open_events: list[tuple[Any, ...]] = []

    def audit_hook(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if isinstance(candidate, (str, bytes, os.PathLike)) and os.path.abspath(os.fsdecode(candidate)) == os.path.abspath(SEALED_TENSOR):
            sealed_open_events.append(arguments)
            raise VerificationError("sealed tensor open/read/hash prohibited")

    sys.addaudithook(audit_hook)
    controller = load_module("v8_controller", CONTROLLER_PATH)
    package = load_json(PACKAGE_PATH)
    runtime = load_json(RUNTIME_PATH)
    verify_schema_validation_stack(package, runtime, controller)

    focused = load_module("v8_focused", FOCUSED_VERIFIER_PATH)
    require(focused.main() == 0, "focused parser verifier")

    c02 = load_module("v8_c02", C02_PATH)
    evaluator = load_module("v8_evaluator", EVALUATOR_PATH)
    schema = load_json(SCHEMA_PATH)
    manifest = load_json(MANIFEST_PATH)
    verify_import_safety()
    verify_schema_contract(schema, controller)
    verify_static_descriptors(package, runtime, manifest, controller)
    resealed_static_binding_mutations = verify_resealed_static_binding_mutations(
        package,
        runtime,
        manifest,
        controller,
    )
    require(controller.PACKAGE_ID == evaluator.PACKAGE_ID, "controller/evaluator package")
    require(controller.RESULT_SCHEMA_ID == evaluator.RESULT_SCHEMA_ID, "controller/evaluator result schema")
    require(tuple(controller.CANDIDATES) == tuple(evaluator.CANDIDATES), "controller/evaluator candidates")
    require(controller.HARD_GATES == evaluator.HARD_GATES, "controller/evaluator gates")
    require(
        controller.FIXED_METRIC_POPULATIONS == evaluator.FIXED_METRIC_POPULATIONS,
        "controller/evaluator fixed metric populations",
    )
    require(
        controller.OFFICIAL_MODEL_IDENTITY_SHA256 == evaluator.OFFICIAL_MODEL_IDENTITY_SHA256,
        "controller/evaluator official model identity",
    )
    require(controller.OFFICIAL_INPUT_BINDINGS == evaluator.OFFICIAL_INPUT_BINDINGS, "controller/evaluator official inputs")
    require(package["official_benchmark"] == controller.OFFICIAL_BENCHMARK, "package official benchmark")

    package_sha256 = sha256_file(PACKAGE_PATH)
    evaluator_sha256 = sha256_file(EVALUATOR_PATH)
    schema_validator = Draft202012Validator(schema)
    canonical_results: dict[str, dict[str, Any]] = {}
    open_counts: dict[str, int] = {}
    official_identity_substitutions = 0

    for label, passing in (("success", True), ("failure", False)):
        records = fixture_records(passing=passing)
        bindings = {record["name"]: binding(record) for record in records}
        bundle = c02.produce_c02_bundle(records)
        require(tuple(c02.parse_c02_producer_bytes(bundle, bindings)) == tuple(sorted(bindings)), f"{label} canonical record order")
        open_counts[label] = 0

        def opener(data: bytes = bundle, fixture_label: str = label) -> bytes:
            open_counts[fixture_label] += 1
            return data

        fixture_inputs = fixture_input_bindings(evaluator.OFFICIAL_INPUT_BINDINGS, bindings, bundle)
        fixture_context = result_context(
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
            action_id=f"ace2:v8:evaluator-fixture:{label}",
            model_identity_sha256=evaluator.OFFICIAL_MODEL_IDENTITY_SHA256,
            input_bindings=fixture_inputs,
        )
        saved_inputs = evaluator.OFFICIAL_INPUT_BINDINGS
        saved_selected = evaluator.OFFICIAL_SELECTED_TENSOR_BINDINGS
        evaluator.OFFICIAL_INPUT_BINDINGS = fixture_inputs
        evaluator.OFFICIAL_SELECTED_TENSOR_BINDINGS = selected_tensor_bindings(fixture_inputs)
        try:
            result_bytes = evaluator.evaluate_bundle_once(opener, bindings, c02, fixture_context)
        finally:
            evaluator.OFFICIAL_INPUT_BINDINGS = saved_inputs
            evaluator.OFFICIAL_SELECTED_TENSOR_BINDINGS = saved_selected
        require(open_counts[label] == 1, f"{label} one-open flow")
        fixture_result = controller.load_canonical_json_bytes(result_bytes, f"{label} fixture result")

        substituted_open_count = 0

        def substituted_opener(data: bytes = bundle) -> bytes:
            nonlocal substituted_open_count
            substituted_open_count += 1
            return data

        official_context = result_context(
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
            action_id=f"ace2:v8:evaluator-fixture:{label}",
            model_identity_sha256=evaluator.OFFICIAL_MODEL_IDENTITY_SHA256,
            input_bindings=evaluator.OFFICIAL_INPUT_BINDINGS,
        )
        expect_reject(evaluator.evaluate_bundle_once, substituted_opener, bindings, c02, official_context)
        require(substituted_open_count == 1, f"{label} substituted bundle one-open rejection")
        official_identity_substitutions += 1

        result_bytes = evaluator.compact_bytes(
            evaluator.build_result(deepcopy(fixture_result["candidate_results"]), official_context)
        )
        result = controller.load_canonical_json_bytes(result_bytes, f"{label} result")
        controller.validate_result_record(
            package,
            result,
            package_sha256=package_sha256,
            evaluator_sha256=evaluator_sha256,
        )
        errors = schema_errors(schema_validator, result)
        require(not errors, f"schema rejected {label}: {errors[0].message if errors else ''}")
        canonical_results[label] = result

    success = canonical_results["success"]
    failure = canonical_results["failure"]
    require(success["selected_candidate"] == "G8", "success fixed-order selection")
    require(success["selection"]["passing_candidates"] == ["G8", "G4", "G2", "G1"], "success all candidates pass")
    require(failure["selected_candidate"] is None and failure["selection"]["passing_candidates"] == [], "complete sweep failure")
    require(failure["terminal"] == {
        "first_record_immutable": True,
        "invocation_count_performed": 1,
        "metrics_published": True,
        "reason_code": "HARD_THRESHOLD_FAILED",
        "retry_replay_resume_repair_permitted": False,
        "status": "FAILED_TERMINAL",
        "tensor_open_count": 1,
        "thresholds_evaluated": True,
    }, "canonical failure terminal")

    official_identity_substitutions += verify_official_identity_substitutions(
        package,
        controller,
        evaluator,
        success["candidate_results"],
        package_sha256,
        evaluator_sha256,
    )

    arithmetic_mutations = verify_arithmetic(evaluator, controller)

    geometry_records = fixture_records(passing=True)
    query = next(record for record in geometry_records if record["name"] == "bf16.q_rope")
    query["shape"] = (1, 14, 40, 64)
    query["payload"] = query["payload"][: 1 * 14 * 40 * 64 * 2]
    geometry_bindings = {record["name"]: binding(record) for record in geometry_records}
    geometry_bundle = c02.produce_c02_bundle(geometry_records)
    geometry_parsed = c02.parse_c02_accepted_reader(geometry_bundle, geometry_bindings)
    expect_reject(evaluator.evaluate_all_candidates, geometry_parsed)
    arithmetic_mutations += 1

    result_mutations, schema_rejections = verify_result_mutations(
        package,
        schema_validator,
        controller,
        evaluator,
        success,
        failure,
        package_sha256,
        evaluator_sha256,
    )

    package_mutations = 0
    for mutation in (
        lambda value: value.__setitem__("unexpected", True),
        lambda value: value["candidates"].reverse(),
        lambda value: value["candidates"][0].__setitem__("bytes_per_head", 81),
        lambda value: value["static_bindings"]["controller_argv"].__setitem__(-1, "REUSED_ACTION_ID"),
        lambda value: value["static_bindings"]["schema_validation_stack"].__setitem__("jsonschema", "4.22.0"),
        lambda value: value["future_namespaces"].__setitem__("result", value["future_namespaces"]["ledger"]),
        lambda value: value["future_protocol"].__setitem__("retry_replay_resume_repair_permitted", True),
    ):
        changed = deepcopy(package)
        mutation(changed)
        expect_reject(controller.validate_package, changed)
        package_mutations += 1
    for _label, mutation in official_identity_mutations():
        changed = deepcopy(package)
        mutation(changed["official_benchmark"])
        expect_reject(controller.validate_package, changed)
        package_mutations += 1

    lifecycle_cases = verify_lifecycle(
        package,
        controller,
        evaluator,
        success,
        failure,
        package_sha256,
        evaluator_sha256,
    )

    require(controller.classify_terminal(0, "a" * 64, success)["status"] == "SUCCEEDED_TERMINAL", "success classification")
    require(controller.classify_terminal(1, "a" * 64, failure)["status"] == "FAILED_TERMINAL", "failure classification")
    expect_reject(controller.classify_terminal, 0, "a" * 64, failure)

    tensor_stat = os.lstat(SEALED_TENSOR)
    require(stat.S_ISREG(tensor_stat.st_mode) and tensor_stat.st_size == 1305797 and stat.S_IMODE(tensor_stat.st_mode) == 0o400, "sealed tensor lstat")
    require(not sealed_open_events, "sealed tensor opened")
    live_paths = [Path(path) for path in package["future_namespaces"].values()]
    require(all(not os.path.lexists(path) for path in live_paths), "live V8 namespace artifact")
    build_root = REPOSITORY_ROOT / "build"
    if build_root.exists():
        require(not list(build_root.glob("*qk-gbfp8-head64-granularity-sweep-v8*")), "V8 build namespace exists")
    require(not (ACTION_ROOT / "build").exists() and not (ACTION_ROOT / "review").exists(), "V8 action-root live state")
    require(os.environ.get("ACE2_V8_EXECUTION_AUTHORITY") is None, "unexpected V8 execution authority")

    print("PASS_V8_BASE_STATIC_CLOSURE")
    print("C02_SORTED_RECORD_ORDER=PASS")
    print("C02_INDEPENDENT_CROSS_PARSERS=2")
    print("EVALUATOR_SHAPED_RESULTS_ACCEPTED=2")
    print("ALL_CANDIDATES_ONE_OPEN=PASS")
    print(f"ARITHMETIC_AND_GEOMETRY_MUTATIONS_REJECTED={arithmetic_mutations}")
    print(f"RESULT_AND_CANONICAL_BYTE_MUTATIONS_REJECTED={result_mutations}")
    print(f"RESULT_SCHEMA_MUTATIONS_REJECTED={schema_rejections}")
    print(f"PACKAGE_MUTATIONS_REJECTED={package_mutations}")
    print(f"RESEALED_STATIC_BINDING_MUTATIONS_REJECTED={resealed_static_binding_mutations}")
    print(f"OFFICIAL_IDENTITY_VALID_CHECKSUM_SUBSTITUTIONS_REJECTED={official_identity_substitutions}")
    print(f"LIFECYCLE_AND_FAULT_CASES={lifecycle_cases}")
    print("CONSUME_BEFORE_PAYLOAD=PASS")
    print("DURABLE_CREATE_ONLY_AND_NO_REPLAY=PASS")
    print("SEALED_TENSOR_ACCESS=LSTAT_ONLY")
    print("LIVE_V8_ARTIFACTS_CREATED=0")
    print("STATIC_ACCEPTANCE_GRANTS_EXECUTION_AUTHORITY=FALSE")
    print(f"PYTHON={platform.python_version()}")
    print("SCHEMA_VALIDATION_STACK=" + json.dumps(controller.SCHEMA_VALIDATION_STACK, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
