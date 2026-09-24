#!/usr/bin/env python3
"""Synthetic-only verifier for the isolated-root Base G16 V5 compatibility package."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import platform
import re
import stat
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY_ROOT = Path("/home/argustest/ace-2")
ACTION_ROOT_ID = "qk-gbfp8-g16-base-cardinality-one-authority-static-v5-action-root"
PACKAGE_REL = "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V5_PACKAGE.json"
SIDECAR_REL = PACKAGE_REL + ".sha256"
PACKAGE_PATH = ROOT / PACKAGE_REL
SIDECAR_PATH = ROOT / SIDECAR_REL
VERIFIER_PATH = Path(__file__).resolve()
CONTROLLER_REL = "reference/qk_gbfp8_g16_base_cardinality_one_authority_controller_static_v5.py"
CONTROLLER_PATH = ROOT / CONTROLLER_REL

TASK_ID = "e14110288e39"
TASK_MISSION_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/e14110288e39/mission.json"
)
TASK_REVIEW_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/e14110288e39/round-0005.json"
)
TASK_MISSION_SHA256 = "4ade4802aaf647c731a417627c94f60963254cb127760aded233cf3439f06e9e"
TASK_REVIEW_SHA256 = "8ce378215be3a55dd50432a3d9f301501133d817e5e343d53929f1fe0eaa4c83"

AUTHORITY_REVIEW_MISSION_ID = "e9bba2b3a6c7"
AUTHORITY_REVIEW_DIRECTORY = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/e9bba2b3a6c7"
)
AUTHORITY_REVIEW_MISSION_PATH = AUTHORITY_REVIEW_DIRECTORY / "mission.json"
AUTHORITY_REVIEW_MISSION_SHA256 = "6e0a8fa01c2c9a52685843d5b6b6422e68456634bf1855798279a012e60a5837"

PRODUCER_PATH = SOURCE_REPOSITORY_ROOT / "tools/diagnose_w4a8_c02_numerical_path.py"
PRODUCER_SHA256 = "5f8c113ebe7d4931b86c7391c677e190a9fd4878117198f1682f27d75690a924"
ACCEPTED_C02_READER_REL = "tools/qk_bfp8_e16_head64_v1_c02_score_path_evaluator_v2.py"
ACCEPTED_C02_READER_PATH = ROOT / ACCEPTED_C02_READER_REL
ACCEPTED_C02_READER_SHA256 = "f27b8662d0137b2c9ca9dc8d977c0b6b6d953c62554658f0e0e3f19d9527d521"
REJECTED_V4_REVIEW_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/4ae7875505a9/round-0001.json"
)
REJECTED_V4_REVIEW_SHA256 = "99a1aac69f76fd14942dab920976a5c775c6517836bc6c74651b21271f2b2268"

V2_ACTION_ROOT = SOURCE_REPOSITORY_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v2_action_root"
V2_FROZEN_BINDINGS = {
    SOURCE_REPOSITORY_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V2_ACTION_ROOT_MANIFEST.json": "7a660215cc0fa944d4c1fc66034b386b93afed100f5518428dd0c16003eb34d4",
    SOURCE_REPOSITORY_ROOT / "verification/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V2_STATIC_REVIEW.json": "eb563991711ead18d4dfda29b95963c9ec621d5205cd45fbc3018c4422269c92",
    V2_ACTION_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V2_PACKAGE.json": "1155e7c469f3605fbc13c8e45da75a12c52f78b06207cf7a3070e35e7190337b",
    V2_ACTION_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V2_PACKAGE.json.sha256": "c789df22fae5d6dbe2b0f82c9d250069d213b21f5ad7df9769b608ba1ddf6af5",
    V2_ACTION_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V2_PACKAGE_SCHEMA.json": "6a7579a593d18a05285f3715807314a8a4f357dff396be43ba813dbd1bd11326",
    V2_ACTION_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_controller_static_v2.py": "750bd1c00b49a1c893369bd57f3876d90a639195acd076427ae9c55e0a8ec01f",
    V2_ACTION_ROOT / "tools/verify_qk_gbfp8_g16_base_cardinality_one_authority_static_v2.py": "151106310b539449abe87cd9da426dee5d47da2c13973533e9038e53cbbfead9",
}

V3_ACTION_ROOT = SOURCE_REPOSITORY_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v3_action_root"
V3_REVIEW_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/33383f584a02/round-0001.json"
)
V3_REVIEW_SHA256 = "c54e5add0dc3aa40b50ce4f0bb6dd479ba9313ab4f4a446cd318a1e06b33ff99"
V3_FROZEN_BINDINGS = {
    SOURCE_REPOSITORY_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V3_ACTION_ROOT_MANIFEST.json": "d2bba775974fb406325652a391b07a42ca8e50e627b171a04c136c75501ba7ba",
    SOURCE_REPOSITORY_ROOT / "verification/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V3_STATIC_REVIEW.json": "d39c5b48fb6d3e8c1e6e51cc763676f1c3c551addc84d29627d078a0ac8d1259",
    V3_ACTION_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V3_PACKAGE.json": "1d3e4412827b3ca7332bff27b9281dcb6ddec7786cd5f1c684eb5f8c6a53da19",
    V3_ACTION_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V3_PACKAGE.json.sha256": "82906357b15a37fd6fa7d2f7fb43f71d9ed7aed528d650e9c5f56cc709ea8f7b",
    V3_ACTION_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V3_PACKAGE_SCHEMA.json": "720a5aafa0ba6d872ce641dbf7a5512ddddd192d6ff4ac4b663368b335cc854b",
    V3_ACTION_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_controller_static_v3.py": "6ed2018a28d3a369a3dea8af6ac9ac6e729ccdb9e08159f4cce62ef34d54b1fa",
    V3_ACTION_ROOT / "reference/qk_gbfp8_g16_head64_successor_diagnostic_static_v1.py": "20e3f4fc891a7a75f2204662c1107a57319f665deafa3de12dfa3ea851d83dbe",
    V3_ACTION_ROOT / "tools/verify_qk_gbfp8_g16_base_cardinality_one_authority_static_v3.py": "fab574f1cc49d98ef03ca13edef0c7f4d2248f6ed4e4ad699a0c0c3436cc378a",
    V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority.json": "12f8d8f94bf7909042803bd35102c1ddd8276abcecca29fa530c2c89e44b04b0",
    V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority-ledger.json": "26989a26d81aa64b3ed9c920ad0bc75ab0c2d6b3e1e8edd890a8d15b5dac1e8d",
    V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1/base/result.json": "2994ae586a47f69ab5fe603c99a2fa2c0151c0d955909b8c97884d1f8438aeb4",
    V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/first-terminal.json": "cb8c4335cb64272662f4baed9cc5e70f92575e0e1451fb2cd739095f8c6ca666",
}

V4_ACTION_ROOT = SOURCE_REPOSITORY_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v4_action_root"
V4_FROZEN_BINDINGS = {
    SOURCE_REPOSITORY_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V4_ACTION_ROOT_MANIFEST.json": "37ca9c7f795c9bd9f544458c76a5c44d1a09f0fb8742b70099a4cf679e631218",
    SOURCE_REPOSITORY_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V4_ACTION_ROOT_MANIFEST.json.sha256": "ed9739a74e0661f31d3d1cca14ea91d78987404d1b176496055f1d3871cf94e7",
    SOURCE_REPOSITORY_ROOT / "verification/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V4_STATIC_REVIEW.json": "84b563bfa99c438400b49ae6cc788180043be1540358038f192d10a0eb8280a6",
    SOURCE_REPOSITORY_ROOT / "verification/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V4_STATIC_REVIEW.json.sha256": "8be10a3998d685e44017d47941043bfe9822dcffb4f190760111ac8e14df92bf",
    V4_ACTION_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V4_PACKAGE.json": "42403dd7ea79d3fff767d82e3893084bcfcd1afe646d18564b5bb729851d7907",
    V4_ACTION_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V4_PACKAGE.json.sha256": "751b4646b4a5f49ec2bd94ecaf5927c094144a1c20583bf2afbad1b266683e55",
    V4_ACTION_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_controller_static_v4.py": "68ddcd77ba270a60664e3e211acd58ea16980c7af5b5c75e99641e9dca7807ac",
    V4_ACTION_ROOT / "reference/qk_gbfp8_g16_head64_successor_diagnostic_static_v4.py": "2c51e985f3aabbfea85767fa2e26c91fb9670e25e77cbf0169185cf2bef48682",
    V4_ACTION_ROOT / "tools/verify_qk_gbfp8_g16_base_cardinality_one_authority_static_v4.py": "576201adc182f6b1b526d3bbf56c9e88d15d7ba115d354b026ed370b38807719",
}

G16_PACKAGE_REL = "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE.json"
G16_SIDECAR_REL = G16_PACKAGE_REL + ".sha256"
G16_CONTRACT_REL = "design/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_CONTRACT.json"
G16_SCHEMA_REL = "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_SCHEMA.json"
G16_EVALUATOR_REL = "reference/qk_gbfp8_g16_head64_successor_diagnostic_static_v5.py"
G16_VERIFIER_REL = "tools/verify_qk_gbfp8_g16_head64_successor_diagnostic_static_v1.py"
G16_PROOF_REL = "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_NO_EXECUTION_PROOF.json"
G16_PACKAGE_SHA256 = "42da06a7b28c889f60f6a8484cdf4c48f4ca03fe548061d76a20beae1370d863"
G16_CONTRACT_SHA256 = "b2f60e2dc54d95193315678df2092755d2eb58b45b8ee86573f000de242f7ccc"
G16_SCHEMA_SHA256 = "8d512af26bf39593236098f850ae5477ae13343ffbdeac4fbb207b1b8fd71840"
G16_EVALUATOR_SHA256 = "3fb622acfef9efe389eeed4d4081d8d4ce93fcc7d696dc6c095129ef46f248f5"
G16_VERIFIER_SHA256 = "bf2bb603a8919b0abb4f0bcdbfc34c278d451945f3bfb34cbc0807b478795f0c"
G16_PROOF_SHA256 = "4544873432051ae92e69d468ec63b1e54acd881ca40574f777cef1a719dd7356"
G16_SIDECAR_SHA256 = "33061cd3443c5a4076ddd2901715a91a89cc79adadf6ba79fe4613f4e453a0b3"
ACCEPTED_G16_PACKAGE_SHA256 = "33c42fe340867f5b89d451339ca2d552f0bfec3309d8a6103066c171d7806513"

AUTHORITY_PACKAGE_ID = "QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V5"
AUTHORITY_ID = "qk-gbfp8-g16-base-cardinality-one-v5"
AUTHORITY_ROOT_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v5-authority"
AUTHORITY_LANE_ROOT_REL = AUTHORITY_ROOT_REL + "/base"
AUTHORITY_REL = AUTHORITY_LANE_ROOT_REL + "/authority.json"
CREDENTIAL_REL = AUTHORITY_LANE_ROOT_REL + "/credential.json"
LEDGER_REL = AUTHORITY_LANE_ROOT_REL + "/authority-ledger.json"
FIRST_TERMINAL_REL = AUTHORITY_LANE_ROOT_REL + "/first-terminal.json"
RESULT_ROOT_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v5"
RESULT_REL = RESULT_ROOT_REL + "/base/result.json"
V1_AUTHORITY_ROOT_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority"
V1_RESULT_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v1/base/result.json"
METADATA_REL = "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
TENSOR_REL = (
    "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/"
    "attention-substage-tensors.bin"
)
FORBIDDEN_PAYLOAD_REL = TENSOR_REL
PACKAGE_SCHEMA_REL = "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V5_PACKAGE_SCHEMA.json"
CREDENTIAL_SCHEMA_REL = "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_CREDENTIAL_V2_SCHEMA.json"
TERMINAL_SCHEMA_REL = "reference/QK_GBFP8_G16_BASE_FIRST_TERMINAL_V5_SCHEMA.json"

INTERPRETER_PATH = Path("/home/argustest/miniconda3/bin/python3.13")
INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}

BASE_INVOCATION_SHA256 = "9b7b3d5f5e0a94aad412b165cd41c75e87e4f936d7d2b4f0af55fb698ea6112f"
BASE_MODEL_SHA256 = "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7"
BASE_METADATA_SHA256 = "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a"
BASE_TENSOR_SHA256 = "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175"
CREDENTIAL_SCHEMA_ID = "QK_GBFP8_G16_BASE_CARDINALITY_ONE_CREDENTIAL_V2"
FIRST_TERMINAL_SCHEMA_ID = "QK_GBFP8_G16_BASE_FIRST_TERMINAL_V5"
VISIBLE_VALID_LEDGER = "VISIBLE_VALID_LEDGER"
NO_VISIBLE_LEDGER_AMBIGUOUS = "NO_VISIBLE_LEDGER_AFTER_EVALUATOR_TREATED_AS_CONSUMED_ORPHAN"
REQUIRED_DECISION = "MATERIALIZE_BASE_AUTHORITY_ONCE"
ACTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,255}$"
SELECTED_ROLES = ("bf16_oracle_scores", "realized_key_source", "realized_query_source")
TENSOR_BUNDLE_MAGIC = b"ACE2-C02-TENSORS-V1\n"

EVALUATOR_ARGV = [
    str(INTERPRETER_PATH),
    "-I",
    "-B",
    G16_EVALUATOR_REL,
    "--package",
    G16_PACKAGE_REL,
    "--authority",
    AUTHORITY_REL,
    "--ledger",
    LEDGER_REL,
    "--lane",
    "Base",
    "--metadata",
    METADATA_REL,
    "--tensor-bundle",
    TENSOR_REL,
    "--output",
    RESULT_REL,
]
VERIFIER_ARGV = [str(INTERPRETER_PATH), "-I", "-B", str(VERIFIER_PATH.relative_to(ROOT))]


class VerificationError(RuntimeError):
    """Fail-closed static verification error."""


SchemaValidationError = VerificationError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    forbidden = {
        (ROOT / FORBIDDEN_PAYLOAD_REL).resolve(),
        (SOURCE_REPOSITORY_ROOT / FORBIDDEN_PAYLOAD_REL).resolve(),
    }
    require(path.resolve() not in forbidden, "sealed tensor payload open")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def pretty_text(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def object_binding(value: Any) -> dict[str, Any]:
    payload = compact_bytes(value)
    return {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def reject_nonfinite(value: Any, context: str = "record") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{context} nonfinite")
    elif isinstance(value, dict):
        for key, nested in value.items():
            require(type(key) is str, f"{context} non-string key")
            reject_nonfinite(nested, f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_nonfinite(nested, f"{context}[{index}]")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        require(key not in value, f"duplicate JSON key {key}")
        value[key] = nested
    return value


def load_canonical_json(path: Path) -> Any:
    verify_plain_regular(path)
    text = path.read_text(encoding="ascii")
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc
    reject_nonfinite(value)
    require(text.encode("ascii") == compact_bytes(value), f"noncanonical JSON {path}")
    return value


def load_static_json(path: Path) -> Any:
    verify_plain_regular(path)
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid static JSON {path}: {exc}") from exc
    reject_nonfinite(value)
    return value


def valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def reason_contains_full_sha(reason: Any, sha256: str) -> bool:
    return (
        type(reason) is str
        and valid_sha256(sha256)
        and re.search(rf"(?<![0-9a-f]){re.escape(sha256)}(?![0-9a-f])", reason) is not None
    )


def _schema_type_matches(instance: Any, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected == "boolean":
        return type(instance) is bool
    if expected == "integer":
        return type(instance) is int
    if expected == "number":
        return (type(instance) is int) or (type(instance) is float and math.isfinite(instance))
    if expected == "string":
        return type(instance) is str
    if expected == "array":
        return type(instance) is list
    if expected == "object":
        return type(instance) is dict
    raise SchemaValidationError(f"unsupported schema type {expected}")


def _resolve_schema_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    require(reference.startswith("#/"), "local schema reference")
    current: Any = root_schema
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        require(type(current) is dict and token in current, "schema reference target")
        current = current[token]
    require(type(current) is dict, "schema reference object")
    return current


def _schema_matches(instance: Any, schema: dict[str, Any], root_schema: dict[str, Any], context: str) -> None:
    if "$ref" in schema:
        _schema_matches(instance, _resolve_schema_ref(root_schema, schema["$ref"]), root_schema, context)
    if "type" in schema:
        expected_types = schema["type"] if type(schema["type"]) is list else [schema["type"]]
        require(
            all(type(item) is str for item in expected_types)
            and any(_schema_type_matches(instance, item) for item in expected_types),
            f"{context} schema type",
        )
    if "const" in schema:
        require(instance == schema["const"] and type(instance) is type(schema["const"]), f"{context} const")
    if "enum" in schema:
        require(any(instance == item and type(instance) is type(item) for item in schema["enum"]), f"{context} enum")
    if "oneOf" in schema:
        matches = 0
        for candidate in schema["oneOf"]:
            try:
                _schema_matches(instance, candidate, root_schema, context)
            except SchemaValidationError:
                continue
            matches += 1
        require(matches == 1, f"{context} oneOf")
    for candidate in schema.get("allOf", []):
        _schema_matches(instance, candidate, root_schema, context)
    if "if" in schema:
        try:
            _schema_matches(instance, schema["if"], root_schema, context)
        except SchemaValidationError:
            condition = False
        else:
            condition = True
        branch = schema.get("then") if condition else schema.get("else")
        if branch is not None:
            _schema_matches(instance, branch, root_schema, context)
    if type(instance) is dict:
        required = schema.get("required", [])
        require(type(required) is list and all(type(item) is str for item in required), f"{context} required schema")
        require(all(item in instance for item in required), f"{context} required")
        properties = schema.get("properties", {})
        require(type(properties) is dict, f"{context} properties schema")
        if schema.get("additionalProperties") is False:
            require(set(instance).issubset(properties), f"{context} additional property")
        for key, nested_schema in properties.items():
            if key in instance:
                _schema_matches(instance[key], nested_schema, root_schema, f"{context}.{key}")
    if type(instance) is list:
        if "minItems" in schema:
            require(len(instance) >= schema["minItems"], f"{context} minItems")
        if "items" in schema:
            for index, item in enumerate(instance):
                _schema_matches(item, schema["items"], root_schema, f"{context}[{index}]")
    if type(instance) is str:
        if "minLength" in schema:
            require(len(instance) >= schema["minLength"], f"{context} minLength")
        if "pattern" in schema:
            require(re.search(schema["pattern"], instance) is not None, f"{context} pattern")
    if type(instance) is int or type(instance) is float:
        if "minimum" in schema:
            require(instance >= schema["minimum"], f"{context} minimum")
        if "maximum" in schema:
            require(instance <= schema["maximum"], f"{context} maximum")


def validate_json_schema(instance: Any, schema: dict[str, Any]) -> None:
    reject_nonfinite(instance)
    _schema_matches(instance, schema, schema, "result")


def validate_result_record(
    package: dict[str, Any], schema: dict[str, Any], result: dict[str, Any], authority_package_sha256: str
) -> None:
    validate_json_schema(result, schema)
    require(valid_sha256(result["result_sha256"]), "result checksum syntax")
    payload = dict(result)
    checksum = payload.pop("result_sha256")
    require(hashlib.sha256(compact_bytes(payload)).hexdigest() == checksum, "result self-checksum")
    lane = package["base_lane"]
    require(result["authority_sha256"] == lane["authority_record"]["canonical_sha256"], "result authority binding")
    require(result["evaluator_sha256"] == G16_EVALUATOR_SHA256, "result evaluator binding")
    require(result["invocation_sha256"] == BASE_INVOCATION_SHA256, "result invocation binding")
    require(result["lane_label"] == "Base" and result["namespace_label"] == "base", "result lane binding")
    require(result["generation_id"] == lane["generation_id"], "result generation binding")
    require(result["model_identity_sha256"] == lane["model_identity_sha256"], "result model binding")
    require(result["package_sha256"] == G16_PACKAGE_SHA256, "result G16 package binding")
    require(result["input_bindings"] == {
        "input_token_ids_sha256": lane["sealed_inputs"]["input_token_ids_sha256"],
        "lane_metadata": lane["sealed_inputs"]["metadata"],
        "sealed_set_id": "w4a8-c02-attention-substage-trace-v2",
        "tensor_bundle": lane["sealed_inputs"]["tensor_bundle"],
        "tensor_records": lane["sealed_inputs"]["authoritative_tensors"],
    }, "result input binding")
    terminal = result["terminal"]
    require(terminal["first_record_immutable"] is True, "result immutability")
    require(terminal["retry_replay_resume_repair_permitted"] is False, "result replay policy")
    require(terminal["invocation_count_performed"] == 1, "result invocation count")
    require(valid_sha256(authority_package_sha256), "controller package binding")


def verify_existing_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.exists() or current.is_symlink():
            require(not current.is_symlink(), f"symlink component {current}")


def verify_plain_regular(path: Path) -> None:
    verify_existing_components(path)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise VerificationError(f"missing regular file {path}") from exc
    require(stat.S_ISREG(mode), f"non-regular file {path}")


def verify_absent_plain(path: Path) -> None:
    verify_existing_components(path)
    require(not path.exists() and not path.is_symlink(), f"pre-existing namespace {path}")


def binding(path: Path, expected_sha256: str, *, hash_file: bool = True) -> dict[str, Any]:
    verify_plain_regular(path)
    result = {"byte_count": path.stat().st_size, "path": path.as_posix(), "sha256": expected_sha256}
    if hash_file:
        require(sha256_file(path) == expected_sha256, f"SHA-256 mismatch {path}")
    return result


def repo_binding(relative: str, expected_sha256: str, *, hash_file: bool = True) -> dict[str, Any]:
    result = binding(ROOT / relative, expected_sha256, hash_file=hash_file)
    result["path"] = relative
    return result


def authority_template() -> dict[str, Any]:
    return {
        "authority_cardinality": 1,
        "authority_id": AUTHORITY_ID,
        "evaluator_sha256": G16_EVALUATOR_SHA256,
        "invocation_sha256": BASE_INVOCATION_SHA256,
        "lane_label": "Base",
        "output_path": RESULT_REL,
        "package_sha256": G16_PACKAGE_SHA256,
        "result_schema_sha256": G16_SCHEMA_SHA256,
        "state": "READY_UNCONSUMED",
    }


def ledger_template() -> dict[str, Any]:
    authority_sha256 = object_binding(authority_template())["sha256"]
    return {
        "authority_id": AUTHORITY_ID,
        "authority_sha256": authority_sha256,
        "invocation_sha256": BASE_INVOCATION_SHA256,
        "lane_label": "Base",
        "schema_id": "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_AUTHORITY_LEDGER_V1",
        "state": "CONSUMED",
    }


def projected_dependency_bindings(
    g16_package: dict[str, Any], contract: dict[str, Any], base_lane: dict[str, Any]
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    compatibility_owned = {G16_CONTRACT_REL, G16_EVALUATOR_REL}
    bindings: list[dict[str, Any]] = [
        value for value in g16_package["artifacts"].values() if value["path"] not in compatibility_owned
    ]
    bindings.append(base_lane["metadata"])
    bindings.extend(
        value
        for value in contract["accepted_g16"].values()
        if isinstance(value, dict) and {"path", "sha256"}.issubset(value) and not Path(value["path"]).is_absolute()
    )
    bindings.extend(contract["preserved_c02_static_bindings"].values())
    for item in bindings:
        relative = item["path"]
        projected = ROOT / relative
        source = SOURCE_REPOSITORY_ROOT / relative
        verify_plain_regular(projected)
        verify_plain_regular(source)
        expected_sha256 = item["sha256"]
        require(sha256_file(projected) == expected_sha256, f"projected dependency checksum {relative}")
        require(sha256_file(source) == expected_sha256, f"source dependency checksum {relative}")
        require(projected.stat().st_size == source.stat().st_size, f"projected dependency size {relative}")
        if "byte_count" in item:
            require(projected.stat().st_size == item["byte_count"], f"accepted dependency bytes {relative}")
        by_path[relative] = {
            "byte_count": projected.stat().st_size,
            "projected_path": relative,
            "sha256": expected_sha256,
            "source_path": relative,
        }
    return [by_path[path] for path in sorted(by_path)]


def compatibility_runtime_bindings() -> dict[str, dict[str, Any]]:
    return {
        "contract": repo_binding(G16_CONTRACT_REL, G16_CONTRACT_SHA256),
        "evaluator": repo_binding(G16_EVALUATOR_REL, G16_EVALUATOR_SHA256),
        "package": repo_binding(G16_PACKAGE_REL, G16_PACKAGE_SHA256),
        "package_sidecar": repo_binding(G16_SIDECAR_REL, G16_SIDECAR_SHA256),
    }


def expected_package() -> dict[str, Any]:
    g16_package = load_static_json(ROOT / G16_PACKAGE_REL)
    contract = load_static_json(ROOT / G16_CONTRACT_REL)
    proof = load_static_json(ROOT / G16_PROOF_REL)
    base_lane = contract["input_set"]["lanes"][0]
    base_invocation = contract["invocations"][0]
    require(base_lane["label"] == "Base", "accepted Base lane")
    require(base_invocation["lane_label"] == "Base", "accepted Base invocation")
    authority = authority_template()
    ledger = ledger_template()
    authority_bytes = object_binding(authority)
    ledger_bytes = object_binding(ledger)
    g16_sidecar = repo_binding(G16_SIDECAR_REL, G16_SIDECAR_SHA256)
    controller_binding = repo_binding(CONTROLLER_REL, sha256_file(CONTROLLER_PATH))
    review_mission_binding = binding(AUTHORITY_REVIEW_MISSION_PATH, AUTHORITY_REVIEW_MISSION_SHA256)
    projected_dependencies = projected_dependency_bindings(g16_package, contract, base_lane)
    schema_bindings = {
        "authority_package": {
            **repo_binding(PACKAGE_SCHEMA_REL, sha256_file(ROOT / PACKAGE_SCHEMA_REL)),
            "schema_id": "QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V5_PACKAGE_SCHEMA",
        },
        "credential": {
            **repo_binding(CREDENTIAL_SCHEMA_REL, sha256_file(ROOT / CREDENTIAL_SCHEMA_REL)),
            "schema_id": "QK_GBFP8_G16_BASE_CARDINALITY_ONE_CREDENTIAL_V2_SCHEMA",
        },
        "first_terminal": {
            **repo_binding(TERMINAL_SCHEMA_REL, sha256_file(ROOT / TERMINAL_SCHEMA_REL)),
            "schema_id": "QK_GBFP8_G16_BASE_FIRST_TERMINAL_V5_SCHEMA",
        },
    }
    return {
        "action_root_projection": {
            "absolute_root": str(ROOT),
            "accepted_dependency_projection": projected_dependencies,
            "action_root_id": ACTION_ROOT_ID,
            "evaluator_argv_compatibility_revision": "V5 exact-c02 evaluator and create-only namespaces only",
            "evaluator_argv_exactly_accepted": False,
            "namespace_separation": "All V5 compatibility evaluator paths resolve beneath this distinct V5 action root, never beneath consumed V3, rejected V4, or closed V1/V2 roots.",
            "sealed_tensor_materialized": False,
            "sealed_tensor_runtime_byte_count": base_lane["tensor_bundle"]["byte_count"],
            "sealed_tensor_runtime_path": TENSOR_REL,
            "sealed_tensor_runtime_sha256": base_lane["tensor_bundle"]["sha256"],
            "source_repository_root": str(SOURCE_REPOSITORY_ROOT),
            "static_template_state": "STATIC_DEPENDENCIES_PROJECTED_SEALED_PAYLOAD_SLOT_ABSENT",
        },
        "accepted_diagnostic": {
            "accepted_baseline_package_sha256": ACCEPTED_G16_PACKAGE_SHA256,
            "artifacts": g16_package["artifacts"],
            "compatibility_status": "CURRENT_MISSION_FRESH_L2_REQUIRED",
            "contract_id": g16_package["contract_id"],
            "evaluator_id": g16_package["evaluator_id"],
            "fresh_l2_handoff": {
                **binding(TASK_REVIEW_PATH, TASK_REVIEW_SHA256),
                "mission_id": TASK_ID,
                "status": "done",
            },
            "package": repo_binding(G16_PACKAGE_REL, G16_PACKAGE_SHA256),
            "package_id": g16_package["package_id"],
            "result_schema_id": g16_package["result_schema_id"],
            "sidecar": g16_sidecar,
            "task_mission": {**binding(TASK_MISSION_PATH, TASK_MISSION_SHA256), "mission_id": TASK_ID},
        },
        "artifact_kind": "qk_gbfp8_g16_base_cardinality_one_execution_authority_static_v5_action_root_freeze",
        "authority_controller": {
            "argv_template": [
                str(INTERPRETER_PATH),
                "-I",
                "-B",
                CONTROLLER_REL,
                "--authority-package",
                PACKAGE_REL,
                "--authority",
                AUTHORITY_REL,
                "--credential",
                CREDENTIAL_REL,
                "--fresh-l2-review",
                "{fresh_l2_review_path}",
                "--irreversible-action-id",
                "{irreversible_action_id}",
                "--first-terminal",
                FIRST_TERMINAL_REL,
            ],
            "byte_count": controller_binding["byte_count"],
            "controller_id": "QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_CONTROLLER_STATIC_V5",
            "environment": ENVIRONMENT,
            "evaluator_argv": EVALUATOR_ARGV,
            "execution_state": "NOT_EXECUTED",
            "implementation": "CPython",
            "interpreter_path": str(INTERPRETER_PATH),
            "interpreter_sha256": INTERPRETER_SHA256,
            "path": CONTROLLER_REL,
            "python_version": "3.13.5",
            "sha256": controller_binding["sha256"],
        },
        "authority_package_id": AUTHORITY_PACKAGE_ID,
        "authority_review_requirement": {
            "acceptance_state": "FUTURE_EXTERNAL_BINDING_REQUIRED",
            "handoff_directory": str(AUTHORITY_REVIEW_DIRECTORY),
            "mission": {**review_mission_binding, "mission_id": AUTHORITY_REVIEW_MISSION_ID},
            "mission_id": AUTHORITY_REVIEW_MISSION_ID,
            "mission_path": str(AUTHORITY_REVIEW_MISSION_PATH),
            "required_package_checksum_in_reason": True,
            "required_producer_role": "reviewer",
            "required_status": "done",
        },
        "base_lane": {
            "authority_cardinality": 1,
            "authority_record": {
                "canonical_byte_count": authority_bytes["byte_count"],
                "canonical_sha256": authority_bytes["sha256"],
                "materialization_state": "NOT_MATERIALIZED",
                "single_use": True,
                "template": authority,
            },
            "credential_requirement": {
                "live_credential": None,
                "materialization_state": "NOT_MATERIALIZED",
                "required_future_bindings": [
                    "authority_package_sha256",
                    "authority_record_sha256",
                    "authority_controller_sha256",
                    "fresh_l2_acceptance_sha256",
                    "irreversible_action_id",
                ],
                "required_future_decision": "MATERIALIZE_BASE_AUTHORITY_ONCE",
                "substitution_permitted": False,
            },
            "generation_id": base_lane["generation_id"],
            "invocation": base_invocation,
            "lane_label": "Base",
            "ledger_record": {
                "canonical_byte_count": ledger_bytes["byte_count"],
                "canonical_sha256": ledger_bytes["sha256"],
                "materialization_state": "NOT_MATERIALIZED",
                "template": ledger,
            },
            "model_identity_sha256": base_lane["model_identity_sha256"],
            "namespace_label": "base",
            "output_identity": {
                "output_path": RESULT_REL,
                "result_schema_id": g16_package["result_schema_id"],
                "result_schema_sha256": G16_SCHEMA_SHA256,
            },
            "sealed_inputs": {
                "authoritative_tensors": base_lane["authoritative_tensors"],
                "input_token_ids_sha256": base_lane["input_token_ids_sha256"],
                "metadata": base_lane["metadata"],
                "tensor_bundle": base_lane["tensor_bundle"],
            },
        },
        "checkpoint_176_policy": {
            "authority_cardinality": 0,
            "authority_materialized": False,
            "lane_included": False,
            "required_before_future_authority": [
                "checksum-valid Base SUCCEEDED_TERMINAL result",
                "complete valid Base CONSUMED ledger",
                "new separately checksum-bound authority package",
                "new explicit Fresh-L2 review",
                "new separately bound irreversible materialization action",
            ],
        },
        "claim_boundary": {
            "authority_or_credential_materialized": False,
            "authority_controller_executed": False,
            "authority_ready_or_consumed": False,
            "c02_replay_or_invocation": False,
            "checkpoint_176_authority_created": False,
            "evaluator_invoked_on_sealed_data": False,
            "ledger_or_first_terminal_created": False,
            "model_loaded": False,
            "result_namespace_created": False,
            "rtl_u280_xrt_hbm2_activity": False,
            "sealed_tensor_payload_accessed": False,
            "stage_transition": False,
            "status": "STATIC_BYTES_ONLY_NO_EXECUTION_AUTHORIZED",
        },
        "consumption_protocol": {
            "commit_order": [
                "controller validates exact authority package, sidecar, controller, accepted task, G16, c02, interpreter, evaluator argv, environment, lane, input, output, and namespace bindings",
                "controller validates canonical external credential, exact authority-record bytes, Fresh-L2 acceptance, and separately supplied irreversible action id",
                "validate every bound path is plain, every bound input is a regular non-symlink, and every create-only output is absent",
                "durably consume the validated credential by unlink plus authority-directory fsync; any failure authorizes no evaluator invocation",
                "create and fully write the exact CONSUMED ledger in a private staging file",
                "fsync the complete ledger staging file",
                "link the ledger create-only at its final path and remove staging",
                "fsync the ledger directory; only this durable completion establishes CONSUMED",
                "open the sealed tensor bundle only after durable CONSUMED",
                "validate complete bundle framing and every complete selected record schema and identity",
                "scan every BF16 word in all three complete selected records and reject NaN or infinity",
                "import the accepted G16 reference only after the complete nonfinite scan",
                "evaluate Base once and publish a self-checksummed result create-only",
                "controller validates the complete nested result schema and all authority/result bindings, classifies visible-ledger and no-visible-ledger evaluator returns through the terminal classifier, then publishes the self-checksummed first-terminal record create-only",
            ],
            "controller_required": True,
            "credential_consumption_before_evaluator": True,
            "durable_transition": "READY_UNCONSUMED -> CONSUMED",
            "success_implies_complete_valid_ledger": True,
            "tensor_access_before_consumed_permitted": False,
        },
        "failure_semantics": {
            "crash_after_consumption": "The durably consumed Base lane becomes an orphan. No result or terminal may be reconstructed, and authority is never restored.",
            "credential_substitution_permitted": False,
            "fsync_or_short_write_before_durable_consumption": "Remove staging/final ledger bytes where possible, publish no success, permanently invalidate the credential, and perform no tensor access.",
            "ledger_rollback_ambiguity": "If final-ledger unlink fails or rollback directory fsync fails after a ledger directory-fsync error, the ledger is surviving or durability-ambiguous. Treat the lane as a permanent consumed orphan, perform no tensor access, and permit no retry, replay, resume, repair, replacement, or credential substitution.",
            "no_visible_ledger_after_evaluator": "The accepted evaluator cannot distinguish a durably rolled-back pre-consume failure from rollback-directory-fsync ambiguity in its terminal rejection. After credential consumption and evaluator return, no valid result plus no visible valid ledger is therefore permanently and conservatively classified as CONSUMED_ORPHAN with explicit no-visible-ledger ambiguity evidence.",
            "permanently_prohibited": [
                "retry",
                "replay",
                "resume",
                "repair",
                "replacement",
                "credential substitution",
                "authority substitution",
                "result substitution",
                "c02 replay",
            ],
            "post_consume_failure": "Publish at most one create-only failed result and one create-only first-terminal record if the live process remains able; otherwise leave a permanent consumed orphan.",
            "pre_consume_failure": "Perform zero tensor access, model load, sealed-data evaluation, result publication, or stage transition.",
        },
        "credential_schema": {
            "additional_properties": False,
            "authority_package_binding": "required_exact_runtime_sha256",
            "canonical_json_required": True,
            "irreversible_action_id_pattern": ACTION_ID_PATTERN,
            "required_decision": REQUIRED_DECISION,
            "required_fields": [
                "authority_controller_sha256",
                "authority_package_sha256",
                "authority_record_sha256",
                "credential_sha256",
                "decision",
                "fresh_l2_acceptance_path",
                "fresh_l2_acceptance_sha256",
                "irreversible_action_id",
                "lane_label",
                "schema_id",
            ],
            "schema_id": CREDENTIAL_SCHEMA_ID,
            "self_checksum": "credential_sha256 is SHA-256 of canonical JSON excluding credential_sha256",
            "single_use": True,
        },
        "freeze_state": "STATIC_FROZEN_BASE_ONLY_NO_LIVE_AUTHORITY_OR_CREDENTIAL",
        "namespaces": {
            "authority": {"create_only": True, "path": AUTHORITY_REL},
            "credential": {"create_only": True, "path": CREDENTIAL_REL},
            "first_terminal": {"create_only": True, "path": FIRST_TERMINAL_REL},
            "ledger": {"create_only": True, "path": LEDGER_REL},
            "result": {"create_only": True, "path": RESULT_REL},
            "roots": {
                "authority": AUTHORITY_ROOT_REL,
                "authority_lane": AUTHORITY_LANE_ROOT_REL,
                "result": RESULT_ROOT_REL,
            },
        },
        "preserved_dependencies": {
            "accepted_g16": contract["accepted_g16"],
            "c02": contract["preserved_c02_static_bindings"],
            "no_execution_proof_bindings": proof["immutable_bindings"],
        },
        "publication_protocol": {
            "first_terminal_record": {
                "additional_properties": False,
                "create_only": True,
                "required_fields": [
                    "authority_consumed",
                    "authority_controller_sha256",
                    "authority_package_sha256",
                    "authority_record_sha256",
                    "consumed_ledger_sha256",
                    "consumption_evidence",
                    "credential_sha256",
                    "evaluator_returncode",
                    "execution_started",
                    "fresh_l2_acceptance_sha256",
                    "invocation_sha256",
                    "irreversible_action_id",
                    "lane_label",
                    "orphaned_after_consumption",
                    "reason_code",
                    "record_sha256",
                    "result_path",
                    "result_sha256",
                    "retry_replay_resume_repair_permitted",
                    "schema_id",
                    "status",
                ],
                "consumption_evidence_values": [VISIBLE_VALID_LEDGER, NO_VISIBLE_LEDGER_AMBIGUOUS],
                "schema_id": FIRST_TERMINAL_SCHEMA_ID,
                "self_checksum": "record_sha256 is SHA-256 of canonical JSON excluding record_sha256",
                "write_mode": "atomic create-if-absent with complete write, file fsync, link, and directory fsync",
            },
            "immutability": "The first durable result and first-terminal records are final and may never be deleted, amended, replaced, or superseded.",
            "result_record": {
                "create_only": True,
                "result_schema_id": g16_package["result_schema_id"],
                "self_checksum_field": "result_sha256",
                "write_mode": "accepted evaluator create-only publication",
            },
            "terminal_states": ["SUCCEEDED_TERMINAL", "FAILED_TERMINAL", "CONSUMED_ORPHAN"],
        },
        "schema_version": 5,
        "schemas": schema_bindings,
        "static_verification": {
            "coverage": [
                "authority cardinality and Base-only lane closure",
                "accepted-task and dependency gating",
                "historical noncanonical accepted JSON semantic loading with exact-byte identity preserved",
                "duplicate-key, malformed, nonfinite, schema, checksum, and substitution rejection",
                "isolated V5 action-root projection and V1-V4 plus consumed V3 immutability",
                "source-derived complete ordered producer/accepted-reader/candidate operation and digest sequences",
                "valid 25-record and boundary 1/256 cross-parser fixtures with zero/257 rejection",
                "field-width boundaries, every framing truncation, malformed UTF-8, trailing bytes, and alternative-layout rejection",
                "executable projected/source sealed-payload guard rejection and guard-weakening mutations",
                "full-SHA Fresh-L2 review acceptance and abbreviated, substituted, or misbound review rejection",
                "all bound identity and path substitutions",
                "leaf and ancestor symlinks for every bound input and output class",
                "READY_UNCONSUMED to durable CONSUMED ordering before payload access",
                "short writes, interrupted writes, zero writes, link failures, and file/directory fsync failures",
                "nested selected-record and result-schema missing/extra/type/nonfinite mutations",
                "rollback final-unlink failure and rollback-directory-fsync ambiguity",
                "coupled persistence-fault outcomes through the controller terminal classifier",
                "create-only namespace preexistence",
                "controller credential/package/review/action validation and first-terminal publication surface",
                "self-checksummed terminal and result binding",
                "permanent no retry/replay/resume/repair/replacement/substitution policy",
                "zero live authority, payload, evaluator, c02, RTL, accelerator, or stage activity",
            ],
            "sealed_payload_policy": "The static verifier may lstat the tensor-bundle path but must never open, read, hash, mmap, deserialize, or import it.",
            "test_mode": "deterministic synthetic-only mutation and fault model",
        },
        "static_verifier": {
            "argv": VERIFIER_ARGV,
            "byte_count": VERIFIER_PATH.stat().st_size,
            "environment": ENVIRONMENT,
            "implementation": "CPython",
            "interpreter_path": str(INTERPRETER_PATH),
            "interpreter_sha256": INTERPRETER_SHA256,
            "path": str(VERIFIER_PATH.relative_to(ROOT)),
            "python_version": "3.13.5",
            "sha256": sha256_file(VERIFIER_PATH),
        },
        "v3_terminal_immutability": {
            "action_id": "33383f584a02",
            "authority_sha256": V3_FROZEN_BINDINGS[
                V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority.json"
            ],
            "ledger_sha256": V3_FROZEN_BINDINGS[
                V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority-ledger.json"
            ],
            "no_replay": True,
            "result_sha256": V3_FROZEN_BINDINGS[
                V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1/base/result.json"
            ],
            "review": {**binding(V3_REVIEW_PATH, V3_REVIEW_SHA256), "status": "done"},
            "terminal_reason_code": "INPUT_SCHEMA_REJECTED",
            "terminal_sha256": V3_FROZEN_BINDINGS[
                V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/first-terminal.json"
            ],
            "terminal_status": "FAILED_TERMINAL",
        },
        "v5_compatibility_runtime": {
            **compatibility_runtime_bindings(),
            "authoritative_sources": {
                "accepted_c02_reader": repo_binding(ACCEPTED_C02_READER_REL, ACCEPTED_C02_READER_SHA256),
                "frozen_c02_producer": binding(PRODUCER_PATH, PRODUCER_SHA256),
                "rejected_v4_review": binding(REJECTED_V4_REVIEW_PATH, REJECTED_V4_REVIEW_SHA256),
            },
            "framing": {
                "count_field": "big-endian >I",
                "dimension_fields": "big-endian >Q",
                "dtype_length_field": "unsigned byte >B",
                "magic": "ACE2-C02-TENSORS-V1\n",
                "name_length_field": "big-endian >H",
                "payload_length_field": "big-endian >Q",
                "rank_field": "unsigned byte >B",
                "record_checksum_semantics": "ascii dtype + NUL + >I rank + >Q dimensions + payload",
                "trailing_bytes": "reject after exact EOF check",
            },
        },
    }


def verify_package(package: Any) -> None:
    reject_nonfinite(package)
    require(package == expected_package(), "authority package semantic mismatch")


def expect_reject(function: Callable[..., Any], *args: Any) -> None:
    try:
        function(*args)
    except (VerificationError, KeyError, ValueError, TypeError):
        return
    raise VerificationError("mutation unexpectedly accepted")


def validate_sha256_guard_source(source: str) -> None:
    tree = ast.parse(source)
    function = _function_node(tree, "sha256_file")

    class ForbiddenPayloadOpen(RuntimeError):
        pass

    class GuardProbePath:
        def __init__(self, parts: tuple[str, ...]):
            self.parts = parts

        def __truediv__(self, value: str) -> GuardProbePath:
            return GuardProbePath((*self.parts, value))

        def __hash__(self) -> int:
            return hash(self.parts)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, GuardProbePath) and self.parts == other.parts

        def resolve(self) -> GuardProbePath:
            return self

        def open(self, *_: Any, **__: Any) -> Any:
            raise ForbiddenPayloadOpen("forbidden payload open attempted")

    projected_root = GuardProbePath(("projected",))
    source_root = GuardProbePath(("source",))
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                copy.deepcopy(function),
            ],
            type_ignores=[],
        )
    )
    namespace: dict[str, Any] = {
        "FORBIDDEN_PAYLOAD_REL": FORBIDDEN_PAYLOAD_REL,
        "ROOT": projected_root,
        "SOURCE_REPOSITORY_ROOT": source_root,
        "hashlib": hashlib,
        "require": require,
    }
    exec(compile(module, "<sha256-guard-mutation>", "exec"), namespace, namespace)
    isolated_sha256_file = namespace["sha256_file"]
    for probe in (
        projected_root / FORBIDDEN_PAYLOAD_REL,
        source_root / FORBIDDEN_PAYLOAD_REL,
    ):
        try:
            isolated_sha256_file(probe)
        except VerificationError as exc:
            require(str(exc) == "sealed tensor payload open", "guard rejection reason")
        except ForbiddenPayloadOpen as exc:
            raise VerificationError("weakened guard reached forbidden open") from exc
        else:
            raise VerificationError("weakened guard accepted forbidden path")

    assignments = [
        statement
        for statement in function.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "forbidden"
    ]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Set), "guard path set")
    guarded_paths = {ast.unparse(element) for element in assignments[0].value.elts}
    require(
        guarded_paths
        == {
            "(ROOT / FORBIDDEN_PAYLOAD_REL).resolve()",
            "(SOURCE_REPOSITORY_ROOT / FORBIDDEN_PAYLOAD_REL).resolve()",
        },
        "complete forbidden payload guard paths",
    )
    expected_compare = ast.dump(
        ast.parse("path.resolve() not in forbidden", mode="eval").body,
        include_attributes=False,
    )
    guard_calls = [
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "require"
        and call.args
        and ast.dump(call.args[0], include_attributes=False) == expected_compare
    ]
    require(len(guard_calls) == 1, "forbidden payload membership guard")
    open_calls = [
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "path"
        and call.func.attr == "open"
    ]
    require(
        len(open_calls) == 1
        and len(open_calls[0].args) == 1
        and ast.unparse(open_calls[0].args[0]) == "'rb'",
        "guarded hash open",
    )
    require(guard_calls[0].lineno < open_calls[0].lineno, "guard before hash open")


def forbidden_payload_guard_checks() -> int:
    verifier_tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    source = ast.unparse(_function_node(verifier_tree, "sha256_file")) + "\n"
    validate_sha256_guard_source(source)
    checks = 1
    for path in (
        ROOT / FORBIDDEN_PAYLOAD_REL,
        SOURCE_REPOSITORY_ROOT / FORBIDDEN_PAYLOAD_REL,
    ):
        expect_reject(sha256_file, path)
        checks += 1

    def replace_once(old: str, new: str, label: str) -> str:
        require(source.count(old) == 1, f"guard mutation anchor {label}")
        return source.replace(old, new, 1)

    mutations = [
        replace_once(
            "    forbidden = {(ROOT / FORBIDDEN_PAYLOAD_REL).resolve(), (SOURCE_REPOSITORY_ROOT / FORBIDDEN_PAYLOAD_REL).resolve()}\n",
            "    forbidden = {(SOURCE_REPOSITORY_ROOT / FORBIDDEN_PAYLOAD_REL).resolve()}\n",
            "projected path removal",
        ),
        replace_once(
            "    forbidden = {(ROOT / FORBIDDEN_PAYLOAD_REL).resolve(), (SOURCE_REPOSITORY_ROOT / FORBIDDEN_PAYLOAD_REL).resolve()}\n",
            "    forbidden = {(ROOT / FORBIDDEN_PAYLOAD_REL).resolve()}\n",
            "source path removal",
        ),
        replace_once(
            "path.resolve() not in forbidden",
            "path.resolve() in forbidden",
            "membership inversion",
        ),
        replace_once(
            "    require(path.resolve() not in forbidden, 'sealed tensor payload open')\n",
            "",
            "guard removal",
        ),
    ]
    for mutated in mutations:
        expect_reject(validate_sha256_guard_source, mutated)
        checks += 1
    return checks


def historical_json_compatibility_checks(package: dict[str, Any]) -> int:
    checks = 0
    semantic = {"alpha": [1, 2, {"ok": True}], "zeta": "historical"}
    with tempfile.TemporaryDirectory(prefix="g16-v5-json-") as temporary:
        root = Path(temporary)
        historical = root / "historical.json"
        historical.write_bytes(b'{\n  "zeta" : "historical",\n  "alpha" : [1,2,{"ok":true}]\n}')
        require(load_static_json(historical) == semantic, "historical noncanonical semantic load")
        checks += 1
        canonical = root / "canonical.json"
        canonical.write_bytes(compact_bytes(semantic))
        require(load_canonical_json(canonical) == semantic, "new V5 canonical load")
        checks += 1
        pretty = root / "pretty.json"
        pretty.write_text(pretty_text(semantic), encoding="ascii")
        expect_reject(load_canonical_json, pretty)
        checks += 1
        for index, raw in enumerate(
            (
                b'{"alpha":1,"alpha":2}',
                b'{"alpha":',
                b'{"alpha":NaN}',
                b'{"alpha":Infinity}',
                b'[] trailing',
            )
        ):
            invalid = root / f"invalid-{index}.json"
            invalid.write_bytes(raw)
            expect_reject(load_static_json, invalid)
            checks += 1
        substituted = root / "substituted.json"
        substituted.write_bytes(b'{"alpha":1}\n')
        expect_reject(binding, substituted, hashlib.sha256(b'{"alpha":2}\n').hexdigest())
        checks += 1
    package_schema = load_canonical_json(ROOT / PACKAGE_SCHEMA_REL)
    validate_json_schema(package, package_schema)
    checks += 1
    package_mutation = copy.deepcopy(package)
    package_mutation["schema_version"] = 3
    expect_reject(validate_json_schema, package_mutation, package_schema)
    checks += 1
    credential_schema = load_canonical_json(ROOT / CREDENTIAL_SCHEMA_REL)
    credential = {
        "authority_controller_sha256": package["authority_controller"]["sha256"],
        "authority_package_sha256": "a" * 64,
        "authority_record_sha256": package["base_lane"]["authority_record"]["canonical_sha256"],
        "credential_sha256": "b" * 64,
        "decision": REQUIRED_DECISION,
        "fresh_l2_acceptance_path": str(AUTHORITY_REVIEW_DIRECTORY / "round-9999.json"),
        "fresh_l2_acceptance_sha256": "c" * 64,
        "irreversible_action_id": "ace2:g16-base:v5:synthetic",
        "lane_label": "Base",
        "schema_id": CREDENTIAL_SCHEMA_ID,
    }
    validate_json_schema(credential, credential_schema)
    checks += 1
    credential["lane_label"] = "checkpoint-176"
    expect_reject(validate_json_schema, credential, credential_schema)
    checks += 1
    terminal_schema = load_canonical_json(ROOT / TERMINAL_SCHEMA_REL)
    terminal = terminal_record(package, "CONSUMED_ORPHAN", None, NO_VISIBLE_LEDGER_AMBIGUOUS)
    validate_json_schema(terminal, terminal_schema)
    checks += 1
    terminal["retry_replay_resume_repair_permitted"] = True
    expect_reject(validate_json_schema, terminal, terminal_schema)
    return checks + 1


def validate_synthetic_review(
    review_path: Path,
    raw: bytes,
    package_sha256: str,
    credential: dict[str, Any],
) -> str:
    require(review_path.parent == AUTHORITY_REVIEW_DIRECTORY, "review directory")
    require(re.fullmatch(r"round-[0-9]{4}\.json", review_path.name) is not None, "review filename")
    review_sha256 = hashlib.sha256(raw).hexdigest()
    require(review_sha256 == credential["fresh_l2_acceptance_sha256"], "credential review checksum")
    require(str(review_path) == credential["fresh_l2_acceptance_path"], "credential review path")
    review = json.loads(
        raw.decode("ascii"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    reject_nonfinite(review)
    require(review.get("kind") == "round_reviewed_handoff", "review kind")
    require(review.get("mission_id") == AUTHORITY_REVIEW_MISSION_ID, "review mission")
    require(review.get("mission_context") == str(AUTHORITY_REVIEW_MISSION_PATH), "review mission context")
    require(review.get("producer_role") == "reviewer", "review role")
    decision = review.get("review")
    require(type(decision) is dict and decision.get("status") == "done", "Fresh-L2 acceptance status")
    require(decision.get("next_action") == "", "Fresh-L2 acceptance next action")
    require(
        reason_contains_full_sha(decision.get("reason"), package_sha256),
        "Fresh-L2 complete package checksum binding",
    )
    return review_sha256


def review_binding_checks(package: dict[str, Any]) -> int:
    package_sha256 = sha256_file(PACKAGE_PATH)
    review_path = AUTHORITY_REVIEW_DIRECTORY / "round-9999.json"
    valid = {
        "checkpoint": {"path": str(AUTHORITY_REVIEW_DIRECTORY / "CHECKPOINT.md")},
        "created_at": 0,
        "kind": "round_reviewed_handoff",
        "mission_context": str(AUTHORITY_REVIEW_MISSION_PATH),
        "mission_id": AUTHORITY_REVIEW_MISSION_ID,
        "producer_role": "reviewer",
        "review": {
            "next_action": "",
            "operator_question": "",
            "reason": f"Fresh-L2 accepts runtime package {package_sha256}. No V5 authority exists.",
            "status": "done",
        },
        "round": 9999,
        "schema_version": 2,
    }

    def encoded(value: dict[str, Any]) -> bytes:
        return (json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")

    valid_raw = encoded(valid)
    credential = {
        "fresh_l2_acceptance_path": str(review_path),
        "fresh_l2_acceptance_sha256": hashlib.sha256(valid_raw).hexdigest(),
    }
    require(validate_synthetic_review(review_path, valid_raw, package_sha256, credential) == credential["fresh_l2_acceptance_sha256"], "full-SHA review acceptance")
    checks = 1

    review_mutations: list[Callable[[dict[str, Any]], None]] = [
        lambda value: value["review"].__setitem__("reason", f"package {package_sha256[:8]}...{package_sha256[-5:]}"),
        lambda value: value["review"].__setitem__("reason", f"package {'0' * 64}"),
        lambda value: value["review"].__setitem__("reason", f"package 0{package_sha256}"),
        lambda value: value.__setitem__("kind", "round_engineered_handoff"),
        lambda value: value.__setitem__("mission_id", "wrong-mission"),
        lambda value: value.__setitem__("mission_context", str(AUTHORITY_REVIEW_DIRECTORY / "wrong.json")),
        lambda value: value.__setitem__("producer_role", "engineer"),
        lambda value: value["review"].__setitem__("status", "continue"),
        lambda value: value["review"].__setitem__("next_action", "execute"),
    ]
    for mutate in review_mutations:
        candidate = copy.deepcopy(valid)
        mutate(candidate)
        raw = encoded(candidate)
        bound = dict(credential)
        bound["fresh_l2_acceptance_sha256"] = hashlib.sha256(raw).hexdigest()
        expect_reject(validate_synthetic_review, review_path, raw, package_sha256, bound)
        checks += 1

    duplicate_raw = valid_raw.replace(
        f'  "mission_id": "{AUTHORITY_REVIEW_MISSION_ID}",'.encode("ascii"),
        f'  "mission_id": "{AUTHORITY_REVIEW_MISSION_ID}",\n  "mission_id": "{AUTHORITY_REVIEW_MISSION_ID}",'.encode("ascii"),
        1,
    )
    duplicate_credential = dict(credential)
    duplicate_credential["fresh_l2_acceptance_sha256"] = hashlib.sha256(duplicate_raw).hexdigest()
    expect_reject(validate_synthetic_review, review_path, duplicate_raw, package_sha256, duplicate_credential)
    checks += 1

    nonfinite_raw = valid_raw.replace(b'"round": 9999', b'"round": NaN', 1)
    nonfinite_credential = dict(credential)
    nonfinite_credential["fresh_l2_acceptance_sha256"] = hashlib.sha256(nonfinite_raw).hexdigest()
    expect_reject(validate_synthetic_review, review_path, nonfinite_raw, package_sha256, nonfinite_credential)
    checks += 1

    wrong_path_credential = dict(credential)
    wrong_path_credential["fresh_l2_acceptance_path"] = str(AUTHORITY_REVIEW_DIRECTORY / "round-9998.json")
    expect_reject(validate_synthetic_review, review_path, valid_raw, package_sha256, wrong_path_credential)
    checks += 1

    wrong_sha_credential = dict(credential)
    wrong_sha_credential["fresh_l2_acceptance_sha256"] = "0" * 64
    expect_reject(validate_synthetic_review, review_path, valid_raw, package_sha256, wrong_sha_credential)
    checks += 1

    expect_reject(validate_synthetic_review, AUTHORITY_REVIEW_DIRECTORY / "review.json", valid_raw, package_sha256, credential)
    checks += 1
    expect_reject(validate_synthetic_review, AUTHORITY_REVIEW_DIRECTORY.parent / "wrong/round-9999.json", valid_raw, package_sha256, credential)
    return checks + 1


def set_nested(value: dict[str, Any], path: tuple[Any, ...], replacement: Any) -> None:
    current: Any = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def mutation_checks(package: dict[str, Any]) -> int:
    checks = 0
    mutations = [
        (("authority_package_id",), "substituted"),
        (("authority_controller", "sha256"), "0" * 64),
        (("authority_controller", "execution_state"), "EXECUTED"),
        (("authority_controller", "evaluator_argv", 7), "build/substituted/authority.json"),
        (("authority_review_requirement", "mission_id"), "wrong-mission"),
        (("authority_review_requirement", "required_status"), "continue"),
        (("base_lane", "authority_cardinality"), 2),
        (("base_lane", "lane_label"), "checkpoint-176"),
        (("checkpoint_176_policy", "authority_cardinality"), 1),
        (("checkpoint_176_policy", "lane_included"), True),
        (("accepted_diagnostic", "fresh_l2_handoff", "mission_id"), "wrong-task"),
        (("accepted_diagnostic", "package", "sha256"), "0" * 64),
        (("accepted_diagnostic", "sidecar", "path"), "reference/substituted.sha256"),
        (("accepted_diagnostic", "artifacts", "contract", "sha256"), "0" * 64),
        (("accepted_diagnostic", "artifacts", "result_schema", "path"), "reference/substituted.json"),
        (("accepted_diagnostic", "artifacts", "reference_evaluator", "sha256"), "0" * 64),
        (("accepted_diagnostic", "artifacts", "static_verifier", "sha256"), "0" * 64),
        (("accepted_diagnostic", "artifacts", "no_execution_proof", "sha256"), "0" * 64),
        (("v5_compatibility_runtime", "evaluator", "sha256"), "0" * 64),
        (("v5_compatibility_runtime", "framing", "rank_fields"), "little-endian <H"),
        (("v3_terminal_immutability", "ledger_sha256"), "0" * 64),
        (("v3_terminal_immutability", "no_replay"), False),
        (("v3_terminal_immutability", "terminal_status"), "SUCCEEDED_TERMINAL"),
        (("preserved_dependencies", "c02", "evaluation_package", "sha256"), "0" * 64),
        (("base_lane", "invocation", "interpreter", "sha256"), "0" * 64),
        (("base_lane", "invocation", "argv", 7), "build/substituted/authority.json"),
        (("base_lane", "invocation", "environment", "TZ"), "US/Pacific"),
        (("base_lane", "sealed_inputs", "metadata", "path"), "evidence/substituted.json"),
        (("base_lane", "sealed_inputs", "tensor_bundle", "sha256"), "0" * 64),
        (("base_lane", "output_identity", "output_path"), "build/substituted/result.json"),
        (("base_lane", "authority_record", "template", "state"), "CONSUMED"),
        (("base_lane", "ledger_record", "template", "state"), "READY_UNCONSUMED"),
        (("namespaces", "credential", "path"), AUTHORITY_REL),
        (("consumption_protocol", "tensor_access_before_consumed_permitted"), True),
        (("consumption_protocol", "success_implies_complete_valid_ledger"), False),
        (("failure_semantics", "credential_substitution_permitted"), True),
        (("credential_schema", "required_decision"), "SUBSTITUTE"),
        (("credential_schema", "authority_package_binding"), "embedded-self-hash"),
        (("publication_protocol", "result_record", "self_checksum_field"), "checksum"),
        (("publication_protocol", "first_terminal_record", "schema_id"), "substituted"),
        (("claim_boundary", "sealed_tensor_payload_accessed"), True),
        (("claim_boundary", "authority_controller_executed"), True),
        (("claim_boundary", "rtl_u280_xrt_hbm2_activity"), True),
        (("static_verification", "sealed_payload_policy"), "payload may be hashed"),
    ]
    for path, replacement in mutations:
        mutation = copy.deepcopy(package)
        set_nested(mutation, path, replacement)
        expect_reject(verify_package, mutation)
        checks += 1
    mutation = copy.deepcopy(package)
    mutation["base_lane"]["extra"] = 1
    expect_reject(verify_package, mutation)
    checks += 1
    mutation = copy.deepcopy(package)
    del mutation["publication_protocol"]["first_terminal_record"]["required_fields"]
    expect_reject(verify_package, mutation)
    checks += 1
    mutation = copy.deepcopy(package)
    mutation["base_lane"]["sealed_inputs"]["metadata"]["byte_count"] = float("nan")
    expect_reject(verify_package, mutation)
    checks += 1
    return checks


TENSOR_OPERATION_SEQUENCE = (
    "magic",
    "count",
    "name_length",
    "name_bytes",
    "dtype_length",
    "dtype_bytes",
    "rank",
    "dimensions",
    "payload_length",
    "payload",
)
READER_OPERATION_SEQUENCE = (*TENSOR_OPERATION_SEQUENCE, "trailing_eof")
DIGEST_OPERATION_SEQUENCE = ("dtype_ascii", "nul", "rank", "dimensions", "payload")


ENCODER_GRAMMAR = {
    "count": (">I", 4),
    "dimension": (">Q", 8),
    "dtype_length": (">B", 1),
    "magic": TENSOR_BUNDLE_MAGIC,
    "name_length": (">H", 2),
    "operation_sequence": TENSOR_OPERATION_SEQUENCE,
    "payload_length": (">Q", 8),
    "rank": (">B", 1),
    "record_digest": ("ascii", b"\0", ">I", ">Q", "payload"),
    "trailing_read": 1,
}


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, f"function cardinality {name}")
    return matches[0]


def _module_bytes_constant(tree: ast.Module, name: str) -> bytes:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            require(isinstance(node.value, ast.Constant) and type(node.value.value) is bytes, f"bytes constant {name}")
            return node.value.value
    raise VerificationError(f"missing bytes constant {name}")


def _struct_formats(node: ast.AST, method: str) -> list[str]:
    formats: list[str] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "struct"
            and child.func.attr == method
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and type(child.args[0].value) is str
        ):
            formats.append(child.args[0].value)
    return formats


def _ordered_calls(node: ast.AST, predicate: Callable[[ast.Call], bool]) -> list[ast.Call]:
    calls = [child for child in ast.walk(node) if isinstance(child, ast.Call) and predicate(child)]
    return sorted(calls, key=lambda child: (child.lineno, child.col_offset))


def _parent_map(node: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}


def _nearest_assignment_target(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.Assign):
            require(len(current.targets) == 1, "reader assignment cardinality")
            return ast.unparse(current.targets[0])
    return ""


def _nearest_struct_unpack_format(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    current = node
    while current in parents:
        current = parents[current]
        if (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and isinstance(current.func.value, ast.Name)
            and current.func.value.id == "struct"
            and current.func.attr == "unpack"
        ):
            require(
                current.args
                and isinstance(current.args[0], ast.Constant)
                and type(current.args[0].value) is str,
                "reader unpack format",
            )
            return current.args[0].value
    return None


def _producer_operation_sequence(writer: ast.FunctionDef) -> tuple[str, ...]:
    calls = _ordered_calls(
        writer,
        lambda call: (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "handle"
            and call.func.attr == "write"
        ),
    )
    operations: list[str] = []
    for call in calls:
        require(len(call.args) == 1 and not call.keywords, "producer write call")
        value = call.args[0]
        if isinstance(value, ast.Name):
            operation = {
                "TENSOR_BUNDLE_MAGIC": "magic",
                "name_raw": "name_bytes",
                "dtype_raw": "dtype_bytes",
                "payload": "payload",
            }.get(value.id)
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "struct"
            and value.func.attr == "pack"
            and len(value.args) == 2
            and isinstance(value.args[0], ast.Constant)
            and type(value.args[0].value) is str
        ):
            operation = {
                (">I", "len(tensors)"): "count",
                (">H", "len(name_raw)"): "name_length",
                (">B", "len(dtype_raw)"): "dtype_length",
                (">B", "tensor.ndim"): "rank",
                (">Q", "int(dimension)"): "dimensions",
                (">Q", "len(payload)"): "payload_length",
            }.get((value.args[0].value, ast.unparse(value.args[1])))
        else:
            operation = None
        require(operation is not None, f"unknown producer write {ast.unparse(value)}")
        operations.append(operation)
    return tuple(operations)


def _reader_operation_sequence(reader: ast.FunctionDef) -> tuple[str, ...]:
    parents = _parent_map(reader)

    def is_read(call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Name)
            and call.func.id == "read_exact"
            or isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "handle"
            and call.func.attr == "read"
        )

    calls = _ordered_calls(reader, is_read)
    operations: list[str] = []
    signatures = {
        ("", "len(TENSOR_BUNDLE_MAGIC)", None): "magic",
        ("count", "4", ">I"): "count",
        ("name_size", "2", ">H"): "name_length",
        ("name", "name_size", None): "name_bytes",
        ("dtype_size", "1", None): "dtype_length",
        ("dtype", "dtype_size", None): "dtype_bytes",
        ("rank", "1", None): "rank",
        ("shape", "8", ">Q"): "dimensions",
        ("payload_size", "8", ">Q"): "payload_length",
        ("payload", "payload_size", None): "payload",
    }
    for call in calls:
        target = _nearest_assignment_target(call, parents)
        if isinstance(call.func, ast.Name):
            require(
                len(call.args) == 2
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "handle"
                and not call.keywords,
                "reader read_exact call",
            )
            signature = (
                target,
                ast.unparse(call.args[1]),
                _nearest_struct_unpack_format(call, parents),
            )
            operation = signatures.get(signature)
        else:
            require(
                target == ""
                and len(call.args) == 1
                and ast.unparse(call.args[0]) == "1"
                and not call.keywords,
                "reader trailing read",
            )
            operation = "trailing_eof"
        require(operation is not None, f"unknown reader operation {ast.unparse(call)}")
        operations.append(operation)
    return tuple(operations)


def _digest_operation_sequence(node: ast.FunctionDef, function_name: str) -> tuple[str, ...]:
    calls = _ordered_calls(
        node,
        lambda call: (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "digest"
            and call.func.attr == "update"
        ),
    )
    operations: list[str] = []
    for call in calls:
        require(len(call.args) == 1 and not call.keywords, f"{function_name} digest update")
        value = call.args[0]
        rendered = ast.unparse(value)
        if ".encode('ascii')" in rendered or '.encode("ascii")' in rendered:
            operation = "dtype_ascii"
        elif isinstance(value, ast.Constant) and value.value == b"\0":
            operation = "nul"
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "struct"
            and value.func.attr == "pack"
            and value.args
            and isinstance(value.args[0], ast.Constant)
        ):
            operation = {">I": "rank", ">Q": "dimensions"}.get(value.args[0].value)
        elif rendered in {"payload", "payload_tensor.numpy().tobytes(order='C')"}:
            operation = "payload"
        else:
            operation = None
        require(operation is not None, f"unknown {function_name} digest update {rendered}")
        operations.append(operation)
    return tuple(operations)


def _digest_grammar(tree: ast.Module, function_name: str) -> tuple[Any, ...]:
    node = _function_node(tree, function_name)
    source = ast.unparse(node)
    require('.encode("ascii")' in source or ".encode('ascii')" in source, f"{function_name} ascii dtype")
    require("b'\\x00'" in source or 'b"\\x00"' in source or "b'\\0'" in source, f"{function_name} NUL separator")
    require(_struct_formats(node, "pack") == [">I", ">Q"], f"{function_name} digest struct sequence")
    payload_terminal = (
        "digest.update(payload_tensor.numpy().tobytes(order='C'))"
        if function_name == "tensor_sha256"
        else "digest.update(payload)"
    )
    require(payload_terminal in source, f"{function_name} payload digest")
    require("len(payload)" not in source, f"{function_name} excludes payload length")
    require(
        _digest_operation_sequence(node, function_name) == DIGEST_OPERATION_SEQUENCE,
        f"{function_name} digest operation sequence",
    )
    return ("ascii", b"\0", ">I", ">Q", "payload")


def extract_producer_grammar(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    writer = _function_node(tree, "write_tensor_bundle")
    require(_struct_formats(writer, "pack") == [">I", ">H", ">B", ">B", ">Q", ">Q"], "producer pack sequence")
    require(
        _producer_operation_sequence(writer) == TENSOR_OPERATION_SEQUENCE,
        "producer complete write operation sequence",
    )
    return {
        **ENCODER_GRAMMAR,
        "magic": _module_bytes_constant(tree, "TENSOR_BUNDLE_MAGIC"),
        "record_digest": _digest_grammar(tree, "tensor_sha256"),
    }


def extract_reader_grammar(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    reader = _function_node(tree, "read_tensor_bundle")
    reader_source = ast.unparse(reader)
    require(_struct_formats(reader, "unpack") == [">I", ">H", ">Q", ">Q"], "reader unpack sequence")
    require(reader_source.count("read_exact(handle, 1)[0]") == 2, "reader byte dtype/rank fields")
    require("handle.read(1) == b''" in reader_source, "reader exact EOF check")
    require(
        _reader_operation_sequence(reader) == READER_OPERATION_SEQUENCE,
        "reader complete read operation sequence",
    )
    return {
        **ENCODER_GRAMMAR,
        "magic": _module_bytes_constant(tree, "TENSOR_BUNDLE_MAGIC"),
        "record_digest": _digest_grammar(tree, "tensor_record_sha256"),
    }


def authoritative_grammar_checks(
    producer_source: str, accepted_reader_source: str, candidate_source: str
) -> dict[str, Any]:
    producer = extract_producer_grammar(producer_source)
    accepted = extract_reader_grammar(accepted_reader_source)
    candidate = extract_reader_grammar(candidate_source)
    require(producer == ENCODER_GRAMMAR, "producer authoritative grammar")
    require(accepted == producer, "accepted reader matches producer grammar")
    require(candidate == producer, "candidate reader matches authoritative grammar")
    return producer


def synthetic_tensor_record_sha256(dtype: str, shape: list[int] | tuple[int, ...], payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack(">I", len(shape)))
    for dimension in shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(payload)
    return digest.hexdigest()


def synthetic_bundle_bytes(
    records: list[tuple[str, str, tuple[int, ...], bytes]],
    *,
    count_override: int | None = None,
    layout: str = "authoritative",
) -> bytes:
    require(layout in {"authoritative", "little", "v4"}, "synthetic layout")
    count_prefix = "<" if layout == "little" else ">"
    payload = bytearray(TENSOR_BUNDLE_MAGIC)
    payload.extend(struct.pack(count_prefix + "I", len(records) if count_override is None else count_override))
    for name, dtype, shape, record_payload in records:
        name_raw = name.encode("utf-8")
        dtype_raw = dtype.encode("ascii")
        if layout == "v4":
            payload.extend(struct.pack(">I", len(name_raw)))
        else:
            payload.extend(struct.pack(("<" if layout == "little" else ">") + "H", len(name_raw)))
        payload.extend(name_raw)
        if layout == "v4":
            payload.extend(struct.pack(">I", len(dtype_raw)))
        else:
            payload.extend(struct.pack(">B", len(dtype_raw)))
        payload.extend(dtype_raw)
        if layout == "v4":
            payload.extend(struct.pack(">H", len(shape)))
        else:
            payload.extend(struct.pack(">B", len(shape)))
        for dimension in shape:
            payload.extend(struct.pack(("<" if layout == "little" else ">") + "Q", dimension))
        payload.extend(struct.pack(("<" if layout == "little" else ">") + "Q", len(record_payload)))
        payload.extend(record_payload)
    return bytes(payload)


def _reader_namespace(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    nodes = [copy.deepcopy(_function_node(tree, name)) for name in ("tensor_record_sha256", "read_exact", "read_tensor_bundle")]
    module = ast.fix_missing_locations(
        ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
    )

    def synthetic_require(condition: bool, message: str, *_: Any) -> None:
        require(condition, message)

    namespace: dict[str, Any] = {
        "__builtins__": {"__import__": __import__, "all": all, "len": len, "range": range, "tuple": tuple},
        "hashlib": hashlib,
        "struct": struct,
        "TENSOR_BUNDLE_MAGIC": TENSOR_BUNDLE_MAGIC,
        "require": synthetic_require,
    }
    exec(compile(module, "<synthetic-reader>", "exec"), namespace, namespace)
    return namespace


def _parse_synthetic(namespace: dict[str, Any], payload: bytes) -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="g16-v5-c02-") as temporary:
        root = Path(temporary).resolve()
        path = root / "fixture.bin"
        path.write_bytes(payload)

        def guard(candidate: Path) -> None:
            require(candidate.resolve() == path, "synthetic reader path guard")

        namespace["verify_plain_file"] = guard
        namespace["verify_plain_path"] = guard
        return namespace["read_tensor_bundle"](path)


def _expect_parse_reject(namespace: dict[str, Any], payload: bytes) -> None:
    try:
        _parse_synthetic(namespace, payload)
    except (VerificationError, UnicodeDecodeError, UnicodeError, ValueError, struct.error):
        return
    raise VerificationError("malformed synthetic fixture unexpectedly accepted")


def _compare_parser_records(accepted: dict[str, Any], candidate: dict[str, Any]) -> None:
    require(candidate == accepted, "accepted/candidate parser divergence")


def _cross_parse(
    accepted_namespace: dict[str, Any], candidate_namespace: dict[str, Any], payload: bytes
) -> dict[str, dict[str, Any]]:
    accepted = _parse_synthetic(accepted_namespace, payload)
    candidate = _parse_synthetic(candidate_namespace, payload)
    _compare_parser_records(accepted, candidate)
    return accepted


def _framing_boundaries(record: tuple[str, str, tuple[int, ...], bytes]) -> list[int]:
    name, dtype, shape, record_payload = record
    cursor = len(TENSOR_BUNDLE_MAGIC)
    boundaries = [cursor]
    cursor += 4
    boundaries.append(cursor)
    cursor += 2
    boundaries.append(cursor)
    cursor += len(name.encode("utf-8"))
    boundaries.append(cursor)
    cursor += 1
    boundaries.append(cursor)
    cursor += len(dtype.encode("ascii"))
    boundaries.append(cursor)
    cursor += 1
    boundaries.append(cursor)
    for _ in shape:
        cursor += 8
        boundaries.append(cursor)
    cursor += 8
    boundaries.append(cursor)
    cursor += len(record_payload)
    boundaries.append(cursor)
    return boundaries


def framing_compatibility_checks() -> int:
    producer_source = PRODUCER_PATH.read_text(encoding="utf-8")
    accepted_source = ACCEPTED_C02_READER_PATH.read_text(encoding="utf-8")
    candidate_source = (ROOT / G16_EVALUATOR_REL).read_text(encoding="utf-8")
    authoritative_grammar_checks(producer_source, accepted_source, candidate_source)
    checks = 3
    accepted_namespace = _reader_namespace(accepted_source)
    candidate_namespace = _reader_namespace(candidate_source)

    def record(index: int, *, shape: tuple[int, ...] = (1,), payload: bytes | None = None) -> tuple[str, str, tuple[int, ...], bytes]:
        return (
            f"synthetic.record.{index:03d}",
            "torch.bfloat16",
            shape,
            b"\x80\x3f" * math.prod(shape) if payload is None else payload,
        )

    for count in (1, 25, 256):
        records = [record(index) for index in range(count)]
        parsed = _cross_parse(accepted_namespace, candidate_namespace, synthetic_bundle_bytes(records))
        require(len(parsed) == count, f"valid authoritative record count {count}")
        for name, item in parsed.items():
            require(
                item["sha256"] == synthetic_tensor_record_sha256(item["dtype"], item["shape"], item["payload"]),
                f"producer-compatible digest {name}",
            )
        checks += 1
    for count in (0, 257):
        payload = synthetic_bundle_bytes([], count_override=count)
        _expect_parse_reject(accepted_namespace, payload)
        _expect_parse_reject(candidate_namespace, payload)
        checks += 1

    boundary_records = [
        ("n", "d", (1,), b""),
        ("n" * 1024, "d" * 64, (1, 1 << 20, 1, 1, 1, 1, 1, 1), b""),
    ]
    for fixture in boundary_records:
        _cross_parse(accepted_namespace, candidate_namespace, synthetic_bundle_bytes([fixture]))
        checks += 1

    first = record(0, shape=(1, 2))
    valid = synthetic_bundle_bytes([first])
    for boundary in _framing_boundaries(first):
        _expect_parse_reject(accepted_namespace, valid[: max(0, boundary - 1)])
        _expect_parse_reject(candidate_namespace, valid[: max(0, boundary - 1)])
        checks += 1
    for malformed in (valid + b"\x00", synthetic_bundle_bytes([record(0), record(0)])):
        _expect_parse_reject(accepted_namespace, malformed)
        _expect_parse_reject(candidate_namespace, malformed)
        checks += 1

    magic_end = len(TENSOR_BUNDLE_MAGIC)
    malformed_utf8 = bytearray(synthetic_bundle_bytes([("n", "d", (1,), b"")]))
    malformed_utf8[magic_end + 4 + 2] = 0xFF
    for namespace in (accepted_namespace, candidate_namespace):
        _expect_parse_reject(namespace, bytes(malformed_utf8))
    checks += 1

    base = bytearray(synthetic_bundle_bytes([("n", "d", (1,), b"")]))
    name_length_offset = magic_end + 4
    dtype_length_offset = name_length_offset + 2 + 1
    rank_offset = dtype_length_offset + 1 + 1
    dimension_offset = rank_offset + 1
    payload_length_offset = dimension_offset + 8
    field_mutations = [
        (name_length_offset, 2, struct.pack(">H", 0)),
        (name_length_offset, 2, struct.pack(">H", 1025)),
        (dtype_length_offset, 1, b"\x00"),
        (dtype_length_offset, 1, b"A"),
        (rank_offset, 1, b"\x00"),
        (rank_offset, 1, b"\x09"),
        (dimension_offset, 8, struct.pack(">Q", 0)),
        (dimension_offset, 8, struct.pack(">Q", (1 << 20) + 1)),
        (payload_length_offset, 8, struct.pack(">Q", (1 << 31) + 1)),
        (payload_length_offset, 8, struct.pack(">Q", 1)),
    ]
    for offset, size, replacement in field_mutations:
        mutated = bytes(base[:offset] + replacement + base[offset + size :])
        _expect_parse_reject(accepted_namespace, mutated)
        _expect_parse_reject(candidate_namespace, mutated)
        checks += 1

    for layout in ("little", "v4"):
        alternative = synthetic_bundle_bytes([first], layout=layout)
        _expect_parse_reject(accepted_namespace, alternative)
        _expect_parse_reject(candidate_namespace, alternative)
        checks += 1

    expect_reject(binding, PRODUCER_PATH, "0" * 64)
    expect_reject(binding, ACCEPTED_C02_READER_PATH, "0" * 64)
    expect_reject(
        authoritative_grammar_checks,
        producer_source.replace('struct.pack(">H", len(name_raw))', 'struct.pack(">I", len(name_raw))', 1),
        accepted_source,
        candidate_source,
    )
    expect_reject(
        authoritative_grammar_checks,
        producer_source,
        accepted_source.replace('struct.unpack(">H", read_exact(handle, 2))[0]', 'struct.unpack(">I", read_exact(handle, 4))[0]', 1),
        candidate_source,
    )
    expect_reject(
        authoritative_grammar_checks,
        producer_source,
        accepted_source,
        candidate_source.replace('rank = read_exact(handle, 1)[0]', 'rank = struct.unpack(">H", read_exact(handle, 2))[0]', 1),
    )
    producer_order = (
        '            handle.write(struct.pack(">H", len(name_raw)))\n'
        "            handle.write(name_raw)\n"
        '            handle.write(struct.pack(">B", len(dtype_raw)))'
    )
    producer_reordered = (
        '            handle.write(struct.pack(">H", len(name_raw)))\n'
        '            handle.write(struct.pack(">B", len(dtype_raw)))\n'
        "            handle.write(name_raw)"
    )
    require(producer_source.count(producer_order) == 1, "producer reorder mutation anchor")
    expect_reject(
        authoritative_grammar_checks,
        producer_source.replace(producer_order, producer_reordered, 1),
        accepted_source,
        candidate_source,
    )
    reader_order = (
        '            name = read_exact(handle, name_size).decode("utf-8")\n'
        '            require(name not in records, "duplicate tensor name")\n'
        "            dtype_size = read_exact(handle, 1)[0]"
    )
    reader_reordered = (
        "            dtype_size = read_exact(handle, 1)[0]\n"
        '            name = read_exact(handle, name_size).decode("utf-8")\n'
        '            require(name not in records, "duplicate tensor name")'
    )
    require(accepted_source.count(reader_order) == 1, "accepted reader reorder mutation anchor")
    expect_reject(
        authoritative_grammar_checks,
        producer_source,
        accepted_source.replace(reader_order, reader_reordered, 1),
        candidate_source,
    )
    require(candidate_source.count(reader_order) == 1, "candidate reader reorder mutation anchor")
    expect_reject(
        authoritative_grammar_checks,
        producer_source,
        accepted_source,
        candidate_source.replace(reader_order, reader_reordered, 1),
    )
    parsed = _cross_parse(accepted_namespace, candidate_namespace, valid)
    divergent = copy.deepcopy(parsed)
    divergent[first[0]]["sha256"] = "0" * 64
    expect_reject(_compare_parser_records, parsed, divergent)
    checks += 9
    return checks


def selected_record_fixture() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for index, role in enumerate(SELECTED_ROLES):
        name = f"synthetic.{role}"
        shape = [1, 2]
        payload = struct.pack("<HH", 0x3F80 + index, 0xBF80 - index)
        checksum = synthetic_tensor_record_sha256("torch.bfloat16", shape, payload)
        records[name] = {
            "dtype": "torch.bfloat16",
            "payload": payload,
            "sha256": checksum,
            "shape": shape,
        }
        bindings[role] = {
            "dtype": "torch.bfloat16",
            "sha256": checksum,
            "shape": shape,
            "tensor_name": name,
        }
    return records, bindings


def validate_synthetic_selected_records(
    records: dict[str, dict[str, Any]], bindings: dict[str, dict[str, Any]]
) -> None:
    require(type(records) is dict and type(bindings) is dict, "selected containers")
    require(set(bindings) == set(SELECTED_ROLES), "selected role set")
    require(set(records) == {bindings[role]["tensor_name"] for role in SELECTED_ROLES}, "selected record set")
    for role in SELECTED_ROLES:
        binding = bindings[role]
        require(
            type(binding) is dict and set(binding) == {"dtype", "sha256", "shape", "tensor_name"},
            f"selected binding schema {role}",
        )
        require(binding["dtype"] == "torch.bfloat16", f"selected binding dtype {role}")
        require(valid_sha256(binding["sha256"]), f"selected binding checksum {role}")
        require(
            type(binding["shape"]) is list
            and len(binding["shape"]) > 0
            and all(type(dimension) is int and dimension > 0 for dimension in binding["shape"]),
            f"selected binding shape {role}",
        )
        require(type(binding["tensor_name"]) is str and binding["tensor_name"] != "", f"selected tensor name {role}")
        record = records[binding["tensor_name"]]
        require(
            type(record) is dict and set(record) == {"dtype", "payload", "sha256", "shape"},
            f"selected record schema {role}",
        )
        require(record["dtype"] == binding["dtype"], f"selected dtype {role}")
        require(record["shape"] == binding["shape"], f"selected shape {role}")
        require(record["sha256"] == binding["sha256"], f"selected checksum binding {role}")
        require(type(record["payload"]) is bytes, f"selected payload type {role}")
        require(len(record["payload"]) == 2 * math.prod(record["shape"]), f"selected payload size {role}")
        require(
            synthetic_tensor_record_sha256(record["dtype"], record["shape"], record["payload"])
            == record["sha256"],
            f"selected framed checksum {role}",
        )
        for word_index in range(len(record["payload"]) // 2):
            word = struct.unpack_from("<H", record["payload"], word_index * 2)[0]
            require(((word >> 7) & 0xFF) != 0xFF, f"selected nonfinite {role}")


def replace_selected_payload(
    records: dict[str, dict[str, Any]], bindings: dict[str, dict[str, Any]], role: str, payload: bytes
) -> None:
    binding = bindings[role]
    record = records[binding["tensor_name"]]
    checksum = synthetic_tensor_record_sha256(record["dtype"], record["shape"], payload)
    record["payload"] = payload
    record["sha256"] = checksum
    binding["sha256"] = checksum


def selected_record_schema_checks() -> int:
    records, bindings = selected_record_fixture()
    validate_synthetic_selected_records(records, bindings)
    checks = 1
    mutations: list[tuple[str, Callable[[dict[str, Any], dict[str, Any]], None]]] = [
        ("missing-role", lambda r, b: b.pop(SELECTED_ROLES[0])),
        ("extra-role", lambda r, b: b.__setitem__("extra", copy.deepcopy(b[SELECTED_ROLES[0]]))),
        ("missing-binding-field", lambda r, b: b[SELECTED_ROLES[0]].pop("tensor_name")),
        ("extra-binding-field", lambda r, b: b[SELECTED_ROLES[0]].__setitem__("extra", 1)),
        ("binding-shape-type", lambda r, b: b[SELECTED_ROLES[0]].__setitem__("shape", [True, 2])),
        ("binding-checksum", lambda r, b: b[SELECTED_ROLES[0]].__setitem__("sha256", "0" * 64)),
        ("missing-record", lambda r, b: r.pop(b[SELECTED_ROLES[0]]["tensor_name"])),
        ("extra-record", lambda r, b: r.__setitem__("synthetic.extra", copy.deepcopy(next(iter(r.values()))))),
        ("missing-record-field", lambda r, b: r[b[SELECTED_ROLES[0]]["tensor_name"]].pop("payload")),
        ("extra-record-field", lambda r, b: r[b[SELECTED_ROLES[0]]["tensor_name"]].__setitem__("extra", 1)),
        ("record-dtype", lambda r, b: r[b[SELECTED_ROLES[0]]["tensor_name"]].__setitem__("dtype", "torch.float16")),
        ("record-shape", lambda r, b: r[b[SELECTED_ROLES[0]]["tensor_name"]].__setitem__("shape", [2, 1])),
        ("payload-type", lambda r, b: r[b[SELECTED_ROLES[0]]["tensor_name"]].__setitem__("payload", bytearray(4))),
        ("payload-size", lambda r, b: r[b[SELECTED_ROLES[0]]["tensor_name"]].__setitem__("payload", b"\x00\x00")),
        ("framed-checksum", lambda r, b: r[b[SELECTED_ROLES[0]]["tensor_name"]].__setitem__("payload", b"\x01\x00\x00\x00")),
        ("positive-infinity", lambda r, b: replace_selected_payload(r, b, SELECTED_ROLES[0], b"\x80\x7f\x00\x00")),
        ("negative-nan", lambda r, b: replace_selected_payload(r, b, SELECTED_ROLES[0], b"\xc1\xff\x00\x00")),
    ]
    for _, mutate in mutations:
        mutated_records = copy.deepcopy(records)
        mutated_bindings = copy.deepcopy(bindings)
        mutate(mutated_records, mutated_bindings)
        expect_reject(validate_synthetic_selected_records, mutated_records, mutated_bindings)
        checks += 1
    return checks


def synthetic_result(package: dict[str, Any]) -> dict[str, Any]:
    lane = package["base_lane"]
    fraction = {"denominator": 1, "numerator": 1}
    integer_check = {"actual": 0, "limit": 0, "pass": True}
    fraction_check = {"actual": fraction, "limit": fraction, "pass": True}
    metrics = {
        "invalid_accounting": {
            "cross_lane_record_count": 0,
            "invalid_or_non_finite_value_count": 0,
            "normalization_rejection_count": 0,
            "positive_centered_realized_score_count": 0,
            "saturation_event_count": 0,
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb": 0,
            "oracle_tied_row_count": 0,
            "preserved_positive_margin_fraction": fraction,
            "preserved_positive_margin_row_count": 1,
            "singleton_valid_key_row_count": 0,
            "unique_oracle_top_row_count": 1,
            "violation_count": 0,
        },
        "score_error": {
            "maximum_absolute_error_q12_20_lsb": 0,
            "sum_absolute_error_q12_20_lsb": 0,
            "sum_signed_error_q12_20_lsb": 0,
            "sum_squared_error_q40_40_lsb2": 0,
            "valid_value_count": 1,
        },
        "top_key": {
            "matching_fraction": fraction,
            "matching_row_count": 1,
            "mismatch_count": 0,
            "row_count": 1,
        },
    }
    thresholds = {
        "all_hard_thresholds_pass": True,
        "cross_lane_record_count_maximum": integer_check,
        "invalid_or_non_finite_value_count_maximum": integer_check,
        "normalization_rejection_count_maximum": integer_check,
        "positive_centered_realized_score_count_maximum": integer_check,
        "rank_margin_violation_count_maximum": integer_check,
        "saturation_event_count_maximum": integer_check,
        "top_key_matching_fraction_minimum": fraction_check,
        "top_key_mismatch_count_maximum": integer_check,
        "unique_oracle_positive_margin_preserved_fraction_minimum": fraction_check,
    }
    result = {
        "authority_sha256": lane["authority_record"]["canonical_sha256"],
        "evaluator_sha256": G16_EVALUATOR_SHA256,
        "generation_id": lane["generation_id"],
        "input_bindings": {
            "input_token_ids_sha256": lane["sealed_inputs"]["input_token_ids_sha256"],
            "lane_metadata": lane["sealed_inputs"]["metadata"],
            "sealed_set_id": "w4a8-c02-attention-substage-trace-v2",
            "tensor_bundle": lane["sealed_inputs"]["tensor_bundle"],
            "tensor_records": lane["sealed_inputs"]["authoritative_tensors"],
        },
        "invocation_sha256": BASE_INVOCATION_SHA256,
        "lane_label": "Base",
        "metrics": metrics,
        "model_identity_sha256": lane["model_identity_sha256"],
        "namespace_label": "base",
        "package_id": "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE",
        "package_sha256": G16_PACKAGE_SHA256,
        "schema_id": "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_RESULT_V1",
        "terminal": {
            "first_record_immutable": True,
            "invocation_count_performed": 1,
            "metrics_published": True,
            "reason_code": "HARD_THRESHOLDS_PASSED",
            "retry_replay_resume_repair_permitted": False,
            "status": "SUCCEEDED_TERMINAL",
            "thresholds_evaluated": True,
        },
        "threshold_evaluation": thresholds,
    }
    result["result_sha256"] = hashlib.sha256(compact_bytes(result)).hexdigest()
    return result


def result_schema_checks(package: dict[str, Any]) -> int:
    schema = load_static_json(ROOT / G16_SCHEMA_REL)
    authority_package_sha256 = sha256_file(PACKAGE_PATH)
    result = synthetic_result(package)
    validate_result_record(package, schema, result, authority_package_sha256)
    checks = 1
    mutations: list[tuple[tuple[Any, ...], Any]] = [
        (("metrics", "top_key", "row_count"), True),
        (("metrics", "invalid_accounting", "saturation_event_count"), -1),
        (("metrics", "rank_margin", "preserved_positive_margin_fraction", "denominator"), 0),
        (("metrics", "score_error", "maximum_absolute_error_q12_20_lsb"), "zero"),
        (("input_bindings", "tensor_records", "realized_query_source", "shape", 2), 0),
        (("threshold_evaluation", "cross_lane_record_count_maximum", "actual"), "zero"),
        (("terminal", "status"), "FAILED_TERMINAL"),
        (("terminal", "retry_replay_resume_repair_permitted"), True),
        (("authority_sha256",), "0" * 64),
        (("lane_label",), "checkpoint-176"),
        (("result_sha256",), "0" * 64),
    ]
    for path, replacement in mutations:
        mutation = copy.deepcopy(result)
        set_nested(mutation, path, replacement)
        expect_reject(validate_result_record, package, schema, mutation, authority_package_sha256)
        checks += 1
    mutation = copy.deepcopy(result)
    del mutation["metrics"]["top_key"]["row_count"]
    expect_reject(validate_result_record, package, schema, mutation, authority_package_sha256)
    checks += 1
    mutation = copy.deepcopy(result)
    mutation["metrics"]["top_key"]["extra"] = 0
    expect_reject(validate_result_record, package, schema, mutation, authority_package_sha256)
    checks += 1
    mutation = copy.deepcopy(result)
    mutation["terminal"]["extra"] = False
    expect_reject(validate_result_record, package, schema, mutation, authority_package_sha256)
    checks += 1
    mutation = copy.deepcopy(result)
    mutation["input_bindings"]["tensor_records"]["bf16_oracle_scores"]["sha256"] = float("nan")
    expect_reject(validate_result_record, package, schema, mutation, authority_package_sha256)
    checks += 1
    return checks


def synthetic_path_checks(package: dict[str, Any]) -> int:
    input_labels = [
        "g16-package",
        "g16-sidecar",
        "g16-contract",
        "g16-schema",
        "g16-evaluator",
        "g16-verifier",
        "g16-proof",
        "authority-controller",
        "authority-review-mission",
        "task-mission",
        "task-review",
        "interpreter",
        "metadata",
        "tensor-bundle",
    ]
    output_labels = ["authority", "credential", "ledger", "first-terminal", "result"]
    checks = 0
    with tempfile.TemporaryDirectory(prefix="ace2-g16-authority-paths-") as temporary:
        temp = Path(temporary)
        target = temp / "target"
        target.write_bytes(b"x")
        for label in input_labels:
            plain_parent = temp / label / "plain"
            plain_parent.mkdir(parents=True)
            plain = plain_parent / "artifact"
            plain.write_bytes(b"x")
            verify_plain_regular(plain)
            checks += 1
            leaf_link = plain_parent / "leaf-link"
            leaf_link.symlink_to(target)
            expect_reject(verify_plain_regular, leaf_link)
            checks += 1
            linked_parent = temp / label / "linked-parent"
            linked_parent.symlink_to(plain_parent, target_is_directory=True)
            expect_reject(verify_plain_regular, linked_parent / "artifact")
            checks += 1
        for label in output_labels:
            plain_parent = temp / (label + "-output") / "plain"
            plain_parent.mkdir(parents=True)
            absent = plain_parent / "artifact"
            verify_absent_plain(absent)
            checks += 1
            absent.symlink_to(target)
            expect_reject(verify_absent_plain, absent)
            checks += 1
            absent.unlink()
            linked_parent = temp / (label + "-output") / "linked-parent"
            linked_parent.symlink_to(plain_parent, target_is_directory=True)
            expect_reject(verify_absent_plain, linked_parent / "artifact")
            checks += 1
    require(package["base_lane"]["authority_cardinality"] == 1, "path package cardinality")
    return checks


def validate_trace(events: list[str], terminal_status: str) -> str:
    forbidden = {"retry", "replay", "resume", "repair", "replacement", "c02", "checkpoint-176"}
    require(not forbidden.intersection(events), "prohibited trace event")
    require(events.count("ledger-directory-fsync") <= 1, "duplicate durable consume")
    credential_fsync_index = events.index("credential-directory-fsync") if "credential-directory-fsync" in events else None
    ledger_write_index = events.index("ledger-complete-write") if "ledger-complete-write" in events else None
    consumed = "ledger-directory-fsync" in events
    payload_index = events.index("tensor-payload-open") if "tensor-payload-open" in events else None
    import_index = events.index("g16-import") if "g16-import" in events else None
    scan_index = events.index("complete-record-nonfinite-scan") if "complete-record-nonfinite-scan" in events else None
    schema_index = events.index("complete-selected-record-schema") if "complete-selected-record-schema" in events else None
    if ledger_write_index is not None:
        required_preflight = {
            "controller-package-validation",
            "credential-validation",
            "fresh-l2-review-validation",
            "irreversible-action-validation",
            "credential-directory-fsync",
        }
        require(required_preflight.issubset(events), "ledger before controller preflight")
        require(credential_fsync_index is not None and credential_fsync_index < ledger_write_index, "ledger before credential consumption")
    if payload_index is not None:
        require(consumed, "payload before consumed")
        require(events.index("ledger-directory-fsync") < payload_index, "payload ordering")
    if import_index is not None:
        require(scan_index is not None and scan_index < import_index, "G16 import before full scan")
    if scan_index is not None:
        require(payload_index is not None and payload_index < scan_index, "full scan before payload access")
        require(schema_index is not None and schema_index < scan_index, "full scan before complete schema")
    if terminal_status == "SUCCEEDED_TERMINAL":
        required = {
            "controller-package-validation",
            "credential-validation",
            "fresh-l2-review-validation",
            "irreversible-action-validation",
            "credential-directory-fsync",
            "ledger-complete-write",
            "ledger-file-fsync",
            "ledger-create-only-link",
            "ledger-directory-fsync",
            "tensor-payload-open",
            "complete-selected-record-schema",
            "complete-record-nonfinite-scan",
            "g16-import",
            "sealed-data-evaluation",
            "result-directory-fsync",
            "first-terminal-directory-fsync",
        }
        require(required.issubset(events), "success without complete ledger/result/terminal")
        return "SUCCEEDED_TERMINAL"
    if terminal_status == "CONSUMED_ORPHAN":
        require(consumed, "orphan without consumption")
        require("result-directory-fsync" not in events, "orphan with result")
        return "CONSUMED_ORPHAN"
    require(terminal_status == "FAILED_TERMINAL", "unknown terminal")
    return terminal_status


def ordering_checks() -> int:
    valid = [
        "preflight-complete",
        "controller-package-validation",
        "credential-validation",
        "fresh-l2-review-validation",
        "irreversible-action-validation",
        "credential-directory-fsync",
        "ledger-complete-write",
        "ledger-file-fsync",
        "ledger-create-only-link",
        "ledger-directory-fsync",
        "tensor-payload-open",
        "complete-selected-record-schema",
        "complete-record-nonfinite-scan",
        "g16-import",
        "sealed-data-evaluation",
        "result-directory-fsync",
        "first-terminal-directory-fsync",
    ]
    require(validate_trace(valid, "SUCCEEDED_TERMINAL") == "SUCCEEDED_TERMINAL", "valid trace")
    checks = 1
    invalid = [
        (["tensor-payload-open"], "FAILED_TERMINAL"),
        (["ledger-directory-fsync", "g16-import"], "FAILED_TERMINAL"),
        (["complete-record-nonfinite-scan", "g16-import"], "FAILED_TERMINAL"),
        (["ledger-directory-fsync", "retry"], "FAILED_TERMINAL"),
        (["ledger-directory-fsync", "c02"], "FAILED_TERMINAL"),
        (["ledger-directory-fsync", "checkpoint-176"], "FAILED_TERMINAL"),
        (["ledger-complete-write"], "FAILED_TERMINAL"),
        (["credential-directory-fsync", "ledger-complete-write"], "FAILED_TERMINAL"),
        (["ledger-directory-fsync"], "SUCCEEDED_TERMINAL"),
        ([], "CONSUMED_ORPHAN"),
    ]
    for events, status in invalid:
        expect_reject(validate_trace, events, status)
        checks += 1
    require(
        validate_trace(["ledger-directory-fsync", "tensor-payload-open"], "CONSUMED_ORPHAN")
        == "CONSUMED_ORPHAN",
        "crash-after-consume orphan",
    )
    return checks + 1


def simulate_commit(
    payload_size: int,
    writes: list[int | str],
    fail_phase: str | None = None,
    namespace_preexists: bool = False,
    rollback_unlink_fails: bool = False,
    rollback_directory_fsync_fails: bool = False,
) -> dict[str, bool]:
    state = {
        "accepted": False,
        "ambiguous": False,
        "durable": False,
        "final": namespace_preexists,
        "preexisting_untouched": namespace_preexists,
        "rollback_durable": False,
        "staging": False,
    }
    if namespace_preexists:
        return state
    state["staging"] = True
    offset = 0
    try:
        for action in writes:
            if action == "interrupt":
                continue
            require(type(action) is int and action > 0, "invalid short write")
            require(action <= payload_size - offset, "overlong write")
            offset += action
            if offset == payload_size:
                break
        require(offset == payload_size, "incomplete write")
        require(fail_phase != "file-fsync", "file fsync failure")
        require(fail_phase != "link", "link failure")
        state["final"] = True
        state["staging"] = False
        require(fail_phase != "directory-fsync", "directory fsync failure")
        state["durable"] = True
        state["accepted"] = True
        return state
    except VerificationError:
        state["staging"] = False
        if state["final"]:
            if rollback_unlink_fails:
                state["final"] = True
            else:
                state["final"] = False
            if rollback_directory_fsync_fails or rollback_unlink_fails:
                state["ambiguous"] = True
            else:
                state["rollback_durable"] = True
        state["durable"] = False
        return state


def persistence_fault_checks() -> int:
    require(simulate_commit(7, [1, "interrupt", 2, 4]) == {
        "accepted": True,
        "ambiguous": False,
        "durable": True,
        "final": True,
        "preexisting_untouched": False,
        "rollback_durable": False,
        "staging": False,
    }, "short write success")
    checks = 1
    cases = [
        (7, [1, 2], None, False, False, False, False, False),
        (7, [0], None, False, False, False, False, False),
        (7, [8], None, False, False, False, False, False),
        (7, [7], "file-fsync", False, False, False, False, False),
        (7, [7], "link", False, False, False, False, False),
        (7, [7], "directory-fsync", False, False, False, False, True),
        (7, [7], "directory-fsync", False, True, False, True, False),
        (7, [7], "directory-fsync", False, False, True, True, False),
        (7, [7], "directory-fsync", False, True, True, True, False),
        (7, [7], None, True, False, False, False, False),
    ]
    for payload_size, writes, fail_phase, preexists, unlink_fails, rollback_fsync_fails, ambiguous, rollback_durable in cases:
        state = simulate_commit(
            payload_size,
            writes,
            fail_phase,
            preexists,
            unlink_fails,
            rollback_fsync_fails,
        )
        require(state["accepted"] is False and state["durable"] is False, "fault accepted")
        require(state["staging"] is False, "fault left staging")
        require(state["ambiguous"] is ambiguous, "fault ambiguity classification")
        require(state["rollback_durable"] is rollback_durable, "fault rollback durability")
        expected_final = preexists or (fail_phase == "directory-fsync" and unlink_fails)
        require(state["final"] is expected_final, "fault final namespace classification")
        require(state["preexisting_untouched"] is preexists, "pre-existing namespace modified")
        checks += 1
    return checks


def classify_terminal(
    evaluator_returncode: int,
    consumed_ledger_sha256: str | None,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    if result is not None:
        require(consumed_ledger_sha256 is not None, "result without consumed ledger")
        status = result["terminal"]["status"]
        reason_code = result["terminal"]["reason_code"]
        require(status in {"SUCCEEDED_TERMINAL", "FAILED_TERMINAL"}, "result terminal status")
        require(
            evaluator_returncode == (0 if status == "SUCCEEDED_TERMINAL" else 1),
            "evaluator return code/result mismatch",
        )
        return {
            "authority_consumed": True,
            "consumed_ledger_sha256": consumed_ledger_sha256,
            "consumption_evidence": VISIBLE_VALID_LEDGER,
            "execution_started": True,
            "reason_code": reason_code,
            "status": status,
        }
    if consumed_ledger_sha256 is not None:
        consumption_evidence = VISIBLE_VALID_LEDGER
        reason_code = "EVALUATOR_EXIT_WITHOUT_VALID_RESULT"
    else:
        consumption_evidence = NO_VISIBLE_LEDGER_AMBIGUOUS
        reason_code = "LEDGER_ROLLBACK_DURABILITY_AMBIGUOUS"
    return {
        "authority_consumed": True,
        "consumed_ledger_sha256": consumed_ledger_sha256,
        "consumption_evidence": consumption_evidence,
        "execution_started": False,
        "reason_code": reason_code,
        "status": "CONSUMED_ORPHAN",
    }


def terminal_record(
    package: dict[str, Any], status: str, result_sha256: str | None, consumption_evidence: str
) -> dict[str, Any]:
    ledger_visible = consumption_evidence == VISIBLE_VALID_LEDGER
    record = {
        "authority_consumed": True,
        "authority_controller_sha256": package["authority_controller"]["sha256"],
        "authority_package_sha256": sha256_file(PACKAGE_PATH),
        "authority_record_sha256": package["base_lane"]["authority_record"]["canonical_sha256"],
        "consumed_ledger_sha256": package["base_lane"]["ledger_record"]["canonical_sha256"] if ledger_visible else None,
        "consumption_evidence": consumption_evidence,
        "credential_sha256": "b" * 64,
        "evaluator_returncode": 0 if status == "SUCCEEDED_TERMINAL" else 1 if status == "FAILED_TERMINAL" else 2,
        "execution_started": result_sha256 is not None,
        "fresh_l2_acceptance_sha256": "c" * 64,
        "invocation_sha256": BASE_INVOCATION_SHA256,
        "irreversible_action_id": "synthetic-action:0001",
        "lane_label": "Base",
        "orphaned_after_consumption": status == "CONSUMED_ORPHAN",
        "reason_code": "HARD_THRESHOLDS_PASSED" if status == "SUCCEEDED_TERMINAL" else "SYNTHETIC_FAILURE",
        "result_path": RESULT_REL,
        "result_sha256": result_sha256,
        "retry_replay_resume_repair_permitted": False,
        "schema_id": FIRST_TERMINAL_SCHEMA_ID,
        "status": status,
    }
    record["record_sha256"] = hashlib.sha256(compact_bytes(record)).hexdigest()
    return record


def validate_terminal_record(package: dict[str, Any], record: dict[str, Any]) -> None:
    required = set(package["publication_protocol"]["first_terminal_record"]["required_fields"])
    require(set(record) == required, "terminal exact keys")
    checksum = record["record_sha256"]
    require(valid_sha256(checksum), "terminal checksum syntax")
    payload = dict(record)
    del payload["record_sha256"]
    require(hashlib.sha256(compact_bytes(payload)).hexdigest() == checksum, "terminal self-checksum")
    require(record["authority_package_sha256"] == sha256_file(PACKAGE_PATH), "terminal package binding")
    require(record["authority_controller_sha256"] == package["authority_controller"]["sha256"], "terminal controller binding")
    require(record["authority_record_sha256"] == package["base_lane"]["authority_record"]["canonical_sha256"], "terminal authority binding")
    require(valid_sha256(record["credential_sha256"]), "terminal credential binding")
    require(valid_sha256(record["fresh_l2_acceptance_sha256"]), "terminal review binding")
    require(re.fullmatch(ACTION_ID_PATTERN, record["irreversible_action_id"]) is not None, "terminal action binding")
    require(record["invocation_sha256"] == BASE_INVOCATION_SHA256, "terminal invocation binding")
    require(record["lane_label"] == "Base" and record["result_path"] == RESULT_REL, "terminal lane/output binding")
    require(record["retry_replay_resume_repair_permitted"] is False, "terminal retry policy")
    require(record["schema_id"] == FIRST_TERMINAL_SCHEMA_ID, "terminal schema identity")
    require(type(record["evaluator_returncode"]) is int, "terminal evaluator return code")
    require(record["authority_consumed"] is True, "terminal permanent consumption disposition")
    require(
        record["consumption_evidence"] in package["publication_protocol"]["first_terminal_record"]["consumption_evidence_values"],
        "terminal consumption evidence",
    )
    if record["consumption_evidence"] == VISIBLE_VALID_LEDGER:
        require(
            record["consumed_ledger_sha256"] == package["base_lane"]["ledger_record"]["canonical_sha256"],
            "terminal ledger binding",
        )
    else:
        require(record["consumption_evidence"] == NO_VISIBLE_LEDGER_AMBIGUOUS, "terminal no-ledger evidence")
        require(record["consumed_ledger_sha256"] is None, "terminal no-visible-ledger checksum")
        require(record["status"] == "CONSUMED_ORPHAN", "terminal no-visible-ledger status")
        require(record["result_sha256"] is None, "terminal no-visible-ledger result")
        require(record["execution_started"] is False, "terminal no-visible-ledger execution")
    if record["status"] == "SUCCEEDED_TERMINAL":
        require(record["consumption_evidence"] == VISIBLE_VALID_LEDGER, "successful terminal ledger evidence")
        require(record["evaluator_returncode"] == 0, "successful evaluator return code")
        require(valid_sha256(record["result_sha256"]), "successful terminal result binding")
        require(record["execution_started"] is True, "successful execution flag")
        require(record["orphaned_after_consumption"] is False, "successful orphan flag")
    elif record["status"] == "CONSUMED_ORPHAN":
        require(record["result_sha256"] is None, "orphan result")
        require(record["execution_started"] is False, "orphan execution flag")
        require(record["orphaned_after_consumption"] is True, "orphan flag")
    else:
        require(record["status"] == "FAILED_TERMINAL", "terminal status")
        require(record["consumption_evidence"] == VISIBLE_VALID_LEDGER, "failed terminal ledger evidence")
        require(record["orphaned_after_consumption"] is False, "failed orphan flag")
        require(valid_sha256(record["result_sha256"]), "consumed failure result binding")
        require(record["evaluator_returncode"] == 1, "failed evaluator return code")
        require(record["execution_started"] is True, "failed execution flag")


def terminal_binding_checks(package: dict[str, Any]) -> int:
    record = terminal_record(package, "SUCCEEDED_TERMINAL", "a" * 64, VISIBLE_VALID_LEDGER)
    validate_terminal_record(package, record)
    checks = 1
    for field, replacement in (
        ("authority_controller_sha256", "0" * 64),
        ("authority_package_sha256", "0" * 64),
        ("authority_record_sha256", "0" * 64),
        ("consumed_ledger_sha256", "0" * 64),
        ("credential_sha256", "0" * 63),
        ("fresh_l2_acceptance_sha256", "0" * 63),
        ("irreversible_action_id", "short"),
        ("invocation_sha256", "0" * 64),
        ("lane_label", "checkpoint-176"),
        ("result_path", "build/substituted/result.json"),
        ("result_sha256", "0" * 64),
        ("retry_replay_resume_repair_permitted", True),
        ("record_sha256", "0" * 64),
    ):
        mutation = copy.deepcopy(record)
        mutation[field] = replacement
        expect_reject(validate_terminal_record, package, mutation)
        checks += 1
    orphan = terminal_record(package, "CONSUMED_ORPHAN", None, VISIBLE_VALID_LEDGER)
    validate_terminal_record(package, orphan)
    checks += 1
    mutation = copy.deepcopy(orphan)
    mutation["result_sha256"] = "a" * 64
    mutation["record_sha256"] = hashlib.sha256(compact_bytes({k: v for k, v in mutation.items() if k != "record_sha256"})).hexdigest()
    expect_reject(validate_terminal_record, package, mutation)
    checks += 1
    ambiguous_orphan = terminal_record(package, "CONSUMED_ORPHAN", None, NO_VISIBLE_LEDGER_AMBIGUOUS)
    validate_terminal_record(package, ambiguous_orphan)
    checks += 1
    mutation = copy.deepcopy(ambiguous_orphan)
    mutation["status"] = "FAILED_TERMINAL"
    mutation["record_sha256"] = hashlib.sha256(compact_bytes({k: v for k, v in mutation.items() if k != "record_sha256"})).hexdigest()
    expect_reject(validate_terminal_record, package, mutation)
    checks += 1
    failed = terminal_record(package, "FAILED_TERMINAL", "d" * 64, VISIBLE_VALID_LEDGER)
    validate_terminal_record(package, failed)
    checks += 1
    mutation = copy.deepcopy(failed)
    mutation["authority_consumed"] = False
    mutation["record_sha256"] = hashlib.sha256(compact_bytes({k: v for k, v in mutation.items() if k != "record_sha256"})).hexdigest()
    expect_reject(validate_terminal_record, package, mutation)
    return checks + 1


def terminal_classifier_fault_checks(package: dict[str, Any]) -> int:
    ledger_sha256 = package["base_lane"]["ledger_record"]["canonical_sha256"]
    checks = 0
    for unlink_fails, rollback_fsync_fails, expected_evidence in (
        (True, False, VISIBLE_VALID_LEDGER),
        (False, True, NO_VISIBLE_LEDGER_AMBIGUOUS),
    ):
        state = simulate_commit(
            7,
            [7],
            "directory-fsync",
            False,
            unlink_fails,
            rollback_fsync_fails,
        )
        require(state["ambiguous"] is True, "coupled fault must be ambiguous")
        observed_ledger_sha256 = ledger_sha256 if state["final"] else None
        classification = classify_terminal(2, observed_ledger_sha256, None)
        require(classification["status"] == "CONSUMED_ORPHAN", "ambiguous fault terminal status")
        require(classification["authority_consumed"] is True, "ambiguous fault consumption disposition")
        require(classification["consumption_evidence"] == expected_evidence, "ambiguous fault evidence")
        record = terminal_record(package, classification["status"], None, classification["consumption_evidence"])
        record["consumed_ledger_sha256"] = classification["consumed_ledger_sha256"]
        record["execution_started"] = classification["execution_started"]
        record["reason_code"] = classification["reason_code"]
        record["record_sha256"] = hashlib.sha256(
            compact_bytes({key: value for key, value in record.items() if key != "record_sha256"})
        ).hexdigest()
        validate_terminal_record(package, record)
        checks += 1
    clean_rollback = simulate_commit(7, [7], "directory-fsync", False, False, False)
    require(clean_rollback["ambiguous"] is False and clean_rollback["final"] is False, "clean rollback model")
    conservative = classify_terminal(2, None, None)
    require(conservative["status"] == "CONSUMED_ORPHAN", "no-visible-ledger conservative terminal")
    require(conservative["consumption_evidence"] == NO_VISIBLE_LEDGER_AMBIGUOUS, "no-visible-ledger evidence")
    return checks + 1


def verify_static_bindings(package: dict[str, Any]) -> int:
    checks = 0
    g16_package = load_static_json(ROOT / G16_PACKAGE_REL)
    contract = load_static_json(ROOT / G16_CONTRACT_REL)
    proof = load_static_json(ROOT / G16_PROOF_REL)
    for name, expected_sha in (
        (CONTROLLER_REL, package["authority_controller"]["sha256"]),
        (G16_PACKAGE_REL, G16_PACKAGE_SHA256),
        (G16_CONTRACT_REL, G16_CONTRACT_SHA256),
        (G16_SCHEMA_REL, G16_SCHEMA_SHA256),
        (G16_EVALUATOR_REL, G16_EVALUATOR_SHA256),
        (G16_VERIFIER_REL, G16_VERIFIER_SHA256),
        (G16_PROOF_REL, G16_PROOF_SHA256),
    ):
        repo_binding(name, expected_sha)
        checks += 1
    expected_sidecar = f"{G16_PACKAGE_SHA256}  {Path(G16_PACKAGE_REL).name}\n"
    require((ROOT / G16_SIDECAR_REL).read_text(encoding="ascii") == expected_sidecar, "accepted G16 sidecar")
    checks += 1
    require(g16_package["artifacts"] == package["accepted_diagnostic"]["artifacts"], "G16 artifact closure")
    require(contract["invocations"][0] == package["base_lane"]["invocation"], "Base invocation closure")
    require(contract["input_set"]["lanes"][0]["tensor_bundle"]["sha256"] == BASE_TENSOR_SHA256, "Base tensor identity")
    require(proof["event_counts"]["sealed_tensor_payload_opens"] == 0, "accepted proof payload count")
    checks += 4
    review_mission_binding = package["authority_review_requirement"]["mission"]
    binding(AUTHORITY_REVIEW_MISSION_PATH, AUTHORITY_REVIEW_MISSION_SHA256)
    require(review_mission_binding["sha256"] == AUTHORITY_REVIEW_MISSION_SHA256, "authority review mission hash")
    require(review_mission_binding["byte_count"] == AUTHORITY_REVIEW_MISSION_PATH.stat().st_size, "authority review mission size")
    require(review_mission_binding["mission_id"] == AUTHORITY_REVIEW_MISSION_ID, "authority review mission id")
    checks += 4
    binding(TASK_MISSION_PATH, TASK_MISSION_SHA256)
    review = load_static_json(TASK_REVIEW_PATH)
    require(sha256_file(TASK_REVIEW_PATH) == TASK_REVIEW_SHA256, "accepted task review hash")
    require(review.get("review", {}).get("status") == "done", "accepted task review status")
    require(ACCEPTED_G16_PACKAGE_SHA256 in review.get("review", {}).get("reason", ""), "accepted task review package")
    checks += 4
    for dependency in contract["accepted_g16"].values():
        if isinstance(dependency, dict) and {"path", "sha256"}.issubset(dependency):
            path = Path(dependency["path"])
            if not path.is_absolute():
                path = ROOT / path
            binding(path, dependency["sha256"])
            require(path.stat().st_size == dependency["byte_count"], "accepted G16 dependency size")
            checks += 1
    for dependency in contract["preserved_c02_static_bindings"].values():
        path = ROOT / dependency["path"]
        binding(path, dependency["sha256"])
        require(path.stat().st_size == dependency["byte_count"], "accepted c02 dependency size")
        checks += 1
    verify_plain_regular(INTERPRETER_PATH)
    require(sha256_file(INTERPRETER_PATH) == INTERPRETER_SHA256, "interpreter hash")
    verify_plain_regular(ROOT / METADATA_REL)
    require(sha256_file(ROOT / METADATA_REL) == BASE_METADATA_SHA256, "metadata hash")
    verify_absent_plain(ROOT / TENSOR_REL)
    source_tensor = SOURCE_REPOSITORY_ROOT / TENSOR_REL
    verify_plain_regular(source_tensor)
    require(source_tensor.stat().st_size == package["action_root_projection"]["sealed_tensor_runtime_byte_count"], "source tensor lstat size")
    require(package["action_root_projection"]["sealed_tensor_runtime_sha256"] == BASE_TENSOR_SHA256, "source tensor package checksum binding")
    checks += 5
    require(
        package["action_root_projection"]["accepted_dependency_projection"]
        == projected_dependency_bindings(g16_package, contract, contract["input_set"]["lanes"][0]),
        "exact action-root dependency projection",
    )
    checks += 1
    expected_runtime = {
        **compatibility_runtime_bindings(),
        "authoritative_sources": {
            "accepted_c02_reader": repo_binding(ACCEPTED_C02_READER_REL, ACCEPTED_C02_READER_SHA256),
            "frozen_c02_producer": binding(PRODUCER_PATH, PRODUCER_SHA256),
            "rejected_v4_review": binding(REJECTED_V4_REVIEW_PATH, REJECTED_V4_REVIEW_SHA256),
        },
        "framing": {
            "count_field": "big-endian >I",
            "dimension_fields": "big-endian >Q",
            "dtype_length_field": "unsigned byte >B",
            "magic": "ACE2-C02-TENSORS-V1\n",
            "name_length_field": "big-endian >H",
            "payload_length_field": "big-endian >Q",
            "rank_field": "unsigned byte >B",
            "record_checksum_semantics": "ascii dtype + NUL + >I rank + >Q dimensions + payload",
            "trailing_bytes": "reject after exact EOF check",
        },
    }
    require(package["v5_compatibility_runtime"] == expected_runtime, "V5 compatibility runtime closure")
    checks += 1
    for binding_record in package["schemas"].values():
        repo_binding(binding_record["path"], binding_record["sha256"])
        checks += 1
    return checks


def verify_closed_v1_immutability() -> int:
    authority_root = SOURCE_REPOSITORY_ROOT / V1_AUTHORITY_ROOT_REL
    authority = authority_root / "base/authority.json"
    credential = authority_root / "base/credential.json"
    ledger = authority_root / "base/authority-ledger.json"
    terminal = authority_root / "base/first-terminal.json"
    result = SOURCE_REPOSITORY_ROOT / V1_RESULT_REL
    verify_plain_regular(authority)
    verify_plain_regular(credential)
    require(stat.S_IMODE(os.lstat(authority).st_mode) == 0o400, "closed V1 authority mode")
    require(stat.S_IMODE(os.lstat(credential).st_mode) == 0o400, "closed V1 credential mode")
    require(authority.stat().st_size == 596, "closed V1 authority bytes")
    require(credential.stat().st_size == 845, "closed V1 credential bytes")
    require(sha256_file(authority) == "f556b410a0a3abf00c306bcef8fbf5feae8e2e90902eb9f3ea037b2ec56c67f7", "closed V1 authority checksum")
    require(sha256_file(credential) == "bbd1fcd9dc9b14da9e0d932cf666b7dd6c82a9031da87b63ef91f6ab1dd3b24c", "closed V1 credential checksum")
    for path in (ledger, terminal, result):
        verify_absent_plain(path)
    return 11


def verify_closed_v2_immutability() -> int:
    files = [path for path in V2_ACTION_ROOT.rglob("*") if path.is_file()]
    require(len(files) == 23, "frozen V2 action-root file count")
    require(all(not path.is_symlink() for path in files), "frozen V2 action-root plain files")
    checks = 2
    for path, expected_sha256 in V2_FROZEN_BINDINGS.items():
        verify_plain_regular(path)
        require(sha256_file(path) == expected_sha256, f"frozen V2 checksum {path}")
        checks += 1
    for relative in (
        TENSOR_REL,
        AUTHORITY_ROOT_REL,
        AUTHORITY_LANE_ROOT_REL,
        AUTHORITY_REL,
        CREDENTIAL_REL,
        LEDGER_REL,
        FIRST_TERMINAL_REL,
        RESULT_ROOT_REL,
        RESULT_REL,
    ):
        verify_absent_plain(V2_ACTION_ROOT / relative)
        checks += 1
    return checks


def verify_consumed_v3_immutability() -> int:
    checks = 0
    for path, expected_sha256 in V3_FROZEN_BINDINGS.items():
        verify_plain_regular(path)
        require(sha256_file(path) == expected_sha256, f"frozen V3 checksum {path}")
        checks += 1
    binding(V3_REVIEW_PATH, V3_REVIEW_SHA256)
    review = load_static_json(V3_REVIEW_PATH)
    require(review.get("producer_role") == "reviewer", "V3 reviewer role")
    require(review.get("review", {}).get("status") == "done", "V3 reviewer status")
    review_reason = review.get("review", {}).get("reason", "")
    require("big-endian c02" in review_reason and "little-endian G16" in review_reason, "V3 root-cause review")
    checks += 4
    ledger_path = V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority-ledger.json"
    result_path = V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1/base/result.json"
    terminal_path = V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/first-terminal.json"
    credential_path = V3_ACTION_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/credential.json"
    ledger = load_static_json(ledger_path)
    result = load_static_json(result_path)
    terminal = load_static_json(terminal_path)
    require(ledger.get("state") == "CONSUMED" and ledger.get("lane_label") == "Base", "V3 consumed ledger")
    require(result.get("terminal", {}).get("status") == "FAILED_TERMINAL", "V3 failed result")
    require(result.get("terminal", {}).get("reason_code") == "INPUT_SCHEMA_REJECTED", "V3 result reason")
    require(terminal.get("status") == "FAILED_TERMINAL", "V3 first terminal status")
    require(terminal.get("reason_code") == "INPUT_SCHEMA_REJECTED", "V3 first terminal reason")
    require(terminal.get("authority_consumed") is True, "V3 terminal consumed")
    require(terminal.get("retry_replay_resume_repair_permitted") is False, "V3 no replay")
    verify_absent_plain(credential_path)
    checks += 8
    return checks


def verify_rejected_v4_immutability() -> int:
    checks = 0
    for path, expected_sha256 in V4_FROZEN_BINDINGS.items():
        verify_plain_regular(path)
        require(sha256_file(path) == expected_sha256, f"frozen V4 checksum {path}")
        checks += 1
    manifest_path = SOURCE_REPOSITORY_ROOT / "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V4_ACTION_ROOT_MANIFEST.json"
    manifest = load_static_json(manifest_path)
    require(manifest["mission_id"] == "4ae7875505a9", "rejected V4 mission identity")
    require(manifest["action_root"]["file_count"] == 23, "rejected V4 file count")
    require(manifest["action_root"]["sealed_tensor_slot_materialized"] is False, "rejected V4 tensor slot")
    for item in manifest["files"]:
        path = SOURCE_REPOSITORY_ROOT / item["path"]
        verify_plain_regular(path)
        require(path.stat().st_size == item["byte_count"], f"rejected V4 byte count {path}")
        require(sha256_file(path) == item["sha256"], f"rejected V4 file checksum {path}")
        checks += 1
    binding(REJECTED_V4_REVIEW_PATH, REJECTED_V4_REVIEW_SHA256)
    review = load_static_json(REJECTED_V4_REVIEW_PATH)
    require(review.get("review", {}).get("status") == "blocked", "rejected V4 review state")
    reason = review.get("review", {}).get("reason", "")
    require(">I name/dtype lengths" in reason and ">H rank" in reason, "rejected V4 defect binding")
    checks += 6
    for relative in (
        "build/qk-gbfp8-g16-head64-successor-diagnostic-v4-authority",
        "build/qk-gbfp8-g16-head64-successor-diagnostic-v4",
    ):
        verify_absent_plain(V4_ACTION_ROOT / relative)
        checks += 1
    return checks


def function_ast(tree: ast.AST, name: str) -> str:
    matches = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    require(len(matches) == 1, f"function surface {name}")
    return ast.dump(matches[0], include_attributes=False)


def normalized_evaluator_ast(source: str) -> str:
    tree = ast.parse(source)
    normalized: list[ast.stmt] = []
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and type(node.value.value) is str:
            continue
        if isinstance(node, ast.FunctionDef) and node.name in {"read_exact", "read_tensor_bundle", "tensor_record_sha256"}:
            continue
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names.intersection({"AUTHORITY_ROOT_REL", "OUTPUT_ROOT_REL"}):
                continue
        normalized.append(node)
    tree.body = normalized
    return ast.dump(tree, include_attributes=False)


def normalized_reader_function_ast(source: str, name: str) -> str:
    node = copy.deepcopy(_function_node(ast.parse(source), name))
    if name == "read_tensor_bundle":
        node.body = [statement for statement in node.body if isinstance(statement, (ast.Assign, ast.With, ast.Return))]

    class NormalizeRequire(ast.NodeTransformer):
        def visit_Call(self, call: ast.Call) -> ast.AST:
            self.generic_visit(call)
            if isinstance(call.func, ast.Name) and call.func.id == "require" and call.args:
                call.args = [call.args[0], ast.Constant(value="message")]
            return call

    return ast.dump(ast.fix_missing_locations(NormalizeRequire().visit(node)), include_attributes=False)


def evaluator_compatibility_surface_checks() -> int:
    v5_source = (ROOT / G16_EVALUATOR_REL).read_text(encoding="utf-8")
    accepted_source = ACCEPTED_C02_READER_PATH.read_text(encoding="utf-8")
    v3_path = V3_ACTION_ROOT / "reference/qk_gbfp8_g16_head64_successor_diagnostic_static_v1.py"
    v3_source = v3_path.read_text(encoding="utf-8")
    require(normalized_evaluator_ast(v5_source) == normalized_evaluator_ast(v3_source), "V5 evaluator non-reader drift")
    v5_tree = ast.parse(v5_source)
    accepted_tree = ast.parse(accepted_source)
    v3_tree = ast.parse(v3_source)
    reader_nodes = [node for node in ast.walk(v5_tree) if isinstance(node, ast.FunctionDef) and node.name == "read_tensor_bundle"]
    old_reader_nodes = [node for node in ast.walk(v3_tree) if isinstance(node, ast.FunctionDef) and node.name == "read_tensor_bundle"]
    require(len(reader_nodes) == 1 and len(old_reader_nodes) == 1, "reader function cardinality")
    old_reader_source = ast.get_source_segment(v3_source, old_reader_nodes[0]) or ""
    authoritative_grammar_checks(PRODUCER_PATH.read_text(encoding="utf-8"), accepted_source, v5_source)
    require(
        normalized_reader_function_ast(v5_source, "read_tensor_bundle")
        == normalized_reader_function_ast(accepted_source, "read_tensor_bundle"),
        "V5 reader control flow matches accepted reader",
    )
    require(
        normalized_reader_function_ast(v5_source, "read_exact")
        == normalized_reader_function_ast(accepted_source, "read_exact"),
        "V5 exact-read control flow matches accepted reader",
    )
    require(
        function_ast(v5_tree, "tensor_record_sha256") == function_ast(accepted_tree, "tensor_record_sha256"),
        "V5 per-record digest matches accepted reader",
    )
    require(old_reader_source.count('"little"') >= 6, "V3 little-endian root-cause surface")
    finiteness = [
        node for node in ast.walk(v5_tree) if isinstance(node, ast.FunctionDef) and node.name == "evaluate_selected_records_from_reference"
    ]
    require(len(finiteness) == 1, "reference evaluation function cardinality")
    finiteness_source = ast.get_source_segment(v5_source, finiteness[0]) or ""
    scan = finiteness_source.find("validate_selected_record_finiteness(selected)")
    import_reference = finiteness_source.find("g16 = import_g16_reference(expected_g16_sha256)")
    require(0 <= scan < import_reference, "nonfinite scan precedes G16 import")
    return 9


def controller_surface_checks() -> int:
    controller_source = CONTROLLER_PATH.read_text(encoding="utf-8")
    controller_tree = ast.parse(controller_source)
    verifier_tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    subprocess_run_calls = 0
    for node in ast.walk(controller_tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
        ):
            subprocess_run_calls += 1
    require("subprocess" in imported, "controller evaluator supervision import")
    require(not imported.intersection({"cocotb", "importlib", "mmap", "numpy", "onnxruntime", "tensorflow", "torch", "transformers"}), "controller payload/model import")
    require(subprocess_run_calls == 1, "controller exact evaluator launch surface")
    required_functions = {
        "classify_terminal",
        "consume_credential",
        "create_only_commit",
        "first_terminal_record",
        "reason_contains_full_sha",
        "supervise_and_publish",
        "validate_authority",
        "validate_authority_package",
        "validate_credential",
        "validate_evaluator_argv_paths",
        "validate_review",
        "validate_result_record",
    }
    controller_functions = {
        node.name for node in ast.walk(controller_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    require(required_functions.issubset(controller_functions), "controller required validation/publication functions")
    mirrored_schema_functions = [
        "_schema_type_matches",
        "_resolve_schema_ref",
        "_schema_matches",
        "classify_terminal",
        "reason_contains_full_sha",
        "validate_json_schema",
        "validate_result_record",
    ]
    for name in mirrored_schema_functions:
        require(function_ast(controller_tree, name) == function_ast(verifier_tree, name), f"controller schema mirror {name}")
    review_surface = function_ast(controller_tree, "validate_review")
    for token in (
        "AUTHORITY_REVIEW_DIRECTORY",
        "AUTHORITY_REVIEW_MISSION_ID",
        "AUTHORITY_REVIEW_MISSION_PATH",
        "fresh_l2_acceptance_path",
        "fresh_l2_acceptance_sha256",
        "producer_role",
        "reason_contains_full_sha",
        "status",
        "next_action",
    ):
        require(token in review_surface, f"controller review binding surface {token}")
    require(
        're.fullmatch(r"round-[0-9]{4}\\.json", review_path.name)' in controller_source,
        "controller review filename surface",
    )
    evaluator_tree = ast.parse((ROOT / G16_EVALUATOR_REL).read_text(encoding="utf-8"))
    evaluator_functions = {
        node.name for node in ast.walk(evaluator_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    require(
        {"read_tensor_bundle", "validate_selected_records", "validate_selected_record_finiteness"}.issubset(evaluator_functions),
        "accepted evaluator selected-record validators",
    )
    evaluator_source = (ROOT / G16_EVALUATOR_REL).read_text(encoding="utf-8")
    for token in (
        "require(set(bindings) == set(SELECTED_ROLES)",
        "require(len(record[\"payload\"]) == 2 * math.prod(record[\"shape\"])",
        "if ((word >> 7) & 0xFF) == 0xFF",
        "evaluate_selected_records_from_reference",
    ):
        require(token in evaluator_source, f"accepted evaluator selected-record surface {token}")
    return 3 + len(required_functions) + len(mirrored_schema_functions) + 10 + 1 + 4


def verify_namespace_absence() -> int:
    paths = [
        ROOT / AUTHORITY_ROOT_REL,
        ROOT / AUTHORITY_LANE_ROOT_REL,
        ROOT / AUTHORITY_REL,
        ROOT / CREDENTIAL_REL,
        ROOT / LEDGER_REL,
        ROOT / FIRST_TERMINAL_REL,
        ROOT / RESULT_ROOT_REL,
        ROOT / RESULT_REL,
    ]
    for path in paths:
        verify_absent_plain(path)
    return len(paths)


def verify_no_execution_surface() -> int:
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {"cocotb", "importlib", "mmap", "numpy", "onnxruntime", "subprocess", "tensorflow", "torch", "transformers"}
    forbidden_calls = {"Popen", "eval", "from_pretrained", "generate", "popen", "run", "system"}
    imported: set[str] = set()
    called: set[str] = set()
    exec_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
                if node.func.id == "exec":
                    exec_calls += 1
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    require(not imported.intersection(forbidden_modules), "execution/model module imported")
    require(not called.intersection(forbidden_calls), "execution call present")
    require(
        exec_calls == 2
        and "<synthetic-reader>" in ast.unparse(_function_node(tree, "_reader_namespace"))
        and "<sha256-guard-mutation>" in ast.unparse(_function_node(tree, "validate_sha256_guard_source")),
        "bounded AST reader/guard extraction",
    )
    return len(forbidden_modules) + len(forbidden_calls) + 1


def verify_sidecar() -> None:
    verify_plain_regular(SIDECAR_PATH)
    expected = f"{sha256_file(PACKAGE_PATH)}  {PACKAGE_PATH.name}\n"
    require(SIDECAR_PATH.read_text(encoding="ascii") == expected, "authority package sidecar")


def main() -> int:
    require(Path.cwd() == ROOT, "canonical cwd")
    require(dict(os.environ) == ENVIRONMENT, "exact verifier environment")
    require(platform.python_implementation() == "CPython", "Python implementation")
    require(platform.python_version() == "3.13.5", "Python version")
    require([sys.executable, "-I", "-B", *sys.argv] == VERIFIER_ARGV, "exact verifier argv")
    verify_plain_regular(INTERPRETER_PATH)
    require(sha256_file(INTERPRETER_PATH) == INTERPRETER_SHA256, "verifier interpreter")
    before_absence = verify_namespace_absence()
    package = load_canonical_json(PACKAGE_PATH)
    verify_package(package)
    verify_sidecar()
    static_binding_checks = verify_static_bindings(package)
    historical_compatibility_checks = historical_json_compatibility_checks(package)
    closed_v1_checks = verify_closed_v1_immutability()
    closed_v2_checks = verify_closed_v2_immutability()
    consumed_v3_checks = verify_consumed_v3_immutability()
    rejected_v4_checks = verify_rejected_v4_immutability()
    fresh_l2_review_binding_checks = review_binding_checks(package)
    semantic_mutation_checks = mutation_checks(package)
    sealed_payload_guard_checks = forbidden_payload_guard_checks()
    framing_checks = framing_compatibility_checks()
    selected_schema_checks = selected_record_schema_checks()
    nested_result_schema_checks = result_schema_checks(package)
    path_checks = synthetic_path_checks(package)
    protocol_ordering_checks = ordering_checks()
    persistence_checks = persistence_fault_checks()
    terminal_checks = terminal_binding_checks(package)
    classifier_fault_checks = terminal_classifier_fault_checks(package)
    controller_checks = controller_surface_checks()
    evaluator_surface_checks = evaluator_compatibility_surface_checks()
    no_execution_checks = verify_no_execution_surface()
    after_absence = verify_namespace_absence()
    print("QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V5=PASS")
    print(f"AUTHORITY_PACKAGE_SHA256={sha256_file(PACKAGE_PATH)}")
    print("LANES=1 BASE_ONLY=1 CHECKPOINT_176_AUTHORITIES=0 AUTHORITY_CARDINALITY=1")
    print(f"STATIC_BINDING_CHECKS={static_binding_checks}")
    print(f"HISTORICAL_JSON_COMPATIBILITY_CHECKS={historical_compatibility_checks}")
    print(f"V1_V2_CONSUMED_V3_REJECTED_V4_IMMUTABILITY_CHECKS={closed_v1_checks + closed_v2_checks + consumed_v3_checks + rejected_v4_checks}")
    print(f"FRESH_L2_REVIEW_BINDING_CHECKS={fresh_l2_review_binding_checks}")
    print(f"SEMANTIC_MUTATION_CHECKS={semantic_mutation_checks}")
    print(f"SEALED_PAYLOAD_GUARD_MUTATION_CHECKS={sealed_payload_guard_checks}")
    print(f"BIG_ENDIAN_FRAMING_CHECKS={framing_checks}")
    print(f"SELECTED_RECORD_SCHEMA_FAULT_CHECKS={selected_schema_checks}")
    print(f"NESTED_RESULT_SCHEMA_FAULT_CHECKS={nested_result_schema_checks}")
    print(f"BOUND_PATH_SYMLINK_CHECKS={path_checks}")
    print(f"PROTOCOL_ORDERING_CHECKS={protocol_ordering_checks}")
    print(f"PERSISTENCE_FAULT_CHECKS={persistence_checks}")
    print(f"TERMINAL_RESULT_BINDING_CHECKS={terminal_checks}")
    print(f"TERMINAL_CLASSIFIER_FAULT_CHECKS={classifier_fault_checks}")
    print(f"AUTHORITY_CONTROLLER_SURFACE_CHECKS={controller_checks}")
    print(f"EVALUATOR_COMPATIBILITY_SURFACE_CHECKS={evaluator_surface_checks}")
    print(f"NO_EXECUTION_SURFACE_CHECKS={no_execution_checks}")
    print(f"LIVE_NAMESPACE_ABSENCE_CHECKS={before_absence + after_absence}")
    print("LIVE_CREDENTIALS=0 READY_RECORDS=0 CONSUMED_LEDGERS=0 RESULT_NAMESPACES=0")
    print("PROJECTED_SEALED_TENSOR_SLOT=ABSENT EVALUATOR_ARGV_COMPATIBILITY_REVISION=V5_EXACT_C02")
    print("SEALED_TENSOR_PAYLOAD_OPENS=0 MODEL_LOADS=0 SEALED_DATA_EVALUATIONS=0 C02_REPLAYS=0")
    print("RTL_ACTIVITY=0 U280_ACTIVITY=0 XRT_ACTIVITY=0 HBM2_ACTIVITY=0 STAGE_TRANSITIONS=0")
    print("MATERIALIZATION_EXECUTION_AUTHORIZED=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"STATIC_INVALID_NO_EXECUTION_PERFORMED: {exc}", file=sys.stderr)
        raise SystemExit(1)
