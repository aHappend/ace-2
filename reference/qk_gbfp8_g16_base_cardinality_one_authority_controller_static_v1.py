#!/usr/bin/env python3
"""Checksum-bound controller for one future Base G16 authority consumption.

This file is frozen as nonexecuted authority-control code.  Static verification
must never invoke it.  A later separately reviewed irreversible action may run
it only with the exact package, authority, credential, review, and action
bindings described by the frozen authority package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_REL = "reference/qk_gbfp8_g16_base_cardinality_one_authority_controller_static_v1.py"
AUTHORITY_PACKAGE_REL = "reference/QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V1_PACKAGE.json"
AUTHORITY_PACKAGE_ID = "QK_GBFP8_G16_BASE_CARDINALITY_ONE_AUTHORITY_STATIC_V1"
AUTHORITY_REVIEW_MISSION_ID = "f882bfd685a1"
AUTHORITY_REVIEW_DIRECTORY = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f882bfd685a1"
)
AUTHORITY_REVIEW_MISSION_PATH = AUTHORITY_REVIEW_DIRECTORY / "mission.json"
AUTHORITY_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority.json"
CREDENTIAL_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/credential.json"
LEDGER_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/authority-ledger.json"
FIRST_TERMINAL_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v1-authority/base/first-terminal.json"
RESULT_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v1/base/result.json"
RESULT_ROOT_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v1"
G16_PACKAGE_SHA256 = "33c42fe340867f5b89d451339ca2d552f0bfec3309d8a6103066c171d7806513"
G16_EVALUATOR_SHA256 = "20e3f4fc891a7a75f2204662c1107a57319f665deafa3de12dfa3ea851d83dbe"
G16_SCHEMA_SHA256 = "8d512af26bf39593236098f850ae5477ae13343ffbdeac4fbb207b1b8fd71840"
BASE_INVOCATION_SHA256 = "675c744dca5cbc8e0823a84b8c94f9b5dbfdb5a9678a170ae5e35983021a25f3"
CREDENTIAL_SCHEMA_ID = "QK_GBFP8_G16_BASE_CARDINALITY_ONE_CREDENTIAL_V1"
FIRST_TERMINAL_SCHEMA_ID = "QK_GBFP8_G16_BASE_FIRST_TERMINAL_V2"
VISIBLE_VALID_LEDGER = "VISIBLE_VALID_LEDGER"
NO_VISIBLE_LEDGER_AMBIGUOUS = "NO_VISIBLE_LEDGER_AFTER_EVALUATOR_TREATED_AS_CONSUMED_ORPHAN"
REQUIRED_DECISION = "MATERIALIZE_BASE_AUTHORITY_ONCE"
ACTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,255}$"
INTERPRETER_PATH = Path("/home/argustest/miniconda3/bin/python3.13")
INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}


class ControllerError(RuntimeError):
    """Fail-closed authority-controller error."""


class AmbiguousCommitError(ControllerError):
    """A create-only publication may have survived an incomplete rollback."""


SchemaValidationError = ControllerError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerError(message)


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


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_existing_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            require(not current.is_symlink(), f"symlink component {current}")


def verify_plain_regular(path: Path) -> None:
    verify_existing_components(path)
    require(path.is_file() and not path.is_symlink(), f"not plain regular file {path}")


def verify_absent_plain(path: Path) -> None:
    verify_existing_components(path)
    require(not path.exists() and not path.is_symlink(), f"namespace exists {path}")


def load_canonical_json(path: Path) -> Any:
    verify_plain_regular(path)
    raw = path.read_bytes()
    require(raw != b"" and raw.endswith(b"\n"), f"canonical newline {path}")
    value = json.loads(raw.decode("ascii"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    reject_nonfinite(value)
    require(raw == compact_bytes(value), f"noncanonical JSON {path}")
    return value


def load_static_json(path: Path) -> Any:
    verify_plain_regular(path)
    raw = path.read_bytes()
    require(raw != b"" and raw.endswith(b"\n"), f"static JSON newline {path}")
    value = json.loads(raw.decode("ascii"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    reject_nonfinite(value)
    require(raw == pretty_bytes(value), f"noncanonical static JSON {path}")
    return value


def exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == expected, f"{context} exact keys")
    return value


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


def validate_authority_package(package: dict[str, Any], package_sha256: str) -> None:
    exact_keys(
        package,
        {
            "accepted_diagnostic",
            "artifact_kind",
            "authority_controller",
            "authority_package_id",
            "authority_review_requirement",
            "base_lane",
            "checkpoint_176_policy",
            "claim_boundary",
            "consumption_protocol",
            "credential_schema",
            "failure_semantics",
            "freeze_state",
            "namespaces",
            "preserved_dependencies",
            "publication_protocol",
            "schema_version",
            "static_verification",
            "static_verifier",
        },
        "authority package",
    )
    require(package["authority_package_id"] == AUTHORITY_PACKAGE_ID, "authority package identity")
    require(package["schema_version"] == 3, "authority package schema version")
    require(package["accepted_diagnostic"]["package"]["sha256"] == G16_PACKAGE_SHA256, "accepted G16 package")
    require(package["base_lane"]["authority_cardinality"] == 1, "Base authority cardinality")
    require(package["base_lane"]["lane_label"] == "Base", "Base lane")
    require(package["checkpoint_176_policy"]["authority_cardinality"] == 0, "checkpoint-176 authority")
    require(package["checkpoint_176_policy"]["lane_included"] is False, "checkpoint-176 exclusion")
    require(all(value is False for key, value in package["claim_boundary"].items() if key != "status"), "static claim boundary")
    controller = package["authority_controller"]
    require(controller["path"] == CONTROLLER_REL, "controller path")
    require(controller["sha256"] == sha256_file(Path(__file__)), "controller checksum")
    require(controller["byte_count"] == Path(__file__).stat().st_size, "controller byte count")
    require(controller["interpreter_sha256"] == INTERPRETER_SHA256, "controller interpreter binding")
    require(controller["environment"] == EXACT_ENVIRONMENT, "controller environment binding")
    require(controller["evaluator_argv"] == package["base_lane"]["invocation"]["argv"], "controller evaluator argv")
    require(controller["execution_state"] == "NOT_EXECUTED", "controller execution state")
    review = package["authority_review_requirement"]
    require(review["mission_id"] == AUTHORITY_REVIEW_MISSION_ID, "authority review mission")
    require(review["handoff_directory"] == str(AUTHORITY_REVIEW_DIRECTORY), "authority review directory")
    require(review["mission_path"] == str(AUTHORITY_REVIEW_MISSION_PATH), "authority review mission path")
    mission = review["mission"]
    require(mission["path"] == str(AUTHORITY_REVIEW_MISSION_PATH), "authority review mission binding path")
    verify_plain_regular(AUTHORITY_REVIEW_MISSION_PATH)
    require(AUTHORITY_REVIEW_MISSION_PATH.stat().st_size == mission["byte_count"], "authority review mission bytes")
    require(sha256_file(AUTHORITY_REVIEW_MISSION_PATH) == mission["sha256"], "authority review mission checksum")
    credential = package["credential_schema"]
    require(credential["schema_id"] == CREDENTIAL_SCHEMA_ID, "credential schema identity")
    require(
        credential["authority_package_binding"] == "required_exact_runtime_sha256",
        "credential package binding rule",
    )
    require(credential["required_decision"] == REQUIRED_DECISION, "credential decision")
    require(credential["irreversible_action_id_pattern"] == ACTION_ID_PATTERN, "credential action pattern")
    require(
        set(credential["required_fields"])
        == {
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
        },
        "credential required fields",
    )
    require(package["namespaces"]["authority"]["path"] == AUTHORITY_REL, "authority namespace")
    require(package["namespaces"]["credential"]["path"] == CREDENTIAL_REL, "credential namespace")
    require(package["namespaces"]["ledger"]["path"] == LEDGER_REL, "ledger namespace")
    require(package["namespaces"]["first_terminal"]["path"] == FIRST_TERMINAL_REL, "terminal namespace")
    require(package["namespaces"]["result"]["path"] == RESULT_REL, "result namespace")
    first_terminal = package["publication_protocol"]["first_terminal_record"]
    require(first_terminal["schema_id"] == FIRST_TERMINAL_SCHEMA_ID, "terminal schema identity")
    require(
        first_terminal["consumption_evidence_values"]
        == [VISIBLE_VALID_LEDGER, NO_VISIBLE_LEDGER_AMBIGUOUS],
        "terminal consumption evidence values",
    )


def validate_bound_static_artifacts(package: dict[str, Any]) -> dict[str, Any]:
    accepted = package["accepted_diagnostic"]
    bindings = [accepted["package"], accepted["sidecar"], accepted["task_mission"], accepted["fresh_l2_handoff"]]
    bindings.extend(accepted["artifacts"].values())
    for binding in bindings:
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        verify_plain_regular(path)
        require(path.stat().st_size == binding["byte_count"], f"artifact byte count {path}")
        require(sha256_file(path) == binding["sha256"], f"artifact checksum {path}")
    schema_binding = accepted["artifacts"]["result_schema"]
    require(schema_binding["sha256"] == G16_SCHEMA_SHA256, "result schema identity")
    return load_static_json(ROOT / schema_binding["path"])


def validate_review(review_path: Path, package_sha256: str, credential: dict[str, Any]) -> str:
    verify_plain_regular(review_path)
    require(review_path.parent == AUTHORITY_REVIEW_DIRECTORY, "review directory")
    require(re.fullmatch(r"round-[0-9]{4}\.json", review_path.name) is not None, "review filename")
    review_sha256 = sha256_file(review_path)
    require(review_sha256 == credential["fresh_l2_acceptance_sha256"], "credential review checksum")
    require(str(review_path) == credential["fresh_l2_acceptance_path"], "credential review path")
    review = load_static_json(review_path)
    require(review.get("kind") == "round_reviewed_handoff", "review kind")
    require(review.get("mission_id") == AUTHORITY_REVIEW_MISSION_ID, "review mission")
    require(review.get("mission_context") == str(AUTHORITY_REVIEW_MISSION_PATH), "review mission context")
    require(review.get("producer_role") == "reviewer", "review role")
    decision = review.get("review")
    require(type(decision) is dict and decision.get("status") == "done", "Fresh-L2 acceptance status")
    require(decision.get("next_action") == "", "Fresh-L2 acceptance next action")
    require(package_sha256 in decision.get("reason", ""), "Fresh-L2 package checksum binding")
    return review_sha256


def validate_authority(package: dict[str, Any], authority_path: Path) -> tuple[dict[str, Any], str]:
    authority = load_canonical_json(authority_path)
    expected = package["base_lane"]["authority_record"]
    require(authority == expected["template"], "authority canonical template")
    authority_sha256 = sha256_file(authority_path)
    require(authority_sha256 == expected["canonical_sha256"], "authority checksum")
    require(authority_path.stat().st_size == expected["canonical_byte_count"], "authority byte count")
    return authority, authority_sha256


def validate_credential(
    package: dict[str, Any],
    package_sha256: str,
    authority_sha256: str,
    credential_path: Path,
    review_path: Path,
    irreversible_action_id: str,
) -> tuple[dict[str, Any], str]:
    credential = load_canonical_json(credential_path)
    required = set(package["credential_schema"]["required_fields"])
    exact_keys(credential, required, "credential")
    credential_sha256 = credential["credential_sha256"]
    require(valid_sha256(credential_sha256), "credential self-checksum syntax")
    payload = dict(credential)
    del payload["credential_sha256"]
    require(hashlib.sha256(compact_bytes(payload)).hexdigest() == credential_sha256, "credential self-checksum")
    require(credential["schema_id"] == CREDENTIAL_SCHEMA_ID, "credential identity")
    require(credential["decision"] == REQUIRED_DECISION, "credential decision")
    require(credential["lane_label"] == "Base", "credential lane")
    require(credential["authority_package_sha256"] == package_sha256, "credential package")
    require(credential["authority_record_sha256"] == authority_sha256, "credential authority")
    require(credential["authority_controller_sha256"] == package["authority_controller"]["sha256"], "credential controller")
    require(credential["irreversible_action_id"] == irreversible_action_id, "credential action id")
    require(re.fullmatch(ACTION_ID_PATTERN, irreversible_action_id) is not None, "irreversible action id syntax")
    review_sha256 = validate_review(review_path, package_sha256, credential)
    require(credential["fresh_l2_acceptance_sha256"] == review_sha256, "credential review")
    return credential, credential_sha256


def consume_credential(credential_path: Path) -> None:
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(
            credential_path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        os.unlink(credential_path)
        os.fsync(directory_descriptor)
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except InterruptedError:
            continue
        if written <= 0 or written > len(payload) - offset:
            raise OSError("invalid zero-length or overlong write")
        offset += written


def create_only_commit(path: Path, payload: bytes) -> None:
    staging_path = path.with_name(f".{path.name}.incomplete")
    descriptor: int | None = None
    directory_descriptor: int | None = None
    staging_created = False
    final_linked = False
    rollback_unlinked = False
    try:
        descriptor = os.open(staging_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
        staging_created = True
        write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.link(staging_path, path, follow_symlinks=False)
        final_linked = True
        os.unlink(staging_path)
        staging_created = False
        os.fsync(directory_descriptor)
        return
    except BaseException as original:
        if final_linked:
            try:
                os.unlink(path)
                rollback_unlinked = True
            except OSError:
                pass
        if staging_created:
            try:
                os.unlink(staging_path)
            except OSError:
                pass
        if final_linked:
            try:
                require(directory_descriptor is not None, "rollback directory descriptor")
                os.fsync(directory_descriptor)
            except BaseException as rollback_error:
                raise AmbiguousCommitError("first-terminal rollback directory fsync failed") from rollback_error
            if not rollback_unlinked:
                raise AmbiguousCommitError("first-terminal rollback unlink failed") from original
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass


def validate_ledger(package: dict[str, Any], ledger_path: Path) -> str:
    ledger = load_canonical_json(ledger_path)
    expected = package["base_lane"]["ledger_record"]
    require(ledger == expected["template"], "consumed ledger template")
    ledger_sha256 = sha256_file(ledger_path)
    require(ledger_sha256 == expected["canonical_sha256"], "consumed ledger checksum")
    require(ledger_path.stat().st_size == expected["canonical_byte_count"], "consumed ledger byte count")
    return ledger_sha256


def first_terminal_record(
    package: dict[str, Any],
    package_sha256: str,
    authority_sha256: str,
    credential_sha256: str,
    review_sha256: str,
    irreversible_action_id: str,
    evaluator_returncode: int,
    classification: dict[str, Any],
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    record = {
        "authority_consumed": classification["authority_consumed"],
        "authority_controller_sha256": package["authority_controller"]["sha256"],
        "authority_package_sha256": package_sha256,
        "authority_record_sha256": authority_sha256,
        "consumed_ledger_sha256": classification["consumed_ledger_sha256"],
        "consumption_evidence": classification["consumption_evidence"],
        "credential_sha256": credential_sha256,
        "evaluator_returncode": evaluator_returncode,
        "execution_started": classification["execution_started"],
        "fresh_l2_acceptance_sha256": review_sha256,
        "invocation_sha256": BASE_INVOCATION_SHA256,
        "irreversible_action_id": irreversible_action_id,
        "lane_label": "Base",
        "orphaned_after_consumption": classification["status"] == "CONSUMED_ORPHAN",
        "reason_code": classification["reason_code"],
        "result_path": RESULT_REL,
        "result_sha256": None if result is None else result["result_sha256"],
        "retry_replay_resume_repair_permitted": False,
        "schema_id": FIRST_TERMINAL_SCHEMA_ID,
        "status": classification["status"],
    }
    record["record_sha256"] = hashlib.sha256(compact_bytes(record)).hexdigest()
    return record


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


def supervise_and_publish(
    package: dict[str, Any],
    schema: dict[str, Any],
    package_sha256: str,
    authority_sha256: str,
    credential_sha256: str,
    review_sha256: str,
    irreversible_action_id: str,
    evaluator_returncode: int,
) -> None:
    ledger_path = ROOT / LEDGER_REL
    result_path = ROOT / RESULT_REL
    terminal_path = ROOT / FIRST_TERMINAL_REL
    consumed_ledger_sha256 = validate_ledger(package, ledger_path) if ledger_path.exists() else None
    result: dict[str, Any] | None = None
    if result_path.exists():
        require(consumed_ledger_sha256 is not None, "result without consumed ledger")
        candidate = load_canonical_json(result_path)
        try:
            validate_result_record(package, schema, candidate, package_sha256)
        except ControllerError:
            candidate = None
        result = candidate
    classification = classify_terminal(evaluator_returncode, consumed_ledger_sha256, result)
    terminal = first_terminal_record(
        package,
        package_sha256,
        authority_sha256,
        credential_sha256,
        review_sha256,
        irreversible_action_id,
        evaluator_returncode,
        classification,
        result,
    )
    create_only_commit(terminal_path, compact_bytes(terminal))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    value.add_argument("--authority-package", required=True)
    value.add_argument("--authority", required=True)
    value.add_argument("--credential", required=True)
    value.add_argument("--fresh-l2-review", required=True)
    value.add_argument("--irreversible-action-id", required=True)
    value.add_argument("--first-terminal", required=True)
    return value


def _main() -> int:
    args = parser().parse_args()
    require(Path.cwd() == ROOT, "canonical cwd")
    require(dict(os.environ) == EXACT_ENVIRONMENT, "exact environment")
    require(platform.python_implementation() == "CPython", "Python implementation")
    require(platform.python_version() == "3.13.5", "Python version")
    require(Path(sys.executable) == INTERPRETER_PATH, "interpreter path")
    verify_plain_regular(INTERPRETER_PATH)
    require(sha256_file(INTERPRETER_PATH) == INTERPRETER_SHA256, "interpreter checksum")
    expected_argv = [
        str(INTERPRETER_PATH),
        "-I",
        "-B",
        CONTROLLER_REL,
        "--authority-package",
        AUTHORITY_PACKAGE_REL,
        "--authority",
        AUTHORITY_REL,
        "--credential",
        CREDENTIAL_REL,
        "--fresh-l2-review",
        args.fresh_l2_review,
        "--irreversible-action-id",
        args.irreversible_action_id,
        "--first-terminal",
        FIRST_TERMINAL_REL,
    ]
    require([sys.executable, "-I", "-B", *sys.argv] == expected_argv, "exact controller argv")
    require(args.authority_package == AUTHORITY_PACKAGE_REL, "authority package argv")
    require(args.authority == AUTHORITY_REL, "authority argv")
    require(args.credential == CREDENTIAL_REL, "credential argv")
    require(args.first_terminal == FIRST_TERMINAL_REL, "first-terminal argv")

    package_path = ROOT / AUTHORITY_PACKAGE_REL
    authority_path = ROOT / AUTHORITY_REL
    credential_path = ROOT / CREDENTIAL_REL
    ledger_path = ROOT / LEDGER_REL
    terminal_path = ROOT / FIRST_TERMINAL_REL
    result_root = ROOT / RESULT_ROOT_REL
    review_path = Path(args.fresh_l2_review)

    verify_plain_regular(package_path)
    package_sha256 = sha256_file(package_path)
    sidecar_path = package_path.with_name(package_path.name + ".sha256")
    verify_plain_regular(sidecar_path)
    require(
        sidecar_path.read_text(encoding="ascii") == f"{package_sha256}  {package_path.name}\n",
        "authority package sidecar",
    )
    package = load_static_json(package_path)
    validate_authority_package(package, package_sha256)
    schema = validate_bound_static_artifacts(package)
    verify_plain_regular(authority_path)
    verify_plain_regular(credential_path)
    verify_absent_plain(ledger_path)
    verify_absent_plain(terminal_path)
    verify_absent_plain(result_root)
    require(
        {entry.name for entry in authority_path.parent.iterdir()} == {authority_path.name, credential_path.name},
        "pre-action authority namespace contents",
    )
    _, authority_sha256 = validate_authority(package, authority_path)
    credential, credential_sha256 = validate_credential(
        package,
        package_sha256,
        authority_sha256,
        credential_path,
        review_path,
        args.irreversible_action_id,
    )
    review_sha256 = credential["fresh_l2_acceptance_sha256"]
    consume_credential(credential_path)
    require(
        {entry.name for entry in authority_path.parent.iterdir()} == {authority_path.name},
        "post-credential authority namespace contents",
    )
    completed = subprocess.run(
        package["authority_controller"]["evaluator_argv"],
        cwd=ROOT,
        env=EXACT_ENVIRONMENT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(completed.stdout == b"", "unexpected evaluator stdout")
    supervise_and_publish(
        package,
        schema,
        package_sha256,
        authority_sha256,
        credential_sha256,
        review_sha256,
        args.irreversible_action_id,
        completed.returncode,
    )
    return completed.returncode


def main() -> int:
    try:
        return _main()
    except (ControllerError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        payload = {
            "message": str(exc),
            "materialization_or_execution_authorized": False,
            "reason_code": "AUTHORITY_CONTROLLER_REJECTED",
            "schema_id": "QK_GBFP8_G16_BASE_AUTHORITY_CONTROLLER_REJECTION_V1",
        }
        sys.stderr.buffer.write(compact_bytes(payload))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
