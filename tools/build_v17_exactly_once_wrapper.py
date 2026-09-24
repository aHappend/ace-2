#!/usr/bin/env python3
"""Build the immutable V17 exactly-once wrapper without executing V17."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31"
BUILD_ROOT = PROJECT_ROOT / "build/v17-exactly-once-wrapper-4c8a6e31-attempt-0001"
CORE_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root"
CORE_MANIFEST = CORE_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json"
CORE_ACCEPTANCE = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/FRESH_L2_STATIC_ACCEPTANCE.json"
CORE_VERIFIER_REPORT = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/independent-inert-verifier-report.json"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
VERIFY_TOOL = PROJECT_ROOT / "tools/verify_v17_exactly_once_wrapper_inert.py"
MANIFEST_NAME = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json"
DISPOSITION = "V17_EXACTLY_ONCE_WRAPPER_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY"
EXACT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
EXPECTED_CORE_MANIFEST_SHA256 = "9e120e0e831ec736df45bd1b9f825217ce3368c5e22dbf3e4fc67ac537ea61f1"
EXPECTED_CORE_CONTENT_SHA256 = "dabd719fde0df6bc075ea8f5d34e09606bef464bbd104dfdf6581268a463898d"
EXPECTED_CORE_ACCEPTANCE_SHA256 = "9e8eb7a1a7c4340ab8d7b70a2ee97a132169711659f0ca7608d6701ad3225761"
EXPECTED_CORE_ACCEPTANCE_SELF_SHA256 = "dcb09b60b3a9010a8360f0fb5c0004f86c70be7b94646f2031dfd868e601bd17"
EXPECTED_CORE_VERIFIER_REPORT_SHA256 = "36191b9bc6b0b99ad9bd37519d5982e0efdb4ed61572f3cc69ee1c3526c6c018"
EXPECTED_INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0))
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def canonical_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, compact_bytes(value))


def inventory_digest(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"preservation root absent: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink in preservation root: {path}")
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({
                "kind": "file",
                "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                "path": relative,
                "sha256": sha256_file(path),
                "size": info.st_size,
            })
        else:
            raise BuildError(f"unsupported preservation entry: {path}")
    return {
        "entry_count": len(records),
        "file_count": sum(record["kind"] == "file" for record in records),
        "root": str(root),
        "tree_sha256": sha256_bytes(compact_bytes(records)),
    }


def runtime_directories() -> list[str]:
    return [
        str(RUNTIME_ROOT),
        str(RUNTIME_ROOT / "primary"),
        str(RUNTIME_ROOT / "primary/authority"),
        str(RUNTIME_ROOT / "primary/authority/base"),
        str(RUNTIME_ROOT / "primary/result"),
        str(RUNTIME_ROOT / "primary/result/base"),
        str(RUNTIME_ROOT / "fallback"),
    ]


def diagnostic_definition(core_terminal_schema: dict[str, Any]) -> dict[str, Any]:
    return core_terminal_schema["$defs"]["diagnostic"]


def lifecycle_schemas(core_terminal_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    authority = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_AUTHORITY_SCHEMA",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action_id": {"const": ACTION_ID},
            "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_authority"},
            "authority_sha256": sha,
            "core_acceptance_file_sha256": {"const": EXPECTED_CORE_ACCEPTANCE_SHA256},
            "core_manifest_file_sha256": {"const": EXPECTED_CORE_MANIFEST_SHA256},
            "execution_package_file_sha256": sha,
            "fresh_l2_acceptance_sha256": sha,
            "interpreter_sha256": {"const": EXPECTED_INTERPRETER_SHA256},
            "invocation_sha256": sha,
            "launcher_invocation_sha256": sha,
            "official_identities": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "input_tokens_sha256": sha,
                    "lane_metadata_sha256": sha,
                    "model_sha256": sha,
                    "tensor_bundle_sha256": sha,
                },
                "required": ["input_tokens_sha256", "lane_metadata_sha256", "model_sha256", "tensor_bundle_sha256"],
            },
            "single_use": {"const": True},
            "state": {"const": "READY_UNCONSUMED"},
            "transport_attestation": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "api": {"const": "os.posix_spawn"},
                    "immediate_parent_argv_sha256": sha,
                    "immediate_parent_environment_sha256": sha,
                    "immediate_parent_executable_sha256": {"const": EXPECTED_INTERPRETER_SHA256},
                    "immediate_parent_pid": {"type": "integer", "minimum": 1},
                    "immediate_parent_transport_sha256": sha,
                    "launcher_argv_sha256": sha,
                    "launcher_environment_sha256": sha,
                    "shell": {"const": False},
                    "transport_attestation_sha256": sha,
                },
                "required": [
                    "api", "immediate_parent_argv_sha256", "immediate_parent_environment_sha256",
                    "immediate_parent_executable_sha256", "immediate_parent_pid",
                    "immediate_parent_transport_sha256", "launcher_argv_sha256",
                    "launcher_environment_sha256", "shell", "transport_attestation_sha256",
                ],
            },
        },
        "required": [
            "action_id", "artifact_kind", "authority_sha256", "core_acceptance_file_sha256",
            "core_manifest_file_sha256", "execution_package_file_sha256", "fresh_l2_acceptance_sha256",
            "interpreter_sha256", "invocation_sha256", "launcher_invocation_sha256",
            "official_identities", "single_use", "state", "transport_attestation",
        ],
    }
    credential = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_CREDENTIAL_SCHEMA",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action_id": {"const": ACTION_ID},
            "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_credential"},
            "authority_sha256": sha,
            "credential_nonce_sha256": sha,
            "credential_sha256": sha,
            "single_use": {"const": True},
            "state": {"const": "READY_UNCONSUMED"},
        },
        "required": ["action_id", "artifact_kind", "authority_sha256", "credential_nonce_sha256", "credential_sha256", "single_use", "state"],
    }
    ledger = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_LEDGER_SCHEMA",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action_id": {"const": ACTION_ID},
            "action_retired": {"const": True},
            "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_consumed_ledger"},
            "authority_sha256": sha,
            "consumed_ledger_sha256": sha,
            "credential_consumption": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"credential_sha256": sha, "state": {"const": "CONSUMED_BEFORE_PAYLOAD"}},
                "required": ["credential_sha256", "state"],
            },
            "execution_package_file_sha256": sha,
            "invocation_cardinality": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"invocation_count_performed_at_publish": {"const": 0}, "invocation_count_permitted": {"const": 1}},
                "required": ["invocation_count_performed_at_publish", "invocation_count_permitted"],
            },
            "payload_cardinality": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"payload_open_count_performed_at_publish": {"const": 0}, "payload_open_count_permitted": {"const": 1}},
                "required": ["payload_open_count_performed_at_publish", "payload_open_count_permitted"],
            },
            "post_consumption_ambiguity_default": {"const": "CONSUMED_ORPHAN"},
            "replay_permitted": {"const": False},
            "retry_replay_resume_repair_replacement_permitted": {"const": False},
            "transport_attestation_sha256": sha,
        },
        "required": [
            "action_id", "action_retired", "artifact_kind", "authority_sha256", "consumed_ledger_sha256",
            "credential_consumption", "execution_package_file_sha256", "invocation_cardinality",
            "payload_cardinality", "post_consumption_ambiguity_default", "replay_permitted",
            "retry_replay_resume_repair_replacement_permitted", "transport_attestation_sha256",
        ],
    }
    wrapper_diagnostic = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "detail_sha256": sha,
            "exception_type": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_.]*$", "maxLength": 96},
            "failure_stage": {"type": "string", "maxLength": 64, "pattern": "^[A-Z_]+$"},
        },
        "required": ["detail_sha256", "exception_type", "failure_stage"],
    }
    terminal = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA",
        "$defs": {"diagnostic": diagnostic_definition(core_terminal_schema), "sha256": sha, "wrapper_diagnostic": wrapper_diagnostic},
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action_id": {"const": ACTION_ID},
            "action_retired": {"const": True},
            "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_first_terminal"},
            "authority_sha256": {"oneOf": [{"$ref": "#/$defs/sha256"}, {"type": "null"}]},
            "consumed_ledger_sha256": {"oneOf": [{"$ref": "#/$defs/sha256"}, {"type": "null"}]},
            "credential_consumed": {"type": "boolean"},
            "evaluator_diagnostic": {"oneOf": [{"$ref": "#/$defs/diagnostic"}, {"type": "null"}]},
            "failure_detail_sha256": {"$ref": "#/$defs/sha256"},
            "failure_stage": {"oneOf": [{"type": "string", "enum": [
                "PREFLIGHT", "AUTHORITY_MATERIALIZATION", "CREDENTIAL_MATERIALIZATION",
                "LEDGER_MATERIALIZATION", "CREDENTIAL_CONSUMPTION", "PAYLOAD_OPEN",
                "EVALUATOR_CALL", "RESULT_PUBLICATION", "TERMINAL_PUBLICATION",
            ]}, {"type": "null"}]},
            "first_record_immutable": {"const": True},
            "first_terminal_sha256": {"$ref": "#/$defs/sha256"},
            "invocation_count_performed": {"type": "integer", "minimum": 0, "maximum": 1},
            "official_target_process_starts": {"type": "integer", "minimum": 0, "maximum": 1},
            "orphaned_after_consumption": {"type": "boolean"},
            "payload_open_count": {"type": "integer", "minimum": 0, "maximum": 1},
            "reason_code": {"type": "string", "enum": [
                "HARD_THRESHOLD_FAILED", "HARD_THRESHOLDS_PASSED", "LIVE_MATERIALIZATION_FAILED",
                "POST_CONSUMPTION_AMBIGUITY", "READ_ONLY_PREFLIGHT_FAILED",
            ]},
            "result_file_sha256": {"oneOf": [{"$ref": "#/$defs/sha256"}, {"type": "null"}]},
            "retry_replay_resume_repair_replacement_permitted": {"const": False},
            "status": {"type": "string", "enum": ["CONSUMED_ORPHAN", "FAILED_TERMINAL", "PREFLIGHT_FAILED_TERMINAL", "SUCCEEDED_TERMINAL"]},
            "wrapper_diagnostic": {"oneOf": [{"$ref": "#/$defs/wrapper_diagnostic"}, {"type": "null"}]},
        },
        "required": [
            "action_id", "action_retired", "artifact_kind", "authority_sha256", "consumed_ledger_sha256",
            "credential_consumed", "evaluator_diagnostic", "failure_detail_sha256", "failure_stage",
            "first_record_immutable", "first_terminal_sha256", "invocation_count_performed",
            "official_target_process_starts", "orphaned_after_consumption", "payload_open_count",
            "reason_code", "result_file_sha256", "retry_replay_resume_repair_replacement_permitted",
            "status", "wrapper_diagnostic",
        ],
        "allOf": [
            {"if": {"properties": {"status": {"const": "PREFLIGHT_FAILED_TERMINAL"}}, "required": ["status"]}, "then": {"properties": {
                "authority_sha256": {"type": "null"}, "consumed_ledger_sha256": {"type": "null"},
                "credential_consumed": {"const": False}, "failure_stage": {"const": "PREFLIGHT"},
                "invocation_count_performed": {"const": 0}, "official_target_process_starts": {"const": 0},
                "orphaned_after_consumption": {"const": False}, "payload_open_count": {"const": 0},
                "reason_code": {"const": "READ_ONLY_PREFLIGHT_FAILED"}, "result_file_sha256": {"type": "null"},
            }}},
            {"if": {"properties": {"status": {"const": "CONSUMED_ORPHAN"}}, "required": ["status"]}, "then": {"properties": {
                "authority_sha256": {"$ref": "#/$defs/sha256"}, "consumed_ledger_sha256": {"$ref": "#/$defs/sha256"},
                "credential_consumed": {"const": True}, "orphaned_after_consumption": {"const": True},
                "reason_code": {"const": "POST_CONSUMPTION_AMBIGUITY"},
            }}},
            {"if": {"properties": {"status": {"enum": ["SUCCEEDED_TERMINAL", "FAILED_TERMINAL"]}}, "required": ["status"]}, "then": {"oneOf": [
                {"properties": {"failure_stage": {"type": "null"}, "invocation_count_performed": {"const": 1},
                    "official_target_process_starts": {"const": 1}, "payload_open_count": {"const": 1},
                    "result_file_sha256": {"$ref": "#/$defs/sha256"}}},
                {"properties": {"consumed_ledger_sha256": {"type": "null"}, "credential_consumed": {"const": False},
                    "invocation_count_performed": {"const": 0}, "official_target_process_starts": {"const": 0},
                    "payload_open_count": {"const": 0}, "reason_code": {"const": "LIVE_MATERIALIZATION_FAILED"},
                    "result_file_sha256": {"type": "null"}}},
            ]}},
        ],
    }
    acceptance = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FRESH_L2_ACCEPTANCE_SCHEMA",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "acceptance_sha256": sha,
            "accepted": {"const": True},
            "action_id": {"const": ACTION_ID},
            "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_fresh_l2_static_acceptance"},
            "claim_boundary": {"const": "STATIC_ONLY_NO_EXECUTION_AUTHORITY"},
            "decision": {"const": "ACCEPT_STATIC_PACKAGE"},
            "execution_package_file_sha256": sha,
            "fixture_report_sha256": sha,
            "independent_verifier_file_sha256": sha,
            "independent_verifier_report_sha256": sha,
            "official_payload_open_count": {"const": 0},
            "official_target_process_starts": {"const": 0},
            "required_disposition": {"const": DISPOSITION},
            "reviewer_role": {"const": "Fresh-L2"},
            "runtime_namespace_file_count": {"const": 0},
            "static_acceptance_grants_execution_authority": {"const": False},
            "v17_executed": {"const": False},
        },
        "required": [
            "acceptance_sha256", "accepted", "action_id", "artifact_kind", "claim_boundary", "decision",
            "execution_package_file_sha256", "fixture_report_sha256", "independent_verifier_file_sha256",
            "independent_verifier_report_sha256", "official_payload_open_count", "official_target_process_starts",
            "required_disposition", "reviewer_role", "runtime_namespace_file_count",
            "static_acceptance_grants_execution_authority", "v17_executed",
        ],
    }
    return {
        "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_AUTHORITY_SCHEMA.json": authority,
        "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_CREDENTIAL_SCHEMA.json": credential,
        "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_LEDGER_SCHEMA.json": ledger,
        "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json": terminal,
        "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FRESH_L2_ACCEPTANCE_SCHEMA.json": acceptance,
    }


def engine_source() -> bytes:
    return textwrap.dedent(r'''
        #!/usr/bin/env python3
        """Atomic exactly-once lifecycle engine; evaluator behavior is injected."""

        from __future__ import annotations

        import hashlib
        import json
        import os
        from pathlib import Path
        from typing import Any, Callable, NamedTuple


        EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


        class LifecycleError(RuntimeError):
            pass


        class DuplicateInvocationError(LifecycleError):
            pass


        class RuntimePaths(NamedTuple):
            authority: Path
            credential: Path
            ledger: Path
            result: Path
            first_terminal: Path
            fallback_terminal: Path


        class Counts:
            def __init__(self) -> None:
                self.invocation_count = 0
                self.payload_open_count = 0
                self.official_target_process_starts = 0


        class Outcome(NamedTuple):
            counts: Counts
            duplicate_rejected: bool
            result_sha256: str | None
            status: str
            terminal_path: str | None
            terminal_publication_failed: bool


        def compact_bytes(value: Any) -> bytes:
            return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


        def sha256_bytes(raw: bytes) -> str:
            return hashlib.sha256(raw).hexdigest()


        def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
            result = dict(value)
            result[field] = sha256_bytes(compact_bytes(result))
            return result


        def verify_self_hash(value: dict[str, Any], field: str) -> None:
            observed = value.get(field)
            candidate = dict(value)
            candidate.pop(field, None)
            if observed != sha256_bytes(compact_bytes(candidate)):
                raise LifecycleError(f"self hash: {field}")


        def _fsync_directory(path: Path) -> None:
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


        def durable_create(path: Path, raw: bytes, *, fault: bool = False) -> None:
            if fault:
                raise OSError("injected create failure")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o400)
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(descriptor, raw[offset:])
                    if written <= 0:
                        raise OSError("short write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(path.parent)


        def durable_unlink(path: Path) -> None:
            os.unlink(path)
            _fsync_directory(path.parent)


        def _diagnostic(error: BaseException, stage: str) -> dict[str, Any]:
            raw = f"{type(error).__name__}:{stage}".encode("ascii", "replace")
            return {"detail_sha256": sha256_bytes(raw), "exception_type": type(error).__name__, "failure_stage": stage}


        def _authority(bindings: dict[str, Any]) -> dict[str, Any]:
            return sealed({
                "action_id": bindings["action_id"],
                "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_authority",
                "core_acceptance_file_sha256": bindings["core_acceptance_file_sha256"],
                "core_manifest_file_sha256": bindings["core_manifest_file_sha256"],
                "execution_package_file_sha256": bindings["execution_package_file_sha256"],
                "fresh_l2_acceptance_sha256": bindings["fresh_l2_acceptance_sha256"],
                "interpreter_sha256": bindings["interpreter_sha256"],
                "invocation_sha256": bindings["invocation_sha256"],
                "launcher_invocation_sha256": bindings["launcher_invocation_sha256"],
                "official_identities": bindings["official_identities"],
                "single_use": True,
                "state": "READY_UNCONSUMED",
                "transport_attestation": bindings["transport_attestation"],
            }, "authority_sha256")


        def _credential(action_id: str, authority: dict[str, Any]) -> dict[str, Any]:
            nonce = sha256_bytes(("credential\0" + authority["authority_sha256"]).encode("ascii"))
            return sealed({
                "action_id": action_id,
                "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_credential",
                "authority_sha256": authority["authority_sha256"],
                "credential_nonce_sha256": nonce,
                "single_use": True,
                "state": "READY_UNCONSUMED",
            }, "credential_sha256")


        def _ledger(bindings: dict[str, Any], authority: dict[str, Any], credential: dict[str, Any]) -> dict[str, Any]:
            return sealed({
                "action_id": bindings["action_id"],
                "action_retired": True,
                "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_consumed_ledger",
                "authority_sha256": authority["authority_sha256"],
                "credential_consumption": {"credential_sha256": credential["credential_sha256"], "state": "CONSUMED_BEFORE_PAYLOAD"},
                "execution_package_file_sha256": bindings["execution_package_file_sha256"],
                "invocation_cardinality": {"invocation_count_performed_at_publish": 0, "invocation_count_permitted": 1},
                "payload_cardinality": {"payload_open_count_performed_at_publish": 0, "payload_open_count_permitted": 1},
                "post_consumption_ambiguity_default": "CONSUMED_ORPHAN",
                "replay_permitted": False,
                "retry_replay_resume_repair_replacement_permitted": False,
                "transport_attestation_sha256": authority["transport_attestation"]["transport_attestation_sha256"],
            }, "consumed_ledger_sha256")


        def _terminal(
            action_id: str,
            *,
            status: str,
            reason_code: str,
            authority_sha256: str | None,
            ledger_sha256: str | None,
            credential_consumed: bool,
            counts: Counts,
            result_sha256: str | None,
            failure_stage: str | None,
            evaluator_diagnostic: dict[str, Any] | None,
            wrapper_diagnostic: dict[str, Any] | None,
        ) -> dict[str, Any]:
            detail = evaluator_diagnostic or wrapper_diagnostic
            return sealed({
                "action_id": action_id,
                "action_retired": True,
                "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_first_terminal",
                "authority_sha256": authority_sha256,
                "consumed_ledger_sha256": ledger_sha256,
                "credential_consumed": credential_consumed,
                "evaluator_diagnostic": evaluator_diagnostic,
                "failure_detail_sha256": EMPTY_SHA256 if detail is None else sha256_bytes(compact_bytes(detail)),
                "failure_stage": failure_stage,
                "first_record_immutable": True,
                "invocation_count_performed": counts.invocation_count,
                "official_target_process_starts": counts.official_target_process_starts,
                "orphaned_after_consumption": status == "CONSUMED_ORPHAN",
                "payload_open_count": counts.payload_open_count,
                "reason_code": reason_code,
                "result_file_sha256": result_sha256,
                "retry_replay_resume_repair_replacement_permitted": False,
                "status": status,
                "wrapper_diagnostic": wrapper_diagnostic,
            }, "first_terminal_sha256")


        def _publish_preconsumption_terminal(paths: RuntimePaths, validate: Callable[[dict[str, Any]], None], record: dict[str, Any], faults: frozenset[str]) -> tuple[str | None, bool]:
            validate(record)
            try:
                durable_create(paths.first_terminal, compact_bytes(record), fault="TERMINAL_PRIMARY" in faults)
                return str(paths.first_terminal), False
            except FileExistsError:
                return None, True
            except OSError:
                try:
                    durable_create(paths.fallback_terminal, compact_bytes(record), fault="TERMINAL_FALLBACK" in faults)
                    return str(paths.fallback_terminal), False
                except (OSError, FileExistsError):
                    return None, True


        def _publish_postconsumption_terminal(
            paths: RuntimePaths,
            validate: Callable[[dict[str, Any]], None],
            record: dict[str, Any],
            orphan_record: dict[str, Any],
            faults: frozenset[str],
        ) -> tuple[str | None, bool, str]:
            validate(record)
            try:
                durable_create(paths.first_terminal, compact_bytes(record), fault="TERMINAL_PRIMARY" in faults)
                return str(paths.first_terminal), False, record["status"]
            except FileExistsError:
                return None, True, "DUPLICATE_REJECTED"
            except OSError:
                validate(orphan_record)
                try:
                    durable_create(paths.fallback_terminal, compact_bytes(orphan_record), fault="TERMINAL_FALLBACK" in faults)
                    return str(paths.fallback_terminal), False, "CONSUMED_ORPHAN"
                except (OSError, FileExistsError):
                    return None, True, "CONSUMED_ORPHAN_UNPUBLISHED"


        def _lifecycle_exists(paths: RuntimePaths) -> bool:
            return any(os.path.lexists(path) for path in paths)


        def run_execution(
            paths: RuntimePaths,
            bindings: dict[str, Any],
            validators: dict[str, Callable[[dict[str, Any]], None]],
            preflight: Callable[[], Any],
            payload_reader: Callable[[], bytes],
            invoke_once: Callable[[bytes, str, str, Callable[[bytes], None]], Any],
            *,
            faults: frozenset[str] = frozenset(),
        ) -> Outcome:
            counts = Counts()
            if _lifecycle_exists(paths):
                return Outcome(counts, True, None, "DUPLICATE_REJECTED", None, False)
            try:
                preflight()
            except BaseException as error:
                record = _terminal(bindings["action_id"], status="PREFLIGHT_FAILED_TERMINAL", reason_code="READ_ONLY_PREFLIGHT_FAILED",
                    authority_sha256=None, ledger_sha256=None, credential_consumed=False, counts=counts, result_sha256=None,
                    failure_stage="PREFLIGHT", evaluator_diagnostic=None, wrapper_diagnostic=_diagnostic(error, "PREFLIGHT"))
                terminal_path, failed = _publish_preconsumption_terminal(paths, validators["terminal"], record, faults)
                return Outcome(counts, False, None, record["status"], terminal_path, failed)

            authority = _authority(bindings)
            credential = _credential(bindings["action_id"], authority)
            ledger = _ledger(bindings, authority, credential)
            validators["authority"](authority)
            validators["credential"](credential)
            validators["ledger"](ledger)
            stage = "AUTHORITY_MATERIALIZATION"
            try:
                durable_create(paths.authority, compact_bytes(authority), fault="AUTHORITY_WRITE" in faults)
                stage = "CREDENTIAL_MATERIALIZATION"
                durable_create(paths.credential, compact_bytes(credential), fault="CREDENTIAL_WRITE" in faults)
                stage = "LEDGER_MATERIALIZATION"
                durable_create(paths.ledger, compact_bytes(ledger), fault="LEDGER_WRITE" in faults)
            except BaseException as error:
                if os.path.lexists(paths.credential) and not os.path.lexists(paths.ledger):
                    try:
                        durable_unlink(paths.credential)
                    except OSError:
                        pass
                record = _terminal(bindings["action_id"], status="FAILED_TERMINAL", reason_code="LIVE_MATERIALIZATION_FAILED",
                    authority_sha256=authority["authority_sha256"] if os.path.lexists(paths.authority) else None,
                    ledger_sha256=None, credential_consumed=False, counts=counts, result_sha256=None,
                    failure_stage=stage, evaluator_diagnostic=None, wrapper_diagnostic=_diagnostic(error, stage))
                terminal_path, failed = _publish_preconsumption_terminal(paths, validators["terminal"], record, faults)
                return Outcome(counts, False, None, record["status"], terminal_path, failed)

            result_sha256: str | None = None
            try:
                stage = "CREDENTIAL_CONSUMPTION"
                if "CREDENTIAL_UNLINK" in faults:
                    raise OSError("injected credential unlink failure")
                durable_unlink(paths.credential)
                stage = "PAYLOAD_OPEN"
                counts.payload_open_count = 1
                payload = payload_reader()
                if type(payload) is not bytes:
                    raise TypeError("payload ABI")
                stage = "EVALUATOR_CALL"
                counts.invocation_count = 1

                def publish_result(raw: bytes) -> None:
                    nonlocal result_sha256, stage
                    stage = "RESULT_PUBLICATION"
                    durable_create(paths.result, raw, fault="RESULT_WRITE" in faults)
                    result_sha256 = sha256_bytes(raw)

                adapter_result = invoke_once(payload, authority["authority_sha256"], ledger["consumed_ledger_sha256"], publish_result)
                diagnostic = adapter_result.diagnostic
                counts.official_target_process_starts = 1 if diagnostic["process"]["started"] else 0
                record_value = adapter_result.record
                terminal_value = record_value["terminal"]
                record = _terminal(bindings["action_id"], status=terminal_value["status"], reason_code=terminal_value["reason_code"],
                    authority_sha256=authority["authority_sha256"], ledger_sha256=ledger["consumed_ledger_sha256"],
                    credential_consumed=True, counts=counts, result_sha256=result_sha256, failure_stage=None,
                    evaluator_diagnostic=diagnostic, wrapper_diagnostic=None)
                orphan = _terminal(bindings["action_id"], status="CONSUMED_ORPHAN", reason_code="POST_CONSUMPTION_AMBIGUITY",
                    authority_sha256=authority["authority_sha256"], ledger_sha256=ledger["consumed_ledger_sha256"],
                    credential_consumed=True, counts=counts, result_sha256=result_sha256, failure_stage="TERMINAL_PUBLICATION",
                    evaluator_diagnostic=diagnostic, wrapper_diagnostic=_diagnostic(OSError("terminal publication"), "TERMINAL_PUBLICATION"))
                terminal_path, failed, status_value = _publish_postconsumption_terminal(paths, validators["terminal"], record, orphan, faults)
                return Outcome(counts, False, result_sha256, status_value, terminal_path, failed)
            except BaseException as error:
                diagnostic = getattr(error, "diagnostic", None)
                if type(diagnostic) is dict:
                    counts.official_target_process_starts = 1 if diagnostic.get("process", {}).get("started") else 0
                    failure_stage = "RESULT_PUBLICATION" if diagnostic.get("failure_class") == "PUBLICATION" else "EVALUATOR_CALL"
                    wrapper = None
                else:
                    failure_stage = stage
                    wrapper = _diagnostic(error, failure_stage)
                orphan = _terminal(bindings["action_id"], status="CONSUMED_ORPHAN", reason_code="POST_CONSUMPTION_AMBIGUITY",
                    authority_sha256=authority["authority_sha256"], ledger_sha256=ledger["consumed_ledger_sha256"],
                    credential_consumed=True, counts=counts, result_sha256=result_sha256, failure_stage=failure_stage,
                    evaluator_diagnostic=diagnostic if type(diagnostic) is dict else None, wrapper_diagnostic=wrapper)
                terminal_path, failed = _publish_preconsumption_terminal(paths, validators["terminal"], orphan, faults)
                return Outcome(counts, False, result_sha256, "CONSUMED_ORPHAN", terminal_path, failed)
    ''').encode("ascii")


def launcher_source(schema_hashes: dict[str, str]) -> bytes:
    return textwrap.dedent(fr'''
        #!/usr/bin/env python3
        """Irreversible V17 wrapper launcher; static acceptance alone is not authority."""

        from __future__ import annotations

        import argparse
        import hashlib
        import importlib.util
        import json
        import os
        import stat
        import sys
        from pathlib import Path
        from typing import Any

        from qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17 import RuntimePaths, compact_bytes, run_execution, sha256_bytes, verify_self_hash


        ROOT = Path({str(ACTION_ROOT)!r})
        CORE_ROOT = Path({str(CORE_ROOT)!r})
        CORE_TOOLS = CORE_ROOT / "tools"
        RUNTIME_ROOT = Path({str(RUNTIME_ROOT)!r})
        PACKAGE = ROOT / {MANIFEST_NAME!r}
        ACCEPTANCE = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
        INTERPRETER = Path({str(INTERPRETER)!r})
        ACTION_ID = {ACTION_ID!r}
        OFFICIAL_PAYLOAD = Path({str(OFFICIAL_PAYLOAD)!r})
        CORE_MANIFEST = Path({str(CORE_MANIFEST)!r})
        CORE_ACCEPTANCE = Path({str(CORE_ACCEPTANCE)!r})
        CORE_VERIFIER_REPORT = Path({str(CORE_VERIFIER_REPORT)!r})
        EXPECTED_ENVIRONMENT = {EXACT_ENVIRONMENT!r}
        EXPECTED_INTERPRETER_SHA256 = {EXPECTED_INTERPRETER_SHA256!r}
        EXPECTED_CORE_MANIFEST_SHA256 = {EXPECTED_CORE_MANIFEST_SHA256!r}
        EXPECTED_CORE_ACCEPTANCE_SHA256 = {EXPECTED_CORE_ACCEPTANCE_SHA256!r}
        EXPECTED_CORE_VERIFIER_REPORT_SHA256 = {EXPECTED_CORE_VERIFIER_REPORT_SHA256!r}
        SCHEMA_HASHES = {schema_hashes!r}
        PATHS = RuntimePaths(
            authority=RUNTIME_ROOT / "primary/authority/base/authority.json",
            credential=RUNTIME_ROOT / "primary/authority/base/credential.json",
            ledger=RUNTIME_ROOT / "primary/authority/base/authority-ledger.json",
            result=RUNTIME_ROOT / "primary/result/base/result.json",
            first_terminal=RUNTIME_ROOT / "primary/authority/base/first-terminal.json",
            fallback_terminal=RUNTIME_ROOT / "fallback/first-terminal.json",
        )


        class LauncherError(RuntimeError):
            pass


        def require(condition: bool, message: str) -> None:
            if not condition:
                raise LauncherError(message)


        def sha256_file(path: Path) -> str:
            descriptor = os.open(path, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0))
            digest = hashlib.sha256()
            try:
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
            finally:
                os.close(descriptor)
            return digest.hexdigest()


        def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
            raw = path.read_bytes()
            value = json.loads(raw.decode("ascii", "strict"))
            require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {{path}}")
            return value, raw


        def load_module(name: str, path: Path) -> Any:
            spec = importlib.util.spec_from_file_location(name, path)
            require(spec is not None and spec.loader is not None, f"module spec: {{name}}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module


        def load_schemas() -> dict[str, Any]:
            from jsonschema import Draft202012Validator
            mapping = {{
                "authority": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_AUTHORITY_SCHEMA.json",
                "credential": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_CREDENTIAL_SCHEMA.json",
                "ledger": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_LEDGER_SCHEMA.json",
                "terminal": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json",
                "acceptance": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
            }}
            validators = {{}}
            for key, path in mapping.items():
                require(sha256_file(path) == SCHEMA_HASHES[path.name], f"schema hash: {{key}}")
                schema, _ = canonical(path)
                Draft202012Validator.check_schema(schema)
                validator = Draft202012Validator(schema)
                validators[key] = validator.validate
            return validators


        def transport_attestation(package: dict[str, Any]) -> dict[str, Any]:
            parent = os.getppid()
            argv_raw = Path(f"/proc/{{parent}}/cmdline").read_bytes()
            env_raw = Path(f"/proc/{{parent}}/environ").read_bytes()
            executable = Path(f"/proc/{{parent}}/exe").resolve()
            require(executable == INTERPRETER, "transport interpreter")
            record = {{
                "api": "os.posix_spawn",
                "immediate_parent_argv_sha256": sha256_bytes(argv_raw),
                "immediate_parent_environment_sha256": sha256_bytes(env_raw),
                "immediate_parent_executable_sha256": sha256_file(executable),
                "immediate_parent_pid": parent,
                "immediate_parent_transport_sha256": package["generated_files"]["tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v17.py"]["sha256"],
                "launcher_argv_sha256": package["launcher_invocation"]["invocation_sha256"],
                "launcher_environment_sha256": sha256_bytes(compact_bytes(EXPECTED_ENVIRONMENT)),
                "shell": False,
            }}
            record["transport_attestation_sha256"] = sha256_bytes(compact_bytes(record))
            return record


        def read_only_preflight(args: argparse.Namespace, validators: dict[str, Any]) -> dict[str, Any]:
            require(Path.cwd() == ROOT, "launcher cwd")
            require(dict(os.environ) == EXPECTED_ENVIRONMENT, "launcher environment")
            require(Path(sys.executable).resolve() == INTERPRETER, "interpreter path")
            require(sha256_file(INTERPRETER) == EXPECTED_INTERPRETER_SHA256, "interpreter hash")
            require(args.package == str(PACKAGE) and args.acceptance == str(ACCEPTANCE) and args.irreversible_action_id == ACTION_ID, "launcher argv")
            package, package_raw = canonical(PACKAGE)
            verify_self_hash(package, "package_content_sha256")
            require(package["action_identity"]["action_id"] == ACTION_ID, "package action")
            require(sha256_file(CORE_MANIFEST) == EXPECTED_CORE_MANIFEST_SHA256, "core manifest hash")
            require(sha256_file(CORE_ACCEPTANCE) == EXPECTED_CORE_ACCEPTANCE_SHA256, "core acceptance hash")
            require(sha256_file(CORE_VERIFIER_REPORT) == EXPECTED_CORE_VERIFIER_REPORT_SHA256, "core verifier report hash")
            core_acceptance, _ = canonical(CORE_ACCEPTANCE)
            require(core_acceptance["acceptance_sha256"] == {EXPECTED_CORE_ACCEPTANCE_SELF_SHA256!r}, "core acceptance self hash")
            acceptance, acceptance_raw = canonical(ACCEPTANCE)
            validators["acceptance"](acceptance)
            verify_self_hash(acceptance, "acceptance_sha256")
            require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_raw), "wrapper acceptance package binding")
            require(acceptance["static_acceptance_grants_execution_authority"] is False, "static acceptance authority boundary")
            for directory in package["runtime_namespace"]["precreated_directories"]:
                info = os.lstat(directory)
                require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"runtime directory: {{directory}}")
                require(stat.S_IMODE(info.st_mode) == 0o700, f"runtime mode: {{directory}}")
            require(not any(os.path.lexists(path) for path in PATHS), "runtime lifecycle already materialized")
            payload_info = os.lstat(OFFICIAL_PAYLOAD)
            require(stat.S_ISREG(payload_info.st_mode) and not stat.S_ISLNK(payload_info.st_mode), "official payload lstat")
            require(payload_info.st_size == package["official_payload"]["byte_count"], "official payload size")
            core_package, core_package_raw = canonical(CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json")
            return {{
                "acceptance_sha256": sha256_bytes(acceptance_raw),
                "core_package": core_package,
                "core_package_sha256": sha256_bytes(core_package_raw),
                "package": package,
                "package_sha256": sha256_bytes(package_raw),
                "transport_attestation": transport_attestation(package),
            }}


        def read_payload_once(expected_sha256: str, expected_size: int) -> bytes:
            flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
            descriptor = os.open(OFFICIAL_PAYLOAD, flags)
            try:
                info = os.fstat(descriptor)
                require(stat.S_ISREG(info.st_mode) and info.st_size == expected_size, "official payload fstat")
                chunks = []
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(descriptor)
            require(sha256_bytes(raw) == expected_sha256, "official payload hash")
            return raw


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--package", required=True)
            parser.add_argument("--acceptance", required=True)
            parser.add_argument("--irreversible-action-id", required=True)
            args = parser.parse_args()
            validators = load_schemas()
            holder: dict[str, Any] = {{}}

            def preflight() -> None:
                holder.update(read_only_preflight(args, validators))

            preflight()
            package = holder["package"]
            accepted = holder["core_package"]
            records = accepted["official_benchmark"]["input_bindings"]["tensor_records"]
            tensor_bindings = {{record["tensor_name"]: {{"dtype": record["dtype"], "sha256": record["sha256"], "shape": list(record["shape"])}} for record in records.values()}}
            input_bindings = accepted["official_benchmark"]["input_bindings"]
            official = package["official_identities"]
            adapter = load_module("qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17", CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py")
            bridge = load_module("qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17", CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17.py")
            result_validation = load_module("qk_gbfp8_head64_granularity_sweep_result_validation_v17", CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_result_validation_v17.py")
            result_schema = result_validation.frozen_result_schema()
            result_validator = result_validation.construct_result_schema_validator(result_schema)

            bindings = {{
                "action_id": ACTION_ID,
                "core_acceptance_file_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256,
                "core_manifest_file_sha256": EXPECTED_CORE_MANIFEST_SHA256,
                "execution_package_file_sha256": holder["package_sha256"],
                "fresh_l2_acceptance_sha256": holder["acceptance_sha256"],
                "interpreter_sha256": EXPECTED_INTERPRETER_SHA256,
                "invocation_sha256": package["production_evaluator_invocation"]["invocation_sha256"],
                "launcher_invocation_sha256": package["launcher_invocation"]["invocation_sha256"],
                "official_identities": official,
                "transport_attestation": holder["transport_attestation"],
            }}

            def invoke_once(payload: bytes, authority_sha256: str, ledger_sha256: str, publish_result: Any) -> Any:
                result_bindings = result_validation.ResultBindings(
                    acceptance_sha256=holder["acceptance_sha256"], authority_sha256=authority_sha256,
                    evaluator_sha256=package["frozen_core_imports"]["official_evaluator_sha256"],
                    invocation_sha256=package["production_evaluator_invocation"]["invocation_sha256"],
                    ledger_sha256=ledger_sha256, model_identity_sha256=official["model_sha256"],
                    package_sha256=holder["core_package_sha256"], tensor_bundle_sha256=official["tensor_bundle_sha256"],
                )
                bound = result_validation.bind_result_validators(result_validator, result_bindings, ACTION_ID)
                context = {{
                    "authority_sha256": authority_sha256, "consumed_ledger_sha256": ledger_sha256,
                    "evaluator_sha256": package["frozen_core_imports"]["official_evaluator_sha256"],
                    "fresh_l2_acceptance_sha256": holder["acceptance_sha256"], "input_bindings": input_bindings,
                    "invocation_sha256": package["production_evaluator_invocation"]["invocation_sha256"],
                    "irreversible_action_id": ACTION_ID, "model_identity_sha256": official["model_sha256"],
                    "package_sha256": holder["core_package_sha256"],
                }}
                return bridge.invoke_production_evaluator(payload, tensor_bindings, context, bound.validate_result_schema, bound.validate_result_record, publish_result)

            outcome = run_execution(PATHS, bindings, validators, lambda: None,
                lambda: read_payload_once(official["tensor_bundle_sha256"], package["official_payload"]["byte_count"]), invoke_once)
            return 0 if outcome.status == "SUCCEEDED_TERMINAL" else 1


        if __name__ == "__main__":
            raise SystemExit(main())
    ''').encode("ascii")


def transport_source(launcher_hash: str) -> bytes:
    launcher = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py"
    package = ACTION_ROOT / MANIFEST_NAME
    acceptance = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
    argv = [str(INTERPRETER), str(launcher), "--package", str(package), "--acceptance", str(acceptance), "--irreversible-action-id", ACTION_ID]
    return textwrap.dedent(fr'''
        #!/usr/bin/env python3
        """Shell-free exact transport for the irreversible V17 wrapper launcher."""

        import hashlib
        import os
        import sys
        from pathlib import Path

        ROOT = Path({str(ACTION_ROOT)!r})
        INTERPRETER = {str(INTERPRETER)!r}
        LAUNCHER = {str(launcher)!r}
        EXACT_ARGV = {argv!r}
        EXACT_ENVIRONMENT = {EXACT_ENVIRONMENT!r}
        LAUNCHER_SHA256 = {launcher_hash!r}

        def sha256_file(path: str) -> str:
            with open(path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()

        def main() -> int:
            if Path.cwd() != ROOT or dict(os.environ) != EXACT_ENVIRONMENT:
                return 64
            if sys.argv != [__file__, "--package", EXACT_ARGV[3], "--acceptance", EXACT_ARGV[5], "--irreversible-action-id", EXACT_ARGV[7]]:
                return 65
            if sha256_file(LAUNCHER) != LAUNCHER_SHA256:
                return 66
            pid = os.posix_spawn(INTERPRETER, EXACT_ARGV, EXACT_ENVIRONMENT)
            _, status = os.waitpid(pid, 0)
            return os.waitstatus_to_exitcode(status)

        if __name__ == "__main__":
            raise SystemExit(main())
    ''').encode("ascii")


def fixture_source() -> bytes:
    return textwrap.dedent(fr'''
        #!/usr/bin/env python3
        """Marker-free synthetic lifecycle regression for the V17 wrapper."""

        from __future__ import annotations

        import copy
        import importlib.util
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from typing import Any, Callable

        ROOT = Path(__file__).resolve().parents[1]
        CORE_TOOLS = Path({str(CORE_ROOT / 'tools')!r})
        ACTION_ID = {ACTION_ID!r}
        sys.path.insert(0, str(ROOT / "tools"))
        sys.path.insert(0, str(CORE_TOOLS))

        from qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17 import RuntimePaths, compact_bytes, run_execution, sha256_bytes
        import qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17 as adapter
        import qk_gbfp8_head64_granularity_sweep_nominal_records_v17 as nominal
        import qk_gbfp8_head64_granularity_sweep_result_validation_v17 as result_validation


        def load_schema(name: str) -> dict[str, Any]:
            raw = (ROOT / "reference" / name).read_bytes()
            value = json.loads(raw.decode("ascii"))
            if compact_bytes(value) != raw:
                raise ValueError("fixture schema canonical")
            return value


        def validators() -> dict[str, Any]:
            from jsonschema import Draft202012Validator
            names = {{
                "authority": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_AUTHORITY_SCHEMA.json",
                "credential": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_CREDENTIAL_SCHEMA.json",
                "ledger": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_LEDGER_SCHEMA.json",
                "terminal": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json",
            }}
            return {{key: Draft202012Validator(load_schema(name)).validate for key, name in names.items()}}


        def make_paths(root: Path) -> RuntimePaths:
            directories = [root, root / "primary", root / "primary/authority", root / "primary/authority/base", root / "primary/result", root / "primary/result/base", root / "fallback"]
            for directory in directories:
                directory.mkdir(mode=0o700, exist_ok=True)
                os.chmod(directory, 0o700)
            return RuntimePaths(root / "primary/authority/base/authority.json", root / "primary/authority/base/credential.json",
                root / "primary/authority/base/authority-ledger.json", root / "primary/result/base/result.json",
                root / "primary/authority/base/first-terminal.json", root / "fallback/first-terminal.json")


        def bindings() -> dict[str, Any]:
            attestation = {{
                "api": "os.posix_spawn", "immediate_parent_argv_sha256": "1" * 64,
                "immediate_parent_environment_sha256": "2" * 64,
                "immediate_parent_executable_sha256": {EXPECTED_INTERPRETER_SHA256!r},
                "immediate_parent_pid": 1, "immediate_parent_transport_sha256": "3" * 64,
                "launcher_argv_sha256": "4" * 64, "launcher_environment_sha256": "5" * 64,
                "shell": False,
            }}
            attestation["transport_attestation_sha256"] = sha256_bytes(compact_bytes(attestation))
            return {{
                "action_id": ACTION_ID, "core_acceptance_file_sha256": {EXPECTED_CORE_ACCEPTANCE_SHA256!r},
                "core_manifest_file_sha256": {EXPECTED_CORE_MANIFEST_SHA256!r},
                "execution_package_file_sha256": "6" * 64, "fresh_l2_acceptance_sha256": "7" * 64,
                "interpreter_sha256": {EXPECTED_INTERPRETER_SHA256!r}, "invocation_sha256": "8" * 64,
                "launcher_invocation_sha256": "9" * 64,
                "official_identities": {{"input_tokens_sha256": "a" * 64, "lane_metadata_sha256": "b" * 64,
                    "model_sha256": "c" * 64, "tensor_bundle_sha256": "d" * 64}},
                "transport_attestation": attestation,
            }}


        def result_bytes(authority_sha256: str, ledger_sha256: str, *, invalid_schema: bool = False) -> tuple[bytes, Any]:
            value = nominal.make_result("G4")
            value["authority_sha256"] = authority_sha256
            value["consumed_ledger_sha256"] = ledger_sha256
            value["evaluator_sha256"] = "e" * 64
            value["fresh_l2_acceptance_sha256"] = "7" * 64
            value["invocation_sha256"] = "8" * 64
            value["model_identity_sha256"] = "c" * 64
            value["package_sha256"] = "f" * 64
            value["input_bindings"]["tensor_bundle_sha256"] = "d" * 64
            if invalid_schema:
                value["candidate_results"][0]["metrics"]["top_key"]["row_count"] = 573
            value.pop("result_sha256", None)
            value["result_sha256"] = sha256_bytes(compact_bytes(value))
            schema_validator = result_validation.construct_result_schema_validator(result_validation.frozen_result_schema())
            bound = result_validation.bind_result_validators(schema_validator, result_validation.ResultBindings(
                acceptance_sha256="7" * 64, authority_sha256=authority_sha256, evaluator_sha256="e" * 64,
                invocation_sha256="8" * 64, ledger_sha256=ledger_sha256, model_identity_sha256="c" * 64,
                package_sha256="f" * 64, tensor_bundle_sha256="d" * 64), ACTION_ID)
            return compact_bytes(value), bound


        def invoker(scenario: str) -> Callable[..., Any]:
            def invoke(payload: bytes, authority_sha256: str, ledger_sha256: str, publish: Any) -> Any:
                invalid = scenario == "schema"
                raw, bound = result_bytes(authority_sha256, ledger_sha256, invalid_schema=invalid)
                if scenario == "decode":
                    raw = b"{{malformed\n"
                return_code = 7 if scenario == "nonzero" else 0
                stderr = b"synthetic nonzero\n" if scenario == "nonzero" else b""
                completed = subprocess.CompletedProcess(["synthetic"], return_code, stdout=raw, stderr=stderr)
                spec = adapter.InvocationSpec(argv=("synthetic-v17-adapter", scenario), cwd="/synthetic", environment=(("LANG", "C"),), max_capture_bytes=1 << 20)
                return adapter.invoke_evaluator(spec, b"synthetic-envelope", bound.validate_result_schema, bound.validate_result_record, publish, runner=lambda *args, **kwargs: completed)
            return invoke


        def read_terminal(path: Path) -> dict[str, Any]:
            return json.loads(path.read_text(encoding="ascii"))


        def run_case(name: str, operation: Callable[[RuntimePaths], dict[str, Any]]) -> dict[str, Any]:
            with tempfile.TemporaryDirectory(prefix="v17-wrapper-fixture-") as temporary:
                paths = make_paths(Path(temporary) / "runtime")
                try:
                    observed = operation(paths)
                    passed = bool(observed.pop("passed"))
                    return {{"name": name, "passed": passed, **observed}}
                except BaseException as error:
                    return {{"name": name, "passed": False, "error_type": type(error).__name__}}


        def execute(paths: RuntimePaths, scenario: str = "nominal", *, faults: frozenset[str] = frozenset(), preflight_error: bool = False) -> Any:
            def preflight() -> None:
                if preflight_error:
                    raise ValueError("synthetic preflight")
            return run_execution(paths, bindings(), validators(), preflight, lambda: b"synthetic-payload", invoker(scenario), faults=faults)


        def run_fixture() -> dict[str, Any]:
            cases = []
            cases.append(run_case("nominal", lambda p: (lambda o: {{"passed": o.status == "SUCCEEDED_TERMINAL" and o.counts.invocation_count == 1 and o.counts.payload_open_count == 1 and p.result.exists() and p.ledger.exists() and not p.credential.exists(), "status": o.status}})(execute(p))))
            cases.append(run_case("preflight_failure", lambda p: (lambda o: {{"passed": o.status == "PREFLIGHT_FAILED_TERMINAL" and o.counts.invocation_count == 0 and not p.authority.exists(), "status": o.status}})(execute(p, preflight_error=True))))
            cases.append(run_case("authority_failure", lambda p: (lambda o: {{"passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and not p.authority.exists(), "status": o.status}})(execute(p, faults=frozenset({{"AUTHORITY_WRITE"}})))))
            cases.append(run_case("credential_failure", lambda p: (lambda o: {{"passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and p.authority.exists() and not p.credential.exists() and not p.ledger.exists(), "status": o.status}})(execute(p, faults=frozenset({{"CREDENTIAL_WRITE"}})))))
            cases.append(run_case("ledger_failure", lambda p: (lambda o: {{"passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and not p.credential.exists() and not p.ledger.exists(), "status": o.status}})(execute(p, faults=frozenset({{"LEDGER_WRITE"}})))))

            def duplicate(p: RuntimePaths) -> dict[str, Any]:
                first = execute(p)
                before = p.first_terminal.read_bytes()
                second = execute(p)
                return {{"passed": first.status == "SUCCEEDED_TERMINAL" and second.duplicate_rejected and second.counts.invocation_count == 0 and p.first_terminal.read_bytes() == before, "status": second.status}}
            cases.append(run_case("duplicate_invocation", duplicate))
            for name, scenario, failure_class in (("nonzero_evaluator", "nonzero", "RETURN_CODE"), ("decode_failure", "decode", "DECODE"), ("schema_failure", "schema", "RESULT_SCHEMA")):
                def operation(p: RuntimePaths, scenario: str = scenario, failure_class: str = failure_class) -> dict[str, Any]:
                    outcome = execute(p, scenario)
                    terminal = read_terminal(Path(outcome.terminal_path))
                    return {{"passed": outcome.status == "CONSUMED_ORPHAN" and outcome.counts.invocation_count == 1 and terminal["evaluator_diagnostic"]["failure_class"] == failure_class, "failure_class": terminal["evaluator_diagnostic"]["failure_class"], "status": outcome.status}}
                cases.append(run_case(name, operation))
            cases.append(run_case("result_publication_failure", lambda p: (lambda o: {{"passed": o.status == "CONSUMED_ORPHAN" and o.counts.invocation_count == 1 and read_terminal(Path(o.terminal_path))["evaluator_diagnostic"]["failure_class"] == "PUBLICATION", "status": o.status}})(execute(p, faults=frozenset({{"RESULT_WRITE"}})))))
            cases.append(run_case("terminal_primary_fallback", lambda p: (lambda o: {{"passed": o.status == "CONSUMED_ORPHAN" and o.terminal_path == str(p.fallback_terminal) and p.result.exists(), "status": o.status}})(execute(p, faults=frozenset({{"TERMINAL_PRIMARY"}})))))
            cases.append(run_case("terminal_total_failure", lambda p: (lambda o: {{"passed": o.status == "CONSUMED_ORPHAN_UNPUBLISHED" and o.terminal_publication_failed and o.counts.invocation_count == 1 and not p.first_terminal.exists() and not p.fallback_terminal.exists(), "status": o.status}})(execute(p, faults=frozenset({{"TERMINAL_PRIMARY", "TERMINAL_FALLBACK"}})))))
            passed = all(case["passed"] for case in cases)
            return {{
                "action_id": ACTION_ID,
                "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_synthetic_fixture_report",
                "case_count": len(cases),
                "cases": cases,
                "official_payload_open_count": 0,
                "official_target_process_starts": 0,
                "production_mode_exercised": False,
                "shared_adapter_sha256": {sha256_file(CORE_ROOT / 'tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py')!r},
                "status": "PASS_V17_EXACTLY_ONCE_SYNTHETIC_FIXTURE" if passed else "FAIL_V17_EXACTLY_ONCE_SYNTHETIC_FIXTURE",
                "synthetic_bytes_only": True,
            }}


        def main() -> int:
            report = run_fixture()
            sys.stdout.buffer.write(compact_bytes(report))
            return 0 if report["status"].startswith("PASS_") else 1


        if __name__ == "__main__":
            raise SystemExit(main())
    ''').encode("ascii")


def build() -> dict[str, Any]:
    require(not ACTION_ROOT.exists(), f"immutable wrapper root already exists: {ACTION_ROOT}")
    require(not BUILD_ROOT.exists(), f"attempt build root already exists: {BUILD_ROOT}")
    require(VERIFY_TOOL.is_file(), f"independent verifier absent: {VERIFY_TOOL}")
    require(sha256_file(CORE_MANIFEST) == EXPECTED_CORE_MANIFEST_SHA256, "accepted core manifest hash")
    require(sha256_file(CORE_ACCEPTANCE) == EXPECTED_CORE_ACCEPTANCE_SHA256, "accepted core acceptance raw hash")
    require(sha256_file(CORE_VERIFIER_REPORT) == EXPECTED_CORE_VERIFIER_REPORT_SHA256, "accepted core verifier raw hash")
    require(sha256_file(INTERPRETER) == EXPECTED_INTERPRETER_SHA256, "interpreter hash")
    core_package, _ = canonical_object(CORE_MANIFEST)
    require(core_package["package_content_sha256"] == EXPECTED_CORE_CONTENT_SHA256, "accepted core content hash")
    core_acceptance, _ = canonical_object(CORE_ACCEPTANCE)
    require(core_acceptance["acceptance_sha256"] == EXPECTED_CORE_ACCEPTANCE_SELF_SHA256, "accepted core acceptance self hash")
    require(core_acceptance["static_acceptance_grants_execution_authority"] is False, "accepted core authority boundary")
    observed_runtime_files = [path for path in RUNTIME_ROOT.rglob("*") if path.is_file()]
    require(not observed_runtime_files, "future runtime is not directories-only")
    require({str(path) for path in [RUNTIME_ROOT, *[Path(item) for item in runtime_directories()[1:]]]} == set(runtime_directories()), "runtime declaration")

    core_terminal_schema, _ = canonical_object(CORE_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_FIRST_TERMINAL_SCHEMA.json")
    schemas = lifecycle_schemas(core_terminal_schema)
    temporary = Path(tempfile.mkdtemp(prefix=f".{ROOT_ID}.", dir=ACTION_ROOT.parent))
    try:
        for relative, value in schemas.items():
            write_json(temporary / relative, value)
        schema_hashes = {Path(relative).name: sha256_file(temporary / relative) for relative in schemas}
        write_bytes(temporary / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17.py", engine_source())
        write_bytes(temporary / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py", launcher_source(schema_hashes))
        launcher_hash = sha256_file(temporary / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py")
        write_bytes(temporary / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v17.py", transport_source(launcher_hash))
        write_bytes(temporary / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v17.py", fixture_source())
        for generated_python in sorted((temporary / "tools").glob("*.py")):
            ast.parse(generated_python.read_text(encoding="ascii"), filename=str(generated_python))

        completed = subprocess.run(
            [str(INTERPRETER), str(temporary / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v17.py")],
            cwd=temporary,
            env=EXACT_ENVIRONMENT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(completed.returncode == 0 and not completed.stderr, f"synthetic fixture failed: {completed.stderr.decode('ascii', 'replace')}")
        fixture_report = json.loads(completed.stdout.decode("ascii", "strict"))
        require(compact_bytes(fixture_report) == completed.stdout, "fixture report canonical")
        require(fixture_report["status"] == "PASS_V17_EXACTLY_ONCE_SYNTHETIC_FIXTURE", "fixture report status")
        write_bytes(temporary / "evidence/SYNTHETIC_EXACTLY_ONCE_FIXTURE_REPORT.json", completed.stdout)

        generated: dict[str, dict[str, Any]] = {}
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                relative = path.relative_to(temporary).as_posix()
                generated[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}

        accepted = core_package["preservation"]
        preservation_roots = [CORE_ROOT]
        for value in accepted.values():
            if type(value) is dict and type(value.get("root")) is str:
                preservation_roots.append(Path(value["root"]))
        preservation = []
        seen: set[str] = set()
        for root in preservation_roots:
            if str(root) not in seen:
                preservation.append(inventory_digest(root))
                seen.add(str(root))

        accepted_package, accepted_package_raw = canonical_object(CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json")
        official_inputs = accepted_package["official_benchmark"]["input_bindings"]
        official_identities = {
            "input_tokens_sha256": official_inputs["input_token_ids_sha256"],
            "lane_metadata_sha256": official_inputs["lane_metadata"]["sha256"],
            "model_sha256": accepted_package["official_benchmark"]["model_identity_sha256"],
            "tensor_bundle_sha256": official_inputs["tensor_bundle"]["sha256"],
        }
        core_adapter = core_package["production_evaluator_adapter"]
        production_record = {"argv": core_adapter["argv"], "cwd": core_adapter["cwd"], "environment": core_adapter["environment"], "shell": False}
        launcher = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py"
        transport = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v17.py"
        acceptance_path = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
        launcher_argv = [str(INTERPRETER), str(launcher), "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(acceptance_path), "--irreversible-action-id", ACTION_ID]
        transport_argv = [str(INTERPRETER), str(transport), "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(acceptance_path), "--irreversible-action-id", ACTION_ID]
        manifest = {
            "action_identity": {"action_id": ACTION_ID, "accepted_core_is_unchanged": True, "retry_or_resume": False, "wrapper_is_additive_and_distinct": True},
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_package",
            "claim_boundary": {
                "authority_materialized": False, "credential_materialized": False, "execution_authorized": False,
                "invocation_count": 0, "official_payload_open_count": 0, "official_target_process_starts": 0,
                "result_materialized": False, "runtime_namespace_file_count": 0,
                "static_acceptance_materialized": False, "v17_executed": False,
            },
            "core_binding": {
                "acceptance_file_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256,
                "acceptance_self_sha256": EXPECTED_CORE_ACCEPTANCE_SELF_SHA256,
                "independent_verifier_report_file_sha256": EXPECTED_CORE_VERIFIER_REPORT_SHA256,
                "manifest_file_sha256": EXPECTED_CORE_MANIFEST_SHA256,
                "package_content_sha256": EXPECTED_CORE_CONTENT_SHA256,
                "root": str(CORE_ROOT),
            },
            "frozen_core_imports": {
                "adapter": {"path": str(CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py"), "sha256": sha256_file(CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py")},
                "bridge": {"path": str(CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17.py"), "sha256": sha256_file(CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17.py")},
                "official_evaluator_sha256": accepted_package["static_bindings"]["evaluator"]["sha256"],
                "result_validation": {"path": str(CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_result_validation_v17.py"), "sha256": sha256_file(CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_result_validation_v17.py")},
                "result_schema_sha256": core_package["frozen_result_validation"]["result_schema_sha256"],
                "worker": {"path": str(CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py"), "sha256": sha256_file(CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py")},
            },
            "generated_files": generated,
            "interpreter": {"path": str(INTERPRETER), "sha256": EXPECTED_INTERPRETER_SHA256, "version": "3.13.5"},
            "launcher_invocation": {"argv": launcher_argv, "cwd": str(ACTION_ROOT), "environment": EXACT_ENVIRONMENT, "invocation_sha256": sha256_bytes(compact_bytes({"argv": launcher_argv, "cwd": str(ACTION_ROOT), "environment": EXACT_ENVIRONMENT, "shell": False})), "shell": False},
            "lifecycle": {
                "authority_credential_ledger_validate_before_materialization": True,
                "consumption_order": ["authority_create_once", "credential_create_once", "ledger_create_once", "credential_durable_unlink", "payload_open_once", "evaluator_invoke_once"],
                "duplicate_invocation_rejected_without_invocation": True,
                "first_terminal_publication": "PRIMARY_CREATE_ONLY_THEN_FALLBACK_CREATE_ONLY",
                "invocation_count_maximum": 1,
                "payload_open_count_maximum": 1,
                "post_consumption_failure": "CONSUMED_ORPHAN",
                "preconsumption_failure": "TERMINAL_RETIRE_WITH_ZERO_INVOCATION",
                "result_publication": "CREATE_ONLY_FSYNC_FILE_AND_PARENT",
                "retry_replay_resume_repair_replacement_permitted": False,
            },
            "official_identities": official_identities,
            "official_payload": {"access_during_static_verification": "PROHIBITED", "byte_count": official_inputs["tensor_bundle"]["byte_count"], "path": str(OFFICIAL_PAYLOAD), "sha256": official_inputs["tensor_bundle"]["sha256"]},
            "preattempt_synthetic_fixture": {"case_count": fixture_report["case_count"], "official_payload_open_count": 0, "official_target_process_starts": 0, "production_mode_exercised": False, "report_path": str(ACTION_ROOT / "evidence/SYNTHETIC_EXACTLY_ONCE_FIXTURE_REPORT.json"), "report_sha256": sha256_bytes(completed.stdout)},
            "preservation": preservation,
            "production_evaluator_invocation": {**production_record, "invocation_sha256": sha256_bytes(compact_bytes(production_record))},
            "required_disposition": DISPOSITION,
            "root_id": ROOT_ID,
            "runtime_namespace": {
                "fallback_terminal": str(RUNTIME_ROOT / "fallback/first-terminal.json"),
                "file_count": 0,
                "paths": {
                    "authority": str(RUNTIME_ROOT / "primary/authority/base/authority.json"),
                    "credential": str(RUNTIME_ROOT / "primary/authority/base/credential.json"),
                    "first_terminal": str(RUNTIME_ROOT / "primary/authority/base/first-terminal.json"),
                    "ledger": str(RUNTIME_ROOT / "primary/authority/base/authority-ledger.json"),
                    "result": str(RUNTIME_ROOT / "primary/result/base/result.json"),
                },
                "precreated_directories": runtime_directories(),
                "runtime_root": str(RUNTIME_ROOT),
            },
            "schema_version": 1,
            "static_acceptance": {"grants_execution_authority": False, "path": str(acceptance_path), "present": False, "schema_path": str(ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FRESH_L2_ACCEPTANCE_SCHEMA.json")},
            "transport_invocation": {"argv": transport_argv, "cwd": str(ACTION_ROOT), "environment": EXACT_ENVIRONMENT, "invocation_sha256": sha256_bytes(compact_bytes({"argv": transport_argv, "cwd": str(ACTION_ROOT), "environment": EXACT_ENVIRONMENT, "shell": False})), "shell": False},
            "verifier": {"path": str(VERIFY_TOOL), "sha256": sha256_file(VERIFY_TOOL)},
        }
        manifest["package_content_sha256"] = sha256_bytes(compact_bytes(manifest))
        write_json(temporary / MANIFEST_NAME, manifest)

        for path in sorted(temporary.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, ACTION_ROOT)
    except BaseException:
        if temporary.exists():
            for path in temporary.rglob("*"):
                try:
                    os.chmod(path, 0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            os.chmod(temporary, 0o755)
            shutil.rmtree(temporary)
        raise

    BUILD_ROOT.mkdir(parents=True, exist_ok=False)
    manifest_path = ACTION_ROOT / MANIFEST_NAME
    fixture_path = ACTION_ROOT / "evidence/SYNTHETIC_EXACTLY_ONCE_FIXTURE_REPORT.json"
    report = {
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_build_report",
        "core_manifest_sha256": sha256_file(CORE_MANIFEST),
        "fixture_report_sha256": sha256_file(fixture_path),
        "manifest_sha256": sha256_file(manifest_path),
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "runtime_namespace_file_count": 0,
        "status": "PASS_V17_EXACTLY_ONCE_WRAPPER_BUILD",
        "wrapper_root": str(ACTION_ROOT),
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    write_json(BUILD_ROOT / "build-report.json", report)
    review_request = {
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_fresh_l2_static_review_request",
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "manifest_path": str(manifest_path),
        "manifest_sha256": report["manifest_sha256"],
        "required_disposition": DISPOSITION,
        "reviewer_must_rerun": str(VERIFY_TOOL),
        "static_acceptance_grants_execution_authority": False,
    }
    write_json(BUILD_ROOT / "fresh-l2-review-request.json", review_request)
    return report


def main() -> int:
    report = build()
    print(compact_bytes(report).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
