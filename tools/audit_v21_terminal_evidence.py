#!/home/argustest/miniconda3/bin/python3.13
"""Read-only verification of the sole accepted V21 terminal and result."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ROOT = PROJECT_ROOT / (
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_"
    "order_independent_binding_terminal_coverage_action_root"
)
TOOLS = ACTION_ROOT / "tools"
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / (
    "qk_gbfp8_head64_granularity_sweep_execution_v21_"
    "order_independent_binding_terminal_coverage_289140ba"
)
PACKAGE = ACTION_ROOT / (
    "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_"
    "ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json"
)
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
OWNER = RUNTIME_ROOT / "primary/authority/base/owner-claim.json"
AUTHORITY = RUNTIME_ROOT / "primary/authority/base/authority.json"
CREDENTIAL = RUNTIME_ROOT / "primary/authority/base/credential.json"
LEDGER = RUNTIME_ROOT / "primary/authority/base/authority-ledger.json"
TERMINAL = RUNTIME_ROOT / "primary/authority/base/first-terminal.json"
FALLBACK = RUNTIME_ROOT / "fallback/first-terminal.json"
RESULT = RUNTIME_ROOT / "primary/result/base/result.json"
ACTION_ID = "ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001"
EXPECTED_PACKAGE_SHA256 = "45caa0015e55979dd6176a319792a3ae3a4da9f25a19baf9bec69b98de828d69"
EXPECTED_ACCEPTANCE_SHA256 = "19663777b84352691030325189ffb397b6386b5443200010cda0e6cd4a286c48"
EXPECTED_RUNTIME_FILES = {OWNER, AUTHORITY, LEDGER, TERMINAL, RESULT}
EXPECTED_V20 = {
    PROJECT_ROOT
    / "reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root/"
    "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json":
        "e1ab2b55dd07b6cfc9f8540d42eb3fd9b387cb57aadf0704933a29fe6c05c314",
    PROJECT_ROOT
    / "reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root/"
    "review/FRESH_L2_STATIC_ACCEPTANCE.json":
        "516f3007e0bedc00b98748d835b5240fb09b08b189ccc87286aef9ea1bfbe963",
    PROJECT_ROOT / "build/v20-bound-path-repair-attempt-0001/sole-v20-transport-terminal-observation.json":
        "289140ba983b6808ef0a2bb56381579857ecfc50c35349e96c266ac2b0c75d33",
    PROJECT_ROOT / "build/v20-bound-path-repair-attempt-0001/sole-v20-transport-stderr.log":
        "6470257db40ccc24d46277d34818419f11cc1a4752dba8263d6c742d321a9e4d",
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


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


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def identity(path: Path) -> tuple[dict[str, Any], bytes]:
    info = os.lstat(path)
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"not regular: {path}")
    raw = path.read_bytes()
    return {
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "size": len(raw),
    }, raw


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    _, raw = identity(path)
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}")
    return value, raw


def main() -> int:
    sys.path.insert(0, str(TOOLS))
    import qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21 as launcher
    import qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v21 as engine
    import qk_gbfp8_head64_granularity_sweep_result_validation_v21 as result_validation

    package, package_raw = canonical(PACKAGE)
    acceptance, acceptance_raw = canonical(ACCEPTANCE)
    require(sha256_bytes(package_raw) == EXPECTED_PACKAGE_SHA256, "V21 package drift")
    require(sha256_bytes(acceptance_raw) == EXPECTED_ACCEPTANCE_SHA256, "V21 acceptance drift")
    engine.verify_self_hash(package, "package_content_sha256")
    engine.verify_self_hash(acceptance, "acceptance_sha256")

    actual_files = {path for path in RUNTIME_ROOT.rglob("*") if path.is_file()}
    require(actual_files == EXPECTED_RUNTIME_FILES, "unexpected V21 runtime file inventory")
    require(not os.path.lexists(CREDENTIAL), "credential was not consumed")
    require(not os.path.lexists(FALLBACK), "fallback terminal exists")

    owner, owner_raw = canonical(OWNER)
    authority, authority_raw = canonical(AUTHORITY)
    ledger, ledger_raw = canonical(LEDGER)
    terminal, terminal_raw = canonical(TERMINAL)
    result, result_raw = canonical(RESULT)
    engine.verify_self_hash(owner, "owner_claim_sha256")
    engine.verify_self_hash(authority, "authority_sha256")
    engine.verify_self_hash(ledger, "consumed_ledger_sha256")
    engine.verify_self_hash(terminal, "first_terminal_sha256")

    validators = launcher.load_schemas()
    validators["owner"](owner)
    validators["authority"](authority)
    validators["ledger"](ledger)
    validators["terminal"](terminal)

    transport_argv = package["transport_invocation"]["argv"]
    transport_environment = package["transport_invocation"]["environment"]
    observed_attestation = authority["transport_attestation"]
    expected_argv_raw = b"\0".join(item.encode("utf-8") for item in transport_argv) + b"\0"
    expected_environment_raw = b"\0".join(
        f"{key}={value}".encode("utf-8") for key, value in transport_environment.items()
    ) + b"\0"
    require(
        observed_attestation["immediate_parent_argv_sha256"] == sha256_bytes(expected_argv_raw),
        "executed transport argv attestation",
    )
    require(
        observed_attestation["immediate_parent_environment_sha256"] == sha256_bytes(expected_environment_raw),
        "executed transport environment attestation",
    )
    require(
        observed_attestation["immediate_parent_transport_sha256"]
        == package["generated_files"]["tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v21.py"]["sha256"],
        "executed transport identity attestation",
    )
    require(
        observed_attestation["launcher_argv_sha256"] == package["launcher_invocation"]["invocation_sha256"],
        "launcher argv attestation",
    )

    core_package_path = launcher.CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
    _, core_package_raw = canonical(core_package_path)
    bindings = result_validation.ResultBindings(
        acceptance_sha256=sha256_bytes(acceptance_raw),
        authority_sha256=authority["authority_sha256"],
        evaluator_sha256=package["core_manifest_closure"]["official_evaluator_sha256"],
        invocation_sha256=package["production_evaluator_invocation"]["invocation_sha256"],
        ledger_sha256=ledger["consumed_ledger_sha256"],
        model_identity_sha256=package["official_identities"]["model_sha256"],
        package_sha256=sha256_bytes(core_package_raw),
        tensor_bundle_sha256=package["official_identities"]["tensor_bundle_sha256"],
    )
    result_validator = result_validation.construct_result_schema_validator(
        result_validation.frozen_result_schema()
    )
    bound = result_validation.bind_result_validators(result_validator, bindings, ACTION_ID)
    bound.validate_result_schema(result)
    bound.validate_result_record(result, result_raw)

    require(terminal["status"] == "FAILED_TERMINAL", "terminal status")
    require(terminal["reason_code"] == "HARD_THRESHOLD_FAILED", "terminal reason")
    require(terminal["invocation_count_performed"] == 1, "terminal invocation count")
    require(terminal["payload_open_count"] == 1, "terminal payload count")
    require(terminal["official_target_process_starts"] == 1, "terminal evaluator count")
    require(terminal["result_file_sha256"] == sha256_bytes(result_raw), "terminal result binding")
    require(terminal["authority_sha256"] == authority["authority_sha256"], "terminal authority binding")
    require(terminal["consumed_ledger_sha256"] == ledger["consumed_ledger_sha256"], "terminal ledger binding")
    require(result["selected_candidate"] is None, "unexpected passing candidate")
    require(result["selection"]["passing_candidates"] == [], "unexpected passing candidate set")
    require(all(not item["threshold_evaluation"]["all_hard_gates_pass"] for item in result["candidate_results"]), "hard-gate classification")

    v20_records = []
    for path, expected in EXPECTED_V20.items():
        record, _ = identity(path)
        record["expected_sha256"] = expected
        record["unchanged"] = record["sha256"] == expected
        require(record["unchanged"], f"V20 drift: {path}")
        v20_records.append(record)

    runtime_identities = {}
    for label, path, raw in (
        ("owner", OWNER, owner_raw),
        ("authority", AUTHORITY, authority_raw),
        ("ledger", LEDGER, ledger_raw),
        ("result", RESULT, result_raw),
        ("terminal", TERMINAL, terminal_raw),
    ):
        info = os.lstat(path)
        runtime_identities[label] = {
            "mode": f"{stat.S_IMODE(info.st_mode):04o}",
            "path": str(path),
            "sha256": sha256_bytes(raw),
            "size": len(raw),
        }

    record: dict[str, Any] = {
        "action_id": ACTION_ID,
        "artifact_kind": "ace2_v21_sole_transport_terminal_observation",
        "claim_boundary": "READ_ONLY_EXTERNAL_OBSERVATION_NOT_A_FRESH_REVIEWER_DECISION",
        "executed_transport_attestation": {
            "argv_matches_accepted_contract": True,
            "environment_matches_accepted_contract": True,
            "launcher_identity_matches_accepted_package": True,
            "transport_identity_matches_accepted_package": True,
        },
        "fresh_reviewer_status": "PENDING",
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "result": {
            "candidate_count": len(result["candidate_results"]),
            "passing_candidates": result["selection"]["passing_candidates"],
            "schema_and_record_valid": True,
            "selected_candidate": result["selected_candidate"],
        },
        "runtime": {
            "credential_consumed_and_absent": True,
            "fallback_terminal_absent": True,
            "file_count": len(actual_files),
            "identities": runtime_identities,
        },
        "schema_version": 1,
        "terminal": {
            "invocation_count_performed": terminal["invocation_count_performed"],
            "official_target_process_starts": terminal["official_target_process_starts"],
            "payload_open_count": terminal["payload_open_count"],
            "reason_code": terminal["reason_code"],
            "schema_and_self_hash_valid": True,
            "status": terminal["status"],
            "transport_exit_code": 1,
            "transport_stderr_sha256": sha256_bytes(b""),
            "transport_stdout_sha256": sha256_bytes(b""),
        },
        "v20_immutability": {
            "all_critical_records_unchanged": True,
            "records": v20_records,
        },
        "verification_status": "PASS_SCHEMA_VALID_V21_FAILED_TERMINAL_HARD_THRESHOLD_FAILED",
    }
    record["record_sha256"] = sha256_bytes(compact_bytes(record))
    sys.stdout.buffer.write(compact_bytes(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
