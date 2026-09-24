#!/usr/bin/env python3
"""Non-consuming Manager admission audit for the sole accepted V21 execution."""

from __future__ import annotations

import argparse
import ast
import errno
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any


PROJECT_ROOT = "/home/argustest/ace-2"
ACTION_ROOT = (
    PROJECT_ROOT
    + "/reference/qk_gbfp8_head64_granularity_sweep_execution_v21_"
    "order_independent_binding_terminal_coverage_action_root"
)
PACKAGE = (
    ACTION_ROOT
    + "/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_"
    "ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json"
)
ACCEPTANCE = ACTION_ROOT + "/review/FRESH_L2_STATIC_ACCEPTANCE.json"
TRANSPORT = ACTION_ROOT + "/tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v21.py"
LAUNCHER = ACTION_ROOT + "/tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21.py"
INTERPRETER = "/home/argustest/miniconda3/bin/python3.13"
PIPELINE_STATE = PROJECT_ROOT + "/research/PIPELINE_STATE.json"
RUNTIME_PARENT = PROJECT_ROOT + "/runtime"
RUNTIME_NAME = (
    "qk_gbfp8_head64_granularity_sweep_execution_v21_"
    "order_independent_binding_terminal_coverage_289140ba"
)
ACTION_ID = "ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001"
MISSION_ID = "0d56568407e7"

EXPECTED_PACKAGE_SHA256 = "45caa0015e55979dd6176a319792a3ae3a4da9f25a19baf9bec69b98de828d69"
EXPECTED_ACCEPTANCE_SHA256 = "19663777b84352691030325189ffb397b6386b5443200010cda0e6cd4a286c48"
EXPECTED_INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
EXPECTED_V20 = {
    PROJECT_ROOT
    + "/reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root/"
    "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json":
        "e1ab2b55dd07b6cfc9f8540d42eb3fd9b387cb57aadf0704933a29fe6c05c314",
    PROJECT_ROOT
    + "/reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root/"
    "review/FRESH_L2_STATIC_ACCEPTANCE.json":
        "516f3007e0bedc00b98748d835b5240fb09b08b189ccc87286aef9ea1bfbe963",
    PROJECT_ROOT + "/build/v20-bound-path-repair-attempt-0001/sole-v20-transport-terminal-observation.json":
        "289140ba983b6808ef0a2bb56381579857ecfc50c35349e96c266ac2b0c75d33",
    PROJECT_ROOT + "/build/v20-bound-path-repair-attempt-0001/sole-v20-transport-stderr.log":
        "6470257db40ccc24d46277d34818419f11cc1a4752dba8263d6c742d321a9e4d",
}


class AdmissionError(RuntimeError):
    pass


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def open_dir_no_follow(path: str) -> int:
    pure = PurePosixPath(path)
    if not pure.is_absolute():
        raise AdmissionError(f"non-absolute directory: {path}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in pure.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_file_no_follow(path: str) -> int:
    pure = PurePosixPath(path)
    parent_descriptor = open_dir_no_follow(str(pure.parent))
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(pure.name, flags, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise AdmissionError(f"not a regular file: {path}")
    return descriptor


def read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def identity(path: str, *, include_raw: bool = False) -> tuple[dict[str, Any], bytes | None]:
    descriptor = open_file_no_follow(path)
    try:
        info = os.fstat(descriptor)
        raw = read_descriptor(descriptor)
        record = {
            "path": path,
            "device": info.st_dev,
            "inode": info.st_ino,
            "mode": f"{stat.S_IMODE(info.st_mode):04o}",
            "size": info.st_size,
            "sha256": sha256_bytes(raw),
            "opened_no_follow_from_directory_descriptors": True,
        }
        return record, raw if include_raw else None
    finally:
        os.close(descriptor)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def verify_self_hash(record: dict[str, Any], field: str) -> None:
    observed = record.get(field)
    candidate = dict(record)
    candidate.pop(field, None)
    require(observed == sha256_bytes(compact_bytes(candidate)), f"self hash: {field}")


def runtime_absence() -> dict[str, Any]:
    parent_descriptor = open_dir_no_follow(RUNTIME_PARENT)
    try:
        parent_info = os.fstat(parent_descriptor)
        try:
            os.stat(RUNTIME_NAME, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            absent = True
        else:
            absent = False
        return {
            "runtime_root": RUNTIME_PARENT + "/" + RUNTIME_NAME,
            "runtime_root_absent": absent,
            "parent_device": parent_info.st_dev,
            "parent_inode": parent_info.st_ino,
            "parent_opened_no_follow": True,
        }
    finally:
        os.close(parent_descriptor)


def source_execution_audit(transport_raw: bytes) -> dict[str, Any]:
    source = transport_raw.decode("utf-8")
    tree = ast.parse(source, filename=TRANSPORT)
    call_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.append(node.func.attr)
    launcher_hash_then_path_spawn = (
        "launcher_hash_provider(Path(LAUNCHER))" in source
        and "spawn_process(INTERPRETER, EXACT_ARGV, EXACT_ENVIRONMENT)" in source
    )
    retained_descriptor_exec = any(name in call_names for name in ("fexecve", "execveat"))
    return {
        "launcher_hash_then_path_spawn": launcher_hash_then_path_spawn,
        "retained_descriptor_exec_present": retained_descriptor_exec,
        "interpreter_pathname_reopen_gap": launcher_hash_then_path_spawn and not retained_descriptor_exec,
        "evidence": {
            "hash_expression": "launcher_hash_provider(Path(LAUNCHER))",
            "spawn_expression": "spawn_process(INTERPRETER, EXACT_ARGV, EXACT_ENVIRONMENT)",
        },
    }


def durable_create(path: str, raw: bytes) -> None:
    pure = PurePosixPath(path)
    parent_descriptor = open_dir_no_follow(str(pure.parent))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(pure.name, flags, 0o400, dir_fd=parent_descriptor)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short admission record write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = os.path.abspath(args.output)

    package_identity, package_raw = identity(PACKAGE, include_raw=True)
    acceptance_identity, acceptance_raw = identity(ACCEPTANCE, include_raw=True)
    transport_identity, transport_raw = identity(TRANSPORT, include_raw=True)
    launcher_identity, _ = identity(LAUNCHER)
    interpreter_identity, _ = identity(INTERPRETER)
    pipeline_identity, pipeline_raw = identity(PIPELINE_STATE, include_raw=True)
    assert package_raw is not None and acceptance_raw is not None
    assert transport_raw is not None and pipeline_raw is not None

    require(package_identity["sha256"] == EXPECTED_PACKAGE_SHA256, "accepted package drift")
    require(acceptance_identity["sha256"] == EXPECTED_ACCEPTANCE_SHA256, "acceptance drift")
    require(interpreter_identity["sha256"] == EXPECTED_INTERPRETER_SHA256, "interpreter drift")
    package = json.loads(package_raw)
    acceptance = json.loads(acceptance_raw)
    pipeline = json.loads(pipeline_raw)
    verify_self_hash(package, "package_content_sha256")
    verify_self_hash(acceptance, "acceptance_sha256")
    require(package["action_identity"]["action_id"] == ACTION_ID, "package action binding")
    require(acceptance["action_id"] == ACTION_ID and acceptance["accepted"] is True, "acceptance action binding")
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "static authority boundary")
    require(package["generated_files"]["tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v21.py"]["sha256"] == transport_identity["sha256"], "transport package binding")
    require(package["generated_files"]["tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21.py"]["sha256"] == launcher_identity["sha256"], "launcher package binding")

    v20_records: list[dict[str, Any]] = []
    for path, expected in EXPECTED_V20.items():
        record, _ = identity(path)
        record["expected_sha256"] = expected
        record["unchanged"] = record["sha256"] == expected
        require(record["unchanged"], f"V20 drift: {path}")
        v20_records.append(record)

    runtime = runtime_absence()
    require(runtime["runtime_root_absent"], "V21 runtime namespace already materialized")
    require(pipeline.get("current_stage") == "rtl", "pipeline stage is not rtl")
    execution_source = source_execution_audit(transport_raw)

    blockers: list[dict[str, str]] = []
    if execution_source["interpreter_pathname_reopen_gap"]:
        blockers.append({
            "code": "INTERPRETER_PATHNAME_REOPEN_GAP",
            "failure_taxonomy": "PRECONSUMPTION_SECURE_ADMISSION_IDENTITY_BINDING_GAP",
            "root_cause_hypothesis": (
                "The accepted transport validates launcher bytes through a no-follow descriptor, closes it, "
                "and then spawns the interpreter and launcher by pathname, so replacement after validation "
                "can change the executed identity."
            ),
            "regression": (
                "A future accepted execution transport must execute the retained verified interpreter and "
                "wrapper descriptors, or run in an independently sealed immutable namespace that preserves "
                "the exact accepted argv/cwd/environment contract."
            ),
        })

    record: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "ace2_v21_manager_execution_admission",
        "mission_id": MISSION_ID,
        "action_id": ACTION_ID,
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "BLOCKED_PRECONSUMPTION" if blockers else "ADMITTED_UNCONSUMED",
        "authority_state": {
            "live_operator_directive_observed": True,
            "authority_consumed": False,
            "v21_invocation_count": 0,
            "reason": "Admission completes before any one-shot authority consumption.",
        },
        "stage_reconciliation": {
            "pipeline_state": pipeline_identity,
            "current_stage": pipeline.get("current_stage"),
            "disposition": "RECONCILED_RTL_STAGE_NO_MANAGER_STATE_EDIT_REQUIRED",
        },
        "accepted_identities": {
            "package": package_identity,
            "acceptance": acceptance_identity,
            "transport": transport_identity,
            "launcher": launcher_identity,
            "interpreter": interpreter_identity,
        },
        "runtime_admission": runtime,
        "execution_source_audit": execution_source,
        "v20_immutability": {
            "all_critical_records_unchanged": all(item["unchanged"] for item in v20_records),
            "records": v20_records,
        },
        "blockers": blockers,
        "claim_boundary": (
            "Non-consuming Manager admission only. No V21 transport, launcher, evaluator, official payload, "
            "authority, credential, owner claim, ledger, result, or terminal was invoked or created; no V20 "
            "artifact or pipeline state was modified."
        ),
    }
    candidate = dict(record)
    record["record_sha256"] = sha256_bytes(compact_bytes(candidate))
    durable_create(output, compact_bytes(record))
    print(json.dumps({"output": output, "record_sha256": record["record_sha256"], "status": record["status"]}, sort_keys=True))
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
