#!/usr/bin/env python3
"""Future V29 one-shot authority wrapper. Never run during static acceptance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jsonschema


ACTION_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load_module(ACTION_ROOT / "tools/v29_contract.py", "v29_future_wrapper_contract")


def _atomic_write(path: Path, raw: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _failure_record(code: str, detail: str, phase: str) -> dict[str, Any]:
    value = {
        "artifact_kind": "qk_gbfp8_head64_b0_v29_first_terminal",
        "claim_boundary": C.CLAIM_BOUNDARY,
        "failure_code": code,
        "failure_detail_sha256": C.sha256_bytes(detail.encode("utf-8", "replace")),
        "phase": phase,
        "retry_replay_resume_repair_permitted": False,
        "schema_version": 1,
        "status": "FAILED_TERMINAL",
    }
    value["terminal_sha256"] = C.sha256_bytes(C.compact_bytes(value))
    return value


def _seal_preclaim_failure(code: str, detail: str) -> None:
    os.mkdir(C.FAILURE_SEAL_NAMESPACE, 0o700)
    _atomic_write(C.FAILURE_SEAL_NAMESPACE / "first-terminal.json", C.compact_bytes(_failure_record(code, detail, "PRECLAIM")))
    os.chmod(C.FAILURE_SEAL_NAMESPACE, 0o500)


def _seal_claimed_failure(runtime: Path, code: str, detail: str, phase: str) -> None:
    terminal_dir = runtime / "terminal"
    terminal_dir.mkdir(mode=0o700, exist_ok=False)
    _atomic_write(terminal_dir / "first-terminal.json", C.compact_bytes(_failure_record(code, detail, phase)))
    os.chmod(terminal_dir, 0o500)
    os.chmod(runtime, 0o500)


def _claim_absent_runtime(runtime: Path) -> None:
    C.require(runtime == C.RUNTIME_NAMESPACE, "RUNTIME_BINDING", str(runtime))
    C.require(not os.path.lexists(C.FAILURE_SEAL_NAMESPACE), "PRIOR_FAILURE_SEAL", str(C.FAILURE_SEAL_NAMESPACE))
    os.mkdir(runtime, 0o700)
    (runtime / "authority").mkdir(mode=0o700)
    (runtime / "authority/consumed").mkdir(mode=0o700)
    (runtime / "authority/internal").mkdir(mode=0o700)


def _read_stable_capsule(path: Path) -> tuple[dict[str, Any], bytes, os.stat_result]:
    path_info = os.lstat(path)
    C.require(stat.S_ISREG(path_info.st_mode) and not stat.S_ISLNK(path_info.st_mode), "CAPSULE_TYPE", str(path))
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    C.require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "CAPSULE_CONCURRENT_SUBSTITUTION",
        str(path),
    )
    raw = b"".join(chunks)
    value = json.loads(raw.decode("ascii", "strict"))
    C.require(type(value) is dict and C.compact_bytes(value) == raw, "CAPSULE_NONCANONICAL", str(path))
    return value, raw, after


def _consume_capsule(path: Path, destination: Path, role: str, package: dict[str, Any], now_ts: float) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    value, raw, before = _read_stable_capsule(path)
    schema_name = "manager_capsule_schema_path" if role == "MANAGER" else "operator_capsule_schema_path"
    schema, _ = C.load_canonical(ACTION_ROOT / package["authority_contract"][schema_name])
    jsonschema.Draft202012Validator(schema).validate(value)
    proof = C.validate_authority_capsule(value, role, package, now_ts)
    os.rename(path, destination)
    moved = os.lstat(destination)
    C.require((moved.st_dev, moved.st_ino) == (before.st_dev, before.st_ino), "CAPSULE_RENAME_SUBSTITUTION", role)
    C.require(not os.path.lexists(path), "CAPSULE_NOT_CONSUMED", role)
    return value, raw, proof


def _write_frozen_envelope(runtime: Path, package: dict[str, Any], acceptance: dict[str, Any], manager: tuple[dict[str, Any], bytes, dict[str, Any]], operator: tuple[dict[str, Any], bytes, dict[str, Any]]) -> Path:
    manager_value, manager_raw, manager_proof = manager
    operator_value, operator_raw, operator_proof = operator
    C.require(manager_value["claim"]["nonce"] == operator_value["claim"]["nonce"], "AUTHORITY_NONCE_DISAGREEMENT", "manager/operator")
    value = {
        "acceptance_file_sha256": C.sha256_file(C.ACCEPTANCE_PATH),
        "acceptance_self_sha256": acceptance["acceptance_sha256"],
        "artifact_kind": "qk_gbfp8_head64_b0_v29_frozen_internal_authority_envelope",
        "execution_binding_sha256": package["future_execution_binding"]["execution_binding_sha256"],
        "manager_capsule_file_sha256": C.sha256_bytes(manager_raw),
        "manager_event_proof": manager_proof,
        "nonce": manager_value["claim"]["nonce"],
        "operator_capsule_file_sha256": C.sha256_bytes(operator_raw),
        "operator_event_proof": operator_proof,
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": C.sha256_file(C.PACKAGE_PATH),
        "protected_access_permitted_only_after_this_file_is_frozen": True,
        "retry_replay_resume_repair_permitted": False,
        "schema_version": 1,
    }
    value["envelope_sha256"] = C.sha256_bytes(C.compact_bytes(value))
    destination = runtime / "authority/internal/frozen-envelope.json"
    _atomic_write(destination, C.compact_bytes(value))
    os.chmod(destination.parent, 0o500)
    return destination


def _protected_boundary(runtime: Path, envelope: Path, package: dict[str, Any]) -> None:
    C.require(envelope.is_file() and stat.S_IMODE(envelope.stat().st_mode) == 0o444, "FROZEN_ENVELOPE_ABSENT", str(envelope))
    target = ACTION_ROOT / package["future_execution_target"]["entrypoint"]
    C.require(target.is_file(), "PROTECTED_TARGET_ABSENT", str(target))
    command = [str(C.INTERPRETER), "-B", str(target), "--runtime", str(runtime), "--envelope", str(envelope)]
    completed = subprocess.run(command, cwd=str(ACTION_ROOT), env=C.EXACT_ENVIRONMENT, shell=False, check=False)
    C.require(completed.returncode == 0, "PROTECTED_TARGET_RETURN_CODE", str(completed.returncode))


def _validate_invocation(args: argparse.Namespace) -> None:
    binding = C.execution_binding()
    expected_tail = binding["argv"][2:]
    C.require(sys.argv == expected_tail, "ARGV_BINDING", repr(sys.argv))
    C.require(Path.cwd() == ACTION_ROOT, "CWD_BINDING", str(Path.cwd()))
    C.require(dict(os.environ) == C.EXACT_ENVIRONMENT, "ENVIRONMENT_BINDING", repr(sorted(os.environ)))
    C.require(Path(sys.executable).resolve() == C.INTERPRETER.resolve(), "INTERPRETER_PATH", sys.executable)
    C.require(C.sha256_file(C.INTERPRETER) == C.INTERPRETER_SHA256, "INTERPRETER_HASH", str(C.INTERPRETER))
    C.require(args.mode == "production", "MODE_BINDING", args.mode)
    C.require(Path(args.package) == C.PACKAGE_PATH and Path(args.acceptance) == C.ACCEPTANCE_PATH, "PACKAGE_ACCEPTANCE_BINDING", "argv")
    C.require(Path(args.manager_capsule) == C.MANAGER_CAPSULE and Path(args.operator_capsule) == C.OPERATOR_CAPSULE, "CAPSULE_PATH_BINDING", "argv")
    C.require(Path(args.runtime) == C.RUNTIME_NAMESPACE, "RUNTIME_ARG_BINDING", args.runtime)
    C.require(args.irreversible_action_id == C.EXECUTION_ACTION_ID, "ACTION_ARG_BINDING", args.irreversible_action_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--manager-capsule", required=True)
    parser.add_argument("--operator-capsule", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--irreversible-action-id", required=True)
    args = parser.parse_args()
    runtime = Path(args.runtime)
    claimed = False
    phase = "INVOCATION"
    try:
        _validate_invocation(args)
        phase = "RUNTIME_CLAIM"
        _claim_absent_runtime(runtime)
        claimed = True
        phase = "STATIC_PACKAGE_AND_ACCEPTANCE"
        package, package_raw = C.load_canonical(C.PACKAGE_PATH)
        C.verify_package(package, package_raw)
        candidate = load_module(ACTION_ROOT / "tools/verify_candidate_v29.py", "v29_future_candidate")
        report = candidate.verify_candidate(True)
        acceptance, acceptance_raw = C.load_canonical(C.ACCEPTANCE_PATH)
        schema, _ = C.load_canonical(ACTION_ROOT / package["fresh_l2_review"]["acceptance_schema_path"])
        jsonschema.Draft202012Validator(schema).validate(acceptance)
        C.verify_acceptance(package, acceptance, acceptance_raw, report)
        phase = "EXTERNAL_CAPSULE_CONSUMPTION"
        now_ts = time.time()
        manager = _consume_capsule(C.MANAGER_CAPSULE, runtime / "authority/consumed/manager-admission.json", "MANAGER", package, now_ts)
        operator = _consume_capsule(C.OPERATOR_CAPSULE, runtime / "authority/consumed/operator-authority.json", "OPERATOR", package, now_ts)
        phase = "FROZEN_INTERNAL_ENVELOPE"
        envelope = _write_frozen_envelope(runtime, package, acceptance, manager, operator)
        phase = "PROTECTED_BOUNDARY"
        _protected_boundary(runtime, envelope, package)
        return 0
    except Exception as error:
        code = error.code if isinstance(error, C.ContractError) else type(error).__name__
        detail = str(error)
        try:
            if claimed:
                _seal_claimed_failure(runtime, code, detail, phase)
            else:
                _seal_preclaim_failure(code, detail)
        except Exception:
            pass
        sys.stderr.write(f"{code}:{C.sha256_bytes(detail.encode('utf-8', 'replace'))}\n")
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
