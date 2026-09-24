#!/usr/bin/env python3
"""Prepare deterministic Fresh-L2 review evidence for the inert V20 candidate."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v20:execute-once:a81f2916:additive-0001"
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root"
BUILD_ROOT = PROJECT_ROOT / "build/v20-bound-path-repair-attempt-0001"
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_a81f2916"
PACKAGE = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_BOUND_CORE_MANIFEST_PREFLIGHT_FIXTURE_REPORT.json"
VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_bound_path_repair_v20.py"
CANDIDATE_REPORT = BUILD_ROOT / "candidate-inert-verifier-report.json"
RERUN_REPORT = BUILD_ROOT / "independent-candidate-inert-rerun.json"
REVIEW_REQUEST = BUILD_ROOT / "fresh-l2-review-request.json"
V19_TERMINAL = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_fdfc33a1/primary/authority/base/first-terminal.json"
V19_TERMINAL_SHA256 = "a81f2916fe3e5f2aa0489caee7d7a52ada1a8760b32ea7faf3a94bad6f4cb862"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TZ": "UTC"}
AUDIT = {"official_payload_opens": 0, "official_target_starts": 0}


class HandoffError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffError(message)


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


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def write_once(path: Path, value: dict[str, Any]) -> None:
    raw = compact_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short handoff write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path == OFFICIAL_PAYLOAD.resolve():
            AUDIT["official_payload_opens"] += 1
            raise HandoffError("official payload open prohibited")
    if event == "subprocess.Popen":
        rendered = repr(args)
        if "evaluator_worker_v20.py" in rendered or "evaluator_static_v8.py" in rendered:
            AUDIT["official_target_starts"] += 1
            raise HandoffError("official evaluator start prohibited")


def action_tree_digest() -> dict[str, Any]:
    records = []
    for path in sorted(ACTION_ROOT.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"action symlink: {path}")
        relative = path.relative_to(ACTION_ROOT).as_posix()
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative, "sha256": sha256_file(path), "size": info.st_size})
        else:
            raise HandoffError(f"unsupported action entry: {path}")
    return {"entry_count": len(records), "file_count": sum(item["kind"] == "file" for item in records), "root": str(ACTION_ROOT), "tree_sha256": sha256_bytes(compact_bytes(records))}


def prepare() -> dict[str, Any]:
    sys.addaudithook(audit_hook)
    require(not os.path.lexists(RERUN_REPORT) and not os.path.lexists(REVIEW_REQUEST), "handoff evidence already exists")
    require(not os.path.lexists(RUNTIME_ROOT), "V20 runtime namespace materialized")
    require(sha256_file(V19_TERMINAL) == V19_TERMINAL_SHA256, "V19 terminal drift")
    package, package_raw = canonical(PACKAGE)
    candidate, candidate_raw = canonical(CANDIDATE_REPORT)
    fixture, fixture_raw = canonical(FIXTURE_REPORT)
    require(package["action_identity"]["action_id"] == ACTION_ID, "package action identity")
    require(candidate["status"] == "PASS_V20_BOUND_PATH_REPAIR_CANDIDATE_INERT", "candidate verifier status")
    require(candidate["manifest_file_sha256"] == sha256_bytes(package_raw), "candidate package binding")
    require(candidate["fixture"]["report_file_sha256"] == sha256_bytes(fixture_raw), "candidate fixture binding")
    command = [str(INTERPRETER), str(VERIFIER)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=ENVIRONMENT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(completed.returncode == 0 and completed.stderr == b"", f"clean verifier rerun: rc={completed.returncode} stderr={completed.stderr!r}")
    require(completed.stdout == candidate_raw, "clean verifier rerun differs from sealed candidate report")
    require(not os.path.lexists(RUNTIME_ROOT), "rerun materialized V20 runtime namespace")
    require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "handoff official effect audit")
    rerun = sealed({
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_independent_inert_rerun",
        "byte_equal_to_candidate_report": True,
        "candidate_report_file_sha256": sha256_bytes(candidate_raw),
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "cwd": str(PROJECT_ROOT),
        "environment": ENVIRONMENT,
        "exit_status": completed.returncode,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "stderr_byte_count": len(completed.stderr),
        "stdout_file_sha256": sha256_bytes(completed.stdout),
        "v20_executed": False,
        "verifier_argv": command,
    }, "report_sha256")
    write_once(RERUN_REPORT, rerun)
    tree = action_tree_digest()
    request = sealed({
        "action_id": ACTION_ID,
        "action_root": str(ACTION_ROOT),
        "action_root_tree_sha256": tree["tree_sha256"],
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_fresh_l2_review_request",
        "candidate_inert_report_file_sha256": sha256_bytes(candidate_raw),
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "core_manifest_closure_entries_sha256": package["core_manifest_closure"]["entries_sha256"],
        "core_manifest_closure_path_count": len(package["core_manifest_closure"]["entries"]),
        "decision_requested": "ACCEPT_STATIC_PACKAGE_OR_SEAL_HONEST_REJECTION",
        "fixture_case_count": fixture["case_count"],
        "fixture_report_file_sha256": sha256_bytes(fixture_raw),
        "independent_rerun_file_sha256": sha256_file(RERUN_REPORT),
        "manifest_file_sha256": sha256_bytes(package_raw),
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "required_checks": [
            "exact path/hash/mode-bound transitive CORE_MANIFEST closure",
            "constant-key production package subscript coverage",
            "exact interpreter/argv/cwd/environment bindings",
            "V18 25-record complete binding and exactly three numerical tensor selections",
            "real shared-preflight positive, absent, wrong-path, wrong-hash, wrong-mode, malformed, inconsistent, and concurrency fixtures",
            "atomic owner and loser no-publication cardinalities",
            "all predecessor roots byte-identical and retired V19 terminal unchanged",
            "V20 runtime namespace absent and zero official effects",
        ],
        "reviewer_role": "Fresh-L2",
        "runtime_namespace_file_count": 0,
        "static_acceptance_grants_execution_authority": False,
        "v20_executed": False,
        "verifier_argv": command,
        "verifier_environment": ENVIRONMENT,
    }, "request_sha256")
    write_once(REVIEW_REQUEST, request)
    return request


def main() -> int:
    try:
        request = prepare()
    except Exception as error:
        sys.stderr.write(f"V20_HANDOFF_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
