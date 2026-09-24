#!/usr/bin/env python3
"""Exactly-once detached coordinator for the reviewed ACE2 case-two recovery."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_prompt_suite_rtl_oracle_regression as suite
from tools import ace2_v73_stage1_chat_attempt as stage1


CONTRACT_SCHEMA = "ace2-prompt-recovery-coordinator-v1"
AUTHORITY_SCHEMA = "ace2-prompt-recovery-single-use-authority-v1"
REVIEW_SCHEMA = "ace2-prompt-recovery-independent-review-v1"
INTENT_SCHEMA = "ace2-prompt-recovery-launch-intent-v1"
PROCESS_SCHEMA = "ace2-prompt-recovery-owner-process-v1"
CHILD_PROCESS_SCHEMA = "ace2-prompt-recovery-child-popen-boundary-v1"
CONSUMPTION_SCHEMA = "ace2-prompt-recovery-authority-consumption-v1"
TERMINAL_SCHEMA = "ace2-prompt-recovery-terminal-v1"


class CoordinationError(RuntimeError):
    pass


class DuplicateLaunch(CoordinationError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoordinationError(message)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        payload = canonical_bytes(value)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def write_atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    write_exclusive_json(temporary, value)
    try:
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        payload = canonical_bytes(value)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def acquire_owner_lock(state_dir: Path) -> int:
    descriptor = os.open(
        state_dir / "owner.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise CoordinationError(
            "another coordinator owner holds the lock"
        ) from error
    return descriptor


def project_path(value: str) -> Path:
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe project path: {value}")
    return ROOT / path


def parse_process_stat(raw: str, pid: int) -> dict[str, Any]:
    closing = raw.rfind(")")
    require(closing > 0, f"cannot parse process identity for pid {pid}")
    fields = raw[closing + 2 :].split()
    require(len(fields) > 19, f"process identity is incomplete for pid {pid}")
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "start_ticks": int(fields[19]),
    }


def process_identity(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    identity = parse_process_stat((proc / "stat").read_text(encoding="ascii"), pid)
    identity.update(
        {
            "argv": [
                item.decode("utf-8")
                for item in (proc / "cmdline").read_bytes().split(b"\0")
                if item
            ],
            "cwd": str((proc / "cwd").resolve()),
            "session_id": os.getsid(pid),
        }
    )
    return identity


def identity_live(identity: dict[str, Any]) -> bool:
    try:
        observed = process_identity(int(identity["pid"]))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return (
        observed["start_ticks"] == identity["start_ticks"]
        and observed["argv"] == identity["argv"]
        and observed["cwd"] == identity["cwd"]
        and observed["state"] != "Z"
    )


def local_python_dependency_closure(seeds: list[Path]) -> set[Path]:
    pending = [path.resolve() for path in seeds]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        observed.add(path)
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module.split(".")[-1])
                names.update(alias.name for alias in node.names)
        for name in names:
            for directory in (ROOT / "tools", ROOT / "scripts"):
                candidate = directory / f"{name}.py"
                if candidate.is_file() and candidate.resolve() not in observed:
                    pending.append(candidate.resolve())
    return observed


def implementation_binding() -> dict[str, Any]:
    python_seeds = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_prompt_suite_rtl_oracle_regression.py",
        ROOT / "tools/ace2_v73_stage1_chat_attempt.py",
        ROOT / "scripts/verify_full_chain_independent_oracle.py",
        ROOT / "scripts/verify_persistent_kv_multitoken_rtl_chat.py",
        ROOT / "tests/test_prompt_recovery_coordinator.py",
    ]
    paths = sorted(local_python_dependency_closure(python_seeds))
    paths.append(ROOT / "Makefile")
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
    }


def execution_source_binding() -> dict[str, Any]:
    base = stage1.source_binding()
    records = {record["path"]: record for record in base["files"]}
    seeds = [
        ROOT / "tools/ace2_stage1_chat_product.py",
        ROOT / "tools/run_rtl_arbitrary_text_generation.py",
        ROOT / "tools/rtl_arbitrary_text_generation_backend.py",
    ]
    for path in local_python_dependency_closure(seeds):
        relative = path.relative_to(ROOT).as_posix()
        records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    for path in sorted((ROOT / "verification/tb").rglob("*.sv")):
        relative = path.relative_to(ROOT).as_posix()
        records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    ordered = [records[path] for path in sorted(records)]
    return {
        "algorithm": "sha256(path-nul-file-sha256-newline)-v1",
        "files": ordered,
        "sha256": sha256_bytes(
            b"".join(
                record["path"].encode("utf-8")
                + b"\0"
                + record["sha256"].encode("ascii")
                + b"\n"
                for record in ordered
            )
        ),
    }


@contextmanager
def absolute_deadline_guard(deadline: float, label: str) -> Any:
    remaining = deadline - time.monotonic()
    require(remaining > 0, f"overall deadline expired before {label}")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum: int, _frame: object) -> None:
        raise CoordinationError(f"overall deadline expired during {label}")

    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def reviewed_bindings(contract_path: Path) -> dict[str, str]:
    contract = load_json(contract_path)
    paths = {
        **{
            str(path): digest
            for path, digest in implementation_binding().items()
        },
        contract["suite_config"]: sha256_file(project_path(contract["suite_config"])),
        contract["preserved_oracle_result"]: sha256_file(
            project_path(contract["preserved_oracle_result"])
        ),
        contract["preserved_source_registry"]: sha256_file(
            project_path(contract["preserved_source_registry"])
        ),
    }
    preserved_oracle = project_path(contract["preserved_oracle_result"]).parent
    preserved_attempt = preserved_oracle.parent / "rtl-attempt"
    for path in (
        preserved_oracle / "SHA256SUMS",
        preserved_attempt / "attempt-manifest.json",
        preserved_attempt / "SHA256SUMS",
        preserved_attempt / "TREE_ROOT.sha256",
    ):
        paths[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for record in execution_source_binding()["files"]:
        paths[record["path"]] = record["sha256"]
    return dict(sorted(paths.items()))


def validate_contract(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_relative_to(ROOT / "tests"), "contract must be under tests")
    contract = load_json(path)
    require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema differs")
    config = suite.load_json(project_path(contract["suite_config"]))
    suite.validate_config(config)
    cases = {case["id"]: case for case in config["cases"]}
    require(contract["preserved_case_id"] == config["cases"][0]["id"], "preserved case differs")
    require(contract["case2_id"] == config["cases"][1]["id"], "case two differs")
    require(contract["case3_id"] == config["cases"][2]["id"], "case three differs")
    require(set(cases) == {contract["preserved_case_id"], contract["case2_id"], contract["case3_id"]}, "case set differs")
    require(contract["case_timeout_seconds"] >= 90_000, "case timeout is below measured runway")
    require(
        10_800 <= contract["oracle_timeout_seconds"] <= 28_800,
        "oracle timeout is outside the reviewed bound",
    )
    require(
        contract["overall_timeout_seconds"]
        >= 2 * (contract["case_timeout_seconds"] + contract["oracle_timeout_seconds"]),
        "overall timeout cannot cover case two, case three, and oracle overhead",
    )
    require(60 <= contract["health_cadence_seconds"] <= 600, "health cadence is out of bounds")
    require(contract["minimum_launch_free_bytes"] >= 8 * 1024**3, "launch capacity floor is too low")
    require(contract["minimum_next_case_free_bytes"] >= 6 * 1024**3, "next-case capacity floor is too low")
    output = project_path(contract["output"]).resolve()
    require(output.is_relative_to(ROOT / "reports/verification"), "output is outside verification reports")
    preserved = load_json(project_path(contract["preserved_oracle_result"]))
    require(preserved.get("status") == "PASS", "preserved first-case oracle is not PASS")
    require(preserved.get("numerical_status") == "PASS", "preserved first-case numerical status is not PASS")
    require(
        preserved["reused_inputs"]["prompt"]["sha256"]
        == sha256_bytes(cases[contract["preserved_case_id"]]["prompt"].encode("utf-8")),
        "preserved first-case prompt differs",
    )
    registry = load_json(project_path(contract["preserved_source_registry"]))
    require(registry.get("state") == "timeout", "0005 registry is not the historical timeout")
    require(registry.get("timeout_seconds") == 72_000, "0005 historical timeout differs")
    require(
        preserved["source_binding"]["sha256"] == stage1.source_binding()["sha256"],
        "current numerical source differs from preserved first case",
    )
    verify_attempt_seal(project_path(contract["preserved_oracle_result"]).parent.parent / "rtl-attempt")
    return contract


def validate_review(review_path: Path, contract_path: Path) -> dict[str, Any]:
    review = load_json(review_path)
    require(review.get("schema") == REVIEW_SCHEMA, "review schema differs")
    require(review.get("status") == "APPROVED", "independent review is not approved")
    require(review.get("independent") is True, "review is not marked independent")
    require(review.get("contract_sha256") == sha256_file(contract_path), "review contract binding differs")
    require(review.get("reviewed_bindings") == reviewed_bindings(contract_path), "review byte bindings differ")
    require(review.get("focused_tests_status") == "PASS", "review did not bind passing focused tests")
    require(
        review.get("focused_test_command") == "make prompt-recovery-coordinator-test",
        "review focused test command differs",
    )
    return review


def issue_authority(
    contract_path: Path,
    review_path: Path,
    authority_path: Path,
    state_dir: Path,
) -> dict[str, Any]:
    contract = validate_contract(contract_path)
    review = validate_review(review_path, contract_path)
    require(not authority_path.exists(), "single-use authority already exists")
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "status": "GRANTED_AFTER_INDEPENDENT_REVIEW",
        "authority_id": contract["authority_id"],
        "scope_authorization_id": contract["scope_authorization_id"],
        "authority_cardinality": 1,
        "case2_execution_limit": 1,
        "retry_replay_relaunch": "FORBIDDEN_AFTER_CONSUMPTION",
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
        },
        "review": {
            "path": str(review_path.resolve()),
            "sha256": sha256_file(review_path),
        },
        "reviewed_bindings": reviewed_bindings(contract_path),
        "state_dir": str(state_dir.resolve()),
        "issued_at_utc": utc_now(),
    }
    write_atomic_create_json(authority_path, authority)
    return authority


def validate_authority(
    contract_path: Path,
    authority_path: Path,
    state_dir: Path,
) -> dict[str, Any]:
    contract = validate_contract(contract_path)
    authority = load_json(authority_path)
    require(authority.get("schema") == AUTHORITY_SCHEMA, "authority schema differs")
    require(authority.get("status") == "GRANTED_AFTER_INDEPENDENT_REVIEW", "authority status differs")
    require(authority.get("authority_id") == contract["authority_id"], "authority id differs")
    require(authority.get("scope_authorization_id") == contract["scope_authorization_id"], "user scope differs")
    require(authority.get("authority_cardinality") == 1, "authority cardinality differs")
    require(authority.get("case2_execution_limit") == 1, "case-two limit differs")
    require(authority.get("retry_replay_relaunch") == "FORBIDDEN_AFTER_CONSUMPTION", "retry policy differs")
    require(authority["contract"]["sha256"] == sha256_file(contract_path), "authority contract binding differs")
    review_path = Path(authority["review"]["path"])
    require(authority["review"]["sha256"] == sha256_file(review_path), "authority review binding differs")
    validate_review(review_path, contract_path)
    require(authority.get("reviewed_bindings") == reviewed_bindings(contract_path), "authority byte bindings differ")
    require(authority.get("state_dir") == str(state_dir.resolve()), "authority state namespace differs")
    return authority


def validate_execution_gate_bindings(
    *,
    contract_path: Path,
    approved_bindings: dict[str, str],
    control_bindings: dict[str, str],
    manifest_source: dict[str, Any],
    manifest_dependencies: dict[str, Any],
) -> None:
    require(sha256_file(contract_path) == control_bindings["contract_sha256"], "contract changed before exec gate")
    require(
        sha256_file(Path(control_bindings["review_path"])) == control_bindings["review_sha256"],
        "review receipt changed before exec gate",
    )
    require(
        sha256_file(Path(control_bindings["authority_path"])) == control_bindings["authority_sha256"],
        "authority changed before exec gate",
    )
    require(
        reviewed_bindings(contract_path) == approved_bindings,
        "reviewed bytes changed before scientific exec gate",
    )
    contract = load_json(contract_path)
    preserved = load_json(project_path(contract["preserved_oracle_result"]))
    current_source = stage1.source_binding()
    require(
        manifest_source["sha256"]
        == current_source["sha256"]
        == preserved["source_binding"]["sha256"],
        "scientific source binding differs at exec gate",
    )
    require(
        manifest_dependencies == execution_source_binding(),
        "execution dependency binding differs at exec gate",
    )


def create_case2_launch_ticket(
    state_dir: Path,
    authority_path: Path,
    contract_path: Path,
    prompt: Path,
    attempt: Path,
    worker_command: list[str],
) -> dict[str, Any]:
    ticket = {
        "schema": "ace2-prompt-recovery-case2-launch-ticket-v1",
        "status": "READY_FOR_EXACTLY_ONE_INNER_POPEN",
        "created_at_utc": utc_now(),
        "case_id": "verification-question",
        "authority_sha256": sha256_file(authority_path),
        "contract_sha256": sha256_file(contract_path),
        "output": str(attempt.resolve()),
        "prompt": str(prompt.resolve()),
        "prompt_sha256": sha256_file(prompt),
        "worker_argv": worker_command,
        "worker_argv_sha256": sha256_bytes(canonical_bytes(worker_command)),
        "source_and_rtl_sha256": execution_source_binding()["sha256"],
        "retry_replay_relaunch": "PERMANENTLY_FORBIDDEN",
    }
    try:
        write_atomic_create_json(state_dir / "case2-launch-ticket.json", ticket)
    except FileExistsError as error:
        raise DuplicateLaunch("case-two launch ticket already exists") from error
    return ticket


def gate_launch_command(
    *,
    gate_path: Path,
    boundary_record_path: Path,
    boundary_role: str,
    prompt: Path,
    output: Path,
    worker_command: list[str],
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        str(Path(__file__).resolve()),
        "_gate_exec",
        "--gate",
        str(gate_path.resolve()),
        "--boundary-record",
        str(boundary_record_path.resolve()),
        "--role",
        boundary_role,
        "--prompt",
        str(prompt.resolve()),
        "--output",
        str(output.resolve()),
        "--",
        *worker_command,
    ]


def publish_child_boundary_identity(
    *,
    boundary_record_path: Path,
    boundary_role: str,
    prompt: Path,
    output: Path,
    worker_command: list[str],
    before_boundary_write: Callable[[], None] | None = None,
) -> dict[str, Any]:
    identity = process_identity(os.getpid())
    launcher_command = identity["argv"]
    require(identity["cwd"] == str(ROOT), "gated child cwd differs")
    require(identity["session_id"] == identity["pid"], "gated child is not a detached session leader")
    require(len(launcher_command) > 2, "gated child command source is absent")
    require(len(worker_command) > 2, "worker command source is absent")
    launcher_source = Path(launcher_command[2]).resolve()
    worker_source = Path(worker_command[2]).resolve()
    require(launcher_source.is_file(), "gated child command source is not a file")
    require(worker_source.is_file(), "worker command source is not a file")
    record = {
        "schema": CHILD_PROCESS_SCHEMA,
        "status": "DETACHED_CHILD_PUBLISHED_IDENTITY_BEFORE_GATE_RELEASE",
        "role": boundary_role,
        "recorded_at_utc": utc_now(),
        "command": {
            "launcher_argv": launcher_command,
            "launcher_argv_sha256": sha256_bytes(canonical_bytes(launcher_command)),
            "worker_argv": worker_command,
            "worker_argv_sha256": sha256_bytes(canonical_bytes(worker_command)),
            "cwd": str(ROOT),
        },
        "input": {
            "prompt": str(prompt.resolve()),
            "prompt_sha256": sha256_file(prompt),
        },
        "output": str(output.resolve()),
        "source": {
            "launcher": {
                "path": str(launcher_source),
                "sha256": sha256_file(launcher_source),
            },
            "worker": {
                "path": str(worker_source),
                "sha256": sha256_file(worker_source),
            },
        },
        **identity,
    }
    if before_boundary_write is not None:
        before_boundary_write()
    write_atomic_create_json(boundary_record_path, record)
    return record


def verify_child_boundary_identity(
    *,
    boundary_record_path: Path,
    process: subprocess.Popen[Any],
    boundary_role: str,
    prompt: Path,
    output: Path,
    worker_command: list[str],
    spawn_command: list[str],
) -> dict[str, Any]:
    record = load_json(boundary_record_path)
    require(record.get("schema") == CHILD_PROCESS_SCHEMA, "child boundary schema differs")
    require(
        record.get("status") == "DETACHED_CHILD_PUBLISHED_IDENTITY_BEFORE_GATE_RELEASE",
        "child boundary status differs",
    )
    require(record.get("role") == boundary_role, "child boundary role differs")
    require(record.get("pid") == process.pid, "child boundary PID differs")
    observed = process_identity(process.pid)
    for field in ("pid", "start_ticks", "argv", "cwd", "session_id"):
        require(record.get(field) == observed[field], f"child boundary {field} differs")
    require(observed["argv"] == spawn_command, "child boundary launcher argv differs")
    require(observed["cwd"] == str(ROOT), "child boundary cwd differs")
    require(observed["session_id"] == observed["pid"], "child boundary is not a detached session leader")
    require(
        record.get("command")
        == {
            "launcher_argv": spawn_command,
            "launcher_argv_sha256": sha256_bytes(canonical_bytes(spawn_command)),
            "worker_argv": worker_command,
            "worker_argv_sha256": sha256_bytes(canonical_bytes(worker_command)),
            "cwd": str(ROOT),
        },
        "child boundary command binding differs",
    )
    require(
        record.get("input")
        == {
            "prompt": str(prompt.resolve()),
            "prompt_sha256": sha256_file(prompt),
        },
        "child boundary input binding differs",
    )
    require(record.get("output") == str(output.resolve()), "child boundary output differs")
    require(
        record.get("source")
        == {
            "launcher": {
                "path": str(Path(spawn_command[2]).resolve()),
                "sha256": sha256_file(Path(spawn_command[2]).resolve()),
            },
            "worker": {
                "path": str(Path(worker_command[2]).resolve()),
                "sha256": sha256_file(Path(worker_command[2]).resolve()),
            },
        },
        "child boundary source binding differs",
    )
    return record


def launch_at_scientific_boundary(
    *,
    state_dir: Path,
    authority_paths: tuple[Path, Path] | None,
    prompt: Path,
    attempt: Path,
    worker_command: list[str],
    spawn_command: list[str],
    boundary_record_path: Path,
    boundary_role: str,
    spawn: Callable[[], subprocess.Popen[Any]],
) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    if authority_paths is not None:
        authority_path, contract_path = authority_paths
        ticket = create_case2_launch_ticket(
            state_dir,
            authority_path,
            contract_path,
            prompt,
            attempt,
            worker_command,
        )
        write_atomic_create_json(
            state_dir / "authority-consumed.json",
            {
                "schema": CONSUMPTION_SCHEMA,
                "status": "CONSUMED_IMMEDIATELY_BEFORE_INNER_RTL_POPEN",
                "consumed_at_utc": utc_now(),
                "authority_sha256": ticket["authority_sha256"],
                "contract_sha256": ticket["contract_sha256"],
                "launch_ticket_sha256": sha256_file(state_dir / "case2-launch-ticket.json"),
                "case2_launch_attempt_count": 1,
                "case2_process_start_count": 0,
                "retry_replay_relaunch": "PERMANENTLY_FORBIDDEN",
            },
        )
    timer = signal.setitimer(signal.ITIMER_REAL, 0)
    timer_paused_at = time.monotonic()
    process: subprocess.Popen[Any] | None = None
    try:
        process = spawn()
        publication_deadline = time.monotonic() + 30
        while (
            not boundary_record_path.is_file()
            and process.poll() is None
            and time.monotonic() < publication_deadline
        ):
            time.sleep(0.05)
        require(
            boundary_record_path.is_file(),
            "gated child did not publish identity before the boundary deadline",
        )
        base_identity = verify_child_boundary_identity(
            boundary_record_path=boundary_record_path,
            process=process,
            boundary_role=boundary_role,
            prompt=prompt,
            output=attempt,
            worker_command=worker_command,
            spawn_command=spawn_command,
        )
        if timer[0] > 0:
            remaining = timer[0] - (time.monotonic() - timer_paused_at)
            if remaining <= 0:
                terminate_and_reap(process, base_identity, [spawn_command, worker_command])
                raise CoordinationError("absolute deadline expired during raw Popen")
            signal.setitimer(signal.ITIMER_REAL, remaining, timer[1])
        timer = (0.0, 0.0)
        if authority_paths is not None:
            write_atomic_create_json(
                state_dir / "case2-popen-succeeded.json",
                {
                    "schema": "ace2-prompt-recovery-scientific-popen-v1",
                    "status": "GATED_RTL_POPEN_SUCCEEDED_BEFORE_SCIENTIFIC_EXEC",
                    "recorded_at_utc": utc_now(),
                    "process": base_identity,
                    "case2_popen_success_count": 1,
                },
            )
    except BaseException:
        if process is not None:
            try:
                identity = process_identity(process.pid)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                terminate_unrecorded_child(process)
            else:
                terminate_and_reap(process, identity, [spawn_command, worker_command])
        if timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, timer[0], timer[1])
        raise
    return process, base_identity


def gate_exec(
    gate_path: Path,
    boundary_record_path: Path,
    boundary_role: str,
    prompt: Path,
    output: Path,
    worker_command: list[str],
    *,
    before_boundary_write: Callable[[], None] | None = None,
) -> int:
    identity = publish_child_boundary_identity(
        boundary_record_path=boundary_record_path,
        boundary_role=boundary_role,
        prompt=prompt,
        output=output,
        worker_command=worker_command,
        before_boundary_write=before_boundary_write,
    )
    deadline = time.monotonic() + 60
    while not gate_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    require(gate_path.is_file(), "scientific execution gate was not released")
    gate = load_json(gate_path)
    require(gate.get("schema") == "ace2-prompt-recovery-science-gate-v1", "science gate schema differs")
    require(gate.get("status") == "RELEASED_AFTER_DURABLE_PROCESS_IDENTITY", "science gate status differs")
    require(gate.get("pid") == identity["pid"], "science gate PID differs")
    require(gate.get("start_ticks") == identity["start_ticks"], "science gate start ticks differ")
    require(
        gate.get("boundary_record_sha256") == sha256_file(boundary_record_path),
        "science gate child boundary hash differs",
    )
    require(
        gate.get("worker_argv_sha256") == sha256_bytes(canonical_bytes(worker_command)),
        "science gate worker argv differs",
    )
    os.execvpe(worker_command[0], worker_command, dict(os.environ))
    raise AssertionError("exec returned unexpectedly")


def record_process(
    path: Path,
    process: subprocess.Popen[Any],
    command: list[str],
    role: str,
    inputs: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    identity = process_identity(process.pid)
    require(identity["argv"] == command, f"{role} argv differs after Popen")
    require(identity["cwd"] == str(ROOT), f"{role} cwd differs after Popen")
    record = {
        "schema": "ace2-prompt-recovery-stage-process-v1",
        "role": role,
        "recorded_at_utc": utc_now(),
        "command": {
            "argv": command,
            "argv_sha256": sha256_bytes(canonical_bytes(command)),
            "cwd": str(ROOT),
        },
        "inputs": inputs,
        "output": str(output.resolve()),
        "source": {
            "path": command[2],
            "sha256": sha256_file(Path(command[2])),
        },
        **identity,
    }
    write_atomic_create_json(path, record)
    return record


def find_inner_worker(worker_result: Path) -> dict[str, Any] | None:
    expected = str(worker_result.resolve())
    matches: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            identity = process_identity(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
        argv = identity["argv"]
        if "--worker-result" not in argv:
            continue
        index = argv.index("--worker-result")
        if index + 1 < len(argv) and str(Path(argv[index + 1]).resolve()) == expected:
            matches.append(identity)
    require(len(matches) <= 1, "multiple inner workers match the exact case output")
    return matches[0] if matches else None


def terminate_identity(identity: dict[str, Any]) -> None:
    require(identity_live(identity), f"process identity changed before termination: {identity['pid']}")
    require(identity["session_id"] == identity["pid"], "refusing to terminate a non-session-leader")
    os.killpg(int(identity["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + 10
    while identity_live(identity) and time.monotonic() < deadline:
        time.sleep(0.2)
    if identity_live(identity):
        os.killpg(int(identity["pid"]), signal.SIGKILL)


def terminate_unrecorded_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def terminate_and_reap(
    process: subprocess.Popen[Any],
    base_identity: dict[str, Any],
    allowed_argv: list[list[str]],
) -> None:
    def matching_identity() -> dict[str, Any] | None:
        try:
            observed = process_identity(int(base_identity["pid"]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return None
        if (
            observed["start_ticks"] != base_identity["start_ticks"]
            or observed["cwd"] != base_identity["cwd"]
            or observed["session_id"] != observed["pid"]
            or observed["argv"] not in allowed_argv
            or observed["state"] == "Z"
        ):
            return None
        return observed

    observed = matching_identity()
    if observed is not None:
        os.killpg(observed["pid"], signal.SIGTERM)
    try:
        process.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        observed = matching_identity()
        if observed is None:
            raise CoordinationError("child did not reap and exact process identity changed")
        os.killpg(observed["pid"], signal.SIGKILL)
        process.wait(timeout=15)


def signal_recorder(
    path: Path,
    cancellation: dict[str, str],
) -> Callable[[int, object], None]:
    def record(signum: int, _frame: object) -> None:
        append_jsonl(
            path,
            {
                "schema": "ace2-prompt-recovery-owner-signal-v1",
                "observed_at_utc": utc_now(),
                "signal": signal.Signals(signum).name,
                "action": "RECORDED_WITHOUT_ABANDONING_OR_KILLING_STAGE",
            },
        )

    return record


def cancellation_request(state_dir: Path) -> str | None:
    path = state_dir / "cancel-request.json"
    if not path.is_file():
        return None
    request = load_json(path)
    require(request.get("schema") == "ace2-prompt-recovery-cancel-request-v1", "cancel schema differs")
    require(request.get("status") == "REQUESTED", "cancel status differs")
    return str(request["reason"])


def request_cancel(state_dir: Path, reason: str) -> dict[str, Any]:
    require(reason.strip(), "cancel reason is empty")
    descriptor = os.open(state_dir / "launch-commit.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        owner = load_json(state_dir / "owner-process.json")
        require(identity_live(owner), "coordinator owner is not live")
        require(not (state_dir / "terminal.json").exists(), "coordinator is already terminal")
        authority_consumed = (state_dir / "authority-consumed.json").exists()
        request = {
            "schema": "ace2-prompt-recovery-cancel-request-v1",
            "status": "REQUESTED",
            "requested_at_utc": utc_now(),
            "reason": reason,
            "owner_pid": owner["pid"],
            "owner_start_ticks": owner["start_ticks"],
            "authority_consumed_at_request": authority_consumed,
        }
        write_atomic_create_json(state_dir / "cancel-request.json", request)
        return request
    finally:
        os.close(descriptor)


def publish_failure_terminal(state_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    terminal = state_dir / "terminal.json"
    descriptor = os.open(state_dir / "launch-commit.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if terminal.is_file():
            return load_json(terminal)
        cancellation = state_dir / "cancel-request.json"
        record = dict(record)
        record["cancellation_requested_before_terminal"] = cancellation.is_file()
        if cancellation.is_file():
            record["cancellation_request"] = load_json(cancellation)
        write_atomic_create_json(terminal, record)
        return record
    finally:
        os.close(descriptor)


def stage_heartbeat(
    state_dir: Path,
    output: Path,
    role: str,
    driver: dict[str, Any] | None,
    worker: dict[str, Any] | None,
    minimum_free_bytes: int,
) -> None:
    disk_free_bytes = shutil.disk_usage(ROOT).free
    output_bytes = 0
    if output.exists():
        for path in output.rglob("*"):
            try:
                if path.is_file():
                    output_bytes += path.stat().st_size
            except FileNotFoundError:
                continue
    append_jsonl(
        state_dir / "heartbeats.jsonl",
        {
            "schema": "ace2-prompt-recovery-heartbeat-v1",
            "observed_at_utc": utc_now(),
            "role": role,
            "owner": process_identity(os.getpid()),
            "driver": {
                "identity": driver,
                "live": bool(driver and identity_live(driver)),
            },
            "inner_worker": {
                "identity": worker,
                "live": bool(worker and identity_live(worker)),
            },
            "capacity": {
                "disk_free_bytes": disk_free_bytes,
                "minimum_free_bytes": minimum_free_bytes,
                "ready": disk_free_bytes >= minimum_free_bytes,
            },
            "output_bytes": output_bytes,
        },
    )


def finalize_orphan_attempt(attempt: Path, elapsed_seconds: float) -> None:
    require(not (attempt / "attempt-result.json").exists(), "attempt was already finalized")
    product_result = load_json(attempt / "worker-result.json")
    require(
        product_result.get("readability", {}).get("accepted") is True,
        "orphaned worker continuation is not accepted",
    )
    runtime_output = attempt / "runtime-output"
    require((runtime_output / "run_summary.json").is_file(), "orphaned worker run summary is absent")
    comparison = stage1.independent_comparison(runtime_output)
    require(comparison.get("status") == "PASS", "orphaned worker comparison failed")
    stage1.write_exclusive(attempt / "independent-reference-comparison.json", comparison)
    result = {
        "schema": "ace2-v73-stage1-chat-attempt-result-v1",
        "status": "PASS",
        "exit_code": 0,
        "timed_out": False,
        "elapsed_wall_seconds": elapsed_seconds,
        "generated_token_ids": product_result["generated_token_ids"],
        "decoded_text": product_result["decoded_text"],
        "independent_comparison_status": "PASS",
        "next_action": "independent Reviewer acceptance",
        "coordinator_orphan_finalization": True,
    }
    stage1.write_exclusive(attempt / "attempt-result.json", result)
    stage1.write_exclusive(
        attempt / "timing.json",
        {
            "supervisor_total_wall_seconds": elapsed_seconds,
            "backend_latency": product_result.get("latency"),
            "coordinator_orphan_finalization": True,
        },
    )
    stage1.seal_attempt(attempt)


def verify_attempt_seal(attempt: Path) -> None:
    sums_path = attempt / "SHA256SUMS"
    root_path = attempt / "TREE_ROOT.sha256"
    require(sums_path.is_file() and root_path.is_file(), "attempt seal is incomplete")
    sums = sums_path.read_bytes()
    require(
        root_path.read_text(encoding="ascii")
        == f"{sha256_bytes(sums)}  SHA256SUMS\n",
        "attempt tree root differs",
    )
    entries = 0
    for line in sums.decode("utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        member = Path(relative)
        require(
            separator == "  "
            and not member.is_absolute()
            and ".." not in member.parts,
            "attempt checksum member is unsafe",
        )
        path = attempt / member
        require(path.is_file() and sha256_file(path) == digest, f"attempt checksum differs: {relative}")
        entries += 1
    require(entries > 0, "attempt checksum manifest is empty")


def monitor_rtl_worker(
    *,
    case_id: str,
    worker: dict[str, Any],
    deadline: float,
    cadence_seconds: int,
    heartbeat: Callable[[], None],
    cancel_requested: Callable[[], str | None],
) -> dict[str, Any] | None:
    next_heartbeat = 0.0
    while identity_live(worker):
        now = time.monotonic()
        if now >= next_heartbeat:
            try:
                heartbeat()
            except BaseException as error:
                terminate_identity(worker)
                return {
                    "case_id": case_id,
                    "stage": "rtl_heartbeat",
                    "status": "HEARTBEAT_FAILED_EXACT_RTL_WORKER_TERMINATED",
                    "error": str(error),
                }
            next_heartbeat = now + cadence_seconds
        cancellation_signal = cancel_requested()
        if cancellation_signal is not None:
            terminate_identity(worker)
            return {
                "case_id": case_id,
                "stage": "owner_cancellation",
                "status": "CANCELLED_NO_RETRY_EXACT_RTL_WORKER_TERMINATED",
                "signal": cancellation_signal,
            }
        if now >= deadline:
            terminate_identity(worker)
            return {
                "case_id": case_id,
                "stage": "rtl_attempt_timeout",
                "returncode": 124,
                "status": "BOUNDED_TIMEOUT_NO_RETRY_EXACT_RTL_WORKER_TERMINATED",
            }
        time.sleep(min(5, cadence_seconds))
    return None


def run_case(
    *,
    case: dict[str, Any],
    output: Path,
    state_dir: Path,
    timeout_seconds: int,
    oracle_timeout_seconds: int,
    cadence_seconds: int,
    overall_deadline: float,
    minimum_free_bytes: int,
    cancel_requested: Callable[[], str | None],
    authority_paths: tuple[Path, Path] | None,
    contract_path: Path,
    approved_bindings: dict[str, str],
    control_bindings: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    disk_free_bytes = shutil.disk_usage(ROOT).free
    if disk_free_bytes < minimum_free_bytes:
        return None, {
            "case_id": case["id"],
            "stage": "capacity_preflight",
            "status": "CAPACITY_FLOOR_NOT_MET_NO_LAUNCH",
            "disk_free_bytes": disk_free_bytes,
            "minimum_free_bytes": minimum_free_bytes,
        }
    observed = time.monotonic()
    if observed >= overall_deadline:
        return None, {
            "case_id": case["id"],
            "stage": "scientific_popen_guard",
            "status": (
                "OVERALL_DEADLINE_EXPIRED_AT_SCIENTIFIC_POPEN_GUARD_"
                "NO_LAUNCH_AUTHORITY_UNCONSUMED"
            ),
            "observed_monotonic": observed,
            "overall_deadline": overall_deadline,
            "authority_consumed": False,
        }
    with absolute_deadline_guard(overall_deadline, f"{case['id']} preparation"):
        case_dir = output / "cases" / case["id"]
        case_dir.mkdir(parents=True)
        attempt = case_dir / "rtl-attempt"
        attempt.mkdir()
        prompt = attempt / "prompt.utf8"
        stage1.write_exclusive(prompt, case["prompt"].encode("utf-8"))
        stage1.generation.validate_max_new_tokens(4)
        position01 = stage1.product.verify_accepted_position01()
        model_snapshot = stage1.generation.resolve_snapshot()
        tokenization, _tokenizer = stage1.generation.tokenizer_record(
            case["prompt"],
            stage1.backend.MAX_CONTEXT_TOKENS - 4,
            4,
            model_snapshot,
        )
    runtime_output = attempt / "runtime-output"
    worker_result = attempt / "worker-result.json"
    stdout = attempt / "stdout.raw.log"
    stderr = attempt / "stderr.raw.log"
    worker_command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(ROOT / "tools/ace2_v73_stage1_chat_attempt.py"),
        "--worker",
        "--worker-prompt-file",
        str(prompt.resolve()),
        "--worker-runtime-output",
        str(runtime_output.resolve()),
        "--worker-result",
        str(worker_result.resolve()),
        "--max-new-tokens",
        "4",
    ]
    with absolute_deadline_guard(overall_deadline, f"{case['id']} manifest binding"):
        source_and_rtl = stage1.source_binding()
        dependency_binding = execution_source_binding()
        tools = {
            "python": stage1.tool_record([str(Path(sys.executable).resolve()), "--version"]),
            "iverilog": stage1.tool_record([str(Path("/usr/bin/iverilog")), "-V"]),
            "vvp": stage1.tool_record([str(Path("/usr/bin/vvp")), "-V"]),
        }
    manifest = {
        "schema": "ace2-v73-stage1-chat-attempt-manifest-v1",
        "manager": "reviewed-prompt-recovery-coordinator-v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "created_at_utc": utc_now(),
        "timeout_seconds": timeout_seconds,
        "pass_policy": (
            "PASS only after the requested RTL-selected tokens, generated-token feedback "
            "across every decode transition, 24-layer RTL integer agreement at every "
            "executed position, persistent RTL-published K/V, full-head agreement, and "
            "readable official tokenizer detokenization; otherwise FAIL or TIMEOUT"
        ),
        "accepted_position01": position01,
        "prompt": stage1.file_record(prompt),
        "tokenization": tokenization,
        "model": stage1.file_record(model_snapshot / "model.safetensors"),
        "model_config": stage1.file_record(model_snapshot / "config.json"),
        "adapter": stage1.file_record(stage1.backend.canonical.ADAPTER.resolve()),
        "source_and_rtl": source_and_rtl,
        "execution_dependency_binding": dependency_binding,
        "command": {
            "argv": worker_command,
            "argv_sha256": sha256_bytes(canonical_bytes(worker_command)),
            "cwd": str(ROOT),
        },
        "tools": tools,
        "environment": {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    manifest["environment_sha256"] = sha256_bytes(canonical_bytes(manifest["environment"]))
    with absolute_deadline_guard(overall_deadline, f"{case['id']} manifest publication"):
        stage1.write_exclusive(attempt / "attempt-manifest.json", manifest)
        stage1.write_exclusive(attempt / "command.json", manifest["command"])
    environment = dict(os.environ)
    environment.update(manifest["environment"])
    gate_path = state_dir / f"{case['id']}-science-gate.json"
    boundary_record_path = state_dir / f"{case['id']}-rtl-popen-boundary.json"
    boundary_role = f"{case['id']}-gated-launcher"
    launch_command = gate_launch_command(
        gate_path=gate_path,
        boundary_record_path=boundary_record_path,
        boundary_role=boundary_role,
        prompt=prompt,
        output=attempt,
        worker_command=worker_command,
    )
    with stdout.open("xb") as stdout_stream, stderr.open("xb") as stderr_stream:
        cancellation_signal = cancel_requested()
        if cancellation_signal is not None:
            return None, {
                "case_id": case["id"],
                "stage": "scientific_popen_guard",
                "status": (
                    "CANCELLED_AT_SCIENTIFIC_POPEN_GUARD_NO_LAUNCH_"
                    "AUTHORITY_UNCONSUMED"
                ),
                "signal": cancellation_signal,
                "authority_consumed": False,
            }
        now = time.monotonic()
        if now >= overall_deadline:
            return None, {
                "case_id": case["id"],
                "stage": "scientific_popen_guard",
                "status": (
                    "OVERALL_DEADLINE_EXPIRED_AT_SCIENTIFIC_POPEN_GUARD_"
                    "NO_LAUNCH_AUTHORITY_UNCONSUMED"
                ),
                "observed_monotonic": now,
                "overall_deadline": overall_deadline,
                "authority_consumed": False,
            }
        disk_free_bytes = shutil.disk_usage(ROOT).free
        if disk_free_bytes < minimum_free_bytes:
            return None, {
                "case_id": case["id"],
                "stage": "scientific_popen_guard",
                "status": (
                    "CAPACITY_FLOOR_NOT_MET_AT_SCIENTIFIC_POPEN_GUARD_"
                    "NO_LAUNCH_AUTHORITY_UNCONSUMED"
                ),
                "disk_free_bytes": disk_free_bytes,
                "minimum_free_bytes": minimum_free_bytes,
                "authority_consumed": False,
            }
        commit_descriptor = os.open(
            state_dir / "launch-commit.lock",
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        process: subprocess.Popen[Any] | None = None
        base_identity: dict[str, Any] | None = None
        try:
            try:
                fcntl.flock(commit_descriptor, fcntl.LOCK_EX)
                require(cancel_requested() is None, "cancellation won the serialized launch race")
                with absolute_deadline_guard(overall_deadline, f"{case['id']} pre-consumption binding validation"):
                    validate_execution_gate_bindings(
                        contract_path=contract_path,
                        approved_bindings=approved_bindings,
                        control_bindings=control_bindings,
                        manifest_source=source_and_rtl,
                        manifest_dependencies=dependency_binding,
                    )
                    started = time.monotonic()
                    process, base_identity = launch_at_scientific_boundary(
                        state_dir=state_dir,
                        authority_paths=authority_paths,
                        prompt=prompt,
                        attempt=attempt,
                        worker_command=worker_command,
                        spawn_command=launch_command,
                        boundary_record_path=boundary_record_path,
                        boundary_role=boundary_role,
                        spawn=lambda: subprocess.Popen(
                            launch_command,
                            cwd=ROOT,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout_stream,
                            stderr=stderr_stream,
                            start_new_session=True,
                            close_fds=True,
                        ),
                    )
            finally:
                os.close(commit_descriptor)
        except BaseException:
            if process is not None and base_identity is not None:
                terminate_and_reap(process, base_identity, [launch_command, worker_command])
            raise
        require(process is not None and base_identity is not None, "launch boundary returned incomplete state")
        try:
            gate_process = record_process(
                state_dir / f"{case['id']}-gate-process.json",
                process,
                launch_command,
                f"{case['id']}-gated-launcher",
                {
                    "case_id": case["id"],
                    "prompt": str(prompt.resolve()),
                    "prompt_sha256": sha256_file(prompt),
                },
                attempt,
            )
            with absolute_deadline_guard(overall_deadline, f"{case['id']} gate validation and release"):
                validate_execution_gate_bindings(
                    contract_path=contract_path,
                    approved_bindings=approved_bindings,
                    control_bindings=control_bindings,
                    manifest_source=source_and_rtl,
                    manifest_dependencies=dependency_binding,
                )
                write_atomic_create_json(
                    gate_path,
                    {
                        "schema": "ace2-prompt-recovery-science-gate-v1",
                        "status": "RELEASED_AFTER_DURABLE_PROCESS_IDENTITY",
                        "released_at_utc": utc_now(),
                        "pid": gate_process["pid"],
                        "start_ticks": gate_process["start_ticks"],
                        "boundary_record_sha256": sha256_file(boundary_record_path),
                        "worker_argv_sha256": sha256_bytes(canonical_bytes(worker_command)),
                    },
                )
                exec_deadline = min(time.monotonic() + 30, overall_deadline)
                worker: dict[str, Any] | None = None
                while time.monotonic() < exec_deadline:
                    try:
                        observed = process_identity(process.pid)
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    if (
                        observed["start_ticks"] == gate_process["start_ticks"]
                        and observed["argv"] == worker_command
                        and observed["cwd"] == str(ROOT)
                    ):
                        worker = record_process(
                            state_dir / f"{case['id']}-inner-worker-process.json",
                            process,
                            worker_command,
                            f"{case['id']}-inner-rtl-worker",
                            {
                                "case_id": case["id"],
                                "prompt": str(prompt.resolve()),
                                "prompt_sha256": sha256_file(prompt),
                            },
                            attempt,
                        )
                        break
                    time.sleep(0.05)
                require(worker is not None, "RTL exec was not observed before the absolute deadline")
            if authority_paths is not None:
                write_atomic_create_json(
                    state_dir / "case2-process-started.json",
                    {
                        "worker": worker,
                        "schema": "ace2-prompt-recovery-scientific-process-started-v1",
                        "status": "INNER_RTL_PROCESS_STARTED",
                        "started_at_utc": utc_now(),
                        "case2_process_start_count": 1,
                    },
                )
        except BaseException as error:
            terminate_and_reap(process, base_identity, [launch_command, worker_command])
            return None, {
                "case_id": case["id"],
                "stage": "scientific_exec_gate",
                "status": "EXEC_GATE_FAILED_EXACT_PROCESS_REAPED_NO_RETRY",
                "error": str(error),
            }
        require(worker is not None, "internal exec gate state is incomplete")
        deadline = min(started + timeout_seconds, overall_deadline)
        try:
            monitor_failure = monitor_rtl_worker(
                case_id=case["id"],
                worker=worker,
                deadline=deadline,
                cadence_seconds=cadence_seconds,
                heartbeat=lambda: stage_heartbeat(
                        state_dir,
                        output,
                        case["id"],
                        worker,
                        None,
                        minimum_free_bytes,
                ),
                cancel_requested=cancel_requested,
            )
            if monitor_failure is not None:
                terminate_and_reap(process, worker, [worker_command])
                return None, monitor_failure
            returncode = process.wait()
        except BaseException:
            terminate_and_reap(process, worker, [worker_command])
            raise
    if returncode != 0 or not worker_result.is_file():
        return None, {
            "case_id": case["id"],
            "stage": "rtl_worker",
            "returncode": returncode,
            "status": "RTL_WORKER_FAILED_NO_RETRY",
        }
    try:
        with absolute_deadline_guard(overall_deadline, f"{case['id']} attempt finalization"):
            finalize_orphan_attempt(attempt, time.monotonic() - started)
            verify_attempt_seal(attempt)
    except CoordinationError as error:
        return None, {
            "case_id": case["id"],
            "stage": "rtl_attempt_finalization",
            "returncode": returncode,
            "status": "RTL_FINALIZATION_FAILED_NO_RETRY",
            "error": str(error),
        }

    oracle_output = case_dir / "full-chain-oracle"
    oracle_stdout = case_dir / "full-chain-oracle.stdout.log"
    oracle_stderr = case_dir / "full-chain-oracle.stderr.log"
    oracle_command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(ROOT / "scripts/verify_full_chain_independent_oracle.py"),
        "--attempt",
        str(attempt.resolve()),
        "--output",
        str(oracle_output.resolve()),
    ]
    oracle_gate_path = state_dir / f"{case['id']}-oracle-science-gate.json"
    oracle_boundary_record_path = (
        state_dir / f"{case['id']}-oracle-popen-boundary.json"
    )
    oracle_boundary_role = f"{case['id']}-oracle-gated-launcher"
    oracle_launch_command = gate_launch_command(
        gate_path=oracle_gate_path,
        boundary_record_path=oracle_boundary_record_path,
        boundary_role=oracle_boundary_role,
        prompt=prompt,
        output=oracle_output,
        worker_command=oracle_command,
    )
    cancellation_signal = cancel_requested()
    if cancellation_signal is not None:
        return None, {
            "case_id": case["id"],
            "stage": "oracle_popen_guard",
            "status": "CANCELLED_BEFORE_ORACLE_LAUNCH",
            "signal": cancellation_signal,
        }
    if time.monotonic() >= overall_deadline:
        return None, {
            "case_id": case["id"],
            "stage": "oracle_popen_guard",
            "status": "OVERALL_DEADLINE_EXPIRED_BEFORE_ORACLE_LAUNCH",
        }
    with oracle_stdout.open("xb") as stdout_stream, oracle_stderr.open("xb") as stderr_stream:
        commit_descriptor = os.open(
            state_dir / "launch-commit.lock",
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        oracle: subprocess.Popen[Any] | None = None
        oracle_base_identity: dict[str, Any] | None = None
        try:
            try:
                fcntl.flock(commit_descriptor, fcntl.LOCK_EX)
                require(cancel_requested() is None, "cancellation won the serialized oracle launch race")
                with absolute_deadline_guard(overall_deadline, f"{case['id']} oracle prelaunch validation"):
                    validate_execution_gate_bindings(
                        contract_path=contract_path,
                        approved_bindings=approved_bindings,
                        control_bindings=control_bindings,
                        manifest_source=source_and_rtl,
                        manifest_dependencies=dependency_binding,
                    )
                    oracle, oracle_base_identity = launch_at_scientific_boundary(
                        state_dir=state_dir,
                        authority_paths=None,
                        prompt=prompt,
                        attempt=oracle_output,
                        worker_command=oracle_command,
                        spawn_command=oracle_launch_command,
                        boundary_record_path=oracle_boundary_record_path,
                        boundary_role=oracle_boundary_role,
                        spawn=lambda: subprocess.Popen(
                            oracle_launch_command,
                            cwd=ROOT,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout_stream,
                            stderr=stderr_stream,
                            start_new_session=True,
                            close_fds=True,
                        ),
                    )
            finally:
                os.close(commit_descriptor)
        except BaseException:
            if oracle is not None and oracle_base_identity is not None:
                terminate_and_reap(
                    oracle,
                    oracle_base_identity,
                    [oracle_launch_command, oracle_command],
                )
            raise
        require(oracle is not None and oracle_base_identity is not None, "oracle launch returned incomplete state")
        try:
            with absolute_deadline_guard(overall_deadline, f"{case['id']} oracle gate validation and release"):
                oracle_gate_identity = record_process(
                    state_dir / f"{case['id']}-oracle-gate-process.json",
                    oracle,
                    oracle_launch_command,
                    f"{case['id']}-oracle-gated-launcher",
                    {"case_id": case["id"]},
                    oracle_output,
                )
                validate_execution_gate_bindings(
                    contract_path=contract_path,
                    approved_bindings=approved_bindings,
                    control_bindings=control_bindings,
                    manifest_source=source_and_rtl,
                    manifest_dependencies=dependency_binding,
                )
                write_atomic_create_json(
                    oracle_gate_path,
                    {
                        "schema": "ace2-prompt-recovery-science-gate-v1",
                        "status": "RELEASED_AFTER_DURABLE_PROCESS_IDENTITY",
                        "released_at_utc": utc_now(),
                        "pid": oracle_gate_identity["pid"],
                        "start_ticks": oracle_gate_identity["start_ticks"],
                        "boundary_record_sha256": sha256_file(
                            oracle_boundary_record_path
                        ),
                        "worker_argv_sha256": sha256_bytes(canonical_bytes(oracle_command)),
                    },
                )
                exec_deadline = min(time.monotonic() + 30, overall_deadline)
                oracle_identity: dict[str, Any] | None = None
                while time.monotonic() < exec_deadline:
                    try:
                        observed = process_identity(oracle.pid)
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    if (
                        observed["start_ticks"] == oracle_gate_identity["start_ticks"]
                        and observed["argv"] == oracle_command
                        and observed["cwd"] == str(ROOT)
                    ):
                        oracle_identity = record_process(
                            state_dir / f"{case['id']}-oracle-process.json",
                            oracle,
                            oracle_command,
                            f"{case['id']}-oracle",
                            {
                                "case_id": case["id"],
                                "attempt_tree_root_sha256": sha256_file(attempt / "TREE_ROOT.sha256"),
                            },
                            oracle_output,
                        )
                        break
                    time.sleep(0.05)
                require(oracle_identity is not None, "oracle exec was not observed before the absolute deadline")
        except BaseException:
            terminate_and_reap(
                oracle,
                oracle_base_identity,
                [oracle_launch_command, oracle_command],
            )
            raise
        try:
            require(oracle_identity is not None, "internal oracle gate state is incomplete")
            oracle_deadline = min(time.monotonic() + oracle_timeout_seconds, overall_deadline)
            next_heartbeat = 0.0
            while identity_live(oracle_identity):
                now = time.monotonic()
                if now >= oracle_deadline:
                    terminate_identity(oracle_identity)
                    terminate_and_reap(oracle, oracle_identity, [oracle_command])
                    return None, {
                        "case_id": case["id"],
                        "stage": "full_chain_independent_oracle_timeout",
                        "returncode": 124,
                        "status": "BOUNDED_ORACLE_TIMEOUT_NO_RTL_RETRY",
                    }
                cancellation_signal = cancel_requested()
                if cancellation_signal is not None:
                    terminate_identity(oracle_identity)
                    terminate_and_reap(oracle, oracle_identity, [oracle_command])
                    return None, {
                        "case_id": case["id"],
                        "stage": "oracle_owner_cancellation",
                        "status": "CANCELLED_NO_RETRY_EXACT_ORACLE_TERMINATED",
                        "signal": cancellation_signal,
                    }
                if now >= next_heartbeat:
                    stage_heartbeat(
                        state_dir,
                        output,
                        f"{case['id']}-oracle",
                        oracle_identity,
                        None,
                        minimum_free_bytes,
                    )
                    next_heartbeat = now + cadence_seconds
                time.sleep(min(cadence_seconds, 5))
            returncode = oracle.wait()
        except BaseException:
            terminate_and_reap(
                oracle,
                oracle_base_identity,
                [oracle_launch_command, oracle_command],
            )
            raise
    if returncode != 0:
        return None, {
            "case_id": case["id"],
            "stage": "full_chain_independent_oracle",
            "returncode": returncode,
            "status": "ORACLE_FAILED_NO_RTL_RETRY",
        }
    with absolute_deadline_guard(overall_deadline, f"{case['id']} oracle validation"):
        oracle_result = load_json(oracle_output / "result.json")
        for record in oracle_result["oracle_implementation"]["files"]:
            require(
                approved_bindings.get(record["path"]) == record["sha256"],
                f"oracle implementation was not authority-approved: {record['path']}",
            )
        validated_oracle = suite.validate_oracle_result(
            case,
            oracle_output,
            oracle_result,
        )
    return validated_oracle, None


def validate_combined_coverage(result: dict[str, Any]) -> None:
    coverage = result["coverage"]
    require(coverage["configured_prompts"] == 3, "combined prompt count differs")
    require(coverage["completed_prompts"] == 3, "combined result is incomplete")
    require(coverage["fresh_rtl_prompts"] == 2, "combined result did not preserve exactly one case")
    require(coverage["generated_tokens"] == 12, "combined result does not contain 12 tokens")
    require(coverage["continuous_24_layer_kv_growth_prompts"] == 3, "K/V coverage differs")
    require(coverage["decode_transitions_checked"] == 9, "decode transition coverage differs")
    require(coverage["decode_layer_growth_checks"] == 216, "decode layer coverage differs")
    require(coverage["integer_byte_mismatches"] == 0, "combined result has integer mismatches")
    require(coverage["selected_token_mismatches"] == 0, "combined result has token mismatches")
    require(coverage["software_transformer_or_logits_fallback"] is False, "combined result used fallback")
    for field in (
        "kv_append_bytes_compared",
        "full_cache_bytes_compared",
        "quantized_layer_output_bytes_compared",
        "full_vocabulary_logit_bytes_compared",
        "full_vocabulary_head_steps_checked",
    ):
        require(coverage[field] > 0, f"combined {field} is zero")


def run_owner(contract_path: Path, authority_path: Path, state_dir: Path) -> int:
    owner_started = time.monotonic()
    raw_contract = load_json(contract_path)
    overall_deadline = owner_started + int(raw_contract["overall_timeout_seconds"])
    with absolute_deadline_guard(overall_deadline, "owner contract and authority validation"):
        contract = validate_contract(contract_path)
        authority = validate_authority(contract_path, authority_path, state_dir)
    owner_record_path = state_dir / "owner-process.json"
    deadline = min(time.monotonic() + 30, overall_deadline)
    while not owner_record_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    require(owner_record_path.is_file(), "owner process record was not published")
    process_record = load_json(owner_record_path)
    require(process_record.get("schema") == PROCESS_SCHEMA, "owner process record schema differs")
    observed_owner = process_identity(os.getpid())
    for field in ("pid", "start_ticks", "argv", "cwd", "session_id"):
        require(process_record[field] == observed_owner[field], f"owner {field} differs")
    require(observed_owner["session_id"] == observed_owner["pid"], "owner is not a detached session leader")
    lock_descriptor = acquire_owner_lock(state_dir)
    try:
        signal_log = state_dir / "signals.jsonl"
        cancellation: dict[str, str] = {}
        handler = signal_recorder(signal_log, cancellation)
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(signum, handler)
        write_atomic_create_json(
            state_dir / "owner-ready.json",
            {
                "schema": "ace2-prompt-recovery-owner-ready-v1",
                "status": "OWNER_LOCKED_SIGNAL_SAFE_AND_READY",
                "ready_at_utc": utc_now(),
                "pid": observed_owner["pid"],
                "start_ticks": observed_owner["start_ticks"],
            },
        )
        with absolute_deadline_guard(overall_deadline, "owner output preparation"):
            output = project_path(contract["output"])
            require(not output.exists(), "recovery output already exists")
            require(shutil.disk_usage(ROOT).free >= contract["minimum_launch_free_bytes"], "launch capacity floor is not met")
            output.mkdir(parents=True)
            config_path = project_path(contract["suite_config"])
            config = suite.load_json(config_path)
            (output / "suite-config.json").write_bytes(suite.canonical_bytes(config))
            cases = {case["id"]: case for case in config["cases"]}
        control_bindings = {
            "contract_sha256": authority["contract"]["sha256"],
            "review_path": authority["review"]["path"],
            "review_sha256": authority["review"]["sha256"],
            "authority_path": str(authority_path.resolve()),
            "authority_sha256": sha256_file(authority_path),
        }

        case2, failure = run_case(
            case=cases[contract["case2_id"]],
            output=output,
            state_dir=state_dir,
            timeout_seconds=contract["case_timeout_seconds"],
            oracle_timeout_seconds=contract["oracle_timeout_seconds"],
            cadence_seconds=contract["health_cadence_seconds"],
            overall_deadline=overall_deadline,
            minimum_free_bytes=contract["minimum_launch_free_bytes"],
            cancel_requested=lambda: cancellation_request(state_dir),
            authority_paths=(authority_path, contract_path),
            contract_path=contract_path,
            approved_bindings=authority["reviewed_bindings"],
            control_bindings=control_bindings,
        )
        if failure is not None:
            raise CoordinationError(json.dumps(failure, sort_keys=True))
        require(case2 is not None, "case two did not pass")
        require((state_dir / "authority-consumed.json").is_file(), "case-two authority was not consumed")
        require((state_dir / "case2-process-started.json").is_file(), "case-two RTL worker was not started")
        require(time.monotonic() < overall_deadline, "overall deadline expired after case two")
        require(
            shutil.disk_usage(ROOT).free >= contract["minimum_next_case_free_bytes"],
            "capacity floor blocks first execution of case three",
        )
        case3, failure = run_case(
            case=cases[contract["case3_id"]],
            output=output,
            state_dir=state_dir,
            timeout_seconds=contract["case_timeout_seconds"],
            oracle_timeout_seconds=contract["oracle_timeout_seconds"],
            cadence_seconds=contract["health_cadence_seconds"],
            overall_deadline=overall_deadline,
            minimum_free_bytes=contract["minimum_next_case_free_bytes"],
            cancel_requested=lambda: cancellation_request(state_dir),
            authority_paths=None,
            contract_path=contract_path,
            approved_bindings=authority["reviewed_bindings"],
            control_bindings=control_bindings,
        )
        if failure is not None:
            raise CoordinationError(json.dumps(failure, sort_keys=True))
        require(case3 is not None, "case three produced no result")
        first_case = cases[contract["preserved_case_id"]]
        first_oracle = project_path(contract["preserved_oracle_result"]).parent
        with absolute_deadline_guard(overall_deadline, "preserved case seal validation"):
            verify_attempt_seal(first_oracle.parent / "rtl-attempt")
            preserved = suite.validate_oracle_result(
                first_case,
                first_oracle,
                load_json(first_oracle / "result.json"),
            )
        preserved["origin"] = "preserved_0005_rtl_and_oracle"
        with absolute_deadline_guard(overall_deadline, "combined result construction"):
            result = suite.build_result(
                config_path,
                output,
                [preserved, case2, case3],
                "PENDING_INDEPENDENT_L2",
                None,
            )
        result["scientific_checks_status"] = "PASS"
        result["final_independent_l2_status"] = "PENDING"
        result["coverage"]["preserved_rtl_prompts"] = 1
        result["reproduction_command"] = (
            "python3 scripts/run_prompt_recovery_coordinator.py status "
            f"--state-dir {state_dir}"
        )
        result["recovery_provenance"] = {
            "authority_id": authority["authority_id"],
            "authority_sha256": sha256_file(authority_path),
            "contract_sha256": sha256_file(contract_path),
            "preserved_case": contract["preserved_case_id"],
            "new_cases": [contract["case2_id"], contract["case3_id"]],
            "historical_0005_timeout_preserved": True,
        }
        commit_descriptor = os.open(
            state_dir / "launch-commit.lock",
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        try:
            fcntl.flock(commit_descriptor, fcntl.LOCK_EX)
            require(cancellation_request(state_dir) is None, "cancellation won before terminal commit")
            with absolute_deadline_guard(overall_deadline, "combined result publication"):
                validate_combined_coverage(result)
                suite.finalize(output, result)
                write_atomic_create_json(
                    state_dir / "terminal.json",
                    {
                        "schema": TERMINAL_SCHEMA,
                        "status": "PASS_PENDING_FINAL_INDEPENDENT_L2",
                        "completed_at_utc": utc_now(),
                        "result_path": str((output / "result.json").resolve()),
                        "result_sha256": sha256_file(output / "result.json"),
                        "authority_consumed": True,
                        "case2_launch_attempt_count": 1,
                        "case2_popen_success_count": 1,
                        "case2_process_start_count": 1,
                        "case2_retry_count": 0,
                        "automatic_retry": False,
                    },
                )
        finally:
            os.close(commit_descriptor)
        return 0
    except BaseException as error:
        publish_failure_terminal(
            state_dir,
            {
                "schema": TERMINAL_SCHEMA,
                "status": (
                    "FAILED_NO_AUTOMATIC_RETRY"
                    if (state_dir / "authority-consumed.json").exists()
                    else "FAILED_BEFORE_CASE2_LAUNCH_AUTHORITY_UNCONSUMED"
                ),
                "completed_at_utc": utc_now(),
                "error": str(error),
                "authority_consumed": (state_dir / "authority-consumed.json").exists(),
                "case2_launch_attempt_count": int((state_dir / "authority-consumed.json").exists()),
                "case2_popen_success_count": int((state_dir / "case2-popen-succeeded.json").exists()),
                "case2_process_start_count": int((state_dir / "case2-process-started.json").exists()),
                "case2_retry_count": 0,
                "automatic_retry": False,
            },
        )
        raise
    finally:
        os.close(lock_descriptor)


def launch(contract_path: Path, authority_path: Path, state_dir: Path) -> dict[str, Any]:
    contract = validate_contract(contract_path)
    validate_authority(contract_path, authority_path, state_dir)
    require(not project_path(contract["output"]).exists(), "recovery output already exists")
    try:
        state_dir.mkdir(mode=0o700)
    except FileExistsError as error:
        raise DuplicateLaunch("coordinator state namespace already exists") from error
    intent = {
        "schema": INTENT_SCHEMA,
        "status": "AUTHORIZED_DETACHED_OWNER_INTENT",
        "created_at_utc": utc_now(),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "authority_path": str(authority_path.resolve()),
        "authority_sha256": sha256_file(authority_path),
        "implementation": implementation_binding(),
        "scientific_process_start_count": 0,
    }
    write_atomic_create_json(state_dir / "intent.json", intent)
    console_path = state_dir / "owner-console.log"
    console = console_path.open("xb", buffering=0)
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(Path(__file__).resolve()),
        "_worker",
        "--contract",
        str(contract_path.resolve()),
        "--authority",
        str(authority_path.resolve()),
        "--state-dir",
        str(state_dir.resolve()),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        try:
            identity = process_identity(process.pid)
        except BaseException:
            terminate_unrecorded_child(process)
            raise
        try:
            record = {
                "schema": PROCESS_SCHEMA,
                "status": "DETACHED_COORDINATOR_OWNER_STARTED",
                "recorded_at_utc": utc_now(),
                **identity,
            }
            write_atomic_create_json(state_dir / "owner-process.json", record)
            ready_path = state_dir / "owner-ready.json"
            deadline = time.monotonic() + 30
            while not ready_path.is_file() and identity_live(record) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not ready_path.is_file():
                if identity_live(record):
                    return {
                        **record,
                        "status": "DETACHED_COORDINATOR_OWNER_RUNNING_READINESS_PENDING",
                        "readiness": "PENDING",
                        "automatic_retry": False,
                    }
                raise CoordinationError("detached coordinator owner exited before acknowledging readiness")
            ready = load_json(ready_path)
            require(
                ready.get("pid") == record["pid"]
                and ready.get("start_ticks") == record["start_ticks"]
                and ready.get("status") == "OWNER_LOCKED_SIGNAL_SAFE_AND_READY",
                "detached owner readiness identity differs",
            )
            return record
        except BaseException:
            terminate_and_reap(process, identity, [command])
            raise
    except BaseException as error:
        publish_failure_terminal(
            state_dir,
            {
                "schema": TERMINAL_SCHEMA,
                "status": "OWNER_LAUNCH_FAILED_NO_SCIENTIFIC_START",
                "completed_at_utc": utc_now(),
                "error": str(error),
                "authority_consumed": False,
                "case2_process_start_count": 0,
                "case2_retry_count": 0,
            },
        )
        raise
    finally:
        console.close()


def live_child_identities(state_dir: Path) -> list[dict[str, Any]]:
    children: dict[tuple[int, int], dict[str, Any]] = {}
    paths = [
        *sorted(state_dir.glob("*-popen-boundary.json")),
        *sorted(state_dir.glob("*-process.json")),
    ]
    for path in paths:
        if path.name == "owner-process.json":
            continue
        identity = load_json(path)
        if identity_live(identity):
            key = (int(identity["pid"]), int(identity["start_ticks"]))
            children.setdefault(key, identity)
    return [children[key] for key in sorted(children)]


def settle_owner_loss(
    state_dir: Path,
) -> dict[str, Any]:
    terminal = state_dir / "terminal.json"
    descriptor = os.open(state_dir / "launch-commit.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if terminal.is_file():
            return {
                "state": "TERMINAL",
                "terminal": load_json(terminal),
                "live_children": live_child_identities(state_dir),
                "automatic_retry": False,
            }
        owner = load_json(state_dir / "owner-process.json")
        if identity_live(owner):
            return {"state": "RUNNING", "owner": owner}
        children = live_child_identities(state_dir)
        authority_consumed = (state_dir / "authority-consumed.json").exists()
        settlement = {
            "schema": TERMINAL_SCHEMA,
            "status": (
                "OWNER_LOST_WITH_LIVE_CHILD_TERMINAL_NO_AUTOMATIC_RETRY"
                if children
                else "OWNER_LOST_NO_LIVE_CHILD_TERMINAL_NO_AUTOMATIC_RETRY"
            ),
            "completed_at_utc": utc_now(),
            "owner": owner,
            "live_children_at_settlement": children,
            "authority_consumed": authority_consumed,
            "case2_launch_attempt_count": int(authority_consumed),
            "case2_popen_success_count": int(
                (state_dir / "case2-popen-succeeded.json").exists()
            ),
            "case2_process_start_count": int(
                (state_dir / "case2-process-started.json").exists()
            ),
            "case2_retry_count": 0,
            "automatic_retry": False,
            "retry_replay_relaunch": "PERMANENTLY_FORBIDDEN",
        }
        write_atomic_create_json(terminal, settlement)
        return {
            "state": "TERMINAL",
            "terminal": settlement,
            "live_children": children,
            "automatic_retry": False,
        }
    finally:
        os.close(descriptor)


def status(state_dir: Path) -> dict[str, Any]:
    if not state_dir.exists():
        return {"state": "NOT_LAUNCHED"}
    terminal = state_dir / "terminal.json"
    if terminal.is_file():
        return {
            "state": "TERMINAL",
            "terminal": load_json(terminal),
            "live_children": live_child_identities(state_dir),
            "automatic_retry": False,
        }
    owner_path = state_dir / "owner-process.json"
    if not owner_path.is_file():
        return {"state": "INTENT_ONLY_FAIL_CLOSED", "intent": load_json(state_dir / "intent.json")}
    owner = load_json(owner_path)
    if identity_live(owner):
        return {"state": "RUNNING", "owner": owner}
    return settle_owner_loss(state_dir)


def preflight(contract_path: Path, state_dir: Path) -> dict[str, Any]:
    contract = validate_contract(contract_path)
    return {
        "schema": "ace2-prompt-recovery-preflight-v1",
        "status": "PASS_PREPARATION_ONLY_NO_AUTHORITY_NO_EXECUTION",
        "contract_sha256": sha256_file(contract_path),
        "reviewed_bindings": reviewed_bindings(contract_path),
        "output_absent": not project_path(contract["output"]).exists(),
        "state_absent": not state_dir.exists(),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "launch_capacity_ready": shutil.disk_usage(ROOT).free >= contract["minimum_launch_free_bytes"],
        "scientific_process_start_count": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "launch"):
        child = subparsers.add_parser(name)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument("--state-dir", type=Path, required=True)
        if name == "launch":
            child.add_argument("--authority", type=Path, required=True)
    issue = subparsers.add_parser("issue-authority")
    issue.add_argument("--contract", type=Path, required=True)
    issue.add_argument("--review", type=Path, required=True)
    issue.add_argument("--authority", type=Path, required=True)
    issue.add_argument("--state-dir", type=Path, required=True)
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--contract", type=Path, required=True)
    worker.add_argument("--authority", type=Path, required=True)
    worker.add_argument("--state-dir", type=Path, required=True)
    gate = subparsers.add_parser("_gate_exec")
    gate.add_argument("--gate", type=Path, required=True)
    gate.add_argument("--boundary-record", type=Path, required=True)
    gate.add_argument("--role", required=True)
    gate.add_argument("--prompt", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("worker_argv", nargs=argparse.REMAINDER)
    probe = subparsers.add_parser("status")
    probe.add_argument("--state-dir", type=Path, required=True)
    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--state-dir", type=Path, required=True)
    cancel.add_argument("--reason", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(args.contract.resolve(), args.state_dir.resolve()), indent=2, sort_keys=True))
        return 0
    if args.command == "issue-authority":
        print(
            json.dumps(
                issue_authority(
                    args.contract.resolve(),
                    args.review.resolve(),
                    args.authority.resolve(),
                    args.state_dir.resolve(),
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "launch":
        print(
            json.dumps(
                launch(args.contract.resolve(), args.authority.resolve(), args.state_dir.resolve()),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "_worker":
        return run_owner(args.contract.resolve(), args.authority.resolve(), args.state_dir.resolve())
    if args.command == "_gate_exec":
        worker_argv = list(args.worker_argv)
        if worker_argv and worker_argv[0] == "--":
            worker_argv.pop(0)
        require(bool(worker_argv), "gated worker argv is absent")
        return gate_exec(
            args.gate.resolve(),
            args.boundary_record.resolve(),
            args.role,
            args.prompt.resolve(),
            args.output.resolve(),
            worker_argv,
        )
    if args.command == "cancel":
        print(
            json.dumps(
                request_cancel(args.state_dir.resolve(), args.reason),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(status(args.state_dir.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_PROMPT_RECOVERY_COORDINATOR_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(1)
