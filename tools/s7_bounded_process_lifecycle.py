#!/usr/bin/python3
"""Fail-closed subprocess lifecycle observation for read-only S7 probes."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


Evaluator = Callable[[bytes, bytes, int], dict[str, Any]]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def digest_streams(return_code: int | None, stdout: bytes, stderr: bytes) -> dict[str, Any]:
    return {
        "raw_stderr_persisted": False,
        "raw_stdout_persisted": False,
        "return_code": return_code,
        "stderr_byte_count": len(stderr),
        "stderr_sha256": sha256_bytes(stderr),
        "stdout_byte_count": len(stdout),
        "stdout_sha256": sha256_bytes(stdout),
    }


def _read_proc_stat(pid: int) -> dict[str, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    right = raw.rfind(")")
    if right < 0:
        return None
    fields = raw[right + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return {
            "pid": pid,
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except ValueError:
        return None


def _proc_snapshot() -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    try:
        entries = os.scandir("/proc")
    except OSError:
        return result
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            stat_record = _read_proc_stat(int(entry.name))
            if stat_record is not None:
                result[stat_record["pid"]] = stat_record
    return result


def _hash_proc_executable(pid: int) -> tuple[str | None, str | None]:
    proc_exe = Path(f"/proc/{pid}/exe")
    try:
        target = os.readlink(proc_exe)
        basename = Path(target.removesuffix(" (deleted)")).name
        digest = hashlib.sha256()
        with proc_exe.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return basename, digest.hexdigest()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None, None


def _hash_proc_cmdline(pid: int) -> tuple[str | None, int]:
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()[:4096]
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None, 0
    return sha256_bytes(payload), len(payload)


def _requested_runtime_identity(
    argv0: str, environment: Mapping[str, str]
) -> tuple[str | None, str | None]:
    candidate = argv0 if "/" in argv0 else shutil.which(argv0, path=environment.get("PATH"))
    if candidate is None:
        return None, None
    try:
        resolved = Path(candidate).resolve(strict=True)
        with resolved.open("rb") as handle:
            first_line = handle.readline(512)
        runtime = resolved
        if first_line.startswith(b"#!"):
            words = first_line[2:].decode("utf-8", errors="strict").strip().split()
            if words:
                interpreter = words[0]
                if Path(interpreter).name == "env" and len(words) > 1:
                    located = shutil.which(words[1], path=environment.get("PATH"))
                    if located is not None:
                        interpreter = located
                runtime = Path(interpreter).resolve(strict=True)
        return runtime.name, sha256_file(runtime)
    except (OSError, UnicodeDecodeError):
        return None, None


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(process_group_id: int, sig: int) -> bool:
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return False
    return True


def _identity_alive(pid: int, start_ticks: int) -> bool:
    current = _read_proc_stat(pid)
    return current is not None and current["start_ticks"] == start_ticks


def _signal_identity(pid: int, start_ticks: int, sig: int) -> bool:
    if not _identity_alive(pid, start_ticks):
        return False
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def _token(identity_salt: str, pid: int, start_ticks: int) -> str:
    return sha256_bytes(f"{identity_salt}:{pid}:{start_ticks}".encode("ascii"))


def _discover(
    snapshot: dict[int, dict[str, int]],
    root_pid: int,
    root_start_ticks: int | None,
    tracked: dict[tuple[int, int], dict[str, Any]],
    identity_salt: str,
    expected_executable_sha256s: set[str],
    root_fallback_identity: tuple[str | None, str | None],
    started: float,
) -> None:
    root_record = snapshot.get(root_pid)
    live_descendant_pids: set[int] = set()
    if root_record is not None and (
        root_start_ticks is None or root_record["start_ticks"] == root_start_ticks
    ):
        live_descendant_pids.add(root_pid)
    for (pid, start_ticks) in tracked:
        current = snapshot.get(pid)
        if current is not None and current["start_ticks"] == start_ticks:
            live_descendant_pids.add(pid)

    changed = True
    while changed:
        changed = False
        for record in snapshot.values():
            if record["pid"] in live_descendant_pids:
                continue
            if record["ppid"] in live_descendant_pids or record["session"] == root_pid:
                live_descendant_pids.add(record["pid"])
                changed = True

    observed_ms = round((time.monotonic() - started) * 1000, 3)
    for pid in sorted(live_descendant_pids):
        record = snapshot[pid]
        key = (pid, record["start_ticks"])
        existing = tracked.get(key)
        if existing is None:
            executable_basename, executable_sha256 = _hash_proc_executable(pid)
            if pid == root_pid and executable_sha256 is None:
                executable_basename, executable_sha256 = root_fallback_identity
            cmdline_sha256, cmdline_byte_count = _hash_proc_cmdline(pid)
            existing = {
                "cmdline_byte_count": cmdline_byte_count,
                "cmdline_sha256": cmdline_sha256,
                "escaped_process_group": record["pgrp"] != root_pid,
                "escaped_session": record["session"] != root_pid,
                "executable_basename": executable_basename,
                "executable_sha256": executable_sha256,
                "exit_observed_ms": None,
                "first_observed_ms": observed_ms,
                "last_observed_ms": observed_ms,
                "parent_process_token_sha256": None,
                "process_token_sha256": _token(identity_salt, pid, record["start_ticks"]),
                "role": "root" if pid == root_pid else "descendant",
                "survived_root_exit": False,
                "unexpected_executable": (
                    executable_sha256 is None
                    or executable_sha256 not in expected_executable_sha256s
                ),
                "_pgrp": record["pgrp"],
                "_pid": pid,
                "_ppid": record["ppid"],
                "_start_ticks": record["start_ticks"],
            }
            tracked[key] = existing
        else:
            existing["last_observed_ms"] = observed_ms
            existing["escaped_process_group"] = bool(
                existing["escaped_process_group"] or record["pgrp"] != root_pid
            )
            existing["escaped_session"] = bool(
                existing["escaped_session"] or record["session"] != root_pid
            )
            existing["_pgrp"] = record["pgrp"]
            existing["_ppid"] = record["ppid"]

    by_pid = {value["_pid"]: value for value in tracked.values()}
    for value in tracked.values():
        parent = by_pid.get(value["_ppid"])
        if parent is not None:
            value["parent_process_token_sha256"] = parent["process_token_sha256"]

    for (pid, start_ticks), value in tracked.items():
        current = snapshot.get(pid)
        if (
            value["exit_observed_ms"] is None
            and (current is None or current["start_ticks"] != start_ticks)
        ):
            value["exit_observed_ms"] = observed_ms


def _alive_tracked(
    snapshot: dict[int, dict[str, int]], tracked: dict[tuple[int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for (pid, start_ticks), value in tracked.items():
        current = snapshot.get(pid)
        if current is not None and current["start_ticks"] == start_ticks:
            result.append(value)
    return result


def _mark_survived_root_exit(alive: list[dict[str, Any]]) -> None:
    for item in alive:
        if item["role"] == "descendant":
            item["survived_root_exit"] = True


def _sanitize_process_records(tracked: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in sorted(
        tracked.values(), key=lambda item: (item["first_observed_ms"], item["process_token_sha256"])
    ):
        records.append({key: item for key, item in value.items() if not key.startswith("_")})
    return records


def run_bounded_process(
    argv: Sequence[str],
    *,
    cwd: str,
    environment: Mapping[str, str],
    evaluator: Evaluator,
    expected_executable_sha256s: set[str],
    identity_salt: str,
    timeout_seconds: float,
    natural_quiescence_seconds: float,
    term_grace_seconds: float,
    kill_grace_seconds: float,
    poll_interval_seconds: float = 0.01,
) -> dict[str, Any]:
    if not argv or natural_quiescence_seconds < 0:
        raise ValueError("invalid bounded-process policy")
    started = time.monotonic()
    root_fallback_identity = _requested_runtime_identity(str(argv[0]), environment)
    process: subprocess.Popen[bytes] | None = None
    tracked: dict[tuple[int, int], dict[str, Any]] = {}
    root_start_ticks: int | None = None
    root_exit_ms: float | None = None
    natural_quiescence_ms: float | None = None
    term_sent = False
    kill_sent = False
    timed_out = False
    cleanup_uncertain = False

    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=False,
                start_new_session=True,
            )
            initial = _read_proc_stat(process.pid)
            if initial is not None:
                root_start_ticks = initial["start_ticks"]
            root_deadline = started + timeout_seconds
            natural_deadline: float | None = None

            while True:
                snapshot = _proc_snapshot()
                _discover(
                    snapshot,
                    process.pid,
                    root_start_ticks,
                    tracked,
                    identity_salt,
                    expected_executable_sha256s,
                    root_fallback_identity,
                    started,
                )
                return_code = process.poll()
                now = time.monotonic()
                if return_code is None:
                    if now >= root_deadline:
                        timed_out = True
                        break
                else:
                    if root_exit_ms is None:
                        root_exit_ms = round((now - started) * 1000, 3)
                        natural_deadline = now + natural_quiescence_seconds
                    live = [item for item in _alive_tracked(snapshot, tracked) if item["role"] != "root"]
                    _mark_survived_root_exit(live)
                    if not live and not _group_exists(process.pid):
                        natural_quiescence_ms = round((now - started) * 1000, 3)
                        break
                    if natural_deadline is not None and now >= natural_deadline:
                        break
                time.sleep(poll_interval_seconds)

            snapshot = _proc_snapshot()
            _discover(
                snapshot,
                process.pid,
                root_start_ticks,
                tracked,
                identity_salt,
                expected_executable_sha256s,
                root_fallback_identity,
                started,
            )
            survivors = _alive_tracked(snapshot, tracked)
            if root_exit_ms is not None:
                _mark_survived_root_exit(survivors)
            cleanup_required = timed_out or bool(survivors) or _group_exists(process.pid)
            if cleanup_required:
                term_sent = _signal_group(process.pid, signal.SIGTERM)
                for item in survivors:
                    if item["_pgrp"] != process.pid:
                        term_sent = bool(
                            _signal_identity(item["_pid"], item["_start_ticks"], signal.SIGTERM)
                            or term_sent
                        )
                term_deadline = time.monotonic() + term_grace_seconds
                while time.monotonic() < term_deadline:
                    snapshot = _proc_snapshot()
                    _discover(
                        snapshot,
                        process.pid,
                        root_start_ticks,
                        tracked,
                        identity_salt,
                        expected_executable_sha256s,
                        root_fallback_identity,
                        started,
                    )
                    if not _alive_tracked(snapshot, tracked) and not _group_exists(process.pid):
                        break
                    time.sleep(poll_interval_seconds)

                snapshot = _proc_snapshot()
                survivors = _alive_tracked(snapshot, tracked)
                if root_exit_ms is not None:
                    _mark_survived_root_exit(survivors)
                if survivors or _group_exists(process.pid):
                    kill_sent = _signal_group(process.pid, signal.SIGKILL)
                    for item in survivors:
                        if item["_pgrp"] != process.pid:
                            kill_sent = bool(
                                _signal_identity(item["_pid"], item["_start_ticks"], signal.SIGKILL)
                                or kill_sent
                            )
                    kill_deadline = time.monotonic() + kill_grace_seconds
                    while time.monotonic() < kill_deadline:
                        snapshot = _proc_snapshot()
                        _discover(
                            snapshot,
                            process.pid,
                            root_start_ticks,
                            tracked,
                            identity_salt,
                            expected_executable_sha256s,
                            root_fallback_identity,
                            started,
                        )
                        if not _alive_tracked(snapshot, tracked) and not _group_exists(process.pid):
                            break
                        time.sleep(poll_interval_seconds)

            try:
                return_code = process.wait(timeout=max(term_grace_seconds, kill_grace_seconds, 0.1))
            except subprocess.TimeoutExpired:
                return_code = process.poll()
                cleanup_uncertain = True

            final_snapshot = _proc_snapshot()
            _discover(
                final_snapshot,
                process.pid,
                root_start_ticks,
                tracked,
                identity_salt,
                expected_executable_sha256s,
                root_fallback_identity,
                started,
            )
            final_survivors = _alive_tracked(final_snapshot, tracked)
            group_absent = not _group_exists(process.pid)
            cleanup_uncertain = bool(cleanup_uncertain or final_survivors or not group_absent)
        except (OSError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
            return_code = None
            cleanup_uncertain = True
            process_exception_type: str | None = type(exc).__name__
        else:
            process_exception_type = None

        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()

    records = _sanitize_process_records(tracked)
    escaped = any(
        item["role"] == "descendant"
        and (item["escaped_process_group"] or item["escaped_session"])
        for item in records
    )
    unexpected_lineage = any(
        item["unexpected_executable"]
        and (item["role"] == "root" or item["survived_root_exit"])
        for item in records
    )
    if cleanup_uncertain:
        process_exception_type = "CleanupUncertain"
    elif timed_out:
        process_exception_type = "TimeoutExpired"
    elif escaped:
        process_exception_type = "EscapedDescendantProcess"
    elif unexpected_lineage:
        process_exception_type = "UnexpectedExecutableLineage"
    elif term_sent or kill_sent:
        process_exception_type = "NaturalQuiescenceTimeout"

    try:
        evaluation = evaluator(stdout, stderr, int(return_code) if return_code is not None else -1)
    except Exception as exc:
        evaluation = {"parse_error_type": type(exc).__name__, "passed": False}

    lifecycle = {
        "cleanup_uncertain": cleanup_uncertain,
        "escaped_descendant_detected": escaped,
        "group_absent_after_cleanup": not cleanup_uncertain,
        "kill_sent": kill_sent,
        "natural_quiescence_bound_ms": round(natural_quiescence_seconds * 1000, 3),
        "natural_quiescence_observed_ms": (
            None
            if root_exit_ms is None or natural_quiescence_ms is None
            else round(natural_quiescence_ms - root_exit_ms, 3)
        ),
        "root_exit_observed_ms": root_exit_ms,
        "root_reaped": process is not None and process.poll() is not None,
        "start_new_session": True,
        "term_sent": term_sent,
        "timeout_seconds": timeout_seconds,
        "unexpected_executable_lineage": unexpected_lineage,
    }
    passed = bool(
        process_exception_type is None
        and return_code == 0
        and evaluation.get("passed") is True
        and lifecycle["group_absent_after_cleanup"]
    )
    return {
        "argv": list(argv),
        "evaluation": evaluation,
        "passed": passed,
        "process_exception_type": process_exception_type,
        "process_lifecycle": lifecycle,
        "process_tree": {
            "identity_salt_persisted": False,
            "observed_process_count": len(records),
            "records": records,
        },
        "sanitized_streams": digest_streams(return_code, stdout, stderr),
    }
