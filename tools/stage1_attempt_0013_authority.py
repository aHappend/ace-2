#!/usr/bin/env python3
"""Create and verify the unconsumed Stage-1 attempt-0013 authority state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/ace2_chat_demo"
PACKAGE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-preparation-r4"
)
QUALIFICATION = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-preparation-r4-qualification"
)
REVIEW = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-a62e0c91-l2-review/fresh-review.json"
)
STATE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-a62e0c91-authority-state"
)
QUALIFICATION_OUTPUT = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-a62e0c91-authority-qualification"
)
TOOL_REL = "tools/stage1_attempt_0013_authority.py"
PACKAGE_REL = PACKAGE.relative_to(ROOT).as_posix()
QUALIFICATION_REL = QUALIFICATION.relative_to(ROOT).as_posix()
REVIEW_REL = REVIEW.relative_to(ROOT).as_posix()
STATE_REL = STATE.relative_to(ROOT).as_posix()
FOCUSED_REL = (
    f"{QUALIFICATION_REL}/01-focused-validation.status.json"
)
EXPECTED_PACKAGE_ROOT = (
    "ee3bebca082356fbf4fa6e9b3fb710725d5a451c257fe88a0150517e6125c828"
)
EXPECTED_QUALIFICATION_ROOT = (
    "b1037fedba0ed15463467bedd2db57259f70c3189eec11350659450bbb558150"
)
EXPECTED_FOCUSED_SHA256 = (
    "c2cb19da9a4ab4dc8385ccf8a3fb2ec8b0996b66e9541c4ff2adb14cecf77866"
)
EXPECTED_REVIEW_SHA256 = (
    "057e0eb99504ca2e3dccaa957dcb6a218d38312202789e028bec80b9c221dbf7"
)
IMMUTABLE_STATE_FILES = {
    "manager-authority.json",
    "reviewer-accept.json",
    "SHA256SUMS",
    "TREE_ROOT.sha256",
}
RUNTIME_STATE_FILES = {
    "child.started.json",
    "execution-consumed.json",
    "invocation.json",
    "run.started.json",
    "stderr.capture.json",
    "stdout.capture.json",
    "submission-release.json",
    "submit-intent.json",
    "submit-receipt.json",
    "terminal-owner.lock",
    "terminal-status.json",
    "worker-handoff.json",
    "worker-waiting.json",
}


class AuthorityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuthorityError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AuthorityError(f"nonfinite JSON value in {path}: {token}")
        ),
    )
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_exclusive(path: Path, raw: bytes, mode: int = 0o400) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, f"short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_sealed_tree(path: Path, expected_root: str) -> None:
    require(path.is_dir() and not path.is_symlink(), f"tree absent: {path}")
    sums_path = path / "SHA256SUMS"
    root_path = path / "TREE_ROOT.sha256"
    require(sha256_file(sums_path) == expected_root, f"tree root drift: {path}")
    root_parts = root_path.read_text(encoding="ascii").split()
    require(
        root_parts == [expected_root, "SHA256SUMS"],
        f"tree root sidecar drift: {path}",
    )
    records: dict[str, str] = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"malformed checksum line: {path}")
        assert match is not None
        require(match[2] not in records, f"duplicate checksum member: {match[2]}")
        records[match[2]] = match[1]
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
        and not item.is_symlink()
        and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(actual == set(records), f"sealed tree member drift: {path}")
    for relative, expected in records.items():
        member = path / relative
        require(
            member.is_file()
            and not member.is_symlink()
            and sha256_file(member) == expected,
            f"sealed tree member changed: {member}",
        )


def file_binding(path: Path) -> dict[str, Any]:
    status = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(status.st_mode), f"regular file required: {path}")
    return {
        "bytes": status.st_size,
        "mode": stat.S_IMODE(status.st_mode),
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def interpreter_binding() -> dict[str, Any]:
    launcher = shutil.which("python3")
    require(launcher is not None, "python3 is absent from PATH")
    launcher_path = Path(launcher)
    resolved = launcher_path.resolve(strict=True)
    status = resolved.stat()
    require(stat.S_ISREG(status.st_mode), "resolved production interpreter is not regular")
    return {
        "argus_skill_python": {
            "present": "ARGUS_SKILL_PYTHON" in os.environ,
            "value": os.environ.get("ARGUS_SKILL_PYTHON"),
        },
        "launcher_token": "python3",
        "launcher_path": str(launcher_path),
        "launcher_symlink_target": (
            os.readlink(launcher_path) if launcher_path.is_symlink() else None
        ),
        "resolved_path": str(resolved),
        "resolved_bytes": status.st_size,
        "resolved_mode": stat.S_IMODE(status.st_mode),
        "resolved_sha256": sha256_file(resolved),
        "version": sys.version.splitlines()[0],
    }


def load_runner() -> Any:
    path = PACKAGE / "terminal_runner.py"
    specification = importlib.util.spec_from_file_location(
        "ace2_attempt_0013_terminal_runner_authority_check", path
    )
    require(specification is not None and specification.loader is not None, "runner import")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def expected_records(created_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    package = load_object(PACKAGE / "package.json")
    commands = load_object(PACKAGE / "commands.json")
    capacity = load_object(PACKAGE / "capacity.json")
    storage = load_object(PACKAGE / "storage-contract.json")
    focused = load_object(ROOT / FOCUSED_REL)
    review = load_object(REVIEW)
    cwd_status = ROOT.stat()
    submit_wrapper = file_binding(PACKAGE / "submit_and_capture.sh")
    launch_wrapper = file_binding(PACKAGE / "launch.sh")
    terminal_runner = file_binding(PACKAGE / "terminal_runner.py")
    tool = file_binding(ROOT / TOOL_REL)
    immutable_bindings = {
        "focused_status": {
            "path": FOCUSED_REL,
            "sha256": EXPECTED_FOCUSED_SHA256,
        },
        "fresh_review": {
            "decision": "PASS",
            "decision_identity": "FRESH_REVIEWER_DECISION",
            "path": REVIEW_REL,
            "sha256": EXPECTED_REVIEW_SHA256,
        },
        "package": {
            "identity": package["identity"],
            "path": PACKAGE_REL,
            "tree_root_sha256": EXPECTED_PACKAGE_ROOT,
        },
        "qualification": {
            "path": QUALIFICATION_REL,
            "tree_root_sha256": EXPECTED_QUALIFICATION_ROOT,
        },
    }
    production = {
        "cwd": {
            "device": cwd_status.st_dev,
            "inode": cwd_status.st_ino,
            "path": str(ROOT),
        },
        "front_door": {
            **submit_wrapper,
            "argv": ["bash", submit_wrapper["path"]],
            "shell": commands["submission_entrypoint"],
        },
        "interpreter": interpreter_binding(),
        "launch_wrapper": launch_wrapper,
        "terminal_runner": terminal_runner,
    }
    submission = {
        "check_with": commands["check_with"],
        "durable_submit_argv": commands["durable_submit_argv"],
        "durable_submit_shell": commands["durable_submit_shell"],
        "package_launch_argv": commands["package_launch_argv"],
        "resolved_first_argv": os.environ.get("ARGUS_SKILL_PYTHON", "python3"),
        "submitter_entrypoint_argv": commands["submitter_entrypoint_argv"],
    }
    storage_limits = {
        "capacity_path": f"{PACKAGE_REL}/capacity.json",
        "capacity_sha256": sha256_file(PACKAGE / "capacity.json"),
        "filesystem_device": capacity["filesystem_device"],
        "package_max_bytes": capacity["package_max_bytes"],
        "required_available_bytes": capacity["required_available_bytes"],
        "runtime_output_max_bytes": capacity["runtime_output_max_bytes"],
        "safety_margin_bytes": capacity["safety_margin_bytes"],
        "storage_contract_path": f"{PACKAGE_REL}/storage-contract.json",
        "storage_contract_sha256": sha256_file(PACKAGE / "storage-contract.json"),
        "temporary_write_max_bytes": capacity["temporary_write_max_bytes"],
        "growth_policy": storage["growth_policy"],
    }
    terminal_policy = {
        "consumption_sentinel": "execution-consumed.json",
        "consumption_sentinel_publication": "O_CREAT_O_EXCL_BEFORE_SUBMISSION_OR_MODEL",
        "execution_limit": 1,
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "second_model_invocation_permitted": False,
        "terminal_cardinality": 1,
        "terminal_status": "SEALED_TERMINAL_NO_RETRY",
        "terminal_status_paths": [
            "terminal-status.json",
            "fallback/terminal-status.json",
        ],
    }
    manager = {
        "authority_state": "UNCONSUMED",
        "binding_tool": tool,
        "created_at_utc": created_at,
        "execution_limit": 1,
        "grant_provenance": {
            "mission_id": "cd3a5253bf31",
            "node_key": "issue-r4-attempt-0013-authority",
            "source_kind": "LIVE_MANAGER_OPERATOR_DIRECTIVE",
            "scope": "ONE_FRESH_R4_NO_EXECUTION_AUTHORITY_STATE",
        },
        "immutable_bindings": immutable_bindings,
        "nonce": package["nonce"],
        "output": package["future_output"],
        "package_identity": package["identity"],
        "package_run_identity": package["package_run_identity"],
        "package_tree_root": EXPECTED_PACKAGE_ROOT,
        "production": production,
        "schema": "ace2-attempt-manager-authority-v1",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "status": "GRANTED_EXACTLY_ONCE",
        "storage_limits": storage_limits,
        "submission": submission,
        "task_id": package["future_task_id"],
        "terminal_policy": terminal_policy,
    }
    reviewer = {
        "artifact_producer_role": "engineer",
        "canonical_review_receipt_path": str(REVIEW),
        "canonical_review_receipt_sha256": EXPECTED_REVIEW_SHA256,
        "decision": "ACCEPT",
        "decision_projection": "FRESH_REVIEWER_DECISION_PASS_TO_CONSUMER_ACCEPT",
        "execution_limit": 1,
        "independent": True,
        "nonce": package["nonce"],
        "output": package["future_output"],
        "package_identity": package["identity"],
        "package_run_identity": package["package_run_identity"],
        "package_tree_root": EXPECTED_PACKAGE_ROOT,
        "preparation_participation": False,
        "producer_role": "reviewer",
        "review_source": immutable_bindings["fresh_review"],
        "reviewer_level": "L2",
        "schema": "ace2-attempt-independent-review-v1",
        "task_id": package["future_task_id"],
    }
    require(focused["status"] == "PASS", "focused status is not PASS")
    require(review["decision"] == "PASS", "fresh review is not PASS")
    return manager, reviewer


def parse_created_at(manager: dict[str, Any]) -> str:
    value = manager.get("created_at_utc")
    require(isinstance(value, str), "authority creation time is absent")
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "authority creation time lacks timezone")
    return value


def authority_candidates(build: Path = BUILD) -> list[Path]:
    pattern = (
        "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
        "attempt-0013-*-authority-state"
    )
    return sorted(
        path for path in build.glob(pattern) if path.is_dir() and not path.is_symlink()
    )


def require_authority_cardinality(build: Path, expected: list[Path]) -> None:
    candidates = authority_candidates(build)
    require(candidates == expected, f"authority cardinality is {len(candidates)}")


def verify_record_values(
    manager: dict[str, Any], reviewer: dict[str, Any]
) -> None:
    expected_manager, expected_reviewer = expected_records(parse_created_at(manager))
    require(manager == expected_manager, "manager authority binding mismatch")
    require(reviewer == expected_reviewer, "review projection binding mismatch")


def verify_no_activity(manager: dict[str, Any]) -> None:
    package = load_object(PACKAGE / "package.json")
    require(not os.path.lexists(ROOT / package["future_output"]), "runtime output exists")
    task_id = package["future_task_id"]
    subagents = ROOT / ".argus_subagents"
    if subagents.is_dir():
        require(
            not any(task_id in item.name for item in subagents.iterdir()),
            "submission state or log exists",
        )
    markers = (
        task_id,
        package["future_output"],
        f"{PACKAGE_REL}/launch.sh",
        f"{PACKAGE_REL}/terminal_runner.py",
    )
    live: list[int] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        if any(marker in command for marker in markers):
            live.append(int(entry.name))
    require(not live, f"live attempt process exists: {live}")
    require(manager["authority_state"] == "UNCONSUMED", "authority is not unconsumed")


def verify_state() -> dict[str, Any]:
    verify_sealed_tree(PACKAGE, EXPECTED_PACKAGE_ROOT)
    verify_sealed_tree(QUALIFICATION, EXPECTED_QUALIFICATION_ROOT)
    require(sha256_file(ROOT / FOCUSED_REL) == EXPECTED_FOCUSED_SHA256, "focused hash")
    require(sha256_file(REVIEW) == EXPECTED_REVIEW_SHA256, "fresh review hash")
    require_authority_cardinality(BUILD, [STATE])
    require(STATE.is_dir() and not STATE.is_symlink(), "authority state is absent")
    actual = {
        item.relative_to(STATE).as_posix()
        for item in STATE.rglob("*")
        if item.is_file() and not item.is_symlink()
    }
    require(actual == IMMUTABLE_STATE_FILES, "authority state member set is not unconsumed")
    require(
        not any(os.path.lexists(STATE / name) for name in RUNTIME_STATE_FILES),
        "runtime lifecycle state exists",
    )
    manager = load_object(STATE / "manager-authority.json")
    reviewer = load_object(STATE / "reviewer-accept.json")
    require(
        canonical_bytes(manager) == (STATE / "manager-authority.json").read_bytes(),
        "manager authority is not canonical",
    )
    require(
        canonical_bytes(reviewer) == (STATE / "reviewer-accept.json").read_bytes(),
        "review projection is not canonical",
    )
    verify_record_values(manager, reviewer)
    lines = [
        f"{sha256_file(STATE / name)}  {name}"
        for name in ("manager-authority.json", "reviewer-accept.json")
    ]
    expected_sums = ("\n".join(lines) + "\n").encode("ascii")
    require((STATE / "SHA256SUMS").read_bytes() == expected_sums, "state checksums")
    tree_root = hashlib.sha256(expected_sums).hexdigest()
    require(
        (STATE / "TREE_ROOT.sha256").read_text(encoding="ascii")
        == f"{tree_root}  SHA256SUMS\n",
        "authority tree root sidecar mismatch",
    )
    output_parent = (ROOT / manager["output"]).parent
    filesystem = os.statvfs(output_parent)
    available = filesystem.f_bavail * filesystem.f_frsize
    require(
        output_parent.stat().st_dev
        == manager["storage_limits"]["filesystem_device"],
        "storage device changed",
    )
    require(
        available >= manager["storage_limits"]["required_available_bytes"],
        "storage is below the frozen launch floor",
    )
    verify_no_activity(manager)
    runner = load_runner()
    with mock.patch.object(sys, "path", [str(PACKAGE), *sys.path]):
        runner.validate_external_gates(require_receipt=False)
    return {
        "authority_cardinality": 1,
        "authority_consumed": False,
        "authority_state": STATE_REL,
        "authority_tree_root_sha256": tree_root,
        "bindings_exact": True,
        "model_activity": 0,
        "rtl_activity": 0,
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "submission_activity": 0,
    }


def negative_controls() -> dict[str, bool]:
    runner = load_runner()
    manager = load_object(STATE / "manager-authority.json")
    reviewer = load_object(STATE / "reviewer-accept.json")
    results: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="ace2-0013-authority-") as directory:
        temporary = Path(directory)
        sentinel = temporary / "execution-consumed.json"
        first = {"identity": "isolated-negative-control"}
        runner.publish(sentinel, first)
        before = sentinel.read_bytes()
        try:
            runner.publish(sentinel, {"identity": "duplicate"})
        except FileExistsError:
            results["exclusive_consumption_sentinel"] = sentinel.read_bytes() == before
        else:
            results["exclusive_consumption_sentinel"] = False

        bad_manager = dict(manager)
        bad_manager["package_tree_root"] = "0" * 64
        write_exclusive(
            temporary / "manager-authority.json", canonical_bytes(bad_manager)
        )
        write_exclusive(
            temporary / "reviewer-accept.json", canonical_bytes(reviewer)
        )
        with mock.patch.object(runner, "STATE", temporary):
            with mock.patch.object(sys, "path", [str(PACKAGE), *sys.path]):
                try:
                    runner.validate_external_gates(require_receipt=False)
                except runner.GateError:
                    results["misbound_manager_rejected"] = True
                else:
                    results["misbound_manager_rejected"] = False

        duplicate_root = temporary / "cardinality"
        duplicate_root.mkdir()
        for suffix in ("a62e0c91", "ffffffff"):
            (duplicate_root / (
                "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
                f"attempt-0013-{suffix}-authority-state"
            )).mkdir()
        try:
            require_authority_cardinality(duplicate_root, [])
        except AuthorityError:
            results["duplicate_authority_rejected"] = True
        else:
            results["duplicate_authority_rejected"] = False

        bad_reviewer = dict(reviewer)
        bad_reviewer["canonical_review_receipt_sha256"] = "0" * 64
        try:
            verify_record_values(manager, bad_reviewer)
        except AuthorityError:
            results["misbound_review_rejected"] = True
        else:
            results["misbound_review_rejected"] = False
    require(all(results.values()), "one or more authority negative controls failed")
    return results


def create_state() -> None:
    require(not os.path.lexists(STATE), f"authority state already exists: {STATE}")
    require_authority_cardinality(BUILD, [])
    require(not os.path.lexists(QUALIFICATION_OUTPUT), "authority qualification exists")
    verify_sealed_tree(PACKAGE, EXPECTED_PACKAGE_ROOT)
    verify_sealed_tree(QUALIFICATION, EXPECTED_QUALIFICATION_ROOT)
    require(sha256_file(ROOT / FOCUSED_REL) == EXPECTED_FOCUSED_SHA256, "focused hash")
    require(sha256_file(REVIEW) == EXPECTED_REVIEW_SHA256, "fresh review hash")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manager, reviewer = expected_records(created_at)
    STATE.mkdir(mode=0o700)
    try:
        write_exclusive(
            STATE / "manager-authority.json", canonical_bytes(manager)
        )
        write_exclusive(
            STATE / "reviewer-accept.json", canonical_bytes(reviewer)
        )
        lines = [
            f"{sha256_file(STATE / name)}  {name}"
            for name in ("manager-authority.json", "reviewer-accept.json")
        ]
        sums = ("\n".join(lines) + "\n").encode("ascii")
        write_exclusive(STATE / "SHA256SUMS", sums)
        tree_root = hashlib.sha256(sums).hexdigest()
        write_exclusive(
            STATE / "TREE_ROOT.sha256",
            f"{tree_root}  SHA256SUMS\n".encode("ascii"),
        )
        fsync_dir(STATE)
        fsync_dir(STATE.parent)
    except BaseException:
        for name in IMMUTABLE_STATE_FILES:
            path = STATE / name
            if path.exists():
                path.unlink()
        STATE.rmdir()
        raise


def create_and_verify() -> dict[str, Any]:
    create_state()
    verification = verify_state()
    controls = negative_controls()
    report = {
        **verification,
        "activity": {
            "durable_runner_submitted": False,
            "model_executed": False,
            "model_process_started": False,
            "rtl_executed": False,
            "rtl_process_started": False,
        },
        "negative_controls": controls,
        "review_status": "PENDING_INDEPENDENT_AUTHORITY_REVIEW",
        "schema": "ace2-attempt-0013-r4-authority-qualification-v1",
        "status": "PASS",
    }
    QUALIFICATION_OUTPUT.mkdir(mode=0o700)
    try:
        write_exclusive(
            QUALIFICATION_OUTPUT / "verification.json",
            canonical_bytes(report),
            0o444,
        )
        fsync_dir(QUALIFICATION_OUTPUT)
        QUALIFICATION_OUTPUT.chmod(0o555)
        fsync_dir(QUALIFICATION_OUTPUT.parent)
    except BaseException:
        path = QUALIFICATION_OUTPUT / "verification.json"
        if path.exists():
            path.unlink()
        QUALIFICATION_OUTPUT.rmdir()
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("create-and-verify", "verify"))
    arguments = parser.parse_args()
    if arguments.operation == "create-and-verify":
        result = create_and_verify()
    else:
        result = verify_state()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
