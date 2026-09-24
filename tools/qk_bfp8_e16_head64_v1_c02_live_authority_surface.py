#!/usr/bin/env python3
"""Fail-closed transaction core for the c02 QK live-authority surface.

This module deliberately contains no live shell argv and no evaluator.  Its public
entry point therefore refuses live use.  The narrow materializer/executor core
loads only a configured checksum-bound launch authority and launches its exact
argv under its exact environment after the selected lane's durable
READY_UNCONSUMED -> CONSUMED transition.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT
CANONICAL_ROOT = REPOSITORY_ROOT
LIVE_SURFACE_REL = (
    "build/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v1"
)
LIVE_SURFACE_ROOT = ROOT / LIVE_SURFACE_REL
OUTPUT_ROOT_REL = "build/qk-bfp8-e16-head64-v1-c02-score-path-evaluation-v1"
OUTPUT_ROOT = ROOT / OUTPUT_ROOT_REL

AUTHORITY_PACKAGE_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EXTERNAL_EXACTLY_ONCE_AUTHORITY_PACKAGE.json"
)
AUTHORITY_VERIFIER_REL = (
    "tools/verify_qk_bfp8_e16_head64_v1_c02_score_path_external_exactly_once_authority.py"
)
AUTHORITY_CHECKSUMS_REL = (
    "evidence/verification/"
    "qk-bfp8-e16-head64-v1-c02-score-path-external-exactly-once-authority-v1/"
    "SHA256SUMS"
)
QUALIFYING_REVIEW_SOURCE = (
    Path.home()
    / ".argus-skill-ace2/projects/s-c8ae985b/handoffs/8c1b4f7e2a96/round-0001.json"
)
PREFLIGHT_REVIEW_SOURCE = (
    Path.home()
    / ".argus-skill-ace2/projects/s-c8ae985b/handoffs/51a7d3bc902e/round-0001.json"
)

# The frozen accepted authority contains no exact argv or executable identity.
# A future authority amendment must set both of these checksum-bound values in
# a new implementation-package revision.  None is deliberately fail closed.
LAUNCH_AUTHORITY_PATH: Path | None = None
LAUNCH_AUTHORITY_SHA256: str | None = None
LAUNCH_AUTHORITY_ID = "QK_BFP8_E16_HEAD64_V1_C02_LIVE_LAUNCH_AUTHORITY_V1"

AUTHORITY_PACKAGE_SHA256 = (
    "3f31da79730ce25040148365091c902fa8ce611b0499d9137a6a053fdbf7285c"
)
AUTHORITY_VERIFIER_SHA256 = (
    "401852322e5b4d421c97e4579bb29a069c49995659eba78c9317df37a373f4aa"
)
CHECKSUM_MANIFEST_SHA256 = (
    "5fe93639f8988f451be8caedaadda641fcc46ef47b6b4333ffaf6cfcde8aa43a"
)
QUALIFYING_AUTHORITY_REVIEW_SHA256 = (
    "88fd0fbc620e2978ba23db223276764e348eda96e37da7a1b39d74c116d2ffbb"
)
FAIL_CLOSED_PREFLIGHT_REVIEW_SHA256 = (
    "89ef3ebee0b07b559fa39c03a0dc84ea81a077fe8ba9d04eabe1fc4e387a2a84"
)
EVALUATOR_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATOR_SPEC_V1"
EVALUATOR_SPEC_SHA256 = (
    "b719d18f571e664b25c926ecb972f9972f2a13c4b8013d89f8bf970cf8a99756"
)
ACCEPTED_EVALUATION_PACKAGE_SHA256 = (
    "0bd51407f20950a93f47a520718a3bb4d5a5006d990d60aef626d16a5f9d976d"
)
COMMAND_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATE_ONCE_V1"
COMMAND_ENCODING = "canonical-json-external-command-descriptor-v1"
EXECUTOR_OPERATION = "evaluate_score_path_once"

ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}

LANES: dict[str, dict[str, str]] = {
    "Base": {
        "command_sha256": (
            "a89602cbcf8d4b5a3d7fd11737615e399ce2d91d869ea1ea46b7a0f1f1b6cb4f"
        ),
        "invocation_sha256": (
            "58b90f06f10d179cb12820ee5a376cf3807eb9f2af929990dacac138a8fa6731"
        ),
        "namespace_label": "base",
        "output_path": OUTPUT_ROOT_REL + "/base/result.json",
        "record_id": "c02-score-path-base-exactly-once-v1",
    },
    "checkpoint-176": {
        "command_sha256": (
            "b9bb0df45304965056f5df21a44f194a22341995923eff0ce20ccd89022523ec"
        ),
        "invocation_sha256": (
            "78e83c2f2ad2a5ec15a96dc320bd3e86f5ed3fbedbf3de9871a03b40225c829b"
        ),
        "namespace_label": "checkpoint-176",
        "output_path": OUTPUT_ROOT_REL + "/checkpoint-176/result.json",
        "record_id": "c02-score-path-checkpoint-176-exactly-once-v1",
    },
}

READY_SCHEMA_ID = "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_READY_V1"
TERMINAL_SCHEMA_ID = "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_TERMINAL_V1"
CONTRACT_SCHEMA_ID = "QK_BFP8_E16_HEAD64_V1_C02_LIVE_SURFACE_CONTRACT_V1"
READY_STATE = "READY_UNCONSUMED"
CONSUMED_STATE = "CONSUMED"
INVALIDATED_STATE = "INVALIDATED_TERMINAL"
TERMINAL_STATES = {"SUCCEEDED_TERMINAL", "FAILED_TERMINAL"}


class SurfaceError(RuntimeError):
    """A fail-closed authority-surface validation or transition failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SurfaceError(message)


def fail_live(message: str) -> NoReturn:
    print(f"LIVE_SURFACE_BLOCKED_NO_EXECUTION_PERFORMED: {message}", file=sys.stderr)
    raise SystemExit(2)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(compact_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_canonical_json(path: Path) -> Any:
    verify_plain_regular_file(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SurfaceError(f"invalid JSON file: {path}: {exc}") from exc
    require(raw == canonical_json_bytes(value), f"noncanonical JSON file: {path}")
    return value


def lane_slug(lane_label: str) -> str:
    require(lane_label in LANES, "unknown lane")
    return LANES[lane_label]["namespace_label"]


def lane_paths(surface_root: Path, lane_label: str) -> dict[str, Path]:
    lane_root = surface_root / "lanes" / lane_slug(lane_label)
    return {
        "lane_root": lane_root,
        "ledger": lane_root / "authority-ledger.json",
        "lock": lane_root / ".transition.lock",
        "terminal": lane_root / "first-terminal.json",
    }


def lexical_exists(path: Path) -> bool:
    return os.path.lexists(path)


def require_canonical_absolute_path(path: Path, expected: Path, label: str) -> None:
    require(isinstance(path, Path), f"{label} path type")
    require(path.is_absolute(), f"{label} is not absolute")
    require(path == expected, f"{label} is not the canonical path")
    require(path == Path(os.path.normpath(str(path))), f"{label} is noncanonical")


def verify_no_symlink_components(path: Path) -> None:
    require(path.is_absolute(), f"path is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        require(not stat.S_ISLNK(info.st_mode), f"symlinked path component: {current}")


def verify_plain_regular_file(path: Path) -> None:
    verify_no_symlink_components(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SurfaceError(f"missing regular file: {path}") from exc
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")


def validate_runtime_roots() -> None:
    require_canonical_absolute_path(ROOT, CANONICAL_ROOT, "repository root")
    require_canonical_absolute_path(
        LIVE_SURFACE_ROOT,
        ROOT / LIVE_SURFACE_REL,
        "live surface root",
    )
    require_canonical_absolute_path(
        OUTPUT_ROOT,
        ROOT / OUTPUT_ROOT_REL,
        "score output root",
    )
    verify_no_symlink_components(ROOT)
    verify_no_symlink_components(LIVE_SURFACE_ROOT)
    verify_no_symlink_components(OUTPUT_ROOT)


def relative_path_text(path: Path, anchor: Path) -> str:
    try:
        relative = path.relative_to(anchor)
    except ValueError as exc:
        raise SurfaceError(f"path escapes anchor: {path}") from exc
    pure = PurePosixPath(relative.as_posix())
    require(".." not in pure.parts and "." not in pure.parts, "noncanonical path")
    return pure.as_posix()


def verify_plain_directory(path: Path, mode: int = 0o700) -> None:
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode), f"not a directory: {path}")
    require(not path.is_symlink(), f"symlinked directory: {path}")
    require(stat.S_IMODE(info.st_mode) == mode, f"directory mode mismatch: {path}")
    require(info.st_uid == os.geteuid(), f"directory owner mismatch: {path}")


def verify_plain_file(path: Path, mode: int = 0o400) -> None:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
    require(not path.is_symlink(), f"symlinked file: {path}")
    require(stat.S_IMODE(info.st_mode) == mode, f"file mode mismatch: {path}")
    require(info.st_uid == os.geteuid(), f"file owner mismatch: {path}")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_write_json(path: Path, value: Any, mode: int = 0o400) -> None:
    payload = canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            require(count > 0, "short create-only write")
            view = view[count:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    verify_plain_file(path, mode)
    fsync_directory(path.parent)


def atomic_replace_json(path: Path, value: Any, mode: int = 0o400) -> None:
    require(lexical_exists(path), f"missing transition source: {path}")
    verify_plain_file(path, mode)
    temporary = path.with_name(
        f".{path.name}.transition-{os.getpid()}-{time.time_ns()}"
    )
    try:
        exclusive_write_json(temporary, value, mode)
        os.replace(temporary, path)
        fsync_directory(path.parent)
        verify_plain_file(path, mode)
    finally:
        if lexical_exists(temporary):
            temporary.unlink()
            fsync_directory(path.parent)


def fixed_bindings() -> dict[str, str]:
    return {
        "authority_package_sha256": AUTHORITY_PACKAGE_SHA256,
        "authority_verifier_sha256": AUTHORITY_VERIFIER_SHA256,
        "checksum_manifest_sha256": CHECKSUM_MANIFEST_SHA256,
        "fail_closed_preflight_review_sha256": FAIL_CLOSED_PREFLIGHT_REVIEW_SHA256,
        "qualifying_authority_review_sha256": QUALIFYING_AUTHORITY_REVIEW_SHA256,
    }


def bound_artifacts() -> tuple[tuple[str, Path, str], ...]:
    return (
        (
            "authority package",
            REPOSITORY_ROOT / AUTHORITY_PACKAGE_REL,
            AUTHORITY_PACKAGE_SHA256,
        ),
        (
            "authority verifier",
            REPOSITORY_ROOT / AUTHORITY_VERIFIER_REL,
            AUTHORITY_VERIFIER_SHA256,
        ),
        (
            "authority checksum manifest",
            REPOSITORY_ROOT / AUTHORITY_CHECKSUMS_REL,
            CHECKSUM_MANIFEST_SHA256,
        ),
        (
            "qualifying authority review",
            QUALIFYING_REVIEW_SOURCE,
            QUALIFYING_AUTHORITY_REVIEW_SHA256,
        ),
        (
            "fail-closed preflight review",
            PREFLIGHT_REVIEW_SOURCE,
            FAIL_CLOSED_PREFLIGHT_REVIEW_SHA256,
        ),
    )


def validate_bound_artifacts() -> None:
    for label, path, expected_sha256 in bound_artifacts():
        require(path.is_absolute(), f"{label} path is not absolute")
        require(path == Path(os.path.normpath(str(path))), f"{label} path is noncanonical")
        verify_plain_regular_file(path)
        require(sha256_file(path) == expected_sha256, f"{label} hash mismatch")


def command_descriptor(lane_label: str) -> dict[str, Any]:
    lane = LANES[lane_label]
    return {
        "arguments": {
            "accepted_evaluation_package_sha256": ACCEPTED_EVALUATION_PACKAGE_SHA256,
            "invocation_sha256": lane["invocation_sha256"],
            "lane_label": lane_label,
            "output_path": lane["output_path"],
        },
        "command_id": COMMAND_ID,
        "encoding": COMMAND_ENCODING,
        "executor_operation": EXECUTOR_OPERATION,
        "shell_argv_materialized": False,
    }


def launch_descriptor(lane_label: str, lane: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_sha256": LANES[lane_label]["command_sha256"],
        "environment": lane["environment"],
        "evaluator_id": EVALUATOR_ID,
        "evaluator_spec_sha256": EVALUATOR_SPEC_SHA256,
        "executable_sha256": lane["executable_sha256"],
        "invocation_sha256": LANES[lane_label]["invocation_sha256"],
        "output_path": LANES[lane_label]["output_path"],
        "shell_argv": lane["shell_argv"],
    }


def validate_contract(contract: dict[str, Any]) -> None:
    validate_runtime_roots()
    require(set(contract) == {
        "bindings",
        "evaluator",
        "launch_authority_id",
        "lanes",
        "schema_id",
        "surface_root",
    }, "contract keys")
    require(contract["schema_id"] == CONTRACT_SCHEMA_ID, "contract schema")
    require(
        contract["launch_authority_id"] == LAUNCH_AUTHORITY_ID,
        "launch authority id",
    )
    require(contract["bindings"] == fixed_bindings(), "contract hash bindings")
    require(
        contract["surface_root"] == str(LIVE_SURFACE_ROOT),
        "contract surface root",
    )
    require(contract["evaluator"] == {
        "evaluator_id": EVALUATOR_ID,
        "evaluator_spec_sha256": EVALUATOR_SPEC_SHA256,
    }, "contract evaluator binding")
    require(set(contract["lanes"]) == set(LANES), "contract lane set")
    argv_values: list[tuple[str, ...]] = []
    for lane_label, expected in LANES.items():
        lane = contract["lanes"][lane_label]
        require(set(lane) == {
            "command_sha256",
            "environment",
            "executable_sha256",
            "invocation_sha256",
            "launch_descriptor_sha256",
            "namespace_label",
            "output_path",
            "record_id",
            "shell_argv",
        }, f"{lane_label} contract keys")
        for key in (
            "command_sha256",
            "invocation_sha256",
            "namespace_label",
            "output_path",
            "record_id",
        ):
            require(lane[key] == expected[key], f"{lane_label} {key}")
        require(lane["environment"] == ENVIRONMENT, f"{lane_label} environment")
        require(is_sha256(lane["executable_sha256"]), f"{lane_label} executable hash")
        require(
            object_sha256(command_descriptor(lane_label)) == expected["command_sha256"],
            f"{lane_label} accepted command descriptor hash",
        )
        argv = lane["shell_argv"]
        require(
            isinstance(argv, list)
            and len(argv) > 0
            and all(
                isinstance(item, str) and item and "\x00" not in item for item in argv
            ),
            f"{lane_label} exact shell argv",
        )
        executable = Path(argv[0])
        require(executable.is_absolute(), f"{lane_label} argv[0] is not absolute")
        require(
            executable == Path(os.path.normpath(str(executable))),
            f"{lane_label} argv[0] is noncanonical",
        )
        require(
            lane["launch_descriptor_sha256"]
            == object_sha256(launch_descriptor(lane_label, lane)),
            f"{lane_label} launch descriptor hash",
        )
        argv_values.append(tuple(argv))
    require(argv_values[0] != argv_values[1], "lane argv alias")
    outputs = [contract["lanes"][lane]["output_path"] for lane in LANES]
    require(len(set(outputs)) == len(outputs), "lane output alias")


def load_authorized_contract() -> dict[str, Any]:
    require(
        isinstance(LAUNCH_AUTHORITY_PATH, Path),
        "checksum-bound launch authority path is unavailable",
    )
    require(
        is_sha256(LAUNCH_AUTHORITY_SHA256),
        "checksum-bound launch authority SHA-256 is unavailable",
    )
    path = LAUNCH_AUTHORITY_PATH
    require(path.is_absolute(), "launch authority path is not absolute")
    require(
        path == Path(os.path.normpath(str(path))),
        "launch authority path is noncanonical",
    )
    verify_plain_regular_file(path)
    require(
        sha256_file(path) == LAUNCH_AUTHORITY_SHA256,
        "launch authority hash mismatch",
    )
    contract = load_canonical_json(path)
    require(isinstance(contract, dict), "launch authority object")
    validate_contract(contract)
    return contract


def make_ready_record(
    contract: dict[str, Any],
    lane_label: str,
    authority_credential_sha256: str,
) -> dict[str, Any]:
    validate_contract(contract)
    require(is_sha256(authority_credential_sha256), "authority credential hash")
    require(is_sha256(LAUNCH_AUTHORITY_SHA256), "launch authority hash")
    lane = contract["lanes"][lane_label]
    paths = lane_paths(LIVE_SURFACE_ROOT, lane_label)
    return {
        **fixed_bindings(),
        "authority_credential_sha256": authority_credential_sha256,
        "command_sha256": lane["command_sha256"],
        "contract_sha256": LAUNCH_AUTHORITY_SHA256,
        "environment": lane["environment"],
        "evaluator_id": EVALUATOR_ID,
        "evaluator_spec_sha256": EVALUATOR_SPEC_SHA256,
        "executable_sha256": lane["executable_sha256"],
        "invocation_sha256": lane["invocation_sha256"],
        "lane_label": lane_label,
        "launch_descriptor_sha256": lane["launch_descriptor_sha256"],
        "ledger_path": relative_path_text(paths["ledger"], LIVE_SURFACE_ROOT),
        "namespace_label": lane["namespace_label"],
        "output_path": lane["output_path"],
        "record_id": lane["record_id"],
        "retry_replay_resume_repair_permitted": False,
        "schema_id": READY_SCHEMA_ID,
        "shell_argv": lane["shell_argv"],
        "state": READY_STATE,
        "terminal_path": relative_path_text(paths["terminal"], LIVE_SURFACE_ROOT),
    }


def validate_ready_record(
    record: dict[str, Any],
    contract: dict[str, Any],
    lane_label: str,
) -> None:
    expected = make_ready_record(
        contract,
        lane_label,
        record.get("authority_credential_sha256", ""),
    )
    require(record == expected, f"{lane_label} READY record mismatch")


def validate_transitioned_ledger(
    record: dict[str, Any],
    contract: dict[str, Any],
    lane_label: str,
    allowed_states: set[str],
) -> None:
    state = record.get("state")
    require(state in allowed_states, f"{lane_label} ledger state")
    timestamp_key = {
        CONSUMED_STATE: "consumed_at_unix_ns",
        INVALIDATED_STATE: "invalidated_at_unix_ns",
    }[state]
    ready = dict(record)
    timestamp = ready.pop(timestamp_key, None)
    ready_record_sha256 = ready.pop("ready_record_sha256", None)
    ready["state"] = READY_STATE
    validate_ready_record(ready, contract, lane_label)
    require(
        is_sha256(ready_record_sha256)
        and ready_record_sha256 == object_sha256(ready),
        f"{lane_label} READY record hash closure",
    )
    require(
        isinstance(timestamp, int) and not isinstance(timestamp, bool) and timestamp > 0,
        f"{lane_label} transition timestamp",
    )


def ensure_output_absence(contract: dict[str, Any]) -> None:
    validate_runtime_roots()
    require(not lexical_exists(OUTPUT_ROOT), "declared output root collision")
    verify_no_symlink_components(OUTPUT_ROOT)
    for lane_label in LANES:
        output = ROOT / contract["lanes"][lane_label]["output_path"]
        expected = ROOT / LANES[lane_label]["output_path"]
        require_canonical_absolute_path(output, expected, f"{lane_label} output")
        require(not lexical_exists(output), f"{lane_label} output collision")
        verify_no_symlink_components(output)


def validate_executable(contract: dict[str, Any], lane_label: str) -> None:
    lane = contract["lanes"][lane_label]
    executable = Path(lane["shell_argv"][0])
    verify_plain_regular_file(executable)
    require(os.access(executable, os.X_OK), f"{lane_label} executable not executable")
    require(
        sha256_file(executable) == lane["executable_sha256"],
        f"{lane_label} executable hash mismatch",
    )


def materialize_lane(
    lane_label: str,
    authority_credential_sha256: str,
) -> Path:
    """Create exactly one READY_UNCONSUMED record for one isolated lane."""

    contract = load_authorized_contract()
    validate_bound_artifacts()
    ensure_output_absence(contract)
    validate_executable(contract, lane_label)
    surface_root = LIVE_SURFACE_ROOT
    paths = lane_paths(surface_root, lane_label)
    if lexical_exists(surface_root):
        verify_plain_directory(surface_root)
    else:
        verify_plain_directory(surface_root.parent)
        surface_root.mkdir(mode=0o700, parents=False)
        verify_plain_directory(surface_root)
        fsync_directory(surface_root.parent)
    lanes_root = surface_root / "lanes"
    if lexical_exists(lanes_root):
        verify_plain_directory(lanes_root)
    else:
        lanes_root.mkdir(mode=0o700)
        verify_plain_directory(lanes_root)
        fsync_directory(surface_root)
    require(not lexical_exists(paths["lane_root"]), f"{lane_label} lane collision")
    paths["lane_root"].mkdir(mode=0o700)
    verify_plain_directory(paths["lane_root"])
    fsync_directory(lanes_root)
    lock_descriptor = os.open(
        paths["lock"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    os.fsync(lock_descriptor)
    os.close(lock_descriptor)
    fsync_directory(paths["lane_root"])
    ready = make_ready_record(contract, lane_label, authority_credential_sha256)
    exclusive_write_json(paths["ledger"], ready)
    require(not lexical_exists(paths["terminal"]), f"{lane_label} terminal collision")
    return paths["ledger"]


def validate_terminal_record(
    record: dict[str, Any],
    ready: dict[str, Any],
    status: str | None = None,
) -> None:
    require(set(record) == {
        "authority_package_sha256",
        "command_sha256",
        "consumption_committed",
        "execution_started",
        "first_record_immutable",
        "invocation_sha256",
        "lane_label",
        "reason_code",
        "record_id",
        "retry_replay_resume_repair_permitted",
        "schema_id",
        "status",
    }, "terminal keys")
    require(record["schema_id"] == TERMINAL_SCHEMA_ID, "terminal schema")
    require(record["status"] in TERMINAL_STATES, "terminal status")
    if status is not None:
        require(record["status"] == status, "terminal expected status")
    for key in (
        "authority_package_sha256",
        "command_sha256",
        "invocation_sha256",
        "lane_label",
        "record_id",
    ):
        require(record[key] == ready[key], f"terminal {key}")
    require(record["first_record_immutable"] is True, "terminal immutability")
    require(
        record["retry_replay_resume_repair_permitted"] is False,
        "terminal retry policy",
    )
    require(isinstance(record["consumption_committed"], bool), "terminal consume flag")
    require(isinstance(record["execution_started"], bool), "terminal execution flag")
    require(
        isinstance(record["reason_code"], str) and record["reason_code"],
        "terminal reason",
    )


def terminal_record(
    ready: dict[str, Any],
    status: str,
    reason_code: str,
    *,
    consumption_committed: bool,
    execution_started: bool,
) -> dict[str, Any]:
    require(status in TERMINAL_STATES, "invalid terminal status")
    record = {
        "authority_package_sha256": ready["authority_package_sha256"],
        "command_sha256": ready["command_sha256"],
        "consumption_committed": consumption_committed,
        "execution_started": execution_started,
        "first_record_immutable": True,
        "invocation_sha256": ready["invocation_sha256"],
        "lane_label": ready["lane_label"],
        "reason_code": reason_code,
        "record_id": ready["record_id"],
        "retry_replay_resume_repair_permitted": False,
        "schema_id": TERMINAL_SCHEMA_ID,
        "status": status,
    }
    validate_terminal_record(record, ready, status)
    return record


def seal_first_terminal(path: Path, record: dict[str, Any], ready: dict[str, Any]) -> None:
    validate_terminal_record(record, ready)
    exclusive_write_json(path, record)


def verify_prior_lane_terminal(contract: dict[str, Any]) -> None:
    base_paths = lane_paths(LIVE_SURFACE_ROOT, "Base")
    verify_plain_file(base_paths["ledger"])
    base_ledger = load_canonical_json(base_paths["ledger"])
    validate_transitioned_ledger(
        base_ledger,
        contract,
        "Base",
        {CONSUMED_STATE, INVALIDATED_STATE},
    )
    verify_plain_file(base_paths["terminal"])
    base_terminal = load_canonical_json(base_paths["terminal"])
    validate_terminal_record(base_terminal, base_ledger)
    require(
        base_terminal["consumption_committed"]
        is (base_ledger["state"] == CONSUMED_STATE),
        "Base terminal consumption closure",
    )
    if base_ledger["state"] == INVALIDATED_STATE:
        require(base_terminal["status"] == "FAILED_TERMINAL", "Base invalidation status")
        require(base_terminal["execution_started"] is False, "Base invalidation execution")
    if base_terminal["status"] == "SUCCEEDED_TERMINAL":
        require(base_terminal["execution_started"] is True, "Base success execution")


def launch_exact_command(committed: dict[str, Any]) -> subprocess.CompletedProcess[Any]:
    lane_label = committed["lane_label"]
    require(tuple(committed["shell_argv"]), f"{lane_label} empty launch argv")
    require(committed["environment"] == ENVIRONMENT, f"{lane_label} launch environment")
    return subprocess.run(
        tuple(committed["shell_argv"]),
        check=False,
        close_fds=True,
        cwd=ROOT,
        env=dict(ENVIRONMENT),
        shell=False,
    )


def seal_post_consume_failure(
    paths: dict[str, Path],
    committed: dict[str, Any],
    reason_code: str,
    *,
    execution_started: bool,
) -> None:
    terminal = terminal_record(
        committed,
        "FAILED_TERMINAL",
        reason_code,
        consumption_committed=True,
        execution_started=execution_started,
    )
    seal_first_terminal(paths["terminal"], terminal, committed)


def consume_and_run(lane_label: str) -> dict[str, Any]:
    """Consume one lane durably, launch exact argv once, and seal first terminal.

    A process or machine crash after the ledger replacement can leave CONSUMED
    without a terminal record.  That is an immutable orphaned-consumption outcome:
    this function rejects it on every later call and never resumes or retries it.
    """

    contract = load_authorized_contract()
    validate_bound_artifacts()
    ensure_output_absence(contract)
    validate_executable(contract, lane_label)
    surface_root = LIVE_SURFACE_ROOT
    paths = lane_paths(surface_root, lane_label)
    verify_plain_directory(surface_root)
    verify_plain_directory(surface_root / "lanes")
    verify_plain_directory(paths["lane_root"])
    verify_plain_file(paths["lock"], 0o600)
    lock_descriptor = os.open(
        paths["lock"], os.O_RDWR | os.O_NOFOLLOW
    )
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    try:
        require(not lexical_exists(paths["terminal"]), f"{lane_label} terminal collision")
        verify_plain_file(paths["ledger"])
        ready = load_canonical_json(paths["ledger"])
        validate_ready_record(ready, contract, lane_label)
        if lane_label == "checkpoint-176":
            verify_prior_lane_terminal(contract)
        consumed = dict(ready)
        consumed["ready_record_sha256"] = object_sha256(ready)
        consumed["state"] = CONSUMED_STATE
        consumed["consumed_at_unix_ns"] = time.time_ns()
        atomic_replace_json(paths["ledger"], consumed)
        committed = load_canonical_json(paths["ledger"])
        require(committed == consumed, "durable consumed readback mismatch")
        validate_transitioned_ledger(
            committed,
            contract,
            lane_label,
            {CONSUMED_STATE},
        )
        try:
            reloaded_contract = load_authorized_contract()
            require(reloaded_contract == contract, "launch authority changed after consume")
            validate_bound_artifacts()
            ensure_output_absence(contract)
            validate_executable(contract, lane_label)
            if lane_label == "checkpoint-176":
                verify_prior_lane_terminal(contract)
            durable = load_canonical_json(paths["ledger"])
            validate_transitioned_ledger(
                durable,
                contract,
                lane_label,
                {CONSUMED_STATE},
            )
            require(durable == committed, "committed ledger changed before launch")
            require(
                not lexical_exists(paths["terminal"]),
                f"{lane_label} terminal appeared before launch",
            )
        except BaseException:
            seal_post_consume_failure(
                paths,
                committed,
                "POST_CONSUME_IDENTITY_CLOSURE_FAILED",
                execution_started=False,
            )
            raise
        try:
            completed = launch_exact_command(committed)
            require(
                isinstance(completed.returncode, int)
                and not isinstance(completed.returncode, bool),
                "launcher return code",
            )
        except BaseException as exc:
            seal_post_consume_failure(
                paths,
                committed,
                "LAUNCH_EXCEPTION_" + type(exc).__name__.upper(),
                execution_started=True,
            )
            raise
        succeeded = completed.returncode == 0
        status = "SUCCEEDED_TERMINAL" if succeeded else "FAILED_TERMINAL"
        reason_code = (
            "PROCESS_EXIT_0"
            if succeeded
            else f"PROCESS_EXIT_NONZERO_{completed.returncode}"
        )
        terminal = terminal_record(
            committed,
            status,
            reason_code,
            consumption_committed=True,
            execution_started=True,
        )
        seal_first_terminal(paths["terminal"], terminal, committed)
        return terminal
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def invalidate_preflight_failure(
    lane_label: str,
    reason_code: str,
) -> dict[str, Any]:
    """Irreversibly invalidate a canonical READY record without running a lane."""

    contract = load_authorized_contract()
    require(isinstance(reason_code, str) and reason_code, "invalid reason code")
    surface_root = LIVE_SURFACE_ROOT
    paths = lane_paths(surface_root, lane_label)
    verify_plain_directory(paths["lane_root"])
    verify_plain_file(paths["lock"], 0o600)
    lock_descriptor = os.open(paths["lock"], os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    try:
        require(not lexical_exists(paths["terminal"]), f"{lane_label} terminal collision")
        ready = load_canonical_json(paths["ledger"])
        validate_ready_record(ready, contract, lane_label)
        invalidated = dict(ready)
        invalidated["ready_record_sha256"] = object_sha256(ready)
        invalidated["state"] = INVALIDATED_STATE
        invalidated["invalidated_at_unix_ns"] = time.time_ns()
        atomic_replace_json(paths["ledger"], invalidated)
        durable = load_canonical_json(paths["ledger"])
        require(durable == invalidated, "durable invalidation readback mismatch")
        validate_transitioned_ledger(
            durable,
            contract,
            lane_label,
            {INVALIDATED_STATE},
        )
        terminal = terminal_record(
            durable,
            "FAILED_TERMINAL",
            reason_code,
            consumption_committed=False,
            execution_started=False,
        )
        seal_first_terminal(paths["terminal"], terminal, durable)
        return terminal
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def main() -> int:
    fail_live(
        "the frozen accepted authority package has shell_argv_materialized=false "
        "and includes no executable evaluator; live materialization and execution "
        "remain disabled until higher authority supplies exact checksum-bound argv "
        "and executable identity"
    )


if __name__ == "__main__":
    raise SystemExit(main())
