#!/home/argustest/miniconda3/bin/python3.13
"""Non-consuming Docker namespace preflight for the accepted V21 invocation."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ROOT = PROJECT_ROOT / (
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_"
    "order_independent_binding_terminal_coverage_action_root"
)
TOOLS = ACTION_ROOT / "tools"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
TRANSPORT = TOOLS / "qk_gbfp8_head64_granularity_sweep_transport_wrapper_v21.py"
LAUNCHER = TOOLS / "qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21.py"
PACKAGE = ACTION_ROOT / (
    "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_"
    "ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json"
)
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
RUNTIME_PARENT = PROJECT_ROOT / "runtime"
RUNTIME_ROOT = RUNTIME_PARENT / (
    "qk_gbfp8_head64_granularity_sweep_execution_v21_"
    "order_independent_binding_terminal_coverage_289140ba"
)
ACTION_ID = "ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001"
EXPECTED_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
EXPECTED_TRANSPORT_ARGV = [
    str(INTERPRETER),
    str(TRANSPORT),
    "--package",
    str(PACKAGE),
    "--acceptance",
    str(ACCEPTANCE),
    "--irreversible-action-id",
    ACTION_ID,
]
EXPECTED_INTERPRETER_SHA256 = (
    "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
)


class PreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


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


def sha256_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(descriptor)


def decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def mount_record(path: Path) -> dict[str, Any]:
    target = str(path.resolve())
    matches: list[dict[str, Any]] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="ascii").splitlines():
        before, after = line.split(" - ", 1)
        fields = before.split()
        mount_point = decode_mount_path(fields[4])
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            post = after.split()
            matches.append(
                {
                    "filesystem": post[0],
                    "mount_options": fields[5].split(","),
                    "mount_point": mount_point,
                    "source": post[1],
                    "super_options": post[2].split(","),
                }
            )
    require(bool(matches), f"no mount record for {path}")
    return max(matches, key=lambda item: len(item["mount_point"]))


def require_read_only_directory(path: Path) -> dict[str, Any]:
    record = mount_record(path)
    require("ro" in record["mount_options"], f"mount is not read-only: {path}")
    probe = path / f".v21-docker-forbidden-write-{os.getpid()}"
    try:
        descriptor = os.open(
            probe,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as error:
        require(error.errno == errno.EROFS, f"write denial was not EROFS: {path}: {error}")
        record["write_probe"] = "DENIED_EROFS_WITHOUT_CREATION"
    else:
        os.close(descriptor)
        probe.unlink(missing_ok=True)
        raise PreflightError(f"read-only write probe unexpectedly succeeded: {path}")
    return record


def require_runtime_parent_writable_without_namespace() -> dict[str, Any]:
    require(not os.path.lexists(RUNTIME_ROOT), "V21 runtime root exists before preflight")
    record = mount_record(RUNTIME_PARENT)
    require("rw" in record["mount_options"], "runtime parent mount is not writable")
    flags = os.O_WRONLY | getattr(os, "O_TMPFILE", 0)
    require(getattr(os, "O_TMPFILE", 0) != 0, "O_TMPFILE unavailable")
    descriptor = os.open(RUNTIME_PARENT, flags, 0o600)
    try:
        os.write(descriptor, b"v21-docker-preflight\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(not os.path.lexists(RUNTIME_ROOT), "V21 runtime root appeared during preflight")
    record["write_probe"] = "PASS_UNNAMED_O_TMPFILE_NO_NAMESPACE_ENTRY"
    return record


def regular_identity(path: Path) -> dict[str, Any]:
    info = os.lstat(path)
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"not regular: {path}")
    return {
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "path": str(path),
        "sha256": sha256_file(path),
        "size": info.st_size,
    }


def main() -> int:
    require(Path.cwd() == ACTION_ROOT, "preflight cwd")
    require(dict(os.environ) == EXPECTED_ENVIRONMENT, "preflight environment")
    require(Path(sys.executable).resolve() == INTERPRETER, "preflight interpreter path")
    require(sha256_file(INTERPRETER) == EXPECTED_INTERPRETER_SHA256, "interpreter hash")

    sys.path.insert(0, str(TOOLS))
    import qk_gbfp8_head64_granularity_sweep_transport_wrapper_v21 as transport
    import qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21 as launcher

    require(transport.EXACT_ARGV == [
        str(INTERPRETER),
        str(LAUNCHER),
        "--package",
        str(PACKAGE),
        "--acceptance",
        str(ACCEPTANCE),
        "--irreversible-action-id",
        ACTION_ID,
    ], "transport launcher argv contract")
    require(transport.EXACT_ENVIRONMENT == EXPECTED_ENVIRONMENT, "transport environment contract")
    require(transport.ROOT == ACTION_ROOT, "transport cwd contract")
    require(transport.expected_transport_arguments() == EXPECTED_TRANSPORT_ARGV[2:], "transport argv contract")

    source_mount = require_read_only_directory(ACTION_ROOT)
    interpreter_mount = require_read_only_directory(INTERPRETER.parent)
    runtime_mount = require_runtime_parent_writable_without_namespace()

    identities = {
        "acceptance": regular_identity(ACCEPTANCE),
        "interpreter": regular_identity(INTERPRETER),
        "launcher": regular_identity(LAUNCHER),
        "package": regular_identity(PACKAGE),
        "transport": regular_identity(TRANSPORT),
    }

    validators = launcher.load_schemas()
    launcher_args = launcher.parse_exact_arguments(transport.EXACT_ARGV[2:])
    original_attestation = launcher.transport_attestation
    launcher.transport_attestation = lambda package: {"preflight_stub": True}
    try:
        holder = launcher.read_only_preflight(
            launcher_args,
            validators,
            launcher.PATHS,
            observed_cwd=ACTION_ROOT,
            observed_environment=EXPECTED_ENVIRONMENT,
        )
    finally:
        launcher.transport_attestation = original_attestation
    launcher.prepare_execution(holder, launcher.load_module)
    require(holder["numerical_tensor_names"] == launcher.EXPECTED_SELECTED_NAMES, "production selection")
    require(len(holder["tensor_bindings"]) == 25, "production binding count")
    require(not os.path.lexists(RUNTIME_ROOT), "V21 runtime root exists after preflight")

    record: dict[str, Any] = {
        "artifact_kind": "ace2_v21_docker_immutable_namespace_preflight",
        "claim_boundary": (
            "NON_CONSUMING_DOCKER_PREFLIGHT_ONLY: no V21 transport, launcher main, authority, "
            "credential, ledger, payload read, evaluator, result, or terminal invocation occurred"
        ),
        "docker_process_contract": {
            "accepted_transport_argv": EXPECTED_TRANSPORT_ARGV,
            "cwd": str(ACTION_ROOT),
            "environment": EXPECTED_ENVIRONMENT,
            "shell": False,
        },
        "identities": identities,
        "mounts": {
            "accepted_sources": source_mount,
            "interpreter": interpreter_mount,
            "runtime_parent": runtime_mount,
        },
        "preowner_checks": {
            "complete_launcher_read_only_preflight": True,
            "production_binding_count": len(holder["tensor_bindings"]),
            "production_module_import_and_prepare_execution": True,
            "selected_tensor_names": holder["numerical_tensor_names"],
        },
        "runtime_namespace_absent_after_preflight": not os.path.lexists(RUNTIME_ROOT),
        "schema_version": 1,
        "status": "PASS_V21_DOCKER_IMMUTABLE_NAMESPACE_PREFLIGHT",
    }
    record["record_sha256"] = hashlib.sha256(compact_bytes(record)).hexdigest()
    sys.stdout.buffer.write(compact_bytes(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
