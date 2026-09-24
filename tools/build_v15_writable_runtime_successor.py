#!/usr/bin/env python3
"""Build the additive V15 static package with an external writable runtime.

The builder never invokes the transport, launcher, controller, evaluator, or
sealed tensor path.  It creates only immutable package files and an empty,
owner-only runtime directory tree with no lifecycle record files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
V13_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_action_root"
V14_FAILED_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v14_shellfree_action_root"
V14_FAILED_PACKAGE = V14_FAILED_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V14_SHELLFREE_PACKAGE.json"
V14_FAILURE_RECORD = PROJECT_ROOT / "build/v14-writable-runtime-pre-review-candidate-0001-failed.json"
V15_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_action_root"
RUNTIME_CONTAINER = PROJECT_ROOT / "runtime"
RUNTIME_ROOT = RUNTIME_CONTAINER / "qk_gbfp8_head64_granularity_sweep_execution_v15_c2dfe170"
PRIMARY_ROOT = RUNTIME_ROOT / "primary"
FALLBACK_ROOT = RUNTIME_ROOT / "fallback"
V8_STATIC_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root"
V9_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root"
V8_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_action_root"
V7_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root"
V6_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root"
VERIFIER_SOURCE = PROJECT_ROOT / "tools/v15_writable_runtime_verify_independent.py"
V13_FAILURE_RECORD = PROJECT_ROOT / ".autors/ace-2/wiki/sources/runs/531ffdcbd930-r002.md"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")

ACTION_ID = "ace2:qk-gbfp8-base-v15:execute-once:c2dfe170:20260814T084500Z"
V13_ACTION_ID = "ace2:qk-gbfp8-base-v13:execute-once:81a8edc1:20260814T075838Z"
V9_ACTION_ID = "ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z"
OLD_ACTION_IDS = [
    "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1610Z",
    "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z",
    "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z",
    V9_ACTION_ID,
]
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXPECTED_UID = os.getuid()
EXPECTED_GID = os.getgid()


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_object(raw: bytes) -> dict[str, Any]:
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    value = json.loads(raw.decode("ascii", "strict"), object_pairs_hook=pairs)
    require(type(value) is dict and not duplicate, "JSON object shape")
    return value


def write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def transform_text(text: str) -> str:
    return (
        text.replace(V13_ACTION_ID, ACTION_ID)
        .replace(str(V13_ROOT), str(V15_ROOT))
        .replace("V13", "V15")
        .replace("v13", "v15")
    )


def transform_value(value: Any) -> Any:
    if isinstance(value, str):
        return transform_text(value)
    if isinstance(value, list):
        return [transform_value(item) for item in value]
    if isinstance(value, dict):
        return {key: transform_value(item) for key, item in value.items()}
    return value


def invocation(argv: list[str]) -> dict[str, Any]:
    base = {
        "argv": argv,
        "command_representation": "ARGV_VECTOR_ONLY",
        "cwd": str(V15_ROOT),
        "environment": EXACT_ENVIRONMENT,
        "shell": False,
    }
    return base | {"invocation_sha256": sha256_bytes(compact_bytes(base))}


def binding(binding_id: str, path: Path) -> dict[str, Any]:
    return {"id": binding_id, "path": str(path), "sha256": sha256_file(path)}


def runtime_directories() -> list[Path]:
    return [
        RUNTIME_CONTAINER,
        RUNTIME_ROOT,
        PRIMARY_ROOT,
        PRIMARY_ROOT / "authority",
        PRIMARY_ROOT / "authority/base",
        PRIMARY_ROOT / "result",
        PRIMARY_ROOT / "result/base",
        FALLBACK_ROOT,
    ]


def runtime_files() -> dict[str, Path]:
    return {
        "authority": PRIMARY_ROOT / "authority/base/authority.json",
        "credential": PRIMARY_ROOT / "authority/base/credential.json",
        "ledger": PRIMARY_ROOT / "authority/base/authority-ledger.json",
        "result": PRIMARY_ROOT / "result/base/result.json",
        "first_terminal": PRIMARY_ROOT / "authority/base/first-terminal.json",
        "fallback_terminal": FALLBACK_ROOT / "first-terminal.json",
    }


def prepare_runtime_tree(staging: Path) -> None:
    if not RUNTIME_CONTAINER.exists():
        os.mkdir(RUNTIME_CONTAINER, 0o700)
    info = os.lstat(RUNTIME_CONTAINER)
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), "runtime container type")
    require(stat.S_IMODE(info.st_mode) == 0o700, "runtime container mode")
    require(info.st_uid == EXPECTED_UID and info.st_gid == EXPECTED_GID, "runtime container owner")
    os.mkdir(staging, 0o700)
    for relative in ("primary", "primary/authority", "primary/authority/base", "primary/result", "primary/result/base", "fallback"):
        os.mkdir(staging / relative, 0o700)


def canonical_paths(stage_root: Path) -> tuple[Path, Path]:
    return (
        stage_root / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json",
        stage_root / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json",
    )


def render_launcher(
    stage_root: Path,
    expected_paths: list[tuple[str, Path]],
    hash_sources: dict[str, Path],
    report_path: Path,
) -> bytes:
    source_path = V13_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v13.py"
    launcher = transform_text(source_path.read_text(encoding="utf-8"))
    root_line = f'ROOT = Path("{V15_ROOT}")\n'
    require(root_line in launcher, "launcher root marker")
    launcher = launcher.replace(
        root_line,
        root_line
        + f'RUNTIME_CONTAINER = Path("{RUNTIME_CONTAINER}")\n'
        + f'RUNTIME_ROOT = Path("{RUNTIME_ROOT}")\n'
        + f'EXPECTED_RUNTIME_UID = {EXPECTED_UID}\n'
        + f'EXPECTED_RUNTIME_GID = {EXPECTED_GID}\n',
        1,
    )
    live_block = f'''LIVE_ROOT = RUNTIME_ROOT / "primary"
FALLBACK_ROOT = RUNTIME_ROOT / "fallback"
FALLBACK_TERMINAL = FALLBACK_ROOT / "first-terminal.json"
RUNTIME_DIRECTORIES = (
    RUNTIME_CONTAINER,
    RUNTIME_ROOT,
    LIVE_ROOT,
    LIVE_ROOT / "authority",
    LIVE_ROOT / "authority/base",
    LIVE_ROOT / "result",
    LIVE_ROOT / "result/base",
    FALLBACK_ROOT,
)
LIVE_PATHS = {{
    "authority": LIVE_ROOT / "authority/base/authority.json",
    "credential": LIVE_ROOT / "authority/base/credential.json",
    "ledger": LIVE_ROOT / "authority/base/authority-ledger.json",
    "result": LIVE_ROOT / "result/base/result.json",
    "first_terminal": LIVE_ROOT / "authority/base/first-terminal.json",
}}
'''
    launcher = replace_block(launcher, "LIVE_ROOT = ROOT / \"live\"", "SEALED_TENSOR =", live_block)

    accepted_lines = ["ACCEPTED_PATHS = {"]
    for binding_id, path in expected_paths:
        accepted_lines.append(f'    "{binding_id}": Path("{path}"),')
    accepted_lines.append("}\n")
    launcher = replace_block(launcher, "ACCEPTED_PATHS = {", "EXPECTED_BINDING_HASHES = {", "\n".join(accepted_lines))
    hashes = "EXPECTED_BINDING_HASHES = {\n" + "".join(
        f'    "{binding_id}": "{sha256_file(hash_sources[binding_id])}",\n' for binding_id, _ in expected_paths
    ) + "}\n"
    launcher = replace_block(launcher, "EXPECTED_BINDING_HASHES = {", "EXPECTED_BINDING_ORDER =", hashes)
    launcher = launcher.replace(
        '    "forensic_report": ROOT / "V9_CANONICAL_BYTE_FAILURE_REPORT.md",',
        '    "forensic_report": ROOT / "V13_RUNTIME_NAMESPACE_FAILURE_REPORT.md",',
    )

    old_identity = '''    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_terminal_action_id": V9_ACTION_ID,
        "retired_action_ids": [RETIRED_PREPACKAGE_ACTION_ID, V7_ACTION_ID, V8_ACTION_ID, V9_ACTION_ID],
        "reuse_permitted": False,
    }, "action identity")'''
    new_identity = '''    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_attempt_action_id": V13_ACTION_ID,
        "prior_terminal_action_id": V9_ACTION_ID,
        "retired_action_ids": [RETIRED_PREPACKAGE_ACTION_ID, V7_ACTION_ID, V8_ACTION_ID, V9_ACTION_ID, V13_ACTION_ID],
        "reuse_permitted": False,
    }, "action identity")'''
    require(old_identity in launcher, "launcher action identity block")
    launcher = launcher.replace(old_identity, new_identity)
    action_line = f'ACTION_ID = "{ACTION_ID}"\n'
    require(action_line in launcher, "launcher action line")
    launcher = launcher.replace(action_line, action_line + f'V13_ACTION_ID = "{V13_ACTION_ID}"\n', 1)

    prior_marker = '''    require(package["prior_terminal_state"] == {
        "action_id": V9_ACTION_ID,
        "invocation_count_performed": 0,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "reason_code": "READ_ONLY_PREFLIGHT_FAILED",
        "result_file_sha256": None,
        "status": "PREFLIGHT_FAILED_TERMINAL",
    }, "prior terminal declaration")'''
    prior_extension = prior_marker + '''
    require(package["prior_attempt_state"] == {
        "action_id": V13_ACTION_ID,
        "evaluator_invocation_count": 0,
        "exit_status": 2,
        "first_terminal_materialized": False,
        "invocation_count_performed": 1,
        "launcher_contract_matched": False,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "reason_code": "IMMUTABLE_ROOT_RUNTIME_NAMESPACE_UNWRITABLE",
        "shell_wrapper_observed": "/bin/bash -c",
    }, "prior V13 attempt declaration")
    runtime = package["runtime_namespace"]
    require(runtime["runtime_root"] == str(RUNTIME_ROOT), "runtime root binding")
    require(runtime["primary_root"] == str(LIVE_ROOT), "primary root binding")
    require(runtime["fallback_terminal"] == str(FALLBACK_TERMINAL), "fallback terminal binding")
    require(runtime["precreated_directories"] == [str(path) for path in RUNTIME_DIRECTORIES], "runtime directory binding")
    require(runtime["owner_uid"] == EXPECTED_RUNTIME_UID and runtime["owner_gid"] == EXPECTED_RUNTIME_GID, "runtime owner binding")
    require(runtime["directory_mode_octal"] == "0700" and runtime["file_mode_octal"] == "0400", "runtime mode binding")
    require(runtime["paths"] == {key: str(path) for key, path in LIVE_PATHS.items()}, "runtime lifecycle path binding")
    require(runtime["terminal_publication_order"] == ["primary", "fallback"], "terminal publication order")'''
    require(prior_marker in launcher, "launcher prior marker")
    launcher = launcher.replace(prior_marker, prior_extension)

    ensure_block = '''def _require_runtime_directory(path: Path) -> None:
    observed = os.lstat(path)
    require(stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode), f"unsafe runtime directory: {path}")
    require(stat.S_IMODE(observed.st_mode) == 0o700, f"runtime directory mode: {path}")
    require(observed.st_uid == EXPECTED_RUNTIME_UID and observed.st_gid == EXPECTED_RUNTIME_GID, f"runtime directory owner: {path}")


def _ensure_directory(path: Path) -> None:
    require(path in RUNTIME_DIRECTORIES, f"unbound runtime directory: {path}")
    _require_runtime_directory(path)


def _probe_create_once(parent: Path, label: str) -> None:
    probe = parent / f".v15-production-{label}-create-once-probe"
    require(not os.path.lexists(probe), f"stale runtime probe: {probe}")
    _durable_create(probe, f"{ACTION_ID}:{label}\\n".encode("ascii"))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    duplicate_rejected = False
    try:
        duplicate = os.open(probe, flags, 0o400)
    except FileExistsError:
        duplicate_rejected = True
    else:
        os.close(duplicate)
    require(duplicate_rejected, f"create-once probe accepted duplicate: {label}")
    _durable_unlink(probe)


def _verify_runtime_namespace_pre_authority() -> None:
    for directory in RUNTIME_DIRECTORIES:
        _require_runtime_directory(directory)
    for path in (*LIVE_PATHS.values(), FALLBACK_TERMINAL):
        require(not os.path.lexists(path), f"pre-authority runtime file exists: {path}")
    _probe_create_once(LIVE_PATHS["first_terminal"].parent, "primary-terminal")
    _probe_create_once(FALLBACK_TERMINAL.parent, "fallback-terminal")
    for path in (*LIVE_PATHS.values(), FALLBACK_TERMINAL):
        require(not os.path.lexists(path), f"runtime probe left lifecycle state: {path}")


'''
    launcher = replace_block(launcher, "def _ensure_directory(path: Path) -> None:", "def _durable_create", ensure_block)
    old_live_check = '''def _require_live_namespace_absent() -> None:
    require(not os.path.lexists(LIVE_ROOT), "new live namespace already exists")
'''
    require(old_live_check in launcher, "launcher live check")
    launcher = launcher.replace(old_live_check, "")
    launcher = launcher.replace("    _require_live_namespace_absent()\n", "    _verify_runtime_namespace_pre_authority()\n")

    publish_block = '''def _publish_terminal(record: dict[str, Any]) -> None:
    raw = compact_bytes(record)
    try:
        _durable_create(LIVE_PATHS["first_terminal"], raw)
        return
    except Exception as primary_error:
        try:
            _durable_create(FALLBACK_TERMINAL, raw)
            return
        except Exception as fallback_error:
            raise ExecutionError(
                f"terminal publication failed: primary={type(primary_error).__name__}:{primary_error}; "
                f"fallback={type(fallback_error).__name__}:{fallback_error}"
            ) from fallback_error


'''
    launcher = replace_block(launcher, "def _publish_terminal(record: dict[str, Any]) -> None:", "def _retire_preflight_failure", publish_block)
    require("ROOT / \"live\"" not in launcher, "launcher retains package-local live path")
    require(str(report_path) == str(V15_ROOT / "V13_RUNTIME_NAMESPACE_FAILURE_REPORT.md"), "report final path")
    return launcher.encode("utf-8")


def render_transport() -> bytes:
    source_path = V13_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v13.py"
    transport = transform_text(source_path.read_text(encoding="utf-8"))
    first_line = f'FIRST_TERMINAL = ROOT / "live/authority/base/first-terminal.json"\n'
    replacement = (
        f'RUNTIME_CONTAINER = Path("{RUNTIME_CONTAINER}")\n'
        f'RUNTIME_ROOT = Path("{RUNTIME_ROOT}")\n'
        'PRIMARY_ROOT = RUNTIME_ROOT / "primary"\n'
        'FALLBACK_ROOT = RUNTIME_ROOT / "fallback"\n'
        'FIRST_TERMINAL = PRIMARY_ROOT / "authority/base/first-terminal.json"\n'
        'FALLBACK_TERMINAL = FALLBACK_ROOT / "first-terminal.json"\n'
        f'EXPECTED_RUNTIME_UID = {EXPECTED_UID}\n'
        f'EXPECTED_RUNTIME_GID = {EXPECTED_GID}\n'
        'RUNTIME_DIRECTORIES = (RUNTIME_CONTAINER, RUNTIME_ROOT, PRIMARY_ROOT, PRIMARY_ROOT / "authority", PRIMARY_ROOT / "authority/base", PRIMARY_ROOT / "result", PRIMARY_ROOT / "result/base", FALLBACK_ROOT)\n'
    )
    require(first_line in transport, "transport terminal marker")
    transport = transport.replace(first_line, replacement, 1)
    ensure_block = '''def _require_runtime_directory(path: Path) -> None:
    observed = os.lstat(path)
    require(stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode), f"unsafe runtime directory: {path}")
    require(stat.S_IMODE(observed.st_mode) == 0o700, f"runtime directory mode: {path}")
    require(observed.st_uid == EXPECTED_RUNTIME_UID and observed.st_gid == EXPECTED_RUNTIME_GID, f"runtime directory owner: {path}")


def _ensure_directory(path: Path) -> None:
    require(path in RUNTIME_DIRECTORIES, f"unbound runtime directory: {path}")
    _require_runtime_directory(path)


def _probe_create_once(parent: Path, label: str) -> None:
    probe = parent / f".v15-transport-{label}-create-once-probe"
    require(not os.path.lexists(probe), f"stale runtime probe: {probe}")
    _durable_create(probe, f"{ACTION_ID}:{label}\\n".encode("ascii"))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    duplicate_rejected = False
    try:
        duplicate = os.open(probe, flags, 0o400)
    except FileExistsError:
        duplicate_rejected = True
    else:
        os.close(duplicate)
    require(duplicate_rejected, f"create-once probe accepted duplicate: {label}")
    os.unlink(probe)
    _fsync_directory(parent)


def _verify_runtime_namespace_pre_authority() -> None:
    for directory in RUNTIME_DIRECTORIES:
        _require_runtime_directory(directory)
    for path in (FIRST_TERMINAL, FALLBACK_TERMINAL, PRIMARY_ROOT / "authority/base/authority.json", PRIMARY_ROOT / "authority/base/credential.json", PRIMARY_ROOT / "authority/base/authority-ledger.json", PRIMARY_ROOT / "result/base/result.json"):
        require(not os.path.lexists(path), f"pre-authority runtime file exists: {path}")
    _probe_create_once(FIRST_TERMINAL.parent, "primary-terminal")
    _probe_create_once(FALLBACK_TERMINAL.parent, "fallback-terminal")


'''
    transport = replace_block(transport, "def _ensure_directory(path: Path) -> None:", "def _durable_create", ensure_block)
    retire_block = '''def _retire_transport_failure(error: BaseException) -> int:
    raw = compact_bytes(_terminal_record(error))
    try:
        _durable_create(FIRST_TERMINAL, raw)
        return 3
    except Exception:
        try:
            _durable_create(FALLBACK_TERMINAL, raw)
            return 3
        except Exception:
            return 4


'''
    transport = replace_block(transport, "def _retire_transport_failure(error: BaseException) -> int:", "def _transport_preflight", retire_block)
    old_absence = '    require(not os.path.lexists(ROOT / "live"), "new action already has live state")\n'
    require(old_absence in transport, "transport live check")
    transport = transport.replace(
        old_absence,
        '    runtime = package["runtime_namespace"]\n'
        '    require(runtime["runtime_root"] == str(RUNTIME_ROOT), "runtime root binding")\n'
        '    require(runtime["primary_root"] == str(PRIMARY_ROOT), "primary root binding")\n'
        '    require(runtime["fallback_terminal"] == str(FALLBACK_TERMINAL), "fallback terminal binding")\n'
        '    _verify_runtime_namespace_pre_authority()\n',
    )
    require("ROOT / \"live\"" not in transport, "transport retains package-local live path")
    return transport.encode("utf-8")


def main() -> int:
    require(not V15_ROOT.exists(), f"refusing to overwrite V15 package: {V15_ROOT}")
    require(not RUNTIME_ROOT.exists(), f"refusing to overwrite V15 runtime: {RUNTIME_ROOT}")
    require(V13_ROOT.is_dir(), "V13 package absent")
    require(VERIFIER_SOURCE.is_file(), "V15 independent verifier source absent")
    require(V13_FAILURE_RECORD.is_file(), "V13 failure record absent")
    require(INTERPRETER.is_file(), "frozen interpreter absent")

    package_stage = V15_ROOT.with_name(f".{V15_ROOT.name}.staging-{os.getpid()}")
    runtime_stage = RUNTIME_CONTAINER / f".{RUNTIME_ROOT.name}.staging-{os.getpid()}"
    require(not package_stage.exists() and not runtime_stage.exists(), "staging path exists")
    package_stage.mkdir(parents=True, mode=0o700)
    prepare_runtime_tree(runtime_stage)
    try:
        canonical_package, canonical_schema = canonical_paths(package_stage)
        source_package = V8_STATIC_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json"
        source_schema = V8_STATIC_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json"
        source_package_raw = source_package.read_bytes()
        source_schema_raw = source_schema.read_bytes()
        canonical_package_raw = compact_bytes(strict_object(source_package_raw))
        canonical_schema_raw = compact_bytes(strict_object(source_schema_raw))
        write_bytes(canonical_package, canonical_package_raw)
        write_bytes(canonical_schema, canonical_schema_raw)

        schema_names = ("AUTHORITY_SCHEMA", "CREDENTIAL_SCHEMA", "FIRST_TERMINAL_SCHEMA", "FRESH_L2_ACCEPTANCE_SCHEMA", "LEDGER_SCHEMA")
        schema_paths: dict[str, Path] = {}
        for name in schema_names:
            source = V13_ROOT / f"reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V13_SHELLFREE_{name}.json"
            target = package_stage / f"reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_{name}.json"
            schema = transform_value(strict_object(source.read_bytes()))
            write_bytes(target, compact_bytes(schema))
            schema_paths[name] = target

        report = f"""# V13 runtime-namespace failure and V15 additive successor

Date: 2026-08-14

The sole V13 action `{V13_ACTION_ID}` was invoked once and returned 2.  The
immutable mode-0555 package root prevented both package-local live-state
creation and fallback first-terminal publication.  The recorded invocation
used `/bin/bash -c`, not the accepted shell-free argv contract.  No immutable
V13 terminal exists, no sealed tensor was opened, and no G8/G4/G2/G1 evaluator
run occurred.  V13 is permanently retired and remains byte-unchanged.

The additive V15 action is `{ACTION_ID}`.  Its package root is immutable and
contains no live directory.  All lifecycle paths are instead bound to the
separately named `{RUNTIME_ROOT}` tree.  The tree is precreated empty with mode
0700 and owner `{EXPECTED_UID}:{EXPECTED_GID}`.  Production preflight and the
independent verifier both check every parent with lstat, reject symlinks or
owner/mode drift, prove O_EXCL create-once behavior in the primary and fallback
terminal parents, remove the probes, and require all lifecycle files absent
before authority publication.

The future invocation is represented only as exact argv, cwd, explicit
environment, `os.posix_spawn`, and `shell=false`.  This static package creates
no execution authority.  Fresh-L2 static acceptance is required before any
separately authorized later execution mission.

The preserved V14 pre-review candidate is not an execution attempt and did not
create acceptance or authority.  Its inert verifier rejected the package after
finding a missing digest cross-link; the V15 verifier adds that cross-link and
binds the V14 failure record without modifying the V14 bytes.
"""
        report_stage = package_stage / "V13_RUNTIME_NAMESPACE_FAILURE_REPORT.md"
        write_bytes(report_stage, report.encode("utf-8"))

        expected_paths = [
            ("accepted_v8_handoff", Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f0fa5c269681/round-0007.json")),
            ("accepted_r2_handoff", Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b8v8execpkg02/round-0003.json")),
            ("accepted_v8_manifest", V8_STATIC_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_MANIFEST.json"),
            ("runtime_package", V8_STATIC_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json"),
            ("package_predecessor_raw", source_package),
            ("result_schema_predecessor_raw", source_schema),
            ("package", V15_ROOT / canonical_package.relative_to(package_stage)),
            ("result_schema", V15_ROOT / canonical_schema.relative_to(package_stage)),
            ("controller", V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py"),
            ("evaluator", V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"),
            ("c02_parser", V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"),
            ("decisive_verifier", V8_STATIC_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v8.py"),
            ("v9_first_terminal", V9_ROOT / "live/authority/base/first-terminal.json"),
            ("v8_authority", V8_ROOT / "live/authority/base/authority.json"),
            ("v8_consumed_ledger", V8_ROOT / "live/authority/base/authority-ledger.json"),
            ("v8_first_terminal", V8_ROOT / "live/authority/base/first-terminal.json"),
            ("v7_consumed_ledger", V7_ROOT / "live/authority/base/authority-ledger.json"),
            ("v7_first_terminal", V7_ROOT / "live/authority/base/first-terminal.json"),
            ("v13_package", V13_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V13_SHELLFREE_PACKAGE.json"),
            ("v13_static_acceptance", V13_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
            ("v13_failure_record", V13_FAILURE_RECORD),
            ("v14_failed_package", V14_FAILED_PACKAGE),
            ("v14_failure_record", V14_FAILURE_RECORD),
        ]
        hash_sources = {binding_id: path for binding_id, path in expected_paths}
        hash_sources["package"] = canonical_package
        hash_sources["result_schema"] = canonical_schema
        for binding_id, path in hash_sources.items():
            require(path.is_file(), f"bound predecessor absent: {binding_id}: {path}")

        launcher_stage = package_stage / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v15.py"
        transport_stage = package_stage / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v15.py"
        verifier_stage = package_stage / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree.py"
        write_bytes(
            launcher_stage,
            render_launcher(
                package_stage,
                expected_paths,
                hash_sources,
                V15_ROOT / report_stage.relative_to(package_stage),
            ),
        )
        write_bytes(transport_stage, render_transport())
        write_bytes(verifier_stage, VERIFIER_SOURCE.read_bytes())

        package_path_stage = package_stage / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_PACKAGE.json"
        old_package = strict_object((V13_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V13_SHELLFREE_PACKAGE.json").read_bytes())
        package = transform_value(old_package)
        package.pop("package_content_sha256", None)
        package["action_identity"] = {
            "future_action_id": ACTION_ID,
            "prior_attempt_action_id": V13_ACTION_ID,
            "prior_terminal_action_id": V9_ACTION_ID,
            "retired_action_ids": [*OLD_ACTION_IDS, V13_ACTION_ID],
            "reuse_permitted": False,
        }
        package["prior_attempt_state"] = {
            "action_id": V13_ACTION_ID,
            "evaluator_invocation_count": 0,
            "exit_status": 2,
            "first_terminal_materialized": False,
            "invocation_count_performed": 1,
            "launcher_contract_matched": False,
            "no_replay_retry_resume_repair_replacement": True,
            "payload_open_count": 0,
            "reason_code": "IMMUTABLE_ROOT_RUNTIME_NAMESPACE_UNWRITABLE",
            "shell_wrapper_observed": "/bin/bash -c",
        }
        package["prior_static_candidate_state"] = {
            "acceptance_materialized": False,
            "action_id": "ace2:qk-gbfp8-base-v14:execute-once:f1eb0abd:20260814T083636Z",
            "authority_materialized": False,
            "classification": "PRE_REVIEW_STATIC_VERIFIER_GAP",
            "failure_record_sha256": sha256_file(V14_FAILURE_RECORD),
            "manifest_raw_sha256": sha256_file(V14_FAILED_PACKAGE),
            "runtime_files_present_after_failure": 0,
            "sealed_tensor_access": "NONE",
            "target_process_starts": 0,
            "transport_or_launcher_invoked": False,
        }
        package["canonical_bindings"] = [
            {"id": binding_id, "path": str(path), "sha256": sha256_file(hash_sources[binding_id])}
            for binding_id, path in expected_paths
        ]
        local_paths = [
            ("authority_schema", V15_ROOT / schema_paths["AUTHORITY_SCHEMA"].relative_to(package_stage)),
            ("credential_schema", V15_ROOT / schema_paths["CREDENTIAL_SCHEMA"].relative_to(package_stage)),
            ("ledger_schema", V15_ROOT / schema_paths["LEDGER_SCHEMA"].relative_to(package_stage)),
            ("first_terminal_schema", V15_ROOT / schema_paths["FIRST_TERMINAL_SCHEMA"].relative_to(package_stage)),
            ("fresh_l2_acceptance_schema", V15_ROOT / schema_paths["FRESH_L2_ACCEPTANCE_SCHEMA"].relative_to(package_stage)),
            ("transport", V15_ROOT / transport_stage.relative_to(package_stage)),
            ("launcher", V15_ROOT / launcher_stage.relative_to(package_stage)),
            ("static_verifier", V15_ROOT / verifier_stage.relative_to(package_stage)),
            ("forensic_report", V15_ROOT / report_stage.relative_to(package_stage)),
        ]
        stage_lookup = {
            V15_ROOT / path.relative_to(package_stage): path
            for path in [*schema_paths.values(), transport_stage, launcher_stage, verifier_stage, report_stage]
        }
        package["local_artifact_bindings"] = [
            {"id": binding_id, "path": str(path), "sha256": sha256_file(stage_lookup[path])}
            for binding_id, path in local_paths
        ]
        package["accepted_byte_derivation"] = {
            "package": {
                "canonical_byte_count": len(canonical_package_raw),
                "canonical_path": str(V15_ROOT / canonical_package.relative_to(package_stage)),
                "canonical_sha256": sha256_bytes(canonical_package_raw),
                "production_reader_context": "accepted V8 package",
                "semantic_json_equal": True,
                "source_byte_count": len(source_package_raw),
                "source_path": str(source_package),
                "source_raw_sha256": sha256_bytes(source_package_raw),
            },
            "result_schema": {
                "canonical_byte_count": len(canonical_schema_raw),
                "canonical_path": str(V15_ROOT / canonical_schema.relative_to(package_stage)),
                "canonical_sha256": sha256_bytes(canonical_schema_raw),
                "production_reader_context": "accepted V8 result schema",
                "semantic_json_equal": True,
                "source_byte_count": len(source_schema_raw),
                "source_path": str(source_schema),
                "source_raw_sha256": sha256_bytes(source_schema_raw),
            },
        }
        transport_final = V15_ROOT / transport_stage.relative_to(package_stage)
        launcher_final = V15_ROOT / launcher_stage.relative_to(package_stage)
        package_final = V15_ROOT / package_path_stage.relative_to(package_stage)
        acceptance_final = V15_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
        transport_argv = [str(INTERPRETER), str(transport_final), "--package", str(package_final), "--acceptance", str(acceptance_final), "--irreversible-action-id", ACTION_ID]
        launcher_argv = [str(INTERPRETER), str(launcher_final), "--package", str(package_final), "--acceptance", str(acceptance_final), "--irreversible-action-id", ACTION_ID]
        package["future_invocation"] = invocation(transport_argv)
        package["launcher_invocation"] = invocation(launcher_argv)
        attestation = {
            "api": "os.posix_spawn",
            "immediate_parent_argv": transport_argv,
            "immediate_parent_environment": EXACT_ENVIRONMENT,
            "immediate_parent_executable_path": str(INTERPRETER),
            "immediate_parent_executable_sha256": sha256_file(INTERPRETER),
            "immediate_parent_transport_path": str(transport_final),
            "immediate_parent_transport_sha256": sha256_file(transport_stage),
            "launcher_argv": launcher_argv,
            "launcher_environment": EXACT_ENVIRONMENT,
            "launcher_sha256": sha256_file(launcher_stage),
            "shell": False,
        }
        package["transport_contract"]["attestation_sha256"] = sha256_bytes(compact_bytes(attestation))
        package["runtime_namespace"] = {
            "directory_mode_octal": "0700",
            "fallback_terminal": str(runtime_files()["fallback_terminal"]),
            "file_mode_octal": "0400",
            "owner_gid": EXPECTED_GID,
            "owner_uid": EXPECTED_UID,
            "paths": {key: str(path) for key, path in runtime_files().items() if key != "fallback_terminal"},
            "precreated_directories": [str(path) for path in runtime_directories()],
            "primary_root": str(PRIMARY_ROOT),
            "publication": "O_CREAT|O_EXCL then file fsync then parent-directory fsync",
            "runtime_root": str(RUNTIME_ROOT),
            "static_preparation_only": True,
            "terminal_publication_order": ["primary", "fallback"],
            "writability_proof": "TRANSIENT_O_EXCL_CREATE_SECOND_CREATE_REJECT_UNLINK_FSYNC",
        }
        package.pop("live_namespace", None)
        package["preflight"]["checks_in_order"][5] = "lstat owner mode and no-symlink checks plus primary and fallback O_EXCL probes before Fresh-L2 acceptance read"
        package["preflight_closure"]["operations_in_order"][6] = "VERIFY_EXTERNAL_RUNTIME_NAMESPACE_AND_PROVE_PRIMARY_FALLBACK_CREATE_ONCE"
        ordered = package["lifecycle"]["ordered_events"]
        authority_index = ordered.index("AUTHORITY_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC")
        ordered[authority_index:authority_index] = [
            "LSTAT_EXTERNAL_RUNTIME_PARENTS_NO_SYMLINK_EXACT_MODE_OWNER",
            "PROVE_PRIMARY_TERMINAL_PARENT_WRITABLE_CREATE_ONCE_AND_REMOVE_PROBE",
            "PROVE_FALLBACK_TERMINAL_PARENT_WRITABLE_CREATE_ONCE_AND_REMOVE_PROBE",
        ]
        for record_name, record in package["record_formats"].items():
            if record_name == "fresh_l2_acceptance":
                continue
            if record_name == "first_terminal":
                record["primary_path"] = str(runtime_files()["first_terminal"])
                record["fallback_path"] = str(runtime_files()["fallback_terminal"])
            elif record_name == "consumed_ledger":
                record["runtime_path"] = str(runtime_files()["ledger"])
            else:
                record["runtime_path"] = str(runtime_files()[record_name])
        package["production_canonical_preflight_proof"] = {
            "accepted_reader_function": "_read_accepted_package_and_result_schema",
            "independent_verifier": str(V15_ROOT / verifier_stage.relative_to(package_stage)),
            "preflight_function": "_preflight",
            "reader_functions": ["decode_canonical_json", "read_canonical_json"],
            "runtime_proof_before_authority": True,
            "runtime_proof_function": "_verify_runtime_namespace_pre_authority",
            "target_process_starts_required": 0,
        }
        package["static_file_policy"] = {
            "allowed_relative_files": [
                "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_PACKAGE.json",
                "V13_RUNTIME_NAMESPACE_FAILURE_REPORT.md",
                "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json",
                "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json",
                "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_AUTHORITY_SCHEMA.json",
                "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_CREDENTIAL_SCHEMA.json",
                "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_FIRST_TERMINAL_SCHEMA.json",
                "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
                "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_LEDGER_SCHEMA.json",
                "review/FRESH_L2_STATIC_ACCEPTANCE.json",
                "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v15.py",
                "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v15.py",
                "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree.py",
            ],
            "forbidden_directories": ["build", "live", "__pycache__"],
            "generated_python_artifacts_permitted": False,
            "runtime_namespace_outside_package_root": str(RUNTIME_ROOT),
            "static_files_only": True,
        }
        package["claim_boundary"] = {
            "acceptance_materialized": False,
            "authority_materialized": False,
            "controller_or_evaluator_invoked": False,
            "credential_materialized": False,
            "execution_authorized": False,
            "first_terminal_materialized": False,
            "ledger_materialized": False,
            "result_materialized": False,
            "runtime_namespace_precreated_empty": True,
            "sealed_tensor_access": "NONE",
            "transport_or_launcher_invoked": False,
        }
        package["root_id"] = "qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_action_root"
        for record_name, record in package["record_formats"].items():
            schema_id = record["schema_id"]
            schema_path = Path(record["schema_path"])
            stage_schema = package_stage / schema_path.relative_to(V15_ROOT)
            require(stage_schema.is_file(), f"record schema absent: {record_name}")
            record["schema_sha256"] = sha256_file(stage_schema)
        package["package_content_sha256"] = sha256_bytes(compact_bytes(package))
        write_bytes(package_path_stage, compact_bytes(package))

        for path in sorted(package_stage.rglob("*"), reverse=True):
            if path.is_file():
                os.chmod(path, 0o444)
        for path in sorted((item for item in package_stage.rglob("*") if item.is_dir()), reverse=True):
            os.chmod(path, 0o555)
        os.chmod(package_stage, 0o555)
        os.rename(runtime_stage, RUNTIME_ROOT)
        os.rename(package_stage, V15_ROOT)
    except Exception:
        if package_stage.exists():
            os.chmod(package_stage, 0o700)
            shutil.rmtree(package_stage)
        if runtime_stage.exists():
            shutil.rmtree(runtime_stage)
        raise

    print(compact_bytes({
        "action_id": ACTION_ID,
        "manifest_raw_sha256": sha256_file(V15_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_PACKAGE.json"),
        "package_root": str(V15_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "status": "BUILT_IMMUTABLE_V15_WRITABLE_RUNTIME_SUCCESSOR_NO_AUTHORITY",
    }).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
