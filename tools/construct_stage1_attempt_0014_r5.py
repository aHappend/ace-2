#!/usr/bin/env python3
"""Construct the disjoint unauthorized Stage-1 lifecycle-repair r5 successor."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/ace2_chat_demo"
SOURCE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-preparation-r4"
)
SOURCE_STATE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-a62e0c91-authority-state"
)
SOURCE_OUTPUT = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "runtime-output-0013-a62e0c91"
)
STAMP = "20260831T134355Z"
ATTEMPT = "0014"
NONCE = "b7e4c2d19a6f4385a0c73e12d89f6b41"
SHORT = NONCE[:8]
DESTINATION = BUILD / (
    f"stage1-official-rtl-backed-chat-demo-{STAMP}-"
    f"attempt-{ATTEMPT}-preparation-r5"
)
STAGE = BUILD / f".{DESTINATION.name}.staging"
QUALIFICATION = BUILD / f"{DESTINATION.name}-qualification"
FAILED_PREPARATION = BUILD / f"{DESTINATION.name}-failed-0001"
OUTPUT_REL = (
    f"build/ace2_chat_demo/stage1-official-rtl-backed-chat-demo-{STAMP}-"
    f"runtime-output-{ATTEMPT}-{SHORT}"
)
STATE_REL = (
    f"build/ace2_chat_demo/stage1-official-rtl-backed-chat-demo-{STAMP}-"
    f"attempt-{ATTEMPT}-{SHORT}-authority-state"
)
REVIEW_REL = (
    f"build/ace2_chat_demo/stage1-official-rtl-backed-chat-demo-{STAMP}-"
    f"attempt-{ATTEMPT}-{SHORT}-independent-review"
)
TASK_ID = f"ace2-stage1-official-rtl-chat-20260831-{ATTEMPT}-{SHORT}"
IDENTITY = f"ace2-stage1-attempt-{ATTEMPT}-{NONCE}"
DESTINATION_REL = DESTINATION.relative_to(ROOT).as_posix()
CREATED_AT = "2026-08-31T13:43:55.765000+00:00"
REGRESSION_SOURCE = ROOT / "tools/stage1_lifecycle_regression.py"
VALIDATOR_SOURCE = ROOT / "tools/validate_stage1_attempt_0014_r5.py"
TEST_SOURCE = ROOT / "tools/test_stage1_attempt_0014_r5.py"
HOST_RECEIPTS = {
    "bound_front_door": ROOT
    / ".argus_subagents/ace2-stage1-attempt-0013-bound-front-door-c3d76914efdb.json",
    "bound_front_door_exit": ROOT
    / (
        ".argus_subagents/"
        "ace2-stage1-attempt-0013-bound-front-door-c3d76914efdb_logs/"
        "exit_code.ace2-stage1-attempt-0013-bound-front-door-c3d76914efdb-"
        "1788182067455590635"
    ),
    "worker": ROOT
    / ".argus_subagents/ace2-stage1-official-rtl-chat-20260831-0013-a62e0c91.json",
    "worker_exit": ROOT
    / (
        ".argus_subagents/"
        "ace2-stage1-official-rtl-chat-20260831-0013-a62e0c91_logs/"
        "exit_code.ace2-stage1-official-rtl-chat-20260831-0013-a62e0c91-"
        "1788182069438434266"
    ),
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_bytes(path: Path, value: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    path.chmod(mode)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_bytes(value), 0o444)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object required: {path}")
    return value


def manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item.relative_to(path).as_posix(): {
            "bytes": item.stat(follow_symlinks=False).st_size,
            "mode": stat.S_IMODE(item.stat(follow_symlinks=False).st_mode),
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }


def receipt_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def build_candidate() -> dict[str, Any]:
    source = load(SOURCE / "package.json")
    replacements = {
        source["identity"]: DESTINATION.name,
        source["future_output"]: OUTPUT_REL,
        source["future_authority_state"]: STATE_REL,
        source["future_review_receipt_namespace"]: REVIEW_REL,
        source["future_task_id"]: TASK_ID,
        source["package_run_identity"]: IDENTITY,
        source["nonce"]: NONCE,
    }
    candidate = replace_strings(copy.deepcopy(source), replacements)
    candidate.update(
        {
            "attempt": ATTEMPT,
            "identity": DESTINATION.name,
            "nonce": NONCE,
            "package_run_identity": IDENTITY,
            "created_at_utc": CREATED_AT,
            "preparation_revision": "missing-path-lifecycle-repair-r5",
            "current_authority_cardinality": 0,
            "future_authority_cardinality": 0,
            "future_authority_state": STATE_REL,
            "future_output": OUTPUT_REL,
            "future_review_receipt_namespace": REVIEW_REL,
            "future_task_id": TASK_ID,
            "future_runner_run_id": None,
            "future_runner_run_id_template": rf"^{re.escape(TASK_ID)}-[0-9]+$",
            "execution_occurred": False,
            "status": "PREPARED_UNAUTHORIZED_UNCONSUMED",
            "successor_reason": (
                "FRESH_DISJOINT_SUCCESSOR_AFTER_ATTEMPT_0013_"
                "TERMINAL_MISSING_PATH_LIFECYCLE_FAILURE"
            ),
            "attempt_0013_status": "SEALED_TERMINAL_NO_RETRY",
            "attempt_0013_classification": "MODEL_EXCEPTION_FileNotFoundError",
            "attempt_0013_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        }
    )
    candidate["prohibitions"] = [
        item
        for item in candidate["prohibitions"]
        if "attempt-0013 model execution" not in item
    ] + [
        "attempt-0013 mutation, retry, replay, resume, relaunch, identity reuse, "
        "authority reuse, package reuse, or output reuse",
        "attempt-0014 model execution, RTL workload, submission, authority "
        "creation, or authority consumption during preparation and review",
        "new execution authority before explicit independent no_execution review PASS",
        "PPA, stage closing, Stage-2, or U280 work",
    ]
    return candidate


def build_commands(candidate: dict[str, Any]) -> dict[str, Any]:
    source = load(SOURCE / "commands.json")
    replacements = {
        load(SOURCE / "package.json")["identity"]: candidate["identity"],
        load(SOURCE / "package.json")["future_output"]: OUTPUT_REL,
        load(SOURCE / "package.json")["future_task_id"]: TASK_ID,
    }
    commands = replace_strings(source, replacements)
    commands.update(
        {
            "schema": "ace2-attempt-0014-future-command-v1",
            "submitted": False,
            "run_id_assigned": None,
            "authorization_state": "ABSENT",
        }
    )
    if "ACE2_ATTEMPT_0014_SUBMIT" not in commands["submit_mode_environment_removed"]:
        commands["submit_mode_environment_removed"].append(
            "ACE2_ATTEMPT_0014_SUBMIT"
        )
    return commands


def adjusted_runner() -> bytes:
    source = (SOURCE / "terminal_runner.py").read_text(encoding="utf-8")
    old_package = load(SOURCE / "package.json")
    for old, new in {
        old_package["identity"]: DESTINATION.name,
        old_package["future_output"]: OUTPUT_REL,
        old_package["future_authority_state"]: STATE_REL,
        old_package["future_task_id"]: TASK_ID,
        old_package["package_run_identity"]: IDENTITY,
        old_package["nonce"]: NONCE,
        "ace2-attempt-0013-consumption-v1": "ace2-attempt-0014-consumption-v1",
        "ace2-attempt-0013-model-invocation-v1": "ace2-attempt-0014-model-invocation-v1",
        "ATTEMPT_0013_EXTERNAL_GATES_VALID": "ATTEMPT_0014_EXTERNAL_GATES_VALID",
    }.items():
        source = source.replace(old, new)
    source = source.replace("import time\n", "import time\nimport traceback\n", 1)
    source = source.replace(
        '    "ACE2_ATTEMPT_0013_SUBMIT",\n)',
        '    "ACE2_ATTEMPT_0013_SUBMIT",\n    "ACE2_ATTEMPT_0014_SUBMIT",\n)',
        1,
    )
    old_size = '''def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )
'''
    new_size = '''def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_symlink():
                continue
            item_stat = item.stat()
        except FileNotFoundError:
            continue
        if stat_is_regular(item_stat.st_mode):
            total += item_stat.st_size
    return total


def _exception_path(error: BaseException) -> str | None:
    filename = getattr(error, "filename", None)
    if filename is None:
        return None
    path = Path(filename)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _exception_traceback(error: BaseException) -> dict[str, object]:
    frames = []
    for frame in traceback.extract_tb(error.__traceback__):
        source = Path(frame.filename)
        try:
            source_name = source.resolve().relative_to(ROOT).as_posix()
        except (OSError, ValueError):
            source_name = f"<external>/{source.name}"
        frames.append(
            {
                "source_file": source_name,
                "line": frame.lineno,
                "function": frame.name,
                "expression": frame.line or "",
            }
        )
    return {"frames": frames}


def publish_exception_record(
    state: Path,
    error: BaseException,
    *,
    phase: str,
    publisher: Publisher = publish,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "ace2-durable-lifecycle-exception-v1",
        "exception_type": type(error).__name__,
        "message": str(error),
        "missing_path": _exception_path(error),
        "phase": phase,
        "traceback": _exception_traceback(error),
    }
    primary = state / "exception-record.json"
    try:
        publisher(primary, value)
        selected = primary
    except BaseException as publication_error:
        value["primary_publication_failure"] = {
            "exception_type": type(publication_error).__name__,
            "message": str(publication_error),
        }
        selected = state / "fallback/exception-record.json"
        publish(selected, value)
    return {
        "path": selected.relative_to(state).as_posix(),
        "sha256": sha256_file(selected),
        "exception_type": value["exception_type"],
        "missing_path": value["missing_path"],
        "phase": value["phase"],
    }
'''
    if old_size not in source:
        raise RuntimeError("source directory_size implementation changed")
    source = source.replace(old_size, new_size, 1)
    source = source.replace(
        "    capture_publication_attempted = False\n"
        "    stream_bindings = unavailable_stream_bindings(\"CHILD_NOT_STARTED\")",
        "    capture_publication_attempted = False\n"
        "    phase = \"lifecycle_setup\"\n"
        "    exception_binding: dict[str, object] | None = None\n"
        "    stream_bindings = unavailable_stream_bindings(\"CHILD_NOT_STARTED\")",
        1,
    )
    source = source.replace(
        "            child = subprocess.Popen(\n",
        "            phase = \"child_spawn\"\n"
        "            child = subprocess.Popen(\n",
        1,
    )
    source = source.replace(
        "            runtime_started = time.monotonic()\n"
        "            while True:\n",
        "            runtime_started = time.monotonic()\n"
        "            phase = \"runtime_monitor\"\n"
        "            while True:\n",
        1,
    )
    source = source.replace(
        "                    if runtime_max_bytes is not None and directory_size(output) > runtime_max_bytes:\n",
        "                    phase = \"runtime_output_size_check\"\n"
        "                    output_bytes = directory_size(output)\n"
        "                    phase = \"runtime_monitor\"\n"
        "                    if runtime_max_bytes is not None and output_bytes > runtime_max_bytes:\n",
        1,
    )
    old_except = '''        except BaseException as error:
            if isinstance(error, StreamPreservationError):
'''
    new_except = '''        except BaseException as error:
            exception_binding = publish_exception_record(
                state,
                error,
                phase=phase,
                publisher=publisher,
            )
            if isinstance(error, StreamPreservationError):
'''
    if old_except not in source:
        raise RuntimeError("source lifecycle exception handler changed")
    source = source.replace(old_except, new_except, 1)
    source = source.replace(
        '                            "second_model_invocation_permitted": False,\n'
        "                        }\n"
        '                        if pending is not None:',
        '                            "second_model_invocation_permitted": False,\n'
        "                        }\n"
        "                        if exception_binding is not None:\n"
        '                            payload["exception_record"] = exception_binding\n'
        "                        if pending is not None:",
        1,
    )
    source = source.replace(
        '                            "attempt_0013_retry_replay_resume_relaunch": "FORBIDDEN",\n',
        '                            "attempt_0013_retry_replay_resume_relaunch": "FORBIDDEN",\n'
        '                            "attempt_0014_retry_replay_resume_relaunch": "FORBIDDEN",\n',
        1,
    )
    required = (
        "runtime_output_size_check",
        "publish_exception_record",
        "attempt_0014_retry_replay_resume_relaunch",
        "ACE2_ATTEMPT_0014_SUBMIT",
    )
    if any(token not in source for token in required):
        raise RuntimeError("successor runner patch is incomplete")
    return source.encode("utf-8")


def build_execution_bindings(candidate: dict[str, Any]) -> dict[str, Any]:
    source = load(SOURCE / "execution-bindings.json")
    old = load(SOURCE / "package.json")
    bindings = replace_strings(
        source,
        {
            old["identity"]: candidate["identity"],
            old["future_task_id"]: TASK_ID,
        },
    )
    bindings["schema"] = "ace2-attempt-0014-execution-bindings-v1"
    bindings["lifecycle_repair"] = {
        "policy": "CREATE_USE_DURABLE_RESULT_PRUNE",
        "supervisor_scan": "DISAPPEARING_TRANSIENTS_ARE_IGNORED",
        "exception_record": (
            "TYPE_MISSING_PATH_PHASE_TRACEBACK_DURABLE_BEFORE_TERMINAL"
        ),
        "regression": "lifecycle-regression.json",
    }
    removed = bindings["process_mode_contract"]["submit_environment_removed"]
    if "ACE2_ATTEMPT_0014_SUBMIT" not in removed:
        removed.append("ACE2_ATTEMPT_0014_SUBMIT")
    return bindings


def seal_attempt_0013(
    package_before: dict[str, Any],
    state_before: dict[str, Any],
    output_before: dict[str, Any],
) -> dict[str, Any]:
    terminal = load(SOURCE_STATE / "terminal-status.json")
    layers_root = SOURCE_OUTPUT / "tokens/position-00"
    layers = sorted(
        path.name for path in layers_root.iterdir() if path.is_dir()
    )
    if layers != [f"layer-{layer:02d}" for layer in range(11)]:
        raise RuntimeError(f"unexpected attempt-0013 layer extent: {layers}")
    receipts = {
        name: receipt_record(path) for name, path in HOST_RECEIPTS.items()
    }
    for name in (
        "execution-consumed.json",
        "submit-intent.json",
        "submit-receipt.json",
        "submission-release.json",
        "worker-handoff.json",
        "terminal-status.json",
    ):
        receipts[f"state_{name.removesuffix('.json').replace('-', '_')}"] = (
            receipt_record(SOURCE_STATE / name)
        )
    return {
        "schema": "ace2-stage1-attempt-0013-terminal-evidence-seal-v1",
        "status": "AUTHENTICATED_SEALED_IMMUTABLE",
        "before_after_byte_exact": True,
        "package": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "tree_root": sha256_file(SOURCE / "SHA256SUMS"),
            "files": package_before,
        },
        "authority_state": {
            "path": SOURCE_STATE.relative_to(ROOT).as_posix(),
            "files": state_before,
            "terminal_sha256": sha256_file(
                SOURCE_STATE / "terminal-status.json"
            ),
            "consumption_sha256": sha256_file(
                SOURCE_STATE / "execution-consumed.json"
            ),
        },
        "runtime_output": {
            "path": SOURCE_OUTPUT.relative_to(ROOT).as_posix(),
            "files": output_before,
            "file_count": len(output_before),
            "position_00_layers": layers,
            "completed_layer_count": len(layers),
            "shared_present": (SOURCE_OUTPUT / "shared").is_dir(),
        },
        "durable_receipts": receipts,
        "terminal": {
            "status": terminal["status"],
            "classification": terminal["classification"],
            "exit_code": terminal["exit_code"],
            "natural_terminal": terminal["natural_terminal"],
            "second_model_invocation_permitted": terminal[
                "second_model_invocation_permitted"
            ],
            "process_group_empty": terminal["child_lifecycle"][
                "process_group_empty"
            ],
            "capture_drains_finished": terminal["child_lifecycle"][
                "capture_drains_finished"
            ],
        },
        "mutation": {
            "package": False,
            "authority_state": False,
            "runtime_output": False,
            "receipts": False,
        },
    }


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return loaded


def seal_package(path: Path) -> str:
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.relative_to(path).as_posix()
        not in {"SHA256SUMS", "TREE_ROOT.sha256"}
        and "__pycache__" not in item.parts
    )
    lines = [
        f"{sha256_file(item)}  {item.relative_to(path).as_posix()}"
        for item in files
    ]
    write_bytes(
        path / "SHA256SUMS",
        ("\n".join(lines) + "\n").encode("ascii"),
        0o444,
    )
    root = sha256_file(path / "SHA256SUMS")
    write_bytes(
        path / "TREE_ROOT.sha256",
        f"{root}  SHA256SUMS\n".encode("ascii"),
        0o444,
    )
    for item in files:
        item.chmod(0o555 if item.suffix in {".py", ".sh"} else 0o444)
    for directory in sorted(
        [path, *(item for item in path.rglob("*") if item.is_dir())],
        reverse=True,
    ):
        if "__pycache__" not in directory.parts:
            directory.chmod(0o555)
    return root


def main() -> int:
    if any(
        path.exists()
        for path in (DESTINATION, QUALIFICATION, FAILED_PREPARATION)
    ):
        raise RuntimeError("r5 destination, qualification, or failed preparation exists")
    failed_root: str | None = None
    if STAGE.exists():
        write_json(
            STAGE / "preparation-failure.json",
            {
                "schema": "ace2-stage1-r5-preparation-failure-v1",
                "status": "FAILED_PRESERVED_NO_EXECUTION",
                "failure_taxonomy": "test_fixture_setup",
                "root_cause_hypothesis": (
                    "The inert exception-capture test called the production "
                    "lifecycle with an absent authority-state directory, while "
                    "the real worker enters with that directory already present."
                ),
                "regression": (
                    "Create the inert authority-state directory before invoking "
                    "the lifecycle runner."
                ),
                "model_executed": False,
                "rtl_executed": False,
                "successor_published": False,
            },
        )
        failed_root = seal_package(STAGE)
        os.replace(STAGE, FAILED_PREPARATION)
    if any(os.path.lexists(ROOT / path) for path in (OUTPUT_REL, STATE_REL, REVIEW_REL)):
        raise RuntimeError("r5 output, authority, or review namespace is not fresh")
    terminal = load(SOURCE_STATE / "terminal-status.json")
    if (
        terminal.get("status") != "SEALED_TERMINAL_NO_RETRY"
        or terminal.get("classification") != "MODEL_EXCEPTION_FileNotFoundError"
        or terminal.get("second_model_invocation_permitted") is not False
    ):
        raise RuntimeError("attempt-0013 is not the expected sealed terminal")
    for path in HOST_RECEIPTS.values():
        if not path.is_file():
            raise RuntimeError(f"attempt-0013 receipt is absent: {path}")

    package_before = manifest(SOURCE)
    state_before = manifest(SOURCE_STATE)
    output_before = manifest(SOURCE_OUTPUT)
    STAGE.mkdir()
    candidate = build_candidate()
    write_json(STAGE / "package.json", candidate)
    write_json(STAGE / "commands.json", build_commands(candidate))
    write_json(
        STAGE / "execution-bindings.json",
        build_execution_bindings(candidate),
    )
    for name in (
        "capacity.json",
        "source-bindings.sha256",
        "source-closure.json",
        "storage-contract.json",
        "prompt.txt",
    ):
        mode = 0o444
        write_bytes(STAGE / name, (SOURCE / name).read_bytes(), mode)
    write_bytes(STAGE / "terminal_runner.py", adjusted_runner(), 0o555)
    write_bytes(
        STAGE / "lifecycle_regression.py",
        REGRESSION_SOURCE.read_bytes(),
        0o555,
    )
    write_bytes(
        STAGE / "validate_package.py",
        VALIDATOR_SOURCE.read_bytes(),
        0o555,
    )
    write_bytes(
        STAGE / "tests/test_candidate.py",
        TEST_SOURCE.read_bytes(),
        0o555,
    )
    launch = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {ROOT}\n"
        f"exec python3 -B {DESTINATION_REL}/terminal_runner.py\n"
    )
    submit = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {ROOT}\n"
        f"exec python3 -B {DESTINATION_REL}/terminal_runner.py --submit\n"
    )
    write_bytes(STAGE / "launch.sh", launch.encode("ascii"), 0o555)
    write_bytes(STAGE / "submit_and_capture.sh", submit.encode("ascii"), 0o555)
    write_json(
        STAGE / "attempt-0013-terminal-seal.json",
        seal_attempt_0013(package_before, state_before, output_before),
    )

    runner = load_module("_r5_staged_runner", STAGE / "terminal_runner.py")
    regression_module = load_module(
        "_r5_staged_regression", STAGE / "lifecycle_regression.py"
    )
    regression = regression_module.run_all(runner)
    write_json(STAGE / "lifecycle-regression.json", regression)
    write_json(
        STAGE / "failure-mechanism.json",
        {
            "schema": "ace2-stage1-attempt-0013-missing-path-mechanism-v1",
            "status": "DETERMINISTICALLY_REPRODUCED_NO_EXECUTION",
            "failure_taxonomy": "file_lifecycle_race",
            "root_cause": (
                "The supervisor used an is_file-then-stat directory-size scan "
                "while the child pruned regenerable layer artifacts after durable "
                "result publication. A transient could disappear between those "
                "operations, raising FileNotFoundError in the supervisor."
            ),
            "original_exact_missing_path": (
                "NOT_DURABLY_CAPTURED_BY_ATTEMPT_0013"
            ),
            "original_capture_limitation": (
                "attempt-0013 sealed only the exception class; its empty streams "
                "and terminal payload contain no path or traceback"
            ),
            "reproduction": regression["legacy_failure_reproduction"],
            "source_evidence": {
                "runner": SOURCE.relative_to(ROOT).as_posix()
                + "/terminal_runner.py",
                "runner_sha256": sha256_file(SOURCE / "terminal_runner.py"),
                "unsafe_function": "directory_size",
                "backend": "tools/rtl_arbitrary_text_generation_backend.py",
                "backend_sha256": sha256_file(
                    ROOT / "tools/rtl_arbitrary_text_generation_backend.py"
                ),
                "prune_function": "prune_transient_execution_artifacts",
                "ordering": "durable_result_then_prune",
            },
            "repair": {
                "supervisor_scan": "single-stat-with-FileNotFoundError-tolerance",
                "exception_record": (
                    "type, exact missing path, phase, and sanitized traceback "
                    "fsynced before terminal publication"
                ),
            },
            "official": False,
            "model_executed": False,
            "rtl_executed": False,
        },
    )
    write_json(
        STAGE / "review-contract.json",
        {
            "schema": "ace2-stage1-attempt-0014-r5-review-request-v1",
            "status": "PENDING_INDEPENDENT_NO_EXECUTION_REVIEW",
            "reviewer": "INDEPENDENT_READ_ONLY",
            "authority_granted_by_review": False,
            "execution_permitted_during_review": False,
            "package": DESTINATION_REL,
            "checks": [
                "attempt-0013 evidence seal authenticates all receipts, "
                "consumption, terminal, output hashes, and 11-layer extent",
                "missing-path lifecycle race is deterministically reproduced "
                "without model or RTL execution",
                "24 inert layers preserve create/use/durable-result/prune order",
                "premature prune and missing generated input fail closed",
                "exception-record primary publication failure uses a durable "
                "fallback before terminal sealing",
                "successor is disjoint, unconsumed, unauthorized, and has "
                "authority cardinality zero",
            ],
        },
    )
    if failed_root is not None:
        write_json(
            STAGE / "preparation-repair-0001.json",
            {
                "schema": "ace2-stage1-r5-preparation-repair-v1",
                "status": "REPAIRED_IN_DISJOINT_PREPARATION_NAMESPACE",
                "failed_preparation": FAILED_PREPARATION.relative_to(
                    ROOT
                ).as_posix(),
                "failed_tree_root": failed_root,
                "failure_taxonomy": "test_fixture_setup",
                "root_cause_hypothesis": (
                    "The inert fixture omitted the production precondition that "
                    "the authority-state directory exists before runner entry."
                ),
                "regression": (
                    "The fixture now creates the state directory before invoking "
                    "the runner; no model or RTL execution is involved."
                ),
                "failed_preparation_mutated_after_seal": False,
            },
        )

    preseal = subprocess.run(
        ["python3", "-B", (STAGE / "tests/test_candidate.py").as_posix()],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if preseal.returncode != 0:
        raise RuntimeError(
            "r5 prepublication tests failed:\n"
            + preseal.stdout
            + preseal.stderr
        )
    if (
        manifest(SOURCE) != package_before
        or manifest(SOURCE_STATE) != state_before
        or manifest(SOURCE_OUTPUT) != output_before
    ):
        raise RuntimeError("attempt-0013 evidence changed during r5 construction")
    tree_root = seal_package(STAGE)
    os.replace(STAGE, DESTINATION)
    descriptor = os.open(BUILD, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    sealed = subprocess.run(
        ["python3", "-B", (DESTINATION / "validate_package.py").as_posix()],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    QUALIFICATION.mkdir()
    write_bytes(
        QUALIFICATION / "prepublication-tests.stdout.log",
        preseal.stdout.encode("utf-8"),
        0o444,
    )
    write_bytes(
        QUALIFICATION / "prepublication-tests.stderr.log",
        preseal.stderr.encode("utf-8"),
        0o444,
    )
    write_bytes(
        QUALIFICATION / "sealed-validation.stdout.log",
        sealed.stdout.encode("utf-8"),
        0o444,
    )
    write_bytes(
        QUALIFICATION / "sealed-validation.stderr.log",
        sealed.stderr.encode("utf-8"),
        0o444,
    )
    write_json(
        QUALIFICATION / "qualification.json",
        {
            "schema": "ace2-stage1-attempt-0014-r5-qualification-v1",
            "status": "PASS" if sealed.returncode == 0 else "FAIL",
            "official": False,
            "model_executed": False,
            "rtl_executed": False,
            "package": DESTINATION_REL,
            "tree_root": tree_root,
            "prepublication_test_exit": preseal.returncode,
            "sealed_validation_exit": sealed.returncode,
            "authority_cardinality": 0,
            "authorized": False,
            "consumed": False,
            "review_status": "PENDING_INDEPENDENT_NO_EXECUTION_REVIEW",
            "qualified_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    if sealed.returncode != 0:
        raise RuntimeError(
            "sealed r5 validation failed:\n" + sealed.stdout + sealed.stderr
        )
    print(
        "ACE2_STAGE1_ATTEMPT_0014_R5_PREPARED "
        f"package={DESTINATION_REL} tree_root={tree_root} "
        "authorized=false consumed=false authority_cardinality=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
