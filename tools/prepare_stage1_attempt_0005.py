#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-official-rtl-backed-chat-demo-20260828T162359Z-attempt-0005-preparation-r1"
)
PROMPT = b"Why must RTL verification be cycle-accurate?\n"
MODEL_SNAPSHOT = (
    ROOT
    / "build/ace2_chat_demo/cf20-bf16-portable-snapshot"
    / "7ae557604adf67be50417f59c2c2f167def9a775"
)
IMAGE_ROOT = (
    ROOT / "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1"
)
RUNTIME_BINARY = (
    ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness"
)
ATTEMPT_0004_ROOTS = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-official-rtl-backed-chat-demo-20260828T101310Z-attempt-0004-preparation",
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-official-rtl-backed-chat-demo-20260828T101310Z-runtime-output-0004-2d7f3071",
    ROOT
    / ".argus_subagents"
    / "ace2-stage1-official-rtl-chat-20260828-0004-2d7f3071.json",
    ROOT
    / ".argus_subagents"
    / "ace2-stage1-official-rtl-chat-20260828-0004-2d7f3071_logs",
)
PROTECTED_INPUTS = (
    "design/BENCHMARK_INTERFACE.json",
    "design/SPEC.md",
    "evidence/reviews/nonofficial-hybrid-0026-l2-reviewer-verdict.json",
    "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/full_model_image.bin",
    "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/image_contract_v2.json",
    "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/validation_report.json",
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/nonofficial-hybrid-0026/live/result.json",
    "research/PIPELINE_STATE.json",
)
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
RTL_COMPILE_SOURCES = (
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_dynamic_scale32_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_dynamic_rope_head_core.sv",
    "rtl/ace2_fixed_q7_rope_score_core.sv",
    "rtl/ace2_relative_rope_score_fusion_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/ace2_layer23_v_rank1_integer_correction_sidecar.sv",
    "rtl/ace2_shell.sv",
    "verification/verilator/ace2_shell_runtime_harness.sv",
)
RUNTIME_MAX_BYTES = 80 * 1024**3
TEMPORARY_MAX_BYTES = 8 * 1024**3
PACKAGE_MAX_BYTES = 32 * 1024**2
SAFETY_MARGIN_PERCENT = 25


class PreparationError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(value):
            written = os.write(descriptor, value[offset:])
            if written <= 0:
                raise PreparationError(f"short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: object) -> None:
    write(path, canonical_bytes(value))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_record(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise PreparationError(f"regular file required: {path}")
    observed = path.stat()
    return {
        "path": relative(path),
        "bytes": observed.st_size,
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": sha256_file(path),
    }


def tree_records(path: Path) -> list[dict[str, object]]:
    if not os.path.lexists(path):
        raise PreparationError(f"attempt-0004 evidence is absent: {path}")
    members = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    records = []
    for member in members:
        if member.is_symlink():
            raise PreparationError(f"attempt-0004 evidence contains symlink: {member}")
        records.append(file_record(member))
    return records


def local_python_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[Path] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
                if node.module == "tools":
                    names.extend(f"tools.{alias.name}" for alias in node.names)
        for name in names:
            if not name.startswith("tools."):
                continue
            candidate = ROOT / (name.replace(".", "/") + ".py")
            if candidate.is_file():
                found.add(candidate)
    return found


def source_paths() -> list[Path]:
    pending = [
        ROOT / "tools/ace2_chat_demo.py",
        ROOT / "tools/ace2_stage1_chat_product.py",
        ROOT / "tools/rtl_arbitrary_text_generation_backend.py",
        ROOT / "tools/run_rtl_arbitrary_text_generation.py",
    ]
    python: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in python:
            continue
        if not path.is_file():
            raise PreparationError(f"execution source is absent: {path}")
        python.add(path)
        pending.extend(local_python_imports(path) - python)
    hdl = {
        path
        for base in (ROOT / "rtl", ROOT / "verification/tb", ROOT / "verification/verilator")
        for path in base.rglob("*")
        if path.is_file() and path.suffix in {".sv", ".svh", ".cpp", ".h"}
    }
    inputs = {ROOT / name for name in PROTECTED_INPUTS}
    inputs.update(MODEL_SNAPSHOT / name for name in MODEL_FILES)
    inputs.add(RUNTIME_BINARY)
    paths = python | hdl | inputs
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise PreparationError(f"source closure member is absent: {missing[0]}")
    return sorted(paths)


def exact_module_header(path: Path, module: str) -> str:
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"\bmodule\s+{re.escape(module)}\b", source)
    if match is None:
        raise PreparationError(f"module is absent: {module}")
    depth = 0
    for offset in range(match.start(), len(source)):
        character = source[offset]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == ";" and depth == 0:
            return source[match.start() : offset + 1]
    raise PreparationError(f"unterminated module header: {module}")


def tool_version(argv: list[str]) -> dict[str, object]:
    executable = shutil.which(argv[0])
    if executable is None:
        return {"argv": argv, "available": False}
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    return {
        "argv": argv,
        "available": True,
        "executable": executable,
        "exit_code": completed.returncode,
        "output": completed.stdout.strip(),
    }


def compile_rtl_contract() -> dict[str, object]:
    argv = [
        "verilator",
        "--lint-only",
        "--language",
        "1800-2017",
        "-Wno-fatal",
        "--top-module",
        "ace2_shell_runtime_harness",
        "-Irtl",
        *RTL_COMPILE_SOURCES,
    ]
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise PreparationError(
            "exact public RTL contract failed compile-only validation: "
            + completed.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    return {
        "schema": "ace2-attempt-0005-rtl-compile-only-v1",
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "simulation_executed": False,
        "model_executed": False,
    }


def runner_source() -> bytes:
    source = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
CONTRACT = json.loads((PACKAGE / "package.json").read_text(encoding="utf-8"))
COMMANDS = json.loads((PACKAGE / "commands.json").read_text(encoding="utf-8"))
CAPACITY = json.loads((PACKAGE / "capacity.json").read_text(encoding="utf-8"))
STATE = ROOT / CONTRACT["future_authority_state"]
OUTPUT = ROOT / CONTRACT["future_output"]
MODEL_ARGV = COMMANDS["model_argv"]
RUN_PATTERN = re.compile(CONTRACT["future_runner_run_id_template"])


class GateError(RuntimeError):
    pass


class InterruptedBySignal(RuntimeError):
    def __init__(self, number: int) -> None:
        super().__init__(f"interrupted by signal {number}")
        self.number = number


Publisher = Callable[[Path, object], None]


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path: Path, value: object) -> None:
    raw = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short create-only publication")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_dir(path.parent)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"object required: {path}")
    return value


def process_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = raw.rfind(")")
    return int(raw[closing + 2 :].split()[19])


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def package_root() -> str:
    return (PACKAGE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()[0]


def validate_external_gates(*, require_receipt: bool) -> dict[str, Any]:
    from validate_package import validate

    validate(require_zero_state=False, verify_preserved_attempt=False)
    if not STATE.is_dir() or STATE.is_symlink():
        raise GateError("separate Manager authority-state namespace is absent")
    manager = load(STATE / "manager-authority.json")
    reviewer = load(STATE / "reviewer-accept.json")
    expected = {
        "package_tree_root": package_root(),
        "package_run_identity": CONTRACT["package_run_identity"],
        "task_id": CONTRACT["future_task_id"],
        "nonce": CONTRACT["nonce"],
        "execution_limit": 1,
        "output": CONTRACT["future_output"],
    }
    if (
        manager.get("schema") != "ace2-attempt-manager-authority-v1"
        or manager.get("status") != "GRANTED_EXACTLY_ONCE"
        or any(manager.get(key) != value for key, value in expected.items())
    ):
        raise GateError("Manager exactly-once authority is absent or misbound")
    if (
        reviewer.get("schema") != "ace2-attempt-independent-review-v1"
        or reviewer.get("producer_role") != "reviewer"
        or reviewer.get("reviewer_level") != "L2"
        or reviewer.get("decision") != "ACCEPT"
        or reviewer.get("package_tree_root") != package_root()
        or reviewer.get("package_run_identity") != CONTRACT["package_run_identity"]
        or reviewer.get("independent") is not True
    ):
        raise GateError("independent L2 Reviewer ACCEPT is absent or misbound")
    if require_receipt:
        receipt = load(STATE / "submit-receipt.json")
        if (
            receipt.get("task_id") != CONTRACT["future_task_id"]
            or receipt.get("state") != "submitted"
            or not isinstance(receipt.get("run_id"), str)
            or RUN_PATTERN.fullmatch(receipt["run_id"]) is None
        ):
            raise GateError("runner-issued run_id receipt is absent or invalid")
        manager["runner_receipt"] = receipt
    return manager


def wait_for_receipt(seconds: int = 120) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while True:
        try:
            return validate_external_gates(require_receipt=True)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise GateError("runner receipt was not atomically captured in time")
            time.sleep(0.1)


def check_capacity(*, available_bytes: int | None = None) -> None:
    usage = os.statvfs(PACKAGE.parent)
    available = (
        usage.f_bavail * usage.f_frsize
        if available_bytes is None
        else available_bytes
    )
    if os.stat(PACKAGE.parent).st_dev != os.stat(OUTPUT.parent).st_dev:
        raise GateError("future output parent moved to another device")
    if available < CAPACITY["required_available_bytes"]:
        raise GateError(
            f"insufficient capacity: {available} < {CAPACITY['required_available_bytes']}"
        )


def consume(manager: dict[str, Any], *, state: Path = STATE, output: Path = OUTPUT) -> None:
    ambiguous = (
        "execution-consumed.json",
        "run.started.json",
        "terminal-status.json",
        "recovery-terminal-status.json",
        "invocation.json",
    )
    if any(os.path.lexists(state / name) for name in ambiguous) or os.path.lexists(output):
        raise GateError("authority is already consumed or runtime state is ambiguous")
    (state / "fallback").mkdir(mode=0o700)
    fsync_dir(state)
    publish(
        state / "execution-consumed.json",
        {
            "schema": "ace2-attempt-0005-consumption-v1",
            "status": "CONSUMED_AND_DURABLE_BEFORE_MODEL_PROCESS",
            "identity": CONTRACT["package_run_identity"],
            "run_id": manager["runner_receipt"]["run_id"],
            "execution_limit": 1,
            "attempt_0004_inheritance": "NONE",
            "supervisor_pid": os.getpid(),
            "supervisor_start_ticks": process_start_ticks(os.getpid()),
            "boot_id": boot_id(),
        },
    )


def fixture_argv(kind: str) -> list[str]:
    if kind == "success":
        return [sys.executable, "-c", "raise SystemExit(0)"]
    if kind == "nonzero":
        return [sys.executable, "-c", "raise SystemExit(7)"]
    if kind == "sleep":
        return [sys.executable, "-c", "import time; time.sleep(60)"]
    raise GateError("unknown inert fixture")


def terminal_payload(
    *,
    prefix: str,
    classification: str,
    exit_code: int,
    child: subprocess.Popen[bytes] | None,
    natural_terminal: bool,
) -> dict[str, object]:
    return {
        "schema": "ace2-package-native-terminal-v2",
        "status": "SEALED_TERMINAL_NO_RETRY",
        "classification": f"{prefix}_{classification}",
        "exit_code": exit_code,
        "natural_terminal": natural_terminal,
        "supervisor_pid": os.getpid(),
        "supervisor_start_ticks": process_start_ticks(os.getpid()),
        "boot_id": boot_id(),
        "child_pid": child.pid if child is not None else None,
        "child_start_ticks": (
            process_start_ticks(child.pid)
            if child is not None and child.poll() is None
            else None
        ),
        "attempt_0004_retry_replay_resume": "FORBIDDEN",
        "second_model_invocation_permitted": False,
    }


def publish_terminal(
    state: Path,
    payload: dict[str, object],
    *,
    publisher: Publisher = publish,
) -> Path:
    primary = state / "terminal-status.json"
    try:
        publisher(primary, payload)
        return primary
    except (GateError, OSError, RuntimeError, ValueError, TypeError) as error:
        payload = dict(payload)
        payload["primary_publication_failure"] = type(error).__name__
        fallback = state / "fallback/terminal-status.json"
        publish(fallback, payload)
        return fallback


def execute(
    argv: list[str],
    *,
    state: Path,
    output: Path,
    prefix: str,
    publisher: Publisher = publish,
    runtime_max_bytes: int | None = None,
) -> int:
    publish(
        state / "run.started.json",
        {
            "schema": "ace2-package-native-durable-start-v2",
            "status": "STARTED_BEFORE_CHILD",
            "supervisor_pid": os.getpid(),
            "supervisor_start_ticks": process_start_ticks(os.getpid()),
            "boot_id": boot_id(),
            "argv_sha256": hashlib.sha256(canonical_bytes(argv)).hexdigest(),
        },
    )
    child: subprocess.Popen[bytes] | None = None
    previous: dict[int, Any] = {}

    def interrupted(number: int, _frame: object) -> None:
        raise InterruptedBySignal(number)

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[number] = signal.signal(number, interrupted)
    classification = "RUNNER_EXCEPTION"
    exit_code = 2
    natural_terminal = False
    try:
        child = subprocess.Popen(
            argv,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        publish(
            state / "child.started.json",
            {
                "schema": "ace2-package-native-child-start-v2",
                "pid": child.pid,
                "start_ticks": process_start_ticks(child.pid),
            },
        )
        while True:
            try:
                stdout, stderr = child.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if (
                    runtime_max_bytes is not None
                    and directory_size(output) > runtime_max_bytes
                ):
                    classification = "DISK_GROWTH_LIMIT"
                    raise GateError("bounded runtime output ceiling exceeded")
        exit_code = int(child.returncode)
        natural_terminal = True
        classification = "SUCCESS" if exit_code == 0 else "NONZERO"
        publish(state / "stdout.sha256.json", {"sha256": hashlib.sha256(stdout).hexdigest()})
        publish(state / "stderr.sha256.json", {"sha256": hashlib.sha256(stderr).hexdigest()})
    except InterruptedBySignal as error:
        classification = "SIGNAL"
        exit_code = 128 + error.number
    except BaseException as error:
        classification = (
            classification
            if classification == "DISK_GROWTH_LIMIT"
            else f"EXCEPTION_{type(error).__name__}"
        )
        exit_code = 2
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        payload = terminal_payload(
            prefix=prefix,
            classification=classification,
            exit_code=exit_code,
            child=child,
            natural_terminal=natural_terminal,
        )
        if classification == "SIGNAL":
            payload["signal"] = exit_code - 128
        publish_terminal(state, payload, publisher=publisher)
        for number, handler in previous.items():
            signal.signal(number, handler)
    return exit_code


def execute_fixture(
    kind: str,
    state: Path,
    *,
    publisher: Publisher = publish,
) -> int:
    state.mkdir(mode=0o700)
    (state / "fallback").mkdir(mode=0o700)
    fsync_dir(state.parent)
    return execute(
        fixture_argv(kind),
        state=state,
        output=state / "fixture-output",
        prefix="FIXTURE",
        publisher=publisher,
    )


def recover_host_loss(state: Path = STATE) -> str:
    consumed = state / "execution-consumed.json"
    started_path = state / "run.started.json"
    terminals = (
        state / "terminal-status.json",
        state / "fallback/terminal-status.json",
        state / "recovery-terminal-status.json",
    )
    if not consumed.is_file() or any(path.exists() for path in terminals):
        raise GateError("no unterminated consumed run is available for recovery")
    started = load(started_path) if started_path.is_file() else load(consumed)
    pid = started.get("supervisor_pid")
    ticks = started.get("supervisor_start_ticks")
    recorded_boot = started.get("boot_id")
    if not isinstance(pid, int) or not isinstance(ticks, int) or not isinstance(recorded_boot, str):
        raise GateError("durable start process identity is incomplete")
    same_process = False
    try:
        same_process = process_start_ticks(pid) == ticks
    except (FileNotFoundError, ProcessLookupError):
        pass
    if same_process:
        raise GateError("recorded supervisor is still live")
    classification = (
        "HOST_POWER_LOSS_CONFIRMED_AFTER_REBOOT"
        if boot_id() != recorded_boot
        else "INTERRUPTED_PROCESS_ABSENT_OR_PID_REUSED"
    )
    publish(
        state / "recovery-terminal-status.json",
        {
            "schema": "ace2-package-native-recovery-terminal-v2",
            "classification": classification,
            "consumption_remains_permanent": True,
            "second_model_invocation_permitted": False,
        },
    )
    return classification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--recover-host-loss", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        validate_external_gates(require_receipt=False)
        check_capacity()
        print("ATTEMPT_0005_EXTERNAL_GATES_VALID")
        return
    if args.recover_host_loss:
        print(recover_host_loss())
        return
    validate_external_gates(require_receipt=False)
    manager = wait_for_receipt()
    check_capacity()
    consume(manager)
    publish(
        STATE / "invocation.json",
        {
            "schema": "ace2-attempt-0005-model-invocation-v1",
            "argv": MODEL_ARGV,
            "cardinality": 1,
        },
    )
    raise SystemExit(
        execute(
            MODEL_ARGV,
            state=STATE,
            output=OUTPUT,
            prefix="MODEL",
            runtime_max_bytes=CAPACITY["runtime_output_max_bytes"],
        )
    )


if __name__ == "__main__":
    main()
'''
    return source.encode("utf-8")


def receipt_io_source() -> bytes:
    source = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def publish(path: Path, value: object) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short create-only receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def main() -> None:
    mode, name = sys.argv[1:]
    path = Path(name)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError("separate Manager authority-state namespace is absent")
    if mode == "reserve":
        publish(path, {"schema": "ace2-submit-intent-v1", "status": "RESERVED_ONCE"})
    elif mode == "receipt":
        value = json.load(sys.stdin)
        if not isinstance(value, dict):
            raise RuntimeError("runner receipt must be an object")
        publish(path, value)
    else:
        raise RuntimeError("invalid receipt operation")


if __name__ == "__main__":
    main()
'''
    return source.encode("utf-8")


def validator_source() -> bytes:
    source = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
EXCLUDED = {"SHA256SUMS", "TREE_ROOT.sha256"}


class ValidationError(RuntimeError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"object required: {path}")
    return value


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in records:
            raise ValidationError(f"invalid checksum line: {line!r}")
        records[match[2]] = match[1]
    return records


def verify_package_closure() -> None:
    sums = parse_sums(PACKAGE / "SHA256SUMS")
    actual = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_file() and path.relative_to(PACKAGE).as_posix() not in EXCLUDED
    }
    if set(sums) != actual:
        raise ValidationError("package file set changed")
    for relative, expected in sums.items():
        path = PACKAGE / relative
        if path.is_symlink() or digest(path) != expected:
            raise ValidationError(f"package file changed: {relative}")
        expected_mode = 0o555 if path.suffix in {".py", ".sh"} else 0o444
        if stat.S_IMODE(path.stat().st_mode) != expected_mode:
            raise ValidationError(f"package mode changed: {relative}")
    expected_root = f"{digest(PACKAGE / 'SHA256SUMS')}  SHA256SUMS"
    if (PACKAGE / "TREE_ROOT.sha256").read_text(encoding="ascii").strip() != expected_root:
        raise ValidationError("package tree root changed")
    for path in (PACKAGE, *(item for item in PACKAGE.rglob("*") if item.is_dir())):
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o555:
            raise ValidationError(f"package directory mode changed: {path}")


def verify_sources() -> None:
    sums = parse_sums(PACKAGE / "source-bindings.sha256")
    closure = load(PACKAGE / "source-closure.json")
    if (
        closure.get("member_count") != len(sums)
        or closure.get("source_bindings_sha256") != digest(PACKAGE / "source-bindings.sha256")
        or closure.get("total_bytes") != sum((ROOT / name).stat().st_size for name in sums)
    ):
        raise ValidationError("source closure metadata changed")
    for relative, expected in sums.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValidationError(f"source binding changed: {relative}")


def verify_attempt_0004() -> None:
    proof = load(PACKAGE / "attempt-0004-preservation.json")
    for root in proof["roots"]:
        records = root["files"]
        if len(records) != root["file_count"]:
            raise ValidationError("attempt-0004 preservation count changed")
        for record in records:
            path = ROOT / record["path"]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != record["bytes"]
                or stat.S_IMODE(path.stat().st_mode) != record["mode"]
                or digest(path) != record["sha256"]
            ):
                raise ValidationError(f"attempt-0004 evidence changed: {record['path']}")
        tree = hashlib.sha256(
            (
                json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        ).hexdigest()
        if tree != root["tree_sha256"]:
            raise ValidationError("attempt-0004 preservation tree changed")


def exact_processes(commands: dict[str, Any]) -> list[int]:
    expected = {tuple(commands["package_launch_argv"]), tuple(commands["model_argv"])}
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = tuple(
                part.decode(errors="surrogateescape")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if argv in expected:
            matches.append(int(entry.name))
    return matches


def verify_contract() -> None:
    package = load(PACKAGE / "package.json")
    commands = load(PACKAGE / "commands.json")
    bindings = load(PACKAGE / "execution-bindings.json")
    storage = load(PACKAGE / "storage-contract.json")
    rtl = load(PACKAGE / "rtl-contract.json")
    compile_only = load(PACKAGE / "rtl-compile.json")
    if (
        package.get("attempt") != "0005"
        or package.get("status") != "PREPARED_AUTHORITY_ABSENT_NOT_EXECUTED"
        or package.get("future_authority_cardinality") != 1
        or package.get("current_authority_cardinality") != 0
        or package.get("future_runner_run_id") is not None
        or package.get("attempt_0004_inheritance") != "NONE"
        or commands.get("submitted") is not False
        or commands.get("submission_entrypoint")
        != f"bash {PACKAGE.relative_to(ROOT).as_posix()}/submit_and_capture.sh"
        or "--resume" in commands["model_argv"]
        or commands["model_argv"][-1] != package["future_output"]
        or sha256_file(PACKAGE / "prompt.txt") != bindings["prompt"]["sha256"]
        or bindings["backend"]["kind"] != "LOCAL_VERILATOR_ACE2_SHELL_RUNTIME"
        or bindings["quantization"]["profile"] != "W4A8_GROUPED_SCALE32"
        or storage["immutable_shared_objects"]["per_attempt_copy_bytes"] != 0
        or storage["runtime_loaded_vectors"]["bounded_by_bytes"]
        != load(PACKAGE / "capacity.json")["runtime_output_max_bytes"]
        or rtl["public_top"] != "ace2_shell"
        or rtl["parameter_count"] != 14
        or rtl["port_count"] != 64
        or compile_only["exit_code"] != 0
        or compile_only["simulation_executed"] is not False
    ):
        raise ValidationError("attempt-0005 execution contract changed")
    package_text = (PACKAGE / "package.json").read_text(encoding="utf-8")
    for forbidden in ("manager-authority.json", "execution-consumed.json", "submit-receipt.json"):
        if (PACKAGE / forbidden).exists():
            raise ValidationError("forbidden authority or consumption artifact exists")
    if "runtime-output-0004" in package_text or "attempt-0004-authority-state" in package_text:
        raise ValidationError("attempt-0005 inherits attempt-0004 runtime namespace")


def verify_capacity() -> None:
    proof = load(PACKAGE / "capacity.json")
    available = os.statvfs(PACKAGE.parent).f_bavail * os.statvfs(PACKAGE.parent).f_frsize
    if proof["required_available_bytes"] != (
        proof["runtime_output_max_bytes"]
        + proof["temporary_write_max_bytes"]
        + proof["package_max_bytes"]
        + proof["safety_margin_bytes"]
    ):
        raise ValidationError("capacity arithmetic changed")
    if available < proof["required_available_bytes"]:
        raise ValidationError("live capacity no longer meets frozen preflight threshold")


def verify_zero_state() -> None:
    package = load(PACKAGE / "package.json")
    commands = load(PACKAGE / "commands.json")
    output = ROOT / package["future_output"]
    state = ROOT / package["future_authority_state"]
    registry = ROOT / ".argus_subagents" / f"{package['future_task_id']}.json"
    review = ROOT / package["future_review_receipt_namespace"]
    if (
        os.path.lexists(output)
        or os.path.lexists(state)
        or os.path.lexists(registry)
        or os.path.lexists(review)
        or exact_processes(commands)
    ):
        raise ValidationError("attempt-0005 runtime or authority state is nonzero")


def sha256_file(path: Path) -> str:
    return digest(path)


def validate(
    *,
    require_zero_state: bool = True,
    verify_preserved_attempt: bool = True,
) -> None:
    verify_package_closure()
    verify_sources()
    verify_contract()
    verify_capacity()
    if verify_preserved_attempt:
        verify_attempt_0004()
    if require_zero_state:
        verify_zero_state()


if __name__ == "__main__":
    validate()
    print("ATTEMPT_0005_PREPARATION_VALID_AUTHORITY_ZERO_NOT_EXECUTED")
'''
    return source.encode("utf-8")


def tests_source() -> bytes:
    source = r'''from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return loaded


runner = load_module("_attempt0005_runner", PACKAGE / "terminal_runner.py")
validator = load_module("_attempt0005_validator", PACKAGE / "validate_package.py")


class PreparationTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(prefix=".attempt0005-test-", dir=PACKAGE.parent)

    def test_static_package_closure_capacity_zero_state_and_attempt4_preservation(self) -> None:
        validator.validate()

    def test_success_nonzero_and_exactly_once_terminal_publication(self) -> None:
        for fixture, expected, classification in (
            ("success", 0, "FIXTURE_SUCCESS"),
            ("nonzero", 7, "FIXTURE_NONZERO"),
        ):
            with self.temporary() as name:
                state = Path(name) / "state"
                self.assertEqual(runner.execute_fixture(fixture, state), expected)
                terminal = json.loads((state / "terminal-status.json").read_text())
                self.assertEqual(terminal["classification"], classification)
                with self.assertRaises(FileExistsError):
                    runner.publish(state / "terminal-status.json", terminal)

    def test_sighup_and_sigterm_are_terminal_and_child_group_is_stopped(self) -> None:
        for number in (signal.SIGHUP, signal.SIGTERM):
            with self.temporary() as name:
                state = Path(name) / "state"
                driver = (
                    "import importlib.util,pathlib,sys;"
                    f"p=pathlib.Path({str(PACKAGE / 'terminal_runner.py')!r});"
                    "s=importlib.util.spec_from_file_location('r',p);"
                    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                    f"raise SystemExit(m.execute_fixture('sleep',pathlib.Path({str(state)!r})))"
                )
                process = subprocess.Popen([sys.executable, "-B", "-c", driver], cwd=runner.ROOT)
                deadline = time.monotonic() + 10
                while not (state / "child.started.json").is_file():
                    self.assertIsNone(process.poll())
                    if time.monotonic() >= deadline:
                        self.fail("inert fixture child did not start")
                    time.sleep(0.02)
                child = json.loads((state / "child.started.json").read_text())
                process.send_signal(number)
                self.assertEqual(process.wait(timeout=10), 128 + number)
                terminal = json.loads((state / "terminal-status.json").read_text())
                self.assertEqual(terminal["classification"], "FIXTURE_SIGNAL")
                with self.assertRaises((FileNotFoundError, ProcessLookupError)):
                    runner.process_start_ticks(child["pid"])

    def test_primary_terminal_publication_failure_uses_distinct_fsync_fallback(self) -> None:
        with self.temporary() as name:
            state = Path(name) / "state"

            def fail_primary(path: Path, value: object) -> None:
                if path == state / "terminal-status.json":
                    raise OSError("injected primary publication failure")
                runner.publish(path, value)

            self.assertEqual(
                runner.execute_fixture("nonzero", state, publisher=fail_primary),
                7,
            )
            fallback = json.loads(
                (state / "fallback/terminal-status.json").read_text()
            )
            self.assertEqual(fallback["primary_publication_failure"], "OSError")
            self.assertEqual(fallback["classification"], "FIXTURE_NONZERO")

    def test_capacity_fails_closed_below_exact_threshold(self) -> None:
        with self.assertRaisesRegex(runner.GateError, "insufficient capacity"):
            runner.check_capacity(
                available_bytes=runner.CAPACITY["required_available_bytes"] - 1
            )

    def test_launch_waits_for_runner_receipt_after_static_gates(self) -> None:
        source = (PACKAGE / "terminal_runner.py").read_text()
        submit = (PACKAGE / "submit_and_capture.sh").read_text()
        self.assertIn("manager = wait_for_receipt()", source)
        self.assertIn("terminal_runner.py\" --preflight", submit)
        self.assertLess(submit.index("terminal_runner.py\" --preflight"), submit.index("submit-intent.json"))
        self.assertLess(submit.index("argus_skill.tools.subagent submit"), submit.index("submit-receipt.json"))

    def test_pid_reuse_recovery_is_terminal_without_reuse(self) -> None:
        with self.temporary() as name:
            state = Path(name) / "state"
            state.mkdir()
            runner.publish(
                state / "execution-consumed.json",
                {
                    "supervisor_pid": os.getpid(),
                    "supervisor_start_ticks": runner.process_start_ticks(os.getpid()) + 1,
                    "boot_id": runner.boot_id(),
                },
            )
            self.assertEqual(
                runner.recover_host_loss(state),
                "INTERRUPTED_PROCESS_ABSENT_OR_PID_REUSED",
            )
            with self.assertRaises(runner.GateError):
                runner.recover_host_loss(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''
    return source.encode("utf-8")


def main() -> int:
    if os.path.lexists(PACKAGE):
        raise PreparationError(f"fresh package path already exists: {PACKAGE}")
    nonce = secrets.token_hex(16)
    short = nonce[:8]
    task_id = f"ace2-stage1-official-rtl-chat-20260828-0005-{short}"
    identity = f"ace2-stage1-attempt-0005-{nonce}"
    output = (
        "build/ace2_chat_demo/"
        f"stage1-official-rtl-backed-chat-demo-20260828T162359Z-runtime-output-0005-{short}"
    )
    authority_state = (
        "build/ace2_chat_demo/"
        f"stage1-official-rtl-backed-chat-demo-20260828T162359Z-attempt-0005-{short}-authority-state"
    )
    review_receipt = (
        "build/ace2_chat_demo/"
        f"stage1-official-rtl-backed-chat-demo-20260828T162359Z-attempt-0005-{short}-l2-review"
    )
    collisions = [
        path
        for path in (
            ROOT / output,
            ROOT / authority_state,
            ROOT / review_receipt,
            ROOT / ".argus_subagents" / f"{task_id}.json",
        )
        if os.path.lexists(path)
    ]
    if collisions:
        raise PreparationError(f"fresh identity collision: {collisions[0]}")

    source_members = source_paths()
    source_records = [file_record(path) for path in source_members]
    source_sums = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in source_records
    ).encode("ascii")
    attempt4_roots = []
    for path in ATTEMPT_0004_ROOTS:
        records = tree_records(path)
        attempt4_roots.append(
            {
                "path": relative(path),
                "file_count": len(records),
                "total_bytes": sum(int(item["bytes"]) for item in records),
                "tree_sha256": sha256_bytes(canonical_bytes(records)),
                "files": records,
            }
        )

    header = exact_module_header(ROOT / "rtl/ace2_shell.sv", "ace2_shell")
    parameters = re.findall(r"\bparameter\s+integer\s+([A-Za-z_][A-Za-z0-9_]*)", header)
    ports = re.findall(
        r"\b(?:input|output|inout)\s+(?:wire|reg)?\s*(?:signed\s*)?"
        r"(?:\[[^\]]+\]\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:,|\))",
        header,
    )
    if len(parameters) != 14 or len(ports) != 64:
        raise PreparationError(
            f"unexpected ace2_shell contract: {len(parameters)} parameters, {len(ports)} ports"
        )

    usage = os.statvfs(PACKAGE.parent)
    available = usage.f_bavail * usage.f_frsize
    base = RUNTIME_MAX_BYTES + TEMPORARY_MAX_BYTES + PACKAGE_MAX_BYTES
    margin = math.ceil(base * SAFETY_MARGIN_PERCENT / 100)
    required = base + margin
    if available < required:
        raise PreparationError(f"insufficient live capacity: {available} < {required}")
    PACKAGE.mkdir(parents=True, mode=0o755)

    model_records = {
        name: file_record(MODEL_SNAPSHOT / name) for name in MODEL_FILES
    }
    image_record = file_record(IMAGE_ROOT / "full_model_image.bin")
    image_contract_record = file_record(IMAGE_ROOT / "image_contract_v2.json")
    runtime_record = file_record(RUNTIME_BINARY)
    now = datetime.now(UTC).isoformat()
    model_argv = [
        "python3",
        "-B",
        "tools/ace2_chat_demo.py",
        "--prompt-file",
        relative(PACKAGE / "prompt.txt"),
        "--max-new-tokens",
        "4",
        "--output",
        output,
    ]
    launch_argv = ["bash", relative(PACKAGE / "launch.sh")]
    submit_argv = [
        "${ARGUS_SKILL_PYTHON:-python3}",
        "-m",
        "argus_skill.tools.subagent",
        "submit",
        "--task-id",
        task_id,
        "--mode",
        "direct",
        "--timeout",
        "172800",
        "--command",
        f"bash {relative(PACKAGE / 'launch.sh')}",
    ]
    write(PACKAGE / "prompt.txt", PROMPT)
    write_json(
        PACKAGE / "package.json",
        {
            "schema": "ace2-stage1-attempt-preparation-v2",
            "attempt": "0005",
            "identity": PACKAGE.name,
            "package_run_identity": identity,
            "nonce": nonce,
            "created_at_utc": now,
            "repo_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip(),
            "git_tree_claim": "DIRTY_TREE_EXECUTION_RELEVANT_BYTES_BOUND_NOT_CLEAN",
            "stage": "STAGE1_VERIFICATION_PREPARATION",
            "status": "PREPARED_AUTHORITY_ABSENT_NOT_EXECUTED",
            "future_task_id": task_id,
            "future_runner_run_id": None,
            "future_runner_run_id_template": rf"^{re.escape(task_id)}-[0-9]+$",
            "future_output": output,
            "future_authority_state": authority_state,
            "future_review_receipt_namespace": review_receipt,
            "future_authority_cardinality": 1,
            "current_authority_cardinality": 0,
            "execution_limit_if_separately_authorized": 1,
            "preparation_revision": "r1",
            "supersedes_preparation_tree_root": (
                "337fb3f83880f978d19a04cbcd31cae47e53a188ce2ac7b2b07905ec853efef9"
            ),
            "supersession_reason": (
                "PREEXECUTION_RECEIPT_ORDER_DEFECT_NO_MODEL_OR_RTL_EXECUTED"
            ),
            "attempt_0004_inheritance": "NONE",
            "attempt_0004_retry_replay_resume": "PERMANENTLY_FORBIDDEN",
            "advance_to_stage": "",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
            "execution_occurred": False,
            "prohibitions": [
                "Manager authority creation in this package",
                "submission intent or consumption in this package",
                "attempt-0004 copy, inheritance, retry, replay, or resume",
                "model execution or RTL simulation during preparation",
                "software fallback",
                "SSH or S7",
                "Stage-2 or U280",
                "proprietary toolchain or deployment",
            ],
        },
    )
    write_json(
        PACKAGE / "commands.json",
        {
            "schema": "ace2-attempt-0005-future-command-v1",
            "package_launch_argv": launch_argv,
            "model_argv": model_argv,
            "durable_submit_argv": submit_argv,
            "durable_submit_shell": (
                f'"${{ARGUS_SKILL_PYTHON:-python3}}" -m argus_skill.tools.subagent '
                f"submit --task-id {task_id} --mode direct --timeout 172800 "
                f"--command 'bash {relative(PACKAGE / 'launch.sh')}'"
            ),
            "check_with": (
                f'"${{ARGUS_SKILL_PYTHON:-python3}}" -m argus_skill.tools.subagent '
                f"status --task-id {task_id}"
            ),
            "submitted": False,
            "run_id_assigned": None,
            "submission_entrypoint": (
                f"bash {relative(PACKAGE / 'submit_and_capture.sh')}"
            ),
        },
    )
    write_json(
        PACKAGE / "execution-bindings.json",
        {
            "schema": "ace2-attempt-0005-execution-bindings-v1",
            "prompt": {
                "path": relative(PACKAGE / "prompt.txt"),
                "bytes": len(PROMPT),
                "sha256": sha256_bytes(PROMPT),
                "max_new_tokens": 4,
                "input_cardinality": 1,
            },
            "evaluator": file_record(ROOT / "tools/ace2_stage1_chat_product.py"),
            "model": {
                "repository": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "files": model_records,
            },
            "tokenizer": {
                "repository": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "tokenizer_json": model_records["tokenizer.json"],
                "tokenizer_config": model_records["tokenizer_config.json"],
                "chat_template_sha256": (
                    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
                ),
            },
            "quantization": {
                "profile": "W4A8_GROUPED_SCALE32",
                "packed_weight_encoding": "SIGNED_INT4_TWO_PER_BYTE",
                "activation_encoding": "SIGNED_INT8",
                "image": image_record,
                "image_contract": image_contract_record,
            },
            "backend": {
                "kind": "LOCAL_VERILATOR_ACE2_SHELL_RUNTIME",
                "binary": runtime_record,
                "public_top": "ace2_shell",
                "runtime_top": "ace2_shell_runtime_harness",
                "streaming_memory_boundary": "ABSTRACT_REQUEST_RESPONSE_INTERFACE",
                "software_fallback": "FORBIDDEN",
            },
        },
    )
    write_json(
        PACKAGE / "storage-contract.json",
        {
            "schema": "ace2-attempt-0005-storage-v1",
            "immutable_shared_objects": {
                "addressing": "SHA256",
                "objects": [
                    {
                        "object_id": f"sha256:{model_records['model.safetensors']['sha256']}",
                        "source": model_records["model.safetensors"],
                        "access": "HASH_VERIFIED_READ_ONLY_REFERENCE",
                    },
                    {
                        "object_id": f"sha256:{image_record['sha256']}",
                        "source": image_record,
                        "access": "HASH_VERIFIED_READ_ONLY_REFERENCE",
                    },
                ],
                "per_attempt_copy_bytes": 0,
                "links_into_attempt_0004": False,
            },
            "runtime_loaded_vectors": {
                "policy": "GENERATE_ONCE_PER_RUNTIME_POSITION_UNDER_FRESH_OUTPUT",
                "loading": "RTL_READMEMH_FROM_RUNTIME_PATHS",
                "package_vector_duplication": False,
                "attempt_0004_vector_reuse": False,
                "bounded_by_bytes": RUNTIME_MAX_BYTES,
            },
            "growth_policy": "RUNNER_TERMINATES_CHILD_GROUP_AND_SEALS_FAILURE_ABOVE_BOUND",
        },
    )
    write_json(
        PACKAGE / "capacity.json",
        {
            "schema": "ace2-attempt-0005-capacity-proof-v1",
            "measured_at_utc": now,
            "filesystem_total_bytes": usage.f_blocks * usage.f_frsize,
            "filesystem_available_bytes": available,
            "filesystem_block_bytes": usage.f_frsize,
            "runtime_output_max_bytes": RUNTIME_MAX_BYTES,
            "temporary_write_max_bytes": TEMPORARY_MAX_BYTES,
            "package_max_bytes": PACKAGE_MAX_BYTES,
            "safety_margin_percent": SAFETY_MARGIN_PERCENT,
            "safety_margin_bytes": margin,
            "required_available_bytes": required,
            "headroom_after_reserve_bytes": available - required,
            "shared_model_and_image_bytes": (
                int(model_records["model.safetensors"]["bytes"])
                + int(image_record["bytes"])
            ),
            "shared_object_incremental_bytes": 0,
            "projection_basis": "ENFORCED_RUNNER_CEILING_PLUS_TEMPORARY_AND_PACKAGE_ALLOWANCES",
            "launch_policy": "FAIL_CLOSED_BELOW_REQUIRED_OR_ON_DEVICE_CHANGE",
        },
    )
    write_json(
        PACKAGE / "zero-state.json",
        {
            "schema": "ace2-attempt-0005-zero-state-proof-v1",
            "observed_at_utc": now,
            "future_output_absent": True,
            "future_authority_state_absent": True,
            "future_review_receipt_absent": True,
            "future_runner_registry_absent": True,
            "future_run_id": None,
            "matching_live_processes": [],
            "current_authority_cardinality": 0,
            "future_identity_cardinality": 1,
            "submission_intent_absent": True,
            "consumption_absent": True,
            "model_execution": "NOT_EXECUTED",
            "rtl_simulation": "NOT_EXECUTED",
        },
    )
    write_json(
        PACKAGE / "attempt-0004-preservation.json",
        {
            "schema": "ace2-attempt-0004-byte-preservation-baseline-v1",
            "observed_at_utc": now,
            "classification": "EXTERNALLY_INTERRUPTED_HOST_POWEROFF_SEALED_NO_REPLAY",
            "access": "READ_ONLY_HASH_OBSERVATION",
            "copy_or_inheritance": "NONE",
            "roots": attempt4_roots,
        },
    )
    write(PACKAGE / "source-bindings.sha256", source_sums)
    write_json(
        PACKAGE / "source-closure.json",
        {
            "schema": "ace2-complete-current-execution-source-closure-v2",
            "method": (
                "recursive AST tools import closure, all RTL/testbench/Verilator "
                "sources, protected public contracts, model/tokenizer/W4A8 image, "
                "and exact local runtime binary"
            ),
            "member_count": len(source_records),
            "python_member_count": sum(
                str(item["path"]).endswith(".py") for item in source_records
            ),
            "rtl_member_count": sum(
                str(item["path"]).endswith((".sv", ".svh")) for item in source_records
            ),
            "total_bytes": sum(int(item["bytes"]) for item in source_records),
            "source_bindings_sha256": sha256_bytes(source_sums),
        },
    )
    write_json(
        PACKAGE / "rtl-contract.json",
        {
            "schema": "ace2-attempt-0005-public-rtl-contract-v1",
            "public_top": "ace2_shell",
            "source": file_record(ROOT / "rtl/ace2_shell.sv"),
            "header_sha256": sha256_bytes(header.encode("utf-8")),
            "header": header,
            "parameter_count": len(parameters),
            "parameters": parameters,
            "port_count": len(ports),
            "ports": ports,
            "non_sram_area_cap_mm2": 2.0,
            "clock_floor_mhz": 100,
            "interface_ambiguity": None,
        },
    )
    write_json(PACKAGE / "rtl-compile.json", compile_rtl_contract())
    write_json(
        PACKAGE / "toolchain.json",
        {
            "schema": "ace2-attempt-0005-toolchain-v1",
            "python": tool_version(["python3", "--version"]),
            "verilator": tool_version(["verilator", "--version"]),
            "iverilog": tool_version(["iverilog", "-V"]),
            "python_packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "safetensors", "tokenizers", "transformers")
            },
            "proprietary_toolchain_used": False,
            "container_runtime_used": False,
        },
    )
    write_json(
        PACKAGE / "review-contract.json",
        {
            "schema": "ace2-attempt-0005-independent-review-request-v1",
            "required_role": "reviewer",
            "required_level": "L2",
            "mode": "READ_ONLY",
            "decision_required": "ACCEPT_OR_REJECT",
            "bind_exactly": [
                "TREE_ROOT.sha256",
                "SHA256SUMS",
                "source-bindings.sha256",
                "attempt-0004-preservation.json",
            ],
            "receipt_namespace": review_receipt,
            "receipt_currently_absent": True,
            "manager_authority": "ABSENT_AND_OUT_OF_SCOPE",
        },
    )
    write(
        PACKAGE / "launch.sh",
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"cd {ROOT}\n"
            f"exec python3 -B {relative(PACKAGE / 'terminal_runner.py')}\n"
        ).encode("utf-8"),
        0o555,
    )
    write(
        PACKAGE / "submit_and_capture.sh",
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"cd {ROOT}\n"
            f'P="{relative(PACKAGE)}"\n'
            f'S="{authority_state}"\n'
            'python3 -B "$P/terminal_runner.py" --preflight\n'
            'python3 -B "$P/receipt_io.py" reserve "$S/submit-intent.json"\n'
            'receipt="$("${ARGUS_SKILL_PYTHON:-python3}" -m '
            "argus_skill.tools.subagent submit "
            f"--task-id {task_id} --mode direct --timeout 172800 "
            f"--command 'bash {relative(PACKAGE / 'launch.sh')}')\"\n"
            'printf \'%s\\n\' "$receipt" | python3 -B "$P/receipt_io.py" '
            'receipt "$S/submit-receipt.json"\n'
            'printf \'%s\\n\' "$receipt"\n'
        ).encode("utf-8"),
        0o555,
    )
    write(PACKAGE / "receipt_io.py", receipt_io_source(), 0o555)
    write(PACKAGE / "terminal_runner.py", runner_source(), 0o555)
    write(PACKAGE / "validate_package.py", validator_source(), 0o555)
    write(
        PACKAGE / "tests/test_preparation.py",
        tests_source(),
        0o555,
    )

    package_files = sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    )
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(PACKAGE).as_posix()}\n"
        for path in package_files
    ).encode("ascii")
    write(PACKAGE / "SHA256SUMS", sums)
    write(
        PACKAGE / "TREE_ROOT.sha256",
        f"{sha256_bytes(sums)}  SHA256SUMS\n".encode("ascii"),
    )
    for path in PACKAGE.rglob("*"):
        if path.is_dir():
            path.chmod(0o555)
    PACKAGE.chmod(0o555)
    print(
        json.dumps(
            {
                "package": relative(PACKAGE),
                "tree_root": sha256_bytes(sums),
                "task_id": task_id,
                "nonce": nonce,
                "future_output": output,
                "future_authority_state": authority_state,
                "available_bytes": available,
                "required_available_bytes": required,
                "model_argv": model_argv,
                "executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
