#!/usr/bin/env python3
"""Run the Stage-1 setup path with every execution route hard-blocked."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DiagnosticError(RuntimeError):
    pass


class ExecutionRouteBlocked(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source_bindings(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in records:
            raise DiagnosticError("source binding manifest is malformed")
        records[relative] = digest
    return records


def context_limit_from_source(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MAX_CONTEXT_TOKENS"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value
    raise DiagnosticError("MAX_CONTEXT_TOKENS is absent from the bound backend source")


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise DiagnosticError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def block_execution_routes() -> list[str]:
    guarded = [
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "os.system",
        "os.exec*",
        "os.spawn*",
        "os.posix_spawn*",
        "simulator/runtime binaries",
        "package launch",
        "subagent submission",
    ]

    def blocked(*args: object, **_kwargs: object) -> NoReturn:
        target = repr(args[0]) if args else "<unknown>"
        raise ExecutionRouteBlocked(f"blocked process execution route: {target}")

    subprocess.run = blocked  # type: ignore[assignment]
    subprocess.Popen = blocked  # type: ignore[assignment,misc]
    subprocess.call = blocked  # type: ignore[assignment]
    subprocess.check_call = blocked  # type: ignore[assignment]
    subprocess.check_output = blocked  # type: ignore[assignment]
    for name in dir(os):
        if (
            name == "system"
            or name.startswith("exec")
            or name.startswith("spawn")
            or name.startswith("posix_spawn")
        ):
            setattr(os, name, blocked)
    return guarded


def write_report(path: Path, value: dict[str, Any]) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short diagnostic report write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def diagnose(package: Path, backend_source: Path) -> dict[str, Any]:
    package = package.resolve()
    backend_source = backend_source.resolve()
    commands = json.loads((package / "commands.json").read_text(encoding="utf-8"))
    package_contract = json.loads((package / "package.json").read_text(encoding="utf-8"))
    bindings = parse_source_bindings(package / "source-bindings.sha256")
    expected_backend_sha256 = bindings["tools/rtl_arbitrary_text_generation_backend.py"]
    observed_backend_sha256 = sha256_file(backend_source)
    if observed_backend_sha256 != expected_backend_sha256:
        raise DiagnosticError("diagnostic backend bytes do not match the package binding")
    argv = commands["model_argv"]
    if argv[:3] != ["python3", "-B", "tools/ace2_chat_demo.py"]:
        raise DiagnosticError("bound model argv entrypoint changed")
    output = ROOT / package_contract["future_output"]
    if os.path.lexists(output):
        raise DiagnosticError("diagnostic refuses an existing bound runtime output")

    guarded_routes = block_execution_routes()
    from tools import ace2_chat_demo as front
    from tools import ace2_stage1_chat_product as product
    from tools import run_rtl_arbitrary_text_generation as generation

    runner = load_module("_inert_terminal_runner", package / "terminal_runner.py")
    runner.check_capacity()
    context_limit = context_limit_from_source(backend_source)
    product.backend.MAX_CONTEXT_TOKENS = context_limit
    generation.MAX_CONTEXT_TOKENS = context_limit
    tokenization: dict[str, Any] = {}
    original_tokenizer_record = generation.tokenizer_record

    def capture_tokenization(*args: object, **kwargs: object) -> tuple[dict[str, Any], Any]:
        record, tokenizer = original_tokenizer_record(*args, **kwargs)
        tokenization.update(record)
        return record, tokenizer

    def block_backend(*_args: object, **_kwargs: object) -> NoReturn:
        raise ExecutionRouteBlocked("blocked Stage-1 model/RTL backend boundary")

    generation.tokenizer_record = capture_tokenization
    product.backend.run_generation = block_backend
    outcome: dict[str, Any]
    try:
        return_code = front.main(argv[3:])
    except BaseException as error:
        frames = traceback.extract_tb(error.__traceback__)
        outcome = {
            "kind": (
                "BACKEND_BOUNDARY_REACHED"
                if isinstance(error, ExecutionRouteBlocked)
                and str(error) == "blocked Stage-1 model/RTL backend boundary"
                else "PRE_RUNTIME_SETUP_EXCEPTION"
            ),
            "exception_type": type(error).__name__,
            "detail": str(error),
            "last_frame": (
                {
                    "path": Path(frames[-1].filename).resolve().relative_to(ROOT).as_posix(),
                    "line": frames[-1].lineno,
                    "function": frames[-1].name,
                }
                if frames and Path(frames[-1].filename).resolve().is_relative_to(ROOT)
                else None
            ),
        }
    else:
        outcome = {"kind": "RETURNED", "return_code": return_code}

    if os.path.lexists(output):
        raise DiagnosticError("inert diagnostic created the bound runtime output")
    return {
        "schema": "ace2-stage1-inert-setup-diagnostic-v1",
        "package_identity": package_contract["identity"],
        "package_run_identity": package_contract["package_run_identity"],
        "package_tree_root": (package / "TREE_ROOT.sha256").read_text(encoding="ascii").split()[0],
        "model_argv_sha256": hashlib.sha256(
            (json.dumps(argv, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "prompt_sha256": bindings.get(
            package_contract.get("prompt_path", ""),
            json.loads((package / "execution-bindings.json").read_text())["prompt"]["sha256"],
        ),
        "backend_source_sha256": observed_backend_sha256,
        "context_limit": context_limit,
        "capacity_checked": True,
        "guarded_routes": guarded_routes,
        "tokenization": {
            "prompt_token_count": tokenization.get("prompt_token_count"),
            "total_context_tokens": tokenization.get("generation_bounds", {}).get(
                "total_context_tokens"
            ),
        },
        "outcome": outcome,
        "runtime_output_absent_before_and_after": True,
        "model_executed": False,
        "rtl_simulator_executed": False,
        "package_launched": False,
        "subagent_submitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--backend-source", type=Path, required=True)
    parser.add_argument(
        "--expect",
        choices=("PRE_RUNTIME_SETUP_EXCEPTION", "BACKEND_BOUNDARY_REACHED"),
        required=True,
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = diagnose(args.package, args.backend_source)
    if report["outcome"]["kind"] != args.expect:
        raise DiagnosticError(
            f"expected {args.expect}, observed {report['outcome']['kind']}"
        )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_report(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
