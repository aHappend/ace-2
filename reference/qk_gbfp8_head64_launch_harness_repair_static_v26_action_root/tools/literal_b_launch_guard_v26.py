#!/usr/bin/env python3
"""Literal ``-B`` launch proof shared by the sealed V26 entry points."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


class LaunchProofError(RuntimeError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise LaunchProofError(code)


def _proc_cmdline() -> list[str]:
    path = Path("/proc/self/cmdline")
    _require(path.is_file() and not path.is_symlink(), "LITERAL_B_PROC_CMDLINE_MISSING")
    raw = path.read_bytes()
    _require(raw.endswith(b"\0"), "LITERAL_B_PROC_CMDLINE_MALFORMED")
    fields = raw.split(b"\0")[:-1]
    _require(bool(fields) and all(fields), "LITERAL_B_PROC_CMDLINE_MALFORMED")
    try:
        return [field.decode("utf-8", errors="strict") for field in fields]
    except UnicodeDecodeError as error:
        raise LaunchProofError("LITERAL_B_PROC_CMDLINE_DECODE") from error


def _resolved_matches(token: str, expected_script: Path) -> bool:
    if not token or token.startswith("-"):
        return False
    try:
        return Path(token).expanduser().resolve(strict=False) == expected_script
    except (OSError, RuntimeError):
        return False


def validate_launch_evidence(
    *,
    expected_script: Path,
    orig_argv: list[str],
    proc_argv: list[str],
    executable: str,
    environment: dict[str, str],
    pycache_prefix: str | None,
    early_dont_write_bytecode: bool,
    current_dont_write_bytecode: bool,
) -> dict[str, Any]:
    expected_script = expected_script.resolve(strict=True)
    _require(bool(orig_argv) and all(type(token) is str and token for token in orig_argv), "LITERAL_B_ORIG_ARGV_MALFORMED")
    _require(orig_argv == proc_argv, "LITERAL_B_ARGV_DISAGREEMENT")
    _require(Path(orig_argv[0]).resolve(strict=True) == Path(executable).resolve(strict=True), "LITERAL_B_INTERPRETER_DISAGREEMENT")

    script_indexes = [index for index, token in enumerate(orig_argv[1:], start=1) if _resolved_matches(token, expected_script)]
    _require(bool(script_indexes), "LITERAL_B_SCRIPT_PATH_MISSING")
    _require(len(script_indexes) == 1, "LITERAL_B_SCRIPT_PATH_AMBIGUOUS")
    script_index = script_indexes[0]
    interpreter_tokens = orig_argv[1:script_index]
    for token in interpreter_tokens:
        _require(token not in {"-", "-c", "-m"} and not token.startswith("-c") and not token.startswith("-m"), "LITERAL_B_SCRIPT_PATH_NOT_INTERPRETER_RESOLVED")
        _require(token == "--" or token.startswith("-"), "LITERAL_B_NON_OPTION_BEFORE_SCRIPT")
    _require("-B" in interpreter_tokens, "LITERAL_B_FLAG_MISSING_BEFORE_SCRIPT")

    _require(environment.get("PYTHONDONTWRITEBYTECODE") == "1", "LITERAL_B_ENV_MISSING")
    _require(environment.get("PYTHONPYCACHEPREFIX") in {None, ""} and pycache_prefix is None, "LITERAL_B_PYCACHEPREFIX_NON_NULL")
    _require(early_dont_write_bytecode, "LITERAL_B_DONT_WRITE_BYTECODE_LATE")
    _require(current_dont_write_bytecode, "LITERAL_B_DONT_WRITE_BYTECODE_FALSE")
    return {
        "argv_sources_agree": True,
        "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
        "interpreter_option_tokens": interpreter_tokens,
        "literal_b_before_resolved_script": True,
        "proc_cmdline_inspected": True,
        "script_path_occurrence_count": 1,
        "sys_orig_argv_inspected": True,
    }


def validate_current_process(expected_script: Path, early_dont_write_bytecode: bool) -> dict[str, Any]:
    original = getattr(sys, "orig_argv", None)
    _require(type(original) is list, "LITERAL_B_ORIG_ARGV_MISSING")
    return validate_launch_evidence(
        expected_script=expected_script,
        orig_argv=list(original),
        proc_argv=_proc_cmdline(),
        executable=sys.executable,
        environment=dict(os.environ),
        pycache_prefix=sys.pycache_prefix,
        early_dont_write_bytecode=early_dont_write_bytecode,
        current_dont_write_bytecode=sys.dont_write_bytecode is True,
    )


def _inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": info.st_size})
        else:
            raise LaunchProofError("LITERAL_B_EFFECTIVENESS_UNSUPPORTED_ENTRY")
    return records


def prove_effective_suppression() -> dict[str, Any]:
    _require(sys.dont_write_bytecode is True, "LITERAL_B_DONT_WRITE_BYTECODE_FALSE")
    with tempfile.TemporaryDirectory(prefix="ace2-v26-literal-b-probe-") as directory:
        root = Path(directory)
        source = root / "suppression_probe.py"
        source.write_bytes(b"VALUE = 25\n")
        before = _inventory(root)
        name = "ace2_v26_bytecode_suppression_probe"
        spec = importlib.util.spec_from_file_location(name, source)
        _require(spec is not None and spec.loader is not None, "LITERAL_B_EFFECTIVENESS_IMPORT_SPEC")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        after = _inventory(root)
        _require(after == before, "LITERAL_B_SUPPRESSION_INEFFECTIVE")
    return {"import_probe_exact_tree_match": True, "sys_dont_write_bytecode": True}
