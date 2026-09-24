#!/usr/bin/env python3
"""Build and validate the cache-free attempt-0014 continuation successor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/ace2_chat_demo"
STEM = "stage1-official-rtl-backed-chat-demo-20260831T134355Z"
IDENTITY = f"{STEM}-attempt-0014-preparation-r5-continuation-repair-0005"
DESTINATION = BUILD / IDENTITY
VALIDATION = BUILD / f"{IDENTITY}-validation"
REVIEW_NAMESPACE = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0005-authority-review"
)
AUTHORITY_NAMESPACE = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0005-authority-state"
)
CONSUMPTION_NAMESPACE = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0005-consumption-state"
)
RUNTIME_NAMESPACE = BUILD / (
    f"{STEM}-runtime-output-0014-r5-continuation-repair-0005"
)

ATTEMPT_0012_TERMINAL = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "attempt-0012-6f2c8d4e-authority-state/terminal-status.json"
)
ATTEMPT_0013_PREPARATION = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-preparation-r4"
)
ATTEMPT_0013_TERMINAL = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-a62e0c91-authority-state/terminal-status.json"
)
R5 = BUILD / f"{STEM}-attempt-0014-preparation-r5"
REPAIR_0002 = BUILD / (
    f"{STEM}-attempt-0014-preparation-r5-continuation-repair-0002"
)
REPAIR_0003 = BUILD / (
    f"{STEM}-attempt-0014-preparation-r5-continuation-repair-0003"
)
REPAIR_0004_FAILED = BUILD / (
    f"{STEM}-attempt-0014-preparation-r5-continuation-repair-0004"
)
REJECTED_AUTHORITY = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0003-authority-state"
)
REJECTION_RECEIPT = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0003-authority-review/"
    "fresh-review.json"
)

EXPECTED_SEALS = {
    R5: ("082a999c7e4f6acbce1672ab6988710db695ff9ad9233aaa1e710d5d508c1ee2", 19),
    REPAIR_0002: (
        "eca23105aa44e4489de20f73125f77dc98acaff4f575a588ccb32118cf798bca",
        34,
    ),
    REPAIR_0003: (
        "0baed352fa192aff3578d8e5f8b7eac39b2ad42726fb0f41fbfbc032f54a03c9",
        34,
    ),
    REPAIR_0004_FAILED: (
        "1c5a68ab9b9c9ea5bc39464398923d45e0220951504b8328028084c5b296330a",
        30,
    ),
    REJECTED_AUTHORITY: (
        "0ab1a018d97d1106605f2e319112fec26e00749f648069ebcfde7f4d3337637f",
        4,
    ),
}
EXPECTED_TERMINALS = {
    ATTEMPT_0012_TERMINAL: (
        "41943596b1065c887b1fbb638f413b639f13271b9618206151a5d438394e110a"
    ),
    ATTEMPT_0013_TERMINAL: (
        "1ae9c3988259155394eab35471f9234f44558309c894d8aa628dedb11a228d25"
    ),
}
EXPECTED_REJECTION = (
    "e25bdaf7861e15b93a1e90663cc45eff7120ed1ad0ab5eef6f84bda68e285f53"
)
OLD_IDENTITY = (
    f"{STEM}-attempt-0014-preparation-r5-continuation-repair-0003"
)
OLD_IDENTITY_FRAGMENT = (
    "attempt-0014-preparation-r5-continuation-repair-0003"
)
NEW_IDENTITY_FRAGMENT = (
    "attempt-0014-preparation-r5-continuation-repair-0005"
)
OLD_PREDECESSOR_ROOT = EXPECTED_SEALS[REPAIR_0002][0]
NEW_PREDECESSOR_ROOT = EXPECTED_SEALS[REPAIR_0003][0]
SEAL_FILES = {"SHA256SUMS", "TREE_ROOT.sha256"}


class ClosureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any, mode: int = 0o444) -> None:
    path.write_bytes(canonical_bytes(value))
    path.chmod(mode)


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"invalid checksum record: {line!r}")
        assert match is not None
        require(match[2] not in records, f"duplicate checksum member: {match[2]}")
        records[match[2]] = match[1]
    return records


def authenticate_sealed_tree(
    path: Path, expected_root: str, expected_count: int
) -> None:
    require(path.is_dir() and not path.is_symlink(), f"sealed tree absent: {path}")
    records = parse_sums(path / "SHA256SUMS")
    require(len(records) == expected_count, f"sealed member count changed: {path}")
    require(
        sha256_file(path / "SHA256SUMS") == expected_root,
        f"sealed root changed: {path}",
    )
    require(
        (path / "TREE_ROOT.sha256").read_text(encoding="ascii").strip()
        == f"{expected_root}  SHA256SUMS",
        f"sealed root sidecar changed: {path}",
    )
    for relative, expected in records.items():
        member = path / relative
        require(
            member.is_file()
            and not member.is_symlink()
            and sha256_file(member) == expected,
            f"sealed member changed: {member}",
        )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_record(path: Path, *, base: Path = ROOT) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "bytes": resolved.stat().st_size,
        "mode": stat.S_IMODE(resolved.stat().st_mode),
        "path": path.relative_to(base).as_posix(),
        "resolved_path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def all_file_tree_record(path: Path) -> dict[str, Any]:
    records = []
    for item in sorted(path.rglob("*")):
        require(not item.is_symlink(), f"predecessor symlink is not accepted: {item}")
        if item.is_file():
            records.append(
                {
                    "bytes": item.stat().st_size,
                    "mode": stat.S_IMODE(item.stat().st_mode),
                    "path": item.relative_to(path).as_posix(),
                    "sha256": sha256_file(item),
                }
            )
    return {
        "file_count": len(records),
        "path": relative(path),
        "records_root_sha256": sha256_bytes(canonical_bytes(records)),
    }


def predecessor_bindings() -> dict[str, Any]:
    for path, (root, count) in EXPECTED_SEALS.items():
        authenticate_sealed_tree(path, root, count)
    for path, expected in EXPECTED_TERMINALS.items():
        require(
            path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == expected,
            f"terminal receipt changed: {path}",
        )
        require(
            load(path).get("status") == "SEALED_TERMINAL_NO_RETRY",
            f"terminal status changed: {path}",
        )
    require(
        REJECTION_RECEIPT.is_file()
        and not REJECTION_RECEIPT.is_symlink()
        and sha256_file(REJECTION_RECEIPT) == EXPECTED_REJECTION,
        "attempt-0014 rejection receipt changed",
    )
    rejection = load(REJECTION_RECEIPT)
    require(
        rejection.get("decision") == "REJECT"
        and rejection.get("verdict") == "REJECTED_NOT_CONSUMABLE_FROM_LAYER_11"
        and rejection["package_integrity"]["execution_relevant_closure"] == "FAIL"
        and rejection["salvage_boundary"]["first_incomplete_unit"]
        == "position-00/layer-11",
        "attempt-0014 rejection semantics changed",
    )
    bytecode = rejection["package_integrity"]["unsealed_execution_relevant_files"]
    require(
        len(bytecode) == 2
        and all((ROOT / item["path"]).is_file() for item in bytecode)
        and all(
            sha256_file(ROOT / item["path"]) == item["sha256"] for item in bytecode
        ),
        "rejected active bytecode evidence changed",
    )
    return {
        "attempt_0012": {
            "terminal_receipt": file_record(ATTEMPT_0012_TERMINAL),
            "status": "SEALED_TERMINAL_NO_RETRY",
        },
        "attempt_0013": {
            "preparation": all_file_tree_record(ATTEMPT_0013_PREPARATION),
            "terminal_receipt": file_record(ATTEMPT_0013_TERMINAL),
            "status": "SEALED_TERMINAL_NO_RETRY",
        },
        "attempt_0014_r5": {
            "sealed_member_count": EXPECTED_SEALS[R5][1],
            "sealed_tree_root_sha256": EXPECTED_SEALS[R5][0],
            **all_file_tree_record(R5),
        },
        "repair_0002": {
            "sealed_member_count": EXPECTED_SEALS[REPAIR_0002][1],
            "sealed_tree_root_sha256": EXPECTED_SEALS[REPAIR_0002][0],
            **all_file_tree_record(REPAIR_0002),
        },
        "repair_0003": {
            "active_unsealed_bytecode": bytecode,
            "sealed_member_count": EXPECTED_SEALS[REPAIR_0003][1],
            "sealed_tree_root_sha256": EXPECTED_SEALS[REPAIR_0003][0],
            **all_file_tree_record(REPAIR_0003),
        },
        "repair_0004_failed_closure": {
            "sealed_member_count": EXPECTED_SEALS[REPAIR_0004_FAILED][1],
            "sealed_tree_root_sha256": EXPECTED_SEALS[REPAIR_0004_FAILED][0],
            "status": "FAILED_ISOLATED_SOURCE_CLOSURE_VALIDATION_IMMUTABLE",
            **all_file_tree_record(REPAIR_0004_FAILED),
        },
        "rejected_authority": {
            "sealed_member_count": EXPECTED_SEALS[REJECTED_AUTHORITY][1],
            "sealed_tree_root_sha256": EXPECTED_SEALS[REJECTED_AUTHORITY][0],
            **all_file_tree_record(REJECTED_AUTHORITY),
        },
        "rejection_receipt": file_record(REJECTION_RECEIPT),
    }


SOURCE_LAUNCH = r'''#!/usr/bin/env python3
"""Isolated source-only loader for the sealed ACE-2 continuation runner."""

from __future__ import annotations

import hashlib
import encodings.ascii
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
SEAL_FILES = {"SHA256SUMS", "TREE_ROOT.sha256"}


class LaunchGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchGateError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        require(
            separator == "  "
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and relative
            and relative not in records,
            f"invalid checksum record: {line!r}",
        )
        records[relative] = digest
    return records


def validate_package_files(manifest: dict[str, Any]) -> str:
    sums = parse_sums(PACKAGE / "SHA256SUMS")
    actual: set[str] = set()
    executable: list[str] = []
    for directory, names, filenames in os.walk(PACKAGE, followlinks=False):
        root = Path(directory)
        for name in names:
            path = root / name
            require(not path.is_symlink(), f"package directory symlink: {path}")
            require(name != "__pycache__", f"bytecode cache directory: {path}")
        for name in filenames:
            path = root / name
            relative = path.relative_to(PACKAGE).as_posix()
            require(path.is_file() and not path.is_symlink(), f"non-regular member: {path}")
            require(path.suffix not in {".pyc", ".pyo"}, f"bytecode member: {path}")
            if relative not in SEAL_FILES:
                actual.add(relative)
            if stat.S_IMODE(path.stat().st_mode) & 0o111:
                executable.append(relative)
    require(actual == set(sums), "sealed package closure changed")
    require(executable == ["source_launch.py"], "alternate executable path present")
    for relative, expected in sums.items():
        require(
            sha256_file(PACKAGE / relative) == expected,
            f"sealed member changed: {relative}",
        )
    root = sha256_file(PACKAGE / "SHA256SUMS")
    require(
        (PACKAGE / "TREE_ROOT.sha256").read_text(encoding="ascii").strip()
        == f"{root}  SHA256SUMS",
        "authority root sidecar changed",
    )
    cache = PACKAGE / manifest["launch"]["empty_bytecode_cache"]
    require(
        cache.is_dir()
        and not cache.is_symlink()
        and not any(cache.iterdir()),
        "redirected bytecode cache is not an empty real directory",
    )
    return root


def loaded_runtime_records() -> tuple[set[str], set[str], dict[str, str]]:
    builtins: set[str] = set()
    frozen: set[str] = set()
    files: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if name == "__main__" or module is None:
            continue
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if origin == "built-in":
            builtins.add(name)
        elif origin == "frozen":
            frozen.add(name)
        elif isinstance(origin, str):
            path = Path(origin)
            require(path.suffix not in {".pyc", ".pyo"}, f"loaded bytecode: {path}")
            cached = getattr(spec, "cached", None)
            require(
                cached is None or not Path(cached).exists(),
                f"selectable module bytecode exists: {cached}",
            )
            resolved = str(path.resolve(strict=True))
            files[resolved] = sha256_file(Path(resolved))
    return builtins, frozen, files


def validate_runtime(runtime: dict[str, Any]) -> None:
    builtins, frozen, files = loaded_runtime_records()
    expected_files = {
        item["resolved_path"]: item["sha256"]
        for item in runtime["python_module_files"]
    }
    require(
        builtins == set(runtime["builtin_modules"]),
        "loaded built-in module closure changed",
    )
    require(
        frozen == set(runtime["frozen_modules"]),
        "loaded frozen module closure changed",
    )
    require(files == expected_files, "loaded source/native module closure changed")


def source_load_runner(manifest: dict[str, Any]) -> dict[str, Any]:
    binding = manifest["execution_members"]["continuation_runner"]
    path = PACKAGE / binding["path"]
    require(
        path.is_file()
        and not path.is_symlink()
        and stat.S_IMODE(path.stat().st_mode) & 0o111 == 0
        and sha256_file(path) == binding["sha256"],
        "bound continuation runner source changed",
    )
    source = path.read_bytes()
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(path),
        "__name__": "ace2_bound_continuation_runner",
        "__package__": None,
    }
    exec(compile(source, str(path), "exec", dont_inherit=True), namespace)
    require(
        callable(namespace.get("run_continuation"))
        and namespace.get("CONTINUATION_IDENTITY") == manifest["identity"],
        "bound continuation callable or identity changed",
    )
    return namespace


def main() -> None:
    manifest = load(PACKAGE / "successor-manifest.json")
    authority = load(PACKAGE / "authority.json")
    runtime = load(PACKAGE / "runtime-closure.json")
    require(
        sys.flags.isolated == 1
        and sys.flags.safe_path is True
        and sys.flags.no_site == 1
        and sys.flags.dont_write_bytecode == 1,
        "launch is not isolated, safe-path, no-site, and bytecode-disabled",
    )
    require(
        sys.pycache_prefix == str(PACKAGE / manifest["launch"]["empty_bytecode_cache"]),
        "bytecode cache redirection changed",
    )
    executable = Path(sys.executable)
    require(
        executable.is_file()
        and not executable.is_symlink()
        and str(executable) == runtime["interpreter"]["resolved_path"]
        and sha256_file(executable) == runtime["interpreter"]["sha256"],
        "bound non-symlink interpreter changed",
    )
    root = validate_package_files(manifest)
    require(
        authority["candidate_authority_root"]["source"] == "TREE_ROOT.sha256"
        and authority["consumable"] is False
        and authority["status"] == "PENDING_FRESH_REVIEW",
        "pending authority gate changed",
    )
    source_load_runner(manifest)
    validate_runtime(runtime)
    require(
        manifest["first_incomplete_unit"] == "position-00/layer-11"
        and manifest["selectable_unit_executor"] is None,
        "layer-11 boundary or no-executor gate changed",
    )
    if sys.argv[1:] == ["--verify-only"]:
        print(
            "ACE2_REPAIR_0005_SOURCE_CLOSURE_PASS "
            "first_incomplete=position-00/layer-11 consumable=false"
        )
        return
    raise LaunchGateError(
        "successor is unconsumable pending Fresh Reviewer approval and a later "
        "separately sealed unit-executor authority"
    )


if __name__ == "__main__":
    main()
'''


RUNTIME_PROBE = r'''from __future__ import annotations
import hashlib
import encodings.ascii
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

runner = Path(sys.argv[1])
source = runner.read_bytes()
namespace = {
    "__builtins__": __builtins__,
    "__file__": str(runner),
    "__name__": "ace2_bound_continuation_runner",
    "__package__": None,
}
exec(compile(source, str(runner), "exec", dont_inherit=True), namespace)
builtins = []
frozen = []
files = {}
for name, module in sorted(sys.modules.items()):
    if name == "__main__" or module is None:
        continue
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin == "built-in":
        builtins.append(name)
    elif origin == "frozen":
        frozen.append(name)
    elif isinstance(origin, str):
        path = Path(origin)
        if path.suffix in {".pyc", ".pyo"}:
            raise RuntimeError(f"bytecode selected: {path}")
        cached = getattr(spec, "cached", None)
        if cached is not None and Path(cached).exists():
            raise RuntimeError(f"selectable bytecode exists: {cached}")
        resolved = path.resolve(strict=True)
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        files[str(resolved)] = {
            "module_names": [],
            "resolved_path": str(resolved),
            "sha256": digest,
            "suffix": resolved.suffix,
        }
        files[str(resolved)]["module_names"].append(name)
print(json.dumps({
    "builtin_modules": builtins,
    "frozen_modules": frozen,
    "python_module_files": list(files.values()),
}, sort_keys=True))
'''


def ldd_paths(path: Path) -> list[Path]:
    completed = subprocess.run(
        ["ldd", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    paths: list[Path] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        require("not found" not in text, f"unresolved native dependency: {text}")
        candidate: str | None = None
        if "=>" in text:
            right = text.split("=>", 1)[1].strip()
            if right.startswith("/"):
                candidate = right.split(" ", 1)[0]
        elif text.startswith("/"):
            candidate = text.split(" ", 1)[0]
        if candidate is not None:
            paths.append(Path(candidate))
    return paths


def native_dependency_closure(initial: list[Path]) -> list[dict[str, Any]]:
    pending = list(initial)
    visited: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    while pending:
        requested = pending.pop()
        resolved = requested.resolve(strict=True)
        key = str(resolved)
        if key in visited:
            continue
        visited.add(key)
        records[key] = {
            "requested_paths": [str(requested)],
            "resolved_path": key,
            "sha256": sha256_file(resolved),
        }
        for dependency in ldd_paths(resolved):
            target = dependency.resolve(strict=True)
            target_key = str(target)
            if target_key in records:
                records[target_key]["requested_paths"] = sorted(
                    set(records[target_key]["requested_paths"] + [str(dependency)])
                )
            else:
                pending.append(dependency)
    return [records[key] for key in sorted(records)]


def build_runtime_closure(package: Path, interpreter: Path) -> dict[str, Any]:
    cache = package / "bytecode-cache"
    with tempfile.TemporaryDirectory(prefix=".ace2-repair-0005-probe-") as temporary:
        probe = Path(temporary) / "probe.py"
        probe.write_text(RUNTIME_PROBE, encoding="utf-8")
        completed = subprocess.run(
            [
                str(interpreter),
                "-I",
                "-B",
                "-S",
                "-X",
                f"pycache_prefix={cache}",
                str(probe),
                str(package / "continuation_runner.py"),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    require(not any(cache.iterdir()), "runtime probe populated bytecode cache")
    modules = json.loads(completed.stdout)
    native_modules = [
        Path(item["resolved_path"])
        for item in modules["python_module_files"]
        if item["suffix"] not in {".py", ".pyw"}
    ]
    return {
        "schema": "ace2-python-source-runtime-closure-v1",
        "interpreter": {
            **file_record(interpreter, base=Path("/")),
            "resolved_path": str(interpreter),
            "version": subprocess.run(
                [str(interpreter), "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ).stdout.strip(),
        },
        "launch_flags": [
            "-I",
            "-B",
            "-S",
            "-X",
            f"pycache_prefix={cache}",
        ],
        "bytecode_policy": {
            "cache_directory_empty": True,
            "cache_reads": "REDIRECTED_TO_SEALED_EMPTY_DIRECTORY",
            "cache_writes": "DISABLED_BY_-B",
            "package_bytecode_members": 0,
        },
        **modules,
        "native_dynamic_libraries": native_dependency_closure(
            [interpreter, *native_modules]
        ),
    }


def copy_successor_inputs(temporary: Path) -> None:
    temporary.mkdir(parents=True)
    (temporary / "bytecode-cache").mkdir()
    checkpoint = load(REPAIR_0003 / "checkpoint-manifest.json")
    checkpoint["continuation_identity"] = {
        "package": IDENTITY,
        "sealed_repair_predecessor_tree_root": NEW_PREDECESSOR_ROOT,
    }
    checkpoint["source_bindings"]["sealed_repair_predecessor"] = {
        "member_count": EXPECTED_SEALS[REPAIR_0003][1],
        "path": relative(REPAIR_0003),
        "status": "SEALED_REPAIR_0003_AUTHENTICATED_UNCHANGED",
        "tree_root": NEW_PREDECESSOR_ROOT,
    }
    write_json(temporary / "checkpoint-manifest.json", checkpoint)

    runner = (REPAIR_0003 / "continuation_runner.py").read_text(encoding="utf-8")
    require(
        runner.count(OLD_IDENTITY_FRAGMENT) == 1
        and runner.count(OLD_PREDECESSOR_ROOT) == 1,
        "repair-0003 runner identity constants changed",
    )
    runner = runner.replace(
        OLD_IDENTITY_FRAGMENT, NEW_IDENTITY_FRAGMENT
    ).replace(OLD_PREDECESSOR_ROOT, NEW_PREDECESSOR_ROOT)
    (temporary / "continuation_runner.py").write_text(runner, encoding="utf-8")
    (temporary / "continuation_runner.py").chmod(0o444)
    (temporary / "source_launch.py").write_text(SOURCE_LAUNCH, encoding="utf-8")
    (temporary / "source_launch.py").chmod(0o555)

    shutil.copytree(
        REPAIR_0003 / "salvage-state",
        temporary / "salvage-state",
        copy_function=shutil.copyfile,
    )
    for item in (temporary / "salvage-state").rglob("*"):
        if item.is_file():
            item.chmod(0o444)


def execution_member_records(package: Path) -> dict[str, Any]:
    salvage = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(package).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted((package / "salvage-state").rglob("*"))
        if path.is_file()
    ]
    return {
        "checkpoint": {
            "path": "checkpoint-manifest.json",
            "sha256": sha256_file(package / "checkpoint-manifest.json"),
        },
        "continuation_runner": {
            "load_policy": "READ_BYTES_COMPILE_EXEC_EXACT_BOUND_SOURCE",
            "path": "continuation_runner.py",
            "sha256": sha256_file(package / "continuation_runner.py"),
        },
        "salvage_state": salvage,
        "source_launch": {
            "path": "source_launch.py",
            "sha256": sha256_file(package / "source_launch.py"),
        },
    }


def seal_package(package: Path) -> str:
    records = []
    for path in sorted(package.rglob("*")):
        require(not path.is_symlink(), f"successor symlink: {path}")
        if path.is_file() and path.name not in SEAL_FILES:
            records.append(
                f"{sha256_file(path)}  {path.relative_to(package).as_posix()}"
            )
    sums = package / "SHA256SUMS"
    sums.write_text("\n".join(records) + "\n", encoding="ascii")
    sums.chmod(0o444)
    root = sha256_file(sums)
    sidecar = package / "TREE_ROOT.sha256"
    sidecar.write_text(f"{root}  SHA256SUMS\n", encoding="ascii")
    sidecar.chmod(0o444)
    return root


def package_files(package: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    executable: list[str] = []
    for directory, names, filenames in os.walk(package, followlinks=False):
        root = Path(directory)
        for name in names:
            path = root / name
            require(not path.is_symlink(), f"successor directory symlink: {path}")
            require(name != "__pycache__", f"successor bytecode cache: {path}")
        for name in filenames:
            path = root / name
            relative_path = path.relative_to(package).as_posix()
            require(path.is_file() and not path.is_symlink(), f"non-file: {path}")
            require(path.suffix not in {".pyc", ".pyo"}, f"bytecode file: {path}")
            if relative_path not in SEAL_FILES:
                files.add(relative_path)
            if stat.S_IMODE(path.stat().st_mode) & 0o111:
                executable.append(relative_path)
    return files, sorted(executable)


def validate_successor(bindings: dict[str, Any]) -> dict[str, Any]:
    package = DESTINATION
    records = parse_sums(package / "SHA256SUMS")
    actual, executable = package_files(package)
    require(actual == set(records), "successor sealed closure changed")
    require(executable == ["source_launch.py"], "alternate executable path present")
    for name, expected in records.items():
        require(
            sha256_file(package / name) == expected,
            f"successor member changed: {name}",
        )
    root = sha256_file(package / "SHA256SUMS")
    require(
        (package / "TREE_ROOT.sha256").read_text(encoding="ascii").strip()
        == f"{root}  SHA256SUMS",
        "successor authority root sidecar changed",
    )
    manifest = load(package / "successor-manifest.json")
    authority = load(package / "authority.json")
    checkpoint = load(package / "checkpoint-manifest.json")
    runtime = load(package / "runtime-closure.json")
    require(
        manifest["identity"] == IDENTITY
        and manifest["predecessor_bindings"] == bindings
        and manifest["first_incomplete_unit"] == "position-00/layer-11"
        and manifest["selectable_unit_executor"] is None
        and manifest["unmanifested_executable_selection"] == "IMPOSSIBLE",
        "successor manifest changed",
    )
    require(
        checkpoint["completed_unit_keys"]
        == [f"position-00/layer-{index:02d}" for index in range(11)]
        and checkpoint["next_incomplete_unit"] == "position-00/layer-11"
        and checkpoint["continuation_identity"]["package"] == IDENTITY
        and checkpoint["continuation_identity"][
            "sealed_repair_predecessor_tree_root"
        ]
        == NEW_PREDECESSOR_ROOT
        and checkpoint["validation"]["status"] == "PASS",
        "authenticated layer-10 checkpoint boundary changed",
    )
    require(
        authority["status"] == "PENDING_FRESH_REVIEW"
        and authority["consumable"] is False
        and authority["candidate_authority_root"]["source"]
        == "TREE_ROOT.sha256"
        and authority["first_incomplete_unit"] == "position-00/layer-11"
        and set(authority["activity"].values()) == {0},
        "successor authority or zero-activity state changed",
    )
    for namespace in (
        REVIEW_NAMESPACE,
        AUTHORITY_NAMESPACE,
        CONSUMPTION_NAMESPACE,
        RUNTIME_NAMESPACE,
    ):
        require(
            not os.path.lexists(namespace),
            f"future state namespace already exists: {namespace}",
        )
    require(
        runtime["bytecode_policy"]["package_bytecode_members"] == 0
        and runtime["bytecode_policy"]["cache_directory_empty"] is True
        and not any((package / "bytecode-cache").iterdir())
        and all(
            item["suffix"] not in {".pyc", ".pyo"}
            for item in runtime["python_module_files"]
        )
        and runtime["native_dynamic_libraries"],
        "runtime source/native closure changed",
    )
    interpreter = Path(runtime["interpreter"]["resolved_path"])
    require(
        interpreter.is_file()
        and not interpreter.is_symlink()
        and sha256_file(interpreter) == runtime["interpreter"]["sha256"],
        "exact non-symlink interpreter changed",
    )
    for item in runtime["python_module_files"]:
        path = Path(item["resolved_path"])
        require(
            path.is_file()
            and sha256_file(path) == item["sha256"],
            f"runtime module changed: {path}",
        )
    for item in runtime["native_dynamic_libraries"]:
        path = Path(item["resolved_path"])
        require(
            path.is_file()
            and sha256_file(path) == item["sha256"],
            f"native runtime member changed: {path}",
        )
    command = [
        str(interpreter),
        "-I",
        "-B",
        "-S",
        "-X",
        f"pycache_prefix={package / 'bytecode-cache'}",
        str(package / "source_launch.py"),
        "--verify-only",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/",
        env={"PATH": "/usr/bin:/bin", "LANG": "C"},
    )
    require(
        completed.stdout.strip()
        == (
            "ACE2_REPAIR_0005_SOURCE_CLOSURE_PASS "
            "first_incomplete=position-00/layer-11 consumable=false"
        )
        and completed.stderr == "",
        "isolated source launch validation failed",
    )
    require(not any((package / "bytecode-cache").iterdir()), "bytecode was emitted")
    current_bindings = predecessor_bindings()
    require(current_bindings == bindings, "immutable predecessor bytes changed")
    return {
        "activity": authority["activity"],
        "alternate_executable_count": 0,
        "authority_root_sha256": root,
        "bytecode_member_count": 0,
        "candidate_consumable": False,
        "executable_member_count": 1,
        "first_incomplete_unit": "position-00/layer-11",
        "immutable_predecessor_authentication": "PASS",
        "isolated_source_launch": "PASS",
        "native_dynamic_library_count": len(
            runtime["native_dynamic_libraries"]
        ),
        "python_module_file_count": len(runtime["python_module_files"]),
        "rejection_receipt_authentication": "PASS",
        "selectable_unit_executor_count": 0,
        "status": "PASS",
        "unbound_execution_relevant_member_count": 0,
    }


def build() -> None:
    require(not DESTINATION.exists(), f"refusing to overwrite: {DESTINATION}")
    require(not VALIDATION.exists(), f"refusing to overwrite: {VALIDATION}")
    for namespace in (
        REVIEW_NAMESPACE,
        AUTHORITY_NAMESPACE,
        CONSUMPTION_NAMESPACE,
        RUNTIME_NAMESPACE,
    ):
        require(
            not os.path.lexists(namespace),
            f"future namespace is not fresh: {namespace}",
        )
    bindings = predecessor_bindings()
    interpreter = Path(sys.executable).resolve(strict=True)
    require(
        interpreter.is_file() and not interpreter.is_symlink(),
        "resolved Python interpreter must be a regular non-symlink file",
    )
    temporary = DESTINATION.with_name(DESTINATION.name + ".building")
    require(not temporary.exists(), f"stale construction path: {temporary}")
    try:
        copy_successor_inputs(temporary)
        runtime = build_runtime_closure(temporary, interpreter)
        runtime["launch_flags"] = [
            "-I",
            "-B",
            "-S",
            "-X",
            f"pycache_prefix={DESTINATION / 'bytecode-cache'}",
        ]
        write_json(temporary / "runtime-closure.json", runtime)
        members = execution_member_records(temporary)
        manifest = {
            "activity": {
                "authority_cardinality": 0,
                "consumption_cardinality": 0,
                "model_execution_cardinality": 0,
                "output_cardinality": 0,
                "rtl_execution_cardinality": 0,
                "submission_cardinality": 0,
            },
            "authenticated_pass_count": 11,
            "execution_members": members,
            "first_incomplete_unit": "position-00/layer-11",
            "identity": IDENTITY,
            "launch": {
                "argv": [
                    str(interpreter),
                    "-I",
                    "-B",
                    "-S",
                    "-X",
                    f"pycache_prefix={DESTINATION / 'bytecode-cache'}",
                    str(DESTINATION / "source_launch.py"),
                ],
                "empty_bytecode_cache": "bytecode-cache",
                "exact_interpreter_regular_non_symlink": True,
                "loader": "READ_BYTES_COMPILE_EXEC_EXACT_BOUND_SOURCE",
                "working_directory": "/",
            },
            "predecessor_bindings": bindings,
            "schema": "ace2-stage1-r5-executable-closure-successor-v1",
            "selectable_unit_executor": None,
            "self_binding": "SUCCESSOR_MANIFEST_BOUND_BY_AUTHORITY_ROOT_SHA256SUMS",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
            "status": "SEALED_PENDING_FRESH_REVIEW",
            "unmanifested_executable_selection": "IMPOSSIBLE",
        }
        write_json(temporary / "successor-manifest.json", manifest)
        review = {
            "acceptance_required_before_consumption": True,
            "checks": [
                "authenticate attempt-0014 rejection receipt",
                "authenticate attempt-0012, attempt-0013, attempt-0014 r5, repair-0002, and repair-0003",
                "verify exact isolated non-symlink interpreter and source-only loader",
                "verify redirected empty bytecode cache and zero package bytecode",
                "verify every loaded Python source/native module and dynamic library hash",
                "verify exactly one executable member and no symlink or alternate runner",
                "verify position-00/layer-11 is the first incomplete unit",
                "verify all activity cardinalities remain zero",
            ],
            "execution_permitted_during_review": False,
            "fresh_review_receipt_path": relative(
                REVIEW_NAMESPACE / "fresh-review.json"
            ),
            "schema": "ace2-stage1-r5-executable-closure-review-request-v1",
            "status": "PENDING_FRESH_REVIEW",
        }
        write_json(temporary / "review-contract.json", review)
        manifest_sha = sha256_file(temporary / "successor-manifest.json")
        authority = {
            "activity": manifest["activity"],
            "candidate_authority_root": {
                "algorithm": "SHA256_OF_SHA256SUMS",
                "source": "TREE_ROOT.sha256",
            },
            "consumable": False,
            "first_incomplete_unit": "position-00/layer-11",
            "fresh_reviewer_approval_present": False,
            "schema": "ace2-stage1-r5-executable-closure-authority-candidate-v1",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
            "status": "PENDING_FRESH_REVIEW",
            "successor_manifest_sha256": manifest_sha,
            "unit_execution_authority": "NOT_GRANTED",
        }
        write_json(temporary / "authority.json", authority)
        seal_package(temporary)
        os.replace(temporary, DESTINATION)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    root = sha256_file(DESTINATION / "SHA256SUMS")
    report_result = validate_successor(bindings)
    VALIDATION.mkdir(parents=True)
    report = {
        **report_result,
        "candidate_authority_root_sha256": root,
        "candidate_path": relative(DESTINATION),
        "command": (
            "python3 -B tools/build_attempt_0014_repair_0005_closure.py --check"
        ),
        "execution_performed": False,
        "model_executed": False,
        "rtl_executed": False,
        "schema": "ace2-stage1-r5-executable-closure-validation-v1",
        "successor_consumable": False,
        "validation_mode": "INERT_SOURCE_LOAD_ONLY",
        "validator_sha256": sha256_file(Path(__file__)),
    }
    write_json(VALIDATION / "executable-closure-validation.json", report)
    digest = sha256_file(VALIDATION / "executable-closure-validation.json")
    (VALIDATION / "executable-closure-validation.json.sha256").write_text(
        f"{digest}  executable-closure-validation.json\n", encoding="ascii"
    )
    (VALIDATION / "executable-closure-validation.json.sha256").chmod(0o444)
    print(
        "ACE2_REPAIR_0005_EXECUTABLE_CLOSURE_PASS "
        f"authority_root={report_result['authority_root_sha256']} "
        "first_incomplete=position-00/layer-11 "
        "authority=0 consumption=0 submission=0 model=0 rtl=0 output=0 "
        "consumable=false review=PENDING_FRESH_REVIEW"
    )


def check() -> None:
    require(DESTINATION.is_dir(), f"successor absent: {DESTINATION}")
    bindings = load(DESTINATION / "successor-manifest.json")[
        "predecessor_bindings"
    ]
    result = validate_successor(bindings)
    print(
        "ACE2_REPAIR_0005_EXECUTABLE_CLOSURE_PASS "
        f"authority_root={result['authority_root_sha256']} "
        "first_incomplete=position-00/layer-11 "
        "authority=0 consumption=0 submission=0 model=0 rtl=0 output=0 "
        "consumable=false review=PENDING_FRESH_REVIEW"
    )


def main() -> None:
    if sys.argv[1:] == ["--check"]:
        check()
    elif not sys.argv[1:]:
        build()
    else:
        raise ClosureError("usage: build_attempt_0014_repair_0005_closure.py [--check]")


if __name__ == "__main__":
    main()
