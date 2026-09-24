#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0019 exclusively."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path("/home/argustest/ace-2")
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0018"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0019"
DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
SOURCE_ROOT = "f2419fb98c58255fd2d86a1142f497e42d82e295efda2851b633efc9778402c8"
ENGINEER_MISSION_ID = "4341527f33f4"
DIRECTIVE_REVISION = "999497e7567e4d1a9a751775239f1cd0"
DIRECTIVE_SHA256 = "059f18166aaddaff10703fa59fa1766d33b5cd18c24cbdacc00f2e96b55d7290"
DIRECTIVE_OBJECTIVE_SHA256 = (
    "43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"
)
SOURCE_MEMBERS = {
    "execute_layer16.py",
    "hostile_controls.py",
    "review_package.py",
    "seal_package.py",
    "validate_nonexecuting.py",
    "validate_review.py",
}
SEALED_MEMBERS = SOURCE_MEMBERS | {
    "acceptance-contract.json",
    "authority.json",
    "failed-predecessor-provenance.json",
    "hostile-controls.json",
    "review-request.json",
}


class PreparationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, value: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_once(text: str, old: str, new: str) -> str:
    require(text.count(old) == 1, f"source fragment count changed: {old[:80]!r}")
    return text.replace(old, new)


def validate_sealed_source() -> None:
    require(
        SOURCE.is_dir()
        and not SOURCE.is_symlink()
        and stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "sealed package0018 directory changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "sealed package0018 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SOURCE).as_posix()
        for item in SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records) == SEALED_MEMBERS, "package0018 members changed")
    for name, digest in records.items():
        path = SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"sealed package0018 member changed: {path}",
        )
    require(
        sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT
        and stat.S_IMODE((SOURCE / "SHA256SUMS").stat().st_mode) == 0o444
        and stat.S_IMODE((SOURCE / "TREE_ROOT.sha256").stat().st_mode) == 0o444
        and (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [SOURCE_ROOT, "SHA256SUMS"],
        "sealed package0018 root changed",
    )
    for path in (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0018",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0018",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0018",
    ):
        require(not os.path.lexists(path), f"package0018 failure state changed: {path}")


def validate_inputs() -> None:
    validate_sealed_source()
    require(DIRECTIVE.is_file() and not DIRECTIVE.is_symlink(), "Manager V8 absent")
    directive = json.loads(DIRECTIVE.read_text(encoding="utf-8"))
    text = directive.get("text")
    require(
        isinstance(directive, dict)
        and directive.get("revision") == DIRECTIVE_REVISION
        and directive.get("objective_sha256") == DIRECTIVE_OBJECTIVE_SHA256
        and isinstance(text, str)
        and "MANAGER GRANT LAYER16 V8" in text
        and "package0019" in text
        and "layer16-package0019-reviewer" in text
        and "agent_type=general-purpose" in text
        and "mode=sync" in text
        and "exactly one launch" in text
        and "Zero workload" in text,
        "active Manager V8 directive changed",
    )
    require(sha256_file(DIRECTIVE) == DIRECTIVE_SHA256, "Manager V8 bytes changed")
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0019",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0019",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0019",
    ):
        require(not os.path.lexists(path), f"package0019 target preexists: {path}")


def transform_executor(text: str) -> str:
    require("0019" not in text, "package0018 source already contains package0019")
    text = text.replace("0018", "0019")
    text = replace_once(
        text,
        'ENGINEER_MISSION_ID = "f1a82b0e164e"',
        f'ENGINEER_MISSION_ID = "{ENGINEER_MISSION_ID}"',
    )
    text = replace_once(
        text,
        'ACTIVE_DIRECTIVE_REVISION = "16f8501da0d748e3bac140c6088bac46"',
        f'ACTIVE_DIRECTIVE_REVISION = "{DIRECTIVE_REVISION}"',
    )
    text = replace_once(
        text,
        '"3f0a1a997a88c38c210b57fa3fe121d097f640ec387d57e5b83e7a8eaf8c350d"',
        f'"{DIRECTIVE_SHA256}"',
    )
    predecessor = '''    "0017": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0017",
        "09aefc03902c4ea11378e2b9be3b948d2573d73a27c268c2a3e97b38294df2df",
        "HOST_READ_AGENT_REQUIRED_MODEL_FIELD",
    ),
'''
    return replace_once(
        text,
        predecessor,
        predecessor
        + '''    "0018": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0018",
        "f2419fb98c58255fd2d86a1142f497e42d82e295efda2851b633efc9778402c8",
        "HOST_SUPPORTED_TASK_NAME_ARGUMENT_MISMATCH",
    ),
''',
    )


def transformed_sources() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in sorted(SOURCE_MEMBERS):
        source = (SOURCE / name).read_text(encoding="utf-8")
        if name == "execute_layer16.py":
            transformed = transform_executor(source)
        else:
            require("0019" not in source, f"package0019 token preexists in {name}")
            transformed = source.replace("0018", "0019")
        compile(transformed, str(TARGET / name), "exec")
        require("package0018-reviewer" not in transformed, f"stale task token in {name}")
        result[name] = transformed
    require(
        'REVIEW_TASK_NAME = "layer16-package0019-reviewer"'
        in result["execute_layer16.py"],
        "exact package0019 review task token absent",
    )
    return result


def main() -> int:
    validate_inputs()
    sources = transformed_sources()
    os.mkdir(TARGET, mode=0o700)
    for name, text in sources.items():
        write_exclusive(TARGET / name, text.encode("utf-8"))
    result = subprocess.run(
        [
            "/home/argustest/miniconda3/bin/python3.13",
            "-B",
            str(TARGET / "seal_package.py"),
        ],
        cwd=ROOT,
        env={
            "LC_ALL": "C",
            "PATH": "/home/argustest/miniconda3/bin:/usr/bin:/bin",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONSAFEPATH": "1",
            "PYTHONUTF8": "1",
        },
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        result.returncode == 0,
        "package0019 seal failed: " + result.stderr.decode("utf-8", "replace"),
    )
    validate_sealed_source()
    sys.stdout.buffer.write(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
