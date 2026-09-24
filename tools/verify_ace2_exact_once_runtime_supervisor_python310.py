#!/usr/bin/env python3
"""Decisive read-only verifier for the Python 3.10 supervisor repair.

This verifier never invokes or validates the retired v3 package/authority.  The
only child-start tests it runs use temporary harmless fake-child packages.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import ace2_exact_once_runtime_supervisor as supervisor  # noqa: E402


EXPECTED_FILE_HASHES = {
    ROOT
    / "research/raw/specification/instruct-rtl-diagnostic-exact-once-execution-package-20260809-v3.json": "d0721e064934b9e51501564e994bc6be3afecfa2aa339c28d38c360a2bfec469",
    ROOT
    / "build/freeze-instruct-rtl-diagnostic-v3-20260809.authority.json": "0bd0a252809915afc799f63d382c3b41f60d5987de676793df10651b3c413a4a",
    ROOT
    / "research/raw/specification/instruct-rtl-diagnostic-v3-authority-fresh-review-binding-20260809.json": "b663c5b8922da70c0d1a658393b1a490ab0d394efaacba10275eebd467edf701",
    ROOT
    / "research/raw/specification/instruct-rtl-diagnostic-exact-once-execution-package-20260809-v3.provenance.json": "c8c693d44d730a4e6ff1049c80a95a2a3a231690dee7a34a34fc81badb406e80",
    ROOT
    / "research/PIPELINE_STATE.json": "5a941e816d114438cd551e0405052e9686513931349c026728b47df715edbcba",
    ROOT
    / "tools/ace2_exact_once_runtime_supervisor.py": "e7c2c5ab3c44463e7546b2677236f0b05bff05ec99a9d803f23a94ae529c9619",
    ROOT
    / "tools/test_ace2_exact_once_runtime_supervisor.py": "8317a992d84ef99f4fa17fbc3893ee27f677d089d43dc88381181ea4a9e14595",
}

EXPECTED_TREES = {
    ROOT / "rtl": (26, 685737, "01bd731570bf149d6173a3705b288718b18a5eeb718596a5da741b38162876d4"),
    ROOT
    / "constraints": (1, 250, "324ed614b2153730a56d6f2ce1c1d1035e2ca16be0d7a107ce3ac2ce3ee74c96"),
    ROOT
    / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001-terminal-evidence": (
        12,
        22018,
        "23965a2ff7bd05ce4d4782a18c2a935e3769ffce482cc0a5b671a458b6ac9662",
    ),
}

V3_RUNTIME_PATHS = [
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.state",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.stdout.log",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.stderr.log",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.terminal.json",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809-runtime-output",
]

CONTAINMENT_DIR = (
    ROOT
    / "evidence/quarantine/execute-instruct-rtl-diagnostic-v3-terminal-no-execution-20260810T001420Z"
)
CONTAINMENT_SHA256 = "6f4f6e7c3317aca64f2a43dc865a798b317fa598a12949f4bfa75fcaf1e254bc"
ACE2_RUNTIME = (ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness").resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def live_runtime_pids() -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            executable = (entry / "exe").resolve(strict=True)
        except (FileNotFoundError, PermissionError):
            continue
        if executable == ACE2_RUNTIME:
            matches.append(int(entry.name))
    return sorted(matches)


def main() -> int:
    require(Path(sys.executable).resolve() == Path("/usr/bin/python3").resolve(), "wrong interpreter")
    require(sys.version_info[:3] == (3, 10, 12), "expected exact Python 3.10.12")

    unit = subprocess.run(
        [
            "/usr/bin/python3",
            "-m",
            "unittest",
            "-v",
            "tools/test_ace2_exact_once_runtime_supervisor.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    require(unit.returncode == 0, unit.stdout + unit.stderr)
    require("Ran 11 tests" in unit.stderr, "unexpected regression count")

    file_results: dict[str, str] = {}
    for path, expected in EXPECTED_FILE_HASHES.items():
        measured = sha256_file(path)
        require(measured == expected, f"file hash mismatch: {path}")
        file_results[str(path.relative_to(ROOT))] = measured

    tree_results: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_TREES.items():
        measured = supervisor.measure_source_tree(path)
        actual = (measured.file_count, measured.byte_count, measured.sha256)
        require(actual == expected, f"tree measurement mismatch: {path}")
        tree_results[str(path.relative_to(ROOT))] = {
            "file_count": measured.file_count,
            "byte_count": measured.byte_count,
            "sha256": measured.sha256,
        }

    pipeline = json.loads((ROOT / "research/PIPELINE_STATE.json").read_text(encoding="utf-8"))
    require(pipeline["current_stage"] == "specification", "pipeline stage changed")

    containment = CONTAINMENT_DIR / "containment.json"
    require(sha256_file(containment) == CONTAINMENT_SHA256, "containment hash mismatch")
    require((containment.stat().st_mode & 0o777) == 0o444, "containment is not read-only")
    checksum_fields = (CONTAINMENT_DIR / "containment.sha256").read_text(encoding="utf-8").split()
    require(checksum_fields == [CONTAINMENT_SHA256, "containment.json"], "bad checksum companion")

    absent = [str(path.relative_to(ROOT)) for path in V3_RUNTIME_PATHS if not os.path.lexists(path)]
    require(len(absent) == len(V3_RUNTIME_PATHS), "a retired v3 runtime path exists")
    runtime_pids = live_runtime_pids()
    require(not runtime_pids, "ACE-2 runtime process is live")

    schemas = {
        "package": supervisor.PACKAGE_SCHEMA,
        "authority": supervisor.AUTHORITY_SCHEMA,
        "intent": supervisor.INTENT_SCHEMA,
        "started": supervisor.STARTED_SCHEMA,
        "terminal": supervisor.TERMINAL_SCHEMA,
    }
    require(
        schemas
        == {
            "package": "ace2-exact-once-execution-package-v1",
            "authority": "ace2-exact-once-external-authority-v1",
            "intent": "ace2-exact-once-pre-start-intent-v1",
            "started": "ace2-exact-once-process-started-v1",
            "terminal": "ace2-exact-once-terminal-record-v1",
        },
        "public schema changed",
    )

    report = {
        "status": "PASS",
        "interpreter": {"path": "/usr/bin/python3", "version": "3.10.12"},
        "regressions": {
            "command": "/usr/bin/python3 -m unittest -v tools/test_ace2_exact_once_runtime_supervisor.py",
            "tests_run": 11,
            "result": "PASS",
            "child_scope": "temporary harmless fake child only",
        },
        "schemas": schemas,
        "file_hashes": file_results,
        "tree_hashes": tree_results,
        "pipeline_stage": pipeline["current_stage"],
        "containment": {
            "path": str(containment.relative_to(ROOT)),
            "sha256": CONTAINMENT_SHA256,
            "mode": "0444",
        },
        "v3_runtime_paths_absent": absent,
        "live_ace2_runtime_pids": runtime_pids,
        "ace2_runtime_process_start_count": 0,
        "v3_package_or_authority_invocations": 0,
        "creates_v4_package_or_authority": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
