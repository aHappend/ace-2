#!/usr/bin/python3
"""Derive the inert v7h schema-parity package from immutable v7g inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
from typing import Any


ROOT = Path("/home/argustest/ace-2")
SOURCE = ROOT / "build/s7-v7g-complete-lineage-fail-closed-successor-offline-20260811-d6d478c6d320"
DESTINATION = ROOT / "build/s7-v7h-authorization-schema-parity-successor-offline-20260811-15a4b0f6f6ff"
V7G_AUTH_ROOT = Path(
    "/home/argustest/.local/state/ace2/s7-v7g-authority-authorizations-20260811-b3c2f91a5d2c/"
    "mgr-s7-reservation-submit-v7g-20260811-bde35b37b5d2"
)
V7G_STATE_PARENT = Path(
    "/home/argustest/.local/state/ace2/s7-v7g-authority-construction-20260811-eeb711c65463"
)
V7G_ORIGIN = V7G_STATE_PARENT / (
    ".mgr-s7-reservation-submit-v7g-20260811-bde35b37b5d2."
    "s7-v7g-origin-exec-capsule-20260811-d4e7b630ea79.process-origin.json"
)
V7G_ATTEMPT = V7G_STATE_PARENT / "s7-v7g-authority-attempt-20260811-e753835dcc16"
V7G_FUTURE_AUTH_TOP = Path(
    "/home/argustest/.local/state/ace2/"
    "s7-v7g-reservation-execution-authorizations-20260811-3996be4b5aaf"
)
V7G_ACCEPTANCE = ROOT / "evidence/s7-v7g-fresh-l2-acceptance-20260811-d6d478c6d320.json"

V7H_AUTH_TOP = Path(
    "/home/argustest/.local/state/ace2/s7-v7h-authority-authorizations-20260811-815cfa247859"
)
V7H_STATE_PARENT = Path(
    "/home/argustest/.local/state/ace2/s7-v7h-authority-construction-20260811-9692cb00d847"
)
V7H_FUTURE_AUTH_TOP = Path(
    "/home/argustest/.local/state/ace2/"
    "s7-v7h-reservation-execution-authorizations-20260811-b5ea020b6cd6"
)

EXPECTED_V7G_HASHES = {
    SOURCE / "runnable-SHA256SUMS": "98c05c4f7a8be5ec022d976db40ae8a152fbd3af5daff49b607682b2f832f276",
    SOURCE / "construct_manager_authority_v7g.py": "6204992332bb46e5873418ec27b76552b42359e7134e479c188a999066545026",
    SOURCE / "launch_capsule_v7g.py": "6c4c1d25d03d33d763e6900965a5037045afcfb8787fa70472d18800c0d90262",
    SOURCE / "process_lifecycle_v7g.py": "0b7e8a873b0bb0ed01ed5a4473c22e94684a4ca28f898b3c42a399cac6a14d5c",
    SOURCE / "test_v7g_offline.py": "1ac3da1f659aa5bda1614144fe71fabfb515539126077cb8b5af8aaf9f1630ea",
    V7G_AUTH_ROOT / "constructor-authorization-source.json": "0112bd73d5359dc366a8dd5e6f69c9e6cd73362b6ed75dc1176d7c970fb28c1e",
    V7G_AUTH_ROOT / "launch-authorization-source.json": "1cf80018401c7ab70b0ef286fb9136dcec5f8f269d21ade8dbcc1c009eda173f",
    ROOT / "research/GROUND_TRUTH.md": "70cd8aae4376ceee1454a08d8e31ab1254b8409dfc29e554f3f902dabc8863d0",
}

REPLACEMENTS = {
    "s7-v7g-complete-lineage-fail-closed-successor-offline-20260811-d6d478c6d320":
        "s7-v7h-authorization-schema-parity-successor-offline-20260811-15a4b0f6f6ff",
    "s7-v7g-complete-lineage-fail-closed-successor-package-20260811-d6d478c6d320":
        "s7-v7h-authorization-schema-parity-successor-package-20260811-15a4b0f6f6ff",
    "mgr-s7-reservation-submit-v7g-20260811-bde35b37b5d2":
        "mgr-s7-reservation-submit-v7h-20260811-f9f25b9364d1",
    "s7-v7g-six-probe-constructor-20260811-3b805b85eb0b":
        "s7-v7h-six-probe-constructor-20260811-ee7877e7f66b",
    "s7-v7g-origin-exec-capsule-20260811-d4e7b630ea79":
        "s7-v7h-origin-exec-capsule-20260811-14209d0a3e25",
    "s7-v7g-authority-attempt-20260811-e753835dcc16":
        "s7-v7h-authority-attempt-20260811-c09746c58180",
    "s7-v7g-six-probe-budget-20260811-823e53f58987":
        "s7-v7h-six-probe-budget-20260811-6d38b02967db",
    "s7-v7g-authority-authorizations-20260811-b3c2f91a5d2c":
        "s7-v7h-authority-authorizations-20260811-815cfa247859",
    "s7-v7g-authority-construction-20260811-eeb711c65463":
        "s7-v7h-authority-construction-20260811-9692cb00d847",
    "s7-v7g-reservation-execution-authorizations-20260811-3996be4b5aaf":
        "s7-v7h-reservation-execution-authorizations-20260811-b5ea020b6cd6",
    "ace2-s7-rsv-v7g-20260811-a0c007dd1d5c":
        "ace2-s7-rsv-v7h-20260811-2480dffe1217",
    "owned-s7-2xg4-24h-v7g-20260811-65647629a3ac":
        "owned-s7-2xg4-24h-v7h-20260811-6d38b02967db",
    "s7-v7g-candidate-intent-20260811-7778656e37ab":
        "s7-v7h-candidate-intent-20260811-b5ea020b6cd6",
    "d9ea79503524": "2480dffe1217",
    "v7g": "v7h",
    "V7G": "V7H",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def assert_v7g_terminal_state() -> None:
    for path, expected in EXPECTED_V7G_HASHES.items():
        if sha(path) != expected:
            raise RuntimeError(f"immutable v7g hash mismatch: {path}")
    for source in V7G_AUTH_ROOT.iterdir():
        if stat.S_IMODE(source.stat().st_mode) != 0o400:
            raise RuntimeError(f"v7g source mode changed: {source}")
    if not V7G_STATE_PARENT.is_dir() or V7G_STATE_PARENT.is_symlink():
        raise RuntimeError("v7g state parent is not the preserved regular directory")
    if list(V7G_STATE_PARENT.iterdir()):
        raise RuntimeError("v7g state parent is no longer empty pre-origin evidence")
    for path in (V7G_ORIGIN, V7G_ATTEMPT, V7G_FUTURE_AUTH_TOP):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"prohibited v7g post-failure artifact exists: {path}")


def rewrite_tree() -> None:
    shutil.copytree(SOURCE, DESTINATION)
    paths = sorted(DESTINATION.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        if "v7g" in path.name:
            path.rename(path.with_name(path.name.replace("v7g", "v7h")))
    for path in sorted(item for item in DESTINATION.rglob("*") if item.is_file()):
        path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for old, new in REPLACEMENTS.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def bind_predecessor() -> None:
    acceptance_copy = DESTINATION / "accepted-v7g-fresh-l2-acceptance.json"
    write(acceptance_copy, V7G_ACCEPTANCE.read_bytes())
    state_stat = V7G_STATE_PARENT.stat()
    terminal = {
        "artifact_kind": "s7_v7g_terminal_pre_origin_authorization_failure_seal_for_v7h",
        "authorization_source_hashes": {
            "constructor": sha(V7G_AUTH_ROOT / "constructor-authorization-source.json"),
            "launch": sha(V7G_AUTH_ROOT / "launch-authorization-source.json"),
        },
        "classification": "AUTHORIZATION_FIELD_SET_MISMATCH_PRE_ORIGIN",
        "exact_traceback": (
            "launch_capsule_v7g.py main line 685 -> launch_once line 677 -> "
            "validate_authorization line 602 -> line 307 "
            "AuthorizationError('authorization field-set mismatch')"
        ),
        "failure_consumed_invocation": True,
        "hash_preflight": "PASSED_BEFORE_FAILURE",
        "invocation_utc": "20260811T135941Z",
        "lifecycle": {
            "attempt_namespace_created": False,
            "future_execution_authorization_created": False,
            "gate_process_invocations": 0,
            "manager_authority_created": False,
            "origin_record_created": False,
            "submission_process_invocations": 0,
        },
        "nonreplayable": True,
        "package_sha256sums_sha256": sha(SOURCE / "runnable-SHA256SUMS"),
        "parent_prerequisite": "PASSED_STATE_PARENT_CREATED",
        "preserved_code_hashes": {
            "capsule": sha(SOURCE / "launch_capsule_v7g.py"),
            "constructor": sha(SOURCE / "construct_manager_authority_v7g.py"),
            "process_lifecycle": sha(SOURCE / "process_lifecycle_v7g.py"),
            "test_suite": sha(SOURCE / "test_v7g_offline.py"),
        },
        "schema_version": 3,
        "state_parent": {
            "entry_count": 0,
            "mode": format(stat.S_IMODE(state_stat.st_mode), "04o"),
            "path": str(V7G_STATE_PARENT),
        },
        "status": "TERMINAL_NONREPLAYABLE_PRE_ORIGIN_NO_AUTHORITY",
    }
    terminal_path = DESTINATION / "v7g-terminal-pre-origin-failure-seal.json"
    write(terminal_path, canonical(terminal))
    ledger = {
        "artifact_kind": "s7_v7h_protected_v7g_terminal_input_hashes",
        "bindings": {
            str(ROOT / "research/GROUND_TRUTH.md"): sha(ROOT / "research/GROUND_TRUTH.md"),
            str(SOURCE / "runnable-SHA256SUMS"): sha(SOURCE / "runnable-SHA256SUMS"),
            str(SOURCE / "process_lifecycle_v7g.py"): sha(SOURCE / "process_lifecycle_v7g.py"),
            str(SOURCE / "test_v7g_offline.py"): sha(SOURCE / "test_v7g_offline.py"),
            str(V7G_ACCEPTANCE): sha(V7G_ACCEPTANCE),
            str(V7G_AUTH_ROOT / "constructor-authorization-source.json"): sha(
                V7G_AUTH_ROOT / "constructor-authorization-source.json"
            ),
            str(V7G_AUTH_ROOT / "launch-authorization-source.json"): sha(
                V7G_AUTH_ROOT / "launch-authorization-source.json"
            ),
        },
        "schema_version": 3,
        "terminal_failure_seal_sha256": sha(terminal_path),
    }
    write(DESTINATION / "protected-input-hashes.json", canonical(ledger))


def update_candidate_hash() -> None:
    candidate = DESTINATION / "candidate-intent-v7h.json"
    constructor = DESTINATION / "construct_manager_authority_v7h.py"
    text = constructor.read_text(encoding="utf-8")
    old = "2fa766ffd28ef09d18fffffd74255baffdc7b15b1af759c2d4c71cc75cc505f1"
    if text.count(old) != 1:
        raise RuntimeError("candidate hash placeholder count mismatch")
    constructor.write_text(text.replace(old, sha(candidate)), encoding="utf-8")


def main() -> None:
    assert_v7g_terminal_state()
    if DESTINATION.exists() and not (
        DESTINATION / "v7g-terminal-pre-origin-failure-seal.json"
    ).exists():
        shutil.rmtree(DESTINATION)
    for path in (DESTINATION, V7H_AUTH_TOP, V7H_STATE_PARENT, V7H_FUTURE_AUTH_TOP):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh v7h path already exists: {path}")
    rewrite_tree()
    bind_predecessor()
    update_candidate_hash()
    print(DESTINATION)


if __name__ == "__main__":
    main()
