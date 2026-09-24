#!/usr/bin/env python3
"""Reviewer-only atomic Fresh-L2 acceptance creator for sealed V29."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import jsonschema


ACTION_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load_module(ACTION_ROOT / "tools/v29_contract.py", "v29_acceptance_contract")


def atomic_create(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o444)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", required=True, type=Path)
    args = parser.parse_args()
    C.require(Path.cwd() == ACTION_ROOT, "ACCEPTANCE_CREATOR_CWD", str(Path.cwd()))
    C.require(args.candidate_report == C.CANDIDATE_REPORT, "CANDIDATE_REPORT_PATH_BINDING", str(args.candidate_report))
    C.require(not os.path.lexists(C.ACCEPTANCE_PATH), "ACCEPTANCE_ALREADY_EXISTS", str(C.ACCEPTANCE_PATH))
    C.require(stat.S_IMODE(C.ACCEPTANCE_PATH.parent.stat().st_mode) == 0o755, "REVIEW_DIRECTORY_NOT_OPEN", str(C.ACCEPTANCE_PATH.parent))
    candidate = load_module(ACTION_ROOT / "tools/verify_candidate_v29.py", "v29_acceptance_candidate")
    report = candidate.verify_candidate(False)
    report_raw = args.candidate_report.read_bytes()
    supplied = json.loads(report_raw.decode("ascii", "strict"))
    C.require(type(supplied) is dict and C.compact_bytes(supplied) == report_raw and supplied == report, "CANDIDATE_REPORT_MISMATCH", str(args.candidate_report))
    package, package_raw = C.load_canonical(C.PACKAGE_PATH)
    C.verify_package(package, package_raw)
    provenance = C.attest_current_reviewer(package, report, report_raw)
    acceptance = C.add_self_hash(C.acceptance_unhashed(package, report, report_raw, provenance), "acceptance_sha256")
    schema, _ = C.load_canonical(ACTION_ROOT / package["fresh_l2_review"]["acceptance_schema_path"])
    jsonschema.Draft202012Validator(schema).validate(acceptance)
    raw = C.compact_bytes(acceptance)
    atomic_create(C.ACCEPTANCE_PATH, raw)
    os.chmod(C.ACCEPTANCE_PATH.parent, 0o555)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
