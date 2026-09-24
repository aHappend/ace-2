#!/usr/bin/env python3
"""Reviewer-only inert preflight that creates no acceptance or live authority."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ACTION_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load_module(ACTION_ROOT / "tools/v29_contract.py", "v29_reviewer_preflight_contract")


def main() -> int:
    C.install_audit_hook()
    C.require(Path.cwd() == ACTION_ROOT, "REVIEWER_PREFLIGHT_CWD", str(Path.cwd()))
    package, package_raw = C.load_canonical(C.PACKAGE_PATH)
    C.verify_package(package, package_raw)
    candidate = load_module(ACTION_ROOT / "tools/verify_candidate_v29.py", "v29_reviewer_preflight_candidate")
    report = candidate.verify_candidate(False)
    report_raw = C.CANDIDATE_REPORT.read_bytes()
    supplied = json.loads(report_raw.decode("ascii", "strict"))
    C.require(
        type(supplied) is dict and C.compact_bytes(supplied) == report_raw and supplied == report,
        "REVIEWER_PREFLIGHT_CANDIDATE_REPORT_MISMATCH",
        str(C.CANDIDATE_REPORT),
    )
    start = C.attest_current_reviewer_start()
    commitment = C.reviewer_preflight_commitment(package, report, report_raw, start["active_call_id"])
    sys.stdout.buffer.write(C.compact_bytes(commitment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
