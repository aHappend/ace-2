#!/usr/bin/env python3
"""Reviewer-only positive V28 post-acceptance verifier."""

from __future__ import annotations

import sys
EARLY_DONT_WRITE_BYTECODE = sys.dont_write_bytecode is True
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import jsonschema

ACTION_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_FRESH_L2_ACCEPTANCE_SCHEMA.json"


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"module spec: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    guard = load(ACTION_ROOT / "tools/literal_b_launch_guard_v28.py", "v28_post_guard")
    launch = {**guard.validate_current_process(Path(__file__), EARLY_DONT_WRITE_BYTECODE), **guard.prove_effective_suppression()}
    candidate = load(ACTION_ROOT / "tools/verify_qk_gbfp8_head64_source_oracle_identity_control_candidate_v28.py", "v28_post_candidate")
    candidate.install_audit_hook()
    acceptance_raw = ACCEPTANCE.read_bytes(); acceptance = json.loads(acceptance_raw.decode("ascii", "strict"))
    schema = json.loads(SCHEMA.read_text(encoding="ascii")); jsonschema.Draft202012Validator(schema).validate(acceptance)
    unhashed = dict(acceptance); observed = unhashed.pop("acceptance_sha256")
    if observed != sha256_bytes(compact_bytes(unhashed)): raise RuntimeError("acceptance self hash")
    report, _ = candidate.verify(True, launch)
    result = {
        "acceptance_file_sha256": sha256_bytes(acceptance_raw),
        "acceptance_sha256": observed,
        "action_id": "ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control:38571d38:additive-0002",
        "artifact_kind": "qk_gbfp8_head64_source_oracle_identity_control_static_v28_post_acceptance_report",
        "candidate_report_sha256": report["report_sha256"],
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_V28_STATIC_POST_ACCEPTANCE_ROLE_ENFORCED_LITERAL_B_MODE_SEALED",
    }
    result["report_sha256"] = sha256_bytes(compact_bytes(result))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(compact_bytes(result)); sys.stdout.buffer.write(compact_bytes(result)); return 0


if __name__ == "__main__": raise SystemExit(main())
