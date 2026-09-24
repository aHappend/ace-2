#!/usr/bin/env python3
"""Reviewer inert post-verifier for V29 acceptance and provenance bindings."""

from __future__ import annotations

import argparse
import importlib.util
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


C = load_module(ACTION_ROOT / "tools/v29_contract.py", "v29_post_contract")


def main() -> int:
    C.install_audit_hook()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    package, package_raw = C.load_canonical(C.PACKAGE_PATH)
    C.verify_package(package, package_raw)
    candidate = load_module(ACTION_ROOT / "tools/verify_candidate_v29.py", "v29_post_candidate")
    report = candidate.verify_candidate(True)
    acceptance, acceptance_raw = C.load_canonical(C.ACCEPTANCE_PATH)
    schema, _ = C.load_canonical(ACTION_ROOT / package["fresh_l2_review"]["acceptance_schema_path"])
    jsonschema.Draft202012Validator(schema).validate(acceptance)
    C.verify_acceptance(package, acceptance, acceptance_raw, report)
    result = {
        "acceptance_file_sha256": C.sha256_bytes(acceptance_raw),
        "acceptance_self_sha256": acceptance["acceptance_sha256"],
        "action_id": C.STATIC_ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_b0_execution_binding_repair_static_v29_post_acceptance_report",
        "candidate_action_tree_sha256": report["preacceptance_tree_sha256"],
        "candidate_report_sha256": report["report_sha256"],
        "claim_boundary": C.CLAIM_BOUNDARY,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_V29_STATIC_POST_ACCEPTANCE_PROVENANCE_RECOMPUTED",
    }
    result["report_sha256"] = C.sha256_bytes(C.compact_bytes(result))
    raw = C.compact_bytes(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
