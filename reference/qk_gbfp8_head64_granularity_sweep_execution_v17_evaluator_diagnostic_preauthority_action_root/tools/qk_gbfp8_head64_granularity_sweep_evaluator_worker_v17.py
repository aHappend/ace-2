#!/usr/bin/env python3
"""V17 evaluator worker with production and explicitly synthetic modes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ACTION_ID = 'ace2:qk-gbfp8-base-v17:execute-once:9b1e6d42:20260814T101609Z'
PACKAGE = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json')
ACCEPTANCE = Path('/home/argustest/ace-2/build/v17-evaluator-diagnostic-preauthority/FRESH_L2_STATIC_ACCEPTANCE.json')
OFFICIAL_EVALUATOR = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py')
OFFICIAL_PARSER = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py')
EXACT_ENVIRONMENT = {'LANG': 'C', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0', 'TZ': 'UTC'}


class WorkerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def canonical_object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, "worker canonical input")
    return value


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(scenario: str, raw: bytes) -> int:
    if scenario == "nominal":
        sys.stdout.buffer.write(raw)
        return 0
    if scenario == "nonzero":
        sys.stderr.buffer.write(b"synthetic evaluator nonzero\n")
        return 7
    if scenario == "no-output":
        return 0
    if scenario == "malformed":
        sys.stdout.buffer.write(b"{malformed\n")
        return 0
    if scenario == "stderr":
        sys.stdout.buffer.write(raw)
        sys.stderr.buffer.write(b"synthetic evaluator stderr\n")
        return 0
    raise WorkerError("unknown synthetic scenario")


def production(args: argparse.Namespace, raw: bytes) -> int:
    require(Path.cwd() == PACKAGE.parent, "production cwd")
    require(dict(os.environ) == EXACT_ENVIRONMENT, "production environment")
    require(args.package == str(PACKAGE), "production package")
    require(args.acceptance == str(ACCEPTANCE), "production acceptance")
    require(args.irreversible_action_id == ACTION_ID, "production action")
    package = canonical_object(PACKAGE.read_bytes())
    acceptance = canonical_object(ACCEPTANCE.read_bytes())
    require(package["action_identity"]["future_action_id"] == ACTION_ID, "bound package action")
    require(acceptance.get("accepted") is True, "Fresh-L2 acceptance")
    require(acceptance.get("static_acceptance_grants_execution_authority") is False, "acceptance authority boundary")
    envelope = canonical_object(raw)
    require(set(envelope) == {"bindings", "context", "payload_base64"}, "production envelope keys")
    payload = base64.b64decode(envelope["payload_base64"], validate=True)
    evaluator = load_module("v17_bound_official_evaluator", OFFICIAL_EVALUATOR)
    parser = load_module("v17_bound_official_parser", OFFICIAL_PARSER)
    result = evaluator.evaluate_bundle_once(lambda: payload, envelope["bindings"], parser, envelope["context"])
    require(type(result) is bytes, "official evaluator return ABI")
    sys.stdout.buffer.write(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("production", "synthetic"))
    parser.add_argument("--scenario", choices=("nominal", "nonzero", "no-output", "malformed", "stderr"))
    parser.add_argument("--package")
    parser.add_argument("--acceptance")
    parser.add_argument("--irreversible-action-id")
    args = parser.parse_args()
    raw = sys.stdin.buffer.read()
    try:
        if args.mode == "synthetic":
            require(args.scenario is not None, "synthetic scenario")
            return synthetic(args.scenario, raw)
        require(args.scenario is None, "production scenario prohibited")
        return production(args, raw)
    except Exception as error:
        detail = f"{type(error).__name__}:{hashlib.sha256(str(error).encode('utf-8', 'replace')).hexdigest()}\n"
        sys.stderr.write(detail)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
