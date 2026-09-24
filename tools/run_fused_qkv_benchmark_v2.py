#!/usr/bin/env python3
"""Evaluator-only repair for the fused-QKV benchmark marker parser."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import run_fused_qkv_benchmark as v1

ROOT = Path(__file__).resolve().parents[1]
V1_CONTRACT = ROOT / "benchmark/fused_qkv_v1/contract.json"
CONTRACT = ROOT / "benchmark/fused_qkv_v2/contract.json"
TOOL = Path(__file__).resolve()

v1.CONTRACT = CONTRACT
v1.TOOL = TOOL
v1.BUILD_ROOT = ROOT / "build/fused-qkv-v2"
v1.EVIDENCE_ROOT = ROOT / "evidence/verification/fused-qkv-v2"
v1.PASS_RE = re.compile(
    r"ACE2_FUSED_QKV_PASS mode=\s*(?P<mode>\w+)"
    r"\s+commands=(?P<commands>\d+)"
    r"\s+outputs_checked=(?P<outputs_checked>\d+)"
    r"\s+simulator_cycles=(?P<simulator_cycles>\d+)"
    r"\s+memory_read_requests=(?P<memory_read_requests>\d+)"
    r"\s+activation_read_requests=(?P<activation_read_requests>\d+)"
    r"\s+weight_read_requests=(?P<weight_read_requests>\d+)"
    r"\s+metadata_read_requests=(?P<metadata_read_requests>\d+)"
    r"\s+memory_write_requests=(?P<memory_write_requests>\d+)"
    r"\s+backpressure_stalls=(?P<backpressure_stalls>\d+)"
)


def freeze() -> None:
    if CONTRACT.exists():
        raise RuntimeError("fused QKV v2 benchmark contract is already frozen")
    if not V1_CONTRACT.is_file():
        raise RuntimeError("missing immutable v1 benchmark contract")
    freeze_output = v1.BUILD_ROOT / "contract-compile"
    if freeze_output.exists():
        shutil.rmtree(freeze_output)
    binary = v1.compile_benchmark(v1.LIVE_SHELL, freeze_output)
    contract = json.loads(V1_CONTRACT.read_text(encoding="utf-8"))
    contract["name"] = "fused-qkv-v2"
    contract["created_at_utc"] = datetime.now(UTC).isoformat()
    contract["evaluator"] = {
        str(v1.TB.relative_to(ROOT)): v1.file_record(v1.TB),
        str(TOOL.relative_to(ROOT)): v1.file_record(TOOL),
    }
    contract["public_rtl_contract"]["compile_probe"] = v1.file_record(binary)
    contract["repair_of"] = {
        "contract": v1.file_record(V1_CONTRACT),
        "first_attempt": (
            "evidence/verification/fused-qkv-v1/attempt-0001"
        ),
        "failure_class": "evaluator_marker_parsing",
        "root_cause": (
            "Icarus right-justified the fused branch of a conditional %s "
            "argument as ' fused'; the v1 parser required the mode token "
            "immediately after '='."
        ),
        "repair": (
            "Accept optional whitespace before the unchanged mode token; "
            "RTL, testbench, oracle, official inputs, and score policy are "
            "unchanged."
        ),
    }
    contract["pre_edit_sources_and_constraint_hashes"] = v1.source_records()
    v1.write_atomic(CONTRACT, v1.canonical_bytes(contract))
    print(
        "ACE2_FUSED_QKV_V2_FREEZE_PASS "
        f"v1_contract_sha256={contract['repair_of']['contract']['sha256']} "
        f"expected_sha256={contract['workload']['expected_output']['sha256']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("freeze")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        v1.run(args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
