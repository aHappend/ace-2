#!/usr/bin/env python3
"""Evidence-integrity repair for the fused-QKV benchmark evaluator."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import run_fused_qkv_benchmark as evaluator

ROOT = Path(__file__).resolve().parents[1]
V2_CONTRACT = ROOT / "benchmark/fused_qkv_v2/contract.json"
CONTRACT = ROOT / "benchmark/fused_qkv_v3/contract.json"
INTERFACE = ROOT / "design/BENCHMARK_INTERFACE.json"
TOOL = Path(__file__).resolve()
IMPORTED_EVALUATOR = Path(evaluator.__file__).resolve()
BASE_READ_CONTRACT = evaluator.read_contract

evaluator.CONTRACT = CONTRACT
evaluator.TOOL = TOOL
evaluator.BUILD_ROOT = ROOT / "build/fused-qkv-v3"
evaluator.EVIDENCE_ROOT = ROOT / "evidence/verification/fused-qkv-v3"
evaluator.PASS_RE = re.compile(
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


def validate_interface(contract: dict[str, object]) -> None:
    expected = contract["public_rtl_contract"]["ports_frozen_by"]  # type: ignore[index]
    if evaluator.file_record(INTERFACE) != expected:
        raise RuntimeError("frozen benchmark interface manifest changed")

    manifest = json.loads(INTERFACE.read_text(encoding="utf-8"))
    shell = manifest["preserved_predecessor_two_token_contract"][
        "rtl_interfaces"
    ]["residual_shell"]
    fused = manifest["fused_qkv_contract"]
    shell_hash = evaluator.sha256_file(evaluator.LIVE_SHELL)
    if shell["elaboration"]["SRAM_ADDR_WIDTH"] != 12:
        raise RuntimeError("benchmark interface SRAM_ADDR_WIDTH is not 12")
    if shell["source_sha256"] != shell_hash:
        raise RuntimeError("benchmark interface shell hash is stale")
    if fused["source_sha256"] != shell_hash or fused["command"]["opcode"] != 11:
        raise RuntimeError("benchmark interface fused-QKV binding is stale")


def read_contract() -> dict[str, object]:
    contract = BASE_READ_CONTRACT()
    validate_interface(contract)
    return contract


evaluator.read_contract = read_contract


def freeze() -> None:
    if CONTRACT.exists():
        raise RuntimeError("fused QKV v3 benchmark contract is already frozen")
    if not V2_CONTRACT.is_file():
        raise RuntimeError("missing immutable v2 benchmark contract")

    freeze_output = evaluator.BUILD_ROOT / "contract-compile"
    if freeze_output.exists():
        shutil.rmtree(freeze_output)
    binary = evaluator.compile_benchmark(evaluator.LIVE_SHELL, freeze_output)

    contract = json.loads(V2_CONTRACT.read_text(encoding="utf-8"))
    predecessor_repair = contract.pop("repair_of")
    contract["name"] = "fused-qkv-v3"
    contract["created_at_utc"] = datetime.now(UTC).isoformat()
    contract["evaluator"] = {
        str(IMPORTED_EVALUATOR.relative_to(ROOT)): evaluator.file_record(
            IMPORTED_EVALUATOR
        ),
        str(TOOL.relative_to(ROOT)): evaluator.file_record(TOOL),
        str(evaluator.TB.relative_to(ROOT)): evaluator.file_record(evaluator.TB),
    }
    contract["public_rtl_contract"]["ports_frozen_by"] = evaluator.file_record(
        INTERFACE
    )
    contract["public_rtl_contract"]["compile_probe"] = evaluator.file_record(binary)
    contract["predecessor_repair"] = predecessor_repair
    contract["repair_of"] = {
        "contract": evaluator.file_record(V2_CONTRACT),
        "preserved_attempts": [
            "evidence/verification/fused-qkv-v1/attempt-0001",
            "evidence/verification/fused-qkv-v2/attempt-0001",
            "evidence/verification/fused-qkv-v2/reviewer-round-0001",
        ],
        "failure_class": "evidence_integrity",
        "root_cause": (
            "The v2 evaluator manifest omitted its imported executable module, "
            "and its frozen interface manifest carried a stale shell hash and "
            "16-bit rather than live 12-bit SRAM address width."
        ),
        "repair": (
            "Freeze and verify the wrapper, imported evaluator, testbench, and "
            "corrected interface manifest without changing RTL, workload, "
            "oracle, official inputs, parser policy, or score policy."
        ),
    }
    contract["pre_edit_sources_and_constraint_hashes"] = evaluator.source_records()
    evaluator.write_atomic(CONTRACT, evaluator.canonical_bytes(contract))
    validate_interface(contract)
    print(
        "ACE2_FUSED_QKV_V3_FREEZE_PASS "
        f"v2_contract_sha256={contract['repair_of']['contract']['sha256']} "
        f"interface_sha256="
        f"{contract['public_rtl_contract']['ports_frozen_by']['sha256']} "
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
        evaluator.run(args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
