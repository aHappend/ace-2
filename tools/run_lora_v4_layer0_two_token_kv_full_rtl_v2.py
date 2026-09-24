#!/usr/bin/env python3
"""Fresh-L2 V2 specification repair and two-position layer-0 RTL replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_layer0_two_token_kv_full_rtl as base


MISSION = "lora_v4_layer0_two_token_kv_full_rtl_specification_repair_v2"
OUT = ROOT / "evidence/verification/lora-v4-layer0-two-token-kv-full-v1"
PARENT_CONTRACT = OUT / "frozen_contract.json"
PRESERVED_ATTEMPT = OUT / "attempt-0001"
REPAIR_CONTRACT = OUT / "repair-0002-contract.json"
ATTEMPT = OUT / "attempt-0002"
PREVIEW_ROOT = ROOT / "build/lora-v4-layer0-two-token-kv-full-repair-v2-preview"

PARENT_SPEC = ROOT / "design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_SPEC.md"
PARENT_INTERFACE = ROOT / "design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_INTERFACE.json"
PARENT_RUNNER = ROOT / "tools/run_lora_v4_layer0_two_token_kv_full_rtl.py"
SPEC = ROOT / "design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_SPEC_V2.md"
INTERFACE = ROOT / "design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_INTERFACE_V2.json"

PARENT_SPEC_SHA256 = "b913b397b4b15d89a5f63647c257da5f169c68be100ba12ae42e8228eed6aa70"
PARENT_INTERFACE_SHA256 = "dfff2c3db0b6d8eee138d964c6557dfa2766f60bdfe0bb05af2ae3c513bf0a37"
PARENT_RUNNER_SHA256 = "afe6ad0fae5db7c8d6e7dc9c9bf5136654a867a546363ea38e19644689334e25"
PARENT_CONTRACT_SHA256 = "6f16f54d2d24061352404dc5a07df07448051b79eda0d61bf06eee1e383d471a"
PRESERVED_SUMS_SHA256 = "8a9eca9fc23a6ed40c9faa18e76dd01ef6e0587472e0488de92aa625db7a09aa"
PRESERVED_MEMBER_COUNT = 143

FAMILIES = ("projection", "rmsnorm", "rope", "score", "softmax", "compose", "silu", "residual")
EXPECTED_CYCLES = {
    "projection": 26764922,
    "rmsnorm": 249060,
    "rope": 757397,
    "score": 5633,
    "softmax": 3071,
    "compose": 34023,
    "silu": 1764423,
    "residual": 3623,
}
EXPECTED_FINAL_HASHES = [
    "87e50216a5fad60d4022be22c7df9a422cd3670d9e3805a6d66dbd3071894bd3",
    "163c71316c812b156f4094d17b8ba23581a7d1e565d5b4ad9373b03764cf04ef",
]
REQUIRED_SCENARIO_CLASSES = {"normal", "boundary", "illegal", "reset", "stall", "recovery"}
REQUIRED_RANGE_INTERFACES = {
    "common",
    "projection",
    "rmsnorm",
    "rope",
    "attention_score",
    "softmax",
    "attention_compose",
    "silu_gate",
    "residual_shell",
}
REQUIRED_RESET_FAMILIES = {
    "projection",
    "rmsnorm",
    "rope",
    "attention_score",
    "softmax",
    "attention_compose",
    "silu_gate",
    "residual_shell",
}

_BASE_SOURCE_BINDINGS = base.source_bindings
_BASE_VALIDATE_PREDECESSORS = base.validate_predecessors
_BASE_RUN_RTL = base.run_rtl


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_preserved_v1() -> int:
    base.require(PARENT_CONTRACT.is_file(), "parent frozen contract is absent")
    base.require(PRESERVED_ATTEMPT.is_dir(), "preserved attempt-0001 is absent")
    base.require(base.sha256_file(PARENT_SPEC) == PARENT_SPEC_SHA256, "parent specification changed")
    base.require(base.sha256_file(PARENT_INTERFACE) == PARENT_INTERFACE_SHA256, "parent interface changed")
    base.require(base.sha256_file(PARENT_RUNNER) == PARENT_RUNNER_SHA256, "parent runner changed")
    base.require(base.sha256_file(PARENT_CONTRACT) == PARENT_CONTRACT_SHA256, "parent contract changed")
    base.require(
        base.sha256_file(PRESERVED_ATTEMPT / "SHA256SUMS") == PRESERVED_SUMS_SHA256,
        "preserved attempt-0001 SHA256SUMS changed",
    )
    members = base.verify_sha256s(PRESERVED_ATTEMPT)
    base.require(members == PRESERVED_MEMBER_COUNT, "preserved attempt-0001 member count changed")
    return members


def validate_interface_v2() -> dict[str, Any]:
    validate_preserved_v1()
    base.require(SPEC.is_file() and INTERFACE.is_file(), "V2 specification or interface is absent")
    inherited = read_json(PARENT_INTERFACE)
    closure = read_json(INTERFACE)

    base.require(closure["schema_version"] == 2, "V2 interface schema changed")
    base.require(closure["kind"] == "versioned_interface_closure", "V2 interface kind changed")
    base.require(closure["mission_id"] == "extend-layer0-two-token-kv-rtl", "V2 mission changed")
    base.require(closure["normative_spec"] == base.relative(SPEC), "V2 normative spec path changed")
    base.require(
        closure["inherits"]["specification"]["sha256"] == PARENT_SPEC_SHA256,
        "V2 parent specification binding changed",
    )
    base.require(
        closure["inherits"]["exact_port_and_workload_manifest"]["sha256"] == PARENT_INTERFACE_SHA256,
        "V2 parent interface binding changed",
    )

    for value in inherited["rtl_interfaces"].values():
        path = ROOT / value["source"]
        base.require(base.sha256_file(path) == value["source_sha256"], f"interface-bound RTL changed: {path}")
        base.require(len(value["ports"]) == value["port_count"], f"interface port count changed: {value['module']}")
    for value in inherited["maintained_testbenches"].values():
        path = ROOT / value["path"]
        base.require(base.sha256_file(path) == value["sha256"], f"interface-bound testbench changed: {path}")
    base.require(inherited["workload"]["projection"]["channels"] == 25344, "projection workload changed")
    base.require(inherited["workload"]["residual_shell"]["output_beats"] == 224, "residual workload changed")

    benchmark = closure["benchmark_applicability"]
    base.require(benchmark["mission_id"] == "extend-layer0-two-token-kv-rtl", "benchmark scope changed")
    base.require(benchmark["applies"] is False, "V2 incorrectly claims benchmark applicability")
    base.require(
        set(closure["legal_semantic_ranges"]) == REQUIRED_RANGE_INTERFACES,
        "V2 legal semantic range coverage changed",
    )
    base.require(
        set(closure["reset_observability"]["families"]) == REQUIRED_RESET_FAMILIES,
        "V2 reset observability coverage changed",
    )
    scenario_classes = {row["class"] for row in closure["scenario_matrix"]}
    base.require(REQUIRED_SCENARIO_CLASSES <= scenario_classes, "V2 scenario classes are incomplete")
    for row in closure["scenario_matrix"]:
        if not row["required"]:
            base.require(
                row.get("current_outcome") == "NOT_EXERCISED_NO_CLAIM",
                f"optional scenario lacks explicit no-claim outcome: {row['id']}",
            )
    performance = closure["latency_throughput_expectations"]
    base.require(performance["expected_simulator_cycles_exact"] == EXPECTED_CYCLES, "V2 expected cycles changed")
    base.require(
        performance["expected_aggregate_family_cycles"] == sum(EXPECTED_CYCLES.values()),
        "V2 aggregate expected cycles changed",
    )
    base.require(closure["fresh_l2"]["compile_families"] == list(FAMILIES), "Fresh-L2 family list changed")
    base.require(closure["fresh_l2"]["reuse_prior_simulator_binaries"] is False, "V1 binary reuse allowed")
    base.require(
        closure["fresh_l2"]["expected_final_layer_output_sha256_by_position"] == EXPECTED_FINAL_HASHES,
        "V2 expected final hashes changed",
    )

    merged = dict(inherited)
    merged["v2_closure"] = closure
    return merged


def validate_predecessors_v2() -> None:
    _BASE_VALIDATE_PREDECESSORS()
    validate_preserved_v1()


def source_bindings_v2() -> dict[str, Any]:
    bindings = _BASE_SOURCE_BINDINGS()
    bindings.update(
        {
            "parent_full_layer_contract": base.file_record(PARENT_CONTRACT),
            "parent_full_layer_specification": base.file_record(PARENT_SPEC),
            "parent_full_layer_interface": base.file_record(PARENT_INTERFACE),
            "parent_full_layer_runner": base.file_record(PARENT_RUNNER),
            "preserved_attempt_0001_sums": base.file_record(PRESERVED_ATTEMPT / "SHA256SUMS"),
        }
    )
    return bindings


def run_rtl_v2(derived: dict[str, Any]) -> dict[str, Any]:
    execution = _BASE_RUN_RTL(derived)
    base.require(execution["simulator_cycles"] == EXPECTED_CYCLES, "Fresh-L2 simulator cycles changed")
    base.require(
        execution["final_layer_output_sha256_by_position"] == EXPECTED_FINAL_HASHES,
        "Fresh-L2 final output hashes changed",
    )
    return execution


def configure_base() -> None:
    base.MISSION = MISSION
    base.OUT = OUT
    base.CONTRACT = REPAIR_CONTRACT
    base.OFFICIAL = ATTEMPT
    base.PREVIEW_ROOT = PREVIEW_ROOT
    base.SPEC = SPEC
    base.INTERFACE = INTERFACE
    base.ACTIVE_ATTEMPT = ATTEMPT
    base.__file__ = str(Path(__file__).resolve())
    base.validate_interface = validate_interface_v2
    base.validate_predecessors = validate_predecessors_v2
    base.source_bindings = source_bindings_v2
    base.run_rtl = run_rtl_v2


def repair_contract_payload() -> dict[str, Any]:
    parent = read_json(PARENT_CONTRACT)
    closure = read_json(INTERFACE)
    return {
        "schema_version": 2,
        "kind": "frozen_specification_interface_repair_contract",
        "mission_increment": MISSION,
        "claim": parent["claim"],
        "repair_scope": "add versioned legal-range, reset-observability, simulator latency/throughput, scenario-outcome, and mission-scoped non-benchmark contracts; recompile and reproduce without RTL, checkpoint, input, quantization, or evaluator-policy change",
        "parent_frozen_contract": base.file_record(PARENT_CONTRACT),
        "preserved_attempt_0001": {
            "path": base.relative(PRESERVED_ATTEMPT),
            "sha256s": base.file_record(PRESERVED_ATTEMPT / "SHA256SUMS"),
            "members_exact": PRESERVED_MEMBER_COUNT,
            "modified": False,
        },
        "closure": {
            "versioned_specification": base.file_record(SPEC),
            "versioned_interface": base.file_record(INTERFACE),
            "parent_specification": base.file_record(PARENT_SPEC),
            "parent_interface": base.file_record(PARENT_INTERFACE),
            "closed_gaps": [
                "legal_semantic_ranges",
                "observable_reset_values",
                "preexecution_simulator_latency_and_throughput_expectations",
                "normal_boundary_illegal_reset_stall_recovery_outcomes",
                "mission_scoped_non_benchmark_statement",
            ],
            "benchmark_applicability": closure["benchmark_applicability"],
        },
        "canonical_checkpoint": parent["canonical_checkpoint"],
        "fixed_input": parent["fixed_input"],
        "architecture": parent["architecture"],
        "quantization": parent["quantization"],
        "specification": base.file_record(SPEC),
        "interface_manifest": base.file_record(INTERFACE),
        "evaluator_policy": parent["evaluator_policy"],
        "fresh_l2_reproduction": {
            "attempt": "attempt-0002",
            "compile_families": list(FAMILIES),
            "fresh_execution_sources": True,
            "reuse_attempt_0001_binaries": False,
            "expected_simulator_cycles_exact": EXPECTED_CYCLES,
            "expected_projection_results": 25344,
            "expected_final_layer_output_sha256_by_position": EXPECTED_FINAL_HASHES,
            "integer_mismatch_max": 0,
        },
        "source_bindings": source_bindings_v2(),
        "commands": {
            "preflight": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl_v2.py --preflight",
            "prepare_repair": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl_v2.py --prepare-repair",
            "run": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl_v2.py --run",
            "check": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl_v2.py --check",
        },
        "attempt_policy": {
            "preserved_official_attempt": "attempt-0001",
            "fresh_l2_repair_attempt": "attempt-0002",
            "immutable_after_execution": True,
            "predecessor_attempts_modified": False,
        },
    }


def prepare_repair() -> None:
    base.require(not REPAIR_CONTRACT.exists(), f"repair contract already exists: {REPAIR_CONTRACT}")
    base.require(not ATTEMPT.exists(), f"Fresh-L2 attempt already exists: {ATTEMPT}")
    validate_interface_v2()
    validate_predecessors_v2()
    contract = repair_contract_payload()
    base.write_json(REPAIR_CONTRACT, contract)
    print(
        json.dumps(
            {
                "status": "FROZEN_REPAIR_0002",
                "contract": base.relative(REPAIR_CONTRACT),
                "sha256": base.sha256_file(REPAIR_CONTRACT),
                "preserved_attempt_0001_members": PRESERVED_MEMBER_COUNT,
                "fresh_attempt": base.relative(ATTEMPT),
            },
            sort_keys=True,
        )
    )


def preflight() -> None:
    base.preflight()


def run() -> None:
    validate_preserved_v1()
    base.execute(ATTEMPT, official=True)


def check_fresh_l2_details() -> dict[str, Any]:
    contract = base.validate_contract()
    base.require(ATTEMPT.is_dir(), "Fresh-L2 attempt-0002 is absent")
    members = base.verify_sha256s(ATTEMPT)
    manifest = read_json(ATTEMPT / "manifest.json")
    comparison = read_json(ATTEMPT / "comparison.json")
    execution = read_json(ATTEMPT / "rtl_execution.json")

    base.require(manifest["contract"]["sha256"] == base.sha256_file(REPAIR_CONTRACT), "repair binding changed")
    base.require(manifest["specification"] == base.file_record(SPEC), "V2 specification manifest binding changed")
    base.require(manifest["interface_manifest"] == base.file_record(INTERFACE), "V2 interface manifest binding changed")
    base.require(manifest["source_bindings"] == source_bindings_v2(), "Fresh-L2 source bindings changed")
    base.require(comparison["status"] == "PASS_EXACT_TWO_POSITION_FULL_LAYER", "Fresh-L2 comparison is not PASS")
    base.require(execution["projection_results"] == 25344, "Fresh-L2 projection result count changed")
    base.require(execution["simulator_cycles"] == EXPECTED_CYCLES, "Fresh-L2 cycle reproduction changed")
    base.require(
        execution["final_layer_output_sha256_by_position"] == EXPECTED_FINAL_HASHES,
        "Fresh-L2 final output reproduction changed",
    )
    base.require(set(execution["executions"]) == set(FAMILIES), "Fresh-L2 execution family set changed")

    for family in FAMILIES:
        record = execution["executions"][family]
        base.require(
            record["binary"]["path"].startswith(base.relative(ATTEMPT) + "/sim/"),
            f"{family} binary is outside attempt-0002",
        )
        compile_command = record["compile_command"]
        simulate_command = record["simulate_command"]
        base.require(base.relative(ATTEMPT) in " ".join(compile_command), f"{family} compile did not target attempt-0002")
        base.require(base.relative(ATTEMPT) in " ".join(simulate_command), f"{family} simulation did not use attempt-0002")
        base.require(
            any(base.relative(ATTEMPT / "execution_sources") in item for item in compile_command),
            f"{family} compile did not use a fresh execution source",
        )
        for phase in ("iverilog", "vvp"):
            result = read_json(ATTEMPT / f"logs/{family}.{phase}.result.json")
            base.require(result["returncode"] == 0, f"{family} {phase} return code changed")

    base.require(
        contract["fresh_l2_reproduction"]["expected_simulator_cycles_exact"] == EXPECTED_CYCLES,
        "repair contract cycle expectation changed",
    )
    preserved_members = validate_preserved_v1()
    return {
        "status": "PASS_FRESH_L2_REPAIR_ATTEMPT_0002",
        "attempt": base.relative(ATTEMPT),
        "members": members,
        "preserved_attempt_0001_members": preserved_members,
        "projection_results": execution["projection_results"],
        "simulator_cycles": execution["simulator_cycles"],
        "final_layer_output_sha256_by_position": execution["final_layer_output_sha256_by_position"],
        "integer_mismatches": comparison["rtl_vs_independent_python_fixed_point"],
        "benchmark_applicability": read_json(INTERFACE)["benchmark_applicability"],
    }


def check() -> None:
    base.check()
    print(json.dumps(check_fresh_l2_details(), sort_keys=True))


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--prepare-repair", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.prepare_repair:
        prepare_repair()
    elif args.run:
        run()
    else:
        check()


if __name__ == "__main__":
    main()
