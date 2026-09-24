#!/usr/bin/env python3
"""Fail-closed specification-stage audit for the revised BF16 LoRA contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V1_CONTRACT.json"
CONTRACT_SUM = CONTRACT.with_suffix(".sha256")
CONTRACT_AUDIT = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v1-contract-audit-20260807T171642Z.json"
TRUNCATION_AUDIT = ROOT / "research/raw/specification/frozen-bf16-v20-generation-truncation-audit-20260807T171153Z.json"
FRESH_REVIEW = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v1-fresh-review-c2710e07d289-20260807.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"

EXPECTED_CONTRACT_SHA256 = "31298b475f2b43d1ba0f5250f0c59e709f6dfb6ac89768feca7d7cd14615c4be"
EXPECTED_CONTRACT_AUDIT_SHA256 = "0c334658281aad90e1136f6a5b9dadeed72894868c69f0a5b22f2cdc886e37d7"
EXPECTED_TRUNCATION_AUDIT_SHA256 = "303490c0ed01f602ca647deeb6e9148ec322fe82c78eddce2595b53330647365"
EXPECTED_FRESH_REVIEW_SHA256 = "09503a0c440c2d143c96a897705b006fc3fe403c52a69d0bc515eb475807f053"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def matrix_classes(spec: str) -> list[str]:
    section = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"}:
            classes.append(cells[0])
    return classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load_json(BENCHMARK)
    contract = load_json(CONTRACT)
    contract_audit = load_json(CONTRACT_AUDIT)
    truncation_audit = load_json(TRUNCATION_AUDIT)
    fresh_review = load_json(FRESH_REVIEW)
    pipeline = load_json(PIPELINE)
    shell = audit_shell_contract()
    observed_classes = matrix_classes(spec)
    local = benchmark.get("local_contract", {})
    successor = local.get("active_same_architecture_successor_contract", {}) if isinstance(local, dict) else {}
    runtime = local.get("required_successor_demo_input_contract", {}) if isinstance(local, dict) else {}
    generation = contract.get("acceptance", {}).get("generation", {})

    companion_fields = CONTRACT_SUM.read_text(encoding="ascii").strip().split()
    ground_lines = GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
    unknowns_only = bool(ground_lines and ground_lines[0] == "# Binding Unknowns") and all(
        not line.strip() or line.startswith("- ") for line in ground_lines[1:]
    )

    protocol_fragments = (
        "A0 contains no internal clock-domain crossing",
        "Reset may assert asynchronously at the accelerator boundary",
        "deassert synchronously before state machines leave reset",
        "earliest read response on clock edge",
        "remain stable until accepted",
        "cmd_done_valid_o",
        "there is no finite completion-latency guarantee unless",
    )
    behavior_fragments = (
        "### Public `ace2_shell` parameter and port contract",
        "Every public port below is declared as an unsigned SystemVerilog scalar or",
        "`ACE2RT3` version 3",
        "host-configurable range is 1 through 256",
        "primary comparable value is 128",
        "token 151643 or 151645 is generated",
        "KV capacity",
    )
    missing_protocol = [fragment for fragment in protocol_fragments if fragment not in spec]
    missing_behavior = [fragment for fragment in behavior_fragments if fragment not in spec]

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS" if shell.get("status") == "PASS" and not missing_behavior else "FAIL",
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "parameter_mismatches": shell.get("checks", {}).get("parameter_mismatches"),
            "port_mismatches": shell.get("checks", {}).get("port_mismatches"),
            "missing_fragments": missing_behavior,
        },
        "spec.clock-reset-protocol": {
            "status": "PASS" if not missing_protocol else "FAIL",
            "clock_domains": 1,
            "missing_fragments": missing_protocol,
        },
        "spec.acceptance-matrix": {
            "status": "PASS" if all(name in observed_classes for name in REQUIRED_CLASSES) else "FAIL",
            "required_classes": list(REQUIRED_CLASSES),
            "observed_classes": observed_classes,
        },
        "spec.benchmark-interface-closure": {
            "status": "PASS"
            if benchmark.get("applies") is False
            and benchmark.get("external_contract") is None
            and bool(benchmark.get("non_benchmark_statement"))
            and isinstance(local, dict)
            and local.get("top_module") == "ace2_shell"
            and local.get("public_interface_source")
            == "design/SPEC.md#public-ace2_shell-parameter-and-port-contract"
            else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "external_contract": benchmark.get("external_contract"),
            "top_module": local.get("top_module") if isinstance(local, dict) else None,
        },
    }

    bindings = {
        "pipeline_current_stage_specification": pipeline.get("current_stage") == "specification",
        "pipeline_specification_in_progress": pipeline.get("stages", {}).get("specification", {}).get("status")
        == "in_progress",
        "contract_hash_matches": sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256,
        "contract_companion_matches": companion_fields
        == [EXPECTED_CONTRACT_SHA256, CONTRACT.name],
        "contract_audit_hash_matches": sha256(CONTRACT_AUDIT) == EXPECTED_CONTRACT_AUDIT_SHA256,
        "contract_audit_pass": contract_audit.get("status") == "PASS",
        "truncation_audit_hash_matches": sha256(TRUNCATION_AUDIT) == EXPECTED_TRUNCATION_AUDIT_SHA256,
        "truncation_audit_artifact_only_pass": truncation_audit.get("status")
        == "PASS_ARTIFACT_ONLY_NO_REPLAY",
        "truncation_audit_keeps_single_sft_pilot": truncation_audit.get("routing_decision", {}).get(
            "single_authorized_sft_pilot_remains_necessary"
        )
        is True,
        "fresh_reviewer_record_hash_matches": sha256(FRESH_REVIEW)
        == EXPECTED_FRESH_REVIEW_SHA256,
        "fresh_reviewer_status_pass": fresh_review.get("status") == "PASS",
        "fresh_reviewer_contract_hash_matches": fresh_review.get("contract_sha256")
        == EXPECTED_CONTRACT_SHA256,
        "generation_primary_128": isinstance(generation, dict)
        and generation.get("primary_max_new_tokens") == 128,
        "generation_diagnostic_256": isinstance(generation, dict)
        and generation.get("diagnostic_max_new_tokens") == 256,
        "generation_eos_stopped": isinstance(generation, dict) and generation.get("eos_stopped") is True,
        "historical_64_preserved": isinstance(generation, dict)
        and generation.get("historical_64_token_results_overwritten") is False,
        "benchmark_contract_binding_matches": isinstance(successor, dict)
        and successor.get("contract_sha256") == EXPECTED_CONTRACT_SHA256,
        "benchmark_runtime_primary_128": isinstance(runtime, dict)
        and runtime.get("primary_comparable_max_new_tokens") == 128,
        "benchmark_runtime_diagnostic_256": isinstance(runtime, dict)
        and runtime.get("diagnostic_max_new_tokens") == 256,
        "benchmark_fresh_reviewer_gate_satisfied": isinstance(successor, dict)
        and successor.get("fresh_reviewer_gate_satisfied") is True
        and successor.get("fresh_reviewer_status") == "done"
        and successor.get("fresh_reviewer_record_sha256") == EXPECTED_FRESH_REVIEW_SHA256
        and successor.get("additional_fresh_reviewer_required_before_pilot") is False,
        "pilot_attempt_namespace_absent": not (
            ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v1/attempt-0001"
        ).exists(),
        "selected_policy_id_null": benchmark.get("product_compression_gate", {}).get("selected_policy_id")
        is None,
        "ground_truth_contains_only_binding_unknowns": unknowns_only,
    }

    status = "PASS" if all(item["status"] == "PASS" for item in checklist.values()) and all(
        bindings.values()
    ) else "FAIL"
    result = {
        "schema_version": 1,
        "audit_id": "specification-stage-bf16-lora-primary-128-v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "current_stage": pipeline.get("current_stage"),
        "checklist": checklist,
        "bindings": dict(sorted(bindings.items())),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                SPEC,
                BENCHMARK,
                CONTRACT,
                CONTRACT_AUDIT,
                TRUNCATION_AUDIT,
                FRESH_REVIEW,
                PIPELINE,
                GROUND_TRUTH,
            ]
        },
        "claim_boundary": "Specification-stage audit only. No training, model evaluation, RTL simulation, formal, synthesis, timing, PPA, FPGA build, hardware emulation, or board execution occurred.",
        "next_gate": "Fresh Reviewer mission c2710e07d289 is done. An Engineer must invoke the exactly-once pilot package on the preflight-qualified A100 endpoint; Planner execution is forbidden.",
    }
    if status != "PASS":
        raise RuntimeError(json.dumps(result, sort_keys=True))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "output": str(output.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
