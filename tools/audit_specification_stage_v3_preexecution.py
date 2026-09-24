#!/usr/bin/env python3
"""Fail-closed audit of the ACE-2 V3 pre-execution specification closure."""

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
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V3_PREEXECUTION_CONTRACT.json"
CONTRACT_SUM = CONTRACT.with_suffix(".sha256")
DIAGNOSIS = ROOT / "research/diagnostics/qwen25_lora_v1_v2_mechanism_diagnostic.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
U280 = ROOT / "research/raw/specification/u280-host-inventory-recheck-20260807T230641Z.json"
U280_SUM = U280.with_suffix(".sha256")
V2_TERMINAL = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v2/TERMINAL_NO_GO.json"

EXPECTED_CONTRACT_SHA256 = "411bf8dfb9722bfa3d18ee87219b9b46d92f5b2a9f0c9fb10f439d89e4e3210d"
EXPECTED_DIAGNOSIS_SHA256 = "43884952117710c471fa4e28559c3e66642b5be2d852bcd37b4347b4411f8263"
EXPECTED_U280_SHA256 = "30da1e0c734ce7c89bb49181a987d4386d378673a6e78e7534f88438d29fda52"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def companion_matches(path: Path, expected_digest: str, expected_name: str) -> bool:
    fields = path.read_text(encoding="ascii").strip().split()
    return fields == [expected_digest, expected_name]


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
    diagnosis = load_json(DIAGNOSIS)
    pipeline = load_json(PIPELINE)
    u280 = load_json(U280)
    terminal = load_json(V2_TERMINAL)
    shell = audit_shell_contract()

    local = benchmark.get("local_contract", {})
    prospective = local.get("prospective_v3_mechanism_response_contract", {}) if isinstance(local, dict) else {}
    compression = benchmark.get("product_compression_gate", {})
    internal = benchmark.get("internal_evaluation_contract", {})
    hypotheses = diagnosis.get("hypotheses", {})
    evaluation = contract.get("evaluation_contract", {})
    observed_classes = matrix_classes(spec)

    ground_lines = GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
    unknowns_only = bool(ground_lines and ground_lines[0] == "# Binding Unknowns") and all(
        not line.strip() or line.startswith("- ") for line in ground_lines[1:]
    )

    behavior_fragments = (
        "### Public `ace2_shell` parameter and port contract",
        "Every public port below is declared as an unsigned SystemVerilog scalar or",
        "`ACE2RT3` version 3",
        "host-configurable range is 1 through 256",
        "primary comparable value is 128",
        "KV capacity",
        "QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V3_PREEXECUTION_CONTRACT.json",
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
    missing_behavior = [fragment for fragment in behavior_fragments if fragment not in spec]
    missing_protocol = [fragment for fragment in protocol_fragments if fragment not in spec]

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
            and isinstance(prospective, dict)
            and prospective.get("contract_sha256") == EXPECTED_CONTRACT_SHA256
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
        "contract_companion_matches": companion_matches(
            CONTRACT_SUM, EXPECTED_CONTRACT_SHA256, CONTRACT.name
        ),
        "diagnosis_hash_matches": sha256(DIAGNOSIS) == EXPECTED_DIAGNOSIS_SHA256,
        "diagnosis_id_matches": diagnosis.get("diagnostic_id")
        == "qwen25-lora-v1-v2-complete-mechanism-diagnostic-v3",
        "diagnosis_precision_mechanism_supported": hypotheses.get(
            "h1_bf16_adapter_storage_degraded_optimization", {}
        ).get("classification")
        == "supported",
        "diagnosis_distribution_mechanism_supported": hypotheses.get(
            "h3_corpus_or_answer_style_catastrophic_tradeoffs", {}
        ).get("classification")
        == "supported",
        "diagnosis_scorer_style_mechanism_supported": hypotheses.get(
            "h4_scorer_generation_target_style_mismatch", {}
        ).get("classification")
        == "supported",
        "contract_has_no_training_authority": contract.get("attempt_policy", {}).get(
            "attempt_marker_creation_authorized"
        )
        is False
        and contract.get("attempt_policy", {}).get("operator_authorization_required") is True,
        "contract_fp32_master_and_moments": contract.get("optimization", {}).get(
            "adam_first_and_second_moments"
        )
        == "float32"
        and contract.get("optimization", {}).get("lora", {}).get("master_parameters")
        == "float32",
        "contract_candidate_epochs_1_to_4": contract.get("checkpoint_policy", {}).get(
            "candidate_epochs"
        )
        == [1, 2, 3, 4],
        "contract_system_message_hash_frozen": contract.get("data_contract", {}).get(
            "system_message_normalized_sha256"
        )
        == "cd9317ad54f5ff5f57e0510ecb86d292edcc60a40b20e313af30b859bd3e7d48"
        and contract.get("data_contract", {}).get("system_message_mismatch_count_maximum") == 0,
        "contract_fixed_evaluator_hashes": contract.get("evaluation_contract", {}).get("dev", {}).get(
            "input_sha256"
        )
        == "bcc21548de475db4c5a11b1c2aed89523136ec7581180e35e7b0005ad5b288a3"
        and contract.get("evaluation_contract", {}).get("holdout", {}).get("input_sha256")
        == "a24cbc11ae8bf0fb49b6f22dbe48f13e724fab1d6c46324f93b41d850234d1f6",
        "contract_evaluator_implementations_match": all(
            sha256(ROOT / evaluation[section]["evaluator_path"])
            == evaluation[section]["evaluator_sha256"]
            for section in ("dev", "holdout", "retention")
        )
        and sha256(ROOT / evaluation["runtime_path"]) == evaluation["runtime_sha256"],
        "contract_environment_lock_matches": sha256(
            ROOT / contract.get("environment_contract", {}).get("source_lock", "")
        )
        == contract.get("environment_contract", {}).get("source_lock_sha256"),
        "contract_primary_128_diagnostic_256": contract.get("evaluation_contract", {}).get(
            "generation", {}
        ).get("primary_max_new_tokens")
        == 128
        and contract.get("evaluation_contract", {}).get("generation", {}).get(
            "diagnostic_max_new_tokens"
        )
        == 256,
        "benchmark_v3_no_execution_authority": isinstance(prospective, dict)
        and prospective.get("attempt_marker_creation_authorized") is False
        and prospective.get("operator_execution_authority") is False
        and prospective.get("training_executed") is False,
        "benchmark_selected_policy_null": isinstance(compression, dict)
        and compression.get("selected_policy_id") is None,
        "internal_evaluator_not_external_benchmark": isinstance(internal, dict)
        and internal.get("classification")
        == "internal_hash_bound_fixed_evaluation_contract_not_external_accelerator_benchmark",
        "v2_terminal_no_go_preserved": terminal.get("status") == "NO_GO"
        and terminal.get("failure_taxonomy") == "DEV_QUALITY_GATE_FAILURE",
        "v3_attempt_namespace_absent": not (
            ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3"
        ).exists(),
        "u280_hash_matches": sha256(U280) == EXPECTED_U280_SHA256,
        "u280_companion_matches": companion_matches(U280_SUM, EXPECTED_U280_SHA256, U280.name),
        "u280_vendor_tools_absent": all(
            u280.get("tool_inventory", {}).get(name) is None
            for name in ("vivado", "vitis", "v++", "xrt-smi", "xbutil", "xbmgmt", "emconfigutil", "xclbinutil")
        ),
        "u280_board_absent": u280.get("board_inventory", {}).get(
            "pci_vendor_0x10ee_function_count"
        )
        == 0
        and u280.get("board_inventory", {}).get("physical_u280_access") is False,
        "ground_truth_contains_only_binding_unknowns": unknowns_only,
    }

    status = "PASS" if all(item["status"] == "PASS" for item in checklist.values()) and all(
        bindings.values()
    ) else "FAIL"
    result = {
        "schema_version": 1,
        "audit_id": "specification-stage-v3-preexecution-closure-v1",
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
                DIAGNOSIS,
                GROUND_TRUTH,
                PIPELINE,
                U280,
                V2_TERMINAL,
            ]
        },
        "claim_boundary": "Specification-stage audit only. No training, model evaluation, RTL simulation, formal, synthesis, timing, PPA, FPGA build, hardware emulation, board execution, or exactly-once marker occurred.",
        "next_gate": "Fresh operator authority plus an exact V3 data/runner/environment freeze and independent L2 review are required before an Engineer may create the sole attempt marker.",
    }
    if status != "PASS":
        raise RuntimeError(json.dumps(result, sort_keys=True))

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "output": str(output.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
