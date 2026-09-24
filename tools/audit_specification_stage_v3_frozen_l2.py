#!/usr/bin/env python3
"""Fail-closed post-L2 audit of the frozen ACE-2 V3 specification state."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V3_PREEXECUTION_CONTRACT.json"
DIAGNOSIS = ROOT / "research/diagnostics/qwen25_lora_v1_v2_mechanism_diagnostic.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-freeze-operator-authority.json"
FREEZE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v3/freeze_manifest.json"
FREEZE_AUDIT = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-freeze-audit.json"
L2_REVIEW = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-freeze-l2-review.json"
DATASET_MANIFEST = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v3/dataset_manifest.json"
RECIPE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v3/training_recipe.json"
SELF_TEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/execution-self-test.json"
U280 = ROOT / "research/raw/specification/u280-host-inventory-recheck-20260807T230641Z.json"
V2_TERMINAL = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v2/TERMINAL_NO_GO.json"
ATTEMPT_AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-attempt-operator-authority.json"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3"

EXPECTED_CONTRACT_SHA256 = "411bf8dfb9722bfa3d18ee87219b9b46d92f5b2a9f0c9fb10f439d89e4e3210d"
EXPECTED_DIAGNOSIS_SHA256 = "43884952117710c471fa4e28559c3e66642b5be2d852bcd37b4347b4411f8263"
EXPECTED_AUTHORITY_SHA256 = "726afc7ec780b5ae247683a830cf4f742acf0c5e90af42959486feb2bd9a18e2"
EXPECTED_FREEZE_SHA256 = "e92bd256da28fd595d640e20fe0398f5950f682ef3a95f7fa846af3eb3ec1bfa"
EXPECTED_FREEZE_AUDIT_SHA256 = "8cdb4d8119f789b4d49e9dabe8934d1949c02eecbd41d91a42d54fc38a22e0d8"
EXPECTED_L2_SHA256 = "c6ca3a1e4cfe8ddf33a5b7dc594cfa61e6bc6b8ed636219c80f21e9e9dc41ba7"
EXPECTED_PACKAGE_TREE_SHA256 = "50667162e3e630cda7d9230ac3e37fb591e78067e292bff36fb3a0e394894891"
EXPECTED_U280_SHA256 = "30da1e0c734ce7c89bb49181a987d4386d378673a6e78e7534f88438d29fda52"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def companion_matches(path: Path, companion: Path) -> bool:
    fields = companion.read_text(encoding="ascii").strip().split() if companion.is_file() else []
    return fields == [sha256(path), path.name]


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
    pipeline = load_json(PIPELINE)
    authority = load_json(AUTHORITY)
    freeze = load_json(FREEZE)
    freeze_audit = load_json(FREEZE_AUDIT)
    l2 = load_json(L2_REVIEW)
    dataset = load_json(DATASET_MANIFEST)
    recipe = load_json(RECIPE)
    self_test = load_json(SELF_TEST)
    u280 = load_json(U280)
    terminal = load_json(V2_TERMINAL)
    shell = audit_shell_contract()

    local = benchmark.get("local_contract", {})
    prospective = local.get("prospective_v3_mechanism_response_contract", {}) if isinstance(local, dict) else {}
    compression = benchmark.get("product_compression_gate", {})
    internal = benchmark.get("internal_evaluation_contract", {})
    observed_classes = matrix_classes(spec)

    ground_lines = GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
    unknowns_only = bool(ground_lines and ground_lines[0] == "# Binding Unknowns") and all(
        not line.strip() or line.startswith("- ") for line in ground_lines[1:]
    )

    accepted_hashes = l2.get("accepted_artifact_sha256", {})
    accepted_hash_map_exact = isinstance(accepted_hashes, dict) and bool(accepted_hashes) and all(
        (ROOT / relative).is_file() and sha256(ROOT / relative) == expected
        for relative, expected in accepted_hashes.items()
    )
    freeze_artifacts = freeze.get("artifacts", {})
    freeze_artifact_map_exact = isinstance(freeze_artifacts, dict) and bool(freeze_artifacts) and all(
        (ROOT / relative).is_file()
        and (ROOT / relative).stat().st_size == metadata.get("bytes")
        and sha256(ROOT / relative) == metadata.get("sha256")
        for relative, metadata in freeze_artifacts.items()
        if isinstance(metadata, dict)
    )

    behavior_fragments = (
        "### Public `ace2_shell` parameter and port contract",
        "Every public port below is declared as an unsigned SystemVerilog scalar or",
        "`ACE2RT3` version 3",
        "host-configurable range is 1 through 256",
        "primary comparable value is 128",
        "KV capacity",
        "V3_PACKAGE_FROZEN_L2_ACCEPTED_AWAITING_FRESH_ATTEMPT_AUTHORITY",
        EXPECTED_PACKAGE_TREE_SHA256,
        "qwen25-05b-instruct-bf16-lora-product-v3-freeze-l2-review.json",
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
            and prospective.get("status")
            == "FROZEN_L2_ACCEPTED_AWAITING_FRESH_ATTEMPT_AUTHORITY_NO_EXECUTION"
            and prospective.get("data_freeze_complete") is True
            and prospective.get("fresh_l2_review_complete") is True
            and prospective.get("operator_execution_authority") is False
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
        "contract_hash_exact": sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256,
        "contract_companion_exact": companion_matches(CONTRACT, CONTRACT.with_suffix(".sha256")),
        "diagnosis_hash_exact": sha256(DIAGNOSIS) == EXPECTED_DIAGNOSIS_SHA256,
        "freeze_authority_hash_exact": sha256(AUTHORITY) == EXPECTED_AUTHORITY_SHA256,
        "freeze_authority_companion_exact": companion_matches(AUTHORITY, AUTHORITY.with_suffix(AUTHORITY.suffix + ".sha256")),
        "freeze_authority_no_training_or_marker": authority.get("status") == "AUTHORIZED_NO_TRAINING_AUTHORITY"
        and authority.get("training_or_attempt_marker_authorized") is False,
        "freeze_manifest_hash_exact": sha256(FREEZE) == EXPECTED_FREEZE_SHA256,
        "freeze_manifest_companion_exact": companion_matches(FREEZE, FREEZE.with_suffix(FREEZE.suffix + ".sha256")),
        "freeze_package_tree_exact": freeze.get("package_tree_sha256") == EXPECTED_PACKAGE_TREE_SHA256,
        "freeze_marker_authority_false": freeze.get("attempt_marker_creation_authorized") is False,
        "freeze_artifact_hash_map_exact": freeze_artifact_map_exact,
        "freeze_audit_hash_exact": sha256(FREEZE_AUDIT) == EXPECTED_FREEZE_AUDIT_SHA256,
        "freeze_audit_companion_exact": companion_matches(FREEZE_AUDIT, FREEZE_AUDIT.with_suffix(FREEZE_AUDIT.suffix + ".sha256")),
        "freeze_audit_133_pass": freeze_audit.get("status") == "PASS_AWAITING_INDEPENDENT_L2"
        and freeze_audit.get("check_count") == 133
        and all(freeze_audit.get("checks", {}).values()),
        "l2_review_hash_exact": sha256(L2_REVIEW) == EXPECTED_L2_SHA256,
        "l2_review_companion_exact": companion_matches(L2_REVIEW, L2_REVIEW.with_suffix(L2_REVIEW.suffix + ".sha256")),
        "l2_review_accepts_exact_package": l2.get("status") == "ACCEPT"
        and l2.get("independent_from_constructor") is True
        and l2.get("execution_interlocks_accepted") is True
        and l2.get("freeze_manifest_sha256") == EXPECTED_FREEZE_SHA256
        and l2.get("package_tree_sha256") == EXPECTED_PACKAGE_TREE_SHA256
        and l2.get("training_executed") is False,
        "l2_accepted_artifact_hash_map_exact": accepted_hash_map_exact,
        "dataset_288_rows_exact": dataset.get("counts", {}).get("train") == 288,
        "recipe_fp32_lora_and_adam": recipe.get("lora", {}).get("master_parameter_dtype") == "float32"
        and recipe.get("optimizer", {}).get("first_moment_dtype") == "float32"
        and recipe.get("optimizer", {}).get("second_moment_dtype") == "float32",
        "recipe_candidate_epochs_1_to_4": recipe.get("checkpoint_policy", {}).get("candidate_epochs")
        == [1, 2, 3, 4],
        "offline_self_test_exact_package": self_test.get("status") == "PASS"
        and self_test.get("check_count") == 25
        and all(self_test.get("checks", {}).values())
        and self_test.get("execution_tree_sha256") == EXPECTED_PACKAGE_TREE_SHA256,
        "attempt_authority_absent": not ATTEMPT_AUTHORITY.exists(),
        "official_run_namespace_absent": not RUN_ROOT.exists(),
        "attempt_marker_absent": not list(
            ROOT.glob("build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/**/ATTEMPT_CONSUMPTION_MARKER.json")
        ),
        "benchmark_v3_frozen_l2_without_execution": isinstance(prospective, dict)
        and prospective.get("freeze_manifest_sha256") == EXPECTED_FREEZE_SHA256
        and prospective.get("package_tree_sha256") == EXPECTED_PACKAGE_TREE_SHA256
        and prospective.get("fresh_l2_review_sha256") == EXPECTED_L2_SHA256
        and prospective.get("attempt_authority_present") is False
        and prospective.get("training_executed") is False,
        "benchmark_selected_policy_null": isinstance(compression, dict)
        and compression.get("selected_policy_id") is None,
        "internal_evaluator_not_external_benchmark": isinstance(internal, dict)
        and internal.get("classification")
        == "internal_hash_bound_fixed_evaluation_contract_not_external_accelerator_benchmark",
        "v2_terminal_no_go_preserved": terminal.get("status") == "NO_GO"
        and terminal.get("failure_taxonomy") == "DEV_QUALITY_GATE_FAILURE",
        "u280_hash_exact": sha256(U280) == EXPECTED_U280_SHA256,
        "u280_vendor_tools_absent": all(
            u280.get("tool_inventory", {}).get(name) is None
            for name in ("vivado", "vitis", "v++", "xrt-smi", "xbutil", "xbmgmt", "emconfigutil", "xclbinutil")
        ),
        "u280_board_absent": u280.get("board_inventory", {}).get("pci_vendor_0x10ee_function_count") == 0
        and u280.get("board_inventory", {}).get("physical_u280_access") is False,
        "ground_truth_contains_only_binding_unknowns": unknowns_only,
        "contract_still_withholds_attempt_marker": contract.get("attempt_policy", {}).get("attempt_marker_creation_authorized")
        is False,
    }

    status = "PASS" if all(item["status"] == "PASS" for item in checklist.values()) and all(
        bindings.values()
    ) else "FAIL"
    result = {
        "schema_version": 1,
        "audit_id": "specification-stage-v3-frozen-l2-closure-v1",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "current_stage": pipeline.get("current_stage"),
        "checklist": checklist,
        "bindings": dict(sorted(bindings.items())),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                SPEC,
                BENCHMARK,
                CONTRACT,
                DIAGNOSIS,
                GROUND_TRUTH,
                PIPELINE,
                AUTHORITY,
                FREEZE,
                FREEZE_AUDIT,
                L2_REVIEW,
                DATASET_MANIFEST,
                RECIPE,
                SELF_TEST,
                U280,
                V2_TERMINAL,
            )
        },
        "claim_boundary": "Specification-stage post-L2 audit only. No attempt authority, marker, training, model evaluation, RTL simulation, formal, synthesis, timing, PPA, FPGA build, hardware emulation, or board execution occurred.",
        "next_gate": "Fresh operator attempt authority naming the exact accepted package tree and L2 review is required before an Engineer may create the sole V3 marker.",
    }
    if status != "PASS":
        raise RuntimeError(json.dumps(result, sort_keys=True))

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    companion = output.with_suffix(output.suffix + ".sha256")
    with companion.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{sha256(output)}  {output.name}\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output), "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
