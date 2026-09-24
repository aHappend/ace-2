#!/usr/bin/env python3
"""Read-only specification-stage audit for terminal V13/V14 and frozen V15."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
V13_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V13_PLAN.json"
V14_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V14_PLAN.json"
V14_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V14_ENGINEER_TASK.json"
V14_MARKER = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14/construction_started.json"
V14_FROZEN_MATRIX = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14/quality-campaign-0001/frozen_matrix.json"
V14_NO_GO = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14/PRODUCT_PREFLIGHT_NO_GO.json"
V14_REVIEW = ROOT / "research/raw/specification/v14-product-quality-matrix-lookup-no-go-review-replan-20260807T070738Z.json"
V15_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V15_PLAN.json"
V15_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V15_ENGINEER_TASK.json"
V15_ADAPTER = ROOT / "tools/v15_v10_transformer_v12_lm_head64_backend.py"
V15_RUNNER = ROOT / "tools/run_option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v15_stage1.py"
V15_SELF_TEST = ROOT / "build/v15-constructor-and-adapter-restoration-self-test.json"
V15_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v15"
U280_RECHECK = ROOT / "research/raw/specification/u280-host-inventory-recheck-20260807T052627Z.json"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    return hashlib.sha256(raw).hexdigest()


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(".sha256")
    expected = f"{sha256(path)}  {path.relative_to(ROOT)}"
    return companion.is_file() and companion.read_text(encoding="utf-8").strip() == expected


def file_record_matches(record: dict[str, Any]) -> bool:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    return path.is_file() and path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"]


def table_classes(section: str) -> list[str]:
    classes: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"} and not set(cells[0]) <= {"-", ":"}:
            classes.append(cells[0])
    return classes


def audit() -> dict[str, Any]:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    v13_plan = load(V13_PLAN)
    v14_plan = load(V14_PLAN)
    v14_task = load(V14_TASK)
    v14_marker = load(V14_MARKER)
    v14_no_go = load(V14_NO_GO)
    v14_review = load(V14_REVIEW)
    v15_plan = load(V15_PLAN)
    v15_task = load(V15_TASK)
    self_test = load(V15_SELF_TEST)
    u280 = load(U280_RECHECK)
    shell = audit_shell_contract()

    shell_source = (ROOT / "rtl/ace2_shell.sv").read_text(encoding="utf-8")
    start = shell_source.index("module ace2_shell")
    shell_header = shell_source[start : shell_source.index(");", start) + 2]
    signed_public_ports = len(re.findall(r"^\s*(?:input|output)[^\n]*\bsigned\b", shell_header, re.MULTILINE))

    matrix_section = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix_section)
    missing_classes = [item for item in REQUIRED_CLASSES if item not in classes]

    terminal_v14 = all(
        (
            sha256(V14_MARKER) == "7c0e2d86c20d0cce1a2f63c043854030afb6184eea595fff19d95cb42a34d8ed",
            sha256(V14_FROZEN_MATRIX) == "e4426dada0be9e7a25e72b598f865e78e3db7cc852083018ba8287e972d0ac99",
            sha256(V14_NO_GO) == "b0cf069fd67802b77375d0fe987713e133bc2c12d990ea669d5fce56e93ad5fe",
            v14_marker.get("candidate_id", "").endswith("_v14"),
            v14_no_go.get("status") == "PRODUCT_PREFLIGHT_NO_GO",
            v14_no_go.get("error_type") == "KeyError",
            v14_no_go.get("error") == "'frozen_product_quality_matrix'",
            v14_no_go.get("quality_campaign_started") is False,
            v14_review.get("review_status") == "replan_requested",
            v14_review.get("failure_taxonomy") == "CONSTRUCTION_OR_CODEC_FAILURE",
            v14_review.get("reviewed_facts", {}).get("quality_conclusion") is None,
        )
    )

    v14_sections = v14_plan["inherited_v13_acceptance_contract"]["sections"]
    v15_inherited = v15_plan["inherited_v14_acceptance_contract"]
    inherited_sections_match = v15_inherited.get("sections") == v14_sections and all(
        canonical_sha256(v13_plan[name]) == digest for name, digest in v14_sections.items()
    )
    execution_binding = v15_plan.get("corrected_execution_binding", {})
    executable_binding = all(file_record_matches(record) for record in execution_binding.values())
    regression = self_test.get("adapter_restoration_regression", {})
    frozen_v15 = all(
        (
            v15_plan.get("status") == "frozen_not_started",
            v15_plan.get("stage") == "specification",
            v15_plan.get("selected_policy_id") is None,
            v15_inherited.get("sha256") == sha256(V14_PLAN),
            v15_inherited.get("acceptance_changes") == [],
            inherited_sections_match,
            v15_task.get("task_id") == "task-c0a354f6ad25",
            v15_task.get("status") == "authorized_not_started",
            v15_task.get("stage") == "specification",
            v15_task.get("candidate_plan", {}).get("sha256") == sha256(V15_PLAN),
            companion_matches(V15_PLAN),
            companion_matches(V15_TASK),
            executable_binding,
            not V15_ROOT.exists(),
            self_test.get("status") == "PASS",
            self_test.get("candidate_id") == v15_plan.get("candidate_id"),
            self_test.get("task_id") == v15_task.get("task_id"),
            regression.get("status") == "PASS",
            regression.get("successor_plan_restored") is True,
            regression.get("exact_visible_matrix_lookup_exercised") is True,
            regression.get("visible_matrix_sha256") == "312942cdb47fb294e6f7d402052c4bfec1cf94b48ecfd64995f9b65e7e3dff36",
            regression.get("official_namespace_created") is False,
            regression.get("quality_marker_created") is False,
        )
    )

    product_gate = benchmark.get("product_compression_gate", {})
    terminal_contract = product_gate.get("terminal_v14_contract", {})
    active_contract = product_gate.get("active_v15_contract", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    planned = internal.get("planned_alias_safe_v10_transformer_v12_lm_head64_product_chat_v15", {})
    benchmark_closure = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure")
            == "explicit_non_benchmark_statement",
            terminal_contract.get("status") == "PRODUCT_PREFLIGHT_NO_GO",
            active_contract.get("plan_sha256") == sha256(V15_PLAN),
            active_contract.get("engineer_task_sha256") == sha256(V15_TASK),
            active_contract.get("status") == "frozen_not_started",
            planned.get("contract_sha256") == sha256(V15_PLAN),
            planned.get("engineer_task_sha256") == sha256(V15_TASK),
            planned.get("namespace_exists") is False,
            internal.get("selected_policy_id") is None,
        )
    )

    missing_spec = [
        item
        for item in (
            v15_plan["candidate_id"],
            v15_task["task_id"],
            "246,980,608 four-bit payload bytes",
            "1,926,352 explicitly accounted weight-",
            "248,906,960 total weight bytes",
            "V14 consumed its sole construction authority",
            "BF16 token identity is not the accelerator oracle",
            "`selected_policy_id` is null",
        )
        if item not in spec
    ]

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS" if shell.get("status") == "PASS" and signed_public_ports == 0 and not missing_spec else "FAIL",
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "signed_public_port_count": signed_public_ports,
            "missing_fragments": missing_spec,
        },
        "spec.clock-reset-protocol": {
            "status": "PASS"
            if all(
                text in spec
                for text in (
                    "single-clock digital accelerator subsystem",
                    "Asynchronous active-low reset",
                    "deassert synchronously before state machines leave reset",
                    "remain stable until accepted",
                    "Completion valid; held until ready",
                )
            )
            else "FAIL",
            "clock_domains": 1,
        },
        "spec.acceptance-matrix": {
            "status": "PASS" if not missing_classes else "FAIL",
            "required_classes": list(REQUIRED_CLASSES),
            "observed_classes": classes,
            "missing_classes": missing_classes,
        },
        "spec.benchmark-interface-closure": {
            "status": "PASS" if benchmark_closure else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "closure": benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure"),
        },
    }

    ground_truth_current = "task-c0a354f6ad25" in ground_truth and "V14 quantized reference satisfies" not in ground_truth
    u280_current = all(
        (
            u280.get("observed_at_utc") == "2026-08-07T05:26:27Z",
            u280.get("u280_access") is False,
            not any(u280.get("vendor_tools_in_path", {}).values()),
        )
    )
    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            terminal_v14,
            frozen_v15,
            benchmark_closure,
            ground_truth_current,
            u280_current,
        )
    )
    inputs = (
        SPEC,
        BENCHMARK,
        PIPELINE,
        GROUND_TRUTH,
        V13_PLAN,
        V14_PLAN,
        V14_TASK,
        V14_MARKER,
        V14_FROZEN_MATRIX,
        V14_NO_GO,
        V14_REVIEW,
        V14_REVIEW.with_suffix(".sha256"),
        V15_PLAN,
        V15_PLAN.with_suffix(".sha256"),
        V15_TASK,
        V15_TASK.with_suffix(".sha256"),
        V15_ADAPTER,
        V15_RUNNER,
        V15_SELF_TEST,
        U280_RECHECK,
        Path(__file__),
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no V15 construction marker, full-model construction, quality campaign, RTL edit or execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, hardware emulation, board execution, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v14_terminal_construction_failure_verified": terminal_v14,
            "v14_quality_campaign_started": v14_no_go.get("quality_campaign_started"),
            "v15_plan_task_and_executable_frozen": frozen_v15,
            "v15_engineer_task_id": v15_task.get("task_id"),
            "v15_namespace_absent": not V15_ROOT.exists(),
            "v15_self_test_status": self_test.get("status"),
            "v15_adapter_restoration_regression_status": regression.get("status"),
            "v15_full_model_and_quality_executed": False,
            "selected_policy_id": internal.get("selected_policy_id"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "u280_local_inventory": {
            "record": str(U280_RECHECK.relative_to(ROOT)),
            "record_sha256": sha256(U280_RECHECK),
            "current_negative_inventory_verified": u280_current,
        },
        "input_bindings": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_v15_product_quality.py --output research/raw/specification/specification-stage-audit-v15-product-quality-20260807T071052Z.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
