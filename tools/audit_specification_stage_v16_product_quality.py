#!/usr/bin/env python3
"""Read-only specification-stage audit for terminal V15 and frozen V16."""

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
V15_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V15_PLAN.json"
V15_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V15_ENGINEER_TASK.json"
V15_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v15"
V15_MARKER = V15_ROOT / "construction_started.json"
V15_QUALITY_MARKER = V15_ROOT / "quality-campaign-0001/execution_started.json"
V15_FAILURE = V15_ROOT / "quality-campaign-0001/failure.json"
V15_SUBMISSION = V15_ROOT / "fresh_reviewer_submission.json"
V15_REVIEW = ROOT / "research/raw/specification/v15-evaluator-counter-mismatch-review-done-20260807T084955Z.json"
V16_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V16_PLAN.json"
V16_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V16_ENGINEER_TASK.json"
V16_ADAPTER = ROOT / "tools/v16_v10_transformer_v12_lm_head64_backend.py"
V16_RUNNER = ROOT / "tools/run_option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v16_stage1.py"
V16_SELF_TEST = ROOT / "build/v16-evaluator-production-path-self-test.json"
V16_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v16"
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
    expected = f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
    return companion.is_file() and companion.read_text(encoding="utf-8").strip() == expected


def file_record_matches(record: dict[str, Any]) -> bool:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    return path.is_file() and path.stat().st_size == int(record["bytes"]) and sha256(path) == record["sha256"]


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
    v15_plan = load(V15_PLAN)
    v15_task = load(V15_TASK)
    v15_marker = load(V15_MARKER)
    v15_quality_marker = load(V15_QUALITY_MARKER)
    v15_failure = load(V15_FAILURE)
    v15_submission = load(V15_SUBMISSION)
    v15_review = load(V15_REVIEW)
    v16_plan = load(V16_PLAN)
    v16_task = load(V16_TASK)
    self_test = load(V16_SELF_TEST)
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

    terminal_v15 = all(
        (
            sha256(V15_MARKER) == "08aeb5a9e5621a137aeb2c7ba39d54c61e33ee8ab230b37567a56f2ba02ed508",
            sha256(V15_QUALITY_MARKER) == "4b10a51df8678821bf7fa61e2898a0805b92fd53fe45fa18b12e32ade0b27d54",
            sha256(V15_FAILURE) == "371c4217c5f7b2a1b823b5536d9523033fb494624427a5197cdc1b428a845446",
            sha256(V15_SUBMISSION) == "b1784394460c36285fec04721a1ead21d1d2b46f38c1096adb45306638726eb8",
            sha256(V15_REVIEW) == "4a5db7140876c3b652c410b36ccda0d9a11003b16c88a7cf33a33b3cb35edca9",
            v15_marker.get("candidate_id", "").endswith("_v15"),
            v15_quality_marker.get("authorization_consumed") is True,
            v15_failure.get("status") == "EVALUATOR_FAILURE_AFTER_CAMPAIGN_CONSUMPTION",
            v15_failure.get("error") == "activation event call counts differ across boundaries",
            v15_submission.get("terminal", {}).get("quality_conclusion") is None,
            v15_submission.get("quality_execution", {}).get("completed_response_generation_trace_count") == 23,
            v15_review.get("review_status") == "done",
            v15_review.get("quality_execution", {}).get("actual_model_forward_count") == 461,
            v15_review.get("quality_conclusion") is None,
            v15_review.get("selected_policy_id") is None,
        )
    )

    v15_sections = v15_plan["inherited_v14_acceptance_contract"]["sections"]
    v16_inherited = v16_plan["inherited_v15_acceptance_contract"]
    inherited_sections_match = v16_inherited.get("sections") == v15_sections and all(
        canonical_sha256(v13_plan[name]) == digest for name, digest in v15_sections.items()
    )
    executable_binding = all(file_record_matches(record) for record in v16_plan.get("corrected_execution_binding", {}).values())
    restoration = self_test.get("adapter_restoration_regression", {})
    evaluator = self_test.get("evaluator_production_path_regression", {})
    frozen_v16 = all(
        (
            v16_plan.get("status") == "frozen_not_started",
            v16_plan.get("stage") == "specification",
            v16_plan.get("claim_boundary", {}).get("selected_policy_id") is None,
            v16_inherited.get("sha256") == sha256(V15_PLAN),
            v16_inherited.get("acceptance_changes") == [],
            inherited_sections_match,
            v16_task.get("task_id") == "task-7c95f6e2a841",
            v16_task.get("status") == "authorized_not_started",
            v16_task.get("stage") == "specification",
            v16_task.get("candidate_plan", {}).get("sha256") == sha256(V16_PLAN),
            companion_matches(V16_PLAN),
            companion_matches(V16_TASK),
            executable_binding,
            not V16_ROOT.exists(),
            self_test.get("status") == "PASS",
            sha256(V16_SELF_TEST) == "78bbd197a40ab9d55cc4a718b6acd6f2df460984fe8c571306afac04232437bf",
            self_test.get("candidate_id") == v16_plan.get("candidate_id"),
            self_test.get("task_id") == v16_task.get("task_id"),
            restoration.get("status") == "PASS",
            restoration.get("visible_matrix_sha256") == "312942cdb47fb294e6f7d402052c4bfec1cf94b48ecfd64995f9b65e7e3dff36",
            evaluator.get("status") == "PASS",
            evaluator.get("response_count") == 3,
            evaluator.get("actual_model_forward_count") == 12,
            evaluator.get("variable_length_model_forward_counts") == [3, 4, 5],
            evaluator.get("failure_safe_raw_output_count") == 3,
            evaluator.get("terminal_quality_bundle_verification") == "VERIFIED",
            evaluator.get("official_namespace_created") is False,
            evaluator.get("quality_marker_created") is False,
        )
    )

    product_gate = benchmark.get("product_compression_gate", {})
    terminal_contract = product_gate.get("terminal_v15_contract", {})
    active_contract = product_gate.get("active_v16_contract", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    planned = internal.get("planned_alias_safe_v10_transformer_v12_lm_head64_product_chat_v16", {})
    benchmark_closure = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure")
            == "explicit_non_benchmark_statement",
            terminal_contract.get("status") == "EVALUATOR_FAILURE_AFTER_CAMPAIGN_CONSUMPTION",
            terminal_contract.get("quality_conclusion") is None,
            active_contract.get("plan_sha256") == sha256(V16_PLAN),
            active_contract.get("engineer_task_sha256") == sha256(V16_TASK),
            active_contract.get("status") == "frozen_not_started",
            active_contract.get("evaluator_production_path_self_test_sha256") == sha256(V16_SELF_TEST),
            planned.get("contract_sha256") == sha256(V16_PLAN),
            planned.get("engineer_task_sha256") == sha256(V16_TASK),
            planned.get("namespace_exists") is False,
            internal.get("selected_policy_id") is None,
        )
    )

    missing_spec = [
        item
        for item in (
            v16_plan["candidate_id"],
            v16_task["task_id"],
            "requiring 461",
            "12 actual model forwards",
            "failure-safe campaign publication",
            "246,980,608 four-bit payload bytes",
            "1,926,352 explicitly accounted weight-",
            "248,906,960 total weight bytes",
            "BF16 token identity is not the accelerator oracle",
            "`selected_policy_id` remains null",
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

    ground_truth_current = all(
        (
            "Whether sealed V16" in ground_truth,
            "task-7c95f6e2a841" in ground_truth,
            "single authorized V16 quality campaign" in ground_truth,
            "Whether sealed successor" not in ground_truth,
            "Whether that fixed V15 quantized reference" not in ground_truth,
        )
    )
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
            terminal_v15,
            frozen_v16,
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
        V15_PLAN,
        V15_TASK,
        V15_MARKER,
        V15_QUALITY_MARKER,
        V15_FAILURE,
        V15_SUBMISSION,
        V15_REVIEW,
        V15_REVIEW.with_suffix(".sha256"),
        V16_PLAN,
        V16_PLAN.with_suffix(".sha256"),
        V16_TASK,
        V16_TASK.with_suffix(".sha256"),
        V16_ADAPTER,
        V16_RUNNER,
        V16_SELF_TEST,
        U280_RECHECK,
        Path(__file__),
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no V16 construction marker, full-model construction, quality campaign, RTL edit or execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, hardware emulation, board execution, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v15_terminal_evaluator_failure_verified": terminal_v15,
            "v15_completed_response_generation_count": v15_review.get("quality_execution", {}).get("completed_response_generation_count"),
            "v15_actual_model_forward_count": v15_review.get("quality_execution", {}).get("actual_model_forward_count"),
            "v15_quality_conclusion": v15_review.get("quality_conclusion"),
            "v16_plan_task_and_executable_frozen": frozen_v16,
            "v16_engineer_task_id": v16_task.get("task_id"),
            "v16_namespace_absent": not V16_ROOT.exists(),
            "v16_self_test_status": self_test.get("status"),
            "v16_evaluator_regression_status": evaluator.get("status"),
            "v16_full_model_and_quality_executed": False,
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
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_specification_stage_v16_product_quality.py --output research/raw/specification/specification-stage-audit-v16-product-quality-20260807T091500Z.json",
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
