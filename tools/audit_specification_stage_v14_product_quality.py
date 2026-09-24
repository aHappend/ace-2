#!/usr/bin/env python3
"""Read-only specification-stage audit for terminal V13 and frozen V14."""

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
V13_MARKER = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v13/construction_started.json"
V13_NO_GO = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v13/PRODUCT_PREFLIGHT_NO_GO.json"
V13_REVIEW = ROOT / "research/raw/specification/v13-product-quality-construction-no-go-review-replan-20260807T060033Z.json"
V14_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V14_PLAN.json"
V14_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V14_ENGINEER_TASK.json"
V14_SELF_TEST = ROOT / "build/v14-constructor-kernel-self-test.json"
V14_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v14"
V14_ADAPTER = ROOT / "tools/v14_v10_transformer_v12_lm_head64_backend.py"
V14_RUNNER = ROOT / "tools/run_option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v14_stage1.py"
ALGORITHM = ROOT / "tools/v13_v10_transformer_v12_lm_head64_backend.py"
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


def table_classes(section: str) -> list[str]:
    classes: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"} and not set(cells[0]) <= {"-", ":"}:
            classes.append(cells[0])
    return classes


def file_record_matches(record: dict[str, Any]) -> bool:
    path = ROOT / record["path"]
    return path.is_file() and path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"]


def audit() -> dict[str, Any]:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    v13_plan = load(V13_PLAN)
    v13_marker = load(V13_MARKER)
    v13_no_go = load(V13_NO_GO)
    v13_review = load(V13_REVIEW)
    v14_plan = load(V14_PLAN)
    v14_task = load(V14_TASK)
    self_test = load(V14_SELF_TEST)
    u280 = load(U280_RECHECK)
    shell = audit_shell_contract()

    shell_source = (ROOT / "rtl/ace2_shell.sv").read_text(encoding="utf-8")
    shell_header = shell_source[
        shell_source.index("module ace2_shell") : shell_source.index(");", shell_source.index("module ace2_shell")) + 2
    ]
    signed_public_ports = len(re.findall(r"^\s*(?:input|output)[^\n]*\bsigned\b", shell_header, re.MULTILINE))

    matrix_section = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix_section)
    missing_classes = [item for item in REQUIRED_CLASSES if item not in classes]

    terminal_v13 = all(
        (
            sha256(V13_MARKER) == "be54b2bfd782cbda0776c4f9c33ea557e134d72dc552d7273bdc96bb6e6c27d9",
            sha256(V13_NO_GO) == "8dcb494e89e1e199206cfcb337671bd1b9cb39454f23132697326aaf1ac7f90c",
            v13_marker.get("candidate_id", "").endswith("_v13"),
            v13_no_go.get("status") == "PRODUCT_PREFLIGHT_NO_GO",
            v13_no_go.get("error_type") == "KeyError",
            v13_no_go.get("error") == "'selector'",
            v13_no_go.get("quality_campaign_started") is False,
            v13_review.get("review_status") == "replan_requested",
            v13_review.get("failure_taxonomy") == "CONSTRUCTION_OR_CODEC_FAILURE",
            v13_review.get("reviewed_facts", {}).get("quality_conclusion") is None,
        )
    )

    inherited = v14_plan.get("inherited_v13_acceptance_contract", {})
    inherited_sections = inherited.get("sections", {})
    section_hashes_match = all(
        canonical_sha256(v13_plan[name]) == digest for name, digest in inherited_sections.items()
    )
    execution_binding = v14_plan.get("corrected_execution_binding", {})
    executable_binding = all(
        file_record_matches(execution_binding[name])
        for name in ("algorithm_backend", "v14_authority_adapter", "v14_runner", "tensor_1_diagnostic")
    )
    frozen_v14 = all(
        (
            v14_plan.get("status") == "frozen_not_started",
            v14_plan.get("stage") == "specification",
            v14_plan.get("selected_policy_id") is None,
            inherited.get("sha256") == sha256(V13_PLAN),
            inherited.get("acceptance_changes") == [],
            section_hashes_match,
            v14_task.get("task_id") == "task-91ec703b42da",
            v14_task.get("status") == "authorized_not_started",
            v14_task.get("stage") == "specification",
            v14_task.get("candidate_plan", {}).get("sha256") == sha256(V14_PLAN),
            companion_matches(V14_PLAN),
            companion_matches(V14_TASK),
            executable_binding,
            not V14_ROOT.exists(),
            self_test.get("status") == "PASS",
            self_test.get("candidate_id") == v14_plan.get("candidate_id"),
            self_test.get("task_id") == v14_task.get("task_id"),
            self_test.get("remaining_required_closure", {}).get("dynamic_scale32_named_event_count") == 312,
            self_test.get("checks", {}).get("quality_campaign_consumed") is False,
        )
    )

    active = benchmark.get("product_compression_gate", {}).get("active_v14_contract", {})
    terminal = benchmark.get("product_compression_gate", {}).get("terminal_v13_contract", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    planned = internal.get("planned_alias_safe_v10_transformer_v12_lm_head64_product_chat_v14", {})
    benchmark_closure = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure")
            == "explicit_non_benchmark_statement",
            terminal.get("status") == "PRODUCT_PREFLIGHT_NO_GO",
            active.get("plan_sha256") == sha256(V14_PLAN),
            active.get("engineer_task_sha256") == sha256(V14_TASK),
            active.get("status") == "frozen_not_started",
            planned.get("contract_sha256") == sha256(V14_PLAN),
            planned.get("engineer_task_sha256") == sha256(V14_TASK),
            planned.get("namespace_exists") is False,
            internal.get("selected_policy_id") is None,
        )
    )

    missing_spec = [
        item
        for item in (
            v14_plan["candidate_id"],
            v14_task["task_id"],
            "246,980,608 four-bit payload bytes",
            "1,926,352 explicitly accounted weight-",
            "248,906,960 total weight bytes",
            "V13 consumed its fresh construction marker",
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

    ground_truth_current = "task-91ec703b42da" in ground_truth and "Whether a fresh `option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v13` construction" not in ground_truth
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
            terminal_v13,
            frozen_v14,
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
        V13_MARKER,
        V13_NO_GO,
        V13_REVIEW,
        V14_PLAN,
        V14_PLAN.with_suffix(".sha256"),
        V14_TASK,
        V14_TASK.with_suffix(".sha256"),
        V14_SELF_TEST,
        V14_ADAPTER,
        V14_RUNNER,
        ALGORITHM,
        U280_RECHECK,
        Path(__file__),
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no V14 construction marker, full-model construction, quality campaign, RTL edit or execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, hardware emulation, board execution, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v13_terminal_construction_failure_verified": terminal_v13,
            "v13_quality_campaign_started": v13_no_go.get("quality_campaign_started"),
            "v14_plan_task_and_executable_frozen": frozen_v14,
            "v14_engineer_task_id": v14_task.get("task_id"),
            "v14_namespace_absent": not V14_ROOT.exists(),
            "v14_kernel_self_test_status": self_test.get("status"),
            "v14_full_model_and_quality_executed": False,
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
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_v14_product_quality.py --output research/raw/specification/specification-stage-audit-v14-product-quality-20260807T061000Z.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT):
            raise RuntimeError("audit output must remain inside the repository")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(raw, encoding="utf-8")
        temporary.replace(output)
    print(raw, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
