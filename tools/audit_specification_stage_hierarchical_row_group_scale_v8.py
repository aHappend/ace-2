#!/usr/bin/env python3
"""Read-only specification audit for terminal V7 and frozen V8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
V7_ROOT = ROOT / "build/stage1-option-b-alias-safe-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v7"
V7_NO_GO = V7_ROOT / "PREATTEMPT_NO_GO.json"
V7_MODEL = V7_ROOT / "preattempt/model_only_input_block_reconstruction.json"
V7_QUALITY = V7_ROOT / "preattempt/non_evaluator_quality.json"
V7_DYNAMIC = V7_ROOT / "preattempt/dynamic_preflight.json"
V7_REVIEW = ROOT / "research/raw/specification/input-block-codebook-v7-review-done-20260806T193852Z.json"
V7_REVIEW_SUM = V7_REVIEW.with_suffix(".sha256")
V8_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_HIERARCHICAL_ROW_GROUP_SCALE_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V8_PLAN.json"
V8_PLAN_SUM = V8_PLAN.with_suffix(".sha256")
V8_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_HIERARCHICAL_ROW_GROUP_SCALE_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V8_ENGINEER_TASK.json"
V8_TASK_SUM = V8_TASK.with_suffix(".sha256")
V8_ROOT = ROOT / "build/stage1-option-b-alias-safe-hierarchical-row-group-scale-w4-grouped-dynamic-scale32-w4a8-v8"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def companion_matches(path: Path, companion: Path) -> bool:
    expected = f"{sha256(path)}  {path.relative_to(ROOT)}"
    return companion.is_file() and companion.read_text(encoding="utf-8").strip() == expected


def table_classes(section: str) -> list[str]:
    out: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"} and not set(cells[0]) <= {"-", ":"}:
            out.append(cells[0])
    return out


def audit() -> dict:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    no_go = load(V7_NO_GO)
    model = load(V7_MODEL)
    quality = load(V7_QUALITY)
    dynamic = load(V7_DYNAMIC)
    review = load(V7_REVIEW)
    plan = load(V8_PLAN)
    task = load(V8_TASK)
    shell = audit_shell_contract()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in classes]
    required_spec = (
        "option_b_alias_safe_hierarchical_row_group_scale_w4_grouped_dynamic_scale32_w4a8_v8",
        "2/8 exact BF16",
        "task-c4731ac4e0b8",
        "one 4-bit selector",
        "signed 4-bit exponent delta",
        "4.03489010989011",
        "`selected_policy_id` is null",
    )
    missing_spec = [item for item in required_spec if item not in spec]

    activation = dynamic.get("activation_execution", {})
    v7_terminal = all(
        (
            no_go.get("status") == "PREATTEMPT_NO_GO",
            no_go.get("attempts_consumed") == 0,
            no_go.get("official_execution_marker_exists") is False,
            no_go.get("gate_passes", {}).get("fresh_v4_and_169_model_closure") is True,
            no_go.get("gate_passes", {}).get("two_build_alias_and_manifest_closure") is True,
            no_go.get("gate_passes", {}).get("dynamic_312_event_closure") is True,
            no_go.get("gate_passes", {}).get("non_evaluator_quality_8_of_8") is False,
            model.get("fresh_v4_exact_reproduction_count") == 169,
            model.get("per_tensor_non_regressing_count") == 169,
            model.get("per_tensor_strictly_improved_count") == 169,
            model.get("aggregate_fresh_v4_bf16_sse") == 2387.058149454451,
            model.get("aggregate_v7_input_block_bf16_sse") == 2303.224223882074,
            quality.get("exact_bf16_sequence_match_count") == 2,
            quality.get("response_gate_outcome_match_count") == 8,
            quality.get("official_nine_prompt_inputs_used") is False,
            activation.get("named_boundary_count") == 312,
            activation.get("saturation_count") == 0,
            activation.get("reserved_minus_128_produced") is False,
            companion_matches(V7_REVIEW, V7_REVIEW_SUM),
            review.get("status") == "done",
            review.get("failure_taxonomy") == "NO_GO_NON_EVALUATOR_QUALITY_GATE",
            review.get("reviewed_facts", {}).get("official_attempts_consumed") == 0,
            review.get("bound_evidence", {}).get("terminal_no_go", {}).get("sha256") == sha256(V7_NO_GO),
            not (V7_ROOT / "PREATTEMPT_READY.json").exists(),
            not (V7_ROOT / "attempt-0001").exists(),
        )
    )

    weight = plan.get("weight_policy", {})
    limits = plan.get("compression_and_accelerator_limits", {})
    namespace = task.get("fresh_attempt_namespace", {})
    computed_standard_total = sum(
        limits.get(name, -1)
        for name in (
            "standard_decoder_layer_packed_w4_payload_bytes",
            "standard_decoder_layer_group_delta_bytes",
            "standard_decoder_layer_row_selector_bytes",
            "standard_decoder_layer_row_base_table_bytes",
        )
    )
    computed_standard_bpw = computed_standard_total * 8 / limits.get("standard_decoder_layer_weight_count", 1)
    v8_frozen = all(
        (
            companion_matches(V8_PLAN, V8_PLAN_SUM),
            companion_matches(V8_TASK, V8_TASK_SUM),
            plan.get("candidate_id") == "option_b_alias_safe_hierarchical_row_group_scale_w4_grouped_dynamic_scale32_w4a8_v8",
            plan.get("stage") == "specification",
            plan.get("official_attempt", {}).get("attempts_authorized_by_this_file") == 0,
            weight.get("payload_bits_per_weight") == 4,
            weight.get("payload_semantics") == "signed two's-complement integer -8 through +7",
            weight.get("weight_group", {}).get("lanes") == 128,
            weight.get("row_base_selector", {}).get("bits_per_output_row") == 4,
            weight.get("group_exponent_delta", {}).get("bits_per_128_weight_group") == 4,
            weight.get("group_exponent_delta", {}).get("semantics") == "signed two's-complement exponent delta -8 through +7",
            limits.get("standard_decoder_layer_total_weight_bytes") == 7519744,
            computed_standard_total == 7519744,
            abs(computed_standard_bpw - 4.03489010989011) < 1e-15,
            limits.get("standard_decoder_layer_exact_bits_per_weight") == 4.03489010989011,
            limits.get("maximum_total_stored_bits_per_weight") == 4.036,
            limits.get("lm_head_exact_bits_per_weight") == 4.035718046696354,
            limits.get("whole_model_exact_bits_per_weight") == 4.03511828750539,
            plan.get("accelerator_facing_format", {}).get("public_ace2_shell_contract_changed") is False,
            task.get("task_id") == "task-c4731ac4e0b8",
            task.get("status") == "authorized_not_started",
            task.get("candidate_plan", {}).get("sha256") == sha256(V8_PLAN),
            namespace.get("attempt_budget") == 1,
            namespace.get("attempts_consumed_at_task_freeze") == 0,
            namespace.get("execution_marker_permitted_during_runner_build_construction_or_preflight") is False,
            not V8_ROOT.exists(),
        )
    )

    internal = benchmark.get("internal_acceptance_contract", {})
    terminal = internal.get("terminal_alias_safe_input_block_codebook_v7", {})
    planned = internal.get("planned_alias_safe_hierarchical_row_group_scale_v8", {})
    benchmark_routing = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("behavior_interface", {}).get("parameter_count") == 12,
            benchmark.get("specification_stage_evidence", {}).get("behavior_interface", {}).get("port_count") == 64,
            terminal.get("preflight_status") == "PREATTEMPT_NO_GO",
            terminal.get("review_record_sha256") == sha256(V7_REVIEW),
            planned.get("contract_sha256") == sha256(V8_PLAN),
            planned.get("engineer_task_sha256") == sha256(V8_TASK),
            planned.get("attempt_marker_exists") is False,
            planned.get("standard_decoder_layer_exact_bits_per_weight") == 4.03489010989011,
            internal.get("selected_policy_id") is None,
        )
    )

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS" if shell.get("status") == "PASS" and not missing_spec else "FAIL",
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "missing_fragments": missing_spec,
        },
        "spec.clock-reset-protocol": {
            "status": "PASS" if all(x in spec for x in (
                "single-clock digital accelerator subsystem",
                "Asynchronous active-low reset",
                "deassert synchronously before state machines leave reset",
                "remain stable until accepted",
                "Completion valid; held until ready",
            )) else "FAIL",
            "clock_domains": 1,
        },
        "spec.acceptance-matrix": {
            "status": "PASS" if not missing_classes else "FAIL",
            "required_classes": list(REQUIRED_CLASSES),
            "observed_classes": classes,
            "missing_classes": missing_classes,
        },
        "spec.benchmark-interface-closure": {
            "status": "PASS" if benchmark_routing else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "closure": benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure"),
        },
    }

    unknowns = [line[2:] for line in GROUND_TRUTH.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]
    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (pipeline.get("current_stage") == "specification", v7_terminal, v8_frozen, benchmark_routing)
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no candidate construction, official input access, RTL execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v7_terminal_fresh_review_verified": v7_terminal,
            "v7_exact_sequence_matches": quality.get("exact_bf16_sequence_match_count"),
            "v7_official_attempts_consumed": no_go.get("attempts_consumed"),
            "v8_plan_and_task_frozen": v8_frozen,
            "v8_engineer_task_id": task.get("task_id"),
            "v8_namespace_absent": not V8_ROOT.exists(),
            "v8_execution_marker_absent": not (V8_ROOT / "attempt-0001/execution_started.json").exists(),
            "v8_standard_decoder_layer_exact_bits_per_weight": limits.get("standard_decoder_layer_exact_bits_per_weight"),
            "benchmark_routing_matches": benchmark_routing,
            "selected_policy_id": internal.get("selected_policy_id"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "binding_unknowns": {"count": len(unknowns), "items": unknowns},
        "input_bindings": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                SPEC, BENCHMARK, PIPELINE, GROUND_TRUTH, V7_NO_GO, V7_MODEL,
                V7_QUALITY, V7_DYNAMIC, V7_REVIEW, V7_REVIEW_SUM, V8_PLAN,
                V8_PLAN_SUM, V8_TASK, V8_TASK_SUM, Path(__file__),
            )
        },
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_hierarchical_row_group_scale_v8.py",
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
