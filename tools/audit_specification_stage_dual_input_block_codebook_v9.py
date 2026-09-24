#!/usr/bin/env python3
"""Read-only specification audit for terminal V8 and frozen V9."""

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
V8_ROOT = ROOT / "build/stage1-option-b-alias-safe-hierarchical-row-group-scale-w4-grouped-dynamic-scale32-w4a8-v8"
V8_NO_GO = V8_ROOT / "PREATTEMPT_NO_GO.json"
V8_MODEL = V8_ROOT / "preattempt/model_only_hierarchical_reconstruction.json"
V8_QUALITY = V8_ROOT / "preattempt/non_evaluator_quality.json"
V8_DYNAMIC = V8_ROOT / "preattempt/dynamic_preflight.json"
V8_REVIEW = ROOT / "research/raw/specification/hierarchical-row-group-scale-v8-review-done-20260806T210801Z.json"
V8_REVIEW_SUM = V8_REVIEW.with_suffix(".sha256")
V9_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_DUAL_INPUT_BLOCK_CODEBOOK_ROW_GROUP_SELECTOR_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V9_PLAN.json"
V9_PLAN_SUM = V9_PLAN.with_suffix(".sha256")
V9_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_DUAL_INPUT_BLOCK_CODEBOOK_ROW_GROUP_SELECTOR_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V9_ENGINEER_TASK.json"
V9_TASK_SUM = V9_TASK.with_suffix(".sha256")
V9_ROOT = ROOT / "build/stage1-option-b-alias-safe-dual-input-block-codebook-row-group-selector-w4-grouped-dynamic-scale32-w4a8-v9"
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
    no_go = load(V8_NO_GO)
    model = load(V8_MODEL)
    quality = load(V8_QUALITY)
    dynamic = load(V8_DYNAMIC)
    review = load(V8_REVIEW)
    plan = load(V9_PLAN)
    task = load(V9_TASK)
    shell = audit_shell_contract()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in classes]
    required_spec = (
        "option_b_alias_safe_dual_input_block_codebook_row_group_selector_w4_grouped_dynamic_scale32_w4a8_v9",
        "2495.1003852741437",
        "task-9d075ab327ee",
        "two complete",
        "selector bit",
        "4.035242101648351",
        "4.035971912985919",
        "`selected_policy_id` is null",
    )
    missing_spec = [item for item in required_spec if item not in spec]

    activation = dynamic.get("activation_execution", {})
    v8_terminal = all(
        (
            no_go.get("status") == "PREATTEMPT_NO_GO",
            no_go.get("attempts_consumed") == 0,
            no_go.get("official_execution_marker_exists") is False,
            no_go.get("gate_passes", {}).get("fresh_v7_and_169_model_closure") is False,
            no_go.get("gate_passes", {}).get("two_build_alias_and_manifest_closure") is True,
            no_go.get("gate_passes", {}).get("dynamic_312_event_closure") is True,
            no_go.get("gate_passes", {}).get("non_evaluator_quality_8_of_8") is False,
            model.get("fresh_v7_exact_reproduction_count") == 169,
            model.get("all_169_tensors_legal") is True,
            model.get("per_tensor_strictly_improved_count") == 23,
            model.get("aggregate_fresh_v7_bf16_sse") == 2303.224223882074,
            model.get("aggregate_v8_hierarchical_bf16_sse") == 2495.1003852741437,
            model.get("aggregate_strictly_improved_against_fresh_v7") is False,
            quality.get("exact_bf16_sequence_match_count") == 2,
            quality.get("response_gate_outcome_match_count") == 8,
            quality.get("official_nine_prompt_inputs_used") is False,
            dynamic.get("status") == "PASS_ALL_312_DYNAMIC_EVENTS",
            activation.get("named_boundary_count") == 312,
            activation.get("saturation_count") == 0,
            activation.get("clipping_count") == 0,
            activation.get("reserved_minus_128_produced") is False,
            companion_matches(V8_REVIEW, V8_REVIEW_SUM),
            review.get("status") == "done",
            review.get("failure_taxonomy") == "NO_GO_MODEL_RECONSTRUCTION_AND_NON_EVALUATOR_QUALITY_GATE",
            review.get("reviewed_facts", {}).get("official_attempts_consumed") == 0,
            review.get("bound_evidence", {}).get("terminal_no_go", {}).get("sha256") == sha256(V8_NO_GO),
            not (V8_ROOT / "PREATTEMPT_READY.json").exists(),
            not (V8_ROOT / "attempt-0001").exists(),
        )
    )

    policy = plan.get("weight_policy", {})
    block = policy.get("input_block", {})
    dual = policy.get("dual_codebook", {})
    selector = policy.get("row_group_selector", {})
    limits = plan.get("compression_and_accelerator_limits", {})
    namespace = task.get("fresh_attempt_namespace", {})
    selector_by_family = limits.get("standard_decoder_layer_selector_bytes_by_family", {})
    computed_selector_bytes = sum(selector_by_family.values())
    computed_standard_total = sum(
        limits.get(name, -1)
        for name in (
            "standard_decoder_layer_packed_w4_payload_bytes",
            "standard_decoder_layer_row_scale_bytes",
            "standard_decoder_layer_dual_codebook_bytes",
            "standard_decoder_layer_selector_bytes",
        )
    )
    computed_standard_bpw = computed_standard_total * 8 / limits.get("standard_decoder_layer_weight_count", 1)
    computed_lm_total = (
        limits.get("lm_head_weight_count", 0) // 2
        + 151936 * 4
        + limits.get("lm_head_dual_codebook_bytes", -1)
        + limits.get("lm_head_selector_bytes_after_16_byte_alignment", -1)
    )
    computed_lm_bpw = computed_lm_total * 8 / limits.get("lm_head_weight_count", 1)
    v9_frozen = all(
        (
            companion_matches(V9_PLAN, V9_PLAN_SUM),
            companion_matches(V9_TASK, V9_TASK_SUM),
            plan.get("candidate_id") == "option_b_alias_safe_dual_input_block_codebook_row_group_selector_w4_grouped_dynamic_scale32_w4a8_v9",
            plan.get("stage") == "specification",
            plan.get("official_attempt", {}).get("attempts_authorized_by_this_file") == 0,
            policy.get("payload_bits_per_weight") == 4,
            policy.get("payload_semantics") == "unsigned codebook index 0 through 15",
            block.get("lanes") == 128,
            dual.get("banks_per_input_block") == 2,
            dual.get("bytes_per_input_block") == 32,
            selector.get("bits_per_row_group_per_input_block") == 1,
            selector.get("transformer_row_group_size_by_family", {}).get("down_proj") == 2,
            selector.get("transformer_row_group_size_by_family", {}).get("q_proj") == 1,
            selector.get("lm_head_row_group_size") == 32,
            computed_selector_bytes == 12432,
            limits.get("standard_decoder_layer_selector_bytes") == 12432,
            computed_standard_total == 7520400,
            limits.get("standard_decoder_layer_total_weight_bytes") == 7520400,
            abs(computed_standard_bpw - 4.035242101648351) < 1e-15,
            limits.get("standard_decoder_layer_exact_bits_per_weight") == 4.035242101648351,
            computed_lm_total == 68679456,
            limits.get("lm_head_total_weight_bytes") == 68679456,
            abs(computed_lm_bpw - 4.035971912985919) < 1e-15,
            limits.get("lm_head_exact_bits_per_weight") == 4.035971912985919,
            limits.get("whole_model_total_weight_bytes") == 249169056,
            limits.get("whole_model_exact_bits_per_weight") == 4.035443236094066,
            limits.get("maximum_total_stored_bits_per_weight") == 4.036,
            plan.get("accelerator_facing_format", {}).get("public_ace2_shell_contract_changed") is False,
            task.get("task_id") == "task-9d075ab327ee",
            task.get("status") == "authorized_not_started",
            task.get("candidate_plan", {}).get("sha256") == sha256(V9_PLAN),
            namespace.get("attempt_budget") == 1,
            namespace.get("attempts_consumed_at_task_freeze") == 0,
            namespace.get("execution_marker_permitted_during_runner_build_construction_or_preflight") is False,
            not V9_ROOT.exists(),
        )
    )

    internal = benchmark.get("internal_acceptance_contract", {})
    terminal = internal.get("terminal_alias_safe_hierarchical_row_group_scale_v8", {})
    planned = internal.get("planned_alias_safe_dual_input_block_codebook_row_group_selector_v9", {})
    benchmark_routing = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("behavior_interface", {}).get("parameter_count") == 12,
            benchmark.get("specification_stage_evidence", {}).get("behavior_interface", {}).get("port_count") == 64,
            terminal.get("preflight_status") == "PREATTEMPT_NO_GO",
            terminal.get("review_record_sha256") == sha256(V8_REVIEW),
            planned.get("contract_sha256") == sha256(V9_PLAN),
            planned.get("engineer_task_sha256") == sha256(V9_TASK),
            planned.get("attempt_marker_exists") is False,
            planned.get("standard_decoder_layer_exact_bits_per_weight") == 4.035242101648351,
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
        (pipeline.get("current_stage") == "specification", v8_terminal, v9_frozen, benchmark_routing)
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no candidate construction, official input access, RTL execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v8_terminal_fresh_review_verified": v8_terminal,
            "v8_aggregate_fresh_v7_bf16_sse": model.get("aggregate_fresh_v7_bf16_sse"),
            "v8_aggregate_bf16_sse": model.get("aggregate_v8_hierarchical_bf16_sse"),
            "v8_exact_sequence_matches": quality.get("exact_bf16_sequence_match_count"),
            "v8_official_attempts_consumed": no_go.get("attempts_consumed"),
            "v9_plan_and_task_frozen": v9_frozen,
            "v9_engineer_task_id": task.get("task_id"),
            "v9_namespace_absent": not V9_ROOT.exists(),
            "v9_execution_marker_absent": not (V9_ROOT / "attempt-0001/execution_started.json").exists(),
            "v9_standard_decoder_layer_exact_bits_per_weight": limits.get("standard_decoder_layer_exact_bits_per_weight"),
            "v9_lm_head_exact_bits_per_weight": limits.get("lm_head_exact_bits_per_weight"),
            "benchmark_routing_matches": benchmark_routing,
            "selected_policy_id": internal.get("selected_policy_id"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "binding_unknowns": {"count": len(unknowns), "items": unknowns},
        "input_bindings": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                SPEC, BENCHMARK, PIPELINE, GROUND_TRUTH, V8_NO_GO, V8_MODEL,
                V8_QUALITY, V8_DYNAMIC, V8_REVIEW, V8_REVIEW_SUM, V9_PLAN,
                V9_PLAN_SUM, V9_TASK, V9_TASK_SUM, Path(__file__),
            )
        },
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_dual_input_block_codebook_v9.py",
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
