#!/usr/bin/env python3
"""Read-only specification audit for terminal V9 and frozen V10."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
SCALE_CONTRACT = ROOT / "tools/ace2_quality_contracts.py"
V9_ROOT = ROOT / "build/stage1-option-b-alias-safe-dual-input-block-codebook-row-group-selector-w4-grouped-dynamic-scale32-w4a8-v9"
V9_NO_GO = V9_ROOT / "PREATTEMPT_NO_GO.json"
V9_MODEL = V9_ROOT / "preattempt/model_only_dual_codebook_reconstruction.json"
V9_QUALITY = V9_ROOT / "preattempt/non_evaluator_quality.json"
V9_DYNAMIC = V9_ROOT / "preattempt/dynamic_preflight.json"
V9_REVIEW = ROOT / "research/raw/specification/dual-input-block-codebook-row-group-selector-v9-review-done-20260806T231214Z.json"
V9_REVIEW_SUM = V9_REVIEW.with_suffix(".sha256")
V10_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_PACKED_SCALE24_PER_ROW_SELECTOR_DUAL_INPUT_BLOCK_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V10_PLAN.json"
V10_PLAN_SUM = V10_PLAN.with_suffix(".sha256")
V10_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_PACKED_SCALE24_PER_ROW_SELECTOR_DUAL_INPUT_BLOCK_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V10_ENGINEER_TASK.json"
V10_TASK_SUM = V10_TASK.with_suffix(".sha256")
V10_ROOT = ROOT / "build/stage1-option-b-alias-safe-packed-scale24-per-row-selector-dual-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v10"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def companion_matches(path: Path, companion: Path) -> bool:
    expected = f"{sha256(path)}  {path.relative_to(ROOT)}"
    return companion.is_file() and companion.read_text(encoding="utf-8").strip() == expected


def aligned_bytes(bit_count: int, alignment: int = 16) -> int:
    return math.ceil(math.ceil(bit_count / 8) / alignment) * alignment


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
    scale_contract = SCALE_CONTRACT.read_text(encoding="utf-8")
    no_go = load(V9_NO_GO)
    model = load(V9_MODEL)
    quality = load(V9_QUALITY)
    dynamic = load(V9_DYNAMIC)
    review = load(V9_REVIEW)
    plan = load(V10_PLAN)
    task = load(V10_TASK)
    shell = audit_shell_contract()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in classes]
    required_spec = (
        "option_b_alias_safe_packed_scale24_per_row_selector_dual_input_block_codebook_w4_grouped_dynamic_scale32_w4a8_v10",
        "2176.5828747795154",
        "task-b87978f9a5a8",
        "lossless storage encoding",
        "one codebook-bank selector per output row",
        "4.029584478021978",
        "4.034611377722951",
        "`selected_policy_id` is null",
    )
    missing_spec = [item for item in required_spec if item not in spec]

    activation = dynamic.get("activation_execution", {})
    v9_terminal = all(
        (
            no_go.get("status") == "PREATTEMPT_NO_GO",
            no_go.get("attempts_consumed") == 0,
            no_go.get("official_execution_marker_exists") is False,
            no_go.get("gate_passes", {}).get("fresh_v7_and_169_model_closure") is True,
            no_go.get("gate_passes", {}).get("two_build_alias_and_manifest_closure") is True,
            no_go.get("gate_passes", {}).get("dynamic_312_event_closure") is True,
            no_go.get("gate_passes", {}).get("non_evaluator_quality_8_of_8") is False,
            model.get("fresh_v7_exact_reproduction_count") == 169,
            model.get("per_tensor_non_regressing_count") == 169,
            model.get("per_tensor_strictly_improved_count") == 169,
            model.get("aggregate_fresh_v7_bf16_sse") == 2303.224223882074,
            model.get("aggregate_v9_dual_codebook_bf16_sse") == 2176.5828747795154,
            model.get("aggregate_strictly_improved_against_fresh_v7") is True,
            quality.get("exact_bf16_sequence_match_count") == 2,
            quality.get("response_gate_outcome_match_count") == 8,
            quality.get("official_nine_prompt_inputs_used") is False,
            dynamic.get("status") == "PASS_ALL_312_DYNAMIC_EVENTS",
            activation.get("named_boundary_count") == 312,
            activation.get("minimum_delta") == -14,
            activation.get("maximum_delta") == 2,
            activation.get("saturation_count") == 0,
            activation.get("clipping_count") == 0,
            activation.get("reserved_minus_128_produced") is False,
            activation.get("transport_metadata_bytes_per_decoder_layer_token") == 832,
            companion_matches(V9_REVIEW, V9_REVIEW_SUM),
            review.get("status") == "done",
            review.get("failure_taxonomy") == "NO_GO_NON_EVALUATOR_QUALITY_GATE",
            review.get("reviewed_facts", {}).get("official_attempts_consumed") == 0,
            review.get("bound_evidence", {}).get("terminal_no_go", {}).get("sha256") == sha256(V9_NO_GO),
            not (V9_ROOT / "PREATTEMPT_READY.json").exists(),
            not (V9_ROOT / "attempt-0001").exists(),
        )
    )

    policy = plan.get("weight_policy", {})
    scale = policy.get("row_scale", {})
    selector = policy.get("row_selector", {})
    limits = plan.get("compression_and_accelerator_limits", {})
    namespace = task.get("fresh_attempt_namespace", {})
    selector_by_family = limits.get("standard_decoder_layer_selector_bytes_by_family", {})
    computed_standard_selector = sum(selector_by_family.values())
    computed_standard_scale = limits.get("standard_decoder_layer_output_row_count", 0) * 3
    computed_standard_total = sum(
        (
            limits.get("standard_decoder_layer_packed_w4_payload_bytes", -1),
            computed_standard_scale,
            limits.get("standard_decoder_layer_dual_codebook_bytes", -1),
            computed_standard_selector,
        )
    )
    computed_standard_bpw = computed_standard_total * 8 / limits.get("standard_decoder_layer_weight_count", 1)
    computed_lm_selector = aligned_bytes(151936 * 7)
    computed_lm_scale = 151936 * 3
    computed_lm_total = 136134656 // 2 + computed_lm_scale + 224 + computed_lm_selector
    computed_lm_bpw = computed_lm_total * 8 / 136134656
    computed_whole_total = sum(
        (
            limits.get("whole_model_packed_w4_payload_bytes", -1),
            limits.get("whole_model_packed_scale24_bytes", -1),
            limits.get("whole_model_dual_codebook_bytes", -1),
            limits.get("whole_model_selector_bytes", -1),
        )
    )
    computed_whole_bpw = computed_whole_total * 8 / limits.get("whole_model_weight_count", 1)
    scale32_reserved_byte_frozen = all(
        fragment in scale_contract
        for fragment in (
            "(record >> 24) != 0",
            "return significand | ((exponent & 0xFF) << 16)",
            "Scale32 reserved byte must be zero",
        )
    )
    v10_frozen = all(
        (
            companion_matches(V10_PLAN, V10_PLAN_SUM),
            companion_matches(V10_TASK, V10_TASK_SUM),
            plan.get("candidate_id") == "option_b_alias_safe_packed_scale24_per_row_selector_dual_input_block_codebook_w4_grouped_dynamic_scale32_w4a8_v10",
            plan.get("stage") == "specification",
            plan.get("official_attempt", {}).get("attempts_authorized_by_this_file") == 0,
            policy.get("payload_bits_per_weight") == 4,
            policy.get("input_block", {}).get("lanes") == 128,
            policy.get("dual_codebook", {}).get("banks_per_input_block") == 2,
            scale.get("stored_bytes_per_output_row") == 3,
            scale.get("lossless_for_all_legal_scale32_records") is True,
            scale.get("reserved_high_byte_required_zero") is True,
            selector.get("bits_per_output_row_per_input_block") == 1,
            selector.get("row_group_size_for_every_linear") == 1,
            scale32_reserved_byte_frozen,
            computed_standard_scale == 38016,
            computed_standard_selector == 14560,
            computed_standard_total == 7509856,
            limits.get("standard_decoder_layer_total_weight_bytes") == 7509856,
            abs(computed_standard_bpw - 4.029584478021978) < 1e-15,
            limits.get("standard_decoder_layer_exact_bits_per_weight") == 4.029584478021978,
            computed_lm_selector == 132944,
            computed_lm_scale == 455808,
            computed_lm_total == 68656304,
            limits.get("lm_head_total_weight_bytes") == 68656304,
            abs(computed_lm_bpw - 4.034611377722951) < 1e-15,
            limits.get("lm_head_exact_bits_per_weight") == 4.034611377722951,
            computed_whole_total == 248892848,
            limits.get("whole_model_total_weight_bytes") == 248892848,
            abs(computed_whole_bpw - 4.030969880841819) < 1e-15,
            limits.get("whole_model_exact_bits_per_weight") == 4.030969880841819,
            max(computed_standard_bpw, computed_lm_bpw, computed_whole_bpw) < 4.036,
            plan.get("accelerator_facing_format", {}).get("public_ace2_shell_contract_changed") is False,
            task.get("task_id") == "task-b87978f9a5a8",
            task.get("status") == "authorized_not_started",
            task.get("candidate_plan", {}).get("sha256") == sha256(V10_PLAN),
            namespace.get("attempt_budget") == 1,
            namespace.get("attempts_consumed_at_task_freeze") == 0,
            namespace.get("execution_marker_permitted_during_runner_build_construction_or_preflight") is False,
            not V10_ROOT.exists(),
        )
    )

    internal = benchmark.get("internal_acceptance_contract", {})
    terminal = internal.get("terminal_alias_safe_dual_input_block_codebook_row_group_selector_v9", {})
    planned = internal.get("planned_alias_safe_packed_scale24_per_row_selector_dual_input_block_codebook_v10", {})
    benchmark_routing = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("behavior_interface", {}).get("parameter_count") == 12,
            benchmark.get("specification_stage_evidence", {}).get("behavior_interface", {}).get("port_count") == 64,
            terminal.get("preflight_status") == "PREATTEMPT_NO_GO",
            terminal.get("review_record_sha256") == sha256(V9_REVIEW),
            planned.get("contract_sha256") == sha256(V10_PLAN),
            planned.get("engineer_task_sha256") == sha256(V10_TASK),
            planned.get("attempt_marker_exists") is False,
            planned.get("stored_row_scale_bytes") == 3,
            planned.get("row_group_size_for_every_linear") == 1,
            planned.get("whole_model_exact_bits_per_weight") == 4.030969880841819,
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
        (pipeline.get("current_stage") == "specification", v9_terminal, v10_frozen, benchmark_routing)
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no candidate construction, official input access, RTL execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v9_terminal_fresh_review_verified": v9_terminal,
            "v9_aggregate_fresh_v7_bf16_sse": model.get("aggregate_fresh_v7_bf16_sse"),
            "v9_aggregate_bf16_sse": model.get("aggregate_v9_dual_codebook_bf16_sse"),
            "v9_exact_sequence_matches": quality.get("exact_bf16_sequence_match_count"),
            "v9_official_attempts_consumed": no_go.get("attempts_consumed"),
            "scale32_reserved_high_byte_frozen_zero": scale32_reserved_byte_frozen,
            "v10_plan_and_task_frozen": v10_frozen,
            "v10_engineer_task_id": task.get("task_id"),
            "v10_namespace_absent": not V10_ROOT.exists(),
            "v10_execution_marker_absent": not (V10_ROOT / "attempt-0001/execution_started.json").exists(),
            "v10_standard_decoder_layer_exact_bits_per_weight": limits.get("standard_decoder_layer_exact_bits_per_weight"),
            "v10_lm_head_exact_bits_per_weight": limits.get("lm_head_exact_bits_per_weight"),
            "v10_whole_model_exact_bits_per_weight": limits.get("whole_model_exact_bits_per_weight"),
            "benchmark_routing_matches": benchmark_routing,
            "selected_policy_id": internal.get("selected_policy_id"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "binding_unknowns": {"count": len(unknowns), "items": unknowns},
        "input_bindings": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                SPEC, BENCHMARK, PIPELINE, GROUND_TRUTH, SCALE_CONTRACT,
                V9_NO_GO, V9_MODEL, V9_QUALITY, V9_DYNAMIC, V9_REVIEW,
                V9_REVIEW_SUM, V10_PLAN, V10_PLAN_SUM, V10_TASK, V10_TASK_SUM,
                Path(__file__),
            )
        },
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_packed_scale24_per_row_selector_v10.py",
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
