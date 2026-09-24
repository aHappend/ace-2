#!/usr/bin/env python3
"""Read-only specification audit for terminal V5 and frozen V6."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
V5_ROOT = ROOT / "build/stage1-option-b-alias-safe-calibration-weighted-joint-row-scale-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v5"
V5_NO_GO = V5_ROOT / "PREATTEMPT_NO_GO.json"
V5_EQUAL = V5_ROOT / "preattempt/calibration_statistics_equality.json"
V5_RECON = V5_ROOT / "preattempt/model_only_weighted_reconstruction.json"
V5_QUALITY = V5_ROOT / "preattempt/non_evaluator_quality.json"
V5_DYNAMIC = V5_ROOT / "preattempt/dynamic_preflight.json"
V5_ALIAS = V5_ROOT / "preattempt/alias_separation_witness.json"
V5_REVIEW = ROOT / "research/raw/specification/calibration-weighted-codebook-v5-review-done-20260806T173457Z.json"
V5_REVIEW_SUM = V5_REVIEW.with_suffix(".sha256")
V6_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_BLOCK_COVARIANCE_ERROR_FEEDBACK_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V6_PLAN.json"
V6_PLAN_SUM = V6_PLAN.with_suffix(".sha256")
V6_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_BLOCK_COVARIANCE_ERROR_FEEDBACK_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V6_ENGINEER_TASK.json"
V6_TASK_SUM = V6_TASK.with_suffix(".sha256")
V6_ROOT = ROOT / "build/stage1-option-b-alias-safe-block-covariance-error-feedback-joint-row-scale-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v6"
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
    no_go = load(V5_NO_GO)
    equality = load(V5_EQUAL)
    reconstruction = load(V5_RECON)
    quality = load(V5_QUALITY)
    dynamic = load(V5_DYNAMIC)
    alias = load(V5_ALIAS)
    review = load(V5_REVIEW)
    plan = load(V6_PLAN)
    task = load(V6_TASK)
    shell = audit_shell_contract()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in classes]

    required_spec = (
        "option_b_alias_safe_block_covariance_error_feedback_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v6",
        "1/8 token arrays",
        "task-0394d005ed34",
        "16-channel BF16 input Gram blocks",
        "`selected_policy_id` is null",
    )
    missing_spec = [item for item in required_spec if item not in spec]

    activation = dynamic.get("activation_execution", {})
    v5_terminal = all(
        (
            no_go.get("status") == "PREATTEMPT_NO_GO",
            no_go.get("attempts_consumed") == 0,
            no_go.get("official_execution_marker_exists") is False,
            no_go.get("gate_passes", {}).get("duplicate_calibration_statistics") is True,
            no_go.get("gate_passes", {}).get("two_build_alias_and_manifest_closure") is True,
            no_go.get("gate_passes", {}).get("v4_relative_weighted_model_169_of_169") is True,
            no_go.get("gate_passes", {}).get("dynamic_312_event_closure") is True,
            no_go.get("gate_passes", {}).get("non_evaluator_quality_8_of_8") is False,
            equality.get("byte_identical") is True,
            equality.get("linear_count") == 169,
            reconstruction.get("per_tensor_non_regressing_count") == 169,
            reconstruction.get("per_tensor_strictly_improved_count") == 169,
            reconstruction.get("aggregate_v5_unweighted_bf16_sse") == 2405.7241206996073,
            quality.get("exact_bf16_sequence_match_count") == 1,
            quality.get("response_gate_outcome_match_count") == 8,
            quality.get("official_nine_prompt_inputs_used") is False,
            activation.get("event_count") == 312,
            activation.get("minimum_delta") == -14,
            activation.get("maximum_delta") == 1,
            activation.get("saturation_count") == 0,
            activation.get("clipping_count") == 0,
            activation.get("reserved_minus_128_produced") is False,
            alias.get("embedding_preserved_in_both") is True,
            alias.get("lm_head_separate_in_both") is True,
            alias.get("state_comparison", {}).get("all_match") is True,
            companion_matches(V5_REVIEW, V5_REVIEW_SUM),
            review.get("status") == "done",
            review.get("failure_taxonomy") == "NO_GO_NON_EVALUATOR_QUALITY_GATE",
            review.get("reviewed_facts", {}).get("official_attempts_consumed") == 0,
            review.get("bound_evidence", {}).get("terminal_no_go", {}).get("sha256") == sha256(V5_NO_GO),
            not (V5_ROOT / "PREATTEMPT_READY.json").exists(),
            not (V5_ROOT / "attempt-0001").exists(),
        )
    )

    algorithm = plan.get("weight_policy", {}).get("block_covariance_error_feedback_algorithm", {})
    namespace = task.get("fresh_attempt_namespace", {})
    v6_frozen = all(
        (
            companion_matches(V6_PLAN, V6_PLAN_SUM),
            companion_matches(V6_TASK, V6_TASK_SUM),
            plan.get("candidate_id") == "option_b_alias_safe_block_covariance_error_feedback_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v6",
            plan.get("stage") == "specification",
            plan.get("official_attempt", {}).get("attempts_authorized_by_this_file") == 0,
            plan.get("calibration_covariance", {}).get("input_channel_block_size") == 16,
            algorithm.get("outer_sweeps") == 4,
            algorithm.get("official_prompt_output_token_or_gate_access_permitted") is False,
            plan.get("accelerator_facing_format", {}).get("public_ace2_shell_contract_changed") is False,
            plan.get("accelerator_facing_format", {}).get("additional_product_runtime_transport_or_sram_bytes") == 0,
            task.get("task_id") == "task-0394d005ed34",
            task.get("status") == "authorized_not_started",
            namespace.get("attempt_budget") == 1,
            namespace.get("attempts_consumed_at_task_freeze") == 0,
            namespace.get("execution_marker_permitted_during_runner_build_covariance_capture_or_preflight") is False,
            not V6_ROOT.exists(),
        )
    )

    internal = benchmark.get("internal_acceptance_contract", {})
    planned = internal.get("planned_alias_safe_block_covariance_codebook_v6", {})
    terminal = internal.get("terminal_alias_safe_calibration_weighted_codebook_v5", {})
    benchmark_routing = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("behavior_interface", {}).get("parameter_count") == 12,
            benchmark.get("specification_stage_evidence", {}).get("behavior_interface", {}).get("port_count") == 64,
            terminal.get("preflight_status") == "PREATTEMPT_NO_GO",
            terminal.get("review_record_sha256") == sha256(V5_REVIEW),
            planned.get("contract_sha256") == sha256(V6_PLAN),
            planned.get("engineer_task_sha256") == sha256(V6_TASK),
            planned.get("attempt_marker_exists") is False,
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
        (pipeline.get("current_stage") == "specification", v5_terminal, v6_frozen, benchmark_routing)
    )
    inputs = [
        SPEC, BENCHMARK, PIPELINE, GROUND_TRUTH, V5_NO_GO, V5_EQUAL, V5_RECON,
        V5_QUALITY, V5_DYNAMIC, V5_ALIAS, V5_REVIEW, V5_REVIEW_SUM, V6_PLAN,
        V6_PLAN_SUM, V6_TASK, V6_TASK_SUM, Path(__file__),
    ]
    return {
        "schema_version": 1,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "read-only specification audit; no candidate construction, official input access, RTL execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v5_terminal_fresh_review_verified": v5_terminal,
            "v5_exact_sequence_matches": quality.get("exact_bf16_sequence_match_count"),
            "v5_official_attempts_consumed": no_go.get("attempts_consumed"),
            "v6_plan_and_task_frozen": v6_frozen,
            "v6_engineer_task_id": task.get("task_id"),
            "v6_namespace_absent": not V6_ROOT.exists(),
            "v6_execution_marker_absent": not (V6_ROOT / "attempt-0001/execution_started.json").exists(),
            "benchmark_routing_matches": benchmark_routing,
            "selected_policy_id": internal.get("selected_policy_id"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "binding_unknowns": {"count": len(unknowns), "items": unknowns},
        "input_bindings": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_block_covariance_codebook_v6.py",
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
