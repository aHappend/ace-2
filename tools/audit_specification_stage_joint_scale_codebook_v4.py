#!/usr/bin/env python3
"""Read-only specification audit for terminal V3 and frozen joint-scale V4."""

from __future__ import annotations

import argparse
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
U280 = ROOT / "research/U280_HOST_INVENTORY.json"

V3_ROOT = ROOT / "build/stage1-option-b-alias-safe-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v3"
V3_NO_GO = V3_ROOT / "PREATTEMPT_NO_GO.json"
V3_RECON = V3_ROOT / "preattempt/model_only_reconstruction.json"
V3_QUALITY = V3_ROOT / "preattempt/non_evaluator_quality.json"
V3_DYNAMIC = V3_ROOT / "preattempt/dynamic_preflight.json"
V3_ALIAS = V3_ROOT / "preattempt/alias_separation_witness.json"
V3_ATTEMPT = V3_ROOT / "attempt-0001"
V3_REVIEW = ROOT / "research/raw/specification/tensor-codebook-v3-review-done-20260806T143956Z.json"
V3_REVIEW_SUM = V3_REVIEW.with_suffix(".sha256")

V4_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V4_PLAN.json"
V4_PLAN_SUM = V4_PLAN.with_suffix(".sha256")
V4_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V4_ENGINEER_TASK.json"
V4_TASK_SUM = V4_TASK.with_suffix(".sha256")
V4_ROOT = ROOT / "build/stage1-option-b-alias-safe-joint-row-scale-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v4"
V4_MARKER = V4_ROOT / "attempt-0001/execution_started.json"

REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def companion_matches(path: Path, companion: Path) -> bool:
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


def audit() -> dict[str, object]:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load_json(BENCHMARK)
    pipeline = load_json(PIPELINE)
    no_go = load_json(V3_NO_GO)
    reconstruction = load_json(V3_RECON)
    quality = load_json(V3_QUALITY)
    dynamic = load_json(V3_DYNAMIC)
    alias = load_json(V3_ALIAS)
    review = load_json(V3_REVIEW)
    plan = load_json(V4_PLAN)
    task = load_json(V4_TASK)
    shell = audit_shell_contract()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    observed_classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in observed_classes]

    required_spec_fragments = (
        "option_b_alias_safe_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v4",
        "exactly eight deterministic alternating iterations",
        "dynamic program",
        "4.027257898351649",
        "4.035715225959803",
        "task-5c2e87fd2509",
        "3/8 exact",
        "all 312 Dynamic Scale32 boundaries",
        "`selected_policy_id` is null",
    )
    missing_fragments = [fragment for fragment in required_spec_fragments if fragment not in spec]

    local = benchmark.get("local_contract", {})
    benchmark_closed = bool(
        benchmark.get("applies") is False
        and benchmark.get("external_contract") is None
        and benchmark.get("non_benchmark_statement")
        and isinstance(local, dict)
        and local.get("top_module") == "ace2_shell"
        and local.get("public_interface_source")
        == "design/SPEC.md#public-ace2_shell-parameter-and-port-contract"
    )

    activation = dynamic.get("activation_execution", {})
    review_facts = review.get("reviewed_facts", {})
    v3_terminal = bool(
        companion_matches(V3_REVIEW, V3_REVIEW_SUM)
        and no_go.get("candidate_id")
        == "option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v3"
        and no_go.get("status") == "PREATTEMPT_NO_GO"
        and no_go.get("attempts_consumed") == 0
        and no_go.get("official_execution_marker_exists") is False
        and no_go.get("official_execution_mode_exposed") is False
        and no_go.get("gate_passes", {}).get("two_build_and_alias_closure") is True
        and no_go.get("gate_passes", {}).get("model_only_reconstruction") is True
        and no_go.get("gate_passes", {}).get("dynamic_312_event_closure") is True
        and no_go.get("gate_passes", {}).get("non_evaluator_quality_8_of_8") is False
        and reconstruction.get("tensor_count") == 169
        and reconstruction.get("per_tensor_non_regressing_count") == 169
        and reconstruction.get("aggregate_uniform_baseline_bf16_sse") == 5435.794767826797
        and reconstruction.get("aggregate_tensor_codebook_bf16_sse") == 2987.0298056936317
        and quality.get("exact_bf16_sequence_match_count") == 3
        and quality.get("required_exact_bf16_sequence_match_count") == 8
        and quality.get("response_gate_outcome_match_count") == 8
        and quality.get("required_response_gate_outcome_match_count") == 8
        and alias.get("embedding_preserved_in_both") is True
        and alias.get("lm_head_separate_in_both") is True
        and activation.get("all_312_layer_event_names_present") is True
        and activation.get("event_count") == 312
        and activation.get("minimum_delta") == -14
        and activation.get("maximum_delta") == 1
        and activation.get("saturation_count") == 0
        and activation.get("clipping_count") == 0
        and activation.get("reserved_minus_128_produced") is False
        and review.get("status") == "done"
        and review.get("failure_taxonomy") == "NO_GO_NON_EVALUATOR_QUALITY_GATE"
        and review_facts.get("official_attempts_consumed") == 0
        and review_facts.get("selected_policy_id") is None
        and review.get("bound_evidence", {}).get("terminal_no_go", {}).get("sha256") == sha256(V3_NO_GO)
        and review.get("bound_evidence", {}).get("model_only_reconstruction", {}).get("sha256") == sha256(V3_RECON)
        and review.get("bound_evidence", {}).get("non_evaluator_quality", {}).get("sha256") == sha256(V3_QUALITY)
        and review.get("bound_evidence", {}).get("dynamic_preflight", {}).get("sha256") == sha256(V3_DYNAMIC)
        and not V3_ATTEMPT.exists()
    )

    joint = plan.get("weight_policy", {}).get("joint_construction_algorithm", {})
    limits = plan.get("compression_and_accelerator_limits", {})
    plan_attempt = plan.get("official_attempt", {})
    plan_claim = plan.get("claim_boundary", {})
    namespace = task.get("fresh_attempt_namespace", {})
    task_claim = task.get("claim_boundary", {})
    v4_frozen = bool(
        companion_matches(V4_PLAN, V4_PLAN_SUM)
        and companion_matches(V4_TASK, V4_TASK_SUM)
        and plan.get("candidate_id")
        == "option_b_alias_safe_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v4"
        and plan.get("status") == "planner_frozen_preexecution_requirements_no_attempt_authority"
        and plan.get("stage") == "specification"
        and plan.get("selected_policy_id") is None
        and joint.get("outer_iterations") == 8
        and joint.get("official_prompt_output_token_or_gate_access_permitted") is False
        and limits.get("maximum_total_stored_bits_per_weight") == 4.036
        and limits.get("standard_decoder_layer_exact_bits_per_weight") == 4.027257898351649
        and limits.get("lm_head_exact_bits_per_weight") == 4.035715225959803
        and limits.get("whole_model_codebook_bytes") == 2704
        and limits.get("additional_row_scale_bytes") == 0
        and limits.get("maximum_transport_metadata_bytes_per_decoder_layer_token") == 832
        and limits.get("maximum_incremental_sram_bytes") == 2368
        and limits.get("new_public_ace2_shell_ports_permitted") is False
        and plan_attempt.get("attempts_authorized_by_this_file") == 0
        and plan_attempt.get("matrix", {}).get("sha256")
        == "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
        and plan_claim.get("attempt_authority_created") is False
        and plan_claim.get("current_stage_remains") == "specification"
        and task.get("task_id") == "task-5c2e87fd2509"
        and task.get("status") == "authorized_not_started"
        and task.get("stage") == "specification"
        and task.get("scope") == "bounded"
        and task.get("candidate_plan", {}).get("sha256") == sha256(V4_PLAN)
        and task.get("terminal_predecessor", {}).get("review_record_sha256") == sha256(V3_REVIEW)
        and namespace.get("attempt_budget") == 1
        and namespace.get("attempts_consumed_at_task_freeze") == 0
        and namespace.get("execution_marker_permitted_during_runner_build_or_preflight") is False
        and namespace.get("replay_resume_or_second_attempt_under_candidate_id_permitted") is False
        and task_claim.get("task_authority_created") is True
        and task_claim.get("official_attempt_started") is False
        and task_claim.get("official_attempt_consumed") is False
        and task_claim.get("selected_policy_id") is None
        and task_claim.get("current_stage_remains") == "specification"
        and not V4_ROOT.exists()
        and not V4_MARKER.exists()
    )

    compression = benchmark.get("product_compression_gate", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    terminal = compression.get("terminal_codebook_v3", {}) if isinstance(compression, dict) else {}
    planned = compression.get("planned_successor_contract", {}) if isinstance(compression, dict) else {}
    internal_v4 = internal.get("planned_alias_safe_joint_row_scale_codebook_v4", {}) if isinstance(internal, dict) else {}
    benchmark_routing = bool(
        isinstance(compression, dict)
        and compression.get("selection_status")
        == "none_codebook_v3_terminal_v4_engineer_task_authorized_not_started"
        and compression.get("selected_policy_id") is None
        and terminal.get("preflight_terminal_sha256") == sha256(V3_NO_GO)
        and terminal.get("review_record_sha256") == sha256(V3_REVIEW)
        and planned.get("sha256") == sha256(V4_PLAN)
        and planned.get("bounded_engineer_task", {}).get("sha256") == sha256(V4_TASK)
        and isinstance(internal, dict)
        and internal_v4.get("contract_sha256") == sha256(V4_PLAN)
        and internal_v4.get("engineer_task_sha256") == sha256(V4_TASK)
        and internal.get("next_candidate_execution_authorized") is True
        and internal.get("next_structurally_distinct_engineer_task_required") is False
        and internal.get("predecessor_fresh_reviewer_adjudication_complete") is True
    )

    ground_lines = GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
    unknowns = [line[2:] for line in ground_lines if line.startswith("- ")]
    unexpected_ground_lines = [
        line
        for line in ground_lines
        if line.strip() and line != "# Binding Unknowns" and not line.startswith("- ")
    ]

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS" if shell.get("status") == "PASS" and not missing_fragments else "FAIL",
            "shell_contract_audit": shell.get("status"),
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "missing_behavior_fragments": missing_fragments,
        },
        "spec.clock-reset-protocol": {
            "status": "PASS"
            if all(
                fragment in spec
                for fragment in (
                    "single-clock digital accelerator subsystem",
                    "Asynchronous active-low reset",
                    "deassert synchronously before state machines leave reset",
                    "earliest read response on clock edge",
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
            "observed_classes": observed_classes,
            "missing_classes": missing_classes,
        },
        "spec.benchmark-interface-closure": {
            "status": "PASS" if benchmark_closed else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "external_contract": benchmark.get("external_contract"),
            "non_benchmark_statement": benchmark.get("non_benchmark_statement"),
        },
    }

    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            v3_terminal,
            v4_frozen,
            benchmark_routing,
            not unexpected_ground_lines,
        )
    )

    inputs = [
        SPEC,
        BENCHMARK,
        PIPELINE,
        GROUND_TRUTH,
        U280,
        V3_NO_GO,
        V3_RECON,
        V3_QUALITY,
        V3_DYNAMIC,
        V3_ALIAS,
        V3_REVIEW,
        V3_REVIEW_SUM,
        V4_PLAN,
        V4_PLAN_SUM,
        V4_TASK,
        V4_TASK_SUM,
        Path(__file__),
    ]
    return {
        "schema_version": 1,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "read-only specification-stage audit; no numerical candidate construction, simulation, formal, synthesis, PPA, chat, FPGA build, emulation, board execution, or attempt marker",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {
            "current_stage": pipeline.get("current_stage"),
            "active_stage_matches": pipeline.get("current_stage") == "specification",
            "manager_owned_state_mutated": False,
        },
        "checklist": checklist,
        "mission_gate": {
            "v3_terminal_fresh_review_verified": v3_terminal,
            "v3_non_evaluator_exact_sequence_matches": quality.get("exact_bf16_sequence_match_count"),
            "v3_non_evaluator_response_gate_matches": quality.get("response_gate_outcome_match_count"),
            "v3_official_attempts_consumed": no_go.get("attempts_consumed"),
            "v4_plan_and_task_frozen": v4_frozen,
            "v4_engineer_task_id": task.get("task_id"),
            "v4_attempts_authorized_by_plan": plan_attempt.get("attempts_authorized_by_this_file"),
            "v4_attempts_authorized_by_task": namespace.get("attempt_budget"),
            "v4_attempts_consumed": namespace.get("attempts_consumed_at_task_freeze"),
            "v4_namespace_absent": not V4_ROOT.exists(),
            "v4_execution_marker_absent": not V4_MARKER.exists(),
            "benchmark_routing_matches": benchmark_routing,
            "selected_policy_id": internal.get("selected_policy_id") if isinstance(internal, dict) else None,
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "binding_unknowns": {
            "count": len(unknowns),
            "items": unknowns,
            "unexpected_non_unknown_lines": unexpected_ground_lines,
        },
        "input_bindings": {
            str(path.relative_to(ROOT)): sha256(path) for path in inputs if path.is_file()
        },
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_joint_scale_codebook_v4.py",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw, encoding="utf-8")
    print(raw, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
