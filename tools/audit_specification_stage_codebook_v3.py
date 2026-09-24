#!/usr/bin/env python3
"""Read-only audit for the ACE-2 codebook-V3 specification replacement."""

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

DYNAMIC_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_POST_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V2_PLAN.json"
DYNAMIC_PLAN_SUM = DYNAMIC_PLAN.with_suffix(".sha256")
DYNAMIC_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_POST_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V2_ENGINEER_TASK.json"
DYNAMIC_TASK_SUM = DYNAMIC_TASK.with_suffix(".sha256")
DYNAMIC_ROOT = ROOT / "build/stage1-option-b-alias-safe-post-w4-grouped-dynamic-scale32-w4a8-v2"
DYNAMIC_READY = DYNAMIC_ROOT / "PREATTEMPT_READY.json"
DYNAMIC_MARKER = DYNAMIC_ROOT / "attempt-0001/execution_started.json"
DYNAMIC_RESULTS = DYNAMIC_ROOT / "attempt-0001/results.json"
DYNAMIC_SUMS = DYNAMIC_ROOT / "attempt-0001/SHA256SUMS"
DYNAMIC_SUMS_SUM = DYNAMIC_ROOT / "attempt-0001/SHA256SUMS.sha256"
DYNAMIC_REVIEW = ROOT / "research/raw/specification/alias-safe-dynamic-v2-review-replan-20260806T140045Z.json"
DYNAMIC_REVIEW_SUM = DYNAMIC_REVIEW.with_suffix(".sha256")

CODEBOOK_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V3_PLAN.json"
CODEBOOK_PLAN_SUM = CODEBOOK_PLAN.with_suffix(".sha256")
CODEBOOK_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V3_ENGINEER_TASK.json"
CODEBOOK_TASK_SUM = CODEBOOK_TASK.with_suffix(".sha256")
CODEBOOK_ROOT = ROOT / "build/stage1-option-b-alias-safe-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v3"

REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def companion_matches(path: Path, companion: Path) -> bool:
    return companion.is_file() and companion.read_text(encoding="utf-8").strip() == (
        f"{sha256(path)}  {path.relative_to(ROOT)}"
    )


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
    dynamic_plan = load_json(DYNAMIC_PLAN)
    dynamic_task = load_json(DYNAMIC_TASK)
    dynamic_ready = load_json(DYNAMIC_READY)
    dynamic_marker = load_json(DYNAMIC_MARKER)
    dynamic_results = load_json(DYNAMIC_RESULTS)
    dynamic_review = load_json(DYNAMIC_REVIEW)
    codebook_plan = load_json(CODEBOOK_PLAN)
    codebook_task = load_json(CODEBOOK_TASK)
    shell = audit_shell_contract()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    observed_classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in observed_classes]

    required_spec_fragments = (
        "option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v3",
        "unsigned 4-bit index",
        "signed-Q4.4",
        "4.027257898351649",
        "4.035715225959803",
        "task-904869143c15",
        "8/8 exact BF16 token arrays",
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

    dynamic_attempt_dir = DYNAMIC_MARKER.parent
    dynamic_immutable = bool(
        dynamic_attempt_dir.stat().st_mode & 0o777 == 0o555
        and all(path.stat().st_mode & 0o777 == 0o444 for path in dynamic_attempt_dir.iterdir())
    )
    aggregate = dynamic_results.get("aggregate", {})
    eligibility = dynamic_results.get("eligibility", {})
    execution = dynamic_results.get("activation_execution", {})
    dynamic_terminal = bool(
        companion_matches(DYNAMIC_PLAN, DYNAMIC_PLAN_SUM)
        and companion_matches(DYNAMIC_TASK, DYNAMIC_TASK_SUM)
        and companion_matches(DYNAMIC_REVIEW, DYNAMIC_REVIEW_SUM)
        and DYNAMIC_SUMS_SUM.read_text(encoding="utf-8").strip()
        == f"{sha256(DYNAMIC_SUMS)}  {DYNAMIC_SUMS.relative_to(ROOT)}"
        and dynamic_plan.get("candidate_id")
        == "option_b_alias_safe_post_w4_grouped_dynamic_scale32_w4a8_v2"
        and dynamic_task.get("task_id") == "task-0ea2df03119e"
        and dynamic_marker.get("authorization_consumed") is True
        and dynamic_results.get("status") == "BLOCKED_EXACT_BF16_DISAGREEMENT"
        and aggregate.get("prompt_count") == 9
        and aggregate.get("exact_bf16_sequence_match_count") == 1
        and aggregate.get("response_gate_outcome_match_count") == 6
        and eligibility.get("eligible") is False
        and eligibility.get("selected_policy_id") is None
        and execution.get("all_312_layer_event_names_present") is True
        and execution.get("minimum_delta") == -13
        and execution.get("maximum_delta") == 2
        and execution.get("saturation_count") == 0
        and execution.get("clipping_count") == 0
        and execution.get("reserved_minus_128_produced") is False
        and execution.get("transport_metadata_bytes_per_decoder_layer_token") == 832
        and dynamic_review.get("status") == "replan_requested"
        and dynamic_review.get("failure_taxonomy") == "NUMERICAL_EXACT_BF16_DISAGREEMENT"
        and dynamic_review.get("reviewed_facts", {}).get("eligible") is False
        and dynamic_review.get("reviewed_facts", {}).get("selected_policy_id") is None
        and dynamic_review.get("reviewed_facts", {}).get("replay_permitted") is False
        and dynamic_review.get("reviewed_facts", {}).get("result", {}).get("sha256")
        == sha256(DYNAMIC_RESULTS)
        and dynamic_review.get("reviewed_facts", {}).get("attempt_manifest", {}).get("sha256")
        == sha256(DYNAMIC_SUMS)
        and dynamic_immutable
    )

    codebook_weight = codebook_plan.get("weight_policy", {})
    codebook_limits = codebook_plan.get("compression_and_accelerator_limits", {})
    codebook_attempt = codebook_plan.get("official_attempt", {})
    codebook_claim = codebook_plan.get("claim_boundary", {})
    codebook_namespace = codebook_task.get("fresh_attempt_namespace", {})
    codebook_task_claim = codebook_task.get("claim_boundary", {})
    expected_standard_bits = (7505408 + 7 * 16) * 8 / 14909440
    expected_lm_head_bits = ((151936 * 896) / 2 + 151936 * 4 + 16) * 8 / (151936 * 896)
    codebook_frozen = bool(
        companion_matches(CODEBOOK_PLAN, CODEBOOK_PLAN_SUM)
        and companion_matches(CODEBOOK_TASK, CODEBOOK_TASK_SUM)
        and codebook_plan.get("candidate_id")
        == "option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v3"
        and codebook_plan.get("status")
        == "planner_frozen_preexecution_requirements_no_attempt_authority"
        and codebook_plan.get("selected_policy_id") is None
        and codebook_weight.get("payload_bits_per_weight") == 4
        and codebook_weight.get("codebook_record", {}).get("bytes") == 16
        and codebook_weight.get("codebook_record", {}).get("memory_beats_at_128_bits") == 1
        and codebook_weight.get("construction_algorithm", {}).get("iterations") == 32
        and codebook_weight.get("construction_algorithm", {}).get(
            "official_prompt_output_or_gate_access_permitted"
        )
        is False
        and codebook_limits.get("maximum_total_stored_bits_per_weight") == 4.036
        and abs(codebook_limits.get("standard_decoder_layer_exact_bits_per_weight", 0) - expected_standard_bits)
        < 1e-15
        and abs(codebook_limits.get("lm_head_exact_bits_per_weight", 0) - expected_lm_head_bits) < 1e-15
        and expected_standard_bits <= 4.036
        and expected_lm_head_bits <= 4.036
        and codebook_limits.get("whole_model_codebook_bytes") == 2704
        and codebook_limits.get("maximum_transport_metadata_bytes_per_decoder_layer_token") == 832
        and codebook_limits.get("maximum_incremental_sram_bytes") == 2368
        and codebook_limits.get("abstract_streaming_memory_boundary_bits") == 128
        and codebook_limits.get("new_public_ace2_shell_ports_permitted") is False
        and codebook_plan.get("accelerator_facing_format", {}).get("floating_point_rtl_permitted")
        is False
        and codebook_attempt.get("attempts_authorized_by_this_file") == 0
        and codebook_attempt.get("matrix", {}).get("sha256")
        == "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
        and codebook_claim.get("attempt_authority_created") is False
        and codebook_claim.get("current_stage_remains") == "specification"
        and codebook_task.get("task_id") == "task-904869143c15"
        and codebook_task.get("status") == "authorized_not_started"
        and codebook_task.get("stage") == "specification"
        and codebook_task.get("scope") == "bounded"
        and codebook_task.get("candidate_plan", {}).get("sha256") == sha256(CODEBOOK_PLAN)
        and codebook_namespace.get("attempt_budget") == 1
        and codebook_namespace.get("attempts_consumed_at_task_freeze") == 0
        and codebook_namespace.get("execution_marker_permitted_during_runner_build_or_preflight")
        is False
        and codebook_namespace.get("replay_resume_or_second_attempt_under_candidate_id_permitted")
        is False
        and codebook_task_claim.get("task_authority_created") is True
        and codebook_task_claim.get("official_attempt_started") is False
        and codebook_task_claim.get("official_attempt_consumed") is False
        and codebook_task_claim.get("selected_policy_id") is None
        and codebook_task_claim.get("current_stage_remains") == "specification"
        and not CODEBOOK_ROOT.exists()
    )

    compression = benchmark.get("product_compression_gate", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    benchmark_routing = bool(
        isinstance(compression, dict)
        and compression.get("selection_status") == "none_codebook_v3_engineer_task_authorized_not_started"
        and compression.get("selected_policy_id") is None
        and compression.get("consumed_alias_safe_dynamic_v2", {}).get("result_sha256")
        == sha256(DYNAMIC_RESULTS)
        and compression.get("consumed_alias_safe_dynamic_v2", {}).get("review_record_sha256")
        == sha256(DYNAMIC_REVIEW)
        and compression.get("planned_successor_contract", {}).get("sha256") == sha256(CODEBOOK_PLAN)
        and compression.get("planned_successor_contract", {}).get("bounded_engineer_task", {}).get("sha256")
        == sha256(CODEBOOK_TASK)
        and isinstance(internal, dict)
        and internal.get("alias_safe_dynamic_v2_attempt", {}).get("eligible") is False
        and internal.get("planned_alias_safe_codebook_successor", {}).get("contract_sha256")
        == sha256(CODEBOOK_PLAN)
        and internal.get("planned_alias_safe_codebook_successor", {}).get("engineer_task_sha256")
        == sha256(CODEBOOK_TASK)
        and internal.get("next_candidate_execution_authorized") is True
        and internal.get("next_structurally_distinct_engineer_task_required") is False
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
            dynamic_terminal,
            codebook_frozen,
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
        DYNAMIC_PLAN,
        DYNAMIC_PLAN_SUM,
        DYNAMIC_TASK,
        DYNAMIC_TASK_SUM,
        DYNAMIC_READY,
        DYNAMIC_MARKER,
        DYNAMIC_RESULTS,
        DYNAMIC_SUMS,
        DYNAMIC_SUMS_SUM,
        DYNAMIC_REVIEW,
        DYNAMIC_REVIEW_SUM,
        CODEBOOK_PLAN,
        CODEBOOK_PLAN_SUM,
        CODEBOOK_TASK,
        CODEBOOK_TASK_SUM,
        Path(__file__),
    ]
    return {
        "schema_version": 1,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "read-only specification-stage audit; no numerical regeneration, simulation, formal, synthesis, PPA, chat, FPGA build, emulation, or board execution",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {
            "current_stage": pipeline.get("current_stage"),
            "active_stage_matches": pipeline.get("current_stage") == "specification",
            "manager_owned_state_mutated": False,
        },
        "checklist": checklist,
        "mission_gate": {
            "dynamic_v2_terminal_failure_verified": dynamic_terminal,
            "dynamic_v2_attempts_consumed": 1,
            "dynamic_v2_exact_sequence_matches": aggregate.get("exact_bf16_sequence_match_count"),
            "dynamic_v2_response_gate_matches": aggregate.get("response_gate_outcome_match_count"),
            "codebook_v3_plan_frozen": codebook_frozen,
            "codebook_v3_engineer_task_id": codebook_task.get("task_id"),
            "codebook_v3_attempts_authorized_by_plan": codebook_attempt.get("attempts_authorized_by_this_file"),
            "codebook_v3_attempts_authorized_by_task": codebook_namespace.get("attempt_budget"),
            "codebook_v3_attempts_consumed": codebook_namespace.get("attempts_consumed_at_task_freeze"),
            "codebook_v3_namespace_absent": not CODEBOOK_ROOT.exists(),
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
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_codebook_v3.py",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
