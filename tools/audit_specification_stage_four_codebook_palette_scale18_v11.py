#!/usr/bin/env python3
"""Read-only specification audit for terminal V10 and frozen V11."""

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
V10_ROOT = ROOT / "build/stage1-option-b-alias-safe-packed-scale24-per-row-selector-dual-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v10"
V10_NO_GO = V10_ROOT / "PREATTEMPT_NO_GO.json"
V10_MODEL = V10_ROOT / "preattempt/model_only_v9_relative_reconstruction.json"
V10_QUALITY = V10_ROOT / "preattempt/non_evaluator_quality.json"
V10_DYNAMIC = V10_ROOT / "preattempt/dynamic_preflight.json"
V10_REVIEW = ROOT / "research/raw/specification/packed-scale24-per-row-selector-v10-review-done-20260807T011340Z.json"
V10_REVIEW_SUM = V10_REVIEW.with_suffix(".sha256")
V11_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_FOUR_INPUT_BLOCK_CODEBOOK_QUATERNARY_ROW_SELECTOR_PALETTE_SCALE18_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V11_PLAN.json"
V11_PLAN_SUM = V11_PLAN.with_suffix(".sha256")
V11_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_FOUR_INPUT_BLOCK_CODEBOOK_QUATERNARY_ROW_SELECTOR_PALETTE_SCALE18_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V11_ENGINEER_TASK.json"
V11_TASK_SUM = V11_TASK.with_suffix(".sha256")
V11_ROOT = ROOT / "build/stage1-option-b-alias-safe-four-input-block-codebook-quaternary-row-selector-palette-scale18-w4-grouped-dynamic-scale32-w4a8-v11"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")
LAYER_SHAPES = (
    (896, 896),
    (128, 896),
    (128, 896),
    (896, 896),
    (4864, 896),
    (4864, 896),
    (896, 4864),
)
LM_HEAD_SHAPE = (151936, 896)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def companion_matches(path: Path, companion: Path) -> bool:
    expected = f"{sha256(path)}  {path.relative_to(ROOT)}"
    return companion.is_file() and companion.read_text(encoding="utf-8").strip() == expected


def aligned_bytes(byte_count: int, alignment: int = 16) -> int:
    return math.ceil(byte_count / alignment) * alignment


def tensor_storage(rows: int, input_features: int) -> dict[str, int]:
    input_blocks = input_features // 128
    payload = rows * input_features // 2
    codebooks = input_blocks * 4 * 16
    selectors = aligned_bytes(math.ceil(rows * input_blocks * 2 / 8))
    scales = aligned_bytes(4 + math.ceil(rows * 18 / 8))
    return {
        "payload": payload,
        "codebooks": codebooks,
        "selectors": selectors,
        "scales": scales,
        "total": payload + codebooks + selectors + scales,
        "weight_count": rows * input_features,
    }


def aggregate_storage(shapes: tuple[tuple[int, int], ...]) -> dict[str, int]:
    records = [tensor_storage(*shape) for shape in shapes]
    return {key: sum(item[key] for item in records) for key in records[0]}


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
    no_go = load(V10_NO_GO)
    model = load(V10_MODEL)
    quality = load(V10_QUALITY)
    dynamic = load(V10_DYNAMIC)
    review = load(V10_REVIEW)
    plan = load(V11_PLAN)
    task = load(V11_TASK)
    shell = audit_shell_contract()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in classes]
    required_spec = (
        "option_b_alias_safe_four_input_block_codebook_quaternary_row_selector_palette_scale18_w4_grouped_dynamic_scale32_w4a8_v11",
        "task-7f2c91d8a44e",
        "2153.4047236346078",
        "4.033731112637363",
        "4.035741552834276",
        "4.034285185661216",
        "four 16-entry signed-Q4.4 codebook banks",
        "18-bit row record",
        "`selected_policy_id` is null",
    )
    missing_spec = [item for item in required_spec if item not in spec]

    activation = dynamic.get("activation_execution", {})
    v10_terminal = all(
        (
            no_go.get("status") == "PREATTEMPT_NO_GO",
            no_go.get("attempts_consumed") == 0,
            no_go.get("official_execution_marker_exists") is False,
            no_go.get("gate_passes", {}).get("fresh_v9_and_169_model_closure") is True,
            no_go.get("gate_passes", {}).get("two_build_alias_codec_and_manifest_closure") is True,
            no_go.get("gate_passes", {}).get("dynamic_312_event_closure") is True,
            no_go.get("gate_passes", {}).get("non_evaluator_quality_8_of_8") is False,
            model.get("fresh_v9_exact_reproduction_count") == 169,
            model.get("per_tensor_non_regressing_count") == 169,
            model.get("per_tensor_strictly_improved_count") == 169,
            model.get("aggregate_fresh_v9_bf16_sse") == 2176.5828747795154,
            model.get("aggregate_v10_bf16_sse") == 2153.4047236346078,
            quality.get("exact_bf16_sequence_match_count") == 4,
            quality.get("response_gate_outcome_match_count") == 7,
            quality.get("official_nine_prompt_inputs_used") is False,
            dynamic.get("status") == "PASS_ALL_312_DYNAMIC_EVENTS",
            activation.get("named_boundary_count") == 312,
            activation.get("minimum_delta") == -14,
            activation.get("maximum_delta") == 2,
            activation.get("saturation_count") == 0,
            activation.get("clipping_count") == 0,
            activation.get("reserved_minus_128_produced") is False,
            activation.get("transport_metadata_bytes_per_decoder_layer_token") == 832,
            review.get("status") == "done",
            review.get("failure_taxonomy") == "NO_GO_NON_EVALUATOR_QUALITY_GATE",
            review.get("fresh_reviewer_handoff", {}).get("sha256")
            == "92434015b6798c098e922be6b97c3348340d9fd289c50a4349ae145e7075f6ce",
            companion_matches(V10_REVIEW, V10_REVIEW_SUM),
        )
    )

    layer = aggregate_storage(LAYER_SHAPES)
    lm_head = aggregate_storage((LM_HEAD_SHAPE,))
    whole = {key: 24 * layer[key] + lm_head[key] for key in layer}
    limits = plan.get("compression_and_accelerator_limits", {})
    accounting = all(
        (
            layer["payload"] == 7454720,
            layer["codebooks"] == 5120,
            layer["selectors"] == 29120,
            layer["scales"] == 28624,
            layer["total"] == 7517584,
            lm_head["payload"] == 68067328,
            lm_head["codebooks"] == 448,
            lm_head["selectors"] == 265888,
            lm_head["scales"] == 341872,
            lm_head["total"] == 68675536,
            whole["payload"] == 246980608,
            whole["codebooks"] == 123328,
            whole["selectors"] == 964768,
            whole["scales"] == 1028848,
            whole["total"] == 249097552,
            math.isclose(layer["total"] * 8 / layer["weight_count"], 4.033731112637363),
            math.isclose(lm_head["total"] * 8 / lm_head["weight_count"], 4.035741552834276),
            math.isclose(whole["total"] * 8 / whole["weight_count"], 4.034285185661216),
            limits.get("standard_decoder_layer_total_weight_bytes") == layer["total"],
            limits.get("lm_head_total_weight_bytes") == lm_head["total"],
            limits.get("whole_model_total_weight_bytes") == whole["total"],
            limits.get("standard_decoder_layer_exact_bits_per_weight") < 4.036,
            limits.get("lm_head_exact_bits_per_weight") < 4.036,
            limits.get("whole_model_exact_bits_per_weight") < 4.036,
        )
    )

    v11_frozen = all(
        (
            plan.get("status") == "frozen_not_started",
            plan.get("stage") == "specification",
            plan.get("selected_policy_id") is None,
            plan.get("weight_policy", {}).get("four_codebooks", {}).get("banks_per_input_block") == 4,
            plan.get("weight_policy", {}).get("quaternary_row_selector", {}).get("bits_per_output_row_per_input_block") == 2,
            plan.get("weight_policy", {}).get("palette_scale18", {}).get("palette_entries_per_tensor") == 4,
            plan.get("weight_policy", {}).get("palette_scale18", {}).get("row_record_bits") == 18,
            companion_matches(V11_PLAN, V11_PLAN_SUM),
            task.get("task_id") == "task-7f2c91d8a44e",
            task.get("status") == "authorized_not_started",
            task.get("stage") == "specification",
            task.get("candidate_plan", {}).get("sha256") == sha256(V11_PLAN),
            companion_matches(V11_TASK, V11_TASK_SUM),
            not V11_ROOT.exists(),
            not (V11_ROOT / "attempt-0001/execution_started.json").exists(),
            accounting,
        )
    )

    internal = benchmark.get("internal_acceptance_contract", {})
    planned = internal.get(
        "planned_alias_safe_four_input_block_codebook_quaternary_row_selector_palette_scale18_v11", {}
    )
    benchmark_routing = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure")
            == "explicit_non_benchmark_statement",
            planned.get("contract_sha256") == sha256(V11_PLAN),
            planned.get("engineer_task_sha256") == sha256(V11_TASK),
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
            "status": "PASS" if benchmark_routing else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "closure": benchmark.get("specification_stage_evidence", {})
            .get("benchmark_interface_closure", {})
            .get("closure"),
        },
    }

    unknowns = [
        line[2:]
        for line in GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]
    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (pipeline.get("current_stage") == "specification", v10_terminal, v11_frozen, benchmark_routing)
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no candidate construction, official input access, RTL execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v10_terminal_fresh_review_verified": v10_terminal,
            "v10_aggregate_bf16_sse": model.get("aggregate_v10_bf16_sse"),
            "v10_exact_sequence_matches": quality.get("exact_bf16_sequence_match_count"),
            "v10_response_gate_matches": quality.get("response_gate_outcome_match_count"),
            "v10_official_attempts_consumed": no_go.get("attempts_consumed"),
            "v11_plan_and_task_frozen": v11_frozen,
            "v11_engineer_task_id": task.get("task_id"),
            "v11_namespace_absent": not V11_ROOT.exists(),
            "v11_execution_marker_absent": not (V11_ROOT / "attempt-0001/execution_started.json").exists(),
            "v11_standard_decoder_layer_exact_bits_per_weight": limits.get(
                "standard_decoder_layer_exact_bits_per_weight"
            ),
            "v11_lm_head_exact_bits_per_weight": limits.get("lm_head_exact_bits_per_weight"),
            "v11_whole_model_exact_bits_per_weight": limits.get("whole_model_exact_bits_per_weight"),
            "benchmark_routing_matches": benchmark_routing,
            "selected_policy_id": internal.get("selected_policy_id"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "byte_accounting": {"standard_decoder_layer": layer, "lm_head": lm_head, "whole_model": whole},
        "binding_unknowns": {"count": len(unknowns), "items": unknowns},
        "input_bindings": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                SPEC,
                BENCHMARK,
                PIPELINE,
                GROUND_TRUTH,
                V10_NO_GO,
                V10_MODEL,
                V10_QUALITY,
                V10_DYNAMIC,
                V10_REVIEW,
                V10_REVIEW_SUM,
                V11_PLAN,
                V11_PLAN_SUM,
                V11_TASK,
                V11_TASK_SUM,
                Path(__file__),
            )
        },
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_four_codebook_palette_scale18_v11.py",
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
