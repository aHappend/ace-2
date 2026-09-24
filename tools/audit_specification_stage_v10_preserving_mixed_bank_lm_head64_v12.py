#!/usr/bin/env python3
"""Read-only specification audit for terminal V11 and frozen V12."""

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
V11_ROOT = ROOT / "build/stage1-option-b-alias-safe-four-input-block-codebook-quaternary-row-selector-palette-scale18-w4-grouped-dynamic-scale32-w4a8-v11"
V11_NO_GO = V11_ROOT / "PREATTEMPT_NO_GO.json"
V11_MODEL = V11_ROOT / "preattempt/model_only_v10_relative_reconstruction.json"
V11_QUALITY = V11_ROOT / "preattempt/non_evaluator_quality.json"
V11_DYNAMIC = V11_ROOT / "preattempt/dynamic_preflight.json"
V11_REVIEW = ROOT / "research/raw/specification/four-codebook-palette-scale18-v11-review-done-20260807T034150Z.json"
V11_REVIEW_SUM = V11_REVIEW.with_suffix(".sha256")
V11_REVIEW_SOURCE = ROOT / ".autors/ace-2/wiki/sources/runs/f8af22ec7fec-r002.md"
V12_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_PRESERVING_MIXED_FOUR_BANK_LM_HEAD64_GROUPED_DUAL_CODEBOOK_PACKED_SCALE24_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V12_PLAN.json"
V12_PLAN_SUM = V12_PLAN.with_suffix(".sha256")
V12_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_PRESERVING_MIXED_FOUR_BANK_LM_HEAD64_GROUPED_DUAL_CODEBOOK_PACKED_SCALE24_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V12_ENGINEER_TASK.json"
V12_TASK_SUM = V12_TASK.with_suffix(".sha256")
V12_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-preserving-mixed-four-bank-lm-head64-grouped-dual-codebook-packed-scale24-w4-grouped-dynamic-scale32-w4a8-v12"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")
LAYER_SHAPES = (
    ("q_proj", 896, 896),
    ("k_proj", 128, 896),
    ("v_proj", 128, 896),
    ("o_proj", 896, 896),
    ("gate_proj", 4864, 896),
    ("up_proj", 4864, 896),
    ("down_proj", 896, 4864),
)
FOUR_BANK_FAMILIES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def companion_matches(path: Path, companion: Path) -> bool:
    expected = f"{sha256(path)}  {path.relative_to(ROOT)}"
    return companion.is_file() and companion.read_text(encoding="utf-8").strip() == expected


def aligned_bytes(byte_count: int, alignment: int = 16) -> int:
    return math.ceil(byte_count / alignment) * alignment


def layer_storage() -> dict[str, int]:
    result = {"payload": 0, "scales": 0, "codebooks": 0, "selectors": 0, "weight_count": 0}
    for name, rows, input_features in LAYER_SHAPES:
        blocks = input_features // 128
        banks = 4 if name in FOUR_BANK_FAMILIES else 2
        selector_bits = 2 if name in FOUR_BANK_FAMILIES else 1
        result["payload"] += rows * input_features // 2
        result["scales"] += aligned_bytes(rows * 3)
        result["codebooks"] += blocks * banks * 16
        result["selectors"] += aligned_bytes(math.ceil(rows * blocks * selector_bits / 8))
        result["weight_count"] += rows * input_features
    result["total"] = result["payload"] + result["scales"] + result["codebooks"] + result["selectors"]
    return result


def lm_head_storage() -> dict[str, int]:
    rows, input_features = 151936, 896
    blocks = input_features // 128
    result = {
        "payload": rows * input_features // 2,
        "scales": aligned_bytes(rows * 3),
        "codebooks": 64 * blocks * 2 * 16,
        "selectors": aligned_bytes(math.ceil(rows * blocks / 8)),
        "weight_count": rows * input_features,
    }
    result["total"] = result["payload"] + result["scales"] + result["codebooks"] + result["selectors"]
    return result


def table_classes(section: str) -> list[str]:
    result: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"} and not set(cells[0]) <= {"-", ":"}:
            result.append(cells[0])
    return result


def bound_artifacts_match(no_go: dict) -> tuple[bool, int]:
    errors = 0
    for record in no_go.get("bound_artifacts", {}).values():
        path = ROOT / record["path"]
        if not path.is_file():
            errors += 1
            continue
        raw = path.read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            errors += 1
    return errors == 0, errors


def audit() -> dict:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    no_go = load(V11_NO_GO)
    model = load(V11_MODEL)
    quality = load(V11_QUALITY)
    dynamic = load(V11_DYNAMIC)
    review = load(V11_REVIEW)
    plan = load(V12_PLAN)
    task = load(V12_TASK)
    shell = audit_shell_contract()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in classes]
    required_spec = (
        "option_b_alias_safe_v10_preserving_mixed_four_bank_lm_head64_grouped_dual_codebook_packed_scale24_w4_grouped_dynamic_scale32_w4a8_v12",
        "task-002f07633b3f",
        "2351.9847979241645",
        "4.035834478021978",
        "4.035440674268865",
        "4.035725946548808",
        "64 fixed contiguous groups",
        "exact three-byte packed Scale24 record",
        "`selected_policy_id` is null",
    )
    missing_spec = [item for item in required_spec if item not in spec]

    bound_ok, bound_errors = bound_artifacts_match(no_go)
    activation = dynamic.get("activation_execution", {})
    v11_terminal = all(
        (
            no_go.get("status") == "PREATTEMPT_NO_GO",
            no_go.get("attempts_consumed") == 0,
            no_go.get("official_execution_marker_exists") is False,
            no_go.get("gate_passes", {}).get("two_build_alias_codec_and_manifest_closure") is True,
            no_go.get("gate_passes", {}).get("v10_ledger_and_169_model_closure") is False,
            no_go.get("gate_passes", {}).get("non_evaluator_quality_8_of_8") is False,
            no_go.get("gate_passes", {}).get("dynamic_312_event_closure") is True,
            model.get("tensor_count") == 169,
            model.get("per_tensor_non_regressing_count") == 0,
            model.get("aggregate_sealed_v10_bf16_sse") == 2153.4047236346078,
            model.get("aggregate_v11_bf16_sse") == 2351.9847979241645,
            quality.get("exact_bf16_sequence_match_count") == 1,
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
            bound_ok,
            len(no_go.get("bound_artifacts", {})) == 32,
            review.get("status") == "done",
            review.get("failure_taxonomy")
            == ["NO_GO_V10_LEDGER_MODEL_ONLY_RECONSTRUCTION_GATE", "NO_GO_NON_EVALUATOR_QUALITY_GATE"],
            review.get("fresh_reviewer_handoff", {}).get("project_local_source_sha256")
            == sha256(V11_REVIEW_SOURCE),
            companion_matches(V11_REVIEW, V11_REVIEW_SUM),
            not (V11_ROOT / "PREATTEMPT_READY.json").exists(),
            not (V11_ROOT / "attempt-0001").exists(),
        )
    )

    layer = layer_storage()
    lm_head = lm_head_storage()
    whole = {key: 24 * layer[key] + lm_head[key] for key in layer}
    limits = plan.get("compression_and_accelerator_limits", {})
    accounting = all(
        (
            layer == {
                "payload": 7454720,
                "scales": 38016,
                "codebooks": 3904,
                "selectors": 24864,
                "weight_count": 14909440,
                "total": 7521504,
            },
            lm_head == {
                "payload": 68067328,
                "scales": 455808,
                "codebooks": 14336,
                "selectors": 132944,
                "weight_count": 136134656,
                "total": 68670416,
            },
            whole["payload"] == 246980608,
            whole["scales"] == 1368192,
            whole["codebooks"] == 108032,
            whole["selectors"] == 729680,
            whole["weight_count"] == 493961216,
            whole["total"] == 249186512,
            math.isclose(layer["total"] * 8 / layer["weight_count"], 4.035834478021978),
            math.isclose(lm_head["total"] * 8 / lm_head["weight_count"], 4.035440674268865),
            math.isclose(whole["total"] * 8 / whole["weight_count"], 4.035725946548808),
            limits.get("standard_decoder_layer_total_weight_bytes") == layer["total"],
            limits.get("lm_head_total_weight_bytes") == lm_head["total"],
            limits.get("whole_model_total_weight_bytes") == whole["total"],
            limits.get("standard_decoder_layer_exact_bits_per_weight") < 4.036,
            limits.get("lm_head_exact_bits_per_weight") < 4.036,
            limits.get("whole_model_exact_bits_per_weight") < 4.036,
        )
    )

    v12_frozen = all(
        (
            plan.get("status") == "frozen_not_started",
            plan.get("stage") == "specification",
            plan.get("selected_policy_id") is None,
            plan.get("weight_policy", {}).get("packed_scale24", {}).get("bytes_per_output_row") == 3,
            plan.get("weight_policy", {}).get("lm_head_bank_policy", {}).get("contiguous_row_group_count") == 64,
            plan.get("weight_policy", {}).get("lm_head_bank_policy", {}).get("rows_per_group") == 2374,
            companion_matches(V12_PLAN, V12_PLAN_SUM),
            task.get("task_id") == "task-002f07633b3f",
            task.get("status") == "authorized_not_started",
            task.get("stage") == "specification",
            task.get("candidate_plan", {}).get("sha256") == sha256(V12_PLAN),
            companion_matches(V12_TASK, V12_TASK_SUM),
            not V12_ROOT.exists(),
            accounting,
        )
    )

    internal = benchmark.get("internal_acceptance_contract", {})
    planned = internal.get(
        "planned_alias_safe_v10_preserving_mixed_four_bank_lm_head64_grouped_dual_codebook_packed_scale24_v12",
        {},
    )
    benchmark_routing = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure")
            == "explicit_non_benchmark_statement",
            planned.get("contract_sha256") == sha256(V12_PLAN),
            planned.get("engineer_task_sha256") == sha256(V12_TASK),
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

    unknowns = [line[2:] for line in ground_truth.splitlines() if line.startswith("- ")]
    unknowns_current = "task-002f07633b3f" in ground_truth and "task-7f2c91d8a44e" not in ground_truth
    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            v11_terminal,
            v12_frozen,
            benchmark_routing,
            unknowns_current,
        )
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no candidate construction, official input access, RTL execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v11_terminal_fresh_review_verified": v11_terminal,
            "v11_bound_artifact_errors": bound_errors,
            "v11_aggregate_bf16_sse": model.get("aggregate_v11_bf16_sse"),
            "v11_exact_sequence_matches": quality.get("exact_bf16_sequence_match_count"),
            "v11_official_attempts_consumed": no_go.get("attempts_consumed"),
            "v12_plan_and_task_frozen": v12_frozen,
            "v12_engineer_task_id": task.get("task_id"),
            "v12_namespace_absent": not V12_ROOT.exists(),
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
                V11_NO_GO,
                V11_MODEL,
                V11_QUALITY,
                V11_DYNAMIC,
                V11_REVIEW,
                V11_REVIEW_SUM,
                V12_PLAN,
                V12_PLAN_SUM,
                V12_TASK,
                V12_TASK_SUM,
                Path(__file__),
            )
        },
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_v10_preserving_mixed_bank_lm_head64_v12.py",
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
