#!/usr/bin/env python3
"""Read-only specification audit for terminal V12 and frozen V13."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
V12_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-preserving-mixed-four-bank-lm-head64-grouped-dual-codebook-packed-scale24-w4-grouped-dynamic-scale32-w4a8-v12"
V12_NO_GO = V12_ROOT / "PREATTEMPT_NO_GO.json"
V12_MODEL = V12_ROOT / "preattempt/model_only_v10_relative_reconstruction.json"
V12_QUALITY = V12_ROOT / "preattempt/non_evaluator_quality.json"
V12_DYNAMIC = V12_ROOT / "preattempt/dynamic_preflight.json"
V12_REVIEW = ROOT / "research/raw/specification/v10-preserving-mixed-bank-lm-head64-v12-review-done-20260807T051730Z.json"
V13_PLAN = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V13_PLAN.json"
V13_PLAN_SUM = V13_PLAN.with_suffix(".sha256")
V13_TASK = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V13_ENGINEER_TASK.json"
V13_TASK_SUM = V13_TASK.with_suffix(".sha256")
V13_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v13"
U280_RECHECK = ROOT / "research/raw/specification/u280-host-inventory-recheck-20260807T052627Z.json"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def companion_matches(path: Path, companion: Path) -> bool:
    expected = f"{sha256(path)}  {path.relative_to(ROOT)}"
    return companion.is_file() and companion.read_text(encoding="utf-8").strip() == expected


def table_classes(section: str) -> list[str]:
    result: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"} and not set(cells[0]) <= {"-", ":"}:
            result.append(cells[0])
    return result


def bound_artifacts_match(record: dict) -> tuple[bool, int]:
    errors = 0
    for item in record.get("bound_artifacts", {}).values():
        path = ROOT / item["path"]
        if not path.is_file():
            errors += 1
            continue
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            errors += 1
    return errors == 0, errors


def storage_accounting() -> dict[str, dict[str, int | float]]:
    layer = {
        "weight_count": 14909440,
        "payload": 7454720,
        "packed_scale24": 38016,
        "codebooks": 2560,
        "selectors": 14560,
    }
    layer["metadata"] = layer["packed_scale24"] + layer["codebooks"] + layer["selectors"]
    layer["total"] = layer["payload"] + layer["metadata"]
    layer["bits_per_weight"] = layer["total"] * 8 / layer["weight_count"]

    lm_head = {
        "weight_count": 136134656,
        "payload": 68067328,
        "packed_scale24": 455808,
        "codebooks": 14336,
        "selectors": 132944,
    }
    lm_head["metadata"] = lm_head["packed_scale24"] + lm_head["codebooks"] + lm_head["selectors"]
    lm_head["total"] = lm_head["payload"] + lm_head["metadata"]
    lm_head["bits_per_weight"] = lm_head["total"] * 8 / lm_head["weight_count"]

    whole = {
        key: 24 * int(layer[key]) + int(lm_head[key])
        for key in ("weight_count", "payload", "packed_scale24", "codebooks", "selectors", "metadata", "total")
    }
    whole["bits_per_weight"] = whole["total"] * 8 / whole["weight_count"]
    return {"standard_decoder_layer": layer, "lm_head": lm_head, "whole_model": whole}


def audit() -> dict:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    no_go = load(V12_NO_GO)
    model = load(V12_MODEL)
    quality = load(V12_QUALITY)
    dynamic = load(V12_DYNAMIC)
    review = load(V12_REVIEW)
    plan = load(V13_PLAN)
    task = load(V13_TASK)
    u280 = load(U280_RECHECK)
    shell = audit_shell_contract()
    storage = storage_accounting()
    shell_source = (ROOT / "rtl/ace2_shell.sv").read_text(encoding="utf-8")
    shell_header = shell_source[
        shell_source.index("module ace2_shell") : shell_source.index(");", shell_source.index("module ace2_shell")) + 2
    ]
    signed_public_port_count = len(
        re.findall(r"^\s*(?:input|output)[^\n]*\bsigned\b", shell_header, re.MULTILINE)
    )

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in classes]
    required_spec = (
        "option_b_alias_safe_v10_transformer_v12_lm_head64_product_chat_w4_grouped_dynamic_scale32_w4a8_v13",
        "task-6b8d1f4a2c73",
        "246,980,608 four-bit payload bytes",
        "1,926,352 explicitly accounted weight-",
        "248,906,960 total weight bytes",
        "4.031198433198448",
        "BF16 token identity is not the accelerator oracle",
        "`selected_policy_id` is null",
    )
    missing_spec = [item for item in required_spec if item not in spec]

    bound_ok, bound_errors = bound_artifacts_match(no_go)
    activation = dynamic.get("activation_execution", {})
    v12_terminal = all(
        (
            no_go.get("status") == "PREATTEMPT_NO_GO",
            no_go.get("attempts_consumed") == 0,
            no_go.get("official_execution_marker_exists") is False,
            no_go.get("gate_passes", {}).get("v10_seed_exact_169") is True,
            no_go.get("gate_passes", {}).get("v10_ledger_and_169_model_closure") is True,
            no_go.get("gate_passes", {}).get("non_evaluator_quality_8_of_8") is False,
            no_go.get("gate_passes", {}).get("dynamic_312_event_closure") is True,
            model.get("tensor_count") == 169,
            model.get("per_tensor_non_regressing_count") == 169,
            model.get("aggregate_v12_bf16_sse") == 2147.8711827687966,
            quality.get("exact_bf16_sequence_match_count") == 0,
            quality.get("response_gate_outcome_match_count") == 7,
            quality.get("official_nine_prompt_inputs_used") is False,
            activation.get("named_boundary_count") == 312,
            activation.get("saturation_count") == 0,
            activation.get("clipping_count") == 0,
            activation.get("reserved_minus_128_produced") is False,
            activation.get("transport_metadata_bytes_per_decoder_layer_token") == 832,
            review.get("status") == "done",
            review.get("failure_taxonomy") == ["NO_GO_NON_EVALUATOR_QUALITY_GATE"],
            review.get("reviewed_facts", {}).get("official_attempts_consumed") == 0,
            bound_ok,
            not (V12_ROOT / "PREATTEMPT_READY.json").exists(),
            not (V12_ROOT / "attempt-0001").exists(),
        )
    )

    whole = storage["whole_model"]
    limits = plan.get("storage_and_accelerator_accounting", {})
    accounting = all(
        (
            storage["standard_decoder_layer"]["total"] == 7509856,
            storage["lm_head"]["total"] == 68670416,
            whole["payload"] == 246980608,
            whole["metadata"] == 1926352,
            whole["total"] == 248906960,
            math.isclose(float(whole["bits_per_weight"]), 4.031198433198448),
            limits.get("whole_model_four_bit_payload_bytes") == whole["payload"],
            limits.get("whole_model_weight_metadata_bytes") == whole["metadata"],
            limits.get("whole_model_total_weight_bytes") == whole["total"],
            limits.get("obsolete_4_036_cap_used_as_acceptance_gate") is False,
        )
    )

    v13_frozen = all(
        (
            plan.get("status") == "frozen_not_started",
            plan.get("stage") == "specification",
            plan.get("selected_policy_id") is None,
            plan.get("weight_policy", {}).get("payload_bits_per_weight") == 4,
            plan.get("quality_rubric", {}).get("bf16_token_identity_is_acceptance_criterion") is False,
            plan.get("frozen_product_quality_matrix", {}).get("frozen_before_candidate_output") is True,
            companion_matches(V13_PLAN, V13_PLAN_SUM),
            task.get("task_id") == "task-6b8d1f4a2c73",
            task.get("status") == "authorized_not_started",
            task.get("stage") == "specification",
            task.get("candidate_plan", {}).get("sha256") == sha256(V13_PLAN),
            companion_matches(V13_TASK, V13_TASK_SUM),
            not V13_ROOT.exists(),
            accounting,
        )
    )

    active = benchmark.get("product_compression_gate", {}).get("active_v13_contract", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    planned = internal.get("planned_alias_safe_v10_transformer_v12_lm_head64_product_chat_v13", {})
    benchmark_routing = all(
        (
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("local_contract", {}).get("top_module") == "ace2_shell",
            benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure")
            == "explicit_non_benchmark_statement",
            active.get("plan_sha256") == sha256(V13_PLAN),
            active.get("engineer_task_sha256") == sha256(V13_TASK),
            active.get("bf16_token_identity_is_acceptance_gate") is False,
            planned.get("contract_sha256") == sha256(V13_PLAN),
            planned.get("engineer_task_sha256") == sha256(V13_TASK),
            planned.get("namespace_exists") is False,
            internal.get("selected_policy_id") is None,
        )
    )

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS" if shell.get("status") == "PASS" and not missing_spec else "FAIL",
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "signed_public_port_count": signed_public_port_count,
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
    unknowns_current = (
        "task-6b8d1f4a2c73" in ground_truth
        and "task-4ea0126f2323" not in ground_truth
        and "exact bf16 token identity" in ground_truth.lower()
    )
    u280_current = all(
        (
            u280.get("observed_at_utc") == "2026-08-07T05:26:27Z",
            u280.get("u280_access") is False,
            not any(u280.get("vendor_tools_in_path", {}).values()),
            u280.get("pci_inventory", {}).get("xilinx_vendor_10ee_functions") == [],
            u280.get("device_nodes", {}).get("xclmgmt") == [],
            u280.get("device_nodes", {}).get("renderD") == [],
        )
    )
    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            v12_terminal,
            v13_frozen,
            benchmark_routing,
            unknowns_current,
            u280_current,
        )
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no V13 construction or quality execution, official input access, RTL edit or execution, simulation, formal, synthesis, PPA, chat retarget, FPGA work, hardware emulation, board execution, or attempt consumption",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v12_terminal_fresh_review_verified": v12_terminal,
            "v12_bound_artifact_errors": bound_errors,
            "v12_official_attempts_consumed": no_go.get("attempts_consumed"),
            "v13_plan_and_task_frozen": v13_frozen,
            "v13_engineer_task_id": task.get("task_id"),
            "v13_namespace_absent": not V13_ROOT.exists(),
            "v13_bf16_identity_gate_active": plan.get("quality_rubric", {}).get("bf16_token_identity_is_acceptance_criterion"),
            "benchmark_routing_matches": benchmark_routing,
            "selected_policy_id": internal.get("selected_policy_id"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "byte_accounting": storage,
        "u280_local_inventory": {
            "record": str(U280_RECHECK.relative_to(ROOT)),
            "record_sha256": sha256(U280_RECHECK),
            "current_negative_inventory_verified": u280_current,
        },
        "binding_unknowns": {"count": len(unknowns), "items": unknowns},
        "input_bindings": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                SPEC,
                BENCHMARK,
                PIPELINE,
                GROUND_TRUTH,
                V12_NO_GO,
                V12_MODEL,
                V12_QUALITY,
                V12_DYNAMIC,
                V12_REVIEW,
                V13_PLAN,
                V13_PLAN_SUM,
                V13_TASK,
                V13_TASK_SUM,
                U280_RECHECK,
                Path(__file__),
            )
        },
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage_v13_product_quality.py",
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
