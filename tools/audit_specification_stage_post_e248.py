#!/usr/bin/env python3
"""Read-only audit for the terminal e248 evidence and frozen post-e248 successor."""

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
CHECKPOINT = ROOT / "CHECKPOINT.md"
PLAN = ROOT / "design/OPTION_B_POST_E248_BLOCK16_AFFINE_BF16_W4A8_V1_PLAN.json"
TASK = ROOT / "design/OPTION_B_POST_E248_BLOCK16_AFFINE_BF16_W4A8_V1_ENGINEER_TASK.json"
RUNNER = ROOT / "tools/run_option_b_post_e248_block16_affine_bf16_w4a8_v1.py"
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-post-e248-block16-affine-bf16-w4a8-v1"
PROBE = ROOT / "research/probes/post-e248-weight-only-quantizer-comparison-20260807.json"
PROBE_TOOL = ROOT / "tools/probe_post_e248_weight_only_quantizers.py"
REVIEW = ROOT / "research/raw/specification/e248e307675a-block32-affine-review-replan-20260807T142652Z.json"
E248_ROOT = ROOT / "build/candidate-e248e307675a-block32-affine-fp32-w4a8-v1"
E248_RESULT = E248_ROOT / "quality-campaign-0001/RESULT.json"
E248_RUBRIC = E248_ROOT / "quality-campaign-0001/source_relative_rubric.json"
E248_ACCOUNTING = E248_ROOT / "codec_and_byte_accounting.json"
U280 = ROOT / "research/raw/specification/u280-host-inventory-recheck-20260807T133413Z.json"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(".sha256")
    if not companion.is_file():
        return False
    fields = companion.read_text(encoding="utf-8").strip().split()
    return bool(fields) and fields[0] == sha256(path)


def table_classes(section: str) -> list[str]:
    classes: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"} and not set(cells[0]) <= {"-", ":"}:
            classes.append(cells[0])
    return classes


def audit() -> dict[str, Any]:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    plan = load(PLAN)
    task = load(TASK)
    probe = load(PROBE)
    review = load(REVIEW)
    e248_result = load(E248_RESULT)
    e248_rubric = load(E248_RUBRIC)
    e248_accounting = load(E248_ACCOUNTING)
    u280 = load(U280)
    shell = audit_shell_contract()

    shell_source = (ROOT / "rtl/ace2_shell.sv").read_text(encoding="utf-8")
    start = shell_source.index("module ace2_shell")
    header = shell_source[start : shell_source.index(");", start) + 2]
    signed_public_ports = len(re.findall(r"^\s*(?:input|output)[^\n]*\bsigned\b", header, re.MULTILINE))

    matrix_section = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix_section)
    missing_classes = [item for item in REQUIRED_CLASSES if item not in classes]
    required_spec_fragments = (
        "Mission `e248e307675a` then executed one fresh output-blind block-32 affine",
        "`SOURCE_RELATIVE_QUALITY_NO_GO`",
        "option_b_post_e248_block16_affine_bf16_offset_step_w4a8_v1",
        "Four blocks consume 48 bytes, exactly three",
        "Only an Engineer may implement the bound runner",
        "No external benchmark contract or score applies",
    )
    missing_spec = [fragment for fragment in required_spec_fragments if fragment not in spec]

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS"
            if shell.get("status") == "PASS" and signed_public_ports == 0 and not missing_spec
            else "FAIL",
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "signed_public_port_count": signed_public_ports,
            "missing_fragments": missing_spec,
        },
        "spec.clock-reset-protocol": {
            "status": "PASS"
            if all(
                fragment in spec
                for fragment in (
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
            "status": "PASS"
            if all(
                (
                    benchmark.get("schema_version") == 20,
                    benchmark.get("stage") == "specification",
                    benchmark.get("applies") is False,
                    benchmark.get("external_contract") is None,
                    benchmark.get("specification_stage_evidence", {})
                    .get("benchmark_interface_closure", {})
                    .get("closure")
                    == "explicit_non_benchmark_statement",
                )
            )
            else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "closure": benchmark.get("specification_stage_evidence", {})
            .get("benchmark_interface_closure", {})
            .get("closure"),
        },
    }

    regressions = e248_rubric.get("source_action_pass_regressions", [])
    e248_terminal = all(
        (
            sha256(E248_RESULT) == "491d2e9920b0becc4a80f6b746da20b02dd60a2c18c3ece3bf1d67a5a64f0ba9",
            sha256(E248_RUBRIC) == "50246c1aff662a4ea26d3b363817ea34f1bd6238638b8ece9fdba4fe6ed4f97c",
            e248_result.get("status") == "SOURCE_RELATIVE_QUALITY_NO_GO",
            e248_result.get("failure_taxonomy") == "SOURCE_RELATIVE_QUALITY_REGRESSION",
            e248_result.get("deterministic_replay_passed") is True,
            e248_result.get("selected_policy_id") is None,
            e248_rubric.get("zero_source_action_pass_regressions") is False,
            regressions == [{"case_id": "concise_summary", "turn_index": 1}],
            e248_rubric.get("response_position_count") == 23,
            e248_rubric.get("delivered_response_position_count") == 22,
            e248_rubric.get("blocked_response_position_count") == 1,
            e248_rubric.get("delivered_integrity_pass_count") == 22,
            e248_accounting.get("four_bit_payload_bytes") == 246980608,
            e248_accounting.get("metadata_bytes") == 123490304,
            e248_accounting.get("total_linear_weight_bytes") == 370470912,
            e248_accounting.get("exact_total_bits_per_weight") == 6.0,
            review.get("review_status") == "replan_requested",
            review.get("failure_taxonomy") == "SOURCE_RELATIVE_QUALITY_REGRESSION",
            review.get("terminal_evidence", {}).get("result", {}).get("sha256") == sha256(E248_RESULT),
            "The genuine regression is the source-passing `concise_summary` turn 1" in checkpoint,
        )
    )

    policy = probe.get("policies", {}).get("block16_affine_bf16", {})
    source_probe_valid = all(
        (
            sha256(PROBE) == "d02028573a617b39a95985f570935a84279344ecdfbe72a305b8c84aea0b326e",
            sha256(PROBE_TOOL) == "f99e59eb09e81eb30ec77c39e9126e5c6849aec63a88f92e070baecad5c97cac",
            probe.get("classification") == "non_consuming_source_weight_only_quantizer_comparison",
            probe.get("linear_tensor_count") == 169,
            probe.get("linear_weight_count") == 493961216,
            policy.get("aggregate_literal_bf16_sse") == 766.2914187726616,
            policy.get("per_tensor_strictly_improved_vs_e248") == 169,
            policy.get("exact_total_bits_per_weight") == 6.0,
            policy.get("four_bit_payload_bytes") == 246980608,
            policy.get("metadata_bytes") == 123490304,
        )
    )

    frozen = benchmark.get("product_compression_gate", {}).get(
        "frozen_post_e248_block16_affine_bf16_candidate", {}
    )
    successor_frozen = all(
        (
            sha256(PLAN) == "d750bf5a260457973c116e4087962da78f0d6649f1f06962fc19eeba1c6b31dc",
            sha256(TASK) == "f0a313519892a9d1ee0a9d7fba7be05f424f94c6a7a61522ae6861c97c2e033c",
            companion_matches(PLAN),
            companion_matches(TASK),
            plan.get("stage") == "specification",
            plan.get("status") == "frozen_not_started",
            plan.get("candidate_id") == task.get("candidate_id"),
            plan.get("fresh_authority", {}).get("planner_may_consume_authority") is False,
            task.get("stage") == "specification",
            task.get("scope") == "bounded",
            task.get("status") == "authorized_not_started",
            task.get("fresh_namespace", {}).get("construction_attempts_authorized") == 1,
            task.get("fresh_namespace", {}).get("quality_campaigns_authorized") == 1,
            not OFFICIAL_ROOT.exists(),
            not RUNNER.exists(),
            frozen.get("status") == "FROZEN_NOT_STARTED",
            frozen.get("official_namespace_exists") is False,
            frozen.get("construction_attempts_consumed") == 0,
            frozen.get("quality_campaigns_consumed") == 0,
            frozen.get("manager_routing_required_before_engineer_execution") is True,
            frozen.get("selected_policy_id") is None,
        )
    )

    unknowns = [line[2:] for line in ground_truth.splitlines() if line.startswith("- ")]
    ground_truth_current = all(
        (
            ground_truth.startswith("# Binding Unknowns\n"),
            len(unknowns) == 5,
            all("unknown" in line.lower() for line in unknowns),
            "option_b_post_e248_block16_affine_bf16_offset_step_w4a8_v1" in ground_truth,
            "No post-`e248e307675a` successor contract is frozen" not in ground_truth,
        )
    )
    u280_current = all(
        (
            sha256(U280) == "34179ffcdbb9772db24fe20ae4f8e10f4734b49c325a890107daaeb457cee1af",
            u280.get("observed_at_utc") == "2026-08-07T13:34:13Z",
            u280.get("u280_access") is False,
            not any(u280.get("vendor_tools_in_path", {}).values()),
        )
    )

    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            pipeline.get("stages", {}).get("specification", {}).get("status") == "in_progress",
            e248_terminal,
            source_probe_valid,
            successor_frozen,
            ground_truth_current,
            u280_current,
        )
    )
    inputs = (
        SPEC,
        BENCHMARK,
        PIPELINE,
        GROUND_TRUTH,
        CHECKPOINT,
        PLAN,
        PLAN.with_suffix(".sha256"),
        TASK,
        TASK.with_suffix(".sha256"),
        PROBE,
        PROBE_TOOL,
        REVIEW,
        REVIEW.with_suffix(".sha256"),
        E248_RESULT,
        E248_RUBRIC,
        E248_ACCOUNTING,
        U280,
        Path(__file__),
    )
    return {
        "schema_version": 1,
        "scope": "non-consuming specification audit; no predecessor replay, candidate construction, text generation, RTL edit or execution, simulation, formal, synthesis, PPA, chat retarget, FPGA build, emulation, board execution, proprietary download, or Manager-owned stage mutation",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {
            "current_stage": pipeline.get("current_stage"),
            "specification_status": pipeline.get("stages", {}).get("specification", {}).get("status"),
            "manager_owned_state_mutated": False,
        },
        "checklist": checklist,
        "mission_gate": {
            "terminal_e248_verified": e248_terminal,
            "source_only_successor_probe_verified": source_probe_valid,
            "successor_contract_frozen": successor_frozen,
            "official_successor_namespace_absent": not OFFICIAL_ROOT.exists(),
            "runner_implementation_pending": not RUNNER.exists(),
            "selected_policy_id": benchmark.get("product_compression_gate", {}).get("selected_policy_id"),
            "project_completion_eligible": False,
        },
        "successor_measurements": {
            "aggregate_literal_bf16_sse": policy.get("aggregate_literal_bf16_sse"),
            "relative_sse_vs_e248": policy.get("relative_sse_vs_e248"),
            "per_tensor_strictly_improved_vs_e248": policy.get("per_tensor_strictly_improved_vs_e248"),
            "exact_total_bits_per_weight": policy.get("exact_total_bits_per_weight"),
            "payload_bytes": policy.get("four_bit_payload_bytes"),
            "metadata_bytes": policy.get("metadata_bytes"),
        },
        "u280_local_inventory": {
            "record": str(U280.relative_to(ROOT)),
            "record_sha256": sha256(U280),
            "current_negative_inventory_verified": u280_current,
        },
        "input_bindings": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools .venv/bin/python tools/audit_specification_stage_post_e248.py --output research/raw/specification/specification-stage-audit-post-e248-20260807T143315Z.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if output.exists():
            raise RuntimeError(f"refusing to overwrite existing audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
