#!/usr/bin/env python3
"""Read-only specification audit for terminal V20 and frozen output-blind V21."""

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
PLAN = ROOT / "design/OPTION_B_POST_V20_BLOCK4_AFFINE_FP16_W4A8_V21_PLAN.json"
TASK = ROOT / "design/OPTION_B_POST_V20_BLOCK4_AFFINE_FP16_W4A8_V21_ENGINEER_TASK.json"
SELECTION_PROBE = ROOT / "research/probes/post-v20-structural-quantizer-comparison-20260807.json"
SELECTION_TOOL = ROOT / "tools/probe_post_v20_structural_quantizers.py"
GEOMETRY = ROOT / "research/probes/post-v20-block4-affine-fp16-source-geometry-20260807.json"
GEOMETRY_TOOL = ROOT / "tools/audit_post_v20_block4_fp16_source_geometry.py"
CODEC = ROOT / "build/post-v20-block4-affine-fp16-w4a8-v21-source-independent-quantizer-regression.json"
CODEC_TOOL = ROOT / "tools/test_block4_affine_fp16_codec.py"
V20_REVIEW = ROOT / "research/raw/specification/v20-block16-affine-bf16-quality-no-go-review-replan-20260807T161101Z.json"
V20_RESULT = ROOT / "build/stage1-option-b-post-v19-block16-affine-bf16-w4a8-v20/quality-campaign-0001/RESULT.json"
V20_SUMS = ROOT / "build/stage1-option-b-post-v19-block16-affine-bf16-w4a8-v20/quality-campaign-0001/SHA256SUMS"
V21_ROOT = ROOT / "build/stage1-option-b-post-v20-block4-affine-fp16-w4a8-v21"
V21_RUNNER = ROOT / "tools/run_option_b_post_v20_block4_affine_fp16_w4a8_v21.py"
DIAGNOSTIC = ROOT / "research/probes/post-v20-bf16-w4-w4a8-summary-divergence-20260807.json"
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
    return len(fields) == 2 and fields[0] == sha256(path) and fields[1] == path.name


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
    selection = load(SELECTION_PROBE)
    geometry = load(GEOMETRY)
    codec = load(CODEC)
    v20_review = load(V20_REVIEW)
    v20_result = load(V20_RESULT)
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
        "The `2026-08-07T16:24:00Z` V21 selection boundary forbids",
        "option_b_post_v20_block4_affine_fp16_offset_step_w4a8_v21",
        "exactly 4.0 payload bits and 12.0 total stored bits",
        "Eight blocks cover\n32 weights",
        "Only a freshly Manager-authorized Engineer may implement or execute",
        "`SHA256SUMS.sha256` companion authenticates that manifest",
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
                    benchmark.get("schema_version") == 22,
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

    v20_terminal = all(
        (
            sha256(V20_REVIEW) == "5690fc8c2a6b6b6b4992c04d08a55d1fd9a1218b9541e16dbb7154297ee2e03e",
            sha256(V20_RESULT) == "7f2d5d5afef2c3e5f932bf0e56e5eba40600c79bf3f0bdde9b20030557a47b14",
            sha256(V20_SUMS) == "a535b7c46237fcade64079f54d07539b57fde295f7d92ef96dfcbcdfca1bc9a4",
            v20_review.get("review_status") == "replan_requested",
            v20_review.get("terminal_status") == "SOURCE_RELATIVE_QUALITY_NO_GO",
            v20_review.get("failure_taxonomy") == "SOURCE_RELATIVE_QUALITY_REGRESSION",
            v20_review.get("terminal_evidence", {}).get("checksum_manifest", {}).get("self_resolving") is False,
            v20_result.get("status") == "SOURCE_RELATIVE_QUALITY_NO_GO",
            v20_result.get("selected_policy_id") is None,
            "V20 consumed one construction and one quality campaign" in checkpoint,
        )
    )

    policies = selection.get("policies", {})
    chosen = policies.get("block4_affine_fp16", {})
    comparison_sse = {
        name: float(record["aggregate_literal_bf16_sse"])
        for name, record in policies.items()
        if isinstance(record, dict) and "aggregate_literal_bf16_sse" in record
    }
    selection_valid = all(
        (
            sha256(SELECTION_PROBE) == "2ec57fd96fb09f2824b4ad9b87822da12b3f2ae731258d3c4c922ee517023b40",
            sha256(SELECTION_TOOL) == "bdc308457c6e570792da1e7f8049cd30f857ae774adbff1f76d5ab180a7ced82",
            selection.get("classification") == "non_consuming_source_weight_only_structural_quantizer_comparison",
            selection.get("linear_tensor_count") == 169,
            selection.get("linear_weight_count") == 493961216,
            len(comparison_sse) == 5,
            chosen.get("aggregate_literal_bf16_sse") == 162.79158489970018,
            chosen.get("relative_sse_vs_v20") == 0.21244082983525642,
            chosen.get("per_tensor_nonregressing_vs_v20") == 169,
            chosen.get("per_tensor_strictly_improved_vs_v20") == 169,
            chosen.get("exact_total_bits_per_weight") == 12.0,
            chosen.get("aggregate_literal_bf16_sse") == min(comparison_sse.values()),
        )
    )

    geometry_record = geometry.get("geometry", {})
    storage = geometry.get("storage", {})
    geometry_valid = all(
        (
            sha256(GEOMETRY) == "9af8217976498c5515b94c1ed80978d2bc28b2d2e6a1bd5eefbe48c62746ee17",
            sha256(GEOMETRY_TOOL) == "759e013cc2cf84656586fd0d26705ca7486174edf8dab4865f9cde6945a1ba93",
            geometry.get("status") == "PASS_NON_CONSUMING_BLOCK4_FP16_SOURCE_GEOMETRY",
            geometry_record.get("block_width") == 4,
            geometry_record.get("linear_tensor_count") == 169,
            geometry_record.get("linear_weight_count") == 493961216,
            geometry_record.get("block_count") == 123490304,
            geometry_record.get("nonconstant_zero_fp16_step_block_count") == 0,
            geometry_record.get("nonfinite_fp16_metadata_block_count") == 0,
            geometry_record.get("geometry_manifest_sha256")
            == "9cb79eeaf5110d175dedb280182470a18bdd369e1ca9e30ed37f4cec2d764c5a",
            storage.get("four_bit_payload_bytes") == 246980608,
            storage.get("metadata_bytes") == 493961216,
            storage.get("total_linear_weight_bytes") == 740941824,
            storage.get("exact_total_bits_per_weight") == 12.0,
            geometry.get("official_namespace_absent") is True,
        )
    )

    codec_valid = all(
        (
            sha256(CODEC) == "9c1db042d8bc52b1a2a6fe0be3cd1a446bda86a74fc76ff869ef70f42573b175",
            sha256(CODEC_TOOL) == "877294c934464f73c3db4be5007c3289fac6218d8f8035b63a64e9738e8d022a",
            codec.get("status") == "PASS_SOURCE_INDEPENDENT_QUANTIZER_REGRESSION",
            len(codec.get("directed_cases", [])) == 7,
            codec.get("deterministic_sweep", {}).get("block_count") == 4096,
            codec.get("deterministic_sweep", {}).get("scalar_tensor_identity") is True,
            codec.get("deterministic_sweep", {}).get("record_digest_sha256")
            == "af3b05251d18e893d39903d20c3a4391012b2bc7c2810c8158536bc6d970d6f0",
            all(codec.get("illegal_case_rejection", {}).values()),
            codec.get("official_namespace_absent") is True,
        )
    )

    frozen = benchmark.get("product_compression_gate", {}).get(
        "frozen_v21_output_blind_block4_fp16_candidate", {}
    )
    contract_frozen = all(
        (
            sha256(PLAN) == "b0579c50a3b565f87b20f2d54ebfce9ed80f7e6280831eb309131af63f84c8bb",
            sha256(TASK) == "ea5db676be5791a0f145ccd2372dc390038805db8e669a00227f4e94b3b0d738",
            companion_matches(PLAN),
            companion_matches(TASK),
            plan.get("stage") == "specification",
            plan.get("status") == "frozen_not_started_awaiting_manager_authorization",
            plan.get("candidate_id") == task.get("candidate_id"),
            plan.get("backlog_id") == task.get("backlog_id"),
            plan.get("selection_boundary", {}).get("candidate_text_generated_before_freeze") is False,
            plan.get("selection_boundary", {}).get("selected_policy") == "block4_affine_fp16",
            plan.get("selection_boundary", {}).get("diagnostic_trace_status")
            == "retained only as non-authoritative failure localization; it did not rank, select, tune, or freeze V21",
            plan.get("authority", {}).get("planner_may_consume_authority") is False,
            plan.get("fresh_namespace", {}).get("construction_attempts_requested") == 1,
            plan.get("fresh_namespace", {}).get("quality_campaigns_requested") == 1,
            plan.get("terminal_integrity", {}).get("required_manifest_companion")
            == "quality-campaign-0001/SHA256SUMS.sha256",
            task.get("stage") == "specification",
            task.get("scope") == "bounded",
            task.get("status") == "AWAITING_MANAGER_AUTHORIZATION",
            task.get("candidate_plan", {}).get("sha256") == sha256(PLAN),
            task.get("fresh_namespace", {}).get("construction_attempts_requested") == 1,
            task.get("fresh_namespace", {}).get("quality_campaigns_requested") == 1,
            not V21_ROOT.exists(),
            not V21_RUNNER.exists(),
            frozen.get("status") == "FROZEN_AWAITING_MANAGER_AUTHORIZATION",
            frozen.get("official_namespace_exists") is False,
            frozen.get("runner_exists") is False,
            frozen.get("candidate_text_generated") is False,
            frozen.get("construction_attempts_consumed") == 0,
            frozen.get("quality_campaigns_consumed") == 0,
            frozen.get("planner_may_consume_authority") is False,
            frozen.get("one_shot_fresh_reviewer_required") is True,
            frozen.get("selected_policy_id") is None,
        )
    )

    unknowns = [line[2:] for line in ground_truth.splitlines() if line.startswith("- ")]
    ground_truth_current = all(
        (
            ground_truth.startswith("# Binding Unknowns\n"),
            len(unknowns) == 5,
            all("unknown" in line.lower() for line in unknowns),
            "frozen V21 block-4 affine FP16 W4A8 successor" in ground_truth,
            "Which source-output-blind quantization geometry" not in ground_truth,
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
    diagnostic_non_authoritative = all(
        (
            DIAGNOSTIC.is_file(),
            "diagnostic trace is non-authoritative localization only" in checkpoint,
            plan.get("selection_boundary", {}).get("prohibited_selection_inputs")[-1]
            == "the diagnostic post-V20 BF16/W4/W4A8 divergence trace",
        )
    )

    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            pipeline.get("stages", {}).get("specification", {}).get("status") == "in_progress",
            v20_terminal,
            selection_valid,
            geometry_valid,
            codec_valid,
            contract_frozen,
            ground_truth_current,
            u280_current,
            diagnostic_non_authoritative,
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
        SELECTION_PROBE,
        SELECTION_TOOL,
        GEOMETRY,
        GEOMETRY_TOOL,
        CODEC,
        CODEC_TOOL,
        V20_REVIEW,
        V20_RESULT,
        V20_SUMS,
        DIAGNOSTIC,
        U280,
        Path(__file__),
    )
    return {
        "schema_version": 1,
        "scope": "non-consuming specification audit; no predecessor replay, V21 construction, candidate text generation, RTL edit or execution, simulation, formal, synthesis, PPA, chat retarget, FPGA build, emulation, board execution, proprietary download, or Manager-owned stage mutation",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {
            "current_stage": pipeline.get("current_stage"),
            "specification_status": pipeline.get("stages", {}).get("specification", {}).get("status"),
            "manager_owned_state_mutated": False,
        },
        "checklist": checklist,
        "mission_gate": {
            "terminal_v20_verified": v20_terminal,
            "source_weight_only_selection_verified": selection_valid,
            "source_geometry_verified": geometry_valid,
            "source_independent_codec_verified": codec_valid,
            "v21_contract_frozen": contract_frozen,
            "diagnostic_trace_non_authoritative": diagnostic_non_authoritative,
            "official_v21_namespace_absent": not V21_ROOT.exists(),
            "v21_runner_absent": not V21_RUNNER.exists(),
            "candidate_text_generated": False,
            "selected_policy_id": benchmark.get("product_compression_gate", {}).get("selected_policy_id"),
            "project_completion_eligible": False,
        },
        "v21_measurements": {
            "aggregate_literal_bf16_sse": chosen.get("aggregate_literal_bf16_sse"),
            "relative_sse_vs_v20": chosen.get("relative_sse_vs_v20"),
            "per_tensor_strictly_improved_vs_v20": chosen.get("per_tensor_strictly_improved_vs_v20"),
            "block_count": geometry_record.get("block_count"),
            "exact_total_bits_per_weight": storage.get("exact_total_bits_per_weight"),
            "payload_bytes": storage.get("four_bit_payload_bytes"),
            "metadata_bytes": storage.get("metadata_bytes"),
            "total_bytes": storage.get("total_linear_weight_bytes"),
        },
        "u280_local_inventory": {
            "record": str(U280.relative_to(ROOT)),
            "record_sha256": sha256(U280),
            "current_negative_inventory_verified": u280_current,
        },
        "input_bindings": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools .venv/bin/python tools/audit_specification_stage_v21.py --output research/raw/specification/specification-stage-audit-v21-output-blind-freeze-20260807T164243Z.json",
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
