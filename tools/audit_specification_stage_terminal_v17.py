#!/usr/bin/env python3
"""Read-only audit of the terminal V17 specification-stage state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"
PLAN = ROOT / "design/OPTION_B_SOURCE_RELATIVE_V2_PER16_SYMMETRIC_SCALE32_W4A8_V17_PLAN.json"
TASK = ROOT / "design/OPTION_B_SOURCE_RELATIVE_V2_PER16_SYMMETRIC_SCALE32_W4A8_V17_ENGINEER_TASK.json"
RUNNER = ROOT / "tools/run_option_b_source_relative_v2_per16_symmetric_scale32_w4a8_v17.py"
PROVENANCE_VERIFIER = ROOT / "tools/verify_v17_redacted_provenance.py"
V17_ROOT = ROOT / "build/stage1-option-b-source-relative-v2-per16-symmetric-scale32-w4a8-v17"
QUALITY_ROOT = V17_ROOT / "quality-campaign-0001"
RESULT = QUALITY_ROOT / "RESULT.json"
RUBRIC = QUALITY_ROOT / "source_relative_noninferiority.json"
LATENCY = QUALITY_ROOT / "token_and_latency_results.json"
CODEC = V17_ROOT / "codec_and_byte_accounting.json"
DYNAMIC = V17_ROOT / "dynamic_preflight.json"
CANDIDATE_A = V17_ROOT / "candidate_a_manifest.json"
CANDIDATE_B = V17_ROOT / "candidate_b_manifest.json"
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


def companion_hash_matches(path: Path) -> bool:
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


def run_provenance_verifier() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(PROVENANCE_VERIFIER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "redaction-aware V17 provenance verifier failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("redaction-aware V17 provenance verifier returned non-object JSON")
    return value


def audit() -> dict[str, Any]:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    result = load(RESULT)
    rubric = load(RUBRIC)
    latency = load(LATENCY)
    codec = load(CODEC)
    dynamic = load(DYNAMIC)
    u280 = load(U280)
    shell = audit_shell_contract()
    provenance = run_provenance_verifier()

    shell_source = (ROOT / "rtl/ace2_shell.sv").read_text(encoding="utf-8")
    start = shell_source.index("module ace2_shell")
    shell_header = shell_source[start : shell_source.index(");", start) + 2]
    signed_public_ports = len(
        re.findall(r"^\s*(?:input|output)[^\n]*\bsigned\b", shell_header, re.MULTILINE)
    )

    matrix_section = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix_section)
    missing_classes = [item for item in REQUIRED_CLASSES if item not in classes]

    required_spec_fragments = (
        "The structurally distinct V17 candidate and its sole quality campaign are now",
        "`SOURCE_RELATIVE_NONINFERIOR_NO_GO`",
        "`selected_policy_id` remains null",
        "No arbitrary-text",
        "multi-turn V17 demo was implemented",
        "tools/verify_v17_redacted_provenance.py",
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
                    benchmark.get("schema_version") == 19,
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

    terminal = benchmark.get("product_compression_gate", {}).get(
        "terminal_v17_source_relative_candidate", {}
    )
    internal = benchmark.get("internal_acceptance_contract", {})
    dynamic_execution = dynamic.get("activation_execution", {})
    campaign_execution = latency.get("campaign_activation_execution", {})
    v17_terminal = all(
        (
            sha256(PLAN) == "606b3045d74a83336852d7445361fed11723141ccaea4928ec9bba98c5f9bb5b",
            sha256(TASK) == "499a42ec45d520f8b32a1f8f26dfbce0061bcf08a05a55fbe2223ac49ec1f719",
            sha256(RUNNER) == "f03cee3c1f17f79fc3a84984ea8c7d19e10d666fde7066633e6a008bb218f2f0",
            sha256(RESULT) == "61291cfd345d5f8a70d43bf88ebdfd85f6db9a21d87f0ad68ba44bc0979cae3c",
            sha256(RUBRIC) == "62d57813c5feee156132d38a246c2cfcb9446de0326ac6db75b91095fe0ea41e",
            sha256(LATENCY) == "f6225951eb6b01ec938814f2d5050900db6d2ce8588479c37b684978ac42acc3",
            CANDIDATE_A.read_bytes() == CANDIDATE_B.read_bytes(),
            result.get("status") == "SOURCE_RELATIVE_NONINFERIOR_NO_GO",
            result.get("selected_policy_id") is None,
            result.get("deterministic_replay_passed") is True,
            rubric.get("source_relative_noninferiority_status") == "NO_GO",
            rubric.get("response_position_count") == 23,
            rubric.get("delivered_response_position_count") == 22,
            rubric.get("blocked_response_position_count") == 1,
            rubric.get("delivered_integrity_pass_count") == 21,
            rubric.get("public_holdout_required_observable_count") == 1,
            len(rubric.get("source_action_pass_regressions", [])) == 2,
            codec.get("status") == "PASS_EXACT_BLOCK16_CODEC_AND_BYTE_CLOSURE",
            codec.get("weight_count") == 493961216,
            codec.get("four_bit_payload_bytes") == 246980608,
            codec.get("scale32_metadata_bytes") == 123490304,
            codec.get("exact_bits_per_weight") == 6.0,
            dynamic_execution.get("event_count") == 312,
            dynamic_execution.get("saturation_count") == 0,
            dynamic_execution.get("clipping_count") == 0,
            dynamic_execution.get("reserved_minus_128_produced") is False,
            latency.get("all_deterministic_replays_match") is True,
            campaign_execution.get("model_forward_count") == 410,
            campaign_execution.get("event_count") == 312,
            campaign_execution.get("saturation_count") == 0,
            campaign_execution.get("clipping_count") == 0,
            campaign_execution.get("reserved_minus_128_produced") is False,
            provenance.get("status") == "PASS_REDACTION_AWARE_SEALED_V17_PROVENANCE",
            provenance.get("source_action_pass_regression_count") == 2,
            terminal.get("status") == "SOURCE_RELATIVE_NONINFERIOR_NO_GO",
            terminal.get("failure_taxonomy") == "valid_noninferiority_failure",
            terminal.get("construction_attempts_consumed") == 1,
            terminal.get("quality_campaigns_consumed") == 1,
            terminal.get("fresh_reviewer_status") == "replan_requested",
            terminal.get("selected_policy_id") is None,
            terminal.get("arbitrary_text_multi_turn_demo_implemented") is False,
            "V17 cannot be used as a credible quantized reference" in checkpoint,
        )
    )

    ground_truth_lines = [line[2:] for line in ground_truth.splitlines() if line.startswith("- ")]
    ground_truth_current = all(
        (
            ground_truth.startswith("# Binding Unknowns\n"),
            len(ground_truth_lines) == 5,
            all("unknown" in line.lower() for line in ground_truth_lines),
            "No V18 or other successor contract is frozen" in ground_truth,
            "official V17 namespace is absent" not in ground_truth,
            "u280-host-inventory-recheck-20260807T133413Z.json" in ground_truth,
        )
    )
    u280_current = all(
        (
            sha256(U280) == "34179ffcdbb9772db24fe20ae4f8e10f4734b49c325a890107daaeb457cee1af",
            companion_hash_matches(U280),
            u280.get("observed_at_utc") == "2026-08-07T13:34:13Z",
            u280.get("u280_access") is False,
            not any(u280.get("vendor_tools_in_path", {}).values()),
            u280.get("pci_inventory", {}).get("xilinx_vendor_10ee_functions") == [],
            u280.get("device_nodes", {}).get("xclmgmt") == [],
            u280.get("device_nodes", {}).get("renderD") == [],
        )
    )
    benchmark_gate = all(
        (
            terminal.get("terminal_result_sha256")
            == "61291cfd345d5f8a70d43bf88ebdfd85f6db9a21d87f0ad68ba44bc0979cae3c",
            benchmark.get("product_compression_gate", {}).get("selected_policy_id") is None,
            internal.get("selected_policy_id") is None,
            internal.get("next_candidate_execution_authorized") is False,
            internal.get("authorized_candidate_id") is None,
            internal.get("planner_execution_permitted") is False,
            internal.get("predecessor_fresh_reviewer_adjudication_complete") is True,
            benchmark.get("stage_order", {}).get("stage_1_local_runtime", "").startswith(
                "blocked_no_selected_policy_source_relative_v2_qualified_v17_terminal_no_go"
            ),
            benchmark.get("stage_order", {}).get("stage_2_u280") == "pending_after_stage_1",
            benchmark.get("u280_environment_gate", {}).get("latest_inventory")
            == "research/raw/specification/u280-host-inventory-recheck-20260807T133413Z.json",
        )
    )

    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            v17_terminal,
            benchmark_gate,
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
        TASK,
        RUNNER,
        PROVENANCE_VERIFIER,
        RESULT,
        RUBRIC,
        LATENCY,
        CODEC,
        DYNAMIC,
        CANDIDATE_A,
        CANDIDATE_B,
        U280,
        U280.with_suffix(".sha256"),
        Path(__file__),
    )
    return {
        "schema_version": 1,
        "scope": "non-consuming specification audit; no V17 replay or resume, no RTL edit or execution, no simulation/formal/synthesis/PPA, no chat retarget, no FPGA build or emulation, no board execution, no proprietary download, and no Manager-owned stage mutation",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {
            "current_stage": pipeline.get("current_stage"),
            "manager_owned_state_mutated": False,
        },
        "checklist": checklist,
        "mission_gate": {
            "terminal_v17_verified": v17_terminal,
            "redaction_aware_provenance_verified": provenance.get("status")
            == "PASS_REDACTION_AWARE_SEALED_V17_PROVENANCE",
            "selected_policy_id": internal.get("selected_policy_id"),
            "next_candidate_execution_authorized": internal.get("next_candidate_execution_authorized"),
            "arbitrary_text_multi_turn_demo_implemented": terminal.get(
                "arbitrary_text_multi_turn_demo_implemented"
            ),
            "project_completion_eligible": False,
            "stage_2_u280": benchmark.get("stage_order", {}).get("stage_2_u280"),
        },
        "v17_terminal_measurements": {
            "response_positions": rubric.get("response_position_count"),
            "delivered_responses": rubric.get("delivered_response_position_count"),
            "model_forwards": campaign_execution.get("model_forward_count"),
            "source_action_pass_regressions": rubric.get("source_action_pass_regressions"),
            "delivered_integrity_pass_count": rubric.get("delivered_integrity_pass_count"),
            "public_holdout_required_observable_count": rubric.get(
                "public_holdout_required_observable_count"
            ),
            "exact_bits_per_weight": codec.get("exact_bits_per_weight"),
            "dynamic_event_count": campaign_execution.get("event_count"),
        },
        "u280_local_inventory": {
            "record": str(U280.relative_to(ROOT)),
            "record_sha256": sha256(U280),
            "current_negative_inventory_verified": u280_current,
        },
        "input_bindings": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_specification_stage_terminal_v17.py --output research/raw/specification/specification-stage-audit-terminal-v17-20260807T134500Z.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
