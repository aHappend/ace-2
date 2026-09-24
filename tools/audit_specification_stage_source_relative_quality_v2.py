#!/usr/bin/env python3
"""Read-only audit of the source-relative V2 specification-stage state."""

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
CONTRACT = ROOT / "design/QWEN_INSTRUCT_SOURCE_RELATIVE_QUALITY_V2.json"
REPAIR = ROOT / "research/probes/qwen-instruct-source-relative-quality-v2-repair1-20260807T111432Z.json"
REVIEWER = ROOT / "research/probes/qwen-instruct-source-relative-quality-v2-reviewer-20260807T112030Z.json"
V16_REVIEW = ROOT / "research/raw/specification/v16-product-quality-review-done-20260807T102540Z.json"
U280 = ROOT / "research/raw/specification/u280-host-inventory-recheck-20260807T103403Z.json"
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


def response_signature(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": item.get("case_id"),
            "turn_index": item.get("turn_index"),
            "delivery_action": item.get("delivery_action"),
            "model_forward_count": item.get("model_forward_count"),
            "generated_token_ids": item.get("generated_token_ids"),
            "decoded_text": item.get("decoded_text"),
        }
        for item in record.get("responses", [])
    ]


def audit() -> dict[str, Any]:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    contract = load(CONTRACT)
    repair = load(REPAIR)
    reviewer = load(REVIEWER)
    v16_review = load(V16_REVIEW)
    u280 = load(U280)
    shell = audit_shell_contract()

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

    rubric = reviewer.get("rubric", {})
    source_relative_closure = all(
        (
            sha256(CONTRACT) == "41b889c048dcb377c29de322028160dde18a7e6f735cccdf684dec653430e03b",
            sha256(REPAIR) == "9058d84c8a0f0a7f483c07486ffc9898c0bb967d03ee8c110042851e84974ecb",
            sha256(REVIEWER) == "fdbba290ef91bd522c0a8e836eeeac7743fad5b5d836084a983b4db1c1cc489e",
            companion_hash_matches(CONTRACT),
            companion_hash_matches(REPAIR),
            companion_hash_matches(REVIEWER),
            contract.get("status") == "FROZEN_NON_CONSUMING_SOURCE_QUALIFICATION",
            contract.get("selection", {}).get("selected_policy_id") is None,
            contract.get("quality_contract_revision", {}).get("source_qualification_authorizes_a_w4a8_attempt") is False,
            contract.get("authority_and_claim_boundary", {}).get("rtl_demo_synthesis_ppa_u280_or_stage_advance_authorized") is False,
            repair.get("status") == "SOURCE_QUALIFIED_RELATIVE_BASELINE",
            reviewer.get("status") == "SOURCE_QUALIFIED_RELATIVE_BASELINE",
            repair.get("response_position_count") == 23,
            reviewer.get("response_position_count") == 23,
            rubric.get("response_position_count") == 23,
            rubric.get("delivered_response_position_count") == 22,
            rubric.get("exact_allowed_response_token_replay_count") == 22,
            rubric.get("blocked_response_position_count") == 1,
            rubric.get("action_pass_count") == 12,
            rubric.get("visible_single_turn_required_observable_count") == 4,
            rubric.get("visible_multi_turn_required_observable_count") == 2,
            rubric.get("public_holdout_required_observable_count") == 2,
            rubric.get("critical_safety_failure_count") == 0,
            response_signature(repair) == response_signature(reviewer),
            all(reviewer.get("guard_self_test", {}).values()),
            all(reviewer.get("detector_self_test", {}).values()),
        )
    )

    v16_terminal = all(
        (
            sha256(V16_REVIEW) == "270744fd9a955be777258146e5cbe5f09a69b0245265348939975bd0d76096cf",
            companion_hash_matches(V16_REVIEW),
            v16_review.get("review", {}).get("status") == "done",
            v16_review.get("review", {}).get("terminal_status") == "PRODUCT_QUALITY_NO_GO",
            v16_review.get("review", {}).get("selected_policy_id") is None,
        )
    )

    product_gate = benchmark.get("product_compression_gate", {})
    source_gate = product_gate.get("source_qualification", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    benchmark_closure = all(
        (
            benchmark.get("schema_version") == 18,
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure")
            == "explicit_non_benchmark_statement",
            source_gate.get("status") == "SOURCE_QUALIFIED_RELATIVE_BASELINE",
            source_gate.get("source_relative_v2_reviewer_result_sha256")
            == "fdbba290ef91bd522c0a8e836eeeac7743fad5b5d836084a983b4db1c1cc489e",
            source_gate.get("next_successor_authorized") is False,
            source_gate.get("candidate_contract_required_before_attempt") is True,
            source_gate.get("arbitrary_prompt_safety_qualified") is False,
            source_gate.get("absolute_product_quality_satisfied") is False,
            product_gate.get("selected_policy_id") is None,
            internal.get("selected_policy_id") is None,
            internal.get("next_candidate_execution_authorized") is False,
            internal.get("source_qualification_status") == "SOURCE_QUALIFIED_RELATIVE_BASELINE",
            benchmark.get("stage_order", {}).get("stage_1_local_runtime", "").startswith(
                "blocked_no_selected_policy_source_relative_v2_qualified"
            ),
        )
    )

    spec_flat = re.sub(r"\s+", " ", spec)
    required_spec_fragments = (
        "23 response positions, 22 exact delivered-source token replays",
        "one pre-generation `BLOCKED_NO_ANSWER`",
        "12 action passes",
        "zero matrix-scored critical safety failures",
        "does not select a W4A8 policy",
        "lexical interlock does not block several semantically equivalent paraphrases",
        "public `ace2_shell` contract remains exact",
        "No external benchmark contract or score applies",
    )
    missing_spec = [item for item in required_spec_fragments if item not in spec_flat]

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS" if shell.get("status") == "PASS" and signed_public_ports == 0 and not missing_spec else "FAIL",
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "signed_public_port_count": signed_public_ports,
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
            "status": "PASS" if benchmark_closure else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "closure": benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure"),
        },
    }

    unknown_lines = [line[2:] for line in ground_truth.splitlines() if line.startswith("- ")]
    ground_truth_current = all(
        (
            ground_truth.startswith("# Binding Unknowns\n"),
            len(unknown_lines) == 4,
            all("unknown" in line.lower() for line in unknown_lines),
            "source-relative V2" in ground_truth,
            "general semantic safety" in ground_truth,
            "operator direction" not in ground_truth,
        )
    )
    u280_current = all(
        (
            sha256(U280) == "a27b8bcb33e40cde56088ad407aedf7d4b3847d7f27b63aa73adba5d9d1134f8",
            companion_hash_matches(U280),
            u280.get("observed_at_utc") == "2026-08-07T10:34:03Z",
            u280.get("u280_access") is False,
            not any(u280.get("vendor_tools_in_path", {}).values()),
        )
    )

    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            source_relative_closure,
            v16_terminal,
            benchmark_closure,
            ground_truth_current,
            u280_current,
        )
    )
    inputs = (
        SPEC,
        BENCHMARK,
        PIPELINE,
        GROUND_TRUTH,
        CONTRACT,
        CONTRACT.with_suffix(".sha256"),
        REPAIR,
        REPAIR.with_suffix(".sha256"),
        REVIEWER,
        REVIEWER.with_suffix(".sha256"),
        V16_REVIEW,
        V16_REVIEW.with_suffix(".sha256"),
        U280,
        U280.with_suffix(".sha256"),
        Path(__file__),
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no attempt consumption, evidence replay, RTL edit or execution, simulation, formal, synthesis/PPA, chat retarget, FPGA build, hardware emulation, board execution, proprietary download, or stage-state mutation",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "source_relative_v2_verified": source_relative_closure,
            "terminal_v16_no_go_preserved": v16_terminal,
            "selected_policy_id": internal.get("selected_policy_id"),
            "next_candidate_execution_authorized": internal.get("next_candidate_execution_authorized"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "u280_local_inventory": {
            "record": str(U280.relative_to(ROOT)),
            "record_sha256": sha256(U280),
            "current_negative_inventory_verified": u280_current,
        },
        "input_bindings": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_specification_stage_source_relative_quality_v2.py --output research/raw/specification/specification-stage-audit-source-relative-quality-v2-20260807T113500Z.json",
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
