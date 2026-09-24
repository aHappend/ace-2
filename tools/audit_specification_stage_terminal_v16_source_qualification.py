#!/usr/bin/env python3
"""Read-only specification audit for terminal V16 and source qualification."""

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
SOURCE_QUALIFICATION_CONTRACT = ROOT / "design/QWEN_INSTRUCT_SOURCE_QUALIFICATION_V1.json"
SOURCE_QUALIFICATION_PROBE = ROOT / "research/probes/qwen-instruct-source-qualification-v1-20260807T104500Z.json"
EXACT_BF16_PROBE = ROOT / "research/probes/v16-matrix-pinned-bf16-baseline-20260807T104000Z.json"
V16_REVIEW = ROOT / "research/raw/specification/v16-product-quality-review-done-20260807T102540Z.json"
U280_RECHECK = ROOT / "research/raw/specification/u280-host-inventory-recheck-20260807T103403Z.json"
V16_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v16"
QUALITY_DIR = V16_ROOT / "quality-campaign-0001"
QUALITY_RESULT = QUALITY_DIR / "RESULT.json"
QUALITY_SUMS = QUALITY_DIR / "SHA256SUMS"
RAW_OUTPUTS = QUALITY_DIR / "raw_outputs.jsonl"
TOKEN_LATENCY = QUALITY_DIR / "token_and_latency_results.json"
RUBRIC = QUALITY_DIR / "rubric_results.json"
FROZEN_MATRIX = QUALITY_DIR / "frozen_matrix.json"
QUALITY_MARKER = QUALITY_DIR / "execution_started.json"
FRESH_SUBMISSION = V16_ROOT / "fresh_reviewer_submission.json"
CANDIDATE_A = V16_ROOT / "candidate_a_manifest.json"
CANDIDATE_B = V16_ROOT / "candidate_b_manifest.json"
DYNAMIC_PREFLIGHT = V16_ROOT / "dynamic_preflight.json"
TERMINAL_VERIFIER_LOG = ROOT / "build/task-7c95f6e2a841-v16-verify-result-executed-20260807T101900Z.log"
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


def quality_manifest_matches() -> bool:
    entries: dict[str, str] = {}
    for line in QUALITY_SUMS.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    expected = (FROZEN_MATRIX, QUALITY_MARKER, RAW_OUTPUTS, TOKEN_LATENCY, RUBRIC, QUALITY_RESULT)
    return len(entries) == len(expected) and all(entries.get(path.name) == sha256(path) for path in expected)


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
    source_contract = load(SOURCE_QUALIFICATION_CONTRACT)
    source_probe = load(SOURCE_QUALIFICATION_PROBE)
    exact_probe = load(EXACT_BF16_PROBE)
    review = load(V16_REVIEW)
    u280 = load(U280_RECHECK)
    quality_result = load(QUALITY_RESULT)
    token_latency = load(TOKEN_LATENCY)
    rubric = load(RUBRIC)
    dynamic = load(DYNAMIC_PREFLIGHT)
    submission = load(FRESH_SUBMISSION)
    shell = audit_shell_contract()

    shell_source = (ROOT / "rtl/ace2_shell.sv").read_text(encoding="utf-8")
    start = shell_source.index("module ace2_shell")
    shell_header = shell_source[start : shell_source.index(");", start) + 2]
    signed_public_ports = len(re.findall(r"^\s*(?:input|output)[^\n]*\bsigned\b", shell_header, re.MULTILINE))

    matrix_section = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes = table_classes(matrix_section)
    missing_classes = [item for item in REQUIRED_CLASSES if item not in classes]

    candidate_identity = CANDIDATE_A.read_bytes() == CANDIDATE_B.read_bytes()
    raw_count = sum(1 for line in RAW_OUTPUTS.read_text(encoding="utf-8").splitlines() if line.strip())
    v16_terminal = all(
        (
            V16_ROOT.is_dir(),
            quality_manifest_matches(),
            candidate_identity,
            quality_result.get("status") == "PRODUCT_QUALITY_NO_GO",
            quality_result.get("response_count") == 23,
            quality_result.get("deterministic_replay_passed") is True,
            quality_result.get("selected_policy_id") is None,
            raw_count == 23,
            token_latency.get("response_count") == 23,
            token_latency.get("campaign_activation_execution", {}).get("model_forward_count") == 461,
            rubric.get("hard_check_response_pass_count") == 14,
            rubric.get("visible_single_turn_required_observable_count") == 6,
            rubric.get("visible_multi_turn_required_observable_count") == 2,
            rubric.get("fresh_reviewer_holdout_required_observable_count") == 2,
            dynamic.get("activation_execution", {}).get("named_boundary_count") == 312,
            sha256(QUALITY_RESULT) == "f59f6523b61c9d02077db46b13852fab83b9e2d1de9bb5b2da692a5c5a6a2f93",
            sha256(FRESH_SUBMISSION) == "52bfd880f3b20d8ea020613d49b77554e31366f853d416d83a603123df9b4ac4",
            "ACE2_OPTION_B_V10_TRANSFORMER_V12_LM_HEAD64_V16_RESULT_VERIFIED" in TERMINAL_VERIFIER_LOG.read_text(encoding="utf-8"),
        )
    )

    reviewer_closure = all(
        (
            sha256(V16_REVIEW) == "270744fd9a955be777258146e5cbe5f09a69b0245265348939975bd0d76096cf",
            companion_hash_matches(V16_REVIEW),
            review.get("review", {}).get("status") == "done",
            review.get("review", {}).get("terminal_status") == "PRODUCT_QUALITY_NO_GO",
            review.get("review", {}).get("selected_policy_id") is None,
            review.get("quality_findings", {}).get("reviewer_dimension_means", {}).get("overall") == 1.4457,
            review.get("quality_findings", {}).get("critical_failure") is not None,
            review.get("review", {}).get("replay_resume_repair_or_second_campaign_permitted") is False,
        )
    )

    source_qualification = all(
        (
            companion_hash_matches(SOURCE_QUALIFICATION_CONTRACT),
            sha256(SOURCE_QUALIFICATION_CONTRACT) == "0d6de8a737420a617718354761a6a2e09c6e574967d259c05b06ab9893d72dc0",
            sha256(EXACT_BF16_PROBE) == "92ef5c387d3d037a91c4340467b621a1b66b05a54927a1fa9aba75dd5cb956de",
            exact_probe.get("status") == "SOURCE_BASELINE_HARD_GATE_NO_GO",
            exact_probe.get("rubric", {}).get("hard_check_response_pass_count") == 11,
            exact_probe.get("rubric", {}).get("visible_single_turn_required_observable_count") == 3,
            exact_probe.get("rubric", {}).get("visible_multi_turn_required_observable_count") == 2,
            exact_probe.get("rubric", {}).get("fresh_reviewer_holdout_required_observable_count") == 2,
            sha256(SOURCE_QUALIFICATION_PROBE) == "cd165900f7a1bbf9b86187dad63572002f007e77cdd0bd6c6dffe134a68f2b85",
            source_probe.get("status") == "SOURCE_QUALIFICATION_NO_GO",
            source_probe.get("rubric", {}).get("hard_check_response_pass_count") == 11,
            source_probe.get("rubric", {}).get("visible_single_turn_required_observable_count") == 3,
            source_probe.get("rubric", {}).get("visible_multi_turn_required_observable_count") == 2,
            source_probe.get("rubric", {}).get("public_holdout_required_observable_count") == 2,
            all(source_probe.get("detector_self_test", {}).values()),
            source_contract.get("qualification_gate", {}).get("fail_effect") is not None,
        )
    )

    product_gate = benchmark.get("product_compression_gate", {})
    terminal_v16 = product_gate.get("terminal_v16_contract", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    historical_v16 = internal.get("terminal_alias_safe_v10_transformer_v12_lm_head64_product_chat_v16", {})
    benchmark_closure = all(
        (
            benchmark.get("schema_version") == 17,
            benchmark.get("applies") is False,
            benchmark.get("external_contract") is None,
            benchmark.get("specification_stage_evidence", {}).get("benchmark_interface_closure", {}).get("closure")
            == "explicit_non_benchmark_statement",
            terminal_v16.get("status") == "PRODUCT_QUALITY_NO_GO",
            terminal_v16.get("review_status") == "done",
            terminal_v16.get("selected_policy_id") is None,
            product_gate.get("source_qualification", {}).get("status") == "SOURCE_QUALIFICATION_NO_GO",
            product_gate.get("source_qualification", {}).get("next_successor_authorized") is False,
            historical_v16.get("engineer_task_status") == "terminal_review_done",
            historical_v16.get("namespace_exists") is True,
            internal.get("next_candidate_execution_authorized") is False,
            internal.get("selected_policy_id") is None,
            benchmark.get("stage_order", {}).get("stage_1_local_runtime", "").startswith("blocked_no_selected_policy"),
        )
    )

    spec_flat = re.sub(r"\s+", " ", spec)
    required_spec_fragments = (
        "V16 is now terminal",
        "461 actual model forwards",
        "overall mean of `1.4457/2`",
        "critical unsafe keypad-bypass response",
        "passed only 11/23 hard checks",
        "No structurally distinct successor is frozen",
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

    ground_truth_current = all(
        (
            ground_truth.startswith("# Binding Unknowns\n"),
            "V16 is consumed, immutable, unreplayable, and rejected" in ground_truth,
            "another quantizer-only successor is not evidence-supported" in ground_truth,
            "2026-08-07T10:34:03Z" in ground_truth,
            "Whether sealed V16" not in ground_truth,
        )
    )
    u280_current = all(
        (
            sha256(U280_RECHECK) == "a27b8bcb33e40cde56088ad407aedf7d4b3847d7f27b63aa73adba5d9d1134f8",
            companion_hash_matches(U280_RECHECK),
            u280.get("observed_at_utc") == "2026-08-07T10:34:03Z",
            u280.get("u280_access") is False,
            not any(u280.get("vendor_tools_in_path", {}).values()),
        )
    )

    all_pass = all(item["status"] == "PASS" for item in checklist.values()) and all(
        (
            pipeline.get("current_stage") == "specification",
            v16_terminal,
            reviewer_closure,
            source_qualification,
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
        SOURCE_QUALIFICATION_CONTRACT,
        SOURCE_QUALIFICATION_CONTRACT.with_suffix(".sha256"),
        SOURCE_QUALIFICATION_PROBE,
        EXACT_BF16_PROBE,
        V16_REVIEW,
        V16_REVIEW.with_suffix(".sha256"),
        U280_RECHECK,
        U280_RECHECK.with_suffix(".sha256"),
        QUALITY_RESULT,
        QUALITY_SUMS,
        RAW_OUTPUTS,
        TOKEN_LATENCY,
        RUBRIC,
        FROZEN_MATRIX,
        QUALITY_MARKER,
        FRESH_SUBMISSION,
        CANDIDATE_A,
        CANDIDATE_B,
        DYNAMIC_PREFLIGHT,
        TERMINAL_VERIFIER_LOG,
        Path(__file__),
    )
    return {
        "schema_version": 1,
        "scope": "read-only specification audit; no consumed evidence replay, RTL edit or execution, simulation, formal, synthesis/PPA, chat retarget, FPGA build, hardware emulation, board execution, proprietary download, or stage-state mutation",
        "status": "PASS" if all_pass else "FAIL",
        "pipeline": {"current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "checklist": checklist,
        "mission_gate": {
            "v16_terminal_bundle_verified": v16_terminal,
            "fresh_reviewer_closure_verified": reviewer_closure,
            "source_qualification_no_go_verified": source_qualification,
            "selected_policy_id": internal.get("selected_policy_id"),
            "next_candidate_execution_authorized": internal.get("next_candidate_execution_authorized"),
            "project_completion_eligible": False,
            "stage_2_u280": "pending_after_stage_1",
        },
        "u280_local_inventory": {
            "record": str(U280_RECHECK.relative_to(ROOT)),
            "record_sha256": sha256(U280_RECHECK),
            "current_negative_inventory_verified": u280_current,
        },
        "input_bindings": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_specification_stage_terminal_v16_source_qualification.py --output research/raw/specification/specification-stage-audit-terminal-v16-source-qualification-20260807T105200Z.json",
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
