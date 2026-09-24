#!/usr/bin/env python3
"""Read-only specification audit for the post-V20 pinned-model blocker."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"
DIAGNOSIS = ROOT / "research/probes/post-v20-bf16-w4-w4a8-summary-divergence-20260807.json"
V20_REVIEW = ROOT / "research/raw/specification/v20-block16-affine-bf16-quality-no-go-review-replan-20260807T161101Z.json"
V21_PLAN = ROOT / "design/OPTION_B_POST_V20_BLOCK4_AFFINE_FP16_W4A8_V21_PLAN.json"
V21_PLAN_SUM = ROOT / "design/OPTION_B_POST_V20_BLOCK4_AFFINE_FP16_W4A8_V21_PLAN.sha256"
V21_TASK = ROOT / "design/OPTION_B_POST_V20_BLOCK4_AFFINE_FP16_W4A8_V21_ENGINEER_TASK.json"
V21_TASK_SUM = ROOT / "design/OPTION_B_POST_V20_BLOCK4_AFFINE_FP16_W4A8_V21_ENGINEER_TASK.sha256"
V21_RUNNER = ROOT / "tools/run_option_b_post_v20_block4_affine_fp16_w4a8_v21.py"
V21_NAMESPACE = ROOT / "build/stage1-option-b-post-v20-block4-affine-fp16-w4a8-v21"

EXPECTED_DIAGNOSIS_SHA256 = "90f339b70229099740642761f8a2f1b40cadd045ffe33ce49c3ada8d7d24b06a"
EXPECTED_PIPELINE_SHA256 = "571314eb005b62945742313b43815247a538acc5a79ee84643f4ab76748f6171"
EXPECTED_V20_REVIEW_SHA256 = "5690fc8c2a6b6b6b4992c04d08a55d1fd9a1218b9541e16dbb7154297ee2e03e"
EXPECTED_V21_PLAN_SHA256 = "6c7748a4a5787aa09ff8c3a2f132a22d450de325e1e01055dcf580bb19b7615f"
EXPECTED_V21_TASK_SHA256 = "9ba719071e10ee9727848f5a0b380dafbd30ac6cdc549e4a7ab1130c2b1d4573"
REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def companion_matches(path: Path, companion: Path) -> bool:
    fields = companion.read_text(encoding="utf-8").strip().split()
    return len(fields) == 2 and fields[0] == sha256(path) and fields[1] == path.name


def matrix_classes(spec: str) -> list[str]:
    section = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    classes: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"}:
            classes.append(cells[0])
    return classes


def audit() -> dict[str, object]:
    pipeline = load_json(PIPELINE)
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load_json(BENCHMARK)
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    diagnosis = load_json(DIAGNOSIS)
    plan = load_json(V21_PLAN)
    task = load_json(V21_TASK)
    shell = audit_shell_contract()

    decision = diagnosis.get("decision", {})
    guards = diagnosis.get("scope_guards", {})
    bindings = diagnosis.get("bindings", {})
    tree_before = bindings.get("immutable_v20_tree_before", {}) if isinstance(bindings, dict) else {}
    tree_after = bindings.get("immutable_v20_tree_after", {}) if isinstance(bindings, dict) else {}
    compression = benchmark.get("product_compression_gate", {})
    containment = compression.get("authoritative_post_v20_containment", {}) if isinstance(compression, dict) else {}
    v21_draft = compression.get("superseded_v21_output_blind_block4_fp16_diagnostic_draft", {}) if isinstance(compression, dict) else {}
    local = benchmark.get("local_contract", {})
    observed_classes = matrix_classes(spec)
    ground_lines = ground_truth.splitlines()
    unknowns_only = bool(ground_lines and ground_lines[0] == "# Binding Unknowns") and all(
        not line.strip() or line.startswith("- ") for line in ground_lines[1:]
    )

    protocol_fragments = (
        "single-clock digital accelerator subsystem",
        "Asynchronous active-low reset",
        "deassert synchronously before state machines leave reset",
        "earliest read response on clock edge",
        "remain stable until accepted",
        "Completion valid; held until ready",
        "extend either latency without changing data, tags, ordering, or completion",
    )
    missing_protocol = [fragment for fragment in protocol_fragments if fragment not in spec]

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS" if shell.get("status") == "PASS" else "FAIL",
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "parameter_mismatches": shell.get("checks", {}).get("parameter_mismatches"),
            "port_mismatches": shell.get("checks", {}).get("port_mismatches"),
        },
        "spec.clock-reset-protocol": {
            "status": "PASS" if not missing_protocol else "FAIL",
            "missing_fragments": missing_protocol,
        },
        "spec.acceptance-matrix": {
            "status": "PASS" if all(item in observed_classes for item in REQUIRED_CLASSES) else "FAIL",
            "required_classes": list(REQUIRED_CLASSES),
            "observed_classes": observed_classes,
        },
        "spec.benchmark-interface-closure": {
            "status": "PASS"
            if benchmark.get("applies") is False
            and benchmark.get("external_contract") is None
            and bool(benchmark.get("non_benchmark_statement"))
            and isinstance(local, dict)
            and local.get("top_module") == "ace2_shell"
            else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "external_contract": benchmark.get("external_contract"),
            "top_module": local.get("top_module") if isinstance(local, dict) else None,
        },
    }

    containment_checks = {
        "diagnosis_hash_matches": sha256(DIAGNOSIS) == EXPECTED_DIAGNOSIS_SHA256,
        "pipeline_hash_unchanged_since_diagnosis": sha256(PIPELINE) == EXPECTED_PIPELINE_SHA256,
        "pipeline_current_stage_specification": pipeline.get("current_stage") == "specification",
        "pipeline_specification_in_progress": pipeline.get("stages", {}).get("specification", {}).get("status") == "in_progress",
        "selected_policy_id_null": isinstance(compression, dict) and compression.get("selected_policy_id") is None,
        "fresh_v21_successor_not_justified": isinstance(decision, dict) and decision.get("fresh_v21_quantization_successor_justified") is False,
        "mechanism_unresolved": isinstance(decision, dict) and decision.get("mechanism_class") == "UNRESOLVED_PINNED_MODEL_OR_QUANTIZATION_MECHANISM",
        "pinned_model_capability_blocker": isinstance(decision, dict) and decision.get("pinned_model_capability_blocker") is True,
        "v20_campaign_not_replayed": isinstance(guards, dict) and guards.get("v20_campaign_replayed") is False,
        "no_authority_consumed": isinstance(guards, dict) and guards.get("attempt_or_authority_consumed") is False,
        "no_candidate_frozen_or_selected": isinstance(guards, dict) and guards.get("candidate_frozen_or_selected") is False,
        "no_downstream_execution": isinstance(guards, dict) and guards.get("rtl_u280_simulation_formal_synthesis_timing_or_ppa_executed") is False,
        "v20_tree_preserved": tree_before.get("manifest_sha256") == tree_after.get("manifest_sha256") and tree_before.get("files") == tree_after.get("files"),
        "v20_review_hash_matches": sha256(V20_REVIEW) == EXPECTED_V20_REVIEW_SHA256,
        "plan_superseded": plan.get("status") == "SUPERSEDED_NON_AUTHORITATIVE_DO_NOT_EXECUTE" and plan.get("backlog_item_created") is False,
        "task_superseded": task.get("status") == "SUPERSEDED_NON_AUTHORITATIVE_DO_NOT_EXECUTE" and task.get("task_created_in_active_backlog") is False,
        "plan_hash_matches": sha256(V21_PLAN) == EXPECTED_V21_PLAN_SHA256 and companion_matches(V21_PLAN, V21_PLAN_SUM),
        "task_hash_matches": sha256(V21_TASK) == EXPECTED_V21_TASK_SHA256 and companion_matches(V21_TASK, V21_TASK_SUM),
        "benchmark_containment_matches": isinstance(containment, dict)
        and containment.get("diagnosis_sha256") == EXPECTED_DIAGNOSIS_SHA256
        and containment.get("quantizer_cycling_permitted") is False,
        "benchmark_v21_draft_superseded": isinstance(v21_draft, dict)
        and v21_draft.get("status") == "SUPERSEDED_NON_AUTHORITATIVE_DO_NOT_EXECUTE"
        and v21_draft.get("engineer_execution_permitted") is False,
        "v21_runner_absent": not V21_RUNNER.exists(),
        "v21_official_namespace_absent": not V21_NAMESPACE.exists(),
        "ground_truth_contains_only_unknowns": unknowns_only,
        "checkpoint_records_blocker": "pinned-model capability blocker" in checkpoint
        and "No V21 backlog item, runner, official namespace, marker, construction, candidate text, or quality campaign" in checkpoint,
    }

    status = "PASS" if all(
        item["status"] == "PASS" for item in checklist.values()
    ) and all(containment_checks.values()) else "FAIL"

    return {
        "schema_version": 1,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "read-only specification-stage containment audit; no candidate construction, prompt execution, RTL edit or execution, simulation, formal, synthesis, timing, PPA, FPGA build, emulation, board execution, or stage mutation",
        "status": status,
        "checklist": checklist,
        "containment_checks": containment_checks,
        "binding_unknowns": [line[2:] for line in ground_lines if line.startswith("- ")],
        "input_hashes": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (
                PIPELINE,
                SPEC,
                BENCHMARK,
                GROUND_TRUTH,
                CHECKPOINT,
                DIAGNOSIS,
                V20_REVIEW,
                V21_PLAN,
                V21_PLAN_SUM,
                V21_TASK,
                V21_TASK_SUM,
            )
        },
        "claim_boundary": "The specification interface checklist and containment rules pass. This audit makes no RTL correctness, product-quality, synthesis, timing, PPA, hardware-emulation, board-deployment, or mission-completion claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(output.relative_to(ROOT).as_posix())
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
