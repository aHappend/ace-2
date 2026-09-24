#!/usr/bin/env python3
"""Audit the current marker-free S5 specification closure without execution."""

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
PIPELINE_SHA = ROOT / "research/PIPELINE_STATE.sha256"

S5_ID = "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5-package-v1"
S5_PACKAGE_IDENTITY = "0487a96540d733eeee88bbef1342b4af5f99204ec334d56c50092be6a5c28a02"
S5_TREE = "54602351773e85fc28e0347aacbee988cc787157306f01913bd74ff190f79a0a"
S5_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S5_EXECUTION_PACKAGE_CONTRACT.json"
S5_PACKAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s5/package-audit.json"
S5_STAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s5/stage-closure-audit.json"
S5_OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5"
S5_MANIFEST = S5_OFFLINE / "s5-execution-package-manifest.json"
S5_SELF_TEST = S5_OFFLINE / "s5-execution-package-self-test.json"
S5_L2 = S5_OFFLINE / "execution-package-l2-acceptance.json"
S5_REVIEW = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-postreview-fresh-review-20260808.json"
S5_AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-attempt-operator-authority.json"
S5_SUBMISSION_ROOT = ROOT / "build/bf16-full-finetune-successor-s5"
S5_RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5"

REQUIRED_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    if not companion.is_file():
        return False
    return companion.read_text(encoding="ascii").strip() == f"{sha256(path)}  {path.name}"


def all_boolean_checks_pass(record: dict[str, object]) -> bool:
    checks = record.get("checks", {})
    return isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())


def table_classes(spec: str) -> list[str]:
    marker = "### Mission-specific acceptance matrix"
    section = spec.split(marker, 1)[1] if marker in spec else ""
    classes: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        first = line.strip().strip("|").split("|", 1)[0].strip().strip("`")
        if first not in {"Class", "---"} and first and not set(first) <= {"-", ":"}:
            classes.append(first)
    return classes


def audit() -> dict[str, object]:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load_json(BENCHMARK)
    pipeline = load_json(PIPELINE)
    shell = audit_shell_contract()
    contract = load_json(S5_CONTRACT)
    package_audit = load_json(S5_PACKAGE_AUDIT)
    stage_audit = load_json(S5_STAGE_AUDIT)
    manifest = load_json(S5_MANIFEST)
    self_test = load_json(S5_SELF_TEST)
    l2 = load_json(S5_L2)
    review = load_json(S5_REVIEW)

    artifacts = (
        S5_CONTRACT,
        S5_PACKAGE_AUDIT,
        S5_STAGE_AUDIT,
        S5_MANIFEST,
        S5_SELF_TEST,
        S5_L2,
        S5_REVIEW,
    )
    artifact_checks = {
        path.relative_to(ROOT).as_posix(): companion_matches(path) for path in artifacts
    }

    state_absent = {
        "authority": not S5_AUTHORITY.exists(),
        "submission_intent": not (S5_SUBMISSION_ROOT / "backend-submission-intent.json").exists(),
        "submission_result": not (S5_SUBMISSION_ROOT / "backend-submission-result.json").exists(),
        "raw_submission_output": not (S5_SUBMISSION_ROOT / "backend-submission.raw.txt").exists(),
        "official_namespace": not S5_RUN_ROOT.exists(),
    }

    s5_record = benchmark.get("active_marker_free_s5_successor", {})
    spec_evidence = benchmark.get("specification_stage_evidence", {})
    spec_s5 = spec_evidence.get("active_marker_free_s5_successor", {}) if isinstance(spec_evidence, dict) else {}

    s5_ok = bool(
        all(artifact_checks.values())
        and contract.get("contract_id") == S5_ID
        and contract.get("attempt_authority", {}).get("granted") is False
        and contract.get("attempt_authority", {}).get("attempt_marker_creation_authorized") is False
        and package_audit.get("status") == "PASS_MARKER_FREE_FULL_TRAINING_SUCCESSOR_S5_PACKAGE"
        and package_audit.get("check_count") == 92
        and package_audit.get("failed_checks") == []
        and all_boolean_checks_pass(package_audit)
        and package_audit.get("package_identity_sha256") == S5_PACKAGE_IDENTITY
        and package_audit.get("runner", {}).get("execution_tree_sha256") == S5_TREE
        and manifest.get("tree_sha256") == S5_TREE
        and self_test.get("status") == "PASS"
        and self_test.get("check_count") == 40
        and all_boolean_checks_pass(self_test)
        and self_test.get("execution_tree_sha256") == S5_TREE
        and l2.get("accepted") is True
        and l2.get("status") == "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S5_PACKAGE_NO_ATTEMPT"
        and l2.get("reviewer_status") == "done"
        and l2.get("package_identity_sha256") == S5_PACKAGE_IDENTITY
        and l2.get("execution_tree_sha256") == S5_TREE
        and l2.get("contract_sha256") == sha256(S5_CONTRACT)
        and stage_audit.get("status") == "PASS_FINAL_POSTREVIEW_MARKER_FREE_S5_TRANSITIVE_STAGE_CLOSURE"
        and stage_audit.get("closure_file_count") == 69
        and stage_audit.get("omission_regression_count") == 69
        and len(stage_audit.get("runtime_sidecar_omission_rejections", {})) == 4
        and stage_audit.get("relocated_readiness", {}).get("status")
        == "PASS_POSTREVIEW_RELOCATED_AUTHORITY_READY_NO_AUTHORITY"
        and all_boolean_checks_pass(stage_audit)
        and review.get("mission_id") == "f68cdf8ac2b5"
        and review.get("review", {}).get("status") == "done"
        and review.get("state", {}).get("attempt_authority_exists") is False
        and review.get("state", {}).get("submission_intent_exists") is False
        and review.get("state", {}).get("official_namespace_exists") is False
        and review.get("state", {}).get("backend_experiment_or_job_exists") is False
        and review.get("state", {}).get("model_execution") is False
        and review.get("state", {}).get("evaluator_execution") is False
        and review.get("state", {}).get("quality_conclusion") is None
        and all(state_absent.values())
        and isinstance(s5_record, dict)
        and s5_record.get("package_id") == S5_ID
        and s5_record.get("package_identity_sha256") == S5_PACKAGE_IDENTITY
        and s5_record.get("execution_tree_sha256") == S5_TREE
        and s5_record.get("attempt_authority_granted") is False
        and s5_record.get("submission_intent_exists") is False
        and s5_record.get("model_quality_conclusion") is None
        and isinstance(spec_s5, dict)
        and spec_s5.get("package_identity_sha256") == S5_PACKAGE_IDENTITY
        and spec_s5.get("execution_tree_sha256") == S5_TREE
        and spec_s5.get("fresh_reviewer_status") == "done"
    )

    observed_classes = table_classes(spec)
    missing_classes = [name for name in REQUIRED_CLASSES if name not in observed_classes]
    behavior_ok = bool(
        shell.get("status") == "PASS"
        and "### Public `ace2_shell` parameter and port contract" in spec
        and "The live runtime instantiates the following complete public parameter set." in spec
        and "there are no `signed` public port declarations" in spec
        and S5_ID in spec
    )
    protocol_ok = all(
        fragment in spec
        for fragment in (
            "Reset may assert asynchronously",
            "deassert synchronously before state machines leave reset",
            "held until ready",
            "no finite completion-latency guarantee",
            "Any latency of one or more cycles is legal",
        )
    )
    benchmark_ok = bool(
        benchmark.get("stage") == "specification"
        and benchmark.get("applies") is False
        and benchmark.get("external_contract") is None
        and benchmark.get("contract_status")
        == "s3_s4_terminal_s5_marker_free_accepted_authority_absent_selected_policy_null_downstream_closed"
        and s5_ok
    )
    pipeline_ok = bool(
        pipeline.get("current_stage") == "specification"
        and PIPELINE_SHA.read_text(encoding="ascii").strip()
        == f"{sha256(PIPELINE)}  {PIPELINE.name}"
    )

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS" if behavior_ok else "FAIL",
            "shell_contract_status": shell.get("status"),
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
        },
        "spec.clock-reset-protocol": {
            "status": "PASS" if protocol_ok else "FAIL",
            "clock_domains": 1,
        },
        "spec.acceptance-matrix": {
            "status": "PASS" if not missing_classes else "FAIL",
            "observed_classes": observed_classes,
            "missing_classes": missing_classes,
        },
        "spec.benchmark-interface-closure": {
            "status": "PASS" if benchmark_ok else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "s5_marker_free_package_verified": s5_ok,
        },
    }
    checklist_pass = all(item["status"] == "PASS" for item in checklist.values())
    status = "PASS" if checklist_pass and pipeline_ok else "FAIL"

    return {
        "schema_version": 1,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "read-only specification-stage S5 audit; no attempt authority, submission, model/evaluator execution, RTL simulation, formal, synthesis, PPA, FPGA build, emulation, or board execution",
        "status": status,
        "checklist": checklist,
        "pipeline_integrity": {
            "status": "PASS" if pipeline_ok else "FAIL",
            "current_stage": pipeline.get("current_stage"),
            "manager_owned_state_mutated": False,
        },
        "s5": {
            "status": "PASS" if s5_ok else "FAIL",
            "package_id": S5_ID,
            "package_identity_sha256": S5_PACKAGE_IDENTITY,
            "execution_tree_sha256": S5_TREE,
            "artifact_companions": artifact_checks,
            "consuming_state_absent": state_absent,
            "attempt_authority_granted": False,
            "planner_execution_permitted": False,
            "next_required_event": "fresh explicit operator authority for exactly one S5 lifecycle",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="ascii"
    )
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
