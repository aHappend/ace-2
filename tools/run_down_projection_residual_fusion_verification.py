#!/usr/bin/env python3
"""Run and bind the verification-stage DPRF discriminator and stress suite."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
BASELINE_MECHANISM = "shared_down_projection_residual_fusion_baseline_v1"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
LATEST = ROOT / "evidence" / CONTRACT / "latest"
PRECHECK = LATEST / "PRECHECK.json"
RTL_REVIEW = (
    ROOT
    / "evidence"
    / "review"
    / "rtl_checklist_shared_down_projection_residual_fusion_v1"
    / "decision.json"
)
RAW = ROOT / "verification" / "raw" / "latest"
RAW_ARCHIVE = ROOT / "verification" / "raw" / "archive"
BASELINE_REPLAY = RAW / "dprf_focused_baseline_replay"
CANDIDATE_REPLAY = RAW / "dprf_focused_candidate_replay"
RESULTS = ROOT / "verification" / "RESULTS.json"
PLAN = ROOT / "verification" / "PLAN.md"
ORACLE_MANIFEST = ROOT / "reference" / "ORACLE_MANIFEST.json"
PROPERTY_MANIFEST = ROOT / "formal" / "ACE2_VERIFICATION_PROPERTIES.json"
DECISION = LATEST / "VERIFICATION_DECISION.json"
PUBLIC = ROOT / "research" / "PUBLIC_STATUS.json"
PIPELINE = ROOT / "research" / "PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
FROZEN_BASELINE = {
    "wikitext2": {
        "model.layers.0.post_mlp_residual": 0.43066111261321344,
        "lm_head": 0.9143753483041323,
    },
    "c4_en_512": {
        "model.layers.0.post_mlp_residual": 0.39761529680619884,
        "lm_head": 1.1934673932274116,
    },
}
SOURCE_PATHS = [
    "rtl/ace2_down_projection_residual_fusion_core.sv",
    "tools/ace2_down_projection_residual_fusion_hook.py",
    "tools/ace2_down_projection_residual_fusion_reference.py",
    "tools/ace2_full_model_fixed_point.py",
    "tools/localize_score_to_lm_head.py",
    "tools/gen_down_projection_residual_fusion_vectors.py",
    "tools/gen_down_projection_residual_fusion_metadata.py",
    "tools/run_down_projection_residual_fusion_formal.py",
    "tools/run_down_projection_residual_fusion_verification.py",
    "verification/test_down_projection_residual_fusion.py",
    "verification/tb/ace2_down_projection_residual_fusion_tb.sv",
    "verification/tb/ace2_down_projection_residual_fusion_stage_tb.sv",
    "formal/ace2_down_projection_residual_fusion_formal.sv",
    "formal/ace2_down_projection_residual_fusion_formal.ys",
    "verification/generated/down_projection_residual_fusion_vectors.json",
    "verification/generated/down_projection_residual_fusion_vectors.svh",
    "reference/generated/down_projection_residual_fusion_metadata.json",
    "reference/generated/down_projection_residual_fusion_metadata.bin",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def evidence_utc() -> str:
    value = os.environ.get("ACE2_EVIDENCE_UTC")
    require(value is not None, "ACE2_EVIDENCE_UTC must bind the evidence date")
    require(
        len(value) == 20
        and value[4] == "-"
        and value[7] == "-"
        and value[10] == "T"
        and value.endswith("Z"),
        "ACE2_EVIDENCE_UTC must use YYYY-MM-DDTHH:MM:SSZ",
    )
    require(value <= "2026-08-02T23:59:59Z", "evidence timestamp is after the current date")
    return value


def archive_raw_latest() -> None:
    if not RAW.exists() or not any(RAW.iterdir()):
        RAW.mkdir(parents=True, exist_ok=True)
        return
    digest = hashlib.sha256()
    for path in sorted(item for item in RAW.rglob("*") if item.is_file()):
        digest.update(path.relative_to(RAW).as_posix().encode())
        digest.update(sha256(path).encode())
    destination = RAW_ARCHIVE / f"pre-dprf-{digest.hexdigest()[:16]}"
    RAW_ARCHIVE.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"raw archive already exists: {destination}")
    shutil.move(str(RAW), str(destination))
    RAW.mkdir(parents=True, exist_ok=True)


def run(command_id: str, argv: list[str]) -> dict[str, Any]:
    RAW.mkdir(parents=True, exist_ok=True)
    log = RAW / f"{command_id}.log"
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    duration = time.perf_counter() - started
    log.write_text(
        "\n".join(
            [
                f"command={' '.join(argv)}",
                f"exit_status={completed.returncode}",
                f"wall_seconds={duration:.2f}",
                "",
                completed.stdout,
            ]
        ),
        encoding="utf-8",
    )
    require(completed.returncode == 0, f"command failed: {' '.join(argv)}")
    return {
        "id": command_id,
        "command": argv,
        "exit_status": completed.returncode,
        "wall_seconds": round(duration, 2),
        "raw_log": artifact(log),
    }


def verify_sha256s(result_path: Path) -> None:
    directory = result_path.parent
    sums = directory / "SHA256SUMS"
    require(sums.is_file(), f"missing SHA256SUMS beside {result_path}")
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = directory / relative
        require(path.is_file(), f"missing hashed replay artifact: {path}")
        require(sha256(path) == digest, f"replay artifact hash differs: {path}")


def comparison(
    dataset: str,
    boundary: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_value = float(baseline["comparisons"][dataset][boundary]["relative_l2_error"])
    candidate_value = float(candidate["comparisons"][dataset][boundary]["relative_l2_error"])
    expected = FROZEN_BASELINE[dataset][boundary]
    baseline_matches = math.isclose(baseline_value, expected, rel_tol=0.0, abs_tol=1e-15)
    improves_fresh_baseline = candidate_value < baseline_value
    beats_frozen_threshold = candidate_value < expected
    return {
        "id": f"{dataset}.{boundary}.strict_improvement",
        "baseline_relative_l2": baseline_value,
        "frozen_baseline_relative_l2": expected,
        "baseline_matches_frozen": baseline_matches,
        "candidate_relative_l2": candidate_value,
        "delta_relative_l2": candidate_value - baseline_value,
        "strictly_improves_fresh_baseline": improves_fresh_baseline,
        "candidate_beats_frozen_threshold": beats_frozen_threshold,
        "passed": baseline_matches and improves_fresh_baseline and beats_frozen_threshold,
    }


def check_replays(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    require(baseline["capture_token_limit"] == 128, "baseline token limit differs")
    require(candidate["capture_token_limit"] == 128, "candidate token limit differs")
    require(baseline["diagnostic_rope_mechanism"] == BASELINE_MECHANISM, "baseline mechanism differs")
    require(candidate["diagnostic_rope_mechanism"] == CONTRACT, "candidate mechanism differs")
    require(baseline["input_observations"] == candidate["input_observations"], "focused inputs differ")
    baseline_contract = baseline["down_projection_residual_fusion_contract"]
    candidate_contract = candidate["down_projection_residual_fusion_contract"]
    require(baseline_contract["mode"] == "isolated_baseline_capture_replay", "baseline replay mode differs")
    require(candidate_contract["mode"] == "composed_recurrent_candidate", "candidate replay mode differs")

    dataset_checks: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    failed_conditions: list[str] = []
    for dataset in ("wikitext2", "c4_en_512"):
        baseline_pass = baseline_contract["passes"][dataset]
        candidate_pass = candidate_contract["passes"][dataset]
        require(baseline_pass["ordered_layer_indices"] == list(range(24)), f"{dataset} baseline layer order differs")
        require(candidate_pass["ordered_layer_indices"] == list(range(24)), f"{dataset} candidate layer order differs")
        require(baseline_pass["all_lane_independent_oracle_match"], f"{dataset} isolated replay failed")
        require(candidate_pass["all_lane_independent_oracle_match"], f"{dataset} composed oracle replay failed")
        require(not baseline_pass["sticky_numeric_overflow"], f"{dataset} baseline replay overflowed")
        require(not candidate_pass["sticky_numeric_overflow"], f"{dataset} candidate replay overflowed")
        baseline_layer0 = baseline_pass["layers"][0]
        candidate_layer0 = candidate_pass["layers"][0]
        layer0_inputs_equal = all(
            baseline_layer0[key] == candidate_layer0[key]
            for key in (
                "accumulator_s32_sha256",
                "residual_s8_sha256",
                "baseline_down_projection_s8_sha256",
                "baseline_post_mlp_s8_sha256",
            )
        )
        score_equal = (
            baseline["fixed_boundary_hashes"][dataset]["model.layers.0.score"]
            == candidate["fixed_boundary_hashes"][dataset]["model.layers.0.score"]
        )
        attention_value_equal = (
            baseline["fixed_boundary_hashes"][dataset]["model.layers.0.attention_value"]
            == candidate["fixed_boundary_hashes"][dataset]["model.layers.0.attention_value"]
        )
        later_layers_causal = all(
            row["input_descended_from_prior_fusion"]
            for row in candidate_pass["layers"][1:]
        )
        dataset_checks[dataset] = {
            "layer0_pre_injection_inputs_bit_identical": layer0_inputs_equal,
            "layer0_score_bit_identical": score_equal,
            "layer0_attention_value_bit_identical": attention_value_equal,
            "later_layer_inputs_marked_as_prior_fusion_descendants": later_layers_causal,
            "isolated_all_lane_oracle_match": baseline_pass["all_lane_independent_oracle_match"],
            "composed_all_lane_oracle_match": candidate_pass["all_lane_independent_oracle_match"],
            "baseline_ordered_trace_sha256": baseline_pass["ordered_trace_sha256"],
            "candidate_ordered_trace_sha256": candidate_pass["ordered_trace_sha256"],
        }
        for name, passed in (
            (f"{dataset}.layer0_pre_injection_inputs_bit_identical", layer0_inputs_equal),
            (f"{dataset}.layer0_score_bit_identical", score_equal),
            (f"{dataset}.layer0_attention_value_bit_identical", attention_value_equal),
            (f"{dataset}.later_layer_inputs_causal", later_layers_causal),
        ):
            if not passed:
                failed_conditions.append(name)
        comparisons[dataset] = {}
        for boundary in ("model.layers.0.post_mlp_residual", "lm_head"):
            item = comparison(dataset, boundary, baseline, candidate)
            comparisons[dataset][boundary] = item
            if not item["baseline_matches_frozen"]:
                failed_conditions.append(
                    f"{dataset}.{boundary}.fresh_baseline_replay_match"
                )
            if not item["strictly_improves_fresh_baseline"]:
                failed_conditions.append(
                    f"{dataset}.{boundary}.strict_improvement_over_fresh_baseline"
                )
            if not item["candidate_beats_frozen_threshold"]:
                failed_conditions.append(
                    f"{dataset}.{boundary}.candidate_below_frozen_threshold"
                )
    return {
        "dataset_checks": dataset_checks,
        "comparisons": comparisons,
        "failed_conditions": failed_conditions,
        "passed": not failed_conditions,
    }


def update_public(results: dict[str, Any], decision_record: dict[str, Any]) -> None:
    public = load(PUBLIC)
    checklist = results["checklist"]
    passed = results["status"] == "pass"
    status = (
        "verification_evidence_green_review_pending"
        if passed
        else "bounded_no_go_focused_discriminator_failed"
    )
    latest_decision = (
        "down_projection_residual_fusion_verification_green"
        if passed
        else "down_projection_residual_fusion_bounded_no_go"
    )
    public["latest_decision"] = latest_decision
    public["supported_layer_operator_prefix"] = PREFIX
    public["ordered_supported_layer_operator_prefix"] = PREFIX
    public["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    public["current_mode"] = "ADVANCE"
    public["verification_checklist"] = checklist
    public["required_operator_action"] = "none"
    public["required_manager_action"] = results["required_manager_action"]
    public["routing_status"] = status
    public["stage"] = {
        "current_stage": "verification",
        "completed_prior_stages": ["definition", "architecture", "environment", "rtl"],
        "current_stage_checklist": checklist,
        "current_stage_evidence": [
            "verification/PLAN.md",
            "verification/RESULTS.json",
            decision_record["path"],
        ],
        "current_stage_status": status,
        "stage_closing": passed,
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    latest_verification = {
        "status": status,
        "contract_id": CONTRACT,
        "candidate_id": results["candidate_id"],
        "candidate_rtl_hash": results["candidate_rtl_hash"],
        "checklist": checklist,
        "quality_gate": results["quality_gate"],
        "coverage": results["coverage"],
        "decision": decision_record,
        "results": "verification/RESULTS.json",
        "paired_smoke_run": False,
        "ppa_run": False,
    }
    public["latest_verification_stage"] = latest_verification
    dashboard = public.setdefault("dashboard_fields", {})
    dashboard.update(
        {
            "current_stage": "verification",
            "current_mode": "ADVANCE",
            "latest_decision": latest_decision,
            "candidate_meets_numeric_acceptance": passed,
            "supported_layer_operator_prefix": PREFIX,
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "verification_checklist": checklist,
            "latest_verification_stage": latest_verification,
        }
    )
    mechanism = dashboard.setdefault("candidate_mechanism", {})
    mechanism.update(
        {
            "contract_id": CONTRACT,
            "exact_scale32_dataset_discriminator_status": status,
            "implementation_authorized": False,
            "status": status,
        }
    )
    rtl_candidate = dashboard.setdefault("latest_rtl_candidate", {})
    rtl_candidate.update(
        {
            "contract_id": CONTRACT,
            "candidate_id": results["candidate_id"],
            "candidate_rtl_hash": results["candidate_rtl_hash"],
            "status": status,
            "required_manager_action": results["required_manager_action"],
            "stage_closing": passed,
        }
    )
    for container_name in ("implementation_frontier",):
        container = public.setdefault(container_name, {})
        container.update(
            {
                "current_stage": "verification",
                "current_mode": "ADVANCE",
                "latest_decision": latest_decision,
                "supported_layer_operator_prefix": PREFIX,
                "ordered_supported_layer_operator_prefix": PREFIX,
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                "verification_checklist": checklist,
                "latest_verification_stage": latest_verification,
            }
        )
    public.setdefault("integrity", {})["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def main() -> int:
    now = evidence_utc()
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "verification", "Manager-owned stage is not verification")
    rtl_review = load(RTL_REVIEW)
    require(rtl_review.get("reviewer_status") == "done", "RTL review is not done")
    require(rtl_review.get("stage_closing") is True, "RTL review is not stage-closing")
    require(rtl_review.get("contract_id") == CONTRACT, "RTL review contract differs")
    candidate_id = rtl_review["candidate_id"]
    candidate_hash = rtl_review["candidate_rtl_hash"]
    require(load(PRECHECK)["candidate_id"] == candidate_id, "precheck candidate differs")
    archive_raw_latest()

    commands: list[dict[str, Any]] = []
    commands.append(run("dprf_vector_generation", [sys.executable, "tools/gen_down_projection_residual_fusion_vectors.py"]))
    commands.append(run("dprf_vector_regeneration", [sys.executable, "tools/gen_down_projection_residual_fusion_vectors.py"]))
    commands.append(run("dprf_metadata_check", [sys.executable, "tools/gen_down_projection_residual_fusion_metadata.py", "--check"]))
    commands.append(run("dprf_hook_self_test", [sys.executable, "tools/ace2_down_projection_residual_fusion_hook.py", "--self-test"]))
    commands.append(run("dprf_reference_tests", [sys.executable, "-m", "unittest", "verification.test_down_projection_residual_fusion", "-v"]))
    commands.append(
        run(
            "dprf_rtl_elaboration",
            [
                "iverilog", "-g2012", "-Wall", "-Iverification/generated",
                "-o", "build/ace2_dprf_verification_tb.vvp",
                "rtl/ace2_down_projection_residual_fusion_core.sv",
                "verification/tb/ace2_down_projection_residual_fusion_tb.sv",
            ],
        )
    )
    commands.append(run("dprf_rtl_simulation", ["vvp", "build/ace2_dprf_verification_tb.vvp"]))
    commands.append(
        run(
            "dprf_stage_stress_elaboration",
            [
                "iverilog", "-g2012", "-Wall",
                "-o", "build/ace2_dprf_stage_tb.vvp",
                "rtl/ace2_down_projection_residual_fusion_core.sv",
                "verification/tb/ace2_down_projection_residual_fusion_stage_tb.sv",
            ],
        )
    )
    commands.append(run("dprf_stage_stress_simulation", ["vvp", "build/ace2_dprf_stage_tb.vvp"]))
    commands.append(
        run(
            "dprf_verilator_lint",
            [
                "verilator", "--lint-only", "--language", "1800-2017", "-Wall", "-Wno-fatal",
                "--top-module", "ace2_down_projection_residual_fusion_core",
                "rtl/ace2_down_projection_residual_fusion_core.sv",
            ],
        )
    )
    commands.append(run("dprf_bounded_formal", [sys.executable, "tools/run_down_projection_residual_fusion_formal.py"]))
    commands.append(
        run(
            "dprf_focused_baseline_replay",
            [
                sys.executable,
                "tools/localize_score_to_lm_head.py",
                "--output-dir",
                BASELINE_REPLAY.relative_to(ROOT).as_posix(),
                "--diagnostic-rope-mechanism",
                BASELINE_MECHANISM,
                "--capture-token-limit",
                "128",
            ],
        )
    )
    commands.append(
        run(
            "dprf_focused_candidate_replay",
            [
                sys.executable,
                "tools/localize_score_to_lm_head.py",
                "--output-dir",
                CANDIDATE_REPLAY.relative_to(ROOT).as_posix(),
                "--diagnostic-rope-mechanism",
                CONTRACT,
                "--capture-token-limit",
                "128",
            ],
        )
    )

    original_log = (RAW / "dprf_rtl_simulation.log").read_text(encoding="utf-8")
    stress_log = (RAW / "dprf_stage_stress_simulation.log").read_text(encoding="utf-8")
    formal_log = (RAW / "dprf_bounded_formal.log").read_text(encoding="utf-8")
    require("ACE2_DOWN_PROJECTION_RESIDUAL_FUSION_RTL_PASS" in original_log, "RTL pass token is missing")
    require("ACE2_DPRF_STAGE_STRESS_PASS" in stress_log, "stress pass token is missing")
    require("ACE2_DPRF_MINIMAL_FORMAL_PASS" in formal_log, "formal pass token is missing")

    baseline_path = BASELINE_REPLAY / "results.json"
    candidate_path = CANDIDATE_REPLAY / "results.json"
    verify_sha256s(baseline_path)
    verify_sha256s(candidate_path)
    baseline = load(baseline_path)
    candidate = load(candidate_path)
    require(baseline["generated_at_utc"] <= "2026-08-02T23:59:59Z", "baseline replay timestamp is in the future")
    require(candidate["generated_at_utc"] <= "2026-08-02T23:59:59Z", "candidate replay timestamp is in the future")
    quality_gate = check_replays(baseline, candidate)
    gate_passed = quality_gate["passed"]
    status = "pass" if gate_passed else "bounded_no_go"
    required_manager_action = (
        "independent_verification_review_then_manager_may_advance_to_ppa"
        if gate_passed
        else "manager_rollback_to_architecture_for_structurally_distinct_successor"
    )
    source_hashes = [artifact(ROOT / relative) for relative in SOURCE_PATHS]

    decision = {
        "schema_version": 1,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "status": status,
        "decision": (
            "focused_discriminator_pass_verification_review_pending"
            if gate_passed
            else "bounded_no_go_focused_discriminator_failed"
        ),
        "stage": "verification",
        "stage_closing": gate_passed,
        "candidate_verified": gate_passed,
        "quality_discriminator_complete": True,
        "quality_gate": quality_gate,
        "required_manager_action": required_manager_action,
        "ordered_supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "mode": "ADVANCE",
        "claim_boundary": [
            "Current-stage standalone, stress, bounded formal, isolated all-layer replay, and composed 128-token two-dataset verification only.",
            "No paired smoke, shell admission, SKY130 PPA, prototype, benchmark, signoff, tapeout, or silicon claim.",
        ],
        "baseline_replay": artifact(baseline_path),
        "candidate_replay": artifact(candidate_path),
        "rtl_review": artifact(RTL_REVIEW),
        "precheck": artifact(PRECHECK),
        "source_hashes": source_hashes,
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    decision["integrity"]["canonical_sha256"] = canonical_sha256(decision)
    dump(DECISION, decision)
    decision_record = artifact(DECISION)

    oracle_manifest = {
        "schema_version": 3,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "claim_boundary": "Independent scalar/reference, bit-exact standalone RTL, and all-lane 24-layer isolated/composed verification; no downstream claim.",
        "independent_reference": artifact(ROOT / "tools/ace2_down_projection_residual_fusion_reference.py"),
        "exact_hook": artifact(ROOT / "tools/ace2_down_projection_residual_fusion_hook.py"),
        "model_image_metadata": {
            "manifest": artifact(ROOT / "reference/generated/down_projection_residual_fusion_metadata.json"),
            "binary": artifact(ROOT / "reference/generated/down_projection_residual_fusion_metadata.bin"),
            "quality_metrics_executed": True,
            "record_count": 21504,
        },
        "standalone_vectors": {
            "json": artifact(ROOT / "verification/generated/down_projection_residual_fusion_vectors.json"),
            "svh": artifact(ROOT / "verification/generated/down_projection_residual_fusion_vectors.svh"),
        },
        "focused_replays": {
            "baseline": artifact(baseline_path),
            "candidate": artifact(candidate_path),
            "all_lane_independent_oracle_match": True,
            "layers": 24,
            "datasets": ["wikitext2", "c4_en_512"],
            "capture_token_limit": 128,
        },
    }
    dump(ORACLE_MANIFEST, oracle_manifest)

    property_manifest = {
        "schema_version": 3,
        "generated_at_utc": now,
        "formal_engine_run": True,
        "formal_evidence": artifact(RAW / "dprf_bounded_formal.log"),
        "coverage_limitations": [
            "No multi-clock CDC exists in the standalone candidate.",
            "Numeric overflow is proved unreachable for valid Scale32 descriptors and signed-32/int8 inputs; fail-closed overflow wiring is retained but cannot be stimulated from the legal input domain.",
        ],
        "properties": [
            {"id": "exact_single_round_fusion", "status": "all_lane_oracle_and_bit_exact_rtl", "evidence": "independent scalar reference, 16 boundary RTL vectors, and both all-24-layer 128-token replays"},
            {"id": "reset_clear_backpressure", "status": "exercised_by_4_state_simulation_and_formal", "evidence": "original and stage stress RTL benches plus bounded SAT"},
            {"id": "invalid_descriptor_fail_closed", "status": "exercised_by_simulation", "evidence": "reserved, unnormalized, low-exponent, and high-exponent Scale32 cases across all descriptor roles"},
            {"id": "stall_stability", "status": "exercised_by_4_state_simulation_and_formal", "evidence": "three-cycle output stalls and bounded SAT stability assertions"},
            {"id": "x_z_known_outputs", "status": "exercised_by_4_state_simulation", "evidence": "post-reset, post-stall, post-clear, and mid-flight reset checks"},
            {"id": "workspace_and_latency_bounds", "status": "proved_and_simulated", "evidence": "signed-96/unsigned-64 static proof, 10-cycle valid latency, and quotient bound"},
            {"id": "single_clock_no_internal_cdc", "status": "specified_not_cdc_applicable", "evidence": "candidate exposes only clk_i"},
        ],
    }
    dump(PROPERTY_MANIFEST, property_manifest)

    checklist = {
        "verification.independent-oracle": gate_passed,
        "verification.coverage-stress": True,
        "verification.reproducible-green": True,
    }
    results = {
        "schema_version": 3,
        "generated_at_utc": now,
        "stage": "verification",
        "pipeline_current_stage": "verification",
        "manager_stage_transition_owner": "Manager; this tool does not edit research/PIPELINE_STATE.json.",
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "status": status,
        "stage_closing": gate_passed,
        "candidate_accepted": gate_passed,
        "required_manager_action": required_manager_action,
        "checklist": checklist,
        "checklist_explanation": {
            "verification.independent-oracle": (
                "Exact standalone arithmetic, all-lane isolated replay, composed all-layer hook agreement, pre-injection equality, causal trace ordering, and both required quality improvements pass."
                if gate_passed
                else "Exact arithmetic and all-lane replay pass, but one or more frozen strict quality or causality conditions fail."
            ),
            "verification.coverage-stress": "Boundary vectors, 256 deterministic randomized RTL cases, reset/clear, stalls, invalid descriptors, X/Z, saturation, workspace bounds, single-clock CDC disposition, all 24 layers, and both 128-token datasets pass.",
            "verification.reproducible-green": "Fresh vector/metadata, Python, Icarus, Verilator, bounded formal, and both focused replay commands exited successfully with hash-bound raw artifacts.",
        },
        "commands": commands,
        "coverage": {
            "standalone_boundary_cases": 16,
            "deterministic_randomized_rtl_cases": 256,
            "stage_stress_total_cases": 263,
            "layers": 24,
            "capture_token_limit": 128,
            "datasets": ["wikitext2", "c4_en_512"],
            "stress": [
                "asynchronous_midflight_reset",
                "midflight_clear",
                "three_cycle_output_backpressure",
                "busy_start_rejection",
                "reserved_unnormalized_low_and_high_exponent_descriptors",
                "positive_and_negative_saturation",
                "x_z_known_outputs",
                "signed96_unsigned64_workspace_bounds",
                "single_clock_no_internal_cdc",
            ],
        },
        "quality_gate": quality_gate,
        "decision": decision_record,
        "oracle_manifest": artifact(ORACLE_MANIFEST),
        "property_manifest": artifact(PROPERTY_MANIFEST),
        "source_hashes": source_hashes,
        "claim_boundary": decision["claim_boundary"],
    }
    dump(RESULTS, results)

    plan_text = f"""# ACE-2 down-projection residual-fusion verification plan

This plan is limited to the active `verification` stage and candidate
`{CONTRACT}` at RTL hash `{candidate_hash}`. It does not run or claim PPA,
prototype, benchmark, signoff, tapeout, or silicon evidence.

## Acceptance gates

- Independent oracle: bit-exact standalone arithmetic, all-lane isolated replay
  from baseline captures, composed ordered execution at all 24 layers, exact
  layer-0 pre-injection equality, and frozen quality constraints.
- Coverage stress: deterministic randomized and boundary arithmetic, reset,
  clear, stalls/backpressure, busy-start rejection, invalid Scale32 records,
  saturation, X/Z, workspace/latency bounds, single-clock CDC disposition, and
  both representative 128-token datasets.
- Reproducibility: the native command writes fresh hash-bound logs under
  `verification/raw/latest/` and binds `verification/RESULTS.json`.

## Native command

```sh
ACE2_EVIDENCE_UTC=2026-08-02T00:00:00Z .venv/bin/python tools/run_down_projection_residual_fusion_verification.py
```

## Current result

Status: `{status}`. Stage closing: `{str(gate_passed).lower()}`. Decision SHA-256:
`{decision_record['sha256']}`.
"""
    PLAN.write_text(plan_text, encoding="utf-8")

    checkpoint = f"""# Goal

Close the verification checklist for `{CONTRACT}` without entering downstream stages.

# Current State

The Manager-owned stage is `verification`. Candidate `{candidate_id}` remains bound to RTL hash
`{candidate_hash}`. Fresh standalone, stress, bounded formal, isolated all-layer,
and composed two-dataset verification is complete.

# Result

Status: `{status}`. Stage closing: `{str(gate_passed).lower()}`.
Failed conditions: `{', '.join(quality_gate['failed_conditions']) if quality_gate['failed_conditions'] else 'none'}`.

# Boundaries

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`. The 2.0 mm2 non-SRAM cap and 100 MHz floor are unchanged.
No paired smoke, shell admission, PPA, prototype, benchmark, signoff, tapeout,
or silicon run was performed. Only the Manager may change `current_stage`.
"""
    CHECKPOINT.write_text(checkpoint, encoding="utf-8")
    update_public(results, decision_record)
    print(
        f"ACE2_DPRF_VERIFICATION_RESULT status={status} "
        f"stage_closing={str(gate_passed).lower()} decision_sha256={decision_record['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
