#!/usr/bin/env python3
"""Run and bind the verification-stage V-residual focused discriminator."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_v_residual_value_correction_attention_v1"
BASELINE_MECHANISM = "shared_v_residual_value_correction_baseline_v1"
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
    / "rtl_checklist_shared_v_residual_value_correction_attention_v1"
    / "decision.json"
)
HISTORICAL_BASELINE = (
    ROOT
    / "evidence"
    / "shared_qk_residual_cross_term_attention_v1"
    / "focused-baseline-all-layer-20260801-v1"
    / "results.json"
)
RAW = ROOT / "verification" / "raw" / "latest"
BASELINE_REPLAY = RAW / "v_focused_baseline_replay"
CANDIDATE_REPLAY = RAW / "v_focused_candidate_replay"
RESULTS = ROOT / "verification" / "RESULTS.json"
ORACLE_MANIFEST = ROOT / "reference" / "ORACLE_MANIFEST.json"
PROPERTY_MANIFEST = ROOT / "formal" / "ACE2_VERIFICATION_PROPERTIES.json"
DECISION = LATEST / "VERIFICATION_DECISION.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    return hashlib.sha256(
        (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()


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


def archive_replay(path: Path) -> None:
    if not path.exists():
        return
    archive = ROOT / "verification" / "raw" / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    path.rename(archive / f"{path.name}-{time.time_ns()}")


def verify_sha256s(result_path: Path) -> None:
    sha_path = result_path.parent / "SHA256SUMS"
    require(sha_path.is_file(), f"missing SHA256SUMS beside {result_path}")
    expected, separator, name = sha_path.read_text(encoding="utf-8").strip().partition("  ")
    require(separator == "  " and name == "results.json", f"malformed {sha_path}")
    require(expected == sha256(result_path), f"SHA256SUMS differs for {result_path}")


def recompute_comparisons(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in ("wikitext2", "c4_en_512"):
        baseline_metrics = baseline["comparisons"][dataset]
        candidate_metrics = candidate["comparisons"][dataset]
        score_base = baseline_metrics["model.layers.0.score"]["relative_l2_error"]
        score_candidate = candidate_metrics["model.layers.0.score"]["relative_l2_error"]
        value_base = baseline_metrics["model.layers.0.attention_value"]["relative_l2_error"]
        value_candidate = candidate_metrics["model.layers.0.attention_value"]["relative_l2_error"]
        lm_base = baseline_metrics["lm_head"]["relative_l2_error"]
        lm_candidate = candidate_metrics["lm_head"]["relative_l2_error"]
        comparisons[dataset] = {
            "layer0_score_strict_improvement": {
                "id": f"{dataset}.layer0_score_strict_improvement",
                "passed": score_candidate < score_base,
                "baseline_relative_l2": score_base,
                "candidate_relative_l2": score_candidate,
                "delta_relative_l2": score_candidate - score_base,
            },
            "layer0_attention_value_strict_improvement": {
                "id": f"{dataset}.layer0_attention_value_strict_improvement",
                "passed": value_candidate < value_base,
                "baseline_relative_l2": value_base,
                "candidate_relative_l2": value_candidate,
                "delta_relative_l2": value_candidate - value_base,
            },
            "lm_head_strict_improvement": {
                "id": f"{dataset}.lm_head_strict_improvement",
                "passed": lm_candidate < lm_base,
                "baseline_relative_l2": lm_base,
                "candidate_relative_l2": lm_candidate,
                "delta_relative_l2": lm_candidate - lm_base,
            },
        }
    return comparisons


def update_human_state(
    *,
    candidate_id: str,
    candidate_hash: str,
    decision_sha: str,
    comparisons: dict[str, Any],
    gate_passed: bool,
    now: str,
) -> None:
    failures = [
        item["id"]
        for dataset in comparisons.values()
        for item in dataset.values()
        if not item["passed"]
    ]
    status_text = "focused discriminator passed" if gate_passed else "bounded verification no-go"
    mission_path = ROOT / "MISSION.md"
    mission = mission_path.read_text(encoding="utf-8")
    markers = (
        "### Active numerical repair verification no-go\n",
        "### Active numerical repair RTL blocker\n",
        "### Active V-residual verification result\n",
    )
    start = next((marker for marker in markers if marker in mission), None)
    end = "## Verification and implementation contract\n"
    require(start is not None and end in mission, "MISSION active repair section differs")
    before, remainder = mission.split(start, 1)
    _, after = remainder.split(end, 1)
    active = f"""### Active V-residual verification result

Candidate `{candidate_id}` at accepted RTL hash `{candidate_hash}` completed
fresh standalone, stress, and all-24-layer 128-token focused verification. The
result is a {status_text}. The accepted prefix remains through
`layer_0.v_proj`; first unsupported remains `layer_0.rope_q`; `ADVANCE` mode,
the 2.0 mm^2 cap, the 100 MHz floor, and the historical PPA frontier are
unchanged.

The exact decision is bound by
`evidence/shared_v_residual_value_correction_attention_v1/latest/VERIFICATION_DECISION.json`
(SHA-256 `{decision_sha}`). Failed focused conditions: `{', '.join(failures) if failures else 'none'}`.
No paired smoke, shell admission/regression, SKY130 PPA, prototype, benchmark,
signoff, tapeout, or silicon result followed. Manager alone changes the stage.

"""
    mission_path.write_text(before + active + end + after, encoding="utf-8")

    next_action = (
        "Continue verification with the legally subsequent paired/full-shell checks before requesting stage closure."
        if gate_passed
        else "Manager rollback from verification to architecture for a structurally distinct successor; this direction is sealed."
    )
    (ROOT / "CHECKPOINT.md").write_text(
        f"""# Goal

Verify `{CONTRACT}` without entering PPA.

# Current State

The Manager-owned stage is `verification`. Candidate `{candidate_id}` at exact
RTL hash `{candidate_hash}` is a {status_text}. The accepted prefix remains
through `layer_0.v_proj`; first unsupported remains `layer_0.rope_q`; mode
remains `ADVANCE`.

# Verified Done

- Independent scalar/tensor arithmetic, deterministic vectors, Icarus RTL,
  Verilator lint, reset/clear/backpressure/error/overflow/X-Z stress all pass.
- Fresh baseline and candidate traces cover all 24 layers, both frozen datasets,
  and 128 tokens with baseline Q/K/V, RoPE, score, and probability checks bound.
- Focused failed conditions: `{', '.join(failures) if failures else 'none'}`.

# Locked / Not Run

Paired smoke, shell admission/regression, canonical SKY130 PPA, prototype,
benchmark, signoff, tapeout, and silicon were not run.

# Next Required Action

{next_action}
""",
        encoding="utf-8",
    )
    (ROOT / "verification" / "PLAN.md").write_text(
        f"""# ACE-2 V-residual verification-stage plan

This plan is limited to the active `verification` stage and candidate
`{CONTRACT}`. It does not run or claim PPA, prototype, benchmark, signoff,
tapeout, or silicon evidence.

## Acceptance gates

- Independent oracle: exact scalar/tensor V projection residual and complete
  sum-then-convert correction agreement, plus bit-exact RTL vectors.
- Coverage stress: all 24 layers, 14:2 head mapping, rows 1/2/63/64/65,
  randomized arithmetic, reset, mid-flight reset, clear, backpressure,
  noncanonical/reserved signed-4, invalid descriptors, checked overflow,
  explicit four-state X/Z checks, and single-clock CDC disposition.
- Focused quality: at 128 tokens on WikiText-2 and C4-en, layer-0 score,
  layer-0 attention value, and final lm-head relative-L2 all strictly improve.
  Any non-improvement is a bounded no-go before downstream work.
- Reproducibility: the native command below writes fresh logs and hash-bound
  results under `verification/raw/latest/` and `verification/RESULTS.json`.

## Native command

```sh
.venv/bin/python tools/run_v_residual_value_correction_verification.py
```

## Current result

Status: {status_text}. Decision SHA-256: `{decision_sha}`.
""",
        encoding="utf-8",
    )
    dump(
        ROOT / ".argus" / "live-view.json",
        {
            "version": 1,
            "title": "V-residual verification result",
            "reason": status_text,
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "evidence_sha256": decision_sha,
            "generated_at_utc": now,
            "paths": [
                "research/PIPELINE_STATE.json",
                "CHECKPOINT.md",
                "verification/RESULTS.json",
                "evidence/shared_v_residual_value_correction_attention_v1/latest/VERIFICATION_DECISION.json",
                "research/PUBLIC_STATUS.json",
            ],
        },
    )


def regenerate_decision_from_retained_raw_outputs() -> int:
    pipeline = load(ROOT / "research" / "PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "verification", "current stage is not verification")
    precheck = load(PRECHECK)
    review = load(RTL_REVIEW)
    current = load(DECISION)
    historical_baseline = load(HISTORICAL_BASELINE)
    baseline_path = BASELINE_REPLAY / "results.json"
    candidate_path = CANDIDATE_REPLAY / "results.json"
    baseline = load(baseline_path)
    candidate = load(candidate_path)

    candidate_id = precheck["candidate_id"]
    candidate_hash = precheck["candidate_rtl_hash"]
    require(precheck.get("contract_id") == CONTRACT, "precheck contract differs")
    require(review.get("stage_closing") is True, "independent RTL review did not close RTL")
    require(review.get("reviewer_status") == "done", "independent RTL review is incomplete")
    require(review.get("candidate_rtl_hash") == candidate_hash, "RTL review hash differs")
    require(current.get("contract_id") == CONTRACT, "current decision contract differs")
    require(current.get("candidate_id") == candidate_id, "current decision candidate differs")
    require(current.get("candidate_rtl_hash") == candidate_hash, "current decision hash differs")
    require(
        current.get("decision") == "bounded_no_go_focused_discriminator_failed",
        "current decision is not the bounded focused NO_GO",
    )
    require(
        all(value is False for value in current.get("downstream_runs", {}).values()),
        "current decision records prohibited downstream work",
    )
    require(
        sha256(ROOT / "rtl" / "ace2_v_residual_value_correction_core.sv")
        == precheck["candidate_rtl_source"]["sha256"],
        "accepted candidate RTL source changed",
    )

    for path in (HISTORICAL_BASELINE, baseline_path, candidate_path):
        verify_sha256s(path)
    require(baseline.get("capture_token_limit") == 128, "baseline token limit differs")
    require(candidate.get("capture_token_limit") == 128, "candidate token limit differs")
    require(baseline.get("diagnostic_rope_mechanism") == BASELINE_MECHANISM, "baseline mechanism differs")
    require(candidate.get("diagnostic_rope_mechanism") == CONTRACT, "candidate mechanism differs")
    require(baseline.get("input_observations") == candidate.get("input_observations"), "focused inputs differ")
    require(
        baseline.get("input_observations") == historical_baseline.get("input_observations")
        and baseline.get("comparisons") == historical_baseline.get("comparisons"),
        "retained V baseline does not preserve the bound authoritative baseline",
    )
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        contract = result.get("v_residual_contract", {})
        require(contract.get("layer_scope") == "all_24_layers", f"{label} layer scope differs")
        require(contract.get("all_baseline_equality_checks_passed") is True, f"{label} baseline equality failed")
        require(len(contract.get("layers", [])) == 24, f"{label} result lacks 24 layers")

    comparisons = recompute_comparisons(baseline, candidate)
    failed_conditions = [
        item["id"]
        for dataset in comparisons.values()
        for item in dataset.values()
        if not item["passed"]
    ]
    expected_failures = [
        "wikitext2.layer0_score_strict_improvement",
        "c4_en_512.layer0_score_strict_improvement",
        "c4_en_512.lm_head_strict_improvement",
    ]
    require(failed_conditions == expected_failures, "retained focused failure set differs")

    now = utc_now()
    decision = copy.deepcopy(current)
    decision.update(
        {
            "bound_at_utc": now,
            "decision": "bounded_no_go_focused_discriminator_failed",
            "accepted_capability": False,
            "accepted_frontier_changed": False,
            "required_next_action": "manager_rollback_verification_to_architecture_for_structurally_distinct_successor",
            "regeneration": {
                "at_utc": now,
                "mode": "retained_raw_outputs_only",
                "discriminator_rerun": False,
            },
        }
    )
    decision["focused_discriminator"].update(
        {
            "passed": False,
            "historical_authoritative_baseline": artifact(HISTORICAL_BASELINE),
            "fresh_baseline_replay": artifact(baseline_path),
            "fresh_candidate_replay": artifact(candidate_path),
            "input_observations_equal": True,
            "baseline_equality_passed": True,
            "comparisons": comparisons,
            "failed_conditions": failed_conditions,
        }
    )
    decision["integrity"]["canonical_sha256"] = None
    decision["integrity"]["canonical_sha256"] = canonical_sha256(decision)
    dump(DECISION, decision)
    decision_sha = sha256(DECISION)
    update_human_state(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        decision_sha=decision_sha,
        comparisons=comparisons,
        gate_passed=False,
        now=now,
    )
    print(
        "ACE2_V_RESIDUAL_RETAINED_DECISION_REGEN_PASS "
        f"failed_conditions={','.join(failed_conditions)} "
        f"decision_sha256={decision_sha} discriminator_rerun=false"
    )
    return 0


def main() -> int:
    if sys.argv[1:] == ["--regenerate-decision-from-retained-raw"]:
        return regenerate_decision_from_retained_raw_outputs()
    require(not sys.argv[1:], f"unsupported arguments: {' '.join(sys.argv[1:])}")
    pipeline = load(ROOT / "research" / "PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "verification", "current stage is not verification")
    precheck = load(PRECHECK)
    review = load(RTL_REVIEW)
    manifest = load(ROOT / "design" / "RTL_MANIFEST.json")
    require(precheck.get("contract_id") == CONTRACT, "precheck contract differs")
    require(all(precheck.get("checklist", {}).values()), "precheck checklist is incomplete")
    require(review.get("stage_closing") is True, "independent RTL review did not close RTL")
    require(review.get("reviewer_status") == "done", "independent RTL review is incomplete")
    candidate_id = precheck["candidate_id"]
    candidate_hash = precheck["candidate_rtl_hash"]
    require(review.get("candidate_rtl_hash") == candidate_hash, "RTL review hash differs")
    require(manifest.get("candidate_rtl_hash") == candidate_hash, "manifest hash differs")
    require(
        sha256(ROOT / "rtl" / "ace2_v_residual_value_correction_core.sv")
        == precheck["candidate_rtl_source"]["sha256"],
        "accepted candidate RTL source changed",
    )

    commands: list[dict[str, Any]] = []
    commands.append(run("v_vector_generation", [sys.executable, "tools/gen_v_residual_value_correction_vectors.py"]))
    vector_paths = (
        ROOT / "verification/generated/v_residual_value_correction_vectors.json",
        ROOT / "verification/generated/v_residual_value_correction_vectors.svh",
    )
    vector_hashes = [sha256(path) for path in vector_paths]
    commands.append(run("v_vector_regeneration", [sys.executable, "tools/gen_v_residual_value_correction_vectors.py"]))
    require(vector_hashes == [sha256(path) for path in vector_paths], "V residual vectors are nondeterministic")
    commands.append(run("v_metadata_check", [sys.executable, "tools/gen_v_residual_scale32_metadata.py", "--check"]))
    commands.append(
        run(
            "v_reference_and_tensor_tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "verification.test_v_residual_value_correction",
                "verification.test_v_residual_value_correction_full_model",
                "-v",
            ],
        )
    )
    commands.append(
        run(
            "v_rtl_elaboration",
            [
                "iverilog",
                "-g2012",
                "-Wall",
                "-Iverification/generated",
                "-o",
                "build/ace2_v_residual_verification_tb.vvp",
                "rtl/ace2_v_residual_value_correction_core.sv",
                "verification/tb/ace2_v_residual_value_correction_tb.sv",
            ],
        )
    )
    commands.append(run("v_rtl_simulation", ["vvp", "build/ace2_v_residual_verification_tb.vvp"]))
    commands.append(
        run(
            "v_stage_stress_elaboration",
            [
                "iverilog",
                "-g2012",
                "-Wall",
                "-o",
                "build/ace2_v_residual_stage_tb.vvp",
                "rtl/ace2_v_residual_value_correction_core.sv",
                "verification/tb/ace2_v_residual_value_correction_stage_tb.sv",
            ],
        )
    )
    commands.append(run("v_stage_stress_simulation", ["vvp", "build/ace2_v_residual_stage_tb.vvp"]))
    for module in (
        "ace2_v_residual_projection_core",
        "ace2_v_residual_value_correction_core",
    ):
        commands.append(
            run(
                f"v_lint_{module}",
                [
                    "verilator",
                    "--lint-only",
                    "--language",
                    "1800-2017",
                    "-Wall",
                    "-Wno-fatal",
                    "--top-module",
                    module,
                    "rtl/ace2_v_residual_value_correction_core.sv",
                ],
            )
        )

    archive_replay(BASELINE_REPLAY)
    archive_replay(CANDIDATE_REPLAY)
    commands.append(
        run(
            "v_focused_baseline_replay",
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
            "v_focused_candidate_replay",
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

    original_rtl_log = (RAW / "v_rtl_simulation.log").read_text(encoding="utf-8")
    stress_log = (RAW / "v_stage_stress_simulation.log").read_text(encoding="utf-8")
    require("ACE2_V_RESIDUAL_RTL_PASS" in original_rtl_log, "original RTL pass token is missing")
    for token in (
        "ACE2_V_RESIDUAL_STAGE_STRESS_PASS",
        "XZ_OUTPUT_CHECK_PASS phases=post_reset,post_backpressure,post_clear,post_midflight_reset",
        "RESET_MIDFLIGHT_CHECK_PASS modules=projection,correction",
        "ERROR_COVERAGE_PASS cases=invalid_scale,noncanonical_s4,lane_count,checked_overflow",
    ):
        require(token in stress_log, f"stage stress token is missing: {token}")

    historical_baseline = load(HISTORICAL_BASELINE)
    baseline = load(BASELINE_REPLAY / "results.json")
    candidate = load(CANDIDATE_REPLAY / "results.json")
    for path in (HISTORICAL_BASELINE, BASELINE_REPLAY / "results.json", CANDIDATE_REPLAY / "results.json"):
        verify_sha256s(path)
    require(baseline.get("capture_token_limit") == 128, "baseline token limit differs")
    require(candidate.get("capture_token_limit") == 128, "candidate token limit differs")
    require(baseline.get("diagnostic_rope_mechanism") == BASELINE_MECHANISM, "baseline mechanism differs")
    require(candidate.get("diagnostic_rope_mechanism") == CONTRACT, "candidate mechanism differs")
    require(baseline["input_observations"] == candidate["input_observations"], "focused inputs differ")
    require(
        baseline["input_observations"] == historical_baseline["input_observations"]
        and baseline["comparisons"] == historical_baseline["comparisons"],
        "fresh V baseline does not preserve the bound authoritative baseline",
    )
    for result in (baseline, candidate):
        contract = result.get("v_residual_contract", {})
        require(contract.get("layer_scope") == "all_24_layers", "focused layer scope differs")
        require(contract.get("all_baseline_equality_checks_passed") is True, "baseline equality failed")
        require(len(contract.get("layers", [])) == 24, "focused result lacks 24 layers")

    comparisons = recompute_comparisons(baseline, candidate)
    gate_passed = all(
        item["passed"] for dataset in comparisons.values() for item in dataset.values()
    )

    now = utc_now()
    decision_name = "focused_discriminator_pass" if gate_passed else "bounded_no_go_focused_discriminator_failed"
    failed_conditions = [
        item["id"]
        for dataset in comparisons.values()
        for item in dataset.values()
        if not item["passed"]
    ]
    decision = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "bound_at_utc": now,
        "manager_owned_stage": "verification",
        "decision": decision_name,
        "accepted_capability": gate_passed,
        "accepted_frontier_changed": False,
        "stage_closing": False,
        "standalone_verification": {
            "passed": True,
            "commands": commands,
            "precheck": artifact(PRECHECK),
            "independent_rtl_review": artifact(RTL_REVIEW),
        },
        "focused_discriminator": {
            "run": True,
            "passed": gate_passed,
            "scope": "all_24_layers_128_tokens_two_frozen_datasets",
            "historical_authoritative_baseline": artifact(HISTORICAL_BASELINE),
            "fresh_baseline_replay": artifact(BASELINE_REPLAY / "results.json"),
            "fresh_candidate_replay": artifact(CANDIDATE_REPLAY / "results.json"),
            "input_observations_equal": True,
            "baseline_equality_passed": True,
            "comparisons": comparisons,
            "failed_conditions": failed_conditions,
        },
        "downstream_runs": {
            "paired_smoke": False,
            "full_shell_regression": False,
            "canonical_sky130_ppa": False,
            "prototype": False,
            "benchmark": False,
            "signoff": False,
        },
        "preserved_frontier": {
            "mode": "ADVANCE",
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "cells": 62199,
            "non_sram_area_mm2": 0.6108746272,
            "setup_slack_ns_at_100mhz": 0.1502,
            "area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "candidate_ppa_run": False,
        },
        "required_next_action": (
            "continue_verification_with_paired_and_full_shell_checks"
            if gate_passed
            else "manager_rollback_verification_to_architecture_for_structurally_distinct_successor"
        ),
        "claim_boundary": "Current-stage focused verification only; no PPA, benchmark acceptance, signoff, tapeout, or silicon claim.",
        "integrity": {"canonical_sha256": None},
    }
    decision["integrity"]["canonical_sha256"] = canonical_sha256(decision)
    dump(DECISION, decision)
    decision_sha = sha256(DECISION)

    generator = ROOT / "tools/gen_v_residual_value_correction_vectors.py"
    reference = ROOT / "tools/ace2_v_residual_value_correction_reference.py"
    oracle = {
        "schema_version": 1,
        "oracles": [
            {
                "case_count": 10,
                "numeric_acceptance": "bit_exact_fixed_point_vector_match",
                "generator": generator.relative_to(ROOT).as_posix(),
                "generator_sha256": sha256(generator),
                "reference": reference.relative_to(ROOT).as_posix(),
                "reference_sha256": sha256(reference),
                "vector_json": vector_paths[0].relative_to(ROOT).as_posix(),
                "vector_json_sha256": sha256(vector_paths[0]),
                "vector_svh": vector_paths[1].relative_to(ROOT).as_posix(),
                "vector_svh_sha256": sha256(vector_paths[1]),
            }
        ],
        "verification_evidence": {
            "generated_at_utc": now,
            "contract_id": CONTRACT,
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "claim_boundary": "Independent scalar/tensor/RTL and focused-quality verification only; no downstream claim.",
            "runner": artifact(ROOT / "tools/ace2_full_model_fixed_point.py"),
            "trace": artifact(ROOT / "tools/localize_score_to_lm_head.py"),
            "hook": artifact(ROOT / "reference/v_residual_value_correction_full_model_hook.json"),
            "metadata": artifact(ROOT / "reference/generated/v_residual_scale32_metadata.json"),
            "baseline": artifact(BASELINE_REPLAY / "results.json"),
            "candidate": artifact(CANDIDATE_REPLAY / "results.json"),
            "baseline_equality_passed_all_24_layers": True,
            "focused_comparisons": comparisons,
            "quality_gate_passed": gate_passed,
        },
    }
    dump(ORACLE_MANIFEST, oracle)
    properties = {
        "schema_version": 2,
        "generated_at_utc": now,
        "formal_engine_run": False,
        "reason_formal_engine_not_run": "The candidate has one clock; closure used independent executable references and four-state RTL simulation. No formal proof is claimed.",
        "properties": [
            {"id": "authoritative_baseline_value_carry", "status": "exercised_by_simulation", "evidence": "scalar/tensor tests and all-layer baseline equality checks"},
            {"id": "sum_then_convert_once", "status": "exercised_by_simulation", "evidence": "verification.test_v_residual_value_correction_full_model"},
            {"id": "reset_clear_backpressure", "status": "exercised_by_4_state_simulation", "evidence": "RESET_MIDFLIGHT_CHECK_PASS and post-backpressure/post-clear checks"},
            {"id": "reserved_noncanonical_descriptor_errors", "status": "exercised_by_simulation", "evidence": "original and stage stress RTL testbenches"},
            {"id": "numeric_overflow_fail_closed", "status": "exercised_by_simulation", "evidence": "checked corrected-accumulator overflow case"},
            {"id": "x_z_known_outputs", "status": "exercised_by_4_state_simulation", "evidence": "XZ_OUTPUT_CHECK_PASS phases=post_reset,post_backpressure,post_clear,post_midflight_reset"},
            {"id": "single_clock_no_internal_cdc", "status": "specified_not_cdc_applicable", "evidence": "Both candidate modules expose only clk_i"},
        ],
    }
    dump(PROPERTY_MANIFEST, properties)

    source_paths = [
        ROOT / "rtl/ace2_v_residual_value_correction_core.sv",
        ROOT / "tools/ace2_v_residual_value_correction_reference.py",
        ROOT / "tools/ace2_full_model_fixed_point.py",
        ROOT / "tools/localize_score_to_lm_head.py",
        Path(__file__).resolve(),
        ROOT / "tools/bind_v_residual_verification_state.py",
        ROOT / "verification/test_v_residual_value_correction.py",
        ROOT / "verification/test_v_residual_value_correction_full_model.py",
        ROOT / "verification/tb/ace2_v_residual_value_correction_tb.sv",
        ROOT / "verification/tb/ace2_v_residual_value_correction_stage_tb.sv",
        ROOT / "reference/generated/v_residual_scale32_metadata.json",
        ROOT / "reference/v_residual_value_correction_full_model_hook.json",
    ]
    checklist = {
        "verification.independent-oracle": gate_passed,
        "verification.coverage-stress": True,
        "verification.reproducible-green": True,
    }
    results = {
        "schema_version": 2,
        "generated_at_utc": now,
        "stage": "verification",
        "pipeline_current_stage": "verification",
        "status": "focused_pass_pending_remaining_verification" if gate_passed else "bounded_no_go",
        "stage_closing": False,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "candidate_accepted": gate_passed,
        "commands": commands,
        "oracle_manifest": artifact(ORACLE_MANIFEST),
        "property_manifest": artifact(PROPERTY_MANIFEST),
        "source_hashes": [artifact(path) for path in source_paths],
        "coverage": {
            "full_model_layers": 24,
            "focused_datasets": ["wikitext2", "c4_en_512"],
            "capture_token_limit": 128,
            "baseline_equality_boundaries": candidate["v_residual_contract"]["baseline_equality_boundaries"],
            "stress": [
                "reset_and_midflight_reset_projection_and_correction",
                "clear",
                "output_backpressure",
                "reserved_and_noncanonical_signed4",
                "invalid_scale_and_lane_count",
                "positive_and_negative_clamps",
                "checked_corrected_accumulator_overflow",
                "x_z_known_outputs_post_reset_backpressure_clear_midflight_reset",
                "deterministic_randomized_scalar_tensor_cases",
                "row_lengths_1_2_63_64_65",
                "all_24_layers_and_14_to_2_head_mapping",
                "single_clock_no_internal_cdc",
            ],
        },
        "quality_gate": {
            "passed": gate_passed,
            "comparisons": comparisons,
            "failed_conditions": failed_conditions,
            "decision": artifact(DECISION),
        },
        "checklist": checklist,
        "checklist_explanation": {
            "verification.independent-oracle": (
                "Exact arithmetic, baseline equality, and all focused quality constraints pass."
                if gate_passed
                else "Exact arithmetic and baseline equality pass, but one or more oracle-owned focused quality constraints fail."
            ),
            "verification.coverage-stress": "Standalone and all-layer stress coverage passed, including explicit four-state X/Z checks and single-clock CDC disposition.",
            "verification.reproducible-green": "Fresh vector, metadata, Python, Icarus, Verilator, and all-layer trace commands exited successfully with raw artifacts.",
        },
        "claim_boundary": [
            "Current-stage focused verification only.",
            "No paired smoke, shell admission, PPA, prototype, benchmark, signoff, tapeout, or silicon claim.",
        ],
        "manager_stage_transition_owner": "Manager; this tool does not edit research/PIPELINE_STATE.json.",
    }
    dump(RESULTS, results)

    manifest["stage"] = "verification"
    manifest["current_stage"] = "verification"
    manifest["candidate_status"] = (
        "focused_discriminator_pass_pending_remaining_verification"
        if gate_passed
        else "authorization_consumed_rejected_focused_discriminator"
    )
    manifest["candidate_meets_numeric_acceptance"] = gate_passed
    manifest["candidate_verification_complete"] = False
    manifest["candidate_requires_fresh_sky130_ppa"] = gate_passed
    manifest["candidate_verification_binding"] = {
        "status": results["status"],
        "evidence": artifact(DECISION),
        "focused_discriminator_passed": gate_passed,
        "paired_smoke_run": False,
        "stage_closing": False,
    }
    manifest.setdefault("latest_evidence", {})["v_residual_value_correction_verification"] = artifact(DECISION)
    dump(ROOT / "design/RTL_MANIFEST.json", manifest)

    status = load(ROOT / "research/PUBLIC_STATUS.json")
    latest_decision = (
        "v_residual_focused_discriminator_pass_pending_remaining_verification"
        if gate_passed
        else "v_residual_focused_discriminator_bounded_no_go"
    )
    required_manager_action = (
        "none_continue_current_verification"
        if gate_passed
        else "rollback_verification_to_architecture_for_structurally_distinct_successor"
    )
    status.update(
        {
            "current_mode": "ADVANCE",
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "latest_decision": latest_decision,
            "latest_ppa_frontier_status": "historical_frontier_preserved_no_candidate_ppa",
            "selected_replacement_contract": {
                "contract_id": CONTRACT,
                "candidate_id": candidate_id,
                "candidate_rtl_hash": candidate_hash,
                "status": results["status"],
                "implementation_authorized": False,
                "operator_approval_consumed": True,
                "result_binding": DECISION.relative_to(ROOT).as_posix(),
                "result_binding_sha256": decision_sha,
            },
            "stage": {
                "completed_prior_stages": ["definition", "architecture", "environment", "rtl"],
                "current_stage": "verification",
                "current_stage_status": results["status"],
                "current_stage_checklist": checklist,
                "current_stage_evidence": [
                    "verification/RESULTS.json",
                    DECISION.relative_to(ROOT).as_posix(),
                    (BASELINE_REPLAY / "results.json").relative_to(ROOT).as_posix(),
                    (CANDIDATE_REPLAY / "results.json").relative_to(ROOT).as_posix(),
                ],
            },
            "blockers": (
                []
                if gate_passed
                else [
                    {
                        "id": "v_residual_focused_quality_no_go",
                        "stage": "verification",
                        "status": "active",
                        "reason": f"Focused failed conditions: {', '.join(failed_conditions)}",
                        "required_resolution": "Manager rollback to architecture for a structurally distinct successor.",
                        "evidence": DECISION.relative_to(ROOT).as_posix(),
                    }
                ]
            ),
            "latest_verification_stage": {
                "status": results["status"],
                "checklist": checklist,
                "results": "verification/RESULTS.json",
                "decision": artifact(DECISION),
            },
        }
    )
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status.get(name)
        if not isinstance(container, dict):
            continue
        container.update(
            {
                "current_stage": "verification",
                "supported_layer_operator_prefix": PREFIX,
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                "current_mode": "ADVANCE",
                "latest_decision": latest_decision,
                "required_manager_action": required_manager_action,
                "required_operator_action": "none",
                "latest_ppa_frontier_status": "historical_frontier_preserved_no_candidate_ppa",
                "verification_checklist": checklist,
            }
        )
    status["routing_status"] = required_manager_action
    status["routing_authorized"] = "manager_only"
    dump(ROOT / "research/PUBLIC_STATUS.json", status)

    update_human_state(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        decision_sha=decision_sha,
        comparisons=comparisons,
        gate_passed=gate_passed,
        now=now,
    )
    from bind_v_residual_verification_state import bind_existing_state

    bind_existing_state()
    print(
        "ACE2_V_RESIDUAL_VERIFICATION "
        f"status={results['status']} candidate={candidate_id} "
        f"failed_conditions={','.join(failed_conditions) if failed_conditions else 'none'} "
        f"decision_sha256={decision_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
