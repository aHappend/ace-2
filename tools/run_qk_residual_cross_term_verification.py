#!/usr/bin/env python3
"""Run and bind the focused verification no-go for the Q/K residual successor."""

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
CONTRACT = "shared_qk_residual_cross_term_attention_v1"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
LATEST = ROOT / "evidence" / CONTRACT / "latest"
PRECHECK = LATEST / "PRECHECK.json"
NO_GO = LATEST / "VERIFICATION_NO_GO.json"
BASELINE = (
    ROOT
    / "evidence"
    / CONTRACT
    / "focused-baseline-all-layer-20260801-v1"
    / "results.json"
)
CANDIDATE = (
    ROOT
    / "evidence"
    / CONTRACT
    / "focused-candidate-all-layer-20260801-v1"
    / "results.json"
)
RAW = ROOT / "verification" / "raw" / "latest"
RESULTS = ROOT / "verification" / "RESULTS.json"
ORACLE_MANIFEST = ROOT / "reference" / "ORACLE_MANIFEST.json"
PROPERTY_MANIFEST = ROOT / "formal" / "ACE2_VERIFICATION_PROPERTIES.json"


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
    destination = archive / f"{path.name}-{time.time_ns()}"
    path.rename(destination)


def verify_sha256s(result_path: Path) -> None:
    sha_path = result_path.parent / "SHA256SUMS"
    require(sha_path.is_file(), f"missing SHA256SUMS beside {result_path}")
    expected, separator, name = sha_path.read_text(encoding="utf-8").strip().partition("  ")
    require(separator == "  " and name == "results.json", f"malformed {sha_path}")
    require(expected == sha256(result_path), f"SHA256SUMS differs for {result_path}")


def update_human_state(
    candidate_id: str,
    candidate_hash: str,
    no_go_sha: str,
    comparisons: dict[str, Any],
    now: str,
) -> None:
    wiki = comparisons["wikitext2"]["lm_head"]
    c4 = comparisons["c4_en_512"]["lm_head"]
    mission_path = ROOT / "MISSION.md"
    mission = mission_path.read_text(encoding="utf-8")
    start = next(
        (
            marker
            for marker in (
                "### Active numerical repair verification no-go\n",
                "### Active numerical repair RTL blocker\n",
            )
            if marker in mission
        ),
        None,
    )
    end = "## Verification and implementation contract\n"
    require(start is not None and end in mission, "MISSION active repair section differs")
    before, remainder = mission.split(start, 1)
    _, after = remainder.split(end, 1)
    active = f"""### Active numerical repair verification no-go

The accepted prefix remains through `layer_0.v_proj`; the first unsupported
operator remains `layer_0.rope_q`. `ADVANCE` mode, the 1.05x quality limit,
the 2.0 mm^2 non-SRAM cap, the 100 MHz floor, and the historical accepted PPA
frontier remain unchanged.

Candidate `{candidate_id}` at ordered RTL hash `{candidate_hash}` passes the
standalone integer reference, deterministic vectors, bit-exact RTL simulation,
warning-free candidate lint, and all-24-layer baseline-equality checks. The
frozen 128-token focused discriminator nevertheless fails: WikiText-2 final
`lm_head` relative-L2 improves from `{wiki['baseline_relative_l2']:.10f}` to
`{wiki['candidate_relative_l2']:.10f}`, while C4-en regresses from
`{c4['baseline_relative_l2']:.10f}` to `{c4['candidate_relative_l2']:.10f}`.

The direction is sealed by
`evidence/shared_qk_residual_cross_term_attention_v1/latest/VERIFICATION_NO_GO.json`
(SHA-256 `{no_go_sha}`). No paired smoke, shell admission/regression, SKY130
PPA, prototype, benchmark, signoff, tapeout, or silicon result followed. The
Manager must route `verification` back to `architecture` before a structurally
distinct successor can be frozen. Planner and Engineer must not edit
`research/PIPELINE_STATE.json`.

"""
    mission_path.write_text(before + active + end + after, encoding="utf-8")

    (ROOT / "CHECKPOINT.md").write_text(
        f"""# Goal

Verify `shared_qk_residual_cross_term_attention_v1` without entering PPA.

# Current State

The current Manager-owned stage is `verification`. Candidate `{candidate_id}`
is a reproducible bounded no-go because the frozen C4-en final-lm-head focused
metric regressed. The accepted prefix remains through `layer_0.v_proj`; first
unsupported remains `layer_0.rope_q`; mode remains `ADVANCE`.

# Verified Done

- Independent scalar/tensor arithmetic agrees for projection residuals,
  residual RoPE, authoritative base-score carry-through, and all three terms.
- Fresh deterministic vector generation, ten Python tests, Icarus simulation,
  explicit 4-state X/Z output checks, and two Verilator lint runs pass.
- Both 128-token all-layer traces reproduce identical inputs and exact baseline
  equality at Q/K projection, RoPE, and base-score boundaries.
- Five mandatory focused metrics improve; C4-en final `lm_head` worsens from
  `{c4['baseline_relative_l2']:.10f}` to `{c4['candidate_relative_l2']:.10f}`.

# Locked / Not Run

Paired smoke, shell admission/regression, canonical SKY130 PPA, prototype,
benchmark, signoff, tapeout, and silicon were not run.

# Next Required Action

Manager rollback from `verification` to `architecture` for a structurally
distinct successor. The sealed residual-cross-term direction must not be
retuned or advanced downstream.
""",
        encoding="utf-8",
    )
    dump(
        ROOT / ".argus" / "live-view.json",
        {
            "version": 1,
            "title": "Q/K residual cross-term verification no-go",
            "reason": "C4-en final lm-head relative-L2 regressed at the frozen all-layer focused gate.",
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "evidence_sha256": no_go_sha,
            "generated_at_utc": now,
            "paths": [
                "research/PIPELINE_STATE.json",
                "CHECKPOINT.md",
                "verification/RESULTS.json",
                "evidence/shared_qk_residual_cross_term_attention_v1/latest/VERIFICATION_NO_GO.json",
                "research/PUBLIC_STATUS.json",
            ],
        },
    )


def main() -> None:
    pipeline = load(ROOT / "research" / "PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "verification", "current stage is not verification")
    precheck = load(PRECHECK)
    manifest = load(ROOT / "design" / "RTL_MANIFEST.json")
    require(precheck.get("contract_id") == CONTRACT, "precheck contract differs")
    require(all(precheck.get("checklist", {}).values()), "precheck checklist is incomplete")
    require(
        precheck.get("candidate_rtl_hash") == manifest.get("candidate_rtl_hash"),
        "precheck and manifest candidate hashes differ",
    )
    candidate_id = precheck["candidate_id"]
    candidate_hash = precheck["candidate_rtl_hash"]

    command_results: list[dict[str, Any]] = []
    command_results.append(
        run("qk_vector_generation", [sys.executable, "tools/gen_qk_residual_cross_term_vectors.py"])
    )
    vector_hashes = {
        path: sha256(ROOT / path)
        for path in (
            "verification/generated/qk_residual_cross_term_vectors.json",
            "verification/generated/qk_residual_cross_term_vectors.svh",
        )
    }
    command_results.append(
        run("qk_vector_regeneration", [sys.executable, "tools/gen_qk_residual_cross_term_vectors.py"])
    )
    require(
        vector_hashes
        == {path: sha256(ROOT / path) for path in vector_hashes},
        "Q/K residual vector regeneration is not deterministic",
    )
    command_results.append(
        run(
            "qk_reference_and_tensor_tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "verification.test_qk_residual_cross_term",
                "verification.test_qk_residual_cross_term_full_model",
                "-v",
            ],
        )
    )
    command_results.append(
        run(
            "qk_rtl_elaboration",
            [
                "iverilog",
                "-g2012",
                "-Wall",
                "-Iverification/generated",
                "-o",
                "build/ace2_qk_residual_cross_term_verification_tb.vvp",
                "rtl/ace2_qk_residual_cross_term_core.sv",
                "verification/tb/ace2_qk_residual_cross_term_tb.sv",
            ],
        )
    )
    command_results.append(
        run(
            "qk_rtl_simulation",
            ["vvp", "build/ace2_qk_residual_cross_term_verification_tb.vvp"],
        )
    )
    for module in (
        "ace2_qk_residual_sidecar_core",
        "ace2_residual_cross_term_score_core",
    ):
        command_results.append(
            run(
                f"qk_lint_{module}",
                [
                    "verilator",
                    "--lint-only",
                    "--language",
                    "1800-2017",
                    "-Wall",
                    "-Wno-fatal",
                    "--top-module",
                    module,
                    "rtl/ace2_qk_residual_cross_term_core.sv",
                ],
            )
        )

    replay_baseline = RAW / "qk_focused_baseline_replay"
    replay_candidate = RAW / "qk_focused_candidate_replay"
    archive_replay(replay_baseline)
    archive_replay(replay_candidate)
    command_results.append(
        run(
            "qk_focused_baseline_replay",
            [
                sys.executable,
                "tools/localize_score_to_lm_head.py",
                "--output-dir",
                replay_baseline.relative_to(ROOT).as_posix(),
                "--diagnostic-rope-mechanism",
                "shared_qk_residual_cross_term_baseline_v1",
                "--capture-token-limit",
                "128",
            ],
        )
    )
    command_results.append(
        run(
            "qk_focused_candidate_replay",
            [
                sys.executable,
                "tools/localize_score_to_lm_head.py",
                "--output-dir",
                replay_candidate.relative_to(ROOT).as_posix(),
                "--diagnostic-rope-mechanism",
                CONTRACT,
                "--capture-token-limit",
                "128",
            ],
        )
    )

    simulation_log = RAW / "qk_rtl_simulation.log"
    simulation_text = simulation_log.read_text(encoding="utf-8")
    require("TB_PASS qk_residual_cross_term" in simulation_text, "RTL pass token is missing")
    require("SCORE_LATENCY_PASS cycles=216" in simulation_text, "216-cycle score assertion is missing")
    require(
        "XZ_OUTPUT_CHECK_PASS phases=post_reset,post_backpressure,post_clear,post_midflight_reset"
        in simulation_text,
        "explicit X/Z output checks are missing",
    )
    require(
        "SIDECAR_OVERFLOW_CHECK_PASS source=divider_remainder_invariant"
        in simulation_text,
        "sidecar overflow invariant check is missing",
    )
    require(
        "SCORE_OVERFLOW_CHECK_PASS source=checked_q20_44_accumulator"
        in simulation_text,
        "score overflow check is missing",
    )
    require(
        "RESET_MIDFLIGHT_CHECK_PASS modules=sidecar,score" in simulation_text,
        "mid-flight reset check is missing",
    )

    baseline = load(BASELINE)
    candidate = load(CANDIDATE)
    replay_baseline_result = load(replay_baseline / "results.json")
    replay_candidate_result = load(replay_candidate / "results.json")
    for path in (
        BASELINE,
        CANDIDATE,
        replay_baseline / "results.json",
        replay_candidate / "results.json",
    ):
        verify_sha256s(path)
    require(
        replay_baseline_result["input_observations"] == baseline["input_observations"]
        and replay_baseline_result["comparisons"] == baseline["comparisons"]
        and replay_baseline_result["qk_residual_contract"] == baseline["qk_residual_contract"],
        "fresh focused baseline replay differs from the bound artifact",
    )
    require(
        replay_candidate_result["input_observations"] == candidate["input_observations"]
        and replay_candidate_result["comparisons"] == candidate["comparisons"]
        and replay_candidate_result["qk_residual_contract"] == candidate["qk_residual_contract"],
        "fresh focused candidate replay differs from the bound artifact",
    )
    require(baseline.get("capture_token_limit") == 128, "baseline is not the frozen 128-token run")
    require(candidate.get("capture_token_limit") == 128, "candidate is not the frozen 128-token run")
    require(
        baseline.get("diagnostic_rope_mechanism")
        == "shared_qk_residual_cross_term_baseline_v1",
        "focused baseline mechanism differs",
    )
    require(
        candidate.get("diagnostic_rope_mechanism") == CONTRACT,
        "focused candidate mechanism differs",
    )
    require(
        baseline.get("input_observations") == candidate.get("input_observations"),
        "focused input observations differ",
    )
    for result in (baseline, candidate):
        contract = result.get("qk_residual_contract", {})
        require(contract.get("layer_scope") == "all_24_layers", "focused layer scope differs")
        require(contract.get("all_baseline_equality_checks_passed") is True, "baseline equality failed")
        require(len(contract.get("layers", [])) == 24, "focused result lacks 24 layer records")

    comparisons: dict[str, Any] = {}
    gate_passed = True
    for dataset in ("wikitext2", "c4_en_512"):
        comparisons[dataset] = {}
        for boundary in (
            "model.layers.0.score",
            "model.layers.0.attention_value",
            "lm_head",
        ):
            baseline_value = baseline["comparisons"][dataset][boundary]["relative_l2_error"]
            candidate_value = candidate["comparisons"][dataset][boundary]["relative_l2_error"]
            improved = candidate_value < baseline_value
            comparisons[dataset][boundary] = {
                "baseline_relative_l2": baseline_value,
                "candidate_relative_l2": candidate_value,
                "delta_relative_l2": candidate_value - baseline_value,
                "strictly_improved": improved,
            }
            gate_passed = gate_passed and improved
    require(not gate_passed, "focused gate unexpectedly passed; paired smoke requires a separate flow")
    require(
        comparisons["c4_en_512"]["lm_head"]["strictly_improved"] is False,
        "expected C4-en final lm-head no-go is absent",
    )
    require(
        all(
            comparisons[dataset][boundary]["strictly_improved"]
            for dataset in comparisons
            for boundary in comparisons[dataset]
            if not (dataset == "c4_en_512" and boundary == "lm_head")
        ),
        "focused no-go differs from the observed single-metric C4 failure",
    )

    now = utc_now()
    source_paths = [
        ROOT / "Makefile",
        ROOT / "verification" / "PLAN.md",
        ROOT / "rtl" / "ace2_qk_residual_cross_term_core.sv",
        ROOT / "tools" / "ace2_qk_residual_cross_term_reference.py",
        ROOT / "tools" / "ace2_full_model_fixed_point.py",
        ROOT / "tools" / "localize_score_to_lm_head.py",
        ROOT / "tools" / "run_qk_residual_cross_term_verification.py",
        ROOT / "verification" / "test_qk_residual_cross_term.py",
        ROOT / "verification" / "test_qk_residual_cross_term_full_model.py",
        ROOT / "verification" / "tb" / "ace2_qk_residual_cross_term_tb.sv",
        ROOT / "verification" / "generated" / "qk_residual_cross_term_vectors.json",
        ROOT / "verification" / "generated" / "qk_residual_cross_term_vectors.svh",
        ROOT / "reference" / "generated" / "qk_residual_scale32_metadata.json",
        ROOT / "reference" / "qk_residual_cross_term_full_model_hook.json",
        PRECHECK,
    ]
    source_hashes = [artifact(path) for path in source_paths]
    oracle = {
        "schema_version": 2,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "claim_boundary": "Independent scalar/tensor/RTL and focused-quality verification only; no downstream claim.",
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "standalone_vector_oracle": {
            "reference": artifact(ROOT / "tools" / "ace2_qk_residual_cross_term_reference.py"),
            "vectors_json": artifact(ROOT / "verification" / "generated" / "qk_residual_cross_term_vectors.json"),
            "vectors_svh": artifact(ROOT / "verification" / "generated" / "qk_residual_cross_term_vectors.svh"),
            "numeric_acceptance": "bit_exact",
        },
        "full_model_oracle": {
            "runner": artifact(ROOT / "tools" / "ace2_full_model_fixed_point.py"),
            "trace": artifact(ROOT / "tools" / "localize_score_to_lm_head.py"),
            "hook": artifact(ROOT / "reference" / "qk_residual_cross_term_full_model_hook.json"),
            "metadata": artifact(ROOT / "reference" / "generated" / "qk_residual_scale32_metadata.json"),
            "baseline": artifact(BASELINE),
            "candidate": artifact(CANDIDATE),
            "input_observations_equal": True,
            "baseline_equality_passed_all_24_layers": True,
            "focused_comparisons": comparisons,
            "quality_gate_passed": False,
        },
    }
    dump(ORACLE_MANIFEST, oracle)
    properties = {
        "schema_version": 2,
        "generated_at_utc": now,
        "formal_engine_run": False,
        "reason_formal_engine_not_run": "The candidate has one clock and current closure used independent executable references plus 4-state RTL simulation; no formal proof is claimed.",
        "properties": [
            {"id": "authoritative_base_score_carry", "status": "exercised_by_simulation", "evidence": "SCORE_LATENCY_PASS and generated base-score vectors"},
            {"id": "reset_clear_backpressure", "status": "exercised_by_simulation", "evidence": "RESET_MIDFLIGHT_CHECK_PASS plus clear and backpressure checks in verification/tb/ace2_qk_residual_cross_term_tb.sv"},
            {"id": "reserved_and_descriptor_errors", "status": "exercised_by_simulation", "evidence": "verification/tb/ace2_qk_residual_cross_term_tb.sv"},
            {"id": "numeric_overflow_fail_closed", "status": "exercised_by_simulation", "evidence": "SIDECAR_OVERFLOW_CHECK_PASS and SCORE_OVERFLOW_CHECK_PASS"},
            {"id": "x_z_known_outputs", "status": "exercised_by_4_state_simulation", "evidence": "XZ_OUTPUT_CHECK_PASS phases=post_reset,post_backpressure,post_clear,post_midflight_reset"},
            {"id": "single_clock_no_internal_cdc", "status": "specified_not_cdc_applicable", "evidence": "Both candidate modules expose only clk_i"},
        ],
    }
    dump(PROPERTY_MANIFEST, properties)

    no_go = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "bound_at_utc": now,
        "manager_owned_stage": "verification",
        "decision": "bounded_no_go_focused_discriminator_c4_lm_head_failed",
        "accepted_capability": False,
        "accepted_frontier_changed": False,
        "stage_closing": False,
        "standalone_verification": {
            "passed": True,
            "commands": command_results,
            "precheck": artifact(PRECHECK),
        },
        "focused_discriminator": {
            "run": True,
            "passed": False,
            "scope": "all_24_layers_128_tokens_two_frozen_datasets",
            "baseline": artifact(BASELINE),
            "candidate": artifact(CANDIDATE),
            "fresh_baseline_replay": artifact(replay_baseline / "results.json"),
            "fresh_candidate_replay": artifact(replay_candidate / "results.json"),
            "input_observations_equal": True,
            "baseline_equality_passed": True,
            "comparisons": comparisons,
            "finding": "Five mandatory metrics improved, but C4-en final lm-head relative-L2 regressed.",
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
            "remaining_area_reserve_mm2": 1.3891253728,
            "candidate_ppa_run": False,
        },
        "required_next_action": "Manager rolls verification back to architecture for a structurally distinct successor; this direction is sealed.",
        "claim_boundary": "Bounded negative verification result only; no PPA, benchmark acceptance, signoff, tapeout, or silicon claim.",
        "integrity": {"canonical_sha256": None},
    }
    no_go["integrity"]["canonical_sha256"] = canonical_sha256(no_go)
    dump(NO_GO, no_go)
    no_go_sha = sha256(NO_GO)

    results = {
        "schema_version": 2,
        "generated_at_utc": now,
        "stage": "verification",
        "pipeline_current_stage": "verification",
        "status": "bounded_no_go",
        "stage_closing": False,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "candidate_accepted": False,
        "error": "C4-en final lm-head relative-L2 failed strict improvement",
        "commands": command_results,
        "oracle_manifest": artifact(ORACLE_MANIFEST),
        "property_manifest": artifact(PROPERTY_MANIFEST),
        "source_hashes": source_hashes,
        "coverage": {
            "standalone": precheck["coverage"],
            "full_model_layers": 24,
            "focused_datasets": ["wikitext2", "c4_en_512"],
            "capture_token_limit": 128,
            "baseline_equality_boundaries": candidate["qk_residual_contract"]["baseline_equality_boundaries"],
            "stress": ["reset_midflight_sidecar_and_score", "clear", "backpressure", "reserved_s4", "descriptor_errors", "clamps", "numeric_overflow_sidecar_divider_remainder_invariant", "numeric_overflow_score_checked_q20_44_accumulator", "x_z_known_outputs_post_reset_backpressure_clear_midflight_reset", "row_lengths_1_2_63_64_65"],
        },
        "quality_gate": {
            "passed": False,
            "comparisons": comparisons,
            "no_go": artifact(NO_GO),
        },
        "checklist": {
            "verification.independent-oracle": False,
            "verification.coverage-stress": True,
            "verification.reproducible-green": True,
        },
        "checklist_explanation": {
            "verification.independent-oracle": "Arithmetic and baseline equality pass, but the oracle-owned C4-en final-lm-head quality constraint fails, so stage closure is false.",
            "verification.coverage-stress": "Standalone and all-layer stress coverage passed, including explicit 4-state X/Z-clean output checks and single-clock CDC disposition.",
            "verification.reproducible-green": "Fresh vector, Python, Icarus, and Verilator commands exited successfully with raw logs.",
        },
        "claim_boundary": [
            "Current-stage bounded verification no-go only.",
            "No paired smoke, shell admission, PPA, prototype, benchmark, signoff, tapeout, or silicon claim.",
        ],
        "manager_stage_transition_owner": "Manager; this tool does not edit research/PIPELINE_STATE.json.",
    }
    dump(RESULTS, results)

    manifest.update(
        {
            "stage": "verification",
            "current_stage": "verification",
            "architecture_contract_status": f"{CONTRACT}_sealed_focused_discriminator_no_go",
            "candidate_status": "authorization_consumed_rejected_focused_discriminator",
            "candidate_meets_numeric_acceptance": False,
            "candidate_verification_complete": False,
            "candidate_requires_fresh_sky130_ppa": False,
            "candidate_verification_binding": {
                "status": "bounded_no_go",
                "evidence": artifact(NO_GO),
                "focused_discriminator_passed": False,
                "paired_smoke_run": False,
                "stage_closing": False,
            },
            "claim_boundaries": [
                "Standalone and all-layer verification executed for the residual cross-term candidate.",
                "The C4-en final-lm-head strict-improvement gate failed; accepted prefix and historical PPA remain unchanged.",
                "No paired smoke, shell regression, candidate PPA, prototype, benchmark, signoff, tapeout, or silicon result followed.",
            ],
            "generated_at_utc": now,
        }
    )
    manifest.setdefault("latest_evidence", {})["qk_residual_cross_term_verification_no_go"] = artifact(NO_GO)
    dump(ROOT / "design" / "RTL_MANIFEST.json", manifest)

    policy = load(ROOT / "design" / "FAST_LOOP_POLICY.json")
    for key in (
        "active_repair_authorization",
        "architecture_proposal_authorization",
        "selected_replacement_contract",
    ):
        policy[key].update(
            {
                "implementation_authorized": False,
                "operator_approval_consumed": True,
                "status": "authorization_consumed_rejected_focused_discriminator",
                "candidate_id": candidate_id,
                "candidate_rtl_hash": candidate_hash,
                "required_manager_action": "rollback_verification_to_architecture_for_structurally_distinct_successor",
                "result_binding": NO_GO.relative_to(ROOT).as_posix(),
                "result_binding_sha256": no_go_sha,
            }
        )
    policy["manager_recommendation"] = "rollback_verification_to_architecture_after_qk_residual_focused_no_go"
    dump(ROOT / "design" / "FAST_LOOP_POLICY.json", policy)

    target = load(ROOT / "design" / "TARGET.json")
    target["current_stage"] = "verification"
    target["current_architecture_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    target["fast_loop_contract"]["manager_recommendation"] = policy["manager_recommendation"]
    target["generated_at_utc"] = now
    dump(ROOT / "design" / "TARGET.json", target)

    scope = load(ROOT / "design" / "CHIP_SCOPE.json")
    scope["stage"] = {
        "current_stage": "verification",
        "current_stage_status": "candidate_rejected_manager_rollback_required",
        "downstream_stages_locked_until_manager_advance": ["ppa", "prototype", "benchmark", "signoff"],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"].update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "required_next_action": "manager_rollback_verification_to_architecture",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
            "result_binding_sha256": no_go_sha,
        }
    )
    scope["implementation_frontier"]["latest_decision"] = "qk_residual_cross_term_focused_discriminator_bounded_no_go"
    scope["implementation_frontier"]["supported_layer_operator_prefix"] = PREFIX
    scope["implementation_frontier"]["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    dump(ROOT / "design" / "CHIP_SCOPE.json", scope)

    status = load(ROOT / "research" / "PUBLIC_STATUS.json")
    status["current_mode"] = "ADVANCE"
    status["supported_layer_operator_prefix"] = PREFIX
    status["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    status["latest_decision"] = "qk_residual_cross_term_focused_discriminator_bounded_no_go"
    status["latest_ppa_frontier_status"] = "historical_frontier_preserved_no_candidate_ppa"
    status["selected_replacement_contract"] = {
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "status": "rejected_focused_discriminator",
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
        "result_binding_sha256": no_go_sha,
    }
    status["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment", "rtl"],
        "current_stage": "verification",
        "current_stage_status": "bounded_no_go_manager_rollback_required",
        "current_stage_checklist": results["checklist"],
        "current_stage_evidence": [
            "verification/RESULTS.json",
            NO_GO.relative_to(ROOT).as_posix(),
            BASELINE.relative_to(ROOT).as_posix(),
            CANDIDATE.relative_to(ROOT).as_posix(),
        ],
    }
    status["blockers"] = [
        {
            "id": "qk_residual_cross_term_c4_lm_head_focused_no_go",
            "stage": "verification",
            "status": "active",
            "reason": "C4-en final lm-head relative-L2 regressed at the frozen all-layer focused gate.",
            "required_resolution": "Manager rollback to architecture for a structurally distinct successor.",
            "evidence": NO_GO.relative_to(ROOT).as_posix(),
        }
    ]
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container["current_mode"] = "ADVANCE"
        container["current_stage"] = "verification"
        container["ordered_supported_layer_operator_prefix"] = PREFIX
        container["supported_layer_operator_prefix"] = PREFIX
        container["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
        container["latest_decision"] = status["latest_decision"]
        container["latest_ppa_frontier_status"] = status["latest_ppa_frontier_status"]
        container["candidate_rtl_hash"] = candidate_hash
        container["routing_status"] = "manager_rollback_to_architecture_required"
        container["candidate_meets_numeric_acceptance"] = False
    status["public_claims"] = [
        {"claim": "the residual cross-term candidate is a sealed focused-quality no-go", "evidence": [NO_GO.relative_to(ROOT).as_posix()]},
        {"claim": "the supported prefix, mode, immutable targets, and historical PPA frontier are unchanged", "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json"]},
        {"claim": "no paired smoke, shell regression, candidate PPA, prototype, benchmark, signoff, tapeout, or silicon result followed", "evidence": ["verification/RESULTS.json"]},
    ]
    public_artifact_paths = {
        entry["path"]
        for entry in status.get("artifact_hashes", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    public_artifact_paths.update(
        {
            "formal/ACE2_VERIFICATION_PROPERTIES.json",
            "reference/ORACLE_MANIFEST.json",
            "verification/PLAN.md",
            "verification/RESULTS.json",
            "verification/tb/ace2_qk_residual_cross_term_tb.sv",
            "tools/run_qk_residual_cross_term_verification.py",
            NO_GO.relative_to(ROOT).as_posix(),
        }
    )
    status["artifact_hashes"] = [
        artifact(ROOT / path)
        for path in sorted(public_artifact_paths)
        if (ROOT / path).is_file()
    ]
    status["generated_at_utc"] = now
    status["last_updated_utc"] = now
    status.setdefault("integrity", {})["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump(ROOT / "research" / "PUBLIC_STATUS.json", status)

    update_human_state(candidate_id, candidate_hash, no_go_sha, comparisons, now)
    print(
        "ACE2_QK_RESIDUAL_VERIFICATION_NO_GO "
        f"candidate={candidate_id} c4_lm_head_delta="
        f"{comparisons['c4_en_512']['lm_head']['delta_relative_l2']:.12f} "
        f"results={RESULTS.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ACE2_QK_RESIDUAL_VERIFICATION_FAIL error={exc}", file=sys.stderr)
        raise
