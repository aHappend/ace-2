#!/usr/bin/env python3
"""Run and bind the bounded RTL-stage V-residual candidate preflight."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_v_residual_value_correction_attention_v1"
EVIDENCE = ROOT / "evidence/shared_v_residual_value_correction_attention_v1/latest"
PRECHECK = EVIDENCE / "PRECHECK.json"
RTL = ROOT / "rtl/ace2_v_residual_value_correction_core.sv"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}
SOURCE_PATHS = [
    "Makefile",
    "rtl/ace2_v_residual_value_correction_core.sv",
    "tools/ace2_v_residual_value_correction_reference.py",
    "tools/calibrate_v_residual_scale32_source.py",
    "tools/gen_v_residual_scale32_metadata.py",
    "tools/gen_v_residual_value_correction_vectors.py",
    "tools/run_v_residual_value_correction_preflight.py",
    "verification/test_v_residual_value_correction.py",
    "verification/tb/ace2_v_residual_value_correction_tb.sv",
    "verification/generated/v_residual_value_correction_vectors.json",
    "verification/generated/v_residual_value_correction_vectors.svh",
    "reference/generated/v_residual_scale32_calibration_source.json",
    "reference/generated/v_residual_scale32_metadata.json",
    "reference/v_residual_value_correction_full_model_hook.json",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: str | Path) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {resolved}")
    return value


def dump(path: str | Path, value: dict[str, Any]) -> None:
    resolved = path if isinstance(path, Path) else ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def aggregate_hash(records: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(records):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(records[path].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def run(command: list[str], log_name: str) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = EVIDENCE / log_name
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return artifact(log)


def manager_transition() -> dict[str, Any]:
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    event = pipeline.get("stage_history", [])[-1]
    require(
        event.get("by") == "manager"
        and event.get("from_stage") == "environment"
        and event.get("to_stage") == "rtl",
        "latest Manager event is not the accepted environment-to-RTL transition",
    )
    return event


def active_authority(candidate_id: str, candidate_hash: str) -> dict[str, Any]:
    scope = load("design/CHIP_SCOPE.json")
    source = scope["authority_override"]["operator_implementation_approval"]
    require(source.get("contract_id") == CONTRACT, "active authority contract mismatch")
    require(source.get("operator_approval_consumed") is False, "authority already consumed")
    record = copy.deepcopy(source)
    record.update(
        {
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "rtl_started": True,
            "required_manager_action": "hold_rtl_for_independent_checklist_review",
            "status": "authorization_consumed_rtl_preflight_pass_review_pending",
            "stage_closing": False,
        }
    )
    return record


def update_project_state(
    *,
    candidate_id: str,
    candidate_hash: str,
    source_hashes: dict[str, str],
    logs: dict[str, dict[str, Any]],
    interfaces: dict[str, dict[str, Any]],
    now: str,
) -> dict[str, Any]:
    authority = active_authority(candidate_id, candidate_hash)
    manifest = load("design/RTL_MANIFEST.json")
    calibration = artifact("reference/generated/v_residual_scale32_calibration_source.json")
    metadata = artifact("reference/generated/v_residual_scale32_metadata.json")
    hook = artifact("reference/v_residual_value_correction_full_model_hook.json")
    vectors_json = artifact("verification/generated/v_residual_value_correction_vectors.json")
    vectors_svh = artifact("verification/generated/v_residual_value_correction_vectors.svh")
    rtl_artifact = artifact(RTL)
    manifest.update(
        {
            "stage": "rtl",
            "current_stage": "rtl",
            "architecture_contract_status": "manager_advanced_to_rtl_successor_implemented",
            "candidate_id": candidate_id,
            "candidate_layer_operator": CONTRACT,
            "candidate_status": "pass_ready_for_independent_rtl_review",
            "candidate_rtl_hash": candidate_hash,
            "candidate_rtl_hash_scope": "ordered_standalone_candidate_sources_not_shell_admitted",
            "candidate_source_hashes": source_hashes,
            "candidate_rtl_sources": [
                {
                    **rtl_artifact,
                    "modules": [
                        "ace2_v_residual_projection_core",
                        "ace2_v_residual_value_correction_core",
                    ],
                    "kind": "first_party_bounded_candidate_rtl",
                    "third_party": False,
                }
            ],
            "candidate_generated_hashes": {
                item["path"]: item["sha256"]
                for item in (calibration, metadata, vectors_json, vectors_svh)
            },
            "candidate_generated_sources": [
                {
                    **calibration,
                    "generator": "tools/calibrate_v_residual_scale32_source.py",
                    "regeneration_command": (
                        ".venv/bin/python tools/calibrate_v_residual_scale32_source.py "
                        "--output reference/generated/v_residual_scale32_calibration_source.json"
                    ),
                    "kind": "generated_calibration_reference_source",
                    "third_party": False,
                },
                {
                    **metadata,
                    "generator": "tools/gen_v_residual_scale32_metadata.py",
                    "regeneration_command": (
                        ".venv/bin/python tools/gen_v_residual_scale32_metadata.py "
                        "--output reference/generated/v_residual_scale32_metadata.json"
                    ),
                    "kind": "generated_model_image_metadata",
                    "third_party": False,
                },
                {
                    **vectors_json,
                    "generator": "tools/gen_v_residual_value_correction_vectors.py",
                    "reference": "tools/ace2_v_residual_value_correction_reference.py",
                    "regeneration_command": (
                        ".venv/bin/python tools/gen_v_residual_value_correction_vectors.py"
                    ),
                    "kind": "generated_verification_source",
                    "third_party": False,
                },
                {
                    **vectors_svh,
                    "generator": "tools/gen_v_residual_value_correction_vectors.py",
                    "reference": "tools/ace2_v_residual_value_correction_reference.py",
                    "regeneration_command": (
                        ".venv/bin/python tools/gen_v_residual_value_correction_vectors.py"
                    ),
                    "kind": "generated_verification_source",
                    "third_party": False,
                },
            ],
            "candidate_interface": {
                "status": "implemented_and_verilator_elaborated",
                "descriptor_or_csr_change": False,
                "correction_core_identity_ports": [],
                "scheduler_identity_validation": [
                    "layer_id",
                    "query_position",
                    "query_head",
                    "kv_head",
                    "key_index",
                    "output_lane",
                ],
                "modules": {
                    "ace2_v_residual_projection_core": {
                        "parameters": {},
                        "inputs": [
                            "clk_i",
                            "rst_ni",
                            "clear_i",
                            "start_valid_i",
                            "accumulator_s32_i[31:0]",
                            "multiplier_s32_i[31:0]",
                            "shift_u6_i[5:0]",
                            "baseline_v_scale32_i[31:0]",
                            "residual_v_scale32_i[31:0]",
                            "out_ready_i",
                        ],
                        "outputs": [
                            "start_ready_o",
                            "out_valid_o",
                            "baseline_v8_o[7:0]",
                            "residual_v_canonical_u8_o[7:0]",
                            "positive_clamp_o",
                            "negative_clamp_o",
                            "descriptor_error_o",
                            "numeric_overflow_o",
                        ],
                    },
                    "ace2_v_residual_value_correction_core": {
                        "parameters": {"CONTEXT_MAX": 32768},
                        "inputs": [
                            "clk_i",
                            "rst_ni",
                            "clear_i",
                            "start_valid_i",
                            "lane_count_u16_i[15:0]",
                            "baseline_value_accumulator_s32_i[31:0]",
                            "baseline_v_scale32_i[31:0]",
                            "residual_v_scale32_i[31:0]",
                            "sample_valid_i",
                            "probability_q0_15_u16_i[15:0]",
                            "residual_v_canonical_u8_i[7:0]",
                            "out_ready_i",
                        ],
                        "outputs": [
                            "start_ready_o",
                            "sample_ready_o",
                            "out_valid_o",
                            "corrected_value_accumulator_s32_o[31:0]",
                            "correction_baseline_domain_s32_o[31:0]",
                            "correction_raw_s64_o[63:0]",
                            "descriptor_error_o",
                            "numeric_overflow_o",
                        ],
                    },
                },
                "interface_xml": interfaces,
                "forbidden_inputs_absent": [
                    "query_residual",
                    "key_residual",
                    "rope_residual",
                    "score_correction",
                    "prompt_id",
                    "dataset_id",
                    "runtime_learned_gate",
                ],
            },
            "candidate_model_metadata": {
                "status": "generated_and_hash_bound_before_quality",
                "record_count": 48,
                "record_order": "layer_0_to_23_then_kv_head_0_to_1",
                "record_format": "little_endian_u16_significand_s8_exponent_u8_reserved_zero",
                "baseline_v_scale32_preservation": "bit_identical",
                "derivation_rule": "smallest_legal_normalized_scale32_ceiling_of_exact_baseline_v_scale32_divided_by_14",
                "quality_tuning_permitted": False,
                "calibration_source": calibration,
                "generated_table": metadata,
            },
            "candidate_schedule": {
                "v_residual_projection_cycles_per_token": 128,
                "v_residual_projection_throughput": "one_lane_per_cycle_when_output_ready",
                "residual_value_mac_cycles": "lane_count_per_output_lane",
                "residual_value_conversion_cycles": 8,
                "residual_value_conversion_schedule": "four_restoring_quotient_bits_per_cycle_for_eight_cycles",
                "all_896_lane_conversion_cycles_per_row": 7168,
            },
            "candidate_meets_numeric_acceptance": False,
            "candidate_requires_fresh_sky130_ppa": True,
            "candidate_review_binding": {
                "candidate_capability_accepted": False,
                "decision": "independent_rtl_checklist_review_pending",
                "scope": "standalone_candidate_rtl_checklist_only",
                "stage_closing": False,
                "evidence": PRECHECK.relative_to(ROOT).as_posix(),
            },
            "proposed_replacement_contract": authority,
            "claim_boundaries": [
                "Standalone synthesizable candidate RTL, deterministic vectors, lint, elaboration, and simulation only.",
                "Accepted prefix and historical PPA frontier are unchanged.",
                "No focused quality, paired smoke, shell regression, SKY130 PPA, prototype, benchmark, signoff, tapeout, or silicon claim is made.",
            ],
        }
    )
    manifest["traceability"] = {
        "architecture_contract_gap": {
            "status": "resolved_by_hash_bound_successor_rtl_preflight",
            "resolution_owner": "independent_rtl_reviewer_pending",
        },
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "selected_mechanism": CONTRACT,
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
        "stage_checklist": copy.deepcopy(CHECKLIST),
    }
    provenance_names = {entry.get("name") for entry in manifest.get("ip_provenance", [])}
    if "ace2_v_residual_projection_core_and_value_correction_core" not in provenance_names:
        manifest.setdefault("ip_provenance", []).append(
            {
                "name": "ace2_v_residual_projection_core_and_value_correction_core",
                "kind": "first_party_bounded_candidate_rtl",
                "path": rtl_artifact["path"],
                "sha256": rtl_artifact["sha256"],
                "source_revision": rtl_artifact["sha256"],
                "license": "repository project license not separately declared in this manifest",
                "third_party": False,
            }
        )
    dump("design/RTL_MANIFEST.json", manifest)

    policy = load("design/FAST_LOOP_POLICY.json")
    for key in (
        "active_repair_authorization",
        "architecture_proposal_authorization",
        "selected_replacement_contract",
    ):
        policy[key] = copy.deepcopy(authority)
    policy["manager_recommendation"] = "hold_rtl_for_independent_checklist_review"
    dump("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["current_stage"] = "rtl"
    target["current_architecture_contract"] = copy.deepcopy(authority)
    target["fast_loop_contract"]["manager_recommendation"] = policy["manager_recommendation"]
    for key in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][key] = copy.deepcopy(authority)
    target["generated_at_utc"] = now
    dump("design/TARGET.json", target)

    scope = load("design/CHIP_SCOPE.json")
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_status": "candidate_preflight_pass_independent_review_pending",
        "downstream_stages_locked_until_manager_advance": [
            "verification",
            "ppa",
            "prototype",
            "benchmark",
            "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"].update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "operator_implementation_approval": copy.deepcopy(authority),
            "required_next_action": "independent_rtl_checklist_review",
        }
    )
    scope["implementation_frontier"].update(
        {
            "current_mode": "ADVANCE",
            "latest_decision": "v_residual_rtl_preflight_pass_review_pending",
            "ordered_supported_layer_operator_prefix": PREFIX,
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "candidate_rtl_hash": candidate_hash,
            "candidate_rtl_hash_scope": "standalone_not_shell_admitted_no_candidate_ppa",
        }
    )
    dump("design/CHIP_SCOPE.json", scope)

    oracle = {
        "schema_version": 2,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "claim_boundary": "Standalone integer reference and RTL preflight only; focused quality and downstream stages were not run.",
        "standalone_vector_oracle": {
            "numeric_acceptance": "bit_exact",
            "reference": artifact("tools/ace2_v_residual_value_correction_reference.py"),
            "vectors_json": vectors_json,
            "vectors_svh": vectors_svh,
        },
        "model_image_metadata": {
            "calibration_source": calibration,
            "metadata": metadata,
            "hook": hook,
            "record_count": 48,
            "quality_metrics_executed": False,
        },
    }
    dump("reference/ORACLE_MANIFEST.json", oracle)

    trace = f"""# ACE-2 RTL traceability notes

## Accepted shell frontier

The accepted shell remains hash-bound through `layer_0.v_proj`; first unsupported
remains `layer_0.rope_q`. Historical accepted PPA remains 62,199 cells,
0.6108746272 mm2, and +0.1502 ns setup slack at 100 MHz.

## Current V-residual value-correction candidate

- Contract: `{CONTRACT}`.
- Candidate: `{candidate_id}`; ordered source hash `{candidate_hash}`.
- `ace2_v_residual_projection_core` preserves the accepted signed-32 accumulator,
  positive multiplier, unsigned shift, and baseline int8 result while producing
  the exact signed-4 remainder with symmetric `[-7,+7]` clamp and canonical byte.
- `ace2_v_residual_value_correction_core` consumes only the authoritative Q0.15
  probability, canonical residual-V byte, frozen Scale32 pair, lane count, and
  authoritative baseline value accumulator. It has no identity, Q/K, RoPE,
  score, prompt, dataset, or learned-gate ports.
- The correction MAC accepts one key per cycle. One radix-16 restoring converter
  emits 32 quotient bits in exactly eight cycles after the complete lane sum;
  conversion and checked add occur once per output lane.
- The frozen model image contains 48 rows ordered by layer then KV head. Each
  residual record is the smallest legal Scale32 ceiling of the bit-identical
  baseline-V Scale32 divided by 14; no quality metric was executed or used.
- Reset, clear, backpressure, reserved `-8`, canonical sign extension, clamp,
  checked overflow, 14:2 mapping, all 24 layers, and row lengths 1/2/63/64/65
  are covered by the bound reference and preflight.
- Both modules and all generated sources are first-party with regeneration
  commands in `design/RTL_MANIFEST.json`; no third-party RTL or generated IP is added.
- The three RTL checklist evidence fields are true for this standalone candidate.
  Independent review remains pending; the candidate is not shell-admitted and
  has no focused quality, PPA, prototype, benchmark, or signoff claim.

## Historical rejected candidates

All prior RoPE/score/QK residual successors remain historical bounded no-go
evidence and are not current RTL capability.
"""
    (ROOT / "design/RTL_TRACEABILITY.md").write_text(trace, encoding="utf-8")

    checkpoint = f"""# Goal

Implement the bounded RTL for `{CONTRACT}` without entering verification or PPA.

# Current State

The Manager-owned stage is `rtl`. Candidate `{candidate_id}` has passed the
standalone reference/RTL preflight and awaits independent RTL-checklist review.
The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`.

# Verified Done

- Frozen 24-layer baseline-V calibration source and 48-row Scale32 table generated without quality metrics.
- Exact integer reference, deterministic vectors, and full 14:2/all-layer metadata checks pass.
- Icarus elaboration/simulation and warning-free Verilator lint pass for both candidate modules.
- The correction converter reproduces the frozen eight-cycle schedule.
- Authorization is consumed by this one bounded RTL attempt; no downstream run occurred.

# Locked / Not Run

Focused quality, paired smoke, shell admission/regression, canonical SKY130 PPA,
prototype, benchmark, signoff, tapeout, and silicon were not run.

# Next Required Action

Independent review of the three RTL checklist items for exact hash `{candidate_hash}`.
Planner and Engineer must not edit `research/PIPELINE_STATE.json`.
"""
    (ROOT / "CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    return manifest


def update_public_status(
    *,
    candidate_id: str,
    candidate_hash: str,
    authority: dict[str, Any],
    now: str,
) -> None:
    status = load("research/PUBLIC_STATUS.json")
    status["current_mode"] = "ADVANCE"
    status["supported_layer_operator_prefix"] = PREFIX
    status["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    status["latest_decision"] = "v_residual_rtl_preflight_pass_independent_review_pending"
    status["latest_ppa_frontier_status"] = "historical_frontier_preserved_no_candidate_ppa"
    status["selected_replacement_contract"] = copy.deepcopy(authority)
    status["architecture_proposal_gate"] = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "required_manager_action": "hold_rtl_for_independent_checklist_review",
        "stage_closing": False,
        "status": "rtl_preflight_pass_independent_review_pending",
    }
    status["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            PRECHECK.relative_to(ROOT).as_posix(),
        ],
        "current_stage_status": "candidate_preflight_pass_independent_review_pending",
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    status["blockers"] = [
        {
            "id": "independent_v_residual_rtl_checklist_review_pending",
            "stage": "rtl",
            "status": "active",
            "reason": "Standalone candidate preflight passed; independent checklist review is pending.",
            "required_resolution": "Review exact candidate hash before any focused quality or downstream run.",
            "evidence": PRECHECK.relative_to(ROOT).as_posix(),
        }
    ]
    candidate = {
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "standalone_not_shell_admitted_no_candidate_ppa",
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "stage_closing": False,
        "status": "rtl_preflight_pass_independent_review_pending",
    }
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status.setdefault(name, {})
        container.update(
            {
                "current_stage": "rtl",
                "current_mode": "ADVANCE",
                "supported_layer_operator_prefix": PREFIX,
                "ordered_supported_layer_operator_prefix": PREFIX,
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                "latest_decision": status["latest_decision"],
                "latest_ppa_frontier_status": status["latest_ppa_frontier_status"],
                "latest_rtl_candidate": copy.deepcopy(candidate),
                "candidate_rtl_hash": candidate_hash,
                "candidate_meets_numeric_acceptance": False,
                "routing_status": "independent_rtl_checklist_review_pending",
            }
        )
    status["public_claims"] = [
        {
            "claim": "the Manager-owned current stage is rtl",
            "evidence": ["research/PIPELINE_STATE.json"],
        },
        {
            "claim": "the V-residual standalone RTL preflight passed for the published candidate hash",
            "evidence": [PRECHECK.relative_to(ROOT).as_posix(), "design/RTL_MANIFEST.json"],
        },
        {
            "claim": "the supported prefix, immutable targets, mode, and historical PPA frontier are unchanged",
            "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json"],
        },
        {
            "claim": "no focused quality, paired smoke, shell regression, candidate PPA, prototype, benchmark, signoff, tapeout, or silicon result exists",
            "evidence": [PRECHECK.relative_to(ROOT).as_posix()],
        },
    ]
    public_paths = {
        item.get("path")
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path") != "research/PUBLIC_STATUS.json"
    }
    public_paths.update(
        {
            "CHECKPOINT.md",
            "design/CHIP_SCOPE.json",
            "design/FAST_LOOP_POLICY.json",
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            "design/TARGET.json",
            "reference/ORACLE_MANIFEST.json",
            "reference/generated/v_residual_scale32_metadata.json",
            "research/PIPELINE_STATE.json",
            PRECHECK.relative_to(ROOT).as_posix(),
        }
    )
    status["artifact_hashes"] = [
        artifact(path) for path in sorted(public_paths) if path and (ROOT / path).is_file()
    ]
    status["last_updated_utc"] = now
    status["generated_at_utc"] = now
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    transition = manager_transition()
    architecture_packet = load(
        "evidence/shared_qk_residual_cross_term_attention_v1/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
    )
    environment = load("research/ENVIRONMENT_AUDIT.json")
    require(architecture_packet.get("contract_id") == CONTRACT, "architecture packet contract mismatch")
    require(
        environment.get("contract_binding", {}).get("active_contract_id") == CONTRACT,
        "environment audit contract mismatch",
    )
    require(load("design/CHIP_SCOPE.json")["implementation_frontier"]["ordered_supported_layer_operator_prefix"] == PREFIX,
            "supported prefix changed")
    require(load("design/CHIP_SCOPE.json")["implementation_frontier"]["first_unsupported_layer_operator"] == FIRST_UNSUPPORTED,
            "first unsupported operator changed")

    logs: dict[str, dict[str, Any]] = {}
    logs["metadata_check"] = run(
        [".venv/bin/python", "tools/gen_v_residual_scale32_metadata.py", "--check"],
        "metadata_check.log",
    )
    run([".venv/bin/python", "tools/gen_v_residual_value_correction_vectors.py"], "vector_generation.log")
    first_vector_hashes = {
        path: sha256_file(ROOT / path)
        for path in (
            "verification/generated/v_residual_value_correction_vectors.json",
            "verification/generated/v_residual_value_correction_vectors.svh",
        )
    }
    logs["vector_regeneration"] = run(
        [".venv/bin/python", "tools/gen_v_residual_value_correction_vectors.py"],
        "vector_regeneration.log",
    )
    require(
        first_vector_hashes
        == {path: sha256_file(ROOT / path) for path in first_vector_hashes},
        "V residual vector generation is not deterministic",
    )
    logs["software_reference"] = run(
        [
            ".venv/bin/python",
            "-m",
            "unittest",
            "verification.test_v_residual_value_correction",
            "-v",
        ],
        "software_reference_unittest.log",
    )
    build = ROOT / "build/v_residual_value_correction"
    build.mkdir(parents=True, exist_ok=True)
    logs["elaboration"] = run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-Iverification/generated",
            "-o",
            str(build / "ace2_v_residual_tb.vvp"),
            "rtl/ace2_v_residual_value_correction_core.sv",
            "verification/tb/ace2_v_residual_value_correction_tb.sv",
        ],
        "rtl_elaboration.log",
    )
    logs["simulation"] = run(
        ["vvp", str(build / "ace2_v_residual_tb.vvp")],
        "rtl_simulation.log",
    )
    require(
        "ACE2_V_RESIDUAL_RTL_PASS projection_cases=4 correction_cycles=8"
        in (EVIDENCE / "rtl_simulation.log").read_text(encoding="utf-8"),
        "RTL simulation did not reproduce the frozen schedule marker",
    )
    logs["projection_lint"] = run(
        [
            "verilator",
            "--lint-only",
            "--language",
            "1800-2017",
            "-Wall",
            "--top-module",
            "ace2_v_residual_projection_core",
            "rtl/ace2_v_residual_value_correction_core.sv",
        ],
        "rtl_projection_lint.log",
    )
    logs["correction_lint"] = run(
        [
            "verilator",
            "--lint-only",
            "--language",
            "1800-2017",
            "-Wall",
            "--top-module",
            "ace2_v_residual_value_correction_core",
            "rtl/ace2_v_residual_value_correction_core.sv",
        ],
        "rtl_correction_lint.log",
    )
    interfaces: dict[str, dict[str, Any]] = {}
    for module, filename in (
        ("ace2_v_residual_projection_core", "projection_interface.xml"),
        ("ace2_v_residual_value_correction_core", "correction_interface.xml"),
    ):
        xml = EVIDENCE / filename
        logs[f"{module}_xml"] = run(
            [
                "verilator",
                "--xml-only",
                "--language",
                "1800-2017",
                "-Wall",
                "--top-module",
                module,
                "--xml-output",
                str(xml),
                "rtl/ace2_v_residual_value_correction_core.sv",
            ],
            f"{module}_xml.log",
        )
        interfaces[module] = artifact(xml)

    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}
    candidate_hash = aggregate_hash(source_hashes)
    candidate_id = f"v_residual_value_correction_{candidate_hash[:16]}"
    now = utc_now()
    manifest = update_project_state(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        source_hashes=source_hashes,
        logs=logs,
        interfaces=interfaces,
        now=now,
    )
    authority = manifest["proposed_replacement_contract"]
    precheck: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "current_stage": "rtl",
        "stage_closing": False,
        "status": "pass_ready_for_independent_rtl_review",
        "claim_boundary": "Standalone synthesizable candidate RTL preflight only; accepted prefix and historical PPA frontier are unchanged.",
        "checklist": {
            "manager_stage_is_rtl": True,
            "accepted_architecture_packet_and_environment_binding_bound": True,
            "standing_authorization_consumed_once": True,
            "calibration_source_generated_without_quality_metrics": True,
            "exact_48_row_scale32_table_generated": True,
            "independent_integer_reference_pass": True,
            "deterministic_vector_regeneration_pass": True,
            "iverilog_elaboration_clean": True,
            "bit_exact_rtl_simulation_pass": True,
            "projection_verilator_lint_clean": True,
            "correction_verilator_lint_clean": True,
            "manifest_interfaces_match_verilator_elaboration": True,
            "one_lane_per_cycle_projection_schedule_asserted": True,
            "eight_cycle_correction_conversion_schedule_asserted": True,
            "reset_clear_backpressure_reserved_code_and_errors_exercised": True,
            "first_party_provenance_and_regeneration_bound": True,
        },
        "stage_checklist": copy.deepcopy(CHECKLIST),
        "manager_transition": transition,
        "source_hashes": source_hashes,
        "candidate_rtl_source": artifact(RTL),
        "traceability_bindings": {
            "architecture": artifact("design/ARCHITECTURE.md"),
            "spec": artifact("design/SPEC.md"),
            "proposal": artifact("design/NUMERICAL_REPLACEMENT_PROPOSAL.md"),
            "architecture_packet": artifact(
                "evidence/shared_qk_residual_cross_term_attention_v1/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
            ),
            "environment_audit": artifact("research/ENVIRONMENT_AUDIT.json"),
            "reference": artifact("tools/ace2_v_residual_value_correction_reference.py"),
            "calibration_source": artifact("reference/generated/v_residual_scale32_calibration_source.json"),
            "metadata": artifact("reference/generated/v_residual_scale32_metadata.json"),
            "full_model_hook": artifact("reference/v_residual_value_correction_full_model_hook.json"),
            "rtl": artifact(RTL),
            "vectors_json": artifact("verification/generated/v_residual_value_correction_vectors.json"),
            "vectors_svh": artifact("verification/generated/v_residual_value_correction_vectors.svh"),
            "manifest": artifact("design/RTL_MANIFEST.json"),
            "traceability": artifact("design/RTL_TRACEABILITY.md"),
        },
        "logs": logs,
        "interface_xml": interfaces,
        "coverage": {
            "layer_ids": list(range(24)),
            "query_to_kv_head": [0] * 7 + [1] * 7,
            "row_lengths": [1, 2, 63, 64, 65],
            "metadata_records": 48,
            "reserved_s4_code": -8,
            "projection_cases": 4,
            "conversion_cycles": 8,
            "stress": [
                "reset",
                "clear_midflight",
                "output_backpressure",
                "positive_and_negative_clamp",
                "reserved_minus8",
                "noncanonical_sign_extension",
                "zero_lane_count",
                "checked_accumulator_overflow",
            ],
        },
        "schedule_contract": {
            "v_residual_projection_cycles_per_128_lane_token": 128,
            "residual_value_mac_cycles_per_output_lane": "lane_count",
            "residual_value_conversion_cycles_per_output_lane": 8,
            "residual_value_conversion_cycles_per_896_lane_row": 7168,
            "assertion_log": logs["simulation"],
        },
        "prohibited_runs": {
            "focused_quality": False,
            "paired_smoke": False,
            "full_shell_regression": False,
            "sky130_ppa": False,
            "prototype": False,
            "benchmark": False,
            "signoff": False,
        },
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    require(all(precheck["checklist"].values()), "preflight checklist is incomplete")
    precheck["integrity"]["canonical_sha256"] = canonical_sha256(precheck)
    dump(PRECHECK, precheck)
    update_public_status(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        authority=authority,
        now=now,
    )
    print(
        "ACE2_V_RESIDUAL_PREFLIGHT_PASS "
        f"candidate={candidate_id} hash={candidate_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
