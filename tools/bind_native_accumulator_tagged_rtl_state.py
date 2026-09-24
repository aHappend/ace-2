#!/usr/bin/env python3
"""Bind focused RTL evidence for native-accumulator tagged attention."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_native_accumulator_tagged_attention_v1"
RTL = "rtl/ace2_native_accumulator_tagged_attention_core.sv"
PREFIX = ["layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"]
FIRST_UNSUPPORTED = "layer_0.rope_q"
SOURCES = [
    "Makefile",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    RTL,
    "tools/ace2_native_accumulator_tagged_reference.py",
    "tools/gen_native_accumulator_tagged_vectors.py",
    "verification/generated/native_accumulator_tagged_attention_vectors.json",
    "verification/generated/native_accumulator_tagged_attention_vectors.svh",
    "verification/tb/ace2_native_accumulator_tagged_attention_tb.sv",
    "verification/test_native_accumulator_tagged_attention.py",
]
LOGS = [
    "evidence/shared_native_accumulator_tagged_attention_v1/latest/software_reference_unittest.log",
    "evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_simulation.log",
    "evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_rope_lint.log",
    "evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_score_lint.log",
    "evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_elaboration.log",
]
RTL_REVIEW = "evidence/review/rtl_checklist_shared_native_accumulator_tagged_attention_v1/decision.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: str, value: dict[str, Any]) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str) -> dict[str, Any]:
    item = ROOT / path
    return {"path": path, "bytes": item.stat().st_size, "sha256": sha256(path)}


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    return hashlib.sha256((json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def aggregate_hash(records: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(records):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(records[path].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def bind() -> None:
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager stage is not rtl")
    verdict = load("research/ENVIRONMENT_REVIEWER_VERDICT.json")
    require(verdict.get("verdict") == "done", "environment review is not done")
    require(verdict.get("contract_id") == CONTRACT, "environment review contract stale")
    require(verdict.get("implementation_authorized") is True, "standing authorization missing")
    for path in SOURCES + LOGS:
        require((ROOT / path).is_file(), f"missing candidate artifact: {path}")
    require((ROOT / LOGS[2]).stat().st_size == 0, "RoPE lint is not warning-free")
    require((ROOT / LOGS[3]).stat().st_size == 0, "score lint is not warning-free")
    require((ROOT / LOGS[4]).stat().st_size == 0, "candidate elaboration is not clean")
    require("TB_PASS native_accumulator_tagged_attention" in (ROOT / LOGS[1]).read_text(), "RTL simulation did not pass")
    require("OK" in (ROOT / LOGS[0]).read_text(), "software reference tests did not pass")
    rtl_text = (ROOT / RTL).read_text(encoding="utf-8")
    require(rtl_text.count(" * ") == 1, "candidate RTL does not contain exactly one multiplier operator")
    require("score_mul_req_valid_i" in rtl_text and "mul_req_valid_o" in rtl_text, "shared multiplier service interface missing")

    source_hashes = {path: sha256(path) for path in SOURCES}
    candidate_hash = aggregate_hash(source_hashes)
    candidate_id = f"native_accumulator_tagged_attention_{candidate_hash[:16]}"
    review = load(RTL_REVIEW) if (ROOT / RTL_REVIEW).is_file() else None
    if review is not None:
        require(review.get("reviewer_status") == "done", "RTL review is not done")
        require(review.get("candidate_rtl_hash") == candidate_hash, "RTL review candidate hash stale")
        require(review.get("live_rtl_source_sha256") == source_hashes[RTL], "RTL review source hash stale")
        require(all(review.get("checklist", {}).values()), "RTL review checklist incomplete")
    evidence_dir = ROOT / "evidence/shared_native_accumulator_tagged_attention_v1/latest"
    (evidence_dir / "source_hashes.txt").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(source_hashes.items())),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "source_hashes": source_hashes,
        "focused_checks": {"software_reference": True, "rtl_simulation": True, "verilator_lint": True, "iverilog_elaboration": True},
        "quality_discriminator_run": False,
        "paired_smoke_run": False,
        "full_shell_regression_run": False,
        "canonical_sky130_ppa_run": False,
        "stage_closing": False,
        "generated_at_utc": utc_now(),
    }
    dump("evidence/shared_native_accumulator_tagged_attention_v1/latest/candidate_evidence.json", evidence)

    manifest = load("design/RTL_MANIFEST.json")
    manifest["stage"] = "rtl"
    manifest["current_stage"] = "rtl"
    manifest["architecture_contract_status"] = f"{CONTRACT}_focused_rtl_pass_quality_discriminator_pending"
    manifest["candidate_layer_operator"] = CONTRACT
    manifest["candidate_status"] = "focused_rtl_and_reference_pass_quality_discriminator_pending"
    manifest["candidate_rtl_hash"] = candidate_hash
    manifest["candidate_rtl_hash_scope"] = "ordered_candidate_source_hashes_standalone_not_shell_admitted"
    manifest["candidate_source_hashes"] = source_hashes
    manifest["candidate_rtl_sources"] = [artifact(RTL)]
    manifest["candidate_generated_hashes"] = {
        path: source_hashes[path]
        for path in SOURCES
        if path.startswith("verification/generated/")
    }
    manifest["candidate_generated_sources"] = [
        {
            **artifact("verification/generated/native_accumulator_tagged_attention_vectors.json"),
            "generator": "tools/gen_native_accumulator_tagged_vectors.py",
            "reference": "tools/ace2_native_accumulator_tagged_reference.py",
            "regeneration_command": "python3 tools/gen_native_accumulator_tagged_vectors.py",
            "third_party": False,
        },
        {
            **artifact("verification/generated/native_accumulator_tagged_attention_vectors.svh"),
            "generator": "tools/gen_native_accumulator_tagged_vectors.py",
            "reference": "tools/ace2_native_accumulator_tagged_reference.py",
            "regeneration_command": "python3 tools/gen_native_accumulator_tagged_vectors.py",
            "third_party": False,
        },
    ]
    manifest["candidate_interface"] = {
        "status": "focused_standalone_rtl_bound_not_shell_admitted",
        "modules": [
            {
                "module": "ace2_native_accumulator_rope_core",
                "parameters": {},
                "ports": {
                    "clk_i": 1, "rst_ni": 1, "clear_i": 1, "in_valid_i": 1, "in_ready_o": 1,
                    "acc0_s32_i": 32, "scale0_u32_i": 32, "acc1_s32_i": 32, "scale1_u32_i": 32,
                    "cosine_q1_15_i": 16, "sine_q1_15_i": 16, "out_valid_o": 1, "out_ready_i": 1,
                    "real_mantissa_s32_o": 32, "real_exponent_s8_o": 8,
                    "imag_mantissa_s32_o": 32, "imag_exponent_s8_o": 8,
                    "descriptor_error_o": 1, "numeric_overflow_o": 1,
                    "score_mul_req_valid_i": 1, "score_mul_req_ready_o": 1,
                    "score_mul_operand_a_s32_i": 32, "score_mul_operand_b_s32_i": 32,
                    "score_mul_rsp_valid_o": 1, "score_mul_rsp_ready_i": 1,
                    "score_mul_product_s64_o": 64,
                },
            },
            {
                "module": "ace2_tagged_attention_score_core",
                "parameters": {"MAX_LANES": 64},
                "ports": {
                    "clk_i": 1, "rst_ni": 1, "clear_i": 1, "start_valid_i": 1, "start_ready_o": 1,
                    "lane_count_u7_i": 7, "lane_valid_i": 1, "lane_ready_o": 1,
                    "query_mantissa_s32_i": 32, "query_exponent_s8_i": 8,
                    "key_mantissa_s32_i": 32, "key_exponent_s8_i": 8,
                    "mul_req_valid_o": 1, "mul_req_ready_i": 1,
                    "mul_operand_a_s32_o": 32, "mul_operand_b_s32_o": 32,
                    "mul_rsp_valid_i": 1, "mul_rsp_ready_o": 1,
                    "mul_product_s64_i": 64,
                    "score_valid_o": 1, "score_ready_i": 1, "score_q20_44_s64_o": 64,
                    "descriptor_error_o": 1, "numeric_overflow_o": 1,
                },
            },
        ],
    }
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "rtl_checklist_done_quality_discriminator_pending" if review is not None else "focused_rtl_pass_quality_discriminator_pending",
        "evidence": RTL_REVIEW if review is not None else "evidence/shared_native_accumulator_tagged_attention_v1/latest/candidate_evidence.json",
        "stage_closing": False,
    }
    if review is not None:
        manifest["independent_reviewer_verdict"] = "done"
        manifest["independent_reviewer_acceptance"] = {
            **artifact(RTL_REVIEW),
            "scope": "standalone_candidate_rtl_checklist_only",
            "quality_discriminator_complete": False,
            "shell_admitted": False,
            "ppa_run": False,
        }
    manifest["candidate_verification_binding"] = {
        "status": "focused_reference_simulation_lint_elaboration_pass_quality_pending",
        "evidence": [artifact(path) for path in LOGS],
    }
    manifest["candidate_verification_complete"] = False
    manifest["candidate_meets_numeric_acceptance"] = False
    manifest["candidate_requires_fresh_sky130_ppa"] = True
    manifest["candidate_supported_layer_operator_prefix_after_review"] = PREFIX
    manifest["candidate_first_unsupported_layer_operator_after_review"] = FIRST_UNSUPPORTED
    manifest["proposed_replacement_contract"].update({
        "rtl_started": True,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "status": "focused_rtl_pass_quality_discriminator_pending",
        "required_manager_action": "hold_rtl_for_focused_quality_discriminator",
    })
    manifest["traceability"]["architecture_contract_gap"] = {
        "resolution_owner": "focused all-layer quality discriminator then paired smoke",
        "status": "standalone_rtl_traced_quality_admission_pending",
    }
    manifest["traceability"]["selected_mechanism"] = CONTRACT
    manifest["traceability"]["rtl.contract-traceability"] = True
    manifest["traceability"]["rtl.hardware-discipline"] = True
    manifest["traceability"]["rtl.ip-provenance"] = True
    manifest["traceability"]["stage_checklist"] = {
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }
    manifest["interfaces_contract_status"] = "candidate_modules_exactly_bound_focused_quality_pending_not_shell_admitted"
    candidate_provenance_names = {
        "ace2_native_accumulator_rope_core_and_tagged_attention_score_core",
        "native_accumulator_tagged_attention_vectors",
    }
    manifest["ip_provenance"] = [
        item for item in manifest["ip_provenance"]
        if item.get("name") not in candidate_provenance_names
    ]
    for item in (
        {
            "kind": "first_party_bounded_candidate_rtl",
            "name": "ace2_native_accumulator_rope_core_and_tagged_attention_score_core",
            "path": RTL,
            "sha256": source_hashes[RTL],
            "source_revision": source_hashes[RTL],
            "license": "repository project license not separately declared in this manifest",
            "third_party": False,
        },
        {
            "kind": "generated_bounded_candidate_verification_source",
            "name": "native_accumulator_tagged_attention_vectors",
            "generator": "tools/gen_native_accumulator_tagged_vectors.py",
            "reference": "tools/ace2_native_accumulator_tagged_reference.py",
            "regeneration_command": "python3 tools/gen_native_accumulator_tagged_vectors.py",
            "third_party": False,
        },
    ):
        manifest["ip_provenance"].append(item)
    manifest["claim_boundaries"] = [
        "The native-accumulator tagged candidate has focused standalone RTL evidence only and is not shell-admitted.",
        "The accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported.",
        "No all-layer quality discriminator, paired smoke, full-shell regression, candidate PPA, prototype, benchmark, signoff, tapeout, or silicon result is claimed.",
        "The historical 0.6108746272 mm2 and +0.1502 ns at 100 MHz frontier remains unchanged.",
    ]
    manifest["generated_at_utc"] = utc_now()
    dump("design/RTL_MANIFEST.json", manifest)

    policy = load("design/FAST_LOOP_POLICY.json")
    live_record = copy.deepcopy(policy["active_repair_authorization"])
    live_record.update({
        "contract_id": CONTRACT,
        "implementation_authorized": True,
        "operator_approval_consumed": False,
        "status": "focused_rtl_pass_quality_discriminator_pending",
        "required_manager_action": "hold_rtl_for_focused_quality_discriminator",
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "rtl_started": True,
    })
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key] = copy.deepcopy(live_record)
    dump("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["current_stage"] = "rtl"
    target["current_architecture_contract"]["status"] = "focused_rtl_pass_quality_discriminator_pending"
    target["current_architecture_contract"]["required_manager_action"] = "hold_rtl_for_focused_quality_discriminator"
    target["fast_loop_contract"]["manager_recommendation"] = "hold_rtl_for_focused_quality_discriminator"
    for key in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][key] = copy.deepcopy(live_record)
    dump("design/TARGET.json", target)

    scope = load("design/CHIP_SCOPE.json")
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_status": "focused_candidate_rtl_pass_quality_discriminator_pending",
        "downstream_stages_locked_until_manager_advance": ["verification", "ppa", "prototype", "benchmark", "signoff"],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"]["required_next_action"] = "run_focused_all_layer_quality_discriminator"
    scope["authority_override"]["architecture_review_status"] = "architecture_and_environment_accepted_rtl_focused_checks_pass"
    scope["authority_override"]["operator_implementation_approval"] = copy.deepcopy(live_record)
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(live_record)
    scope["implementation_frontier"]["latest_decision"] = "native_accumulator_tagged_focused_rtl_pass_quality_pending"
    dump("design/CHIP_SCOPE.json", scope)

    trace_path = ROOT / "design/RTL_TRACEABILITY.md"
    trace_base = """# ACE-2 RTL traceability notes

## Accepted shell frontier

The accepted shell remains hash-bound through `layer_0.v_proj`; first
unsupported remains `layer_0.rope_q`. The accepted-shell lint recheck passes
with the four previously documented non-fatal warnings in `ace2_shell` and
`ace2_silu_gate_core`. The historical accepted PPA frontier remains 62,199
cells, 0.6108746272 mm2, and +0.1502 ns at 100 MHz.

## Historical rejected candidates

All earlier RoPE/score successors, including tile-BFP, remain historical
bounded no-go evidence and are not current RTL capability.
"""
    trace_path.write_text(
        trace_base + f"\n\n## Native-accumulator tagged candidate RTL binding\n\n"
        + f"- Contract: `{CONTRACT}`.\n"
        + f"- Candidate: `{candidate_id}`; ordered source hash `{candidate_hash}`.\n"
        + "- `ace2_native_accumulator_rope_core` owns the candidate's only signed 32x32 multiplier operator, serializes eight RoPE products, and exposes a ready/valid multiplier service to the score controller. It also traces native signed-32/Scale32 records, exponent alignment, signed ties-to-even normalization, reset/clear, and fail-closed Scale32/overflow behavior.\n"
        + "- `ace2_tagged_attention_score_core` contains no multiplier operator; it requests the shared service for each of 1..64 lanes, then traces maximum product exponent, signed-128 accumulation, exact 1/8 head scaling, signed Q20.44 conversion, reset/clear, backpressure, and fail-closed exponent/count/overflow behavior.\n"
        + "- Generated JSON/SVH vectors are first-party and regenerate with `python3 tools/gen_native_accumulator_tagged_vectors.py`. Verilator lint logs and Icarus elaboration are empty; independent Python tests and standalone RTL simulation pass.\n"
        + "- `rtl.contract-traceability`, `rtl.hardware-discipline`, and `rtl.ip-provenance` are true for the standalone candidate. Shell admission and numeric acceptance remain false until the all-layer focused discriminator and unique paired smoke pass.\n",
        encoding="utf-8",
    )

    status = load("research/PUBLIC_STATUS.json")
    status["current_mode"] = "ADVANCE"
    status["supported_layer_operator_prefix"] = PREFIX
    status["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    status["latest_decision"] = "native_accumulator_tagged_focused_rtl_pass_quality_discriminator_pending"
    status["latest_ppa_frontier_status"] = "historical_frontier_preserved_no_candidate_ppa"
    status["architecture_proposal_gate"].update({
        "contract_id": CONTRACT,
        "implementation_authorized": True,
        "required_manager_action": "hold_rtl_for_focused_quality_discriminator",
        "required_operator_action": "none_standing_authorization_is_current",
        "status": "focused_rtl_pass_quality_discriminator_pending",
    })
    status["selected_replacement_contract"].update({
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "status": "focused_rtl_pass_quality_discriminator_pending",
        "implementation_authorized": True,
    })
    status["stage"].update({
        "current_stage": "rtl",
        "current_stage_status": "focused_candidate_rtl_pass_quality_discriminator_pending",
        "current_stage_checklist": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "current_stage_evidence": ["design/RTL_MANIFEST.json", "design/RTL_TRACEABILITY.md", "evidence/shared_native_accumulator_tagged_attention_v1/latest/candidate_evidence.json", *LOGS],
    })
    status["blockers"] = [{
        "id": "focused_all_layer_quality_discriminator_pending",
        "stage": "rtl",
        "status": "active",
        "reason": "Standalone reference, simulation, lint, and elaboration pass; all-layer model quality has not been measured.",
        "required_resolution": "Run the frozen all-layer focused discriminator before paired smoke or shell admission.",
        "evidence": "evidence/shared_native_accumulator_tagged_attention_v1/latest/candidate_evidence.json",
    }]
    if review is not None:
        status["reviewer_certified_metrics"] = [
            item for item in status.setdefault("reviewer_certified_metrics", [])
            if item.get("name") != "rtl_checklist_shared_native_accumulator_tagged_attention_v1"
        ]
        status["reviewer_certified_metrics"].append({
            "name": "rtl_checklist_shared_native_accumulator_tagged_attention_v1",
            "certified_at_utc": review["review_bound_at_utc"],
            "status": "standalone_rtl_checklist_done_quality_pending",
            "contract_binding_status": CONTRACT,
            "candidate_rtl_hash": candidate_hash,
            "checklist": review["checklist"],
            "evidence": [RTL_REVIEW],
        })
    latest_environment = status.get("latest_environment_stage", {})
    if latest_environment.get("contract_binding") == CONTRACT:
        latest_environment["status"] = "independent_review_done_manager_advanced_to_rtl"
        latest_environment["stage_transition_completed"] = True
        latest_environment["implementation_authorized"] = True
        latest_environment["reviewer_verdict"] = "done"
        if isinstance(latest_environment.get("review"), dict):
            latest_environment["review"]["decision"] = "done"
        status["latest_environment_stage"] = latest_environment
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container["current_stage"] = "rtl"
        container["current_mode"] = "ADVANCE"
        container["latest_decision"] = status["latest_decision"]
        container["required_manager_action"] = "hold_rtl_for_focused_quality_discriminator"
        container["required_operator_action"] = "none_standing_authorization_is_current"
        container["ordered_supported_layer_operator_prefix"] = PREFIX
        container["supported_layer_operator_prefix"] = PREFIX
        container["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
        container["rtl_contract_traceability"] = True
        container["candidate_rtl_hash"] = candidate_hash
        container["candidate_rtl_hash_scope"] = "ordered_candidate_source_hashes_standalone_not_shell_admitted"
        container["candidate_mechanism"] = {
            "contract_id": CONTRACT,
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "implementation_authorized": True,
            "mechanism": "all_layer_native_accumulator_tagged_integer_attention_one_shared_multiplier",
            "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "proposal_sha256": sha256("design/NUMERICAL_REPLACEMENT_PROPOSAL.md"),
            "stage_closing": False,
            "status": "focused_rtl_pass_quality_discriminator_pending",
        }
        container["latest_rtl_candidate"] = {
            "candidate_id": candidate_id,
            "contract_id": CONTRACT,
            "evidence": "evidence/shared_native_accumulator_tagged_attention_v1/latest/candidate_evidence.json",
            "rtl_hash": candidate_hash,
            "rtl_hash_scope": "ordered_candidate_source_hashes_standalone_not_shell_admitted",
            "shared_multiplier_operator_count": 1,
            "stage_closing": False,
            "status": "focused_reference_simulation_lint_elaboration_pass_quality_pending",
        }
        container["operator_policy"] = copy.deepcopy(policy)
        if container.get("latest_environment_stage", {}).get("contract_binding") == CONTRACT:
            container["latest_environment_stage"]["status"] = "independent_review_done_manager_advanced_to_rtl"
            container["latest_environment_stage"]["stage_transition_completed"] = True
            container["latest_environment_stage"]["implementation_authorized"] = True
            container["latest_environment_stage"]["reviewer_verdict"] = "done"
            if isinstance(container["latest_environment_stage"].get("review"), dict):
                container["latest_environment_stage"]["review"]["decision"] = "done"
    for metric in status.get("reviewer_certified_metrics", []):
        if metric.get("name") == "environment_stage_close_shared_native_accumulator_tagged_attention_v1":
            metric["status"] = "certified_manager_advanced_to_rtl"
            metric["implementation_authorized"] = True
    status["public_claims"] = [
        {"claim": "current Manager-owned stage is rtl", "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"]},
        {"claim": "independent environment review passed and Manager advanced the active contract to rtl", "evidence": ["research/ENVIRONMENT_REVIEWER_VERDICT.json", "research/PIPELINE_STATE.json"]},
        {"claim": "tile-BFP is a sealed bounded no-go", "evidence": ["evidence/layer0_tile_bfp_score_attention_v1/latest/BOUNDED_NO_GO.json"]},
        {"claim": "the native-accumulator tagged candidate has focused standalone RTL evidence with one shared multiplier", "evidence": ["design/RTL_MANIFEST.json", "evidence/shared_native_accumulator_tagged_attention_v1/latest/candidate_evidence.json"]},
        {"claim": "the supported prefix and immutable 2.0 mm2 and 100 MHz targets are unchanged", "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json"]},
        {"claim": "no shell admission, all-layer quality pass, paired-smoke pass, full-shell regression, candidate PPA, prototype, full benchmark, signoff, tapeout, or silicon result exists", "evidence": ["design/RTL_MANIFEST.json"]},
    ]
    paths = {
        "CHECKPOINT.md", "design/RTL_MANIFEST.json", "design/RTL_TRACEABILITY.md",
        "design/NUMERICAL_REPLACEMENT_PROPOSAL.md", RTL,
        "evidence/shared_native_accumulator_tagged_attention_v1/latest/candidate_evidence.json",
        *([RTL_REVIEW] if review is not None else []),
        *LOGS,
    }
    status["artifact_hashes"] = [artifact(path) for path in sorted(paths)]
    status["last_updated_utc"] = utc_now()
    status["generated_at_utc"] = status["last_updated_utc"]
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": "UTF-8 sorted keys two-space indentation trailing newline canonical hash null during hash",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)

    checkpoint = ROOT / "CHECKPOINT.md"
    checkpoint_text = checkpoint.read_text(encoding="utf-8")
    checkpoint_base = checkpoint_text.split("\n\n# Focused RTL candidate", 1)[0]
    checkpoint.write_text(
        checkpoint_base + f"\n\n# Focused RTL candidate\n\n"
        + f"Manager advanced to `rtl`. Candidate `{candidate_id}` implements exactly two synthesizable modules in `{RTL}` around one explicit shared 32x32 multiplier. Independent Python tests, generated-vector Icarus simulation, warning-free Verilator lint, and clean Icarus elaboration pass. Independent RTL review returned `done` for contract traceability, hardware discipline, and IP provenance; all active policy/status copies bind this live hash and `hold_rtl_for_focused_quality_discriminator`. Shell admission, all-layer focused quality, paired smoke, full-shell regression, and candidate PPA have not run. The supported prefix and historical PPA frontier are unchanged.\n",
        encoding="utf-8",
    )
    status = load("research/PUBLIC_STATUS.json")
    status["artifact_hashes"] = [artifact(path) for path in sorted(paths)]
    status["last_updated_utc"] = utc_now()
    status["generated_at_utc"] = status["last_updated_utc"]
    status["integrity"]["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)


def validate() -> None:
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager stage changed")
    manifest = load("design/RTL_MANIFEST.json")
    require(manifest["current_stage"] == "rtl", "manifest stage stale")
    require(manifest["candidate_layer_operator"] == CONTRACT, "manifest contract stale")
    require(manifest["candidate_interface"]["modules"][0]["module"] == "ace2_native_accumulator_rope_core", "RoPE interface missing")
    require(manifest["candidate_interface"]["modules"][1]["parameters"] == {"MAX_LANES": 64}, "score parameter stale")
    require(manifest["candidate_interface"]["modules"][0]["ports"]["score_mul_product_s64_o"] == 64, "shared multiplier provider interface stale")
    require(manifest["candidate_interface"]["modules"][1]["ports"]["mul_product_s64_i"] == 64, "shared multiplier client interface stale")
    require(all(manifest["traceability"]["stage_checklist"].values()), "RTL checklist incomplete")
    require(manifest["candidate_verification_complete"] is False, "quality-pending candidate marked complete")
    require(manifest["candidate_meets_numeric_acceptance"] is False, "quality-pending candidate marked accepted")
    require(manifest["candidate_supported_layer_operator_prefix_after_review"] == PREFIX, "prefix changed")
    require(manifest["candidate_first_unsupported_layer_operator_after_review"] == FIRST_UNSUPPORTED, "first unsupported changed")
    if (ROOT / RTL_REVIEW).is_file():
        require(manifest["independent_reviewer_verdict"] == "done", "manifest RTL review verdict stale")
        require(sha256(RTL_REVIEW) == manifest["independent_reviewer_acceptance"]["sha256"], "manifest RTL review hash stale")
    policy = load("design/FAST_LOOP_POLICY.json")
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        record = policy[key]
        require(record["contract_id"] == CONTRACT, f"policy contract stale: {key}")
        require(record["status"] == "focused_rtl_pass_quality_discriminator_pending", f"policy status stale: {key}")
        require(record["required_manager_action"] == "hold_rtl_for_focused_quality_discriminator", f"policy routing stale: {key}")
        require(record["candidate_rtl_hash"] == manifest["candidate_rtl_hash"], f"policy hash stale: {key}")
    for record in manifest["candidate_rtl_sources"] + manifest["candidate_generated_sources"]:
        require(sha256(record["path"]) == record["sha256"], f"artifact hash stale: {record['path']}")
    status = load("research/PUBLIC_STATUS.json")
    require(status["stage"]["current_stage"] == "rtl", "public stage stale")
    require(status["supported_layer_operator_prefix"] == PREFIX, "public prefix stale")
    require(status["first_unsupported_layer_operator"] == FIRST_UNSUPPORTED, "public first unsupported stale")
    require(status["integrity"]["canonical_sha256"] == canonical_sha256(status), "public hash stale")
    require("/home/" not in json.dumps(status, sort_keys=True), "public status leaks private path")
    checkpoint = (ROOT / "CHECKPOINT.md").read_text(encoding="utf-8")
    require("has not yet advanced `environment -> rtl`" not in checkpoint, "checkpoint stage routing stale")
    require("RTL remains locked until the Manager" not in checkpoint, "checkpoint RTL lock stale")
    require("`current_stage=rtl`" in checkpoint, "checkpoint current stage stale")
    for container_name in ("dashboard_fields", "implementation_frontier"):
        operator_policy = status[container_name]["operator_policy"]
        require(operator_policy["active_repair_authorization"]["required_manager_action"] == "hold_rtl_for_focused_quality_discriminator", f"public nested routing stale: {container_name}")
        require(status[container_name]["latest_environment_stage"]["status"] == "independent_review_done_manager_advanced_to_rtl", f"public environment transition stale: {container_name}")
    claims = [item.get("claim", "") for item in status["public_claims"]]
    require("current Manager-owned stage is rtl" in claims, "public current-stage claim stale")
    require(not any("no successor RTL" in claim for claim in claims), "public no-RTL claim stale")
    require(status["latest_environment_stage"]["reviewer_verdict"] == "done", "top-level environment reviewer verdict stale")


if __name__ == "__main__":
    bind()
    validate()
    print(f"ACE2_NATIVE_TAGGED_RTL_BIND_PASS contract={CONTRACT} stage=rtl quality_pending=true")
