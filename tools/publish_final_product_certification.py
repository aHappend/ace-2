#!/usr/bin/env python3
"""Atomically publish the final ACE-2 demonstrated-scope product certificate.

This tool is metadata-only. It validates sealed functional, repair, exactly-once
SKY130, and Fresh Reviewer PPA evidence; it never invokes an engineering flow
or mutates engineering evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "design/PPA_FRONTIER_LEDGER.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
MANIFEST_COMPANION = ROOT / "design/RTL_MANIFEST.sha256"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
TRACEABILITY_COMPANION = ROOT / "design/RTL_TRACEABILITY.sha256"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
PIPELINE_COMPANION = ROOT / "research/PIPELINE_STATE.sha256"
CERTIFICATE = ROOT / "research/FINAL_PRODUCT_CERTIFICATE.json"
CERTIFICATE_COMPANION = ROOT / "research/FINAL_PRODUCT_CERTIFICATE.sha256"
INTEGRITY_CHECKER = ROOT / "tools/check_publication_integrity.py"
INTEGRITY_ORDERING_REGRESSION = ROOT / "tools/test_publication_integrity_ordering.py"

PPA_REL = "evidence/canonical_sky130/rtl-final-sumsq-repaired-tree-canonical-sky130-ppa-v1"
PPA_DIR = ROOT / PPA_REL
REPAIR_REL = "evidence/verification/rtl-rmsnorm-final-sumsq-carry-cone-repair-v1"
REPAIR_DIR = ROOT / REPAIR_REL
HANDOFF_ROOT = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs")
REPAIR_REVIEW = HANDOFF_ROOT / "rtl-rmsnorm-final-sumsq-carry-cone-repair-v1/round-0002.json"
PPA_REVIEW = HANDOFF_ROOT / "rtl-final-sumsq-repaired-tree-canonical-sky130-ppa-v1/round-0001.json"

PUBLICATION_AT = "2026-08-04T07:26:46Z"
INTEGRITY_CORRECTION_AT = "2026-08-04T07:43:34Z"
PUBLICATION_ID = f"final-product-publication-{PUBLICATION_AT}"
CERTIFICATE_ID = "ace2-qwen2p5-0p5b-w4a8-final-product-certificate-v1"
PUBLICATION_MISSION = "manager-final-formal-product-publication-certification-v1"
PREVIOUS_PUBLICATION_ID = "frontier-publication-2026-08-04T06:00:28Z"
PREVIOUS_TREE = "a42bc8469b929a1cc193fefb45ff630457db9d298b0f611571992a79bacb360a"
TREE = "bf12e2c83b4d569b27bbbc14835ed8d36c39ec4e8820725cc7ad054fd7ffb4f6"
RMSNORM = "b09fe7073fd6509f0ca83d5b0982b1952aa624d4883631690df35c0bfabb014d"
SHELL = "8bf63ef4bb98700e2f3b86ae4faf1883da76d3875e8632f086453cc51d15f091"
CONSTRAINT = "b21f8da39a0bdc5bf4db70b82013efb3d06a686347749c0070f37f37eca9e9ea"
SCHEDULE = "838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002"
IMAGE = "e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb"
MODEL = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

CELLS = 62283
AREA = 0.614082704
AREA_RESERVE = 1.385917296
SLACK = 0.6966
WNS = 0.0
TNS = 0.0

CLAIM = (
    "ACE-2 is complete for the demonstrated scope: functionally complete 24-layer/two-token "
    "Qwen2.5-0.5B W4A8 command integration and canonical mapped SKY130 synthesis/OpenSTA at "
    "100 MHz within the 2.0 mm2 non-SRAM cap."
)
PPA_CLAIM = (
    "The accepted canonical evidence is mapped SKY130 HD synthesis/OpenSTA at TT 25C 1.80V: "
    "62,283 cells, 0.614082704 mm2 non-SRAM area, +0.6966 ns detailed setup slack, WNS 0.00 ns, "
    "TNS 0.00 ns, and a 10.000 ns clock."
)
NON_CLAIMS = [
    "routed_timing",
    "power_signoff",
    "drc_lvs",
    "gds_or_tapeout",
    "silicon_validation",
    "longer_than_two_token_generation",
    "external_deployment_interfaces",
    "fpga_prototype",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def encode_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, logical_path: str | None = None) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": logical_path or path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def virtual_artifact(content: bytes, logical_path: str) -> dict[str, Any]:
    return {"bytes": len(content), "path": logical_path, "sha256": sha256_bytes(content)}


def reviewer_artifact(path: Path, logical_path: str) -> dict[str, Any]:
    record = artifact(path, logical_path)
    record.update({"producer_role": "reviewer", "status": "done"})
    return record


def canonical_public_hash(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    return sha256_bytes(encode_json(candidate))


def build_integrity_guard() -> dict[str, Any]:
    return {
        "checker": artifact(INTEGRITY_CHECKER),
        "historical_consumed_runner_policy": {
            "hash_locked_pre_guard_runner_count": 5,
            "sealed_evidence_mutated": False,
            "status": "historical_consumed_runners_exempt_by_exact_path_hash_and_marker_only",
        },
        "makefile": artifact(ROOT / "Makefile"),
        "ordering": "integrity_check_before_exactly_once_authorization_marker_creation",
        "ordering_regression": artifact(INTEGRITY_ORDERING_REGRESSION),
        "pre_marker_target": "publication-authorization-preflight",
        "schema_version": 1,
        "status": "PASS",
    }


def replace_artifact_records(value: Any, logical_path: str, replacement: dict[str, Any]) -> None:
    if isinstance(value, dict):
        if value.get("path") == logical_path and "sha256" in value:
            value.clear()
            value.update(copy.deepcopy(replacement))
            return
        for child in value.values():
            replace_artifact_records(child, logical_path, replacement)
    elif isinstance(value, list):
        for child in value:
            replace_artifact_records(child, logical_path, replacement)


def verify_sha256_manifest(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    require(manifest.is_file(), f"missing aggregate manifest {manifest}")
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        digest, relative = raw.split(None, 1)
        relative = relative.strip().lstrip("*")
        if relative.startswith("./"):
            relative = relative[2:]
        target = ROOT / relative if relative.startswith("evidence/") else directory / relative
        require(target.is_file(), f"{manifest}:{number} missing {relative}")
        require(sha256_file(target) == digest, f"{manifest}:{number} hash mismatch for {relative}")


def verify_tree_manifest() -> None:
    tree_manifest = REPAIR_DIR / "rtl_tree_manifest.sha256"
    require(sha256_file(tree_manifest) == TREE, "sealed final RTL-tree manifest digest differs")
    for number, raw in enumerate(tree_manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        digest, relative = raw.split(None, 1)
        target = ROOT / relative.strip()
        require(target.is_file(), f"live RTL path missing at tree line {number}: {relative}")
        require(sha256_file(target) == digest, f"live RTL hash mismatch at tree line {number}: {relative}")
    require(sha256_file(ROOT / "rtl/ace2_rmsnorm_core.sv") == RMSNORM, "live RMSNorm source hash differs")
    require(sha256_file(ROOT / "rtl/ace2_shell.sv") == SHELL, "live shell source hash differs")
    require(sha256_file(ROOT / "constraints/ace2_rmsnorm_core.sdc") == CONSTRAINT, "constraint hash differs")


def verify_sealed_inputs() -> None:
    verify_sha256_manifest(REPAIR_DIR)
    verify_sha256_manifest(PPA_DIR)
    verify_tree_manifest()

    ppa_companion = (PPA_DIR / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()
    require(ppa_companion and ppa_companion[0] == sha256_file(PPA_DIR / "SHA256SUMS"),
            "canonical PPA SHA256SUMS companion differs")

    result = load_json(PPA_DIR / "RESULT.json")
    metrics = load_json(PPA_DIR / "METRICS.json")
    independent = load_json(PPA_DIR / "INDEPENDENT_METRICS.json")
    marker = load_json(PPA_DIR / "authorization_consumed/CONSUMPTION_MARKER.json")
    repair_review = load_json(REPAIR_REVIEW)
    ppa_review = load_json(PPA_REVIEW)

    require(result.get("make_exit_status") == 0 and result.get("protected_state") == "MATCH",
            "canonical PPA result did not complete with protected state MATCH")
    require(result.get("explicit_100mhz_gate") == "PASS", "canonical PPA gate is not PASS")
    require(marker.get("exactly_once_consumed") is True, "PPA authorization was not consumed exactly once")
    require(marker.get("locked_rtl_tree_sha256") == TREE, "PPA marker tree differs")
    require(marker.get("locked_ace2_rmsnorm_core_sha256") == RMSNORM, "PPA marker RMSNorm hash differs")
    require(marker.get("locked_ace2_shell_sha256") == SHELL, "PPA marker shell hash differs")
    require(marker.get("locked_sdc_sha256") == CONSTRAINT, "PPA marker constraint differs")
    require(repair_review.get("producer_role") == "reviewer" and repair_review.get("review", {}).get("status") == "done",
            "final repair Fresh Reviewer acceptance is missing")
    require(ppa_review.get("producer_role") == "reviewer" and ppa_review.get("review", {}).get("status") == "done",
            "canonical PPA Fresh Reviewer acceptance is missing")

    expected = {
        "top_level_cell_count": CELLS,
        "non_sram_area_mm2": "0.614082704",
        "worst_setup_slack_ns": "0.6966",
        "wns_ns": "0.00",
        "tns_ns": "0.00",
        "timing_100mhz_status": "PASS",
        "area_cap_status": "PASS",
        "canonical_gate_verdict": "PASS",
    }
    for key, value in expected.items():
        require(metrics.get(key) == value, f"METRICS.json {key} differs")
        require(independent.get(key) == value, f"INDEPENDENT_METRICS.json {key} differs")
    require(independent.get("reconciliation", {}).get("all_required_metrics_match") is True,
            "independent metric reconciliation differs")
    require((PPA_DIR / "preflight/protected_state_pre.sha256").read_bytes()
            == (PPA_DIR / "postflight/protected_state_post.sha256").read_bytes(),
            "canonical PPA protected state changed")


def verify_functional_authority(full: dict[str, Any]) -> None:
    runtime = full.get("runtime", {})
    layer0 = full.get("layer0", {})
    require(layer0.get("accepted_operator_count") == 18 and layer0.get("total_operator_count") == 18,
            "Layer-0 accepted operator count differs")
    require(layer0.get("result") == "PASS_EXACT" and layer0.get("first_unsupported_layer_operator") is None,
            "Layer-0 exact acceptance differs")
    require(runtime.get("commands_passed") == 13914 and runtime.get("commands_total") == 13914,
            "full-Qwen command count differs")
    require(runtime.get("layers") == 24 and runtime.get("token_steps") == 2,
            "full-Qwen layer/token scope differs")
    require(runtime.get("generated_tokens") == [0, 0] and runtime.get("first_failure") is None,
            "full-Qwen output or failure status differs")
    require(runtime.get("simulator_cycles") == 1240410384, "full-Qwen cycle count differs")
    require(runtime.get("schedule_sha256") == SCHEDULE, "schedule hash differs")
    require(runtime.get("image_sha256") == IMAGE, "image hash differs")
    require(runtime.get("raw_model_sha256") == MODEL, "raw model hash differs")


def build_repair() -> dict[str, Any]:
    return {
        "aggregate_manifest": artifact(REPAIR_DIR / "SHA256SUMS"),
        "changed_rtl_paths": ["rtl/ace2_rmsnorm_core.sv"],
        "fresh_reviewer_binding": reviewer_artifact(
            REPAIR_REVIEW, "handoff:rtl-rmsnorm-final-sumsq-carry-cone-repair-v1/round-0002.json"
        ),
        "from_rtl_tree_sha256": PREVIOUS_TREE,
        "mission_id": "rtl-rmsnorm-final-sumsq-carry-cone-repair-v1",
        "results": artifact(REPAIR_DIR / "RESULTS.txt"),
        "rmsnorm_rtl_sha256": RMSNORM,
        "rtl_tree_manifest": artifact(REPAIR_DIR / "rtl_tree_manifest.sha256"),
        "rtl_tree_sha256": TREE,
        "scope": "insert_ST_MEAN_PRELOAD_to_cut_final_sumsq_carry_cone_with_one_permitted_internal_cycle",
        "source_to_netlist_attribution": artifact(REPAIR_DIR / "source_to_netlist_attribution.md"),
        "to_rtl_tree_sha256": TREE,
    }


def build_ppa(repair: dict[str, Any]) -> dict[str, Any]:
    metrics = load_json(PPA_DIR / "METRICS.json")
    return {
        "aggregate_manifest": artifact(PPA_DIR / "SHA256SUMS"),
        "aggregate_manifest_companion": artifact(PPA_DIR / "SHA256SUMS.sha256"),
        "area_cap_status": "PASS",
        "authorization_consumption": artifact(PPA_DIR / "authorization_consumed/CONSUMPTION_MARKER.json"),
        "canonical_gate_verdict": "PASS",
        "claim_boundary": PPA_CLAIM + " This is not routed or physical-design signoff timing.",
        "clock_period_ns": 10.0,
        "constraint_sha256": CONSTRAINT,
        "corner": "SKY130 HD TT 25C / 1.80V",
        "critical_path": {
            "data_arrival_ns": float(metrics["critical_path"]["data_arrival_ns"]),
            "data_required_ns": float(metrics["critical_path"]["data_required_ns"]),
            "endpoint": metrics["critical_path"]["endpoint"],
            "path_group": metrics["critical_path"]["path_group"],
            "path_type": metrics["critical_path"]["path_type"],
            "reported_slack_ns": SLACK,
            "startpoint": metrics["critical_path"]["startpoint"],
        },
        "evidence_directory": f"{PPA_REL}/",
        "exactly_once_consumed": True,
        "final_metrics": artifact(PPA_DIR / "METRICS.json"),
        "frequency_target_mhz": 100.0,
        "fresh_reviewer_binding": reviewer_artifact(
            PPA_REVIEW, "handoff:rtl-final-sumsq-repaired-tree-canonical-sky130-ppa-v1/round-0001.json"
        ),
        "independent_metrics": artifact(PPA_DIR / "INDEPENDENT_METRICS.json"),
        "mapped_synthesis_opensta_100mhz_certified": True,
        "mission_id": "rtl-final-sumsq-repaired-tree-canonical-sky130-ppa-v1",
        "non_sram_area_cap_mm2": 2.0,
        "non_sram_area_mm2": AREA,
        "product_timing_certified": True,
        "remaining_area_reserve_mm2": AREA_RESERVE,
        "result": artifact(PPA_DIR / "RESULT.json"),
        "rmsnorm_rtl_sha256": RMSNORM,
        "rtl_tree_sha256": TREE,
        "scope": "mapped_sky130_synthesis_and_opensta_only",
        "source_repair": repair,
        "timing_100mhz_status": "PASS",
        "tns_ns": TNS,
        "top_level_cell_count": CELLS,
        "wns_ns": WNS,
        "worst_setup_slack_ns": SLACK,
    }


def build_functional(full: dict[str, Any], ppa: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(full)
    verify_functional_authority(value)
    value["canonical_ppa"] = ppa
    value["claim_boundary"] = CLAIM
    value["final_product_certified_for_demonstrated_scope"] = True
    value["final_product_timing_certified"] = True
    value["publication_review_status"] = "PENDING_FRESH_REVIEWER"
    value["rtl_tree_sha256"] = TREE
    value["status"] = "FUNCTIONAL_FULL_QWEN_TWO_TOKEN_INTEGRATION_COMPLETE"
    value["runtime"]["raw_model_revision"] = MODEL_REVISION
    value["layer0"]["publication_rtl_tree_sha256"] = TREE
    value["layer0"]["report_review_status_resolution"] = (
        "sealed_functional_review_plus_accepted_post_runtime_repair_chain_through_final_sumsq_tree"
    )
    repairs = value.setdefault("accepted_post_runtime_rtl_repairs", [])
    if not any(item.get("to_rtl_tree_sha256") == TREE for item in repairs):
        repairs.append(repair)
    return value


def build_certificate(full: dict[str, Any], ppa: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    return {
        "certificate_id": CERTIFICATE_ID,
        "certificate_status": "ISSUED_PENDING_FRESH_REVIEWER",
        "certified_scope": {
            "functional": "24_layer_two_token_qwen2p5_0p5b_w4a8_command_integration",
            "ppa": "mapped_sky130_hd_synthesis_opensta_100mhz_and_non_sram_area_at_tt_25c_1p80v",
        },
        "claim": CLAIM,
        "decision": "FORMAL_PRODUCT_CERTIFICATE_ISSUED_PENDING_FRESH_REVIEWER",
        "explicit_non_claims": NON_CLAIMS,
        "fresh_reviewer": {"requested_verdict": "FORMAL_PRODUCT_CERTIFIED", "status": "PENDING"},
        "functional_authority": full,
        "issued_at_utc": PUBLICATION_AT,
        "metadata_only_publication": True,
        "no_engineering_execution_performed": True,
        "no_successor_enqueued": True,
        "ppa_authority": ppa,
        "product_status": "COMPLETE_FOR_DEMONSTRATED_SCOPE",
        "publication_id": PUBLICATION_ID,
        "publication_mission_id": PUBLICATION_MISSION,
        "publication_surfaces": [
            "design/PPA_FRONTIER_LEDGER.json",
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            "research/PUBLIC_STATUS.json",
            "research/PIPELINE_STATE.json",
        ],
        "repair_authority": repair,
        "schema_version": 1,
    }


def update_manifest_source(manifest: dict[str, Any]) -> None:
    live = ROOT / "rtl/ace2_rmsnorm_core.sv"
    found = False
    for record in manifest.get("source_hashes", []):
        if record.get("path") == "rtl/ace2_rmsnorm_core.sv":
            old = copy.deepcopy(record)
            record.update({"bytes": live.stat().st_size, "sha256": RMSNORM})
            found = True
            historical = manifest.setdefault("historical_source_hashes", [])
            if not any(item.get("path") == old.get("path") and item.get("sha256") == old.get("sha256")
                       for item in historical):
                old["superseded_by_rtl_tree_sha256"] = TREE
                historical.append(old)
    require(found, "manifest RMSNorm source record is missing")


def build_payloads() -> dict[Path, bytes]:
    verify_sealed_inputs()
    ledger = load_json(LEDGER)
    manifest = load_json(MANIFEST)
    public = load_json(PUBLIC_STATUS)
    pipeline = load_json(PIPELINE_STATE)
    trace = TRACEABILITY.read_text(encoding="utf-8")

    require(ledger.get("entries", [])[-1].get("id") == PREVIOUS_PUBLICATION_ID, "ledger predecessor differs")
    require(ledger["entries"][-1].get("rtl_hash") == PREVIOUS_TREE, "ledger predecessor tree differs")
    require(manifest.get("rtl_hash") == PREVIOUS_TREE, "manifest predecessor differs")
    require(public.get("candidate_rtl_hash") == PREVIOUS_TREE, "PUBLIC_STATUS predecessor differs")
    require(pipeline.get("current_publication", {}).get("id") == PREVIOUS_PUBLICATION_ID,
            "PIPELINE_STATE predecessor publication differs")
    require(not CERTIFICATE.exists() and not CERTIFICATE_COMPANION.exists(),
            "canonical final product certificate already exists")
    verify_functional_authority(manifest["full_qwen_integration"])

    for target, companion in [(MANIFEST, MANIFEST_COMPANION), (TRACEABILITY, TRACEABILITY_COMPANION)]:
        expected = f"{sha256_file(target)}  {target.name}\n"
        require(companion.read_text(encoding="utf-8") == expected, f"stale companion: {companion}")
    require(
        PIPELINE_COMPANION.read_text(encoding="utf-8")
        != f"{sha256_file(PIPELINE_STATE)}  {PIPELINE_STATE.name}\n",
        "PIPELINE_STATE companion is not the authorized stale pair",
    )

    repair = build_repair()
    ppa = build_ppa(repair)
    full = build_functional(manifest["full_qwen_integration"], ppa, repair)
    certificate_value = build_certificate(full, ppa, repair)
    certificate_bytes = encode_json(certificate_value)
    certificate_companion = f"{sha256_bytes(certificate_bytes)}  {CERTIFICATE.name}\n".encode("utf-8")
    certificate_record = virtual_artifact(certificate_bytes, "research/FINAL_PRODUCT_CERTIFICATE.json")

    entry = copy.deepcopy(ledger["entries"][-1])
    entry.update({
        "area_cap_status": "PASS",
        "canonical_gate_verdict": "PASS",
        "cells": CELLS,
        "claim_boundary": f"{CLAIM} {PPA_CLAIM} Explicit non-claims: {', '.join(NON_CLAIMS)}.",
        "comparison_limitation": (
            "This final publication performed metadata/checksum work only and did not run or mutate "
            "RTL, simulation, formal, synthesis, STA, PPA, runtime, model, image, benchmark, or evidence flows."
        ),
        "decision": "FINAL_PRODUCT_SCOPE_COMPLETE_CERTIFICATE_ISSUED_PENDING_FRESH_REVIEWER",
        "evidence": copy.deepcopy(entry.get("evidence", {})),
        "formal_product_certificate": certificate_record,
        "functional_authority": {
            "full_runtime_commands_passed": 13914,
            "full_runtime_commands_total": 13914,
            "generated_tokens": [0, 0],
            "image_sha256": IMAGE,
            "layer0_commands_passed": 18,
            "layer0_commands_total": 18,
            "raw_model_revision": MODEL_REVISION,
            "raw_model_sha256": MODEL,
            "schedule_sha256": SCHEDULE,
            "simulator_cycles": 1240410384,
            "status": "COMPLETE_FOR_24_LAYER_TWO_TOKEN_SCOPE",
        },
        "id": PUBLICATION_ID,
        "mode": "PRODUCT_COMPLETE",
        "next_action": None,
        "non_sram_area_mm2": AREA,
        "non_sram_area_um2": AREA * 1_000_000,
        "publication_review_status": "PENDING_FRESH_REVIEWER",
        "recorded_at_utc": PUBLICATION_AT,
        "remaining_area_reserve_mm2": AREA_RESERVE,
        "repair_binding": repair,
        "repaired_rmsnorm_sha256": RMSNORM,
        "rtl_file_count": 23,
        "rtl_hash": TREE,
        "sky130_sta": {
            "clock_period_ns": 10.0,
            "critical_path": ppa["critical_path"],
            "tns_ns": TNS,
            "wns_ns": WNS,
            "worst_setup_slack_ns": SLACK,
        },
        "timing_100mhz_status": "PASS",
    })
    entry["evidence"].update({
        "canonical_ppa_aggregate_manifest": ppa["aggregate_manifest"],
        "canonical_ppa_authorization_consumption": ppa["authorization_consumption"],
        "canonical_ppa_evidence_directory": ppa["evidence_directory"],
        "canonical_ppa_final_metrics": ppa["final_metrics"],
        "canonical_ppa_fresh_review": ppa["fresh_reviewer_binding"],
        "canonical_ppa_independent_metrics": ppa["independent_metrics"],
        "canonical_ppa_result": ppa["result"],
        "final_repair": repair,
        "formal_product_certificate": certificate_record,
    })
    ledger["entries"].append(entry)

    manifest.update({
        "candidate_capability_accepted": True,
        "candidate_id": "final_sumsq_repaired_tree_bf12e2c8",
        "candidate_meets_numeric_acceptance": True,
        "candidate_requires_fresh_sky130_ppa": False,
        "candidate_rtl_hash": TREE,
        "candidate_status": "PRODUCT_COMPLETE_FOR_DEMONSTRATED_SCOPE_CERTIFICATE_ISSUED",
        "candidate_verification_complete": True,
        "canonical_ppa": ppa,
        "claim_boundaries": [CLAIM, PPA_CLAIM, "Explicit non-claims: " + ", ".join(NON_CLAIMS) + "."],
        "current_stage": "final_certification",
        "first_unsupported_layer_operator": None,
        "formal_product_certificate": certificate_record,
        "full_qwen_integration": full,
        "generated_at_utc": PUBLICATION_AT,
        "next_action_queue": [],
        "next_product_boundary": None,
        "ppa_evidence_binding": {
            "canonical_ppa": ppa,
            "claim_boundary": CLAIM,
            "formal_product_certificate": certificate_record,
            "functional_full_qwen_integration": full,
            "publication_id": PUBLICATION_ID,
            "publication_mission_id": PUBLICATION_MISSION,
        },
        "publication_review_status": "PENDING_FRESH_REVIEWER",
        "required_manager_action": "none",
        "rtl_hash": TREE,
        "stage": "final_certification",
        "stage_closing": True,
        "status": "PRODUCT_COMPLETE_FOR_DEMONSTRATED_SCOPE_CERTIFICATE_ISSUED_PENDING_FRESH_REVIEWER",
    })
    update_manifest_source(manifest)
    manifest["traceability"].update({
        "final_product_certified_for_demonstrated_scope": True,
        "final_product_timing_certified": True,
        "next_product_boundary": None,
        "selected_mechanism": "final_sumsq_repaired_tree_bf12e2c8",
        "timing_contract_gap": None,
    })
    manifest["latest_evidence"].update({
        "final_product_certificate": certificate_record,
        "final_repair_aggregate_manifest": repair["aggregate_manifest"],
        "final_repair_fresh_review": repair["fresh_reviewer_binding"],
        "final_repair_tree_manifest": repair["rtl_tree_manifest"],
        "final_tree_canonical_ppa_aggregate_manifest": ppa["aggregate_manifest"],
        "final_tree_canonical_ppa_final_metrics": ppa["final_metrics"],
        "final_tree_canonical_ppa_fresh_review": ppa["fresh_reviewer_binding"],
        "final_tree_canonical_ppa_independent_metrics": ppa["independent_metrics"],
        "final_tree_canonical_ppa_result": ppa["result"],
    })
    manifest["frontier_publication_history"].append({
        "canonical_ppa_mission_id": ppa["mission_id"],
        "certification_status": "ISSUED_PENDING_FRESH_REVIEWER",
        "formal_product_certificate": certificate_record,
        "functional_status": "COMPLETE_FOR_24_LAYER_TWO_TOKEN_SCOPE",
        "non_sram_area_mm2": AREA,
        "publication_id": PUBLICATION_ID,
        "rtl_tree_sha256": TREE,
        "successor_task_id": None,
        "timing_100mhz_status": "PASS",
        "tns_ns": TNS,
        "top_level_cell_count": CELLS,
        "wns_ns": WNS,
        "worst_setup_slack_ns": SLACK,
    })

    trace_section = f"""

## Final product publication for the demonstrated mapped-SKY130 scope

Publication `{PUBLICATION_ID}` supersedes the current pointers of
`{PREVIOUS_PUBLICATION_ID}` while preserving that NO_GO record in append-only
history. The accepted final 23-file RTL tree is SHA-256 `{TREE}`. Its final
bounded repair is `rtl-rmsnorm-final-sumsq-carry-cone-repair-v1`, bound by
`{REPAIR_REL}/rtl_tree_manifest.sha256` (SHA-256 `{TREE}`), aggregate
`SHA256SUMS` (SHA-256 `{repair['aggregate_manifest']['sha256']}`), and Fresh
Reviewer handoff `handoff:rtl-rmsnorm-final-sumsq-carry-cone-repair-v1/round-0002.json`
(SHA-256 `{repair['fresh_reviewer_binding']['sha256']}`).

Functional authority is Layer-0 18/18 exact PASS plus full-Qwen 13,914/13,914
commands across 24 layers and two token steps, generated tokens `[0, 0]`, no
first failure, and 1,240,410,384 cycles. The schedule, image, and raw-model
SHA-256 values are `{SCHEDULE}`, `{IMAGE}`, and `{MODEL}`; the raw model
revision is `{MODEL_REVISION}`.

The exactly-once canonical packet is `{PPA_REL}/`. It reports {CELLS:,} cells,
{AREA:.9f} mm2 non-SRAM area, +{SLACK:.4f} ns detailed setup slack, WNS 0.00 ns,
TNS 0.00 ns, and PASS at 10.000 ns / 100 MHz in SKY130 HD TT 25C 1.80V.
`RESULT.json`, `METRICS.json`, `INDEPENDENT_METRICS.json`, `SHA256SUMS`, and the
authorization-consumption marker hash to `{ppa['result']['sha256']}`,
`{ppa['final_metrics']['sha256']}`, `{ppa['independent_metrics']['sha256']}`,
`{ppa['aggregate_manifest']['sha256']}`, and `{ppa['authorization_consumption']['sha256']}`.
Fresh Reviewer PPA acceptance hashes to `{ppa['fresh_reviewer_binding']['sha256']}`.

The product is complete only for functionally integrated 24-layer/two-token
Qwen command execution plus mapped SKY130 synthesis/OpenSTA at 100 MHz and
within 2.0 mm2 non-SRAM. This does not claim routed timing, power signoff,
DRC/LVS, GDS or tapeout, silicon validation, longer-token generation, FPGA, or
external deployment interfaces. This publication ran no engineering flow,
changed no engineering evidence, enqueued no successor, and issued
`research/FINAL_PRODUCT_CERTIFICATE.json` pending independent Fresh Reviewer
verdict `FORMAL_PRODUCT_CERTIFIED`.
"""
    trace = trace.rstrip() + trace_section

    old_publication = copy.deepcopy(pipeline["current_publication"])
    old_successor = copy.deepcopy(pipeline.get("successor"))
    if isinstance(old_successor, dict):
        old_successor.update({
            "completed_at_utc": PUBLICATION_AT,
            "completed_rtl_tree_sha256": TREE,
            "status": "completed_and_superseded_by_formal_product_publication",
            "superseded_by": PUBLICATION_ID,
        })
        pipeline.setdefault("historical_successors", []).append(old_successor)
    pipeline.setdefault("historical_authoritative_state_reconciliations", []).append(
        copy.deepcopy(pipeline.get("authoritative_state_reconciliation", {}))
    )
    current_publication = {
        "canonical_ppa": ppa,
        "claim_boundary": CLAIM,
        "final_product_certified_for_demonstrated_scope": True,
        "final_product_timing_certified": True,
        "formal_product_certificate": certificate_record,
        "full_qwen_integration": full,
        "id": PUBLICATION_ID,
        "next_action_queue": [],
        "performed_without_engineering_execution": True,
        "product_status": "COMPLETE_FOR_DEMONSTRATED_SCOPE",
        "publication_mission_id": PUBLICATION_MISSION,
        "publication_review_status": "PENDING_FRESH_REVIEWER",
        "repair_binding": repair,
        "rtl_tree_sha256": TREE,
        "status": "certificate_issued_pending_fresh_reviewer",
        "supersedes_publication_id": PREVIOUS_PUBLICATION_ID,
    }
    pipeline["authoritative_state_reconciliation"] = {
        "at": PUBLICATION_AT,
        "canonical_ppa": ppa,
        "first_unsupported_layer_operator": None,
        "formal_product_certificate": certificate_record,
        "full_qwen_integration": full,
        "no_successor_enqueued": True,
        "publication_mission_id": PUBLICATION_MISSION,
        "rtl_tree_sha256": TREE,
        "status": "COMPLETE_FOR_DEMONSTRATED_SCOPE_PENDING_FRESH_REVIEWER",
    }
    pipeline["current_publication"] = current_publication
    pipeline["current_stage"] = "final_certification"
    pipeline["stages"]["ppa"]["status"] = "done_pass_mapped_synthesis_opensta_100mhz"
    pipeline["stages"]["prototype"]["status"] = "not_claimed_outside_demonstrated_scope"
    pipeline["stages"]["benchmark"]["status"] = "not_claimed_outside_demonstrated_scope"
    pipeline["stages"]["signoff"]["status"] = "not_claimed_outside_demonstrated_scope"
    pipeline["successor"] = None
    pipeline["workflow_mode"] = "PRODUCT_COMPLETE"
    pipeline["stage_history"].append({
        "at": PUBLICATION_AT,
        "by": "manager",
        "direction": "publication",
        "from_stage": "ppa",
        "publication_id": PUBLICATION_ID,
        "reason": (
            "Publish the accepted final functional and exactly-once mapped-SKY130 PASS evidence, "
            "issue the demonstrated-scope product certificate, and enqueue no successor."
        ),
        "to_stage": "final_certification",
    })

    previous_history = {
        "canonical_sky130_ppa": {
            "aggregate_sha256": old_publication["canonical_ppa"]["aggregate_manifest"]["sha256"],
            "mission_id": old_publication["canonical_ppa"]["mission_id"],
            "non_sram_area_mm2": old_publication["canonical_ppa"]["non_sram_area_mm2"],
            "timing_100mhz_status": old_publication["canonical_ppa"]["timing_100mhz_status"],
            "tns_ns": old_publication["canonical_ppa"]["tns_ns"],
            "top_level_cell_count": old_publication["canonical_ppa"]["top_level_cell_count"],
            "wns_ns": old_publication["canonical_ppa"]["wns_ns"],
            "worst_setup_slack_ns": old_publication["canonical_ppa"]["worst_setup_slack_ns"],
        },
        "functional_status": "FUNCTIONAL_FULL_QWEN_TWO_TOKEN_INTEGRATION_ACCEPTED",
        "publication_id": PREVIOUS_PUBLICATION_ID,
        "rtl_tree_sha256": PREVIOUS_TREE,
        "status": f"SUPERSEDED_BY_{PUBLICATION_ID}",
        "successor_task_id": "rtl-rmsnorm-final-sumsq-carry-cone-repair-v1",
    }
    public.setdefault("historical_frontier_publications", []).append(previous_history)
    historical_fields = public.setdefault("historical_dashboard_fields", {}).setdefault("fields", {})
    historical_fields["rmsnorm_repaired_tree_current_no_go_before_final_certification"] = copy.deepcopy(
        public.get("terminal_baseline_no_go")
    )
    public.pop("terminal_baseline_no_go", None)
    dashboard = {
        "canonical_ppa": ppa,
        "canonical_sky130_ppa": ppa,
        "current_mode": "PRODUCT_COMPLETE",
        "current_publication": current_publication,
        "current_stage": "final_certification",
        "final_product_certified_for_demonstrated_scope": True,
        "final_product_timing_certified": True,
        "formal_product_certificate": certificate_record,
        "functional_integration": full,
        "headline": "ACE-2 demonstrated scope complete: full-Qwen two-token integration and mapped SKY130 100 MHz/area PASS.",
        "next_action_queue": [],
        "privacy": public.get("dashboard_fields", {}).get("privacy"),
        "product_status": "COMPLETE_FOR_DEMONSTRATED_SCOPE",
        "publication_id": PUBLICATION_ID,
        "publication_review_status": "PENDING_FRESH_REVIEWER",
    }
    public.update({
        "blockers": [],
        "candidate_rtl_hash": TREE,
        "current_mode": "PRODUCT_COMPLETE",
        "current_stage": "final_certification",
        "dashboard_fields": dashboard,
        "explicit_non_claims": NON_CLAIMS,
        "final_product_certificate": certificate_record,
        "first_unsupported_layer_operator": None,
        "generated_at_utc": PUBLICATION_AT,
        "implementation_frontier": {
            "canonical_ppa": ppa,
            "claim_boundary": CLAIM,
            "final_product_certified_for_demonstrated_scope": True,
            "final_product_timing_certified": True,
            "formal_product_certificate": certificate_record,
            "full_qwen_integration": full,
            "next_action_queue": [],
            "rtl_tree_sha256": TREE,
        },
        "last_updated_utc": PUBLICATION_AT,
        "latest_decision": "FORMAL_PRODUCT_CERTIFICATE_ISSUED_PENDING_FRESH_REVIEWER",
        "latest_ppa_frontier": ppa,
        "latest_ppa_frontier_status": "accepted_canonical_final_tree_pass_timing_and_area",
        "latest_rtl_candidate": {
            "candidate_id": "final_sumsq_repaired_tree_bf12e2c8",
            "rmsnorm_rtl_sha256": RMSNORM,
            "rtl_tree_sha256": TREE,
            "status": "accepted_functional_and_canonical_ppa_pass",
            "verification_complete": True,
        },
        "next_action_queue": [],
        "public_claims": [
            {"claim": "Layer-0 passes 18/18 ordered operators exactly.", "evidence": [full["layer0"]["evidence"]["path"]]},
            {"claim": "Full-Qwen passes 13,914/13,914 commands across 24 layers/two tokens with token IDs [0,0].",
             "evidence": [full["runtime"]["runtime_summary"]["path"], full["runtime"]["runtime_validation"]["path"]]},
            {"claim": "Canonical mapped SKY130 synthesis/OpenSTA passes 100 MHz and the 2.0 mm2 non-SRAM cap.",
             "evidence": [ppa["final_metrics"]["path"], ppa["independent_metrics"]["path"]]},
        ],
        "required_manager_action": "none",
        "required_operator_action": "none",
        "required_reviewer_action": "verify_atomic_publication_and_issue_FORMAL_PRODUCT_CERTIFIED_or_one_metadata_only_correction",
        "reviewer_certified_metrics": {
            "canonical_sky130": ppa,
            "functional_integration": {
                "full_runtime_commands_passed": 13914,
                "full_runtime_commands_total": 13914,
                "generated_tokens": [0, 0],
                "layer0_commands_passed": 18,
                "layer0_commands_total": 18,
            },
        },
        "routing_authorized": False,
        "routing_status": "not_claimed_outside_demonstrated_scope",
        "stage": "final_certification",
        "stage_closing": True,
    })

    ledger_bytes = encode_json(ledger)
    manifest_bytes = encode_json(manifest)
    trace_bytes = trace.encode("utf-8")
    pipeline_bytes = encode_json(pipeline)
    manifest_companion = f"{sha256_bytes(manifest_bytes)}  {MANIFEST.name}\n".encode("utf-8")
    trace_companion = f"{sha256_bytes(trace_bytes)}  {TRACEABILITY.name}\n".encode("utf-8")
    pipeline_companion = f"{sha256_bytes(pipeline_bytes)}  {PIPELINE_STATE.name}\n".encode("utf-8")

    records = list(public.get("artifact_hashes", []))
    additions = [
        repair["aggregate_manifest"], repair["rtl_tree_manifest"], repair["results"],
        ppa["aggregate_manifest"], ppa["aggregate_manifest_companion"], ppa["authorization_consumption"],
        ppa["result"], ppa["final_metrics"], ppa["independent_metrics"],
        virtual_artifact(ledger_bytes, "design/PPA_FRONTIER_LEDGER.json"),
        virtual_artifact(manifest_bytes, "design/RTL_MANIFEST.json"),
        virtual_artifact(manifest_companion, "design/RTL_MANIFEST.sha256"),
        virtual_artifact(trace_bytes, "design/RTL_TRACEABILITY.md"),
        virtual_artifact(trace_companion, "design/RTL_TRACEABILITY.sha256"),
        virtual_artifact(pipeline_bytes, "research/PIPELINE_STATE.json"),
        virtual_artifact(pipeline_companion, "research/PIPELINE_STATE.sha256"),
        certificate_record,
        virtual_artifact(certificate_companion, "research/FINAL_PRODUCT_CERTIFICATE.sha256"),
    ]
    by_path = {item["path"]: item for item in records}
    for item in additions:
        by_path[item["path"]] = item
    public["artifact_hashes"] = [by_path[path] for path in sorted(by_path)]
    public["integrity"] = {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None}
    public["integrity"]["canonical_sha256"] = canonical_public_hash(public)
    public_bytes = encode_json(public)

    return {
        LEDGER: ledger_bytes,
        MANIFEST: manifest_bytes,
        MANIFEST_COMPANION: manifest_companion,
        TRACEABILITY: trace_bytes,
        TRACEABILITY_COMPANION: trace_companion,
        PIPELINE_STATE: pipeline_bytes,
        PIPELINE_COMPANION: pipeline_companion,
        CERTIFICATE: certificate_bytes,
        CERTIFICATE_COMPANION: certificate_companion,
        PUBLIC_STATUS: public_bytes,
    }


def atomic_commit(payloads: dict[Path, bytes]) -> None:
    staged: dict[Path, Path] = {}
    try:
        for path, content in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged[path] = Path(name)
        for path in payloads:
            if path == PUBLIC_STATUS:
                continue
            os.replace(staged[path], path)
        os.replace(staged[PUBLIC_STATUS], PUBLIC_STATUS)
        for directory in {path.parent for path in payloads}:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        for path in staged.values():
            if path.exists():
                path.unlink()


def build_integrity_correction_payloads() -> dict[Path, bytes]:
    verify_sealed_inputs()
    ledger = load_json(LEDGER)
    manifest = load_json(MANIFEST)
    public = load_json(PUBLIC_STATUS)
    pipeline = load_json(PIPELINE_STATE)
    certificate = load_json(CERTIFICATE)
    trace = TRACEABILITY.read_text(encoding="utf-8")

    require(ledger.get("entries", [])[-1].get("id") == PUBLICATION_ID, "ledger final publication differs")
    require(manifest.get("rtl_hash") == TREE, "manifest final tree differs before integrity correction")
    require(pipeline.get("current_publication", {}).get("id") == PUBLICATION_ID,
            "PIPELINE_STATE final publication differs before integrity correction")
    require(certificate.get("certificate_id") == CERTIFICATE_ID, "canonical certificate differs")
    require(
        certificate.get("publication_integrity_guard") is None
        or isinstance(certificate.get("publication_integrity_guard"), dict),
        "publication integrity correction has an invalid existing guard",
    )
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_public_hash(public),
            "PUBLIC_STATUS integrity differs before correction")
    for target, companion in [
        (MANIFEST, MANIFEST_COMPANION),
        (TRACEABILITY, TRACEABILITY_COMPANION),
        (PIPELINE_STATE, PIPELINE_COMPANION),
        (CERTIFICATE, CERTIFICATE_COMPANION),
    ]:
        verify_companion(target, companion)

    guard = build_integrity_guard()
    certificate["publication_integrity_guard"] = copy.deepcopy(guard)
    certificate_bytes = encode_json(certificate)
    certificate_companion = f"{sha256_bytes(certificate_bytes)}  {CERTIFICATE.name}\n".encode("utf-8")
    certificate_record = virtual_artifact(certificate_bytes, "research/FINAL_PRODUCT_CERTIFICATE.json")

    for surface in (ledger, manifest, public, pipeline):
        replace_artifact_records(surface, "research/FINAL_PRODUCT_CERTIFICATE.json", certificate_record)

    ledger["entries"][-1]["publication_integrity_guard"] = copy.deepcopy(guard)

    manifest["publication_integrity_guard"] = copy.deepcopy(guard)
    correction_history = manifest.setdefault("metadata_correction_history", [])
    if not any(item.get("publication_id") == PUBLICATION_ID
               and item.get("at") == INTEGRITY_CORRECTION_AT for item in correction_history):
        correction_history.append({
            "at": INTEGRITY_CORRECTION_AT,
            "publication_id": PUBLICATION_ID,
            "reason": "Enforce publication integrity before future exactly-once marker creation and bind the metadata-only ordering regression.",
            "status": "APPLIED_WITHOUT_ENGINEERING_OR_EVIDENCE_MUTATION",
        })

    correction_heading = "## Publication-integrity ordering correction"
    if correction_heading not in trace:
        trace = trace.rstrip() + f"""

{correction_heading}

The pending publication was metadata-corrected at `{INTEGRITY_CORRECTION_AT}`.
`make publication-authorization-preflight` now runs the companion/integrity
checker and its ordering regression before any future exactly-once runner may
create an authorization marker. The checker hash-locks the five already
consumed historical runners and requires every non-historical runner to place
that guard before marker creation. No RTL, constraints, flow output, canonical
PPA packet, verification evidence, or other engineering evidence was changed.
"""

    pipeline["publication_integrity_guard"] = copy.deepcopy(guard)
    pipeline["current_publication"]["publication_integrity_guard"] = copy.deepcopy(guard)
    pipeline["authoritative_state_reconciliation"]["publication_integrity_guard"] = copy.deepcopy(guard)
    if not any(item.get("direction") == "metadata_integrity_correction"
               and item.get("publication_id") == PUBLICATION_ID for item in pipeline["stage_history"]):
        pipeline["stage_history"].append({
            "at": INTEGRITY_CORRECTION_AT,
            "by": "manager",
            "direction": "metadata_integrity_correction",
            "from_stage": "final_certification",
            "publication_id": PUBLICATION_ID,
            "reason": "Move publication integrity enforcement ahead of future exactly-once marker creation; no engineering execution.",
            "to_stage": "final_certification",
        })

    public["publication_integrity_guard"] = copy.deepcopy(guard)
    public["dashboard_fields"]["publication_integrity_guard"] = copy.deepcopy(guard)
    public["dashboard_fields"]["current_publication"]["publication_integrity_guard"] = copy.deepcopy(guard)
    public["implementation_frontier"]["publication_integrity_guard"] = copy.deepcopy(guard)
    public["last_updated_utc"] = INTEGRITY_CORRECTION_AT

    ledger_bytes = encode_json(ledger)
    manifest_bytes = encode_json(manifest)
    trace_bytes = trace.encode("utf-8")
    pipeline_bytes = encode_json(pipeline)
    manifest_companion = f"{sha256_bytes(manifest_bytes)}  {MANIFEST.name}\n".encode("utf-8")
    trace_companion = f"{sha256_bytes(trace_bytes)}  {TRACEABILITY.name}\n".encode("utf-8")
    pipeline_companion = f"{sha256_bytes(pipeline_bytes)}  {PIPELINE_STATE.name}\n".encode("utf-8")

    records = list(public.get("artifact_hashes", []))
    additions = [
        guard["makefile"],
        guard["checker"],
        guard["ordering_regression"],
        virtual_artifact(ledger_bytes, "design/PPA_FRONTIER_LEDGER.json"),
        virtual_artifact(manifest_bytes, "design/RTL_MANIFEST.json"),
        virtual_artifact(manifest_companion, "design/RTL_MANIFEST.sha256"),
        virtual_artifact(trace_bytes, "design/RTL_TRACEABILITY.md"),
        virtual_artifact(trace_companion, "design/RTL_TRACEABILITY.sha256"),
        virtual_artifact(pipeline_bytes, "research/PIPELINE_STATE.json"),
        virtual_artifact(pipeline_companion, "research/PIPELINE_STATE.sha256"),
        certificate_record,
        virtual_artifact(certificate_companion, "research/FINAL_PRODUCT_CERTIFICATE.sha256"),
    ]
    by_path = {item["path"]: item for item in records}
    for item in additions:
        by_path[item["path"]] = item
    public["artifact_hashes"] = [by_path[path] for path in sorted(by_path)]
    public["integrity"] = {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None}
    public["integrity"]["canonical_sha256"] = canonical_public_hash(public)
    public_bytes = encode_json(public)

    return {
        LEDGER: ledger_bytes,
        MANIFEST: manifest_bytes,
        MANIFEST_COMPANION: manifest_companion,
        TRACEABILITY: trace_bytes,
        TRACEABILITY_COMPANION: trace_companion,
        PIPELINE_STATE: pipeline_bytes,
        PIPELINE_COMPANION: pipeline_companion,
        CERTIFICATE: certificate_bytes,
        CERTIFICATE_COMPANION: certificate_companion,
        PUBLIC_STATUS: public_bytes,
    }


def verify_companion(target: Path, companion: Path) -> None:
    expected = f"{sha256_file(target)}  {target.name}\n"
    require(companion.read_text(encoding="utf-8") == expected, f"stale companion: {companion}")


def verify_artifact_record(record: dict[str, Any]) -> None:
    path = record.get("path")
    if not isinstance(path, str) or path.startswith("handoff:"):
        return
    target = ROOT / path
    require(target.is_file(), f"PUBLIC_STATUS artifact path missing: {path}")
    require(target.stat().st_size == record.get("bytes"), f"PUBLIC_STATUS artifact bytes differ: {path}")
    require(sha256_file(target) == record.get("sha256"), f"PUBLIC_STATUS artifact hash differs: {path}")


def check_published() -> None:
    verify_sealed_inputs()
    ledger = load_json(LEDGER)
    manifest = load_json(MANIFEST)
    public = load_json(PUBLIC_STATUS)
    pipeline = load_json(PIPELINE_STATE)
    certificate = load_json(CERTIFICATE)
    trace = TRACEABILITY.read_text(encoding="utf-8")
    expected_guard = build_integrity_guard()
    certified = certificate.get("certificate_status") == "CERTIFIED"

    entries = ledger.get("entries", [])
    require(entries[-1].get("id") == PUBLICATION_ID and entries[-2].get("id") == PREVIOUS_PUBLICATION_ID,
            "ledger append-only publication tail differs")
    require(sum(item.get("id") == PUBLICATION_ID for item in entries) == 1, "ledger publication is not unique")
    require(entries[-1].get("rtl_hash") == TREE and entries[-1].get("timing_100mhz_status") == "PASS",
            "ledger current PASS binding differs")
    require(entries[-1].get("next_action") is None, "ledger enqueues a successor")
    require(entries[-1].get("publication_integrity_guard") == expected_guard,
            "ledger publication integrity guard differs")

    require(manifest.get("rtl_hash") == TREE and manifest.get("candidate_rtl_hash") == TREE,
            "manifest final tree differs")
    expected_manifest_status = (
        "PRODUCT_COMPLETE_FOR_DEMONSTRATED_SCOPE_CERTIFIED"
        if certified
        else "PRODUCT_COMPLETE_FOR_DEMONSTRATED_SCOPE_CERTIFICATE_ISSUED_PENDING_FRESH_REVIEWER"
    )
    require(manifest.get("status") == expected_manifest_status, "manifest product status differs")
    if certified:
        manifest_certification = manifest.get("final_product_certification", {})
        require(manifest_certification.get("status") == "done", "manifest certification is not closed")
        require(manifest_certification.get("decision") == "FORMAL_PRODUCT_CERTIFIED",
                "manifest certification decision differs")
    require(manifest.get("canonical_ppa", {}).get("timing_100mhz_status") == "PASS",
            "manifest canonical PPA is not PASS")
    require(manifest.get("next_action_queue") == [] and manifest.get("next_product_boundary") is None,
            "manifest retains a successor projection")
    require(manifest.get("first_unsupported_layer_operator") is None, "manifest retains first unsupported operator")
    verify_functional_authority(manifest["full_qwen_integration"])
    require(manifest["full_qwen_integration"]["runtime"].get("raw_model_revision") == MODEL_REVISION,
            "manifest raw model revision differs")
    require(manifest.get("frontier_publication_history", [])[-1].get("publication_id") == PUBLICATION_ID,
            "manifest publication history tail differs")
    require(manifest.get("publication_integrity_guard") == expected_guard,
            "manifest publication integrity guard differs")

    require(trace.count("## Final product publication for the demonstrated mapped-SKY130 scope") == 1,
            "traceability final publication section is missing or duplicated")
    for text in [TREE, "62,283", "0.614082704", "+0.6966", MODEL_REVISION,
                 "does not claim routed timing", "enqueued no successor"]:
        require(text in trace, f"traceability is missing {text}")
    require("## Publication-integrity ordering correction" in trace,
            "traceability publication-integrity correction is missing")

    require(certificate.get("certificate_id") == CERTIFICATE_ID, "certificate id differs")
    require(certificate.get("product_status") == "COMPLETE_FOR_DEMONSTRATED_SCOPE", "certificate status differs")
    if certified:
        require(certificate.get("decision") == "FORMAL_PRODUCT_CERTIFIED",
                "certificate decision differs")
        require(certificate.get("fresh_reviewer", {}).get("status") == "done",
                "certificate Fresh Reviewer status differs")
        require(certificate.get("fresh_reviewer", {}).get("verdict_binding", {}).get("status") == "done",
                "certificate Fresh Reviewer verdict binding is incomplete")
    else:
        require(certificate.get("certificate_status") == "ISSUED_PENDING_FRESH_REVIEWER",
                "pending certificate status differs")
        require(certificate.get("fresh_reviewer", {}).get("status") == "PENDING",
                "pending certificate fabricates Fresh Reviewer acceptance")
    require(certificate.get("no_successor_enqueued") is True, "certificate successor claim differs")
    require(certificate.get("explicit_non_claims") == NON_CLAIMS, "certificate non-claims differ")
    require(certificate.get("publication_integrity_guard") == expected_guard,
            "certificate publication integrity guard differs")

    require(pipeline.get("current_stage") == "final_certification" and pipeline.get("workflow_mode") == "PRODUCT_COMPLETE",
            "PIPELINE_STATE current completion projection differs")
    require(pipeline.get("current_publication", {}).get("id") == PUBLICATION_ID,
            "PIPELINE_STATE current publication differs")
    if certified:
        require(pipeline.get("current_publication", {}).get("status") == "certified",
                "PIPELINE_STATE publication is not certified")
        require(pipeline.get("current_publication", {}).get("publication_review_status") == "done",
                "PIPELINE_STATE publication review is not closed")
        require(pipeline.get("final_product_certification", {}).get("decision") == "FORMAL_PRODUCT_CERTIFIED",
                "PIPELINE_STATE certification decision differs")
    require(pipeline.get("successor") is None, "PIPELINE_STATE enqueues a successor")
    require(pipeline.get("authoritative_state_reconciliation", {}).get("first_unsupported_layer_operator") is None,
            "PIPELINE_STATE retains first unsupported operator")
    require(pipeline.get("stage_history", [])[-1].get("publication_id") == PUBLICATION_ID,
            "PIPELINE_STATE history tail differs")
    require(pipeline.get("current_publication", {}).get("publication_integrity_guard") == expected_guard,
            "PIPELINE_STATE publication integrity guard differs")

    expected_public_decision = (
        "FORMAL_PRODUCT_CERTIFIED"
        if certified
        else "FORMAL_PRODUCT_CERTIFICATE_ISSUED_PENDING_FRESH_REVIEWER"
    )
    require(public.get("candidate_rtl_hash") == TREE and public.get("latest_decision")
            == expected_public_decision,
            "PUBLIC_STATUS current decision differs")
    require(public.get("next_action_queue") == [] and public.get("blockers") == [],
            "PUBLIC_STATUS retains obsolete action/blocker projections")
    require("terminal_baseline_no_go" not in public, "PUBLIC_STATUS retains current terminal NO_GO projection")
    require(public.get("dashboard_fields", {}).get("publication_id") == PUBLICATION_ID,
            "PUBLIC_STATUS dashboard publication differs")
    if certified:
        require(public.get("dashboard_fields", {}).get("current_publication", {}).get("status") == "certified",
                "PUBLIC_STATUS dashboard publication is not certified")
        require(public.get("dashboard_fields", {}).get("current_publication", {}).get("publication_review_status") == "done",
                "PUBLIC_STATUS dashboard publication review is not closed")
    require(public.get("dashboard_fields", {}).get("next_action_queue") == [],
            "PUBLIC_STATUS dashboard retains a successor")
    require(public.get("latest_ppa_frontier", {}).get("timing_100mhz_status") == "PASS",
            "PUBLIC_STATUS current PPA is not PASS")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_public_hash(public),
            "PUBLIC_STATUS canonical integrity differs")
    require(public.get("publication_integrity_guard") == expected_guard,
            "PUBLIC_STATUS publication integrity guard differs")
    paths = [item.get("path") for item in public.get("artifact_hashes", [])]
    require(len(paths) == len(set(paths)), "PUBLIC_STATUS artifact paths are duplicated")
    for record in public.get("artifact_hashes", []):
        verify_artifact_record(record)

    for target, companion in [
        (MANIFEST, MANIFEST_COMPANION),
        (TRACEABILITY, TRACEABILITY_COMPANION),
        (PIPELINE_STATE, PIPELINE_COMPANION),
        (CERTIFICATE, CERTIFICATE_COMPANION),
    ]:
        verify_companion(target, companion)

    print(json.dumps({
        "area_mm2": AREA,
        "cells": CELLS,
        "certificate_id": CERTIFICATE_ID,
        "fresh_reviewer_status": "done" if certified else "PENDING",
        "product_status": (
            "COMPLETE_FOR_DEMONSTRATED_SCOPE_CERTIFIED"
            if certified
            else "COMPLETE_FOR_DEMONSTRATED_SCOPE"
        ),
        "publication_id": PUBLICATION_ID,
        "status": "PASS",
        "timing_100mhz": "PASS",
        "tree": TREE,
        "worst_setup_slack_ns": SLACK,
    }, indent=2, sort_keys=True))


def protected_hashes() -> dict[str, str]:
    paths = [LEDGER, MANIFEST, MANIFEST_COMPANION, TRACEABILITY, TRACEABILITY_COMPANION,
             PUBLIC_STATUS, PIPELINE_STATE, PIPELINE_COMPANION, CERTIFICATE, CERTIFICATE_COMPANION]
    return {path.as_posix(): sha256_file(path) for path in paths if path.exists()}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.apply:
        if CERTIFICATE.exists():
            require(load_json(CERTIFICATE).get("certificate_id") == CERTIFICATE_ID,
                    "a different canonical certificate already exists")
            if load_json(CERTIFICATE).get("publication_integrity_guard") != build_integrity_guard():
                atomic_commit(build_integrity_correction_payloads())
        else:
            atomic_commit(build_payloads())
        check_published()
        return

    before = protected_hashes()
    check_published()
    after = protected_hashes()
    require(before == after, "--check mutated a protected publication surface")


if __name__ == "__main__":
    main()
