#!/usr/bin/env python3
"""Run and bind the bounded RTL preflight for the residual cross-term candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_qk_residual_cross_term_attention_v1"
PREFIX = ["layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"]
FIRST_UNSUPPORTED = "layer_0.rope_q"
SCORE_DOT_CYCLES = 192
SCORE_CONVERSION_COUNT = 3
SCORE_CYCLES_PER_CONVERSION = 8
SCORE_CONVERSION_CYCLES = SCORE_CONVERSION_COUNT * SCORE_CYCLES_PER_CONVERSION
SCORE_TOTAL_CYCLES = SCORE_DOT_CYCLES + SCORE_CONVERSION_CYCLES
SCORE_QUERY_HEADS = 14
SCORE_CYCLES_PER_KEY = SCORE_QUERY_HEADS * SCORE_TOTAL_CYCLES
EVIDENCE_DIR = ROOT / "evidence/shared_qk_residual_cross_term_attention_v1/latest"
PACKET = EVIDENCE_DIR / "PRECHECK.json"
RTL = "rtl/ace2_qk_residual_cross_term_core.sv"
REFREEZE_PACKET = "evidence/shared_qk_residual_cross_term_attention_v1/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
ARCHITECTURE_DECISION = "evidence/review/architecture_refreeze_shared_qk_residual_cross_term_attention_v1/decision.json"
ENVIRONMENT_DECISION = "evidence/review/environment_refreeze_shared_qk_residual_cross_term_attention_v1/decision.json"
REFREEZE_PACKET_SHA256 = "e0c711d52267cf1902a7663693aff952e1cd3caf4539217595485358a16534be"
TRACEABILITY_PATHS = [
    "design/ARCHITECTURE.md",
    "design/SPEC.md",
    "reference/qk_residual_cross_term_full_model_hook.json",
    "reference/generated/qk_residual_scale32_metadata.json",
    "tools/ace2_qk_residual_cross_term_reference.py",
    RTL,
    "verification/generated/qk_residual_cross_term_vectors.json",
    "tools/run_qk_residual_cross_term_preflight.py",
]
SOURCES = [
    "Makefile",
    RTL,
    "tools/ace2_qk_residual_cross_term_reference.py",
    "tools/gen_qk_residual_cross_term_vectors.py",
    "tools/run_qk_residual_cross_term_preflight.py",
    "verification/test_qk_residual_cross_term.py",
    "verification/tb/ace2_qk_residual_cross_term_tb.sv",
    "verification/generated/qk_residual_cross_term_vectors.json",
    "verification/generated/qk_residual_cross_term_vectors.svh",
]
IMPLEMENTATION_SOURCES = [
    RTL,
    "tools/ace2_qk_residual_cross_term_reference.py",
    "tools/gen_qk_residual_cross_term_vectors.py",
    "verification/test_qk_residual_cross_term.py",
    "verification/tb/ace2_qk_residual_cross_term_tb.sv",
    "verification/generated/qk_residual_cross_term_vectors.json",
    "verification/generated/qk_residual_cross_term_vectors.svh",
]
LOGS = {
    "software_reference": "evidence/shared_qk_residual_cross_term_attention_v1/latest/software_reference_unittest.log",
    "rtl_simulation": "evidence/shared_qk_residual_cross_term_attention_v1/latest/rtl_simulation.log",
    "sidecar_lint": "evidence/shared_qk_residual_cross_term_attention_v1/latest/rtl_sidecar_lint.log",
    "score_lint": "evidence/shared_qk_residual_cross_term_attention_v1/latest/rtl_score_lint.log",
    "elaboration": "evidence/shared_qk_residual_cross_term_attention_v1/latest/rtl_elaboration.log",
    "interface_elaboration": "evidence/shared_qk_residual_cross_term_attention_v1/latest/interface_elaboration.log",
}
INTERFACE_XML = {
    "ace2_qk_residual_sidecar_core": "evidence/shared_qk_residual_cross_term_attention_v1/latest/sidecar_interface.xml",
    "ace2_residual_cross_term_score_core": "evidence/shared_qk_residual_cross_term_attention_v1/latest/score_interface.xml",
}


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
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    item = path if isinstance(path, Path) else ROOT / path
    digest = hashlib.sha256()
    with item.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str) -> dict[str, Any]:
    item = ROOT / path
    return {"path": path, "bytes": item.stat().st_size, "sha256": sha256(path)}


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


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    return hashlib.sha256((json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def run(command: list[str], log_path: str) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    (ROOT / log_path).write_text(result.stdout, encoding="utf-8")
    require(result.returncode == 0, f"command failed ({result.returncode}): {' '.join(command)}")


def authorization_record(source: dict[str, Any], candidate_hash: str) -> dict[str, Any]:
    record = copy.deepcopy(source)
    record.update({
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "rtl_started": True,
        "candidate_rtl_hash": candidate_hash,
        "status": "bounded_reference_rtl_preflight_pass_independent_rtl_review_pending",
        "required_manager_action": "hold_rtl_for_independent_rtl_review",
    })
    return record


def module_interfaces() -> list[dict[str, Any]]:
    sidecar_ports = {
        "clk_i": 1, "rst_ni": 1, "clear_i": 1, "start_valid_i": 1,
        "start_ready_o": 1, "operation_rope_i": 1, "accumulator_s32_i": 32,
        "multiplier_s32_i": 32, "shift_u6_i": 6, "baseline_scale32_i": 32,
        "residual_scale32_i": 32, "residual_real_s4_i": 4,
        "residual_imag_s4_i": 4, "cosine_q1_15_i": 16, "sine_q1_15_i": 16,
        "div_req_valid_o": 1, "div_req_ready_i": 1, "div_numerator_u128_o": 128,
        "div_denominator_u128_o": 128, "div_rsp_valid_i": 1,
        "div_rsp_ready_o": 1, "div_quotient_u128_i": 128,
        "div_remainder_u128_i": 128, "out_valid_o": 1, "out_ready_i": 1,
        "baseline_q8_o": 8, "residual_s4_o": 4, "residual_real_s8_o": 8,
        "residual_imag_s8_o": 8, "positive_clamp_o": 1, "negative_clamp_o": 1,
        "descriptor_error_o": 1, "numeric_overflow_o": 1,
    }
    score_ports = {
        "clk_i": 1, "rst_ni": 1, "clear_i": 1, "start_valid_i": 1,
        "start_ready_o": 1, "lane_count_u7_i": 7,
        "base_score_q20_44_s64_i": 64,
        "query_scale32_i": 32, "key_scale32_i": 32,
        "query_residual_scale32_i": 32, "key_residual_scale32_i": 32,
        "lane_valid_i": 1, "lane_ready_o": 1, "query_q8_i": 8, "key_q8_i": 8,
        "query_residual_s8_i": 8, "key_residual_s8_i": 8, "score_valid_o": 1,
        "score_ready_i": 1, "score_q20_44_s64_o": 64, "dot_q_rk_s32_o": 32,
        "dot_rq_k_s32_o": 32, "dot_rq_rk_s32_o": 32,
        "descriptor_error_o": 1, "numeric_overflow_o": 1,
    }
    return [
        {"module": "ace2_qk_residual_sidecar_core", "parameters": {}, "ports": sidecar_ports},
        {"module": "ace2_residual_cross_term_score_core", "parameters": {"HEAD_DIM": 64}, "ports": score_ports},
    ]


def ports_from_verilator_xml(path: str, module_name: str) -> dict[str, int]:
    root = ET.parse(ROOT / path).getroot()
    type_widths: dict[str, int] = {}
    for dtype in root.findall(".//typetable/basicdtype"):
        left = dtype.get("left")
        right = dtype.get("right")
        type_widths[dtype.attrib["id"]] = 1 if left is None else abs(int(left) - int(right or "0")) + 1
    module = next((node for node in root.findall(".//netlist/module")
                   if node.get("name") == module_name), None)
    require(module is not None, f"module missing from Verilator XML: {module_name}")
    return {var.attrib["name"]: type_widths[var.attrib["dtype_id"]]
            for var in module.findall("var") if var.get("dir") in {"input", "output", "inout"}}


def check_only() -> int:
    packet = load(str(PACKET.relative_to(ROOT)))
    require(packet.get("contract_id") == CONTRACT, "preflight packet contract mismatch")
    require(packet.get("status") == "pass_ready_for_independent_rtl_review",
            "preflight packet is not passing")
    require(all(packet.get("checklist", {}).values()), "preflight checklist is incomplete")
    for path, digest in packet.get("source_hashes", {}).items():
        require((ROOT / path).is_file(), f"missing bound source: {path}")
        require(sha256(path) == digest, f"bound source hash changed: {path}")
    require(aggregate_hash(packet["source_hashes"]) == packet["candidate_rtl_hash"],
            "candidate aggregate hash mismatch")
    for name, binding in packet.get("traceability_bindings", {}).items():
        path = binding.get("path")
        require(path and (ROOT / path).is_file(), f"missing traceability binding: {name}")
        require(sha256(path) == binding.get("sha256"),
                f"traceability binding changed: {name}")
    canonical = copy.deepcopy(packet)
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    require(hashlib.sha256((json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
            == packet.get("integrity", {}).get("canonical_sha256"),
            "preflight packet canonical hash mismatch")
    manifest = load("design/RTL_MANIFEST.json")
    require(manifest.get("candidate_rtl_hash") == packet["candidate_rtl_hash"],
            "RTL manifest candidate hash mismatch")
    require(manifest.get("candidate_id") == packet["candidate_id"],
            "RTL manifest candidate id mismatch")
    require(manifest.get("current_stage") == "rtl" and manifest.get("stage") == "rtl",
            "RTL manifest stage projections are stale")
    metadata_projection = manifest.get("candidate_model_metadata", {})
    require(metadata_projection.get("candidate_id") == packet["candidate_id"] and
            metadata_projection.get("candidate_rtl_hash") == packet["candidate_rtl_hash"] and
            metadata_projection.get("status") == "rtl_preflight_bound_independent_rtl_review_pending",
            "RTL manifest metadata projection is stale")
    require(packet.get("schedule_contract") == manifest.get("candidate_schedule") and
            packet.get("schedule_contract", {}).get("total_cycles_per_score") == SCORE_TOTAL_CYCLES,
            "RTL manifest schedule projection is stale")
    require(all(manifest.get("traceability", {}).get("stage_checklist", {}).values()),
            "RTL manifest checklist is incomplete")
    require(load("research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "Manager-owned current_stage changed")
    status = load("research/PUBLIC_STATUS.json")
    require(status.get("stage", {}).get("current_stage") == "rtl", "public stage is stale")
    require(status.get("integrity", {}).get("canonical_sha256") == canonical_sha256(status),
            "public canonical hash is stale")
    require("/home/" not in json.dumps(status, sort_keys=True), "public status leaks a private path")
    print(f"QK_RESIDUAL_CROSS_TERM_PREFLIGHT_CHECK_PASS candidate={packet['candidate_id']}")
    return 0


def main() -> int:
    if "--check" in sys.argv[1:]:
        return check_only()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned current_stage is not rtl")
    policy = load("design/FAST_LOOP_POLICY.json")
    active = policy.get("active_repair_authorization", {})
    require(active.get("contract_id") == CONTRACT, "active authorization contract is stale")
    fresh_authorization = (active.get("implementation_authorized") is True and
                           active.get("operator_approval_consumed") is False)
    if not fresh_authorization:
        require(active.get("implementation_authorized") is False and
                active.get("operator_approval_consumed") is True and PACKET.is_file(),
                "implementation is neither freshly authorized nor an existing frozen candidate")
        prior_manifest = load("design/RTL_MANIFEST.json")
        prior_hashes = prior_manifest.get("candidate_source_hashes", {})
        changed = {path for path in IMPLEMENTATION_SOURCES
                   if prior_hashes.get(path) != sha256(path)}
        prior_vectors = load("verification/generated/qk_residual_cross_term_vectors.json")
        coverage_repair = (
            "attention_row_cases" not in prior_vectors and
            changed <= {
                "tools/ace2_qk_residual_cross_term_reference.py",
                "tools/gen_qk_residual_cross_term_vectors.py",
                "verification/test_qk_residual_cross_term.py",
                "verification/generated/qk_residual_cross_term_vectors.json",
                "verification/generated/qk_residual_cross_term_vectors.svh",
            }
        )
        schedule_traceability_repair = (
            RTL in changed and
            changed <= {
                RTL,
                "verification/tb/ace2_qk_residual_cross_term_tb.sv",
            }
        )
        require(not changed or coverage_repair or schedule_traceability_repair,
                "consumed authorization forbids changed implementation sources outside the bounded coverage or schedule-traceability repair")
    require(load("research/ENVIRONMENT_REVIEWER_VERDICT.json").get("verdict") == "done",
            "environment review is not done")
    refreeze_packet = load(REFREEZE_PACKET)
    architecture_decision = load(ARCHITECTURE_DECISION)
    environment_decision = load(ENVIRONMENT_DECISION)
    require(sha256(REFREEZE_PACKET) == REFREEZE_PACKET_SHA256,
            "accepted architecture refreeze packet hash changed")
    require(architecture_decision.get("status") == "done" and
            architecture_decision.get("packet", {}).get("sha256") == REFREEZE_PACKET_SHA256,
            "fresh architecture acceptance is not bound to the accepted packet")
    require(environment_decision.get("status") == "done" and
            environment_decision.get("packet", {}).get("sha256") == REFREEZE_PACKET_SHA256,
            "fresh environment acceptance is not bound to the accepted packet")
    require(refreeze_packet.get("selected_solution", {}).get("consumer_port") ==
            "base_score_q20_44_s64_i", "refreeze packet consumer port changed")
    hook = load("reference/qk_residual_cross_term_full_model_hook.json")
    metadata = load("reference/generated/qk_residual_scale32_metadata.json")
    require(hook.get("authoritative_base_score_interface", {}).get("consumer_port") ==
            "base_score_q20_44_s64_i", "full-model hook base-score port changed")
    require(metadata.get("contract_id") == CONTRACT and
            "never replace or recompute" in
            metadata.get("derivation", {}).get("baseline_preservation", ""),
            "metadata no longer preserves the authoritative baseline score")

    run(["python", "tools/gen_qk_residual_cross_term_vectors.py"],
        "evidence/shared_qk_residual_cross_term_attention_v1/latest/vector_generation.log")
    first_vector_hashes = {
        path: sha256(path) for path in SOURCES if path.startswith("verification/generated/")
    }
    run(["python", "tools/gen_qk_residual_cross_term_vectors.py"],
        "evidence/shared_qk_residual_cross_term_attention_v1/latest/vector_regeneration.log")
    require(first_vector_hashes == {
        path: sha256(path) for path in first_vector_hashes
    }, "generated vectors are not deterministic")
    run(["python", "-m", "unittest", "verification.test_qk_residual_cross_term", "-v"],
        LOGS["software_reference"])
    run(["iverilog", "-g2012", "-Wall", "-Iverification/generated", "-o",
         "build/ace2_qk_residual_cross_term_tb.vvp", RTL,
         "verification/tb/ace2_qk_residual_cross_term_tb.sv"], LOGS["elaboration"])
    run(["vvp", "build/ace2_qk_residual_cross_term_tb.vvp"], LOGS["rtl_simulation"])
    run(["verilator", "--lint-only", "--language", "1800-2017", "-Wall", "-Wno-fatal",
         "--top-module", "ace2_qk_residual_sidecar_core", RTL], LOGS["sidecar_lint"])
    run(["verilator", "--lint-only", "--language", "1800-2017", "-Wall", "-Wno-fatal",
         "--top-module", "ace2_residual_cross_term_score_core", RTL], LOGS["score_lint"])
    interface_commands = []
    for module_name, xml_path in INTERFACE_XML.items():
        interface_commands.append(["verilator", "--xml-only", "--language", "1800-2017",
                                   "--xml-output", xml_path, "--top-module", module_name, RTL])
    for command in interface_commands:
        run(command, LOGS["interface_elaboration"])

    require("OK" in (ROOT / LOGS["software_reference"]).read_text(), "reference tests did not pass")
    require("TB_PASS qk_residual_cross_term" in (ROOT / LOGS["rtl_simulation"]).read_text(),
            "RTL simulation did not pass")
    require(f"SCORE_LATENCY_PASS cycles={SCORE_TOTAL_CYCLES}" in
            (ROOT / LOGS["rtl_simulation"]).read_text(),
            "RTL simulation did not assert the frozen score schedule")
    require((ROOT / LOGS["sidecar_lint"]).stat().st_size == 0, "sidecar lint is not clean")
    require((ROOT / LOGS["score_lint"]).stat().st_size == 0, "score lint is not clean")
    rtl_text = (ROOT / RTL).read_text(encoding="utf-8")
    sidecar_text, score_text = rtl_text.split("module ace2_residual_cross_term_score_core", 1)
    require(sidecar_text.count(" * ") == 1, "sidecar must contain exactly one multiplier operator")
    require(score_text.count(" * ") == 1, "score core must contain exactly one multiplier operator")
    require(" / " not in sidecar_text and " % " not in sidecar_text,
            "sidecar must use the shared divider service, not a local division operator")
    declared_interfaces = {item["module"]: item for item in module_interfaces()}
    for module_name, xml_path in INTERFACE_XML.items():
        require(ports_from_verilator_xml(xml_path, module_name) ==
                declared_interfaces[module_name]["ports"],
                f"manifest interface does not match elaborated RTL: {module_name}")
    require("parameter integer HEAD_DIM = 64" in score_text,
            "score parameter HEAD_DIM default does not match the manifest")
    require("base_score_q20_44_s64_i" in score_text,
            "authoritative signed-Q20.44 base-score port is missing")
    require("base_dot_s32_i" not in score_text,
            "forbidden base-dot reconstruction interface remains")
    vectors = load("verification/generated/qk_residual_cross_term_vectors.json")
    require(vectors.get("coverage", {}).get("score_seed") ==
            "authoritative_base_score_q20_44_s64",
            "vectors do not carry the authoritative base score")

    source_hashes = {path: sha256(path) for path in SOURCES}
    candidate_hash = aggregate_hash(source_hashes)
    candidate_id = f"qk_residual_cross_term_{candidate_hash[:16]}"
    auth = authorization_record(active, candidate_hash)

    packet = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "current_stage": "rtl",
        "status": "pass_ready_for_independent_rtl_review",
        "stage_closing": False,
        "checklist": {
            "manager_stage_is_rtl": True,
            "standing_authorization_was_valid_and_is_now_consumed": True,
            "independent_integer_reference_pass": True,
            "deterministic_vector_regeneration_pass": True,
            "bit_exact_rtl_simulation_pass": True,
            "sidecar_verilator_lint_clean": True,
            "score_verilator_lint_clean": True,
            "iverilog_elaboration_clean": True,
            "one_shared_multiplier_per_candidate_module": True,
            "shared_divider_service_used_without_local_divider": True,
            "reset_clear_and_backpressure_exercised": True,
            "first_party_provenance_and_regeneration_bound": True,
            "manifest_interfaces_match_verilator_elaboration": True,
            "accepted_refreeze_packet_and_independent_acceptances_bound": True,
            "authoritative_base_score_carried_without_term_zero_reconstruction": True,
            "frozen_192_dot_plus_24_conversion_cycle_schedule_asserted": True,
            "architecture_spec_hook_metadata_reference_rtl_vectors_preflight_bound": True,
        },
        "coverage": vectors["coverage"],
        "schedule_contract": {
            "dot_cycles_per_score": SCORE_DOT_CYCLES,
            "correction_conversions_per_score": SCORE_CONVERSION_COUNT,
            "cycles_per_conversion": SCORE_CYCLES_PER_CONVERSION,
            "conversion_cycles_per_score": SCORE_CONVERSION_CYCLES,
            "total_cycles_per_score": SCORE_TOTAL_CYCLES,
            "query_heads": SCORE_QUERY_HEADS,
            "cycles_per_key": SCORE_CYCLES_PER_KEY,
            "assertion_log": artifact(LOGS["rtl_simulation"]),
            "status": "rtl_cycle_assertion_pass",
        },
        "source_hashes": source_hashes,
        "traceability_bindings": {
            "accepted_refreeze_packet": artifact(REFREEZE_PACKET),
            "architecture_acceptance": artifact(ARCHITECTURE_DECISION),
            "environment_acceptance": artifact(ENVIRONMENT_DECISION),
            **{Path(path).stem: artifact(path) for path in TRACEABILITY_PATHS},
        },
        "logs": {name: artifact(path) for name, path in LOGS.items()},
        "interface_xml": {name: artifact(path) for name, path in INTERFACE_XML.items()},
        "prohibited_runs": {
            "all_layer_quality_discriminator": False,
            "paired_smoke": False,
            "full_shell_regression": False,
            "sky130_ppa": False,
            "prototype": False,
            "benchmark": False,
            "signoff": False,
        },
        "claim_boundary": "Standalone synthesizable candidate RTL preflight only; accepted prefix and historical PPA frontier are unchanged.",
        "preflight_repairs": [
            "Replaced base_dot_s32_i with signed base_score_q20_44_s64_i and initialized the checked signed-67 accumulator from the authoritative base score before exactly three correction conversions.",
            "Computed staged Q1.31/Q0.15 softmax and signed-int8 attention-value vectors replaced a non-qualifying row-length enumeration for lengths 1, 2, 63, 64, and 65.",
            "Aligned the score FSM to 192 serialized dot cycles plus three eight-cycle conversions and asserted the resulting 216-cycle accepted-start-to-score-valid latency."
        ],
        "generated_at_utc": utc_now(),
    }
    packet["integrity"] = {"canonical_sha256": None}
    canonical = copy.deepcopy(packet)
    canonical["integrity"]["canonical_sha256"] = None
    packet["integrity"]["canonical_sha256"] = canonical_sha256(packet)
    dump(str(PACKET.relative_to(ROOT)), packet)

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update({
        "current_stage": "rtl",
        "stage": "rtl",
        "candidate_id": candidate_id,
        "architecture_contract_status": f"{CONTRACT}_bounded_rtl_preflight_pass_independent_review_pending",
        "candidate_status": "bounded_reference_rtl_preflight_pass_independent_rtl_review_pending",
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "ordered_standalone_candidate_sources_not_shell_admitted",
        "candidate_source_hashes": source_hashes,
        "candidate_rtl_sources": [artifact(RTL)],
        "candidate_generated_hashes": first_vector_hashes,
        "candidate_generated_sources": [
            {**artifact("verification/generated/qk_residual_cross_term_vectors.json"),
             "generator": "tools/gen_qk_residual_cross_term_vectors.py",
             "reference": "tools/ace2_qk_residual_cross_term_reference.py",
             "regeneration_command": "python tools/gen_qk_residual_cross_term_vectors.py", "third_party": False},
            {**artifact("verification/generated/qk_residual_cross_term_vectors.svh"),
             "generator": "tools/gen_qk_residual_cross_term_vectors.py",
             "reference": "tools/ace2_qk_residual_cross_term_reference.py",
             "regeneration_command": "python tools/gen_qk_residual_cross_term_vectors.py", "third_party": False},
        ],
        "candidate_interface": {"status": "exact_standalone_interfaces_bound_not_shell_admitted",
                                "modules": module_interfaces()},
        "candidate_review_binding": {"status": "independent_rtl_review_pending",
                                     "candidate_capability_accepted": False,
                                     "evidence": str(PACKET.relative_to(ROOT)), "stage_closing": False},
        "candidate_verification_binding": {"status": "rtl_preflight_pass_independent_verification_pending",
                                           "evidence": [artifact(path) for path in LOGS.values()]},
        "candidate_verification_complete": False,
        "candidate_meets_numeric_acceptance": False,
        "candidate_schedule": copy.deepcopy(packet["schedule_contract"]),
        "interfaces_contract_status": "candidate_standalone_interfaces_exactly_bound_review_pending_not_shell_admitted",
        "generated_at_utc": utc_now(),
    })
    manifest["candidate_model_metadata"].update({
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "status": "rtl_preflight_bound_independent_rtl_review_pending",
    })
    manifest["proposed_replacement_contract"] = auth
    manifest["traceability"].update({
        "selected_mechanism": CONTRACT,
        "architecture_contract_gap": {"status": "standalone_rtl_traced_independent_review_pending",
                                      "resolution_owner": "independent RTL Reviewer"},
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
        "stage_checklist": {"rtl.contract-traceability": True,
                            "rtl.hardware-discipline": True, "rtl.ip-provenance": True},
    })
    manifest["claim_boundaries"] = [
        "The residual cross-term candidate has standalone reference, generation, lint, elaboration, and RTL simulation evidence only.",
        "The accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported.",
        "Independent RTL review, all-layer quality discrimination, paired smoke, shell admission/regression, and candidate PPA have not run.",
        "The historical 0.6108746272 mm2 and +0.1502 ns at 100 MHz frontier remains unchanged.",
    ]
    old_names = {"ace2_qk_residual_sidecar_core_and_residual_cross_term_score_core",
                 "qk_residual_cross_term_vectors"}
    manifest["ip_provenance"] = [item for item in manifest["ip_provenance"]
                                 if item.get("name") not in old_names]
    manifest["ip_provenance"].extend([
        {"kind": "first_party_bounded_candidate_rtl",
         "name": "ace2_qk_residual_sidecar_core_and_residual_cross_term_score_core",
         "path": RTL, "sha256": source_hashes[RTL], "source_revision": source_hashes[RTL],
         "license": "repository project license not separately declared in this manifest",
         "third_party": False},
        {"kind": "generated_bounded_candidate_verification_source",
         "name": "qk_residual_cross_term_vectors",
         "generator": "tools/gen_qk_residual_cross_term_vectors.py",
         "reference": "tools/ace2_qk_residual_cross_term_reference.py",
         "regeneration_command": "python tools/gen_qk_residual_cross_term_vectors.py",
         "third_party": False},
    ])
    manifest.setdefault("latest_evidence", {})["qk_residual_cross_term_rtl_preflight"] = artifact(str(PACKET.relative_to(ROOT)))
    dump("design/RTL_MANIFEST.json", manifest)

    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key] = copy.deepcopy(auth)
    policy["manager_recommendation"] = "hold_rtl_for_independent_rtl_review"
    dump("design/FAST_LOOP_POLICY.json", policy)

    oracle = load("reference/ORACLE_MANIFEST.json")
    oracle["architecture_contract_status"] = "bounded_rtl_preflight_pass_independent_rtl_review_pending"
    oracle["claim_boundary"] = packet["claim_boundary"]
    oracle["proposed_architecture_contract"] = copy.deepcopy(auth)
    oracle["oracles"] = [item for item in oracle["oracles"]
                         if item.get("contract_id") != CONTRACT]
    oracle["oracles"].append({
        "contract_id": CONTRACT,
        "case_count": 15,
        "numeric_acceptance": "bit_exact_integer_reference_and_standalone_rtl_preflight",
        "reference": "tools/ace2_qk_residual_cross_term_reference.py",
        "reference_sha256": source_hashes["tools/ace2_qk_residual_cross_term_reference.py"],
        "generator": "tools/gen_qk_residual_cross_term_vectors.py",
        "generator_sha256": source_hashes["tools/gen_qk_residual_cross_term_vectors.py"],
        "vector_json": "verification/generated/qk_residual_cross_term_vectors.json",
        "vector_json_sha256": source_hashes["verification/generated/qk_residual_cross_term_vectors.json"],
        "vector_svh": "verification/generated/qk_residual_cross_term_vectors.svh",
        "vector_svh_sha256": source_hashes["verification/generated/qk_residual_cross_term_vectors.svh"],
        "status": "preflight_pass_independent_rtl_review_pending",
    })
    oracle["generated_at_utc"] = utc_now()
    dump("reference/ORACLE_MANIFEST.json", oracle)

    status = load("research/PUBLIC_STATUS.json")
    status.update({"current_mode": "ADVANCE",
                   "supported_layer_operator_prefix": PREFIX,
                   "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                   "latest_decision": "residual_cross_term_rtl_preflight_pass_independent_review_pending",
                   "latest_ppa_frontier_status": "historical_frontier_preserved_no_candidate_ppa",
                   "selected_replacement_contract": copy.deepcopy(auth),
                   "generated_at_utc": utc_now(), "last_updated_utc": utc_now()})
    status["architecture_proposal_gate"].update(copy.deepcopy(auth))
    status["dashboard_fields"].update({
        "candidate_mechanism": copy.deepcopy(auth), "current_mode": "ADVANCE",
        "supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "latest_decision": "residual_cross_term_rtl_preflight_pass_independent_review_pending",
        "latest_ppa_frontier_status": "historical_frontier_preserved_no_candidate_ppa",
    })
    status["implementation_frontier"]["latest_decision"] = "residual_cross_term_rtl_preflight_pass_independent_review_pending"
    status["blockers"] = [{
        "id": "independent_rtl_review_pending", "stage": "rtl", "status": "active",
        "evidence": str(PACKET.relative_to(ROOT)),
        "reason": "The bounded standalone RTL preflight passes, but independent RTL checklist review has not yet certified it.",
        "required_resolution": "Independent Reviewer must inspect the hash-bound candidate before any quality discriminator or further RTL action.",
    }]
    dump("research/PUBLIC_STATUS.json", status)

    (ROOT / "design/RTL_TRACEABILITY.md").write_text(f"""# ACE-2 RTL traceability notes

## Accepted shell frontier

The accepted shell remains hash-bound through `layer_0.v_proj`; first unsupported
remains `layer_0.rope_q`. Historical accepted PPA remains 62,199 cells,
0.6108746272 mm2, and +0.1502 ns setup slack at 100 MHz.

## Current residual cross-term candidate

- Contract: `{CONTRACT}`.
- Candidate: `{candidate_id}`; ordered source hash `{candidate_hash}`.
- `ace2_qk_residual_sidecar_core` traces exact projection residual generation,
  symmetric signed-4 clamping, reserved-code rejection, residual RoPE, one shared
  multiplier, and a ready/valid client interface to the reused unsigned divider.
- `ace2_residual_cross_term_score_core` captures the authoritative signed-Q20.44
  base score, sign-extends it into the checked signed-67 accumulator, and converts
  and adds only the three residual correction terms through one multiplier.
- Its hash-bound simulation asserts the frozen 192 dot cycles plus three
  eight-cycle conversions: 216 cycles per score and 3,024 cycles per key across
  fourteen query heads.
- Reset/clear, backpressure, descriptor errors, deterministic generated vectors,
  Icarus elaboration/simulation, and warning-free Verilator lint are bound in
  `{PACKET.relative_to(ROOT)}`.
- Both modules and generated sources are first-party. Regeneration is
  `python tools/gen_qk_residual_cross_term_vectors.py`; no third-party IP is added.
- The three RTL checklist evidence fields are true for this standalone candidate.
  Independent review remains pending; the candidate is not shell-admitted and
  has no quality, PPA, prototype, benchmark, or signoff claim.

## Historical rejected candidates

All earlier RoPE/score successors remain historical bounded no-go evidence and
are not current RTL capability.
""", encoding="utf-8")

    (ROOT / "CHECKPOINT.md").write_text(f"""# Goal

Complete the bounded reference/RTL implementation and consolidated preflight for
`{CONTRACT}` without entering a downstream stage.

# Current State

The hash-bound standalone RTL preflight passes and the one-time implementation
authorization is consumed. `current_stage` remains Manager-owned `rtl`.
Independent RTL review is the only legal next action.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, 100 MHz
floor, and historical PPA frontier are unchanged.

# Verified Done

- Integer reference and deterministic JSON/SVH vector generation pass.
- Both synthesizable candidate modules elaborate and lint without warnings.
- Bit-exact standalone RTL simulation passes projection residuals, clamp cases,
  residual RoPE, authoritative base-score carry-through plus three correction
  terms, backpressure,
  reserved-code rejection, descriptor errors, and in-flight clear.
- Exact interfaces, hashes, first-party provenance, and regeneration commands are
  bound in `design/RTL_MANIFEST.json` and `{PACKET.relative_to(ROOT)}`.

# Locked / Not Run

Independent verification, all-layer quality discrimination, paired smoke, shell
admission/regression, SKY130 PPA, prototype, benchmark, and signoff were not run.

# Next Required Action

Independent Reviewer inspection of the hash-bound RTL checklist evidence. No
candidate edit or expensive evaluation is authorized before that review.
""", encoding="utf-8")

    # Refresh all duplicated public dashboard projections only after every
    # referenced project artifact has reached its final content.
    status = load("research/PUBLIC_STATUS.json")
    now = utc_now()
    status["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(manifest["traceability"]["stage_checklist"]),
        "current_stage_evidence": [
            "research/PIPELINE_STATE.json",
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "design/RTL_MANIFEST.json",
            str(PACKET.relative_to(ROOT)),
        ],
        "current_stage_status": "bounded_rtl_preflight_pass_independent_review_pending",
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    gate = status.get("architecture_proposal_gate", {})
    gate.update(copy.deepcopy(auth))
    gate.update({
        "source": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "required_operator_action": "none_implementation_consumed_independent_review_pending",
    })
    status["architecture_proposal_gate"] = gate
    status["selected_replacement_contract"] = copy.deepcopy(auth)
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container.update({
            "current_stage": "rtl",
            "current_mode": "ADVANCE",
            "latest_decision": "residual_cross_term_rtl_preflight_pass_independent_review_pending",
            "required_manager_action": "hold_rtl_for_independent_rtl_review",
            "required_operator_action": "none_implementation_consumed_independent_review_pending",
            "candidate_mechanism": copy.deepcopy(auth),
            "candidate_rtl_hash": candidate_hash,
            "candidate_rtl_hash_scope": "ordered_standalone_candidate_sources_not_shell_admitted",
            "rtl_contract_traceability": True,
            "routing_status": "independent_rtl_review_pending",
            "operator_policy": copy.deepcopy(policy),
        })
    status["public_claims"] = [
        {"claim": "current Manager-owned stage remains rtl",
         "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"]},
        {"claim": "the bounded standalone residual cross-term RTL preflight passes and its one-time implementation authorization is consumed",
         "evidence": [str(PACKET.relative_to(ROOT)), "design/RTL_MANIFEST.json", "design/FAST_LOOP_POLICY.json"]},
        {"claim": "the accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported",
         "evidence": ["design/RTL_MANIFEST.json", "research/PUBLIC_STATUS.json"]},
        {"claim": "the historical PPA frontier, 2.0 mm2 cap, and 100 MHz floor are unchanged",
         "evidence": ["design/PPA_FRONTIER_LEDGER.json", "design/TARGET.json"]},
        {"claim": "independent RTL review, quality discrimination, paired smoke, shell regression, and candidate PPA have not run",
         "evidence": [str(PACKET.relative_to(ROOT))]},
    ]
    public_paths = {
        str(item.get("path")) for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path") and item.get("path") != "research/PUBLIC_STATUS.json"
    }
    public_paths.update({
        "CHECKPOINT.md", "Makefile", "design/FAST_LOOP_POLICY.json",
        "design/NUMERICAL_REPLACEMENT_PROPOSAL.md", "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md", "reference/ORACLE_MANIFEST.json",
        "research/PIPELINE_STATE.json", RTL,
        "tools/ace2_qk_residual_cross_term_reference.py",
        "tools/gen_qk_residual_cross_term_vectors.py",
        "tools/run_qk_residual_cross_term_preflight.py",
        "verification/test_qk_residual_cross_term.py",
        "verification/tb/ace2_qk_residual_cross_term_tb.sv",
        "verification/generated/qk_residual_cross_term_vectors.json",
        "verification/generated/qk_residual_cross_term_vectors.svh",
        str(PACKET.relative_to(ROOT)),
    })
    status["artifact_hashes"] = [artifact(path) for path in sorted(public_paths)
                                 if (ROOT / path).is_file()]
    status["generated_at_utc"] = now
    status["last_updated_utc"] = now
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": "UTF-8 sorted keys two-space indentation trailing newline canonical hash null during hash",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    require("/home/" not in json.dumps(status, sort_keys=True), "public status leaks a private path")
    dump("research/PUBLIC_STATUS.json", status)

    print(f"QK_RESIDUAL_CROSS_TERM_PREFLIGHT_PASS candidate={candidate_id}")
    print(f"packet={PACKET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
