#!/usr/bin/env python3
"""Refresh and validate the frozen down-projection residual-fusion contract.

This maintainer is intentionally architecture-local.  It repairs nested probe
artifact records, refreshes the directly bound sidecars, recomputes the
authoritative packet/redirect/freeze canonical digests, and validates the live
public projection without modifying it.  It does not generate RTL, run
datasets, update public status, or advance any downstream stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

from validate_down_projection_residual_fusion_evidence import validate_artifact_tree


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
PREDECESSOR_CONTRACT = "shared_v_residual_value_correction_attention_v1"
LATEST = ROOT / f"evidence/{CONTRACT}/latest"

PACKET = LATEST / "ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
REDIRECT = LATEST / "ARCHITECTURE_REVIEW_PACKET.json"
BINDINGS = LATEST / "CONTRACT_BINDINGS.json"
PROVENANCE = LATEST / "FIRST_PARTY_PROVENANCE.json"
HOOK_CONTRACT = LATEST / "FULL_MODEL_HOOK_CONTRACT.json"
FREEZE = ROOT / f"evidence/{CONTRACT}/architecture_freeze.json"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
RTL_PRECHECK = LATEST / "PRECHECK.json"
ARCH_DECISION = ROOT / "evidence/review/architecture_refreeze_shared_down_projection_residual_fusion_v1/decision.json"

PROBE_REL = "evidence/diagnostics/shared_down_proj_residual_fusion_probe_20260801/results.json"
REFREEZE_PACKET_REL = f"evidence/{CONTRACT}/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
HOOK_REL = "tools/ace2_down_projection_residual_fusion_hook.py"
QK_NO_GO_REL = "evidence/shared_qk_residual_cross_term_attention_v1/latest/VERIFICATION_NO_GO.json"
V_NO_GO_REL = "evidence/shared_v_residual_value_correction_attention_v1/latest/VERIFICATION_DECISION.json"

EXPECTED_IMMUTABLE_HASHES = {
    PROBE_REL: "29171254c34f59900f634b8ca41d083a3f0129c27459373d5f748b0f6e136aaa",
    HOOK_REL: "b4e222db99a5457a84183b8069654a3881fe5afc1e1caf9c23f49d6aa1463e18",
    QK_NO_GO_REL: "252b19e608b4251da033b44523d7805d2c482ceee3486a55f8b270210f855fae",
    V_NO_GO_REL: "e5f018f90e8c0b65858572fc9fb21c665094178161608ac0712902b9eaec15a6",
}
EXPECTED_SEALED_BINDINGS_HASH = "159fe1b816dcf7f8d7503aaeeaa9f7ae6815dfd0192e0545b5bbe8a119f02681"
EXPECTED_ACCEPTED_ARCHITECTURE_DECISION_HASH = "3193281d67ee83e9f7a22a1a8ecf956e9d9de94172ddbbf676638edf364a0e5f"

PROBE_CLASSIFICATION = "exploratory_float64_proxy_not_scale32_construct_evidence"
PROBE_PERMITTED_USE = "historical_context_only"
STALE_PREDECESSOR_MODEL_KEYS = {
    "candidate_core_fraction_at_32768_keys",
    "candidate_workspace_bytes",
    "decode_context_32768_external_bytes",
    "decode_context_32768_total_cycles_at_16_bytes_per_cycle",
    "dma_fraction_at_16_bytes_per_cycle_at_32768_keys",
    "fourteen_path_speedup_bound_at_32768_keys",
    "prefill_128_dma_cycles_at_16_bytes_per_cycle",
    "prefill_128_total_cycles_at_16_bytes_per_cycle",
    "selected_incremental_logic_area_proxy_units",
}

BOUND_JSON = (
    ROOT / "design/TARGET.json",
    ROOT / "design/FAST_LOOP_POLICY.json",
    ROOT / "design/MEMORY_MODEL.json",
    PROVENANCE,
    HOOK_CONTRACT,
    PACKET,
    REDIRECT,
    FREEZE,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_close(actual: float, expected: float, message: str) -> None:
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12),
        f"{message}: expected {expected!r}, found {actual!r}",
    )


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(encoded.encode()).hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    require(path.is_file(), f"bound artifact missing: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def artifact_records(value: Any, trail: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value) and isinstance(value["path"], str):
            yield trail, value
        for key, child in value.items():
            yield from artifact_records(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from artifact_records(child, f"{trail}[{index}]")


def refresh_record(record: dict[str, Any]) -> None:
    current = artifact(record["path"])
    record["bytes"] = current["bytes"]
    record["sha256"] = current["sha256"]


def refresh_all_artifact_records(value: dict[str, Any]) -> int:
    records = list(artifact_records(value))
    for _, record in records:
        refresh_record(record)
    return len(records)


def refresh_probe_records(path: Path, expected_count: int) -> int:
    value = load(path)
    matches = [
        record
        for _, record in artifact_records(value)
        if record["path"] == PROBE_REL
    ]
    require(
        len(matches) == expected_count,
        f"unexpected probe binding count in {path.relative_to(ROOT)}: "
        f"expected {expected_count}, found {len(matches)}",
    )
    for record in matches:
        refresh_record(record)
    dump(path, value)
    return len(matches)


def refresh_sidecars() -> None:
    provenance = load(PROVENANCE)
    refresh_all_artifact_records(provenance)
    workload = ROOT / provenance["model_identity"]["source_contract"]
    provenance["model_identity"]["source_contract_sha256"] = sha256(workload)
    dump(PROVENANCE, provenance)

    hook_contract = load(HOOK_CONTRACT)
    refresh_all_artifact_records(hook_contract)
    dump(HOOK_CONTRACT, hook_contract)


def refresh_integrity() -> None:
    packet = load(PACKET)
    packet.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    packet["integrity"]["canonical_sha256"] = canonical_sha256(packet)
    dump(PACKET, packet)

    redirect = load(REDIRECT)
    refresh_record(redirect["compatibility_redirect"])
    redirect.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    redirect["integrity"]["canonical_sha256"] = canonical_sha256(redirect)
    dump(REDIRECT, redirect)

    freeze = load(FREEZE)
    refresh_all_artifact_records(freeze)
    freeze.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    freeze["integrity"]["canonical_sha256"] = canonical_sha256(freeze)
    dump(FREEZE, freeze)


def refresh() -> None:
    target_count = refresh_probe_records(ROOT / "design/TARGET.json", expected_count=4)
    policy_count = refresh_probe_records(ROOT / "design/FAST_LOOP_POLICY.json", expected_count=3)
    require(target_count + policy_count == 7, "expected seven nested probe artifact records")
    refresh_sidecars()
    refresh_integrity()


def validate_artifact_records() -> int:
    checked = 0
    failures: list[str] = []
    for json_path in BOUND_JSON:
        value = load(json_path)
        for trail, record in artifact_records(value):
            relative = record["path"]
            path = ROOT / relative
            if not path.is_file():
                failures.append(f"{json_path.relative_to(ROOT)}:{trail}: missing {relative}")
                continue
            actual_bytes = path.stat().st_size
            actual_sha = sha256(path)
            if record["bytes"] != actual_bytes or record["sha256"] != actual_sha:
                failures.append(
                    f"{json_path.relative_to(ROOT)}:{trail}: {relative} "
                    f"recorded bytes={record['bytes']} sha256={record['sha256']} "
                    f"actual bytes={actual_bytes} sha256={actual_sha}"
                )
            checked += 1
    require(not failures, "recursive artifact binding failures:\n" + "\n".join(failures))
    return checked


def validate_public_probe_record(record: Any, location: str) -> None:
    require(isinstance(record, dict), f"missing public probe binding: {location}")
    require(record.get("path") == PROBE_REL, f"unexpected public probe path: {location}")
    require(record.get("bytes") == 529880, f"stale public probe byte count: {location}")
    require(
        record.get("sha256") == EXPECTED_IMMUTABLE_HASHES[PROBE_REL],
        f"stale public probe hash: {location}",
    )
    require(
        record.get("classification") == PROBE_CLASSIFICATION,
        f"invalid public probe classification: {location}",
    )
    require(
        record.get("permitted_use") == PROBE_PERMITTED_USE,
        f"invalid public probe permitted use: {location}",
    )


def expected_public_performance_model(memory: dict[str, Any]) -> dict[str, Any]:
    compute = memory["compute_model"]
    hierarchy = memory["memory_hierarchy"]
    sram = hierarchy["sram_live_set_budget_bytes"]
    roofline = memory["roofline_amdahl"]
    bandwidth_16 = next(
        row for row in compute["bandwidth_sweep"] if row["bytes_per_cycle"] == 16
    )
    return {
        "contract_id": CONTRACT,
        "source": "design/MEMORY_MODEL.json",
        "status": "architecture_estimate_reconciled_not_measured_no_candidate_ppa",
        "claim_status": memory["claim_status"],
        "cycle_estimate_status": compute["cycle_estimate_status"],
        "quality_evidence_status": roofline["quality_evidence_status"],
        "area_status": "unmeasured_no_candidate_ppa",
        "frequency_status": "unmeasured_100mhz_is_operator_floor_not_evidence",
        "macs_per_layer_token": compute["macs_per_layer_token"],
        "external_bytes_per_layer_token": compute["external_bytes_per_layer_token"],
        "down_projection_cycles_per_layer_token": compute["down_dot_cycles_per_layer_token"],
        "fusion_cycles_per_layer_token": compute["fusion_cycles_per_layer_token"],
        "combined_cycles_per_layer_token": compute["combined_cycles_per_layer_token"],
        "all_24_layer_fusion_cycles_per_token": compute["all_24_layer_fusion_cycles_per_token"],
        "dma_payload_floor_cycles_at_16_bytes_per_cycle": bandwidth_16["dma_payload_cycles"],
        "dma_planning_cycles_at_16_bytes_per_cycle": bandwidth_16["dma_cycles"],
        "arithmetic_intensity_macs_per_external_byte": compute[
            "arithmetic_intensity_macs_per_external_byte"
        ],
        "compute_memory_balance_bytes_per_cycle": roofline[
            "compute_memory_balance_bytes_per_cycle"
        ],
        "maximum_slice_speedup_if_fusion_were_free": roofline[
            "max_slice_speedup_if_fusion_were_free"
        ],
        "kv_record_bytes": hierarchy["kv_record_bytes"],
        "candidate_metadata_bytes": hierarchy["model_image_increment_bytes"],
        "incremental_kv_or_token_payload_bytes": hierarchy[
            "incremental_kv_or_token_payload_bytes"
        ],
        "planned_sram_peak_bytes": sram["total_planned_peak"],
        "remaining_sram_margin_bytes": sram["remaining_planning_margin"],
    }


def validate_current_architecture_performance_model(
    model: Any,
    location: str,
    expected: dict[str, Any],
) -> None:
    require(isinstance(model, dict), f"missing current architecture model: {location}")
    require(
        model.get("contract_id") == CONTRACT,
        f"stale predecessor current architecture contract: {location}",
    )
    require(
        model.get("contract_id") != PREDECESSOR_CONTRACT,
        f"sealed predecessor remains current: {location}",
    )
    stale_keys = sorted(STALE_PREDECESSOR_MODEL_KEYS.intersection(model))
    require(
        not stale_keys,
        f"stale predecessor-only current architecture fields at {location}: {stale_keys}",
    )
    require(
        set(model) == set(expected),
        f"current architecture model schema mismatch at {location}: "
        f"missing={sorted(set(expected) - set(model))} "
        f"extra={sorted(set(model) - set(expected))}",
    )
    for key, expected_value in expected.items():
        require(
            model[key] == expected_value,
            f"current architecture model value mismatch at {location}.{key}: "
            f"expected {expected_value!r}, found {model[key]!r}",
        )


def validate_stale_predecessor_rejection(expected: dict[str, Any]) -> None:
    stale_contract = copy.deepcopy(expected)
    stale_contract["contract_id"] = PREDECESSOR_CONTRACT
    try:
        validate_current_architecture_performance_model(
            stale_contract,
            "negative_test.stale_contract",
            expected,
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative stale-predecessor contract test did not fail")

    stale_schema = copy.deepcopy(expected)
    stale_schema["selected_incremental_logic_area_proxy_units"] = 521
    try:
        validate_current_architecture_performance_model(
            stale_schema,
            "negative_test.stale_schema",
            expected,
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative stale-predecessor schema test did not fail")


def validate_public_status() -> None:
    public = load(PUBLIC_STATUS)
    memory = load(ROOT / "design/MEMORY_MODEL.json")
    expected_model = expected_public_performance_model(memory)
    mission = public["architecture_certification_mission"]
    require(
        mission["contract_id"] == CONTRACT,
        "public architecture certification mission contract mismatch",
    )
    require(
        mission["review_packet"] == REFREEZE_PACKET_REL,
        "public architecture certification mission does not point to the refreeze packet",
    )
    require(
        mission["implementation_authorized"] is False,
        "public architecture certification mission authorizes implementation",
    )

    validate_public_probe_record(
        public["architecture_proposal_gate"].get("probe"),
        "architecture_proposal_gate.probe",
    )
    validate_public_probe_record(
        public["selected_replacement_contract"].get("probe"),
        "selected_replacement_contract.probe",
    )

    artifact_hashes = public.get("artifact_hashes")
    require(isinstance(artifact_hashes, list), "public artifact_hashes is not a list")
    public_probe_records = [
        record
        for record in artifact_hashes
        if isinstance(record, dict) and record.get("path") == PROBE_REL
    ]
    require(len(public_probe_records) == 1, "unexpected public artifact_hashes probe binding count")
    public_probe = public_probe_records[0]
    require(public_probe.get("bytes") == 529880, "stale public artifact_hashes probe byte count")
    require(
        public_probe.get("sha256") == EXPECTED_IMMUTABLE_HASHES[PROBE_REL],
        "stale public artifact_hashes probe hash",
    )

    refreeze_records = [
        record
        for record in artifact_hashes
        if isinstance(record, dict) and record.get("path") == REFREEZE_PACKET_REL
    ]
    require(len(refreeze_records) == 1, "public artifact_hashes refreeze packet binding missing")
    refreeze_record = refreeze_records[0]
    require(
        refreeze_record.get("bytes") == PACKET.stat().st_size,
        "stale public refreeze packet byte count",
    )
    require(
        refreeze_record.get("sha256") == sha256(PACKET),
        "stale public refreeze packet hash",
    )

    for location, candidate in (
        ("dashboard_fields.candidate_mechanism", public["dashboard_fields"]["candidate_mechanism"]),
        (
            "implementation_frontier.candidate_mechanism",
            public["implementation_frontier"]["candidate_mechanism"],
        ),
    ):
        require(candidate.get("contract_id") == CONTRACT, f"public candidate contract mismatch: {location}")
        require(
            candidate.get("exact_scale32_dataset_discriminator_status") == "not_run",
            f"public exact Scale32 dataset status is not not_run: {location}",
        )
        require(
            "probe_final_relative_l2" not in candidate,
            f"positive exploratory-probe metrics remain public: {location}",
        )
        validate_public_probe_record(candidate.get("exploratory_probe"), f"{location}.exploratory_probe")

    for location, model in (
        (
            "dashboard_fields.current_architecture_performance_model",
            public["dashboard_fields"].get("current_architecture_performance_model"),
        ),
        (
            "implementation_frontier.current_architecture_performance_model",
            public["implementation_frontier"].get("current_architecture_performance_model"),
        ),
    ):
        validate_current_architecture_performance_model(model, location, expected_model)
    validate_stale_predecessor_rejection(expected_model)

    claims = public.get("public_claims")
    require(isinstance(claims, list), "public_claims is not a list")
    positive_tokens = ("improve", "better", "pass", "meets", "positive")
    for index, claim_record in enumerate(claims):
        require(isinstance(claim_record, dict), f"invalid public claim record: public_claims[{index}]")
        claim = claim_record.get("claim", "")
        require(isinstance(claim, str), f"invalid public claim text: public_claims[{index}]")
        lowered = claim.lower()
        require(
            not ("probe" in lowered and any(token in lowered for token in positive_tokens)),
            f"positive exploratory-probe claim remains public: public_claims[{index}]",
        )

    exploratory_claims = [
        claim_record
        for claim_record in claims
        if isinstance(claim_record, dict)
        and "retained float64 probe is exploratory historical context only" in claim_record.get("claim", "")
    ]
    require(len(exploratory_claims) == 1, "authoritative exploratory-only public probe claim missing")
    exploratory_evidence = exploratory_claims[0].get("evidence")
    require(isinstance(exploratory_evidence, list), "exploratory-only public claim evidence is not a list")
    require(PROBE_REL in exploratory_evidence, "exploratory-only public claim omits retained probe evidence")
    require(
        REFREEZE_PACKET_REL in exploratory_evidence,
        "exploratory-only public claim omits refreeze packet evidence",
    )
    require(
        public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public),
        "public status canonical digest stale",
    )


def validate_rtl_transition_binding() -> int:
    pipeline = load(PIPELINE_STATE)
    if pipeline.get("current_stage") != "rtl":
        return 0
    require(RTL_PRECHECK.is_file(), "RTL stage has no PRECHECK transition binding")
    precheck = load(RTL_PRECHECK)
    record = precheck.get("transition_binding")
    require(isinstance(record, dict), "PRECHECK transition binding missing")
    relative = record.get("path")
    require(isinstance(relative, str) and relative, "PRECHECK transition path missing")
    path = ROOT / relative
    require(path.is_file(), "immutable RTL transition binding missing")
    require(record.get("bytes") == path.stat().st_size, "RTL transition byte binding differs")
    require(record.get("sha256") == sha256(path), "RTL transition hash binding differs")
    transition = load(path)
    summary = validate_artifact_tree(transition, label=relative)
    sealed = transition.get("sealed_architecture", {})
    require(
        sealed.get("contract_bindings", {}).get("sha256") == EXPECTED_SEALED_BINDINGS_HASH,
        "transition does not preserve sealed CONTRACT_BINDINGS",
    )
    require(
        sealed.get("accepted_decision", {}).get("sha256")
        == EXPECTED_ACCEPTED_ARCHITECTURE_DECISION_HASH,
        "transition does not preserve the accepted architecture verdict",
    )
    historical = sealed.get("historical_mutable_path_observations", {})
    bindings = load(BINDINGS)["architecture_sources"]
    for name, source_name in (("target", "target"), ("fast_loop_policy", "fast_loop_policy")):
        observed = historical.get(name, {})
        source = bindings[source_name]
        require(
            observed.get("bytes") == source["bytes"] and observed.get("sha256") == source["sha256"],
            f"transition historical {name} observation differs from sealed architecture evidence",
        )
        require(
            observed.get("binding_scope") == "historical_mutable_path_observation",
            f"transition historical {name} observation is not explicitly scoped",
        )
    candidate = transition.get("candidate_transition", {})
    require(
        candidate.get("successor_candidate_rtl_hash") == precheck.get("candidate_rtl_hash"),
        "transition successor candidate differs from PRECHECK",
    )
    require(candidate.get("semantic_sources_unchanged") is True, "transition does not preserve RTL semantics")
    require(
        transition.get("manager_transition", {}).get("event") == pipeline.get("stage_history", [])[-1],
        "transition Manager event differs from live pipeline history",
    )
    return summary.current_records + summary.historical_records


def validate_contract() -> tuple[int, int]:
    for relative, expected in EXPECTED_IMMUTABLE_HASHES.items():
        require(sha256(ROOT / relative) == expected, f"immutable artifact changed: {relative}")
    require(sha256(BINDINGS) == EXPECTED_SEALED_BINDINGS_HASH, "sealed CONTRACT_BINDINGS changed")
    require(
        sha256(ARCH_DECISION) == EXPECTED_ACCEPTED_ARCHITECTURE_DECISION_HASH,
        "accepted architecture decision changed",
    )

    target = load(ROOT / "design/TARGET.json")
    policy = load(ROOT / "design/FAST_LOOP_POLICY.json")
    for path, value, expected_count in (
        ("design/TARGET.json", target, 4),
        ("design/FAST_LOOP_POLICY.json", policy, 3),
    ):
        records = [record for _, record in artifact_records(value) if record["path"] == PROBE_REL]
        require(len(records) == expected_count, f"unexpected probe binding count in {path}")
        for record in records:
            require(record["bytes"] == 529880, f"stale probe byte count in {path}")
            require(record["sha256"] == EXPECTED_IMMUTABLE_HASHES[PROBE_REL], f"stale probe hash in {path}")

    workload = (ROOT / "design/WORKLOAD.md").read_text(encoding="utf-8")
    require("For the current `shared_native_accumulator_tagged_attention_v1` contract" not in workload, "stale workload current-contract narrative")
    require(CONTRACT in workload, "successor workload contract missing")

    packet = load(PACKET)
    control = packet["control_semantics"]
    require("assert asynchronously" in control["reset"], "packet reset assertion semantics stale")
    require("deasserts synchronously" in control["reset"], "packet reset deassertion semantics stale")
    require("completion-gated rather than write-atomic" in control["completion"], "packet completion semantics stale")
    require(
        "discard" in control["completion"] and "overwrite" in control["completion"],
        "packet provisional destination recovery missing",
    )
    per_layer = packet["cycle_bandwidth_and_resource_budget"]["per_layer"]
    require("Unmeasured architecture planning assumption" in per_layer["fusion_cycle_basis"], "fusion latency overclaimed")
    require(per_layer["w4a8_macs"] == 4864 * 896, "per-layer MAC count equation mismatch")
    require(per_layer["projection_lanes"] == 4, "projection lane count changed")
    require(
        per_layer["down_projection_floor_cycles"]
        == per_layer["w4a8_macs"] // per_layer["projection_lanes"],
        "down-projection floor cycle equation mismatch",
    )
    require(per_layer["fusion_lanes"] == 1, "fusion lane count changed")
    require(
        per_layer["fusion_cycles"]
        == 896 * per_layer["fusion_cycles_per_output_lane"] // per_layer["fusion_lanes"],
        "fusion cycle equation mismatch",
    )
    require(
        per_layer["combined_cycles"]
        == per_layer["down_projection_floor_cycles"] + per_layer["fusion_cycles"],
        "combined cycle equation mismatch",
    )
    require(
        per_layer["token_dependent_external_bytes"] == 2179072 + 4864 + 896 + 896,
        "token-dependent external byte equation mismatch",
    )
    require(
        per_layer["dma_payload_floor_cycles_at_16_bytes_per_cycle"]
        == (per_layer["token_dependent_external_bytes"] + 15) // 16,
        "DMA payload floor equation mismatch",
    )
    require(
        per_layer["dma_planning_cycles_at_16_bytes_per_cycle"]
        == per_layer["dma_payload_floor_cycles_at_16_bytes_per_cycle"]
        + per_layer["dma_setup_cycles_per_descriptor"],
        "DMA planning cycle equation mismatch",
    )
    require_close(
        per_layer["fusion_over_down_projection_percent"],
        100.0 * per_layer["fusion_cycles"] / per_layer["down_projection_floor_cycles"],
        "fusion-over-down-projection ratio mismatch",
    )
    require_close(
        per_layer["arithmetic_intensity_macs_per_external_byte"],
        per_layer["w4a8_macs"] / per_layer["token_dependent_external_bytes"],
        "arithmetic intensity mismatch",
    )
    all_layers = packet["cycle_bandwidth_and_resource_budget"]["all_24_layers"]
    require(
        all_layers["added_fusion_cycles_per_token"] == 24 * per_layer["fusion_cycles"],
        "all-layer fusion cycle equation mismatch",
    )
    roofline = packet["cycle_bandwidth_and_resource_budget"]["roofline"]
    require_close(
        roofline["balance_point_bytes_per_cycle"],
        per_layer["token_dependent_external_bytes"] / per_layer["combined_cycles"],
        "roofline balance-point equation mismatch",
    )
    require_close(
        roofline["maximum_slice_speedup_if_fusion_cost_were_zero"],
        per_layer["combined_cycles"] / per_layer["down_projection_floor_cycles"],
        "maximum slice speedup equation mismatch",
    )
    timing = packet["cycle_bandwidth_and_resource_budget"]["area_timing_risks"]["timing"]
    require("not a frozen requirement" in timing, "divider schedule still frozen without derivation")

    layout = packet["interface_and_memory_layout"]["immutable_model_image_layout"]
    require(layout["address_space"] == "canonical_on_chip_sram_tagged_address", "fusion address space not frozen")
    require(layout["canonical_base_address"] == "0x800000000005c300", "fusion canonical base changed")
    require(layout["base_sram_byte_offset"] == 0x5C300, "fusion SRAM base offset mismatch")
    require(layout["allocation_bytes"] == 86720, "fusion allocation size mismatch")
    require(layout["immutable_image_bytes"] == 86592, "fusion immutable image size mismatch")
    require(layout["mutable_runtime_state_bytes"] == 128, "fusion runtime state size mismatch")
    require(
        layout["exclusive_end_sram_byte_offset"]
        == layout["base_sram_byte_offset"] + layout["allocation_bytes"]
        == 464320,
        "fusion allocation endpoint mismatch",
    )
    sections = layout["sections"]
    header = sections["checked_layer_headers"]
    pair = sections["layer_residual_destination_scale32_pairs"]
    accumulator = sections["accumulator_scale32_records"]
    state = sections["staging_and_error_state"]
    require(
        header["offset_bytes"] == 0
        and header["record_count"] == 24
        and header["record_bytes"] == header["record_stride_bytes"] == 16
        and header["bytes"] == header["record_count"] * header["record_bytes"],
        "fusion layer-header layout mismatch",
    )
    require(
        pair["offset_bytes"] == header["offset_bytes"] + header["bytes"]
        and pair["record_count"] == 24
        and pair["record_bytes"] == pair["record_stride_bytes"] == 8
        and pair["bytes"] == pair["record_count"] * pair["record_bytes"],
        "fusion residual/destination pair layout mismatch",
    )
    require(
        accumulator["offset_bytes"] == pair["offset_bytes"] + pair["bytes"]
        and accumulator["record_count"] == 24 * 896
        and accumulator["record_bytes"] == accumulator["lane_stride_bytes"] == 4
        and accumulator["layer_stride_bytes"] == 896 * accumulator["record_bytes"]
        and accumulator["bytes"] == accumulator["record_count"] * accumulator["record_bytes"],
        "fusion accumulator Scale32 layout mismatch",
    )
    require(
        state["offset_bytes"] == accumulator["offset_bytes"] + accumulator["bytes"]
        and state["offset_bytes"] == layout["immutable_image_bytes"]
        and state["bytes"] == layout["mutable_runtime_state_bytes"]
        and state["offset_bytes"] + state["bytes"] == layout["allocation_bytes"],
        "fusion runtime-state layout mismatch",
    )
    load_identity = layout["load_and_identity"]
    require("DMA_COPY" in load_identity["loader"], "fusion metadata loader not frozen")
    require("SHA-256" in load_identity["metadata_digest"], "fusion metadata digest not frozen")
    require("digest bytes 0..7" in load_identity["metadata_identity64"], "fusion identity64 derivation not frozen")
    require("fusion_metadata_valid" in load_identity["software_validation"], "fusion loader valid latch missing")
    require("descriptor_error before payload issue" in load_identity["hardware_prestart_validation"], "fusion pre-start identity failure action missing")
    require("scale_addr" in load_identity["descriptor_relationship"] and "does not" in load_identity["descriptor_relationship"], "fusion/ordinary metadata separation missing")
    lifetime = layout["lifetime"]
    require("read-only" in lifetime["immutable_sections"], "fusion immutable lifetime missing")
    require("reset" in lifetime["invalidation"] and "revalidation" in lifetime["invalidation"], "fusion invalidation lifetime missing")
    require("illegal" in lifetime["concurrency"], "fusion loader concurrency rule missing")

    discriminator = packet["minimal_two_dataset_discriminator"]
    baseline_requirements = discriminator["baseline_requirements"]
    require(len(baseline_requirements) == 3, "unexpected discriminator baseline requirement count")
    require("isolated per-layer construct replay" in baseline_requirements[1], "isolated construct replay requirement missing")
    require("layer-0 fusion injection boundary" in baseline_requirements[2], "composed layer-0 equality boundary missing")
    require("may diverge" in baseline_requirements[2], "recurrent downstream divergence permission missing")
    phases = discriminator["execution_phases"]
    isolated = phases["isolated_per_layer_construct_replay"]
    composed = phases["composed_recurrent_propagation"]
    require(isolated["layers"] == "0..23 inclusive", "isolated replay layer coverage mismatch")
    require("bit-identical" in isolated["required_equalities"], "isolated replay equality missing")
    require("independent scalar integer oracle" in isolated["required_result"], "isolated replay oracle missing")
    require("every layer 0..23" in composed["layers"], "composed hook coverage mismatch")
    require("layer-0" in composed["initial_equality_boundary"], "composed initial equality boundary missing")
    require("may differ" in composed["permitted_downstream_difference"], "composed downstream divergence permission missing")
    require(
        "remain baseline" in composed["unchanged_operator_requirement"]
        and "no independent candidate modification" in composed["unchanged_operator_requirement"],
        "composed unchanged-operator rule missing",
    )
    require("lm_head" in composed["trace_requirement"], "composed final trace requirement missing")

    memory = load(ROOT / "design/MEMORY_MODEL.json")
    require("unmeasured 12-cycle" in memory["compute_model"]["cycle_estimate_status"], "memory cycle uncertainty missing")
    require(memory["interface_contract"]["reset"].startswith("asynchronous_assertion"), "memory reset semantics stale")
    memory_mapping = memory["memory_hierarchy"]["candidate_metadata_mapping"]
    require(memory_mapping["base_sram_byte_offset"] == layout["base_sram_byte_offset"], "memory-model fusion base mismatch")
    require(memory_mapping["exclusive_end_sram_byte_offset"] == layout["exclusive_end_sram_byte_offset"], "memory-model fusion endpoint mismatch")
    require(memory_mapping["immutable_image_bytes"] == layout["immutable_image_bytes"], "memory-model immutable size mismatch")
    require(memory_mapping["runtime_state_bytes"] == layout["mutable_runtime_state_bytes"], "memory-model runtime size mismatch")
    for key, packet_key in (
        ("checked_layer_headers", "checked_layer_headers"),
        ("layer_residual_output_scale32_pairs", "layer_residual_destination_scale32_pairs"),
        ("accumulator_scale32_records", "accumulator_scale32_records"),
        ("staging_and_error_state", "staging_and_error_state"),
    ):
        require(
            memory_mapping["sections"][key]["offset_bytes"] == sections[packet_key]["offset_bytes"]
            and memory_mapping["sections"][key]["bytes"] == sections[packet_key]["bytes"],
            f"memory-model section mismatch: {key}",
        )

    hook_contract = load(HOOK_CONTRACT)
    hook_layout = hook_contract["metadata_image_contract"]
    require(hook_layout["canonical_base_address"] == layout["canonical_base_address"], "hook-contract fusion base mismatch")
    require(hook_layout["allocation_bytes"] == layout["allocation_bytes"], "hook-contract fusion allocation mismatch")
    require("isolated_per_layer_construct_replay" in hook_contract["discriminator_execution_phases"], "hook isolated replay phase missing")
    require("composed_recurrent_propagation" in hook_contract["discriminator_execution_phases"], "hook composed phase missing")
    require(
        "baseline_and_candidate_observe_identical_inputs_through_the_injection_boundary"
        not in hook_contract["construct_fidelity_checks"],
        "obsolete all-layer baseline-equality rule remains in hook contract",
    )

    provenance = load(PROVENANCE)
    metadata_origin = provenance["metadata_origin"]
    require("0x800000000005c300" in metadata_origin["layout_binding"], "provenance fusion base missing")
    require("SHA-256" in metadata_origin["load_and_identity_binding"], "provenance identity mechanism missing")
    require("revalidation" in metadata_origin["lifetime_binding"], "provenance lifetime rule missing")

    validate_public_status()

    binder_text = Path(__file__).read_text(encoding="utf-8")
    obsolete_area_token = "four_lane_added_area_" + "proxy_units"
    invalid_probe_claim = "probe_improvement_" + "percent"
    require(obsolete_area_token not in binder_text, "obsolete area proxy remains in binder")
    require(invalid_probe_claim not in binder_text, "invalid probe quality claim remains in binder")

    canonical_files = (PACKET, REDIRECT, FREEZE)
    for path in canonical_files:
        value = load(path)
        require(
            value["integrity"]["canonical_sha256"] == canonical_sha256(value),
            f"canonical digest stale: {path.relative_to(ROOT)}",
        )

    checked = validate_artifact_records() + validate_rtl_transition_binding()
    return checked, len(canonical_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate without modifying files")
    args = parser.parse_args()

    if not args.check:
        refresh()
    records, digests = validate_contract()
    print(
        "ACE2_DOWN_PROJECTION_RESIDUAL_FUSION_BINDING_PASS "
        f"contract={CONTRACT} artifact_records={records} canonical_digests={digests} "
        "implementation_authorized=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
