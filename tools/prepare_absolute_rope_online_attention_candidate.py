#!/usr/bin/env python3
"""Bind the focused absolute-RoPE candidate without publishing the frontier."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "layer0_absolute_rope_online_attention_v1"
PROPOSAL_SHA256 = "b14489f8c882ff53f27db87612238efa06ef8d102930da15b2916ef314b683de"
EVIDENCE = ROOT / "evidence" / CONTRACT / "latest"
APPROVAL = ROOT / "evidence" / "authorization" / CONTRACT / "operator_approval.json"
SOURCES = [
    "Makefile",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    f"evidence/authorization/{CONTRACT}/operator_approval.json",
    "rtl/ace2_absolute_rope_score_core.sv",
    "rtl/ace2_absolute_rope_online_attention_core.sv",
    "rtl/generated/ace2_exp_q31_lut.svh",
    "tools/ace2_absolute_rope_online_attention_reference.py",
    "tools/ace2_full_model_fixed_point.py",
    "tools/ace2_quality_contracts.py",
    "tools/gen_absolute_rope_online_attention_vectors.py",
    "verification/generated/absolute_rope_online_attention_vectors.json",
    "verification/generated/absolute_rope_online_attention_vectors.svh",
    "verification/tb/ace2_absolute_rope_score_tb.sv",
    "verification/tb/ace2_absolute_rope_online_attention_tb.sv",
    "verification/test_absolute_rope_online_attention.py",
]
LOGS = {
    "software_reference": "evidence/layer0_absolute_rope_online_attention_v1/latest/software_reference_unittest.log",
    "rtl_score_simulation": "evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_absolute_rope_score.log",
    "rtl_online_simulation": "evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_online_attention.log",
    "rtl_score_lint": "evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_score_lint.log",
    "rtl_online_lint": "evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_online_lint.log",
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {relative}")
    return value


def write(relative: str, value: dict[str, Any]) -> None:
    (ROOT / relative).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"required artifact missing: {relative}")
    return {"bytes": path.stat().st_size, "path": relative, "sha256": sha256(path)}


def main() -> None:
    if load("research/PIPELINE_STATE.json").get("current_stage") != "rtl":
        raise RuntimeError("candidate binder requires Manager-owned rtl stage")
    if sha256(ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md") != PROPOSAL_SHA256:
        raise RuntimeError("proposal hash changed")
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if approval.get("contract_id") != CONTRACT or approval.get("authority") != "operator":
        raise RuntimeError("exact operator approval is not bound")
    for label, relative in LOGS.items():
        item = artifact(relative)
        text = (ROOT / relative).read_text(encoding="utf-8")
        if label == "software_reference" and "Ran 5 tests" not in text:
            raise RuntimeError("software focused test count differs")
        if label == "rtl_score_simulation" and "TB_PASS cases=3" not in text:
            raise RuntimeError("score RTL simulation did not pass")
        if label == "rtl_online_simulation" and "TB_PASS" not in text:
            raise RuntimeError("online RTL simulation did not pass")
        if "lint" in label and text.strip():
            raise RuntimeError(f"{label} is not warning-free")
        if item["sha256"] == hashlib.sha256(b"").hexdigest() and "lint" not in label:
            raise RuntimeError(f"unexpected empty evidence: {label}")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    source_lines = []
    source_hashes: dict[str, str] = {}
    for relative in sorted(SOURCES):
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"candidate source missing: {relative}")
        digest = sha256(path)
        source_hashes[relative] = digest
        source_lines.append(f"{digest}  {relative}")
    source_list = EVIDENCE / "source_hashes.txt"
    source_list.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    rtl_hash = sha256(source_list)
    candidate_id = f"layer0_absolute_rope_online_attention_{rtl_hash[:16]}"

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update({
        "candidate_layer_operator": CONTRACT,
        "candidate_status": "focused_software_rtl_pass_unique_paired_smoke_pending",
        "candidate_rtl_hash": rtl_hash,
        "candidate_rtl_hash_scope": "hash_bound_absolute_rope_score_online_recurrence_finalizer_generated_lut_and_full_model_candidate_not_shell_admitted",
        "candidate_source_hashes": source_hashes,
        "candidate_generated_hashes": {
            key: source_hashes[key]
            for key in (
                "rtl/generated/ace2_exp_q31_lut.svh",
                "verification/generated/absolute_rope_online_attention_vectors.json",
                "verification/generated/absolute_rope_online_attention_vectors.svh",
            )
        },
        "candidate_geometry": {
            "layer_scope": 0,
            "query_heads": 14,
            "kv_heads": 2,
            "head_dim": 64,
            "rope_pairs": 32,
            "score_width_bits": 54,
            "logit_format": "signed_Q12_20_in_s64",
            "weight_format": "unsigned_Q1_31",
            "online_denominator_width_bits": 48,
            "online_numerator_width_bits": 56,
            "stored_wide_query_bytes": 3584,
            "online_state_bytes": 7616,
            "planned_sram_peak_bytes": 504400,
            "planned_sram_margin_bytes": 19888,
            "shared_multiplier": "one_time_multiplexed_unsigned_32x32_with_signed_magnitude_control",
        },
        "candidate_interface": {
            "source_files": [
                "rtl/ace2_absolute_rope_score_core.sv",
                "rtl/ace2_absolute_rope_online_attention_core.sv",
            ],
            "modules": [
                {
                    "module": "ace2_absolute_rope_score_core",
                    "parameter": {"PAIR_COUNT": 32},
                    "role": "stream_32_split_half_pairs_to_signed54_score_and_q12_20_logit",
                    "handshake": "start_ready_valid_pair_ready_valid_out_ready_valid",
                },
                {
                    "module": "ace2_absolute_rope_online_attention_core",
                    "parameter": {"LANE_COUNT": 64},
                    "role": "one_key_online_max_denominator_and_64_numerator_update",
                    "handshake": "start_ready_valid_lane_ready_valid_lane_out_ready_valid",
                },
                {
                    "module": "ace2_absolute_rope_online_attention_finalize_core",
                    "parameter": {},
                    "role": "serialized_56_step_signed_rne_divide_and_int8_saturation",
                    "handshake": "start_ready_valid_out_ready_valid",
                },
            ],
            "shell_descriptor_integration": "not_admitted_before_unique_paired_smoke_pass",
            "status": "focused_candidate_arithmetic_interfaces_frozen",
        },
        "candidate_verification_complete": True,
        "candidate_verification_binding": {
            "classification": "focused_software_and_standalone_rtl_not_independent_verification_stage",
            **{name: artifact(path) for name, path in LOGS.items()},
        },
        "candidate_meets_numeric_acceptance": False,
        "candidate_supported_layer_operator_prefix_after_review": manifest[
            "supported_layer_operator_prefix"
        ],
        "candidate_first_unsupported_layer_operator_after_review": "layer_0.rope_q",
        "candidate_requires_fresh_sky130_ppa": False,
        "interfaces_contract_status": "focused_arithmetic_bound_shell_descriptor_integration_gated_by_paired_smoke",
        "generated_at_utc": utc_now(),
    })
    replacement = manifest["proposed_replacement_contract"]
    replacement["status"] = "implementation_complete_focused_tests_pass_paired_smoke_pending"
    replacement["implementation_result"] = {
        "rtl_change_started": True,
        "focused_software_rtl_tests_run": True,
        "focused_software_rtl_tests_passed": True,
        "paired_smoke_run": False,
        "accepted_rtl_change": False,
        "accepted_verification_evidence": False,
        "accepted_ppa_evidence": False,
    }
    manifest["traceability"].update({
        "architecture_contract_gap": {
            "status": "arithmetic_implemented_shell_descriptor_integration_gated_by_paired_smoke",
            "resolution_owner": "paired smoke stop rule",
        },
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    })
    manifest["traceability"]["stage_checklist"] = {
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }
    for entry in (
        {
            "kind": "new_project_rtl",
            "license": "repository project license not separately declared in this manifest",
            "name": "ace2_absolute_rope_score_core",
            "path": "rtl/ace2_absolute_rope_score_core.sv",
            "sha256": source_hashes["rtl/ace2_absolute_rope_score_core.sv"],
            "source_revision": "active_worktree",
            "third_party": False,
        },
        {
            "kind": "new_project_rtl",
            "license": "repository project license not separately declared in this manifest",
            "name": "ace2_absolute_rope_online_attention_core_and_finalizer",
            "path": "rtl/ace2_absolute_rope_online_attention_core.sv",
            "sha256": source_hashes["rtl/ace2_absolute_rope_online_attention_core.sv"],
            "source_revision": "active_worktree",
            "third_party": False,
        },
        {
            "kind": "generated_project_rtl_include",
            "license": "repository project license not separately declared in this manifest",
            "name": "ace2_exp_q31_lut",
            "path": "rtl/generated/ace2_exp_q31_lut.svh",
            "sha256": source_hashes["rtl/generated/ace2_exp_q31_lut.svh"],
            "generator": "tools/gen_absolute_rope_online_attention_vectors.py",
            "regeneration_command": "python tools/gen_absolute_rope_online_attention_vectors.py",
            "source_revision": "active_worktree",
            "third_party": False,
        },
    ):
        manifest["ip_provenance"] = [
            item for item in manifest["ip_provenance"] if item.get("name") != entry["name"]
        ]
        manifest["ip_provenance"].append(entry)
    manifest["claim_boundaries"] = list(dict.fromkeys(
        manifest["claim_boundaries"] + [
            "The absolute-RoPE candidate has focused arithmetic evidence only and is not an accepted shell capability.",
            "The historical 0.6108746272 mm2/+0.1502 ns frontier is unchanged; no PPA or full-shell run occurred.",
            "A failure or equality on either unique paired-smoke dataset seals this candidate as a bounded no-go.",
        ]
    ))
    write("design/RTL_MANIFEST.json", manifest)
    manifest_hash = sha256(ROOT / "design/RTL_MANIFEST.json")

    candidate = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "contract_id": CONTRACT,
        "proposal_sha256": PROPOSAL_SHA256,
        "recorded_at_utc": utc_now(),
        "status": "focused_software_rtl_pass_unique_paired_smoke_pending",
        "acceptance_claim": False,
        "accepted_publication_frontier_changed": False,
        "official_14_item_evaluation_run": False,
        "full_shell_regression_run": False,
        "canonical_sky130_ppa_run": False,
        "operator_approval": {
            "authority": "operator",
            "approved_at_utc": approval["approved_at_utc"],
            "evidence": APPROVAL.relative_to(ROOT).as_posix(),
            "evidence_sha256": sha256(APPROVAL),
            "scope": approval["scope"],
        },
        "source_binding": {
            "ordered_source_hash_list_sha256": rtl_hash,
            "source_hash_list": source_list.relative_to(ROOT).as_posix(),
        },
        "focused_verification": {
            name: {"status": "pass", **artifact(path)} for name, path in LOGS.items()
        },
        "accepted_frontier_preservation": {
            "rtl_manifest_path": "design/RTL_MANIFEST.json",
            "rtl_manifest_sha256": manifest_hash,
            "ppa_frontier_ledger_path": "design/PPA_FRONTIER_LEDGER.json",
            "ppa_frontier_ledger_sha256": sha256(ROOT / "design/PPA_FRONTIER_LEDGER.json"),
            "non_sram_area_mm2": 0.6108746272,
            "setup_slack_ns_at_100mhz": 0.1502,
        },
        "frozen_stop_rule": {
            "wikitext2_ratio_strictly_below": 13549.939049967887,
            "c4_en_512_ratio_strictly_below": 4477.990517308544,
            "failure_action": "seal_bounded_no_go_without_full_shell_ppa_or_official_14",
        },
    }
    write(f"evidence/{CONTRACT}/latest/candidate_evidence.json", candidate)

    policy = load("design/FAST_LOOP_POLICY.json")
    policy["active_repair_authorization"]["execution_status"] = (
        "focused_software_rtl_pass_unique_paired_smoke_pending"
    )
    policy["active_repair_authorization"]["observed_candidate_rtl_hash"] = rtl_hash
    policy["selected_replacement_contract"]["execution_result"] = {
        "rtl_change_started": True,
        "focused_software_rtl_tests_run": True,
        "focused_software_rtl_tests_passed": True,
        "paired_smoke_run": False,
        "full_shell_regression_run": False,
        "canonical_sky130_ppa_run": False,
        "official_14_item_evaluation_run": False,
    }
    policy["selected_replacement_contract"]["status"] = (
        "focused_software_rtl_pass_unique_paired_smoke_pending"
    )
    write("design/FAST_LOOP_POLICY.json", policy)

    status = load("research/PUBLIC_STATUS.json")
    status["dashboard_fields"]["candidate_rtl_hash"] = rtl_hash
    status["dashboard_fields"]["candidate_mechanism"]["status"] = (
        "focused_software_rtl_pass_unique_paired_smoke_pending"
    )
    status["dashboard_fields"]["latest_decision"] = (
        "absolute_rope_focused_pass_unique_paired_smoke_pending"
    )
    status["implementation_frontier"]["latest_decision"] = (
        "absolute_rope_focused_pass_unique_paired_smoke_pending"
    )
    status["implementation_frontier"]["latest_rtl_candidate"] = {
        "candidate_id": candidate_id,
        "evidence": f"evidence/{CONTRACT}/latest/candidate_evidence.json",
        "rtl_hash": rtl_hash,
        "rtl_hash_scope": manifest["candidate_rtl_hash_scope"],
        "stage_closing": False,
        "status": "focused_pass_unique_paired_smoke_pending_not_accepted_frontier",
    }
    status["latest_decision"] = "absolute_rope_focused_pass_unique_paired_smoke_pending"
    status["stage"]["current_stage_checklist"] = manifest["traceability"]["stage_checklist"]
    status["stage"]["current_stage_status"] = "focused_candidate_paired_smoke_pending"
    write("research/PUBLIC_STATUS.json", status)

    oracle = load("reference/ORACLE_MANIFEST.json")
    oracle["proposed_architecture_contract"]["status"] = (
        "focused_vectors_generated_and_passed_unique_paired_smoke_pending"
    )
    oracle["proposed_architecture_contract"]["focused_vector_binding"] = {
        "generator": artifact("tools/gen_absolute_rope_online_attention_vectors.py"),
        "json": artifact("verification/generated/absolute_rope_online_attention_vectors.json"),
        "svh": artifact("verification/generated/absolute_rope_online_attention_vectors.svh"),
        "exp_lut": artifact("rtl/generated/ace2_exp_q31_lut.svh"),
        "regeneration_command": "python tools/gen_absolute_rope_online_attention_vectors.py",
    }
    write("reference/ORACLE_MANIFEST.json", oracle)

    trace_path = ROOT / "design/RTL_TRACEABILITY.md"
    trace = trace_path.read_text(encoding="utf-8")
    trace += f"""

## Bounded implementation candidate

- Candidate: `{candidate_id}`.
- Ordered source-list SHA-256: `{rtl_hash}`.
- `ace2_absolute_rope_score_core` traces the split-half signed-25 rotation,
  signed-54 score, Scale32 validation, shared 32x32 limb scaling, and Q12.20 RNE.
- `ace2_absolute_rope_online_attention_core` traces the Q1.31 interpolation,
  both online maximum branches, unsigned-48 denominator, signed-56 numerators,
  single shared 32x32 limb rescaling, and ready/valid stability.
- `ace2_absolute_rope_online_attention_finalize_core` traces the 56-step
  signed RNE ratio and signed-int8 saturation.
- The generated 257-entry LUT is bound to generator command
  `python tools/gen_absolute_rope_online_attention_vectors.py`.
- Focused software, score RTL, online-state RTL, and warning-free lint pass.
- `rtl.contract-traceability` remains false because shell descriptor/workspace
  integration is deliberately gated by the mandatory paired smoke. The
  accepted prefix and historical PPA frontier are unchanged.
"""
    trace_path.write_text(trace, encoding="utf-8")
    print(
        "ACE2_ABSOLUTE_ROPE_CANDIDATE_BOUND "
        f"candidate_id={candidate_id} rtl_hash={rtl_hash} paired_smoke=pending"
    )


if __name__ == "__main__":
    main()
