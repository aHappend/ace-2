#!/usr/bin/env python3
"""Bind the focused tile-BFP RTL implementation into RTL_MANIFEST.json."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
CONTRACT = "layer0_tile_bfp_score_attention_v1"
FILES = [
    "rtl/ace2_tile_bfp_score_attention_core.sv",
    "tools/ace2_tile_bfp_reference.py",
    "tools/gen_tile_bfp_attention_vectors.py",
    "verification/generated/tile_bfp_attention_vectors.json",
    "verification/generated/tile_bfp_attention_vectors.svh",
    "verification/tb/ace2_tile_bfp_attention_tb.sv",
    "verification/test_tile_bfp_attention.py",
]
EVIDENCE = [
    f"evidence/{CONTRACT}/latest/software_reference_unittest.log",
    f"evidence/{CONTRACT}/latest/full_model_self_test.log",
    f"evidence/{CONTRACT}/latest/runtime_dependency.log",
    f"evidence/{CONTRACT}/latest/rtl_tile_bfp.log",
    f"evidence/{CONTRACT}/latest/rtl_score_lint.log",
    f"evidence/{CONTRACT}/latest/rtl_softmax_lint.log",
    f"evidence/{CONTRACT}/focused-candidate-full-20260801-v1/results.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"missing tile-BFP artifact: {relative}")
    return {"bytes": path.stat().st_size, "path": relative, "sha256": sha256(path)}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def updated_manifest() -> dict[str, object]:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_hashes = {relative: sha256(ROOT / relative) for relative in FILES}
    value.update({
        "stage": "rtl",
        "current_stage": "rtl",
        "architecture_contract_status": f"{CONTRACT}_focused_rtl_gate_pass_paired_smoke_pending",
        "interfaces_contract_status": "accepted_shell_prefix_through_v_proj_tile_bfp_focused_standalone_rtl_bound_not_shell_admitted",
        "candidate_status": "focused_rtl_and_two_dataset_discriminator_pass_unique_paired_smoke_pending",
        "candidate_rtl_hash": source_hashes["rtl/ace2_tile_bfp_score_attention_core.sv"],
        "candidate_rtl_hash_scope": "tile_bfp_reference_generated_vectors_standalone_rtl_and_focused_quality_not_shell_admitted",
        "candidate_source_hashes": source_hashes,
        "candidate_generated_hashes": {
            relative: source_hashes[relative]
            for relative in FILES if relative.startswith("verification/generated/")
        },
        "candidate_meets_numeric_acceptance": False,
        "candidate_requires_fresh_sky130_ppa": True,
        "candidate_verification_complete": False,
        "generated_at_utc": utc_now(),
    })
    value["candidate_interface"] = {
        "status": "focused_standalone_rtl_bound_not_shell_admitted",
        "modules": [
            {
                "module": "ace2_tile_bfp_score_core",
                "parameters": {"MAX_KEYS": 64},
                "ports": {
                    "clk_i": 1, "rst_ni": 1, "clear_i": 1,
                    "start_valid_i": 1, "start_ready_o": 1,
                    "key_count_u7_i": 7, "score_exponent_s8_i": 8,
                    "score_valid_i": 1, "score_ready_o": 1,
                    "score_num_s106_i": 106, "metadata_valid_o": 1,
                    "metadata_ready_i": 1, "tile_max_s128_o": 128,
                    "fraction_bits_u5_o": 5, "mantissa_valid_o": 1,
                    "mantissa_ready_i": 1, "mantissa_index_u6_o": 6,
                    "mantissa_s24_o": 24, "mantissa_last_o": 1,
                    "descriptor_error_o": 1, "numeric_overflow_o": 1,
                },
            },
            {
                "module": "ace2_bfp_hierarchical_softmax_core",
                "parameters": {},
                "ports": {
                    "clk_i": 1, "rst_ni": 1, "clear_i": 1,
                    "start_valid_i": 1, "start_ready_o": 1,
                    "key_count_u7_i": 7, "score_exponent_s8_i": 8,
                    "fraction_bits_u5_i": 5, "row_max_s128_i": 128,
                    "tile_max_s128_i": 128, "mantissa_valid_i": 1,
                    "mantissa_ready_o": 1, "mantissa_s24_i": 24,
                    "weight_valid_o": 1, "weight_ready_i": 1,
                    "weight_q1_31_o": 32, "weight_last_o": 1,
                    "sum_valid_o": 1, "sum_ready_i": 1,
                    "weight_sum_u48_o": 48, "norm_valid_i": 1,
                    "norm_ready_o": 1, "norm_weight_q1_31_i": 32,
                    "norm_denominator_u48_i": 48, "prob_valid_o": 1,
                    "prob_ready_i": 1, "probability_q0_15_o": 15,
                    "descriptor_error_o": 1, "numeric_overflow_o": 1,
                },
            },
        ],
    }
    value["candidate_verification_binding"] = {
        "status": "focused_native_and_two_dataset_discriminator_pass_paired_smoke_pending",
        "evidence": [record(relative) for relative in EVIDENCE],
    }
    value["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "focused_discriminator_passed": True,
        "paired_smoke_run": False,
        "stage_closing": False,
        "status": "unique_paired_smoke_authorized",
    }
    value["traceability"].update({
        "architecture_contract_gap": {
            "resolution_owner": "bounded RTL task",
            "status": "resolved_for_focused_standalone_candidate",
        },
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
        "stage_checklist": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
    })
    value["candidate_rtl_sources"] = [record("rtl/ace2_tile_bfp_score_attention_core.sv")]
    value["candidate_generated_sources"] = [
        {
            **record("verification/generated/tile_bfp_attention_vectors.json"),
            "generator": "tools/gen_tile_bfp_attention_vectors.py",
            "reference": "tools/ace2_tile_bfp_reference.py",
            "regeneration_command": ".venv/bin/python tools/gen_tile_bfp_attention_vectors.py",
            "third_party": False,
        },
        {
            **record("verification/generated/tile_bfp_attention_vectors.svh"),
            "generator": "tools/gen_tile_bfp_attention_vectors.py",
            "reference": "tools/ace2_tile_bfp_reference.py",
            "regeneration_command": ".venv/bin/python tools/gen_tile_bfp_attention_vectors.py",
            "third_party": False,
        },
    ]
    value["ip_provenance"] = [
        item for item in value["ip_provenance"]
        if item.get("name") not in {"ace2_tile_bfp_score_core_and_softmax", "tile_bfp_attention_vectors"}
    ] + [
        {
            "kind": "first_party_bounded_candidate_rtl",
            "license": "repository project license not separately declared in this manifest",
            "name": "ace2_tile_bfp_score_core_and_softmax",
            "path": "rtl/ace2_tile_bfp_score_attention_core.sv",
            "sha256": source_hashes["rtl/ace2_tile_bfp_score_attention_core.sv"],
            "source_revision": source_hashes["rtl/ace2_tile_bfp_score_attention_core.sv"],
            "third_party": False,
        },
        {
            "generator": "tools/gen_tile_bfp_attention_vectors.py",
            "kind": "generated_bounded_candidate_verification_source",
            "name": "tile_bfp_attention_vectors",
            "reference": "tools/ace2_tile_bfp_reference.py",
            "regeneration_command": ".venv/bin/python tools/gen_tile_bfp_attention_vectors.py",
            "third_party": False,
        },
    ]
    value["latest_evidence"]["tile_bfp_focused_gate"] = record(
        f"evidence/{CONTRACT}/focused-candidate-full-20260801-v1/results.json"
    )
    return value


def main() -> None:
    expected = json.dumps(updated_manifest(), indent=2, sort_keys=True) + "\n"
    MANIFEST.write_text(expected, encoding="utf-8")
    print(f"ACE2_TILE_BFP_RTL_STATE_BOUND manifest_sha256={sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
