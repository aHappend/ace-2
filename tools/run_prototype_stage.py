#!/usr/bin/env python3
"""Bind ACE-2 current-stage prototype evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_STATE = ROOT / "research" / "PIPELINE_STATE.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
TARGET = ROOT / "design" / "TARGET.json"
CHIP_SCOPE = ROOT / "design" / "CHIP_SCOPE.json"
PPA_RESULTS = ROOT / "ppa" / "RESULTS.json"
VERIFICATION_RESULTS = ROOT / "verification" / "RESULTS.json"
RESULTS = ROOT / "prototype" / "RESULTS.json"
RAW_DIR = ROOT / "prototype" / "raw" / "latest"
RAW_LOG = RAW_DIR / "prototype_structured_na.log"
LIVE_VIEW = ROOT / ".argus" / "live-view.json"

STAGE_ORDER = [
    "definition",
    "architecture",
    "environment",
    "rtl",
    "verification",
    "ppa",
    "prototype",
    "benchmark",
    "signoff",
]

TRACKED_ARTIFACTS = [
    "MISSION.md",
    "research/PIPELINE_STATE.json",
    "research/PUBLIC_STATUS.json",
    "design/TARGET.json",
    "design/CHIP_SCOPE.json",
    "design/RTL_MANIFEST.json",
    "design/PPA_FRONTIER_LEDGER.json",
    "verification/RESULTS.json",
    "ppa/PROTOCOL.md",
    "ppa/RESULTS.json",
    "ppa/raw/latest/sky130_yosys.log",
    "ppa/raw/latest/sky130_sta.log",
    "ppa/raw/latest/sky130_power.log",
    "prototype/RESULTS.json",
    "prototype/raw/latest/prototype_structured_na.log",
    "tools/run_prototype_stage.py",
    "Makefile",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return {
        "path": rel,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def tree_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        item = artifact(rel)
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def stage_lists(pipeline: dict[str, Any]) -> tuple[list[str], list[str]]:
    current_stage = str(pipeline.get("current_stage", "unknown"))
    completed = [
        stage
        for stage in STAGE_ORDER
        if stage != current_stage and pipeline.get("stages", {}).get(stage, {}).get("status") == "done"
    ]
    index = STAGE_ORDER.index(current_stage) if current_stage in STAGE_ORDER else -1
    downstream = STAGE_ORDER[index + 1 :] if index >= 0 else []
    return completed, downstream


def require_current_stage(pipeline: dict[str, Any]) -> None:
    if pipeline.get("current_stage") != "prototype":
        raise RuntimeError("run_prototype_stage.py must only bind evidence while current_stage is prototype")


def require_pre_tapeout_structured_na(target: dict[str, Any], chip_scope: dict[str, Any]) -> None:
    delivery = target.get("delivery_contract", {})
    if delivery.get("delivery_level") != "pre_tapeout":
        raise RuntimeError("prototype structured N/A binder requires delivery_level=pre_tapeout")
    if delivery.get("fpga_claim_required_for_current_delivery_level") is not False:
        raise RuntimeError("prototype structured N/A binder requires FPGA claim not required for current delivery level")

    scope_delivery = chip_scope.get("delivery_scope", {})
    excluded = set(scope_delivery.get("excluded_claims", []))
    if "fabricated silicon" not in excluded:
        raise RuntimeError("prototype structured N/A binder requires fabricated silicon to remain excluded")


def require_prior_stage_evidence(ppa: dict[str, Any], verification: dict[str, Any]) -> None:
    if ppa.get("stage") != "ppa" or not all(ppa.get("checklist", {}).values()):
        raise RuntimeError("prototype binder requires passed PPA stage evidence in ppa/RESULTS.json")
    if verification.get("stage") != "verification" or not all(verification.get("checklist", {}).values()):
        raise RuntimeError("prototype binder requires passed verification evidence in verification/RESULTS.json")


def write_raw_log(generated_at: str, pipeline: dict[str, Any], target: dict[str, Any], ppa: dict[str, Any]) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_record = {
        "generated_at_utc": generated_at,
        "command": "make prototype-stage",
        "script": "tools/run_prototype_stage.py",
        "purpose": "Bind current prototype-stage evidence as structured N/A for the frozen pre_tapeout scope.",
        "manager_owned_current_stage": pipeline.get("current_stage"),
        "delivery_contract": target.get("delivery_contract", {}),
        "prototype_delivery_level": "structured_n_a",
        "hardware_claims": {
            "fpga": False,
            "emulator": False,
            "open_pdk_gds": False,
            "fabricated_silicon": False,
        },
        "applicability_decision": (
            "No FPGA, emulator, GDS, tapeout, or fabricated-silicon prototype is applicable "
            "to this current stage because the frozen delivery contract is pre_tapeout and "
            "explicitly does not require an FPGA claim. Current evidence remains bounded to "
            "verified RTL plus mapped SKY130 synthesis/STA/power from the prior PPA stage."
        ),
        "prior_ppa_summary": {
            "status": ppa.get("status"),
            "cells": ppa.get("frontier", {}).get("cells"),
            "non_sram_area_mm2": ppa.get("frontier", {}).get("non_sram_area_mm2"),
            "fmax_mhz": ppa.get("frontier", {}).get("fmax_mhz"),
            "first_unsupported_layer_operator": ppa.get("frontier", {}).get("first_unsupported_layer_operator"),
        },
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "input_artifacts": [
            artifact(rel)
            for rel in [
                "research/PIPELINE_STATE.json",
                "design/TARGET.json",
                "design/CHIP_SCOPE.json",
                "verification/RESULTS.json",
                "ppa/RESULTS.json",
                "tools/run_prototype_stage.py",
                "Makefile",
            ]
        ],
    }
    RAW_LOG.write_text(json.dumps(raw_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact("prototype/raw/latest/prototype_structured_na.log")


def update_public_status(results: dict[str, Any], pipeline: dict[str, Any]) -> None:
    public = load_json(PUBLIC_STATUS, {})
    completed, downstream = stage_lists(pipeline)
    frontier = results["prior_ppa_frontier"]
    dashboard = dict(public.get("dashboard_fields", {}))
    latest_ppa_frontier = dict(dashboard.get("latest_ppa_frontier", {}))
    latest_ppa_frontier.update({
        "status": results["prior_ppa_status"],
        "cells": frontier.get("cells"),
        "non_sram_area_mm2": frontier.get("non_sram_area_mm2"),
        "fmax_mhz": frontier.get("fmax_mhz"),
        "rtl_hash": frontier.get("rtl_hash"),
        "constraint_hash": frontier.get("constraint_hash"),
        "remaining_area_reserve_mm2": frontier.get("remaining_area_reserve_mm2"),
        "remaining_frequency_reserve_mhz": frontier.get("remaining_frequency_reserve_mhz"),
        "prototype_stage_binding": "structured_n_a_no_hardware_claim",
    })
    dashboard.update({
        "current_stage": "prototype",
        "current_mode": frontier.get("mode", dashboard.get("current_mode")),
        "latest_decision": results["decision"],
        "latest_ppa_frontier": latest_ppa_frontier,
        "latest_ppa_frontier_status": results["prior_ppa_status"],
        "latest_prototype_stage": {
            "status": results["status"],
            "delivery_level": results["prototype_delivery_level"]["declared_level"],
            "checklist": results["checklist"],
            "results": "prototype/RESULTS.json",
            "raw_log": "prototype/raw/latest/prototype_structured_na.log",
        },
        "ordered_supported_layer_operator_prefix": frontier.get(
            "ordered_supported_layer_operator_prefix",
            dashboard.get("ordered_supported_layer_operator_prefix", []),
        ),
        "first_unsupported_layer_operator": frontier.get("first_unsupported_layer_operator"),
    })

    public["schema_version"] = 1
    public["project"] = "ACE-2"
    public["vertical"] = "chip_design"
    public["generated_at_utc"] = results["generated_at_utc"]
    public["last_updated_utc"] = results["generated_at_utc"]
    public["stage"] = {
        "current_stage": "prototype",
        "completed_prior_stages": completed,
        "current_stage_source": "research/PIPELINE_STATE.json",
        "stage_transition_owner": "Manager",
        "planner_may_advance_stage": False,
        "downstream_locked_until_manager_advance": downstream,
        "current_stage_checklist": results["checklist"],
        "current_stage_evidence": [
            "prototype/RESULTS.json",
            "prototype/raw/latest/prototype_structured_na.log",
        ],
    }
    public["dashboard_fields"] = dashboard
    public["implementation_frontier"] = {
        "tracking_order_definition": "design/WORKLOAD.md#ordered-layer/operator-frontier",
        "ordered_supported_layer_operator_prefix": frontier.get("ordered_supported_layer_operator_prefix", []),
        "first_unsupported_layer_operator": frontier.get("first_unsupported_layer_operator"),
        "current_mode": frontier.get("mode", "ADVANCE"),
        "latest_decision": results["decision"],
        "latest_ppa_frontier": latest_ppa_frontier,
        "decision_policy": {
            "area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "target_authority": "operator_only",
        },
    }
    public["privacy_policy"] = {
        "contains_credentials": False,
        "contains_private_paths": False,
        "contains_private_pdk_contents": False,
        "contains_prompts": False,
        "contains_raw_private_logs": False,
        "public_safe": True,
    }

    old_claims = [
        claim
        for claim in public.get("public_claims", [])
        if isinstance(claim, dict)
        and not str(claim.get("claim", "")).startswith("current Manager-owned stage")
        and not str(claim.get("claim", "")).startswith("current prototype-stage")
        and not str(claim.get("claim", "")).startswith("prototype stage is structured N/A")
    ]
    public["public_claims"] = [
        {
            "claim": "current Manager-owned stage is prototype",
            "evidence": [
                "research/PIPELINE_STATE.json",
                "research/PUBLIC_STATUS.json",
                "prototype/RESULTS.json",
            ],
        },
        {
            "claim": (
                "current prototype-stage evidence is a structured N/A for the frozen pre_tapeout "
                "scope; no FPGA, emulator, GDS, tapeout, or fabricated-silicon prototype is claimed"
            ),
            "evidence": [
                "design/TARGET.json",
                "design/CHIP_SCOPE.json",
                "prototype/RESULTS.json",
                "prototype/raw/latest/prototype_structured_na.log",
            ],
        },
        {
            "claim": (
                "prior PPA evidence remains bounded to mapped SKY130 synthesis/OpenSTA timing "
                "and activity power for the supported prefix; benchmark and signoff remain locked downstream"
            ),
            "evidence": [
                "ppa/RESULTS.json",
                "ppa/raw/latest/",
                "design/PPA_FRONTIER_LEDGER.json",
            ],
        },
        *old_claims,
    ]

    explicit_non_claims = [
        claim
        for claim in public.get("explicit_non_claims", [])
        if "FPGA prototype" not in str(claim) and "routed Open-PDK GDS" not in str(claim)
    ]
    public["explicit_non_claims"] = [
        "no FPGA, emulator, board, GDS, tapeout-readiness, or fabricated-silicon prototype is claimed by the prototype stage",
        "no on-hardware frequency, resource, power, correctness, host-runtime, or benchmark measurement exists because hardware prototype evidence is not applicable to the frozen pre_tapeout delivery scope",
        "benchmark and signoff stages remain locked until Manager advance",
        *explicit_non_claims,
    ]

    old_hashes = {
        item["path"]: item
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and "path" in item
    }
    for rel in TRACKED_ARTIFACTS:
        old_hashes[rel] = artifact(rel)
    public["artifact_hashes"] = [old_hashes[key] for key in sorted(old_hashes)]
    public["blockers"] = []
    write_json(PUBLIC_STATUS, public)


def update_live_view() -> None:
    write_json(
        LIVE_VIEW,
        {
            "version": 1,
            "title": "Prototype structured N/A bound",
            "paths": [
                "prototype/RESULTS.json",
                "prototype/raw/latest/prototype_structured_na.log",
                "design/TARGET.json",
                "ppa/RESULTS.json",
                "research/PUBLIC_STATUS.json",
            ],
            "reason": (
                "Shows the current prototype-stage structured N/A evidence for the frozen "
                "pre_tapeout scope, bound to prior PPA evidence without broader hardware claims."
            ),
        },
    )


def main() -> None:
    if sys.argv[1:]:
        raise RuntimeError("usage: run_prototype_stage.py")

    pipeline = load_json(PIPELINE_STATE, {})
    target = load_json(TARGET, {})
    chip_scope = load_json(CHIP_SCOPE, {})
    ppa = load_json(PPA_RESULTS, {})
    verification = load_json(VERIFICATION_RESULTS, {})

    require_current_stage(pipeline)
    require_pre_tapeout_structured_na(target, chip_scope)
    require_prior_stage_evidence(ppa, verification)

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    raw_log_artifact = write_raw_log(generated_at, pipeline, target, ppa)
    frontier = dict(ppa.get("frontier", {}))

    results = {
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "prototype",
        "generated_at_utc": generated_at,
        "status": "structured_na_pass_pending_manager_review",
        "decision": "prototype_structured_na_pre_tapeout_no_hardware_claim_manager_review_next",
        "checklist": {
            "prototype.delivery-level": True,
            "prototype.hardware-evidence": True,
        },
        "prototype_delivery_level": {
            "declared_level": "structured_n_a",
            "reason": (
                "The frozen delivery contract is pre_tapeout and design/TARGET.json states "
                "that an FPGA claim is not required for the current delivery level. A GDS, "
                "tapeout, fabricated silicon, or on-board prototype would be a broader claim "
                "than the current stage can make."
            ),
            "consistent_with_frozen_scope": True,
            "contains_broader_hardware_claim": False,
            "non_applicable_levels": {
                "fpga": "not_claimed_not_required",
                "emulator": "not_claimed_not_required",
                "open_pdk_gds": "not_claimed_signoff_stage_locked",
                "fabricated_silicon": "not_claimed_excluded_from_delivery_scope",
            },
        },
        "hardware_evidence": {
            "applicability": "not_applicable_for_structured_n_a",
            "tool_identity": "not_applicable_no_hardware_or_gds_tool_run",
            "board_identity": "not_applicable_no_fpga_board",
            "chip_identity": "not_applicable_no_fabricated_chip",
            "build_artifact": None,
            "clocks": "not_applicable_no_hardware_clock",
            "resources": "not_applicable_no_fpga_or_emulator_resource_report",
            "host_runtime_integration": "not_applicable_no_hardware_runtime",
            "on_hardware_correctness": "not_run_not_applicable",
            "power": "not_measured_not_applicable",
            "raw_commands_and_logs": [
                raw_log_artifact,
            ],
            "claim_boundary": (
                "The prototype stage records structured N/A only. It does not transform prior "
                "RTL, verification, or PPA evidence into an FPGA, emulator, GDS, tapeout, or "
                "silicon claim."
            ),
        },
        "prior_ppa_status": ppa.get("status"),
        "prior_ppa_frontier": frontier,
        "source_hashes": {
            "prototype_binding_hash": tree_hash(
                [
                    "research/PIPELINE_STATE.json",
                    "design/TARGET.json",
                    "design/CHIP_SCOPE.json",
                    "verification/RESULTS.json",
                    "ppa/RESULTS.json",
                    "prototype/raw/latest/prototype_structured_na.log",
                    "tools/run_prototype_stage.py",
                    "Makefile",
                ]
            ),
            "input_artifacts": [
                artifact(rel)
                for rel in [
                    "research/PIPELINE_STATE.json",
                    "design/TARGET.json",
                    "design/CHIP_SCOPE.json",
                    "verification/RESULTS.json",
                    "ppa/RESULTS.json",
                    "prototype/raw/latest/prototype_structured_na.log",
                    "tools/run_prototype_stage.py",
                    "Makefile",
                ]
            ],
        },
        "manager_stage_transition_owner": "Manager; this tool does not edit research/PIPELINE_STATE.json.",
        "downstream_locked": ["benchmark", "signoff"],
    }

    write_json(RESULTS, results)
    update_public_status(results, pipeline)
    update_live_view()
    print(
        "ACE2_PROTOTYPE_STAGE_PASS "
        "delivery_level=structured_n_a "
        f"prior_area_mm2={frontier.get('non_sram_area_mm2')} "
        f"prior_fmax_mhz={frontier.get('fmax_mhz')} "
        "hardware_claims=none"
    )


if __name__ == "__main__":
    main()
