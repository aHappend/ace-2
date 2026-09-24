#!/usr/bin/env python3
"""Build the compact environment-stage review packet for the active contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
OUT = ROOT / "evidence/environment_preflight/latest/PACKET.json"
AUDIT = ROOT / "research/ENVIRONMENT_AUDIT.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
TOOLCHAIN = ROOT / "research/TOOLCHAIN_CANDIDATES.md"
IP_REUSE = ROOT / "research/IP_REUSE_PLAN.md"
PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
ARCH_REVIEW = ROOT / f"evidence/review/architecture_refreeze_{CONTRACT}/decision.json"
PROBE = ROOT / "research/raw/environment/down_projection_residual_fusion_compatibility/RESULTS.json"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_record(record: dict[str, Any], label: str) -> None:
    path = ROOT / str(record.get("path", ""))
    require(path.is_file(), f"{label} missing")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label} byte mismatch")
    require(sha256(path) == record.get("sha256"), f"{label} hash mismatch")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    return hashlib.sha256((json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def run_check(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=180,
    )
    require(completed.returncode == 0, f"check failed: {' '.join(command)}\n{completed.stdout}")
    return completed.stdout.strip()


def build() -> None:
    audit = load(AUDIT)
    pipeline = load(PIPELINE)
    public = load(PUBLIC)
    packet = load(PACKET)
    review = load(ARCH_REVIEW)
    probe = load(PROBE)

    require(pipeline.get("current_stage") == "environment", "Manager stage is not environment")
    require(audit.get("contract_binding", {}).get("active_contract_id") == CONTRACT, "audit contract mismatch")
    require(packet.get("contract_id") == CONTRACT, "architecture packet contract mismatch")
    require(review.get("reviewer_status") == "done" and review.get("architecture_accepted") is True,
            "architecture review is not accepted")
    require(review.get("packet", {}).get("sha256") == sha256(PACKET), "architecture review packet binding stale")
    require(probe.get("status") == "pass", "contract delta probe is not pass")
    require(public.get("stage", {}).get("current_stage") == "environment", "public stage is stale")
    require(public.get("selected_replacement_contract", {}).get("contract_id") == CONTRACT, "public contract mismatch")
    require(public.get("supported_layer_operator_prefix") == PREFIX, "public supported prefix changed")
    require(public.get("first_unsupported_layer_operator") == "layer_0.rope_q", "first unsupported operator changed")

    architecture_check = run_check([sys.executable, "tools/bind_down_projection_residual_fusion_architecture.py", "--check"])
    environment_check = run_check([sys.executable, "tools/bind_environment_compatibility.py", "--check"])
    probe_check = run_check([sys.executable, "tools/run_down_projection_environment_probe.py", "--check"])
    project_python = shutil.which("python") or sys.executable
    exact_hook_check = run_check([project_python, "tools/ace2_down_projection_residual_fusion_hook.py", "--self-test"])

    preserved = packet["preserved_operator_contract"]
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "project": "ACE-2",
        "stage": "environment",
        "stage_closing": True,
        "contract_id": CONTRACT,
        "status": "pass_ready_for_independent_environment_review",
        "implementation_authorized": False,
        "checklist": {
            "environment.eda-capabilities": True,
            "environment.tool-ip-selection": True,
        },
        "compatibility_findings": {
            "bootstrap_capability_artifacts_hash_verified": 27,
            "bootstrap_raw_probes_rerun": False,
            "contract_specific_wide_arithmetic_probe": "pass",
            "new_eda_pdk_board_license_compiler_runtime_or_external_ip_class": False,
            "maintained_tool_and_ip_selection_current": True,
            "architecture_review_current_and_accepted": True,
            "implementation_remains_stage_locked": True,
            "forbidden_downstream_work_respected": True,
        },
        "contract_delta": {
            "signed_numerator_bits_minimum": 96,
            "unsigned_denominator_bits_minimum": 64,
            "division": "variable_integer_divide_and_remainder",
            "rounding": "signed_round_to_nearest_ties_to_even",
            "saturation": "signed_int8_final_only",
            "metadata_allocation_bytes": 86720,
            "planned_sram_peak_bytes": 464320,
            "remaining_sram_bytes": 59968,
            "divider_schedule_status": "unmeasured_architecture_assumption_not_environment_claim",
        },
        "preserved_operator_contract": preserved,
        "public_frontier": {
            "mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "candidate_ppa_run": False,
        },
        "decisive_checks": {
            "architecture_contract": architecture_check,
            "environment_binding": environment_check,
            "wide_arithmetic_probe": probe_check,
            "exact_scale32_hook": exact_hook_check,
        },
        "bound_artifacts": [
            artifact(AUDIT),
            artifact(TOOLCHAIN),
            artifact(IP_REUSE),
            artifact(PACKET),
            artifact(ARCH_REVIEW),
            artifact(PROBE),
            artifact(PIPELINE),
            artifact(ROOT / "tools/bind_environment_compatibility.py"),
            artifact(ROOT / "tools/run_down_projection_environment_probe.py"),
        ],
        "claim_boundaries": audit.get("claim_boundaries", []),
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    payload["integrity"]["canonical_sha256"] = canonical_sha256(payload)
    dump(OUT, payload)
    validate()


def validate() -> None:
    payload = load(OUT)
    require(payload.get("contract_id") == CONTRACT, "preflight contract mismatch")
    require(payload.get("stage") == "environment", "preflight stage mismatch")
    require(payload.get("stage_closing") is True, "preflight is not stage-closing")
    require(payload.get("status") == "pass_ready_for_independent_environment_review", "preflight status is not pass")
    require(payload.get("implementation_authorized") is False, "preflight authorizes implementation")
    require(payload.get("checklist") == {
        "environment.eda-capabilities": True,
        "environment.tool-ip-selection": True,
    }, "preflight checklist incomplete")
    require(canonical_sha256(payload) == payload.get("integrity", {}).get("canonical_sha256"),
            "preflight canonical hash mismatch")
    require(payload.get("compatibility_findings", {}).get("bootstrap_capability_artifacts_hash_verified") == 27,
            "bootstrap artifact count mismatch")
    require(payload.get("compatibility_findings", {}).get("contract_specific_wide_arithmetic_probe") == "pass",
            "wide arithmetic probe missing")
    require(payload.get("contract_delta", {}).get("signed_numerator_bits_minimum") == 96,
            "signed numerator width mismatch")
    require(payload.get("contract_delta", {}).get("unsigned_denominator_bits_minimum") == 64,
            "denominator width mismatch")
    for index, record in enumerate(payload.get("bound_artifacts", [])):
        verify_record(record, f"bound_artifact[{index}]")
    require(load(PIPELINE).get("current_stage") == "environment", "Manager stage changed")
    print(
        "ACE2_DOWN_PROJECTION_ENVIRONMENT_PREFLIGHT_PASS "
        f"contract={CONTRACT} checklist=2/2 bootstrap_artifacts=27 "
        "delta_probe=pass implementation_authorized=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate()
    else:
        build()


if __name__ == "__main__":
    main()
