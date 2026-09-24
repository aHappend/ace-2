#!/usr/bin/env python3
"""Run and bind the bounded environment audit for the frozen fusion contract.

The probes use only tiny synthetic designs and existing public SKY130 collateral.
They do not compile or execute ACE-2 RTL and do not produce candidate PPA.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
REVIEW_DIR = ROOT / "evidence/review" / f"environment_stage_closing_{CONTRACT}"
PROBE_DIR = REVIEW_DIR / "fresh_environment_probe"
PROBE_RESULTS = PROBE_DIR / "RESULTS.json"
AUDIT = REVIEW_DIR / "audit.json"
IMAGE = "openroad/orfs@sha256:3bc303869d5e4caac8f72c854f2b1614c726b2961bbb372f54bc8fbc0e725e71"
BOOTSTRAP_DIR = ROOT / "research/raw/environment/20260801T190754Z"
DELTA_RESULTS = ROOT / "research/raw/environment/down_projection_residual_fusion_compatibility/RESULTS.json"
ENVIRONMENT_AUDIT = ROOT / "research/ENVIRONMENT_AUDIT.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
ARCH_REVIEW = ROOT / f"evidence/review/architecture_refreeze_{CONTRACT}/decision.json"
PRIOR_ENV_REVIEW = REVIEW_DIR / "decision.json"
CONTEXT_DIR = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b74f837e41bd")
CONTEXT_INDEX = CONTEXT_DIR / "latest.json"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
PDK_HASHES = {
    "/OpenROAD-flow-scripts/flow/platforms/sky130hd/config.mk": "1a36b8fbd58ee4b6b961bc14f3ca44e7b44a4ea47faf6f116f827eb5177509d5",
    "/OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef": "8e99b4e8b016db0713029ebcae6b2cc2aedd9c2c49682e2f23521dd0b1a2085e",
    "/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib": "ec0e1067a35c8bf20b11e58d1e8ac53326067e4dac84a125cc1b917a3518d0d9",
    "/OpenROAD-flow-scripts/flow/platforms/sky130hd/drc/sky130hd.lydrc": "029722ea1fc2cf8c48f09fc670c0b72efb295f799c4eb236b7e9767930ecf772",
    "/OpenROAD-flow-scripts/flow/platforms/sky130hd/lvs/sky130hd.lylvs": "1afade11dd24ea4e64d1ffba54fc17a8f110286712ab942928d2f5e021bb3caf",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def resolve_context_pointer(index: dict[str, Any], key: str) -> Path:
    pointer = index.get(key)
    require(isinstance(pointer, dict), f"context index missing {key} object")
    raw = pointer.get("path")
    require(isinstance(raw, str) and raw, f"context index missing {key}.path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = CONTEXT_DIR / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"context index {key}.path does not resolve") from error
    context_root = CONTEXT_DIR.resolve()
    require(context_root in resolved.parents, f"context index {key}.path escapes context directory")
    require(resolved.is_file(), f"context index {key}.path is not a file")
    return resolved


def context_reference_matches(raw: Any, expected: Path) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = CONTEXT_DIR / candidate
    try:
        return candidate.resolve(strict=True) == expected
    except OSError:
        return False


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            rel = f"operator_context/{path.relative_to(CONTEXT_DIR).as_posix()}"
        except ValueError:
            raise RuntimeError("refusing to bind an external private path") from None
    return {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}


def artifact_path(record: dict[str, Any]) -> Path:
    raw = str(record.get("path", ""))
    require(raw, "bound artifact path is empty")
    if raw.startswith("operator_context/"):
        path = (CONTEXT_DIR / raw.removeprefix("operator_context/")).resolve()
        require(CONTEXT_DIR.resolve() in path.parents, f"bound artifact escapes context directory: {raw}")
    else:
        path = (ROOT / raw).resolve()
        require(ROOT.resolve() in path.parents, f"bound artifact escapes workspace: {raw}")
    return path


def verify_artifact(record: dict[str, Any]) -> None:
    raw = str(record.get("path", ""))
    path = artifact_path(record)
    require(path.is_file(), f"bound artifact missing: {raw}")
    require(path.stat().st_size == record.get("bytes"), f"bound artifact size mismatch: {raw}")
    require(sha256(path) == record.get("sha256"), f"bound artifact hash mismatch: {raw}")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = json.dumps(clone, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def projection_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def run_logged(name: str, command: list[str], timeout: int = 180) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    log = PROBE_DIR / f"{name}.log"
    log.write_text(completed.stdout, encoding="utf-8")
    require(completed.returncode == 0, f"{name} failed; inspect {log.relative_to(ROOT)}")
    return {"name": name, "exit_status": completed.returncode, "log": artifact(log)}


def docker_command(script: str, *, bind_z3: bool = False) -> list[str]:
    command = [
        "docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{ROOT}:/work",
    ]
    if bind_z3:
        z3 = shutil.which("z3")
        require(z3 is not None, "Z3 is required for the selected formal flow")
        command.extend(["-v", f"{z3}:/usr/local/bin/z3:ro"])
    command.extend([IMAGE, "bash", "-lc", script])
    return command


def prepare_probe_inputs() -> list[dict[str, Any]]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    inputs = []
    for name in ("formal_probe.sv", "formal.sby"):
        source = BOOTSTRAP_DIR / name
        destination = PROBE_DIR / name
        shutil.copyfile(source, destination)
        inputs.append(artifact(destination))

    sta = PROBE_DIR / "sky130_sta.tcl"
    sta.write_text(
        "read_liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib\n"
        "read_verilog /work/research/raw/environment/20260801T190754Z/env_probe_mapped_sta.v\n"
        "link_design env_probe\n"
        "read_sdc /work/research/raw/environment/20260801T190754Z/sky130_probe.sdc\n"
        "report_checks -path_delay max -group_count 2\n"
        "report_checks -path_delay min -group_count 2\n"
        "report_wns\nreport_tns\nexit\n",
        encoding="utf-8",
    )
    physical = PROBE_DIR / "sky130_openroad.tcl"
    physical.write_text(
        "read_lef /OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef\n"
        "read_lef /OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef\n"
        "read_liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib\n"
        "read_verilog /work/research/raw/environment/20260801T190754Z/env_probe_mapped_sta.v\n"
        "link_design env_probe\n"
        "initialize_floorplan -die_area {0 0 120 120} -core_area {10 10 110 110} -site unithd\n"
        "make_tracks\nplace_pins -hor_layers met3 -ver_layers met2\n"
        "global_placement -skip_io -density 0.40\ndetailed_placement\ncheck_placement -verbose\n"
        "write_def /work/evidence/review/environment_stage_closing_shared_down_projection_residual_fusion_v1/fresh_environment_probe/env_probe_placed.def\n"
        "exit\n",
        encoding="utf-8",
    )
    inputs.extend([artifact(sta), artifact(physical)])
    return inputs


def run_probes() -> None:
    inputs = prepare_probe_inputs()
    delta = load(DELTA_RESULTS)
    require(delta.get("status") == "pass", "fresh contract-delta probe is not pass")
    checks = []
    checks.append(run_logged(
        "tool_pdk_license",
        docker_command(
            "source /OpenROAD-flow-scripts/env.sh && "
            "yosys -V && sby --version && openroad -version && sta -version && klayout -v && "
            "test -r /OpenROAD-flow-scripts/flow/platforms/sky130hd/config.mk && "
            "test -r /OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef && "
            "test -r /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib && "
            "test -r /OpenROAD-flow-scripts/flow/platforms/sky130hd/drc/sky130hd.lydrc && "
            "test -r /OpenROAD-flow-scripts/flow/platforms/sky130hd/lvs/sky130hd.lylvs && "
            "sha256sum /OpenROAD-flow-scripts/flow/platforms/sky130hd/config.mk "
            "/OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef "
            "/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib "
            "/OpenROAD-flow-scripts/flow/platforms/sky130hd/drc/sky130hd.lydrc "
            "/OpenROAD-flow-scripts/flow/platforms/sky130hd/lvs/sky130hd.lylvs"
        ),
    ))
    checks.append(run_logged(
        "formal",
        docker_command(
            "source /OpenROAD-flow-scripts/env.sh && "
            "cd /work/evidence/review/environment_stage_closing_shared_down_projection_residual_fusion_v1/fresh_environment_probe && "
            "rm -rf formal_work && sby -f -d formal_work formal.sby"
        , bind_z3=True),
    ))
    checks.append(run_logged(
        "sky130_sta",
        docker_command(
            "source /OpenROAD-flow-scripts/env.sh && "
            "sta -no_splash -exit /work/evidence/review/environment_stage_closing_shared_down_projection_residual_fusion_v1/fresh_environment_probe/sky130_sta.tcl"
        ),
    ))
    checks.append(run_logged(
        "sky130_physical",
        docker_command(
            "source /OpenROAD-flow-scripts/env.sh && "
            "openroad -no_init -exit /work/evidence/review/environment_stage_closing_shared_down_projection_residual_fusion_v1/fresh_environment_probe/sky130_openroad.tcl"
        ),
    ))
    placed = PROBE_DIR / "env_probe_placed.def"
    require(placed.is_file() and placed.stat().st_size > 0, "physical probe did not emit DEF")
    tool_log = (PROBE_DIR / "tool_pdk_license.log").read_text(encoding="utf-8")
    for path, digest in PDK_HASHES.items():
        require(f"{digest}  {path}" in tool_log, f"PDK hash not observed: {path}")
    require("DONE (PASS" in (PROBE_DIR / "formal.log").read_text(encoding="utf-8"), "formal pass marker missing")
    require("wns" in (PROBE_DIR / "sky130_sta.log").read_text(encoding="utf-8").lower(), "STA summary missing")
    physical_log = (PROBE_DIR / "sky130_physical.log").read_text(encoding="utf-8").lower()
    require("negotiation phase 1 converged" in physical_log and "|          0 |         0 |         0" in physical_log,
            "placement legalization pass marker missing")
    license_variables = ["LM_LICENSE_FILE", "SNPSLMD_LICENSE_FILE", "CDS_LIC_FILE", "MGLS_LICENSE_FILE"]
    payload = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "contract_id": CONTRACT,
        "status": "pass",
        "claim_boundary": "synthetic_environment_executability_only_not_ace2_rtl_or_candidate_ppa",
        "fresh_capabilities": {
            "simulation": "pass_via_contract_delta_iverilog_compile_and_run",
            "lint": "pass_via_contract_delta_verilator_lint",
            "formal": "pass_symbiyosys_yosys_smtbmc_z3",
            "synthesis": "pass_via_contract_delta_pinned_orfs_yosys",
            "sta": "pass_opensta_sky130_mapped_synthetic_netlist",
            "physical_flow": "pass_openroad_sky130_floorplan_and_placement_synthetic_netlist",
        },
        "contract_delta_probe": artifact(DELTA_RESULTS),
        "probe_inputs": inputs + [artifact(BOOTSTRAP_DIR / "env_probe_mapped_sta.v"), artifact(BOOTSTRAP_DIR / "sky130_probe.sdc")],
        "checks": checks,
        "outputs": [artifact(placed)],
        "runtime": {
            "python": sys.version.splitlines()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "workspace_writable": os.access(ROOT, os.W_OK),
            "docker_image": IMAGE,
        },
        "pdk": {
            "platform": "ORFS sky130hd public platform",
            "expected_file_sha256": PDK_HASHES,
            "availability": "pass_readable_and_hash_matched",
        },
        "licenses": {
            "proprietary_license_required": False,
            "selected_stack": "public_open_source_tools_and_public_sky130_collateral",
            "host_license_variables": {name: ("set" if os.environ.get(name) else "unset") for name in license_variables},
            "assumption": "No credentialed or proprietary license server is required for the selected environment path.",
        },
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    payload["integrity"]["canonical_sha256"] = canonical_sha256(payload)
    dump(PROBE_RESULTS, payload)
    validate_probe()


def validate_probe() -> dict[str, Any]:
    probe = load(PROBE_RESULTS)
    require(probe.get("contract_id") == CONTRACT, "probe contract mismatch")
    require(probe.get("status") == "pass", "probe status is not pass")
    require(canonical_sha256(probe) == probe.get("integrity", {}).get("canonical_sha256"), "probe canonical hash mismatch")
    for key in ("simulation", "lint", "formal", "synthesis", "sta", "physical_flow"):
        require(str(probe.get("fresh_capabilities", {}).get(key, "")).startswith("pass"), f"capability failed: {key}")
    for group in ("probe_inputs", "outputs"):
        for record in probe.get(group, []):
            verify_artifact(record)
    verify_artifact(probe["contract_delta_probe"])
    for check in probe.get("checks", []):
        require(check.get("exit_status") == 0, f"probe check failed: {check.get('name')}")
        verify_artifact(check["log"])
    return probe


def bind_audit() -> None:
    subprocess.run([sys.executable, "tools/bind_environment_compatibility.py"], cwd=ROOT, check=True)
    probe = validate_probe()
    environment = load(ENVIRONMENT_AUDIT)
    packet = load(PACKET)
    arch_review = load(ARCH_REVIEW)
    public = load(PUBLIC)
    pipeline = load(PIPELINE)
    target = load(ROOT / "design/TARGET.json")
    chip = load(ROOT / "design/CHIP_SCOPE.json")
    memory = load(ROOT / "design/MEMORY_MODEL.json")
    index = load(CONTEXT_INDEX)
    handoff_path = resolve_context_pointer(index, "handoff")
    mission_path = resolve_context_pointer(index, "mission")
    handoff = load(handoff_path)
    mission = load(mission_path)
    public_hashes = {
        "dashboard_fields.current_architecture_performance_model": projection_sha256(public["dashboard_fields"]["current_architecture_performance_model"]),
        "implementation_frontier.current_architecture_performance_model": projection_sha256(public["implementation_frontier"]["current_architecture_performance_model"]),
    }
    recorded_public_hashes = arch_review["review_binding"]["public_projection_hashes"]
    context_pointer_check = {
        "index_schema_version_2": index.get("schema_version") == 2,
        "index_kind_handoff_ref": index.get("kind") == "handoff_ref",
        "handoff_path_present": isinstance(index.get("handoff"), dict) and isinstance(index["handoff"].get("path"), str),
        "mission_path_present": isinstance(index.get("mission"), dict) and isinstance(index["mission"].get("path"), str),
        "handoff_kind_round_reviewed_handoff": handoff.get("kind") == "round_reviewed_handoff",
        "mission_kind_mission_context": mission.get("kind") == "mission_context",
        "handoff_mission_context_matches_pointed_mission": context_reference_matches(handoff.get("mission_context"), mission_path),
        "handoff_mission_id_matches_pointed_mission": isinstance(mission.get("mission_id"), str)
        and bool(mission["mission_id"])
        and handoff.get("mission_id") == mission["mission_id"],
        "handoff_snapshot_is_round_file": handoff_path.name.startswith("round-") and handoff_path.suffix == ".json",
        "mission_snapshot_is_mission_file": mission_path.name == "mission.json",
    }
    preserved = packet["preserved_operator_contract"]
    target_contract = target["current_architecture_contract"]
    chip_frontier = chip["implementation_frontier"]
    memory_override = memory["authority_override"]
    contracts_preserved = all([
        preserved["ordered_supported_layer_operator_prefix"] == PREFIX,
        target_contract["ordered_supported_layer_operator_prefix"] == PREFIX,
        chip_frontier["ordered_supported_layer_operator_prefix"] == PREFIX,
        preserved["area_cap_non_sram_mm2"] == target_contract["area_cap_non_sram_mm2"] == 2.0,
        preserved["frequency_floor_mhz"] == target_contract["frequency_floor_mhz"] == 100.0,
        preserved["abstract_streaming_memory_boundary_bits"] == chip["interfaces"]["external_memory_stream"]["data_width_bits"] == 128,
        memory_override["area_cap_non_sram_mm2"] == 2.0,
        memory_override["frequency_floor_mhz"] == 100.0,
        chip_frontier["latest_ppa_frontier"]["cells"] == preserved["historical_ppa_frontier"]["cells"],
        chip_frontier["latest_ppa_frontier"]["non_sram_area_mm2"] == preserved["historical_ppa_frontier"]["non_sram_area_mm2"],
        chip_frontier["latest_ppa_frontier"]["setup_slack_ns_at_100mhz"] == preserved["historical_ppa_frontier"]["setup_slack_ns_at_100mhz"],
    ])
    capability_records = environment.get("evidence", {}).get("artifacts", [])
    for record in capability_records:
        verify_artifact(record)
    environment_items_true = environment.get("stage_gate_items") == {
        "environment.eda-capabilities": True,
        "environment.tool-ip-selection": True,
    }
    architecture_acknowledged = all([
        arch_review.get("architecture_accepted") is True,
        arch_review.get("reviewer_status") == "done",
        arch_review.get("implementation_authorized") is False,
        sha256(PACKET) == arch_review.get("packet", {}).get("sha256"),
        public_hashes == recorded_public_hashes,
        len(set(public_hashes.values())) == 1,
    ])
    implementation_locked = all([
        packet.get("implementation_authorized") is False,
        arch_review.get("implementation_authorized") is False,
        environment.get("contract_binding", {}).get("implementation_authorized") is False,
        environment.get("readiness_summary", {}).get("implementation_authorized") is False,
        public.get("selected_replacement_contract", {}).get("implementation_authorized") is False,
        target_contract.get("implementation_authorized") is False,
        memory_override.get("implementation_authorized") is False,
    ])
    context_complete = all(context_pointer_check.values())
    checklist = {
        "environment.eda-capabilities": environment_items_true and probe.get("status") == "pass" and len(capability_records) == 27,
        "environment.tool-ip-selection": environment_items_true and environment.get("tool_ip_selection", {}).get("proprietary_or_credentialed_dependency") is False,
    }
    ready_except_context = all(checklist.values()) and architecture_acknowledged and implementation_locked and contracts_preserved
    prior_decision = artifact(PRIOR_ENV_REVIEW) if PRIOR_ENV_REVIEW.is_file() else None
    audit = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "project": "ACE-2",
        "stage": "environment",
        "contract_id": CONTRACT,
        "status": "blocked_context_index_missing_required_pointers" if not context_complete else "ready_for_exactly_one_fresh_independent_l2_environment_review",
        "claim_boundary": "environment compatibility only; no RTL implementation, candidate verification, PPA, prototype, benchmark, signoff, GDS, tapeout, or silicon claim",
        "architecture_acceptance_acknowledged": architecture_acknowledged,
        "implementation_authorized": False,
        "environment_checklist": checklist,
        "preserved_operator_contract": preserved,
        "contracts_preserved": contracts_preserved,
        "public_projection_binding": {
            "algorithm": "sha256-json-sort-keys-compact-utf8-v1",
            "current_hashes": public_hashes,
            "accepted_architecture_hashes": recorded_public_hashes,
            "equal_and_current": public_hashes == recorded_public_hashes and len(set(public_hashes.values())) == 1,
        },
        "fresh_probe": artifact(PROBE_RESULTS),
        "first_party_capability_evidence": {
            "audit": artifact(ENVIRONMENT_AUDIT),
            "bootstrap_artifact_count": len(capability_records),
            "all_bootstrap_artifacts_hash_verified": len(capability_records) == 27,
            "bootstrap_probes_rerun": False,
            "fresh_capabilities": probe["fresh_capabilities"],
        },
        "bindings": [
            artifact(PACKET), artifact(ARCH_REVIEW), artifact(ROOT / f"evidence/{CONTRACT}/latest/CONTRACT_BINDINGS.json"),
            artifact(ROOT / f"evidence/{CONTRACT}/latest/FIRST_PARTY_PROVENANCE.json"),
            artifact(ROOT / "design/TARGET.json"), artifact(ROOT / "design/CHIP_SCOPE.json"),
            artifact(ROOT / "design/MEMORY_MODEL.json"), artifact(PUBLIC), artifact(PIPELINE),
            artifact(mission_path), artifact(handoff_path), artifact(Path(__file__)),
        ],
        "runtime_and_dependencies": probe["runtime"],
        "pdk": probe["pdk"],
        "licenses": probe["licenses"],
        "context_binding": {
            "binding_mode": "immutable_handoff_and_mission_snapshots_discovered_via_mutable_index",
            "discovery_index": {
                "path": "operator_context/latest.json",
                "schema_version": index.get("schema_version"),
                "kind": index.get("kind"),
                "observed_handoff_path": artifact(handoff_path)["path"],
                "observed_mission_path": artifact(mission_path)["path"],
                "mutable_pointer_not_integrity_bound": True,
            },
            "mission": artifact(mission_path),
            "handoff": artifact(handoff_path),
            "pointer_check": context_pointer_check,
            "fail_closed": not context_complete,
        },
        "preexisting_environment_decision": {
            "artifact": prior_decision,
            "classification": "historical_pre_mission_decision_not_a_fresh_decision_for_this_audit",
        },
        "decisive_readiness_check": {
            "environment_criteria_pass": ready_except_context,
            "context_binding_complete": context_complete,
            "ready_for_independent_l2": ready_except_context and context_complete,
            "independent_l2_decision": "pending_not_performed_by_auditor",
            "manager_environment_to_rtl_permission": False,
            "reason": "The EDA, PDK, license, architecture, public-projection, implementation-lock, and preserved-contract checks pass, but stage closure fails closed because the schema-v2 context index or its pointed mission/handoff identity checks failed." if not context_complete else "All audit criteria pass; exactly one fresh independent L2 environment decision is required before Manager-controlled environment-to-rtl entry.",
        },
        "forbidden_work_respected": True,
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    audit["integrity"]["canonical_sha256"] = canonical_sha256(audit)
    dump(AUDIT, audit)
    validate_audit()


def validate_audit() -> None:
    validate_probe()
    audit = load(AUDIT)
    require(audit.get("contract_id") == CONTRACT, "audit contract mismatch")
    require(audit.get("implementation_authorized") is False, "audit authorizes implementation")
    require(all(audit.get("environment_checklist", {}).values()), "environment checklist item failed")
    require(audit.get("architecture_acceptance_acknowledged") is True, "architecture acceptance not acknowledged")
    require(audit.get("contracts_preserved") is True, "operator contract changed")
    require(canonical_sha256(audit) == audit.get("integrity", {}).get("canonical_sha256"), "audit canonical hash mismatch")
    for record in audit.get("bindings", []):
        verify_artifact(record)
    for key in ("audit",):
        verify_artifact(audit["first_party_capability_evidence"][key])
    verify_artifact(audit["fresh_probe"])
    context = audit.get("context_binding", {})
    require(
        context.get("binding_mode") == "immutable_handoff_and_mission_snapshots_discovered_via_mutable_index",
        "audit context binding is not snapshot-stable",
    )
    discovery = context.get("discovery_index", {})
    require(discovery.get("path") == "operator_context/latest.json", "unexpected context discovery index")
    require(discovery.get("schema_version") == 2, "context discovery index schema mismatch")
    require(discovery.get("kind") == "handoff_ref", "context discovery index kind mismatch")
    require(discovery.get("mutable_pointer_not_integrity_bound") is True, "mutable context index is integrity-bound")
    mission_record = context.get("mission", {})
    handoff_record = context.get("handoff", {})
    for key, record in (("mission", mission_record), ("handoff", handoff_record)):
        verify_artifact(record)
        require(record in audit.get("bindings", []), f"immutable context {key} missing from bindings")
    require(discovery.get("observed_mission_path") == mission_record.get("path"), "discovery mission path mismatch")
    require(discovery.get("observed_handoff_path") == handoff_record.get("path"), "discovery handoff path mismatch")
    mission_path = artifact_path(mission_record)
    handoff_path = artifact_path(handoff_record)
    mission = load(mission_path)
    handoff = load(handoff_path)
    snapshot_check = {
        "handoff_kind_round_reviewed_handoff": handoff.get("kind") == "round_reviewed_handoff",
        "mission_kind_mission_context": mission.get("kind") == "mission_context",
        "handoff_mission_context_matches_bound_mission": context_reference_matches(handoff.get("mission_context"), mission_path),
        "handoff_mission_id_matches_bound_mission": isinstance(mission.get("mission_id"), str)
        and bool(mission["mission_id"])
        and handoff.get("mission_id") == mission["mission_id"],
        "handoff_snapshot_is_round_file": handoff_path.name.startswith("round-") and handoff_path.suffix == ".json",
        "mission_snapshot_is_mission_file": mission_path.name == "mission.json",
    }
    require(all(context.get("pointer_check", {}).values()), "generation-time schema-v2 context pointer check failed")
    require(all(snapshot_check.values()), "bound immutable context snapshot check failed")
    require(context["fail_closed"] is False, "schema-v2 context binding remains fail-closed")
    require(audit.get("status") == "ready_for_exactly_one_fresh_independent_l2_environment_review", "audit is not ready for independent L2 review")
    require(audit["decisive_readiness_check"]["manager_environment_to_rtl_permission"] is False, "audit grants stage permission")
    require(audit["decisive_readiness_check"]["independent_l2_decision"] == "pending_not_performed_by_auditor", "auditor performed reviewer decision")
    expected_ready = not audit["context_binding"]["fail_closed"]
    require(audit["decisive_readiness_check"]["ready_for_independent_l2"] is expected_ready, "context fail-closed status inconsistent")
    print(
        "ACE2_DOWN_PROJECTION_ENVIRONMENT_AUDIT_BINDING_PASS "
        f"contract={CONTRACT} checklist=2/2 architecture_accepted=true "
        f"implementation_authorized=false context_complete={str(expected_ready).lower()} "
        "l2_decision=pending"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-probes", action="store_true")
    group.add_argument("--bind", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.run_probes:
        run_probes()
    elif args.bind:
        bind_audit()
    else:
        validate_audit()


if __name__ == "__main__":
    main()
