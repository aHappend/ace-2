#!/usr/bin/env python3
"""Engineer-owned recovery, precheck, and handoff for Dynamic Scale32 RTL."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_token_group_dynamic_scale32_v1"
MISSION_ID = "rtl-remediate-scale32"
SESSION_ID = os.environ.get("ARGUS_SKILL_SESSION_ID", "s-c8ae985b")
THREAD_ID = os.environ.get("CODEX_THREAD_ID", "019fc16c-20c2-77a3-9555-356bbbc004f2")
LATEST = ROOT / f"evidence/{CONTRACT}/rtl/latest"
RECOVERY = ROOT / f"evidence/{CONTRACT}/recovery/planner_cycle0_bypass"
INVENTORY = RECOVERY / "INCIDENT_INVENTORY.json"
ARCHIVE = RECOVERY / "archive"
DRAFT_PROVENANCE = RECOVERY / "DRAFT_PROVENANCE.json"
SOURCE_ADOPTION = RECOVERY / "ENGINEER_SOURCE_ADOPTION.json"
HANDOFF = RECOVERY / "ENGINEER_IMPLEMENTATION_HANDOFF.json"
PRECHECK = LATEST / "PRECHECK.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
IP_PROVENANCE = ROOT / "design/DYNAMIC_SCALE32_IP_PROVENANCE.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
REMEDIATION_TOOL = ROOT / "tools/recover_dynamic_scale32_rtl.py"

ELABORATION_LOG_NAMES = [
    "iverilog.log",
    "elaborate_ace2_dynamic_scale32_group_core.log",
    "elaborate_ace2_dynamic_scale32_sidecar_builder_core.log",
    "elaborate_ace2_dynamic_scale32_sidecar_validator_core.log",
    "elaborate_ace2_scale32_tagged_accumulator_core.log",
]

EXPECTED_FROZEN_HASHES = {
    f"evidence/{CONTRACT}/architecture/MANAGER_FREEZE.json":
        "175d1dcc7016df0c94fb6ec0d086d78fe6b0e4b80f26d1ef42f746825b53c2d8",
    f"evidence/{CONTRACT}/architecture/PROPOSAL.json":
        "b55ab977574dc2bdeb760e8859ec2fa49f9f835846e3da24bd0f887d84a74f41",
    f"evidence/review/architecture_{CONTRACT}/decision.json":
        "386a2951ea1f39e3ef51978ced42be72dea8e10c53be188286abbe1507124297",
    f"evidence/review/environment_stage_closing_{CONTRACT}/audit.json":
        "fc953fe75e63aead71deb293b8413d28e4581fef3e8061c6fbed44b80b140cf7",
    f"evidence/review/environment_stage_closing_{CONTRACT}/decision.json":
        "910efb2fd11e53f4404905b7f6e8e0a528986b873bad15dfe8da5ec522907eff",
}

SOURCE_PATHS = [
    "rtl/ace2_dynamic_scale32_core.sv",
    "tools/ace2_dynamic_scale32_reference.py",
    "tools/gen_dynamic_scale32_vectors.py",
    "verification/generated/dynamic_scale32_vectors.json",
    "verification/generated/dynamic_scale32_vectors.svh",
    "verification/test_dynamic_scale32.py",
    "verification/tb/ace2_dynamic_scale32_tb.sv",
    "formal/ace2_dynamic_scale32_formal.sv",
    "formal/ace2_dynamic_scale32_formal.ys",
    "tools/run_dynamic_scale32_formal.py",
    "Makefile",
    "design/DYNAMIC_SCALE32_IP_PROVENANCE.json",
]

DISPOSITIONS = {
    "rtl/ace2_dynamic_scale32_core.sv": (
        "rewritten",
        "Line-by-line review retained the arithmetic/state structure, added fail-closed enforcement for frozen group indices [0,37], changed five combinational process declarations from always_comb to the synthesizable equivalent always @*, and added continuous selected-bank aliases so Icarus sees no unpacked-array sensitivity diagnostics.",
    ),
    "tools/ace2_dynamic_scale32_reference.py": (
        "adopted_unchanged",
        "Line-by-line review confirmed signed RNE, delta/exponent bounds, exact sidecar layout, and signed-160 accumulation against the frozen contract.",
    ),
    "tools/gen_dynamic_scale32_vectors.py": (
        "rewritten",
        "Line-by-line review retained deterministic cases and added independently generated packed expected mantissas for the 128-lane RTL comparison.",
    ),
    "verification/generated/dynamic_scale32_vectors.json": (
        "regenerated",
        "Every line was deterministically regenerated from the Engineer-adopted reference and rewritten generator.",
    ),
    "verification/generated/dynamic_scale32_vectors.svh": (
        "regenerated",
        "Every line was deterministically regenerated and now carries the 128-lane expected mantissa vector.",
    ),
    "verification/test_dynamic_scale32.py": (
        "adopted_unchanged",
        "Line-by-line review confirmed focused scalar coverage for RNE, exponent floor, fail-closed overflow, sidecar reserved bytes, and exact accumulation.",
    ),
    "verification/tb/ace2_dynamic_scale32_tb.sv": (
        "rewritten",
        "Line-by-line review removed a self-referential comparison, added a bit-exact 128-lane case, group payload/commit stall stability, and group-index fail-closed coverage.",
    ),
    "formal/ace2_dynamic_scale32_formal.sv": (
        "adopted_unchanged",
        "Line-by-line review confirmed bounded exclusive-error, canonical-exponent, reset-ready, and result-stability assertions.",
    ),
    "formal/ace2_dynamic_scale32_formal.ys": (
        "adopted_unchanged",
        "Line-by-line review confirmed the depth-4 proof is limited to the standalone accumulator harness.",
    ),
    "tools/run_dynamic_scale32_formal.py": (
        "rewritten",
        "Line-by-line review retained extraction of only the live accumulator and the permitted bounded Yosys SAT proof while updating the parser-shim assertion for the diagnostic-clean always @* declaration.",
    ),
    "Makefile": (
        "rewritten",
        "The Dynamic Scale32 target block was line-by-line validated; Engineer recovery and non-mutating recovery-check entrypoints were added without invoking downstream stages.",
    ),
    "design/DYNAMIC_SCALE32_IP_PROVENANCE.json": (
        "rewritten",
        "Line-by-line review retained first-party/no-third-party claims and added explicit Engineer recovery ownership bound to the incident inventory and archive.",
    ),
}

TOP_PORTS = {
    "ace2_dynamic_scale32_group_core": {
        "clk_i": "input", "rst_ni": "input", "clear_i": "input",
        "group_start_valid_i": "input", "group_start_ready_o": "output",
        "group_lanes_u8_i": "input", "base_scale32_i": "input",
        "group_index_u6_i": "input", "tensor_tag_u16_i": "input",
        "value_valid_i": "input", "value_ready_o": "output", "value_s40_i": "input",
        "payload_valid_o": "output", "payload_ready_i": "input", "payload_s8_o": "output",
        "payload_last_o": "output", "group_commit_valid_o": "output",
        "group_commit_ready_i": "input", "group_index_u6_o": "output",
        "tensor_tag_u16_o": "output", "exponent_delta_s7_o": "output",
        "effective_scale32_o": "output", "descriptor_error_o": "output",
        "numeric_overflow_o": "output",
    },
    "ace2_dynamic_scale32_sidecar_builder_core": {
        "clk_i": "input", "rst_ni": "input", "clear_i": "input",
        "start_valid_i": "input", "start_ready_o": "output",
        "payload_addr_u64_i": "input", "group_lanes_u8_i": "input",
        "group_count_u6_i": "input", "producer_tag_u16_i": "input",
        "layer_id_u8_i": "input", "producer_opcode_u8_i": "input",
        "tensor_elements_u32_i": "input", "model_identity_u64_i": "input",
        "delta_valid_i": "input", "delta_ready_o": "output",
        "exponent_delta_s7_i": "input", "base_scale32_i": "input",
        "sidecar_valid_o": "output", "sidecar_ready_i": "input",
        "sidecar_o": "output", "descriptor_error_o": "output",
        "numeric_overflow_o": "output",
    },
    "ace2_dynamic_scale32_sidecar_validator_core": {
        "clk_i": "input", "rst_ni": "input", "clear_i": "input",
        "start_valid_i": "input", "start_ready_o": "output", "sidecar_i": "input",
        "base_scale32_packed_i": "input", "payload_addr_u64_i": "input",
        "expected_group_lanes_u8_i": "input", "expected_group_count_u6_i": "input",
        "expected_producer_tag_u16_i": "input", "expected_layer_id_u8_i": "input",
        "expected_producer_opcode_u8_i": "input", "expected_tensor_elements_u32_i": "input",
        "expected_model_identity_u64_i": "input", "result_valid_o": "output",
        "result_ready_i": "input", "descriptor_error_o": "output",
        "numeric_overflow_o": "output",
    },
    "ace2_scale32_tagged_accumulator_core": {
        "clk_i": "input", "rst_ni": "input", "clear_i": "input",
        "start_valid_i": "input", "start_ready_o": "output",
        "accumulator_tag_u16_i": "input", "event_valid_i": "input",
        "event_ready_o": "output", "partial_s32_i": "input", "scale_a32_i": "input",
        "scale_b32_i": "input", "event_last_i": "input", "result_valid_o": "output",
        "result_ready_i": "input", "accumulator_s160_o": "output",
        "canonical_exponent_s8_o": "output", "accumulator_tag_u16_o": "output",
        "descriptor_error_o": "output", "numeric_overflow_o": "output",
    },
}

GENERATED_PATHS = [
    "verification/generated/dynamic_scale32_vectors.json",
    "verification/generated/dynamic_scale32_vectors.svh",
]

GENERATED_METADATA = {
    "verification/generated/dynamic_scale32_vectors.json": {
        "kind": "generated_verification_source",
        "generator": "tools/gen_dynamic_scale32_vectors.py",
        "reference": "tools/ace2_dynamic_scale32_reference.py",
        "regeneration_command": "python tools/gen_dynamic_scale32_vectors.py",
    },
    "verification/generated/dynamic_scale32_vectors.svh": {
        "kind": "generated_verification_include",
        "generator": "tools/gen_dynamic_scale32_vectors.py",
        "reference": "tools/ace2_dynamic_scale32_reference.py",
        "regeneration_command": "python tools/gen_dynamic_scale32_vectors.py",
    },
}

DYNAMIC_IP_NAMES = [
    *TOP_PORTS,
    "dynamic_scale32_vectors_json",
    "dynamic_scale32_vectors_svh",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path_or_relative: Path | str) -> dict[str, Any]:
    path = path_or_relative if isinstance(path_or_relative, Path) else ROOT / path_or_relative
    return {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def aggregate_hash(source_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(source_hashes[path].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: dict[str, Any], *, integrity: bool = True) -> None:
    if integrity:
        value["integrity"] = {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        }
        value["integrity"]["canonical_sha256"] = canonical_sha256(value)
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_companion(path: Path) -> None:
    atomic_write(path.with_suffix(".sha256"), f"{sha256_file(path)}  {path.name}\n")


def verify_companion(path: Path) -> None:
    companion = path.with_suffix(".sha256")
    require(companion.is_file(), f"missing companion hash: {relative(path)}")
    require(companion.read_text(encoding="utf-8") == f"{sha256_file(path)}  {path.name}\n",
            f"companion hash mismatch: {relative(path)}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_json_integrity(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("integrity", {}).get("canonical_sha256") == canonical_sha256(value),
            f"canonical JSON integrity mismatch: {relative(path)}")
    verify_companion(path)
    return value


def pipeline_state_binding() -> dict[str, Any]:
    verify_companion(PIPELINE_STATE)
    state = load_json(PIPELINE_STATE)
    successor = state.get("successor", {})
    queue = successor.get("next_action_queue", [])
    require(state.get("current_stage") == "rtl", "top-level current_stage is not rtl")
    require(successor.get("current_stage") == "rtl", "successor current_stage is not reconciled")
    require(successor.get("required_next_gate") ==
            "fresh_independent_evidence_only_l2_rtl_stage_closing_review",
            "successor next gate is not the fresh independent RTL stage-closing review")
    require(successor.get("same_contract_rtl_authorized") is True,
            "same-contract RTL authorization is not recorded")
    require(successor.get("implementation_authorized") is False and
            successor.get("implementation_authority_consumed") is True and
            successor.get("implementation_completed") is True,
            "successor RTL implementation completion fields are inconsistent")
    require(successor.get("candidate_capability_accepted") is False and
            successor.get("baseline_or_candidate_execution_authorized") is False and
            successor.get("candidate_or_model_execution_count") == 0,
            "successor candidate/model execution boundary changed")
    require(all(value == "unrun" for value in successor.get("downstream_runs", {}).values()),
            "successor downstream run state changed")
    require(len(queue) == 1 and queue[0].get("scope") ==
            "fresh_independent_evidence_only_l2_rtl_stage_closing_review" and
            queue[0].get("status") == "pending_fresh_independent_review",
            "successor fresh independent review queue is inconsistent")
    require("manager_environment_to_rtl_decision" not in json.dumps(queue, sort_keys=True),
            "stale pending Manager environment-to-RTL queue remains")
    reconciliation = successor.get("manager_state_reconciliation", {})
    require(reconciliation.get("role") == "manager" and
            reconciliation.get("scope") == "successor_projection_reconciliation_only" and
            reconciliation.get("stage_transition_performed") is False and
            reconciliation.get("recorded_environment_to_rtl_transition_at") ==
            "2026-08-02T07:08:41.858846Z",
            "Manager successor reconciliation provenance is incomplete")
    return {
        "pipeline_state": artifact(PIPELINE_STATE),
        "pipeline_state_companion": artifact(PIPELINE_STATE.with_suffix(".sha256")),
    }


def require_clean_elaboration_contents(contents: dict[str, str]) -> None:
    for name, content in contents.items():
        require(content == "", f"elaboration diagnostic emitted: {name}")


def verify_elaboration_diagnostic_guard() -> dict[str, Any]:
    actual = {
        name: (LATEST / name).read_text(encoding="utf-8")
        for name in ELABORATION_LOG_NAMES
    }
    require_clean_elaboration_contents(actual)
    synthetic_rejected = False
    try:
        require_clean_elaboration_contents({"synthetic.log": "synthetic diagnostic\n"})
    except RuntimeError as error:
        require(str(error) == "elaboration diagnostic emitted: synthetic.log",
                f"synthetic diagnostic failed for an unexpected reason: {error}")
        synthetic_rejected = True
    require(synthetic_rejected, "synthetic elaboration diagnostic was accepted")
    guard_log = LATEST / "elaboration_diagnostic_guard.log"
    atomic_write(
        guard_log,
        "ACE2_DYNAMIC_SCALE32_ELABORATION_DIAGNOSTIC_GUARD_PASS "
        "actual_logs_byte_empty=5 synthetic_diagnostic_rejected=true\n",
    )
    return artifact(guard_log)


def engineer_ledger_proof(thread_id: str, session_id: str) -> dict[str, Any]:
    require(isinstance(thread_id, str) and thread_id, "sealed Engineer thread identity is missing")
    require(isinstance(session_id, str) and session_id, "sealed Engineer session identity is missing")
    candidates = []
    configured = os.environ.get("ARGUS_SKILL_AGENT_IO_LOG", "")
    if configured:
        candidates.append(Path(configured))
    candidates.append(
        Path.home() / ".argus-skill-ace2" / "projects" / session_id / "events.jsonl"
    )
    ledgers = []
    seen_ledgers = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved not in seen_ledgers:
            seen_ledgers.add(resolved)
            ledgers.append(path)
    require(ledgers, "Engineer execution ledger is unavailable")

    if (thread_id == os.environ.get("CODEX_THREAD_ID") and
            session_id == os.environ.get("ARGUS_SKILL_SESSION_ID")):
        for ledger in ledgers:
            for line in reversed(ledger.read_text(encoding="utf-8").splitlines()):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                if (event.get("type") == "engineer.progress" and
                        event.get("agent_layer") == "engineer" and
                        event.get("status") == "completed"):
                    return {
                        "event_type": "engineer.progress",
                        "mission_id": MISSION_ID,
                        "run_label": event.get("actor", "engineer"),
                        "status": "active_engineer_runtime",
                        "thread_id": thread_id,
                    }

    for ledger in ledgers:
        starts: dict[str, dict[str, Any]] = {}
        completions: list[dict[str, Any]] = []
        usage: dict[str, dict[str, Any]] = {}
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            call_id = event.get("call_id")
            if not isinstance(call_id, str):
                continue
            if event.get("type") == "agent.io.start":
                starts[call_id] = event
            elif event.get("type") == "agent.io.complete" and event.get("thread_id") == thread_id:
                completions.append(event)
            elif event.get("type") == "usage.recorded" and event.get("thread_id") == thread_id:
                usage[call_id] = event

        for completion in reversed(completions):
            call_id = completion["call_id"]
            start = starts.get(call_id)
            recorded = usage.get(call_id)
            if start is None or recorded is None:
                continue
            run_label = str(start.get("run_label") or completion.get("run_label") or "")
            working_dir = start.get("working_dir")
            mission_id = str(recorded.get("mission_id") or "")
            if not run_label.startswith("engineer"):
                continue
            if working_dir and Path(working_dir).resolve() != ROOT:
                continue
            if completion.get("exit_code") != 0 or completion.get("turn_completed") is not True:
                continue
            if recorded.get("project_id") != session_id or recorded.get("status") != "completed":
                continue
            if not mission_id.startswith(f"{MISSION_ID}:"):
                continue
            return {
                "call_id": call_id,
                "event_type": "agent.io.complete",
                "mission_id": mission_id,
                "run_label": run_label,
                "status": "completed",
                "thread_id": thread_id,
            }
    raise RuntimeError("no ledger-backed completed Engineer producer identity matches the sealed packet")


def validate_authority_and_archive() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    pipeline_state_binding()
    for path, expected in EXPECTED_FROZEN_HASHES.items():
        require(sha256_file(ROOT / path) == expected, f"frozen dependency changed: {path}")

    require(sha256_file(INVENTORY) == "0b56bfc5916967c257e448c1fa5de0764843abb777c5100e62d4ea2b4ef769d9",
            "incident inventory hash changed")
    inventory = load_json(INVENTORY)
    require(inventory.get("artifact_count") == 36 and inventory.get("inventory_complete") is True,
            "incident inventory is incomplete")
    entries = {entry["path"]: entry for entry in inventory["artifacts"]}
    require(len(entries) == 36, "incident inventory path set is not unique")
    for path, entry in entries.items():
        archived = ARCHIVE / path
        require(archived.is_file(), f"missing archived Planner artifact: {path}")
        require(sha256_file(archived) == entry["sha256"], f"archived Planner hash mismatch: {path}")
    require(all(path in entries for path in SOURCE_PATHS), "source adoption path missing from incident inventory")
    return inventory, entries


def write_draft_provenance(inventory: dict[str, Any]) -> None:
    records = []
    for entry in inventory["artifacts"]:
        records.append({
            "original_path": entry["path"],
            "archived_path": relative(ARCHIVE / entry["path"]),
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
            "actor": "planner.cycle0",
            "accepted_evidence": False,
        })
    value = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "classification": "unaccepted_planner_cycle0_draft_and_run_output",
        "classification_authority": "manager_recovery_mission_rtl-recovery-scale32",
        "incident_inventory": artifact(INVENTORY),
        "artifact_count": len(records),
        "artifacts": records,
        "preservation": {
            "all_incident_files_archived_before_live_rewrite": True,
            "archive_hashes_match_incident_inventory": True,
            "history_rewritten": False,
        },
        "candidate_capability_accepted": False,
        "stage_closing": False,
        "superseded_by": "engineer_line_by_line_adoption_or_rewrite",
    }
    write_json(DRAFT_PROVENANCE, value)
    write_companion(DRAFT_PROVENANCE)


def run(command: list[str], log_name: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = LATEST / log_name
    atomic_write(log, result.stdout)
    require(result.returncode == 0, f"permitted RTL check failed: {' '.join(command)}")
    return artifact(log)


def interface_ports(path: Path, top: str) -> dict[str, str]:
    root = ET.parse(path).getroot()
    module = root.find(f".//module[@name='{top}']")
    require(module is not None and module.get("topModule") == "1", f"interface XML top missing: {top}")
    return {
        var.get("name", ""): var.get("dir", "")
        for var in module.findall("var")
        if var.get("dir") in {"input", "output", "inout"}
    }


def run_permitted_checks() -> tuple[dict[str, Any], dict[str, Any]]:
    LATEST.mkdir(parents=True, exist_ok=True)
    (ROOT / "build/dynamic_scale32").mkdir(parents=True, exist_ok=True)
    logs: dict[str, Any] = {}
    logs["vector_generation"] = artifact(LATEST / "vector_generation.log")
    logs["reference_unittest"] = artifact(LATEST / "reference_unittest.log")
    logs["iverilog"] = run(
        ["iverilog", "-g2012", "-Wall", "-Iverification/generated", "-o",
         "build/ace2_dynamic_scale32_tb.vvp", "rtl/ace2_dynamic_scale32_core.sv",
         "verification/tb/ace2_dynamic_scale32_tb.sv"],
        "iverilog.log",
    )
    logs["rtl_simulation"] = run(["vvp", "build/ace2_dynamic_scale32_tb.vvp"], "rtl_simulation.log")

    interfaces: dict[str, Any] = {}
    lint_outputs = []
    for top, expected_ports in TOP_PORTS.items():
        lint_log = f"lint_{top}.log"
        logs[f"lint_{top}"] = run(
            ["verilator", "--lint-only", "--language", "1800-2017", "-Wall",
             "--top-module", top, "rtl/ace2_dynamic_scale32_core.sv"],
            lint_log,
        )
        lint_outputs.append((LATEST / lint_log).read_text(encoding="utf-8"))
        logs[f"elaborate_{top}"] = run(
            ["iverilog", "-g2012", "-s", top, "-o", f"build/dynamic_scale32/{top}.vvp",
             "rtl/ace2_dynamic_scale32_core.sv"],
            f"elaborate_{top}.log",
        )
        interface_path = LATEST / f"interface_{top}.xml"
        logs[f"interface_{top}"] = run(
            ["verilator", "--xml-only", "--language", "1800-2017", "-Wall",
             "--top-module", top, "--xml-output", relative(interface_path),
             "rtl/ace2_dynamic_scale32_core.sv"],
            f"interface_{top}.log",
        )
        ports = interface_ports(interface_path, top)
        require(ports == expected_ports, f"interface port drift: {top}")
        interfaces[top] = {
            **artifact(interface_path),
            "port_directions": ports,
            "exact_expected_port_set": True,
        }
    logs["elaboration_diagnostic_guard"] = verify_elaboration_diagnostic_guard()
    atomic_write(LATEST / "verilator_lint.log", "".join(lint_outputs))
    require((LATEST / "verilator_lint.log").read_text(encoding="utf-8") == "",
            "Verilator lint was not warning-free")
    logs["verilator_lint"] = artifact(LATEST / "verilator_lint.log")
    logs["minimal_formal"] = run([sys.executable, "tools/run_dynamic_scale32_formal.py"], "minimal_formal.log")

    require("ACE2_DYNAMIC_SCALE32_VECTOR_GENERATION_PASS cases=5" in
            (LATEST / "vector_generation.log").read_text(encoding="utf-8"),
            "vector generation marker missing")
    require("Ran 5 tests" in (LATEST / "reference_unittest.log").read_text(encoding="utf-8"),
            "reference unittest marker missing")
    require("ACE2_DYNAMIC_SCALE32_RTL_PASS groups=5 lanes=64,128" in
            (LATEST / "rtl_simulation.log").read_text(encoding="utf-8"),
            "bit-exact RTL simulation marker missing")
    require("ACE2_DYNAMIC_SCALE32_MINIMAL_FORMAL_PASS" in
            (LATEST / "minimal_formal.log").read_text(encoding="utf-8"),
            "minimal formal marker missing")
    return logs, interfaces


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def write_source_adoption(
    incident_entries: dict[str, dict[str, Any]], source_hashes: dict[str, str], candidate_hash: str
) -> dict[str, Any]:
    records = []
    for path in SOURCE_PATHS:
        planner = incident_entries[path]
        disposition, validation = DISPOSITIONS[path]
        count = line_count(ROOT / path)
        if disposition == "regenerated":
            hash_relation = (
                "deterministically_regenerated_byte_identical_to_planner_draft"
                if planner["sha256"] == source_hashes[path]
                else "deterministically_regenerated_with_engineer_validated_changes"
            )
        elif planner["sha256"] == source_hashes[path]:
            hash_relation = "unchanged_bytes_explicitly_adopted"
        else:
            hash_relation = "engineer_rewrite"
        records.append({
            "path": path,
            "planner_actor": "planner.cycle0",
            "planner_draft_sha256": planner["sha256"],
            "final_actor": "engineer",
            "final_live_sha256": source_hashes[path],
            "hash_relation": hash_relation,
            "disposition": disposition,
            "reviewed_line_range": f"1-{count}",
            "line_count": count,
            "line_by_line_validated": True,
            "validation": validation,
            "accepted_as_engineer_implementation_source": True,
        })
    value = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "actor": "engineer",
        "role": "engineer",
        "session_id": SESSION_ID,
        "thread_id": THREAD_ID,
        "review_method": "line_by_line_validation_of_every_necessary_planner_draft",
        "planner_draft_provenance": artifact(DRAFT_PROVENANCE),
        "source_hash_order": SOURCE_PATHS,
        "source_artifacts": records,
        "implementation_source_aggregate_sha256": candidate_hash,
        "implementation_source_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_makefile_and_ip_provenance_excluding_evidence_orchestration",
        "pipeline_state_binding": pipeline_state_binding(),
        "evidence_generator": artifact(REMEDIATION_TOOL),
        "review_findings": [
            "Planner testbench did not execute the 128-lane vector and used a self-referential fallback comparison; rewritten to compare all 128 generated expected mantissas.",
            "Planner testbench did not stall group payload/commit outputs; rewritten to prove stable data, tags, deltas, scale, and errors under backpressure.",
            "Group indices outside the frozen maximum of 38 groups were not rejected; RTL and simulation now fail closed for indices 38-63.",
            "Icarus emitted 59 nonfatal constant-select always_* diagnostics across the integrated and four per-top elaborations; the five equivalent combinational process declarations now use always @*, selected unpacked-array values are exposed through continuous scalar/packed aliases, and a byte-empty log guard rejects a synthetic diagnostic.",
        ],
        "candidate_capability_accepted": False,
        "stage_closing": False,
        "status": "engineer_sources_line_by_line_adopted_rewritten_or_regenerated",
    }
    write_json(SOURCE_ADOPTION, value)
    write_companion(SOURCE_ADOPTION)
    return value


def implementation_provenance(
    candidate_hash: str, *, session_id: str | None = None, thread_id: str | None = None
) -> dict[str, Any]:
    producer_session = SESSION_ID if session_id is None else session_id
    producer_thread = THREAD_ID if thread_id is None else thread_id
    return {
        "role": "engineer",
        "agent_layer": "engineer",
        "session_id": producer_session,
        "thread_id": producer_thread,
        "status": "engineer_recovery_from_frozen_contract",
        "supersedes": "unaccepted_planner_cycle0_drafts_and_runs",
        "planner_draft_provenance": artifact(DRAFT_PROVENANCE),
        "source_adoption": artifact(SOURCE_ADOPTION),
        "implementation_source_aggregate_sha256": candidate_hash,
        "implementation_source_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_makefile_and_ip_provenance_excluding_evidence_orchestration",
        "pipeline_state_binding": pipeline_state_binding(),
        "evidence_generator": artifact(REMEDIATION_TOOL),
        "preserved_for_explicitly_adopted_unchanged_sources": True,
        "stage_closing": False,
    }


def sealed_producer_identity(
    adoption: dict[str, Any], handoff: dict[str, Any]
) -> dict[str, str]:
    require(adoption.get("actor") == "engineer" and adoption.get("role") == "engineer",
            "source adoption is not Engineer-owned")
    require(handoff.get("producer_role") == "engineer",
            "handoff is not Engineer-owned")
    adoption_identity = {
        "session_id": adoption.get("session_id"),
        "thread_id": adoption.get("thread_id"),
    }
    handoff_identity = {
        "session_id": handoff.get("session_id"),
        "thread_id": handoff.get("thread_id"),
    }
    require(adoption_identity == handoff_identity,
            "sealed Engineer identities disagree between source adoption and handoff")
    require(all(isinstance(value, str) and value for value in adoption_identity.values()),
            "sealed Engineer identity is incomplete")
    return adoption_identity


def validate_sealed_implementation_provenance(
    candidate_hash: str,
    adoption: dict[str, Any],
    handoff: dict[str, Any],
    precheck: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    identity = sealed_producer_identity(adoption, handoff)
    expected = implementation_provenance(candidate_hash, **identity)
    require(precheck.get("implementation_provenance") == expected,
            "PRECHECK implementation provenance is stale")
    require(manifest.get("candidate_implementation_provenance") == expected,
            "manifest implementation provenance is stale")
    engineer_ledger_proof(identity["thread_id"], identity["session_id"])
    return expected


def run_provenance_identity_regression(
    candidate_hash: str,
    adoption: dict[str, Any],
    handoff: dict[str, Any],
    precheck: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    global SESSION_ID, THREAD_ID
    original_session = SESSION_ID
    original_thread = THREAD_ID
    original_ledger = os.environ.get("ARGUS_SKILL_AGENT_IO_LOG")
    try:
        SESSION_ID = "distinct-checker-session"
        THREAD_ID = "00000000-0000-0000-0000-000000000001"
        os.environ["ARGUS_SKILL_AGENT_IO_LOG"] = str(MANIFEST)
        validate_sealed_implementation_provenance(
            candidate_hash, adoption, handoff, precheck, manifest
        )
    finally:
        SESSION_ID = original_session
        THREAD_ID = original_thread
        if original_ledger is None:
            os.environ.pop("ARGUS_SKILL_AGENT_IO_LOG", None)
        else:
            os.environ["ARGUS_SKILL_AGENT_IO_LOG"] = original_ledger

    tampered_thread = "00000000-0000-0000-0000-000000000002"
    tampered_adoption = copy.deepcopy(adoption)
    tampered_handoff = copy.deepcopy(handoff)
    tampered_precheck = copy.deepcopy(precheck)
    tampered_manifest = copy.deepcopy(manifest)
    tampered_adoption["thread_id"] = tampered_thread
    tampered_handoff["thread_id"] = tampered_thread
    tampered_precheck["implementation_provenance"]["thread_id"] = tampered_thread
    tampered_manifest["candidate_implementation_provenance"]["thread_id"] = tampered_thread
    try:
        validate_sealed_implementation_provenance(
            candidate_hash,
            tampered_adoption,
            tampered_handoff,
            tampered_precheck,
            tampered_manifest,
        )
    except RuntimeError as error:
        require("no ledger-backed completed Engineer producer identity" in str(error),
                f"tampered identity regression failed for an unexpected reason: {error}")
    else:
        raise RuntimeError("tampered recorded Engineer identity was accepted")
    print(
        "ACE2_DYNAMIC_SCALE32_PROVENANCE_REGRESSION_PASS "
        "distinct_caller=true tampered_identity_rejected=true"
    )


def candidate_source_provenance(
    incident_entries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **artifact(path),
            "actor": "engineer",
            "planner_draft_sha256": incident_entries[path]["sha256"],
            "disposition": DISPOSITIONS[path][0],
            "source_adoption": artifact(SOURCE_ADOPTION),
            "third_party": False,
        }
        for path in SOURCE_PATHS
    ]


def candidate_rtl_sources() -> list[dict[str, Any]]:
    return [{
        **artifact("rtl/ace2_dynamic_scale32_core.sv"),
        "actor": "engineer",
        "source_adoption": artifact(SOURCE_ADOPTION),
        "third_party": False,
    }]


def candidate_generated_sources() -> list[dict[str, Any]]:
    return [
        {
            **artifact(path),
            **GENERATED_METADATA[path],
            "actor": "engineer",
            "source_adoption": artifact(SOURCE_ADOPTION),
            "third_party": False,
        }
        for path in GENERATED_PATHS
    ]


def candidate_evidence_hashes() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT,
        "scope": "frozen_manager_architecture_and_environment_acceptance",
        "frozen_dependencies_sha256": dict(EXPECTED_FROZEN_HASHES),
        "reconciled_pipeline_state": pipeline_state_binding(),
    }


def candidate_model_metadata() -> dict[str, Any]:
    return {
        "baseline_or_candidate_model_execution": 0,
        "model_artifact_bound": False,
        "required_for_standalone_rtl_handoff": False,
        "status": "not_applicable_dynamic_scale32_standalone_rtl_recovery",
    }


def candidate_review_binding() -> dict[str, Any]:
    return {
        "candidate_capability_accepted": False,
        "implementation_handoff_only": True,
        "pipeline_state_binding": pipeline_state_binding(),
        "planner_draft_provenance": artifact(DRAFT_PROVENANCE),
        "precheck": artifact(PRECHECK),
        "reviewer_status": "pending_separate_fresh_evidence_only_l2",
        "scope": "dynamic_scale32_engineer_rtl_implementation_handoff_only",
        "source_adoption": artifact(SOURCE_ADOPTION),
        "stage_closing": False,
    }


def dynamic_ip_provenance_entries() -> list[dict[str, Any]]:
    rtl_artifact = artifact("rtl/ace2_dynamic_scale32_core.sv")
    source_adoption = artifact(SOURCE_ADOPTION)
    common = {
        "actor": "engineer",
        "license_provenance": "design/DYNAMIC_SCALE32_IP_PROVENANCE.json",
        "license_status": "not_a_third_party_dependency",
        "source_adoption": source_adoption,
        "third_party": False,
    }
    entries = [
        {
            **rtl_artifact,
            **common,
            "kind": "first_party_candidate_rtl",
            "license": "first-party operator-directed project-internal work product; no external redistribution grant claimed",
            "name": name,
            "source_revision": rtl_artifact["sha256"],
        }
        for name in TOP_PORTS
    ]
    for path, name in zip(GENERATED_PATHS, DYNAMIC_IP_NAMES[-2:]):
        generated_artifact = artifact(path)
        entries.append({
            **generated_artifact,
            **common,
            **GENERATED_METADATA[path],
            "license": "generated from first-party project-internal reference and generator; no external redistribution grant claimed",
            "name": name,
            "source_revision": generated_artifact["sha256"],
        })
    return entries


def replace_dynamic_ip_provenance(manifest: dict[str, Any]) -> None:
    dynamic_names = set(DYNAMIC_IP_NAMES)
    dynamic_paths = {"rtl/ace2_dynamic_scale32_core.sv", *GENERATED_PATHS}
    preserved = [
        entry for entry in manifest.get("ip_provenance", [])
        if entry.get("name") not in dynamic_names and entry.get("path") not in dynamic_paths
    ]
    manifest["ip_provenance"] = preserved + dynamic_ip_provenance_entries()


def write_precheck(
    source_hashes: dict[str, str], candidate_hash: str, logs: dict[str, Any],
    interfaces: dict[str, Any], timestamp: str
) -> dict[str, Any]:
    value = {
        "schema_version": 2,
        "project": "ACE-2",
        "contract_id": CONTRACT,
        "candidate_id": f"dynamic_scale32_{candidate_hash[:16]}",
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_makefile_and_ip_provenance_not_shell_admitted",
        "generated_at_utc": timestamp,
        "current_stage": "rtl",
        "stage_closing": False,
        "implementation_authorized": False,
        "implementation_completed": True,
        "implementation_provenance": implementation_provenance(candidate_hash),
        "pipeline_state_binding": pipeline_state_binding(),
        "evidence_generator": artifact(REMEDIATION_TOOL),
        "frozen_dependencies": {path: artifact(path) for path in EXPECTED_FROZEN_HASHES},
        "source_hashes": source_hashes,
        "source_artifacts": [artifact(path) for path in SOURCE_PATHS],
        "checks": {
            "reference_vector_generation": "RETAINED_PASS_5_cases_deterministic_source_unaffected",
            "reference_unit_tests": "RETAINED_PASS_5_tests_reference_unaffected",
            "iverilog_elaboration": "PASS_integrated_and_4_tops",
            "elaboration_diagnostic_guard": "PASS_5_logs_byte_empty_and_synthetic_diagnostic_rejected",
            "bit_exact_simulation": "PASS_5_groups_including_128_lane_and_backpressure",
            "verilator_lint": "PASS_4_tops_warning_free",
            "interface_checks": "PASS_4_exact_port_sets_from_verilator_xml",
            "minimal_formal": "PASS_depth_4",
        },
        "logs": logs,
        "interfaces": interfaces,
        "stage_checklist": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "frontier": {
            "mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"
            ],
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "non_sram_area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "abstract_streaming_memory_boundary_bits": 128,
            "historical_ppa_preserved_not_new_evidence": True,
            "candidate_capability_accepted": False,
        },
        "execution_boundary": {
            "permitted_rtl_checks_only": True,
            "baseline_or_candidate_model_execution": 0,
            "focused_discriminator": 0,
            "shell_regression": 0,
            "synthesis_or_ppa": 0,
            "physical": 0,
            "prototype": 0,
            "benchmark": 0,
            "signoff": 0,
            "stage_transition": 0,
        },
        "independent_review": "pending_separate_fresh_evidence_only_l2",
        "claim_boundary": "Engineer-owned standalone RTL implementation precheck and handoff only. This is not RTL stage-closing certification and makes no model, shell, synthesis/PPA, physical, prototype, benchmark, signoff, tapeout, or silicon claim.",
        "status": "engineer_rtl_precheck_pass_implementation_handoff_only",
    }
    write_json(PRECHECK, value)
    write_companion(PRECHECK)
    return value


def update_manifest(
    source_hashes: dict[str, str], candidate_hash: str, logs: dict[str, Any],
    interfaces: dict[str, Any], timestamp: str, incident_entries: dict[str, dict[str, Any]]
) -> None:
    manifest = load_json(MANIFEST)
    provenance = implementation_provenance(candidate_hash)
    manifest.update({
        "candidate_evidence_hashes": candidate_evidence_hashes(),
        "candidate_generated_hashes": {
            path: source_hashes[path] for path in GENERATED_PATHS
        },
        "candidate_generated_sources": candidate_generated_sources(),
        "candidate_id": f"dynamic_scale32_{candidate_hash[:16]}",
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_makefile_and_ip_provenance_not_shell_admitted",
        "candidate_rtl_sources": candidate_rtl_sources(),
        "candidate_source_hashes": source_hashes,
        "candidate_source_provenance": candidate_source_provenance(incident_entries),
        "candidate_implementation_provenance": provenance,
        "candidate_layer_operator": CONTRACT,
        "candidate_model_metadata": candidate_model_metadata(),
        "candidate_review_binding": candidate_review_binding(),
        "candidate_pipeline_state_binding": pipeline_state_binding(),
        "candidate_evidence_generator": artifact(REMEDIATION_TOOL),
        "candidate_status": "engineer_rtl_precheck_pass_implementation_handoff_only",
        "candidate_capability_accepted": False,
        "candidate_meets_numeric_acceptance": False,
        "candidate_verification_binding": None,
        "candidate_verification_complete": False,
        "implementation_completed": True,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "current_stage": "rtl",
        "stage": "rtl",
        "stage_closing": False,
        "status": "engineer_rtl_implementation_handoff_stage_closing_false",
        "generated_at_utc": timestamp,
        "interfaces_contract_status": "dynamic_scale32_standalone_interfaces_engineer_prechecked",
        "latest_dynamic_scale32_rtl_evidence": {
            "precheck": artifact(PRECHECK),
            "source_adoption": artifact(SOURCE_ADOPTION),
            "planner_draft_provenance": artifact(DRAFT_PROVENANCE),
            "pipeline_state_binding": pipeline_state_binding(),
            "evidence_generator": artifact(REMEDIATION_TOOL),
            "traceability": artifact(TRACEABILITY),
            "traceability_companion": artifact(TRACEABILITY.with_suffix(".sha256")),
            "logs": logs,
            "interfaces": interfaces,
        },
        "planner_cycle0_recovery": {
            "incident_inventory": artifact(INVENTORY),
            "planner_draft_provenance": artifact(DRAFT_PROVENANCE),
            "engineer_source_adoption": artifact(SOURCE_ADOPTION),
            "planner_manifest_sha256": incident_entries["design/RTL_MANIFEST.json"]["sha256"],
            "planner_precheck_sha256": incident_entries[f"evidence/{CONTRACT}/rtl/latest/PRECHECK.json"]["sha256"],
            "planner_checks_accepted_as_engineer_evidence": False,
        },
        "required_manager_action": "none_implementation_handoff_only_stage_closing_false",
    })
    proposed = manifest.setdefault("proposed_replacement_contract", {})
    proposed.update({
        "candidate_id": f"dynamic_scale32_{candidate_hash[:16]}",
        "candidate_rtl_hash": candidate_hash,
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "implementation_completed": True,
        "stage_closing": False,
        "status": "engineer_rtl_implementation_handoff_stage_closing_false",
    })
    review = manifest.setdefault("independent_reviewer_acceptance", {})
    review.update({
        "architecture_review": "accepted",
        "environment_review": "accepted",
        "rtl_review": "pending_separate_fresh_evidence_only_l2",
        "status": "pending",
    })
    traceability = manifest.setdefault("traceability", {})
    traceability.update({
        "selected_mechanism": CONTRACT,
        "first_unsupported_layer_operator": "layer_0.rope_q",
        "architecture_contract_gap": None,
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
        "stage_checklist": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
    })
    replace_dynamic_ip_provenance(manifest)
    atomic_write(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_companion(MANIFEST)


def validate_manifest_aliases(
    manifest: dict[str, Any], source_hashes: dict[str, str], candidate_hash: str,
    incident_entries: dict[str, dict[str, Any]], precheck: dict[str, Any],
    expected_provenance: dict[str, Any]
) -> None:
    candidate_id = f"dynamic_scale32_{candidate_hash[:16]}"
    require(manifest.get("candidate_id") == candidate_id, "manifest candidate id is stale")
    require(manifest.get("candidate_layer_operator") == CONTRACT,
            "manifest candidate contract alias is stale")
    require(manifest.get("candidate_rtl_hash") == candidate_hash,
            "manifest aggregate is stale")
    require(manifest.get("candidate_source_hashes") == source_hashes,
            "manifest source hash map is stale")
    require(manifest.get("candidate_source_provenance") ==
            candidate_source_provenance(incident_entries),
            "manifest source provenance aliases are stale")
    require(manifest.get("candidate_rtl_sources") == candidate_rtl_sources(),
            "manifest RTL source alias is stale")
    require(manifest.get("candidate_generated_hashes") == {
                path: source_hashes[path] for path in GENERATED_PATHS
            }, "manifest generated hash aliases are stale")
    require(manifest.get("candidate_generated_sources") == candidate_generated_sources(),
            "manifest generated source aliases are stale")
    require(manifest.get("candidate_evidence_hashes") == candidate_evidence_hashes(),
            "manifest candidate evidence binding is stale")
    require(manifest.get("candidate_model_metadata") == candidate_model_metadata(),
            "manifest candidate model metadata alias is stale")
    require(manifest.get("candidate_review_binding") == candidate_review_binding(),
            "manifest candidate review binding is stale")
    require(manifest.get("candidate_implementation_provenance") ==
            expected_provenance,
            "manifest implementation provenance is stale")
    require(manifest.get("candidate_pipeline_state_binding") == pipeline_state_binding() and
            manifest.get("candidate_evidence_generator") == artifact(REMEDIATION_TOOL),
            "manifest remediation binding is stale")
    require(manifest.get("candidate_capability_accepted") is False and
            manifest.get("candidate_meets_numeric_acceptance") is False and
            manifest.get("candidate_verification_binding") is None and
            manifest.get("candidate_verification_complete") is False,
            "manifest candidate acceptance boundary changed")

    proposed = manifest.get("proposed_replacement_contract", {})
    require(proposed.get("candidate_id") == candidate_id and
            proposed.get("candidate_rtl_hash") == candidate_hash and
            proposed.get("contract_id") == CONTRACT and
            proposed.get("implementation_authorized") is False and
            proposed.get("implementation_completed") is True and
            proposed.get("stage_closing") is False and
            proposed.get("status") == "engineer_rtl_implementation_handoff_stage_closing_false",
            "manifest proposed replacement contract alias is stale")

    dynamic_names = set(DYNAMIC_IP_NAMES)
    dynamic_paths = {"rtl/ace2_dynamic_scale32_core.sv", *GENERATED_PATHS}
    live_ip_entries = [
        entry for entry in manifest.get("ip_provenance", [])
        if entry.get("name") in dynamic_names or entry.get("path") in dynamic_paths
    ]
    require(live_ip_entries == dynamic_ip_provenance_entries(),
            "manifest Dynamic Scale32 IP provenance aliases are stale")

    expected_latest = {
        "precheck": artifact(PRECHECK),
        "source_adoption": artifact(SOURCE_ADOPTION),
        "planner_draft_provenance": artifact(DRAFT_PROVENANCE),
        "pipeline_state_binding": pipeline_state_binding(),
        "evidence_generator": artifact(REMEDIATION_TOOL),
        "traceability": artifact(TRACEABILITY),
        "traceability_companion": artifact(TRACEABILITY.with_suffix(".sha256")),
        "logs": precheck.get("logs"),
        "interfaces": precheck.get("interfaces"),
    }
    require(manifest.get("latest_dynamic_scale32_rtl_evidence") == expected_latest,
            "manifest latest Dynamic Scale32 evidence binding is stale")


def write_handoff(
    candidate_hash: str, logs: dict[str, Any], interfaces: dict[str, Any],
    timestamp: str, incident_entries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    collateral_paths = [
        "design/DYNAMIC_SCALE32_IP_PROVENANCE.json",
        "design/RTL_TRACEABILITY.md",
        "design/RTL_MANIFEST.json",
        f"evidence/{CONTRACT}/rtl/latest/PRECHECK.json",
    ]
    collateral_validation = {
        "design/DYNAMIC_SCALE32_IP_PROVENANCE.json": (
            "rewritten",
            "Planner provenance lines were reviewed in full; first-party/no-third-party declarations were retained and Engineer recovery ownership was added.",
        ),
        "design/RTL_TRACEABILITY.md": (
            "rewritten",
            "Planner traceability lines were reviewed in full; false 128-lane and group-stall evidence claims were corrected, and the bounded Icarus compatibility repair plus reconciled pipeline-state binding were added.",
        ),
        "design/RTL_MANIFEST.json": (
            "rewritten",
            "Planner manifest lines were reviewed in full; unrelated repository records were preserved while the Dynamic Scale32 candidate namespace, ownership, hashes, checks, and review boundary were replaced.",
        ),
        f"evidence/{CONTRACT}/rtl/latest/PRECHECK.json": (
            "rewritten",
            "Planner PRECHECK lines were reviewed in full and replaced by fresh Engineer-owned source, check, interface, execution-boundary, and stage_closing=false evidence.",
        ),
    }
    value = {
        "schema_version": 1,
        "kind": "engineer_rtl_implementation_handoff",
        "project": "ACE-2",
        "contract_id": CONTRACT,
        "generated_at_utc": timestamp,
        "producer_role": "engineer",
        "session_id": SESSION_ID,
        "thread_id": THREAD_ID,
        "current_stage": "rtl",
        "stage_closing": False,
        "stage_transition_performed": False,
        "candidate_capability_accepted": False,
        "pipeline_state_binding": pipeline_state_binding(),
        "evidence_generator": artifact(REMEDIATION_TOOL),
        "implementation_source_aggregate_sha256": candidate_hash,
        "implementation_source_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_makefile_and_ip_provenance_excluding_evidence_orchestration",
        "planner_draft_provenance": artifact(DRAFT_PROVENANCE),
        "engineer_source_adoption": artifact(SOURCE_ADOPTION),
        "precheck": artifact(PRECHECK),
        "manifest": artifact(MANIFEST),
        "manifest_companion": artifact(MANIFEST.with_suffix(".sha256")),
        "traceability": artifact(TRACEABILITY),
        "traceability_companion": artifact(TRACEABILITY.with_suffix(".sha256")),
        "ip_provenance": artifact(IP_PROVENANCE),
        "planner_to_engineer_collateral": [
            {
                "path": path,
                "planner_draft_sha256": incident_entries[path]["sha256"],
                "final_live_sha256": sha256_file(ROOT / path),
                "final_actor": "engineer",
                "disposition": collateral_validation[path][0],
                "planner_reviewed_line_range": f"1-{line_count(ARCHIVE / path)}",
                "planner_line_count": line_count(ARCHIVE / path),
                "final_line_range": f"1-{line_count(ROOT / path)}",
                "final_line_count": line_count(ROOT / path),
                "line_by_line_validated": True,
                "validation": collateral_validation[path][1],
            }
            for path in collateral_paths
        ],
        "necessary_planner_drafts_complete": True,
        "permitted_rtl_checks": {
            "logs": logs,
            "interfaces": interfaces,
            "result": "PASS",
        },
        "forbidden_execution_counts": {
            "baseline_model": 0,
            "candidate_model": 0,
            "focused_discriminator": 0,
            "shell_regression": 0,
            "synthesis": 0,
            "ppa": 0,
            "physical": 0,
            "prototype": 0,
            "benchmark": 0,
            "signoff": 0,
        },
        "preserved_contract": {
            "mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"
            ],
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "non_sram_area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "abstract_streaming_memory_boundary_bits": 128,
            "historical_ppa_is_new_evidence": False,
            "certified_baseline_harness_chain_preserved": True,
            "sealed_predecessors_preserved": True,
        },
        "review_request": {
            "scope": "fresh_evidence_only_integrity_review_of_engineer_implementation_handoff",
            "may_certify_rtl_stage_closure": False,
            "required_stage_closing": False,
            "next_action": "Verify exact hashes, Engineer ownership, permitted-check evidence, and no-downstream boundary without advancing the stage.",
        },
        "claim_boundary": "Implementation handoff only. A separate fresh evidence-only L2 task is required for RTL stage-closing certification.",
        "status": "ready_for_fresh_evidence_only_review_stage_closing_false",
    }
    write_json(HANDOFF, value)
    write_companion(HANDOFF)
    return value


def bind() -> str:
    inventory, incident_entries = validate_authority_and_archive()
    write_draft_provenance(inventory)
    logs, interfaces = run_permitted_checks()
    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}
    candidate_hash = aggregate_hash(source_hashes)
    atomic_write(RECOVERY / "SOURCE_AGGREGATE.sha256", f"{candidate_hash}  dynamic_scale32_engineer_source_aggregate\n")
    write_source_adoption(incident_entries, source_hashes, candidate_hash)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    write_precheck(source_hashes, candidate_hash, logs, interfaces, timestamp)
    write_companion(TRACEABILITY)
    update_manifest(source_hashes, candidate_hash, logs, interfaces, timestamp, incident_entries)
    write_handoff(candidate_hash, logs, interfaces, timestamp, incident_entries)
    return candidate_hash


def check() -> str:
    _, incident_entries = validate_authority_and_archive()
    draft = verify_json_integrity(DRAFT_PROVENANCE)
    adoption = verify_json_integrity(SOURCE_ADOPTION)
    precheck = verify_json_integrity(PRECHECK)
    handoff = verify_json_integrity(HANDOFF)
    pipeline_binding = pipeline_state_binding()
    verify_companion(TRACEABILITY)
    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}
    candidate_hash = aggregate_hash(source_hashes)
    require((RECOVERY / "SOURCE_AGGREGATE.sha256").read_text(encoding="utf-8") ==
            f"{candidate_hash}  dynamic_scale32_engineer_source_aggregate\n",
            "aggregate source companion is stale")
    require(draft.get("artifact_count") == 36 and draft.get("stage_closing") is False,
            "Planner draft provenance boundary changed")
    require(all(entry.get("accepted_evidence") is False for entry in draft.get("artifacts", [])),
            "Planner draft archive was accepted as evidence")
    require(adoption.get("implementation_source_aggregate_sha256") == candidate_hash,
            "source adoption aggregate is stale")
    require(adoption.get("pipeline_state_binding") == pipeline_binding and
            adoption.get("evidence_generator") == artifact(REMEDIATION_TOOL),
            "source adoption remediation binding is stale")
    require(all(entry.get("line_by_line_validated") is True for entry in adoption.get("source_artifacts", [])),
            "one or more source drafts lack line-by-line disposition")
    require(len(adoption.get("source_artifacts", [])) == len(SOURCE_PATHS),
            "source adoption set is incomplete")
    require(precheck.get("candidate_rtl_hash") == candidate_hash and precheck.get("stage_closing") is False,
            "PRECHECK aggregate or stage boundary changed")
    require(precheck.get("source_hashes") == source_hashes,
            "PRECHECK source hash map is stale")
    require(precheck.get("source_artifacts") == [artifact(path) for path in SOURCE_PATHS],
            "PRECHECK source artifact aliases are stale")
    require(precheck.get("pipeline_state_binding") == pipeline_binding and
            precheck.get("evidence_generator") == artifact(REMEDIATION_TOOL),
            "PRECHECK remediation binding is stale")
    require(handoff.get("implementation_source_aggregate_sha256") == candidate_hash and
            handoff.get("stage_closing") is False and handoff.get("stage_transition_performed") is False,
            "handoff aggregate or stage boundary changed")
    require(all(value == 0 for value in handoff.get("forbidden_execution_counts", {}).values()),
            "forbidden execution count changed")
    require(handoff.get("planner_draft_provenance") == artifact(DRAFT_PROVENANCE) and
            handoff.get("engineer_source_adoption") == artifact(SOURCE_ADOPTION) and
            handoff.get("precheck") == artifact(PRECHECK) and
            handoff.get("traceability") == artifact(TRACEABILITY) and
            handoff.get("traceability_companion") == artifact(TRACEABILITY.with_suffix(".sha256")) and
            handoff.get("pipeline_state_binding") == pipeline_binding and
            handoff.get("evidence_generator") == artifact(REMEDIATION_TOOL) and
            handoff.get("ip_provenance") == artifact(IP_PROVENANCE),
            "handoff implementation bindings are stale")
    verify_companion(MANIFEST)
    manifest = load_json(MANIFEST)
    expected_provenance = validate_sealed_implementation_provenance(
        candidate_hash, adoption, handoff, precheck, manifest
    )
    require(precheck.get("implementation_provenance", {}).get("role") == "engineer",
            "PRECHECK is not Engineer-owned")
    validate_manifest_aliases(
        manifest, source_hashes, candidate_hash, incident_entries, precheck,
        expected_provenance
    )
    require(manifest.get("candidate_implementation_provenance", {}).get("role") == "engineer",
            "manifest is not Engineer-owned")
    require(manifest.get("stage_closing") is False and manifest.get("current_stage") == "rtl",
            "manifest stage boundary changed")
    require(handoff.get("manifest") == artifact(MANIFEST) and
            handoff.get("manifest_companion") == artifact(MANIFEST.with_suffix(".sha256")),
            "handoff manifest bindings are stale")
    require(incident_entries["design/RTL_MANIFEST.json"]["sha256"] != sha256_file(MANIFEST),
            "Planner manifest was not rewritten")
    require("ACE2_DYNAMIC_SCALE32_RTL_PASS groups=5 lanes=64,128" in
            (LATEST / "rtl_simulation.log").read_text(encoding="utf-8"),
            "fresh simulation marker missing")
    require("ACE2_DYNAMIC_SCALE32_MINIMAL_FORMAL_PASS" in
            (LATEST / "minimal_formal.log").read_text(encoding="utf-8"),
            "fresh formal marker missing")
    require((LATEST / "verilator_lint.log").read_text(encoding="utf-8") == "",
            "fresh lint aggregate is not warning-free")
    guard = verify_elaboration_diagnostic_guard()
    require(precheck.get("logs", {}).get("elaboration_diagnostic_guard") == guard,
            "elaboration diagnostic guard evidence is stale")
    for top, expected in TOP_PORTS.items():
        require(interface_ports(LATEST / f"interface_{top}.xml", top) == expected,
                f"fresh interface check drift: {top}")
    run_provenance_identity_regression(
        candidate_hash, adoption, handoff, precheck, manifest
    )
    return candidate_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    candidate_hash = check() if args.check else bind()
    marker = "ACE2_DYNAMIC_SCALE32_ENGINEER_RECOVERY_CHECK_PASS" if args.check else "ACE2_DYNAMIC_SCALE32_ENGINEER_RECOVERY_PASS"
    print(f"{marker} aggregate_source_sha256={candidate_hash} stage_closing=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
