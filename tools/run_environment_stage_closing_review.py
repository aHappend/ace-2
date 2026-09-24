#!/usr/bin/env python3
"""Run and publish one fresh independent environment-stage review."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import argus_skill
from argus_skill.adapters.agent_cli_backend import AgentCliBackend
from argus_skill.core.knobs import (
    resolve_role_backend,
    resolve_role_model,
    resolve_role_reasoning_effort,
    resolve_runner_bin_setting,
)
from argus_skill.core.paths import global_root
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
MISSION_ID = "environment-stage-close-shared-down-projection-residual-fusion-v1"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
AUDIT = ROOT / "research/ENVIRONMENT_AUDIT.json"
AUDIT_MD = ROOT / "research/ENVIRONMENT_AUDIT.md"
PREFLIGHT = ROOT / "evidence/environment_preflight/latest/PACKET.json"
PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
ARCH_REVIEW = ROOT / f"evidence/review/architecture_refreeze_{CONTRACT}/decision.json"
TOOLCHAIN = ROOT / "research/TOOLCHAIN_CANDIDATES.md"
IP_REUSE = ROOT / "research/IP_REUSE_PLAN.md"
PROBE = ROOT / "research/raw/environment/down_projection_residual_fusion_compatibility/RESULTS.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
RAW_DECISION = ROOT / f"evidence/review/environment_stage_closing_{CONTRACT}/decision.json"
REVIEW_BINDING_AUDIT = ROOT / f"evidence/review/environment_stage_closing_{CONTRACT}/audit.json"
VERDICT = ROOT / "research/ENVIRONMENT_REVIEWER_VERDICT.json"
CERT = ROOT / "research/ENVIRONMENT_L2_CERTIFICATION.md"
ARCHIVE = ROOT / f"research/archive/environment/historical_before_{CONTRACT}"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
TOP_LEVEL_SCOPE = "manager_controlled_environment_to_rtl_entry_only"
ENVIRONMENT_CHECKLIST = {
    "environment.eda-capabilities": True,
    "environment.tool-ip-selection": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def resolve_record_path(record: dict[str, Any]) -> Path:
    raw = Path(str(record.get("path", "")))
    return raw if raw.is_absolute() else ROOT / raw


def verify_any_record(record: dict[str, Any], label: str) -> None:
    path = resolve_record_path(record)
    require(path.is_file(), f"{label} missing")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label} byte mismatch")
    require(sha256(path) == record.get("sha256"), f"{label} hash mismatch")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    return hashlib.sha256((json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def compact_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_public_projection_hashes(public: dict[str, Any]) -> dict[str, str]:
    return {
        f"{container}.current_architecture_performance_model": compact_json_sha256(
            public[container]["current_architecture_performance_model"]
        )
        for container in ("dashboard_fields", "implementation_frontier")
    }


def apply_top_level_verdict_schema(
    verdict: dict[str, Any],
    raw: dict[str, Any],
    review_audit: dict[str, Any],
) -> dict[str, Any]:
    architecture = load(ARCH_REVIEW)
    require(architecture.get("contract_id") == CONTRACT, "accepted architecture contract mismatch")
    require(architecture.get("reviewer_status") == "done", "accepted architecture reviewer is not done")
    require(architecture.get("status") == "accepted", "final architecture decision is not accepted")
    require(architecture.get("implementation_authorized") is False,
            "accepted architecture decision authorizes implementation")

    projection_binding = review_audit.get("public_projection_binding", {})
    expected_projection_hashes = projection_binding.get("accepted_architecture_hashes", {})
    current_projection_hashes = current_public_projection_hashes(load(PUBLIC))
    require(projection_binding.get("equal_and_current") is True,
            "reviewed public architecture projections were not equal and current")
    require(current_projection_hashes == expected_projection_hashes,
            "current public architecture projection hashes changed after review")

    preserved = review_audit.get("preserved_operator_contract", {})
    require(preserved.get("ordered_supported_layer_operator_prefix") == PREFIX,
            "reviewed operator prefix changed")
    require(preserved.get("first_unsupported_layer_operator") == "layer_0.rope_q",
            "reviewed first unsupported operator changed")
    require(preserved.get("area_cap_non_sram_mm2") == 2.0, "reviewed area cap changed")
    require(preserved.get("frequency_floor_mhz") == 100.0, "reviewed frequency floor changed")
    require(preserved.get("abstract_streaming_memory_boundary_bits") == 128,
            "reviewed streaming-memory boundary changed")
    require(preserved.get("historical_ppa_frontier") == {
        "candidate_ppa_run": False,
        "cells": 62199,
        "non_sram_area_mm2": 0.6108746272,
        "setup_slack_ns_at_100mhz": 0.1502,
    }, "reviewed historical PPA frontier changed")

    normalized = copy.deepcopy(verdict)
    normalized.update({
        "schema_version": max(int(verdict.get("schema_version", 0)), 5),
        "project": "ACE-2",
        "vertical": "chip_design",
        "stage": "environment",
        "contract_id": CONTRACT,
        "scope": TOP_LEVEL_SCOPE,
        "reviewer_role": "independent_l2_environment",
        "reviewer_status": "done",
        "status": "accepted",
        "verdict": "done",
        "stage_closing": True,
        "implementation_authorized": False,
        "checklist": copy.deepcopy(ENVIRONMENT_CHECKLIST),
        "environment_checklist": copy.deepcopy(ENVIRONMENT_CHECKLIST),
        "audit": artifact(REVIEW_BINDING_AUDIT),
        "architecture_decision": artifact(ARCH_REVIEW),
        "public_projection_hash_algorithm": "sha256-json-sort-keys-compact-utf8-v1",
        "public_projection_hashes": current_projection_hashes,
        "preserved_contracts": copy.deepcopy(preserved),
        "manager_stage_transition_owner": "Manager",
        "planner_stage_transition_authorized": False,
        "downstream_work_started": False,
    })
    normalized["reviewed_at_utc"] = raw.get("reviewed_at_utc", verdict.get("reviewed_at_utc"))
    return normalized


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def session_id() -> str:
    configured = os.environ.get("ARGUS_SKILL_SESSION_ID")
    if configured:
        return configured
    session_file = ROOT / ".ace2-session.json"
    if session_file.is_file():
        value = load(session_file).get("sid")
        if isinstance(value, str) and value:
            return value
    return ROOT.name


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


def preflight() -> dict[str, Any]:
    pipeline = load(PIPELINE)
    audit = load(AUDIT)
    packet = load(PREFLIGHT)
    architecture_review = load(ARCH_REVIEW)
    public = load(PUBLIC)
    require(pipeline.get("current_stage") == "environment", "Manager stage is not environment")
    require(audit.get("contract_binding", {}).get("active_contract_id") == CONTRACT, "audit contract mismatch")
    require(audit.get("readiness_summary", {}).get("environment_evidence_ready_for_independent_review") is True,
            "environment evidence is not review-ready")
    require(audit.get("readiness_summary", {}).get("fresh_independent_environment_acceptance") is False,
            "environment audit already reports independent acceptance")
    require(packet.get("contract_id") == CONTRACT, "preflight contract mismatch")
    require(packet.get("status") == "pass_ready_for_independent_environment_review", "preflight is not pass")
    require(all(packet.get("checklist", {}).values()), "preflight checklist incomplete")
    require(architecture_review.get("reviewer_status") == "done", "architecture review is not done")
    require(public.get("routing_status") == "independent_environment_review_pending", "public routing is not review-pending")
    require(public.get("selected_replacement_contract", {}).get("implementation_authorized") is False,
            "public state authorizes implementation")

    checks = {
        "environment_contract": run_check([sys.executable, "tools/bind_environment_compatibility.py", "--check"]),
        "environment_preflight": run_check([sys.executable, "tools/run_environment_preflight.py", "--check"]),
        "wide_arithmetic_probe": run_check([sys.executable, "tools/run_down_projection_environment_probe.py", "--check"]),
    }
    return {
        "stage": "environment",
        "contract_id": CONTRACT,
        "checklist": packet["checklist"],
        "compatibility_findings": packet["compatibility_findings"],
        "contract_delta": packet["contract_delta"],
        "preserved_operator_contract": packet["preserved_operator_contract"],
        "claim_boundaries": packet["claim_boundaries"],
        "decisive_checks": checks,
        "artifacts": {
            "audit": artifact(AUDIT),
            "preflight": artifact(PREFLIGHT),
            "architecture_packet": artifact(PACKET),
            "architecture_review": artifact(ARCH_REVIEW),
            "toolchain": artifact(TOOLCHAIN),
            "ip_reuse": artifact(IP_REUSE),
            "delta_probe": artifact(PROBE),
            "pipeline": artifact(PIPELINE),
            "public_status_reviewed": artifact(PUBLIC),
        },
        "required_decision": {
            "done_only_if_both_environment_items_supported": True,
            "implementation_authorized_must_remain_false": True,
            "manager_owns_stage_transition": True,
            "downstream_work_started": False,
        },
    }


def archive_prior() -> list[dict[str, Any]]:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in (VERDICT, CERT):
        if not source.is_file():
            continue
        digest = sha256(source)
        destination = ARCHIVE / f"{source.stem}-{digest[:12]}{source.suffix}"
        if not destination.exists():
            shutil.copy2(source, destination)
        records.append(artifact(destination))
    if records:
        dump(ARCHIVE / "ARCHIVE_MANIFEST.json", {
            "schema_version": 1,
            "archived_at_utc": utc_now(),
            "reason": f"Preserve prior environment certification before binding {CONTRACT}.",
            "artifacts": records,
        })
    return records


def merge_artifact_records(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        for record in group:
            key = (str(record.get("path", "")), str(record.get("sha256", "")))
            merged[key] = record
    return list(merged.values())


def validate_external_review(
    raw: dict[str, Any],
    *,
    reviewed_input_archives: list[dict[str, Any]] | None = None,
    require_live_review_inputs: bool,
) -> dict[str, Any]:
    require(raw.get("schema_version") == 1, "external review schema mismatch")
    require(raw.get("project") == "ACE-2", "external review project mismatch")
    require(raw.get("stage") == "environment", "external review stage mismatch")
    require(raw.get("contract_id") == CONTRACT, "external review contract mismatch")
    require(raw.get("reviewer_role") == "independent_l2_environment", "external reviewer role mismatch")
    independence = raw.get("independence", {})
    require(independence.get("fresh_review") is True, "external review is not fresh")
    require(independence.get("reviewer_count") == 1, "external review count mismatch")
    require(independence.get("delegation_or_subagents_used") is False,
            "external review independence declaration changed")

    decision = raw.get("decision", {})
    require(decision.get("status") == "done", "external reviewer is not done")
    require(decision.get("reviewer_status") == "done", "external reviewer status is not done")
    require(decision.get("stage_closing") is True, "external review is not stage-closing")
    authority = raw.get("authority_and_scope", {})
    require(authority.get("implementation_authorized") is False, "external review authorizes implementation")
    require(authority.get("environment_to_rtl_permission_granted_by_this_decision") is False,
            "external review grants Manager-owned stage permission")
    require(authority.get("only_manager_controls_environment_to_rtl_permission") is True,
            "external review changes stage-transition ownership")
    require(authority.get("forbidden_work_respected") is True,
            "external review reports forbidden downstream work")
    require(authority.get("rtl_or_candidate_downstream_work_started_by_reviewer") is False,
            "external reviewer started downstream work")
    for item in ("environment.eda-capabilities", "environment.tool-ip-selection"):
        result = raw.get("environment_checklist_verdicts", {}).get(item, {})
        require(result.get("verdict") == "done" and result.get("supported") is True,
                f"external review did not accept {item}")

    audit_record = raw.get("bindings", {}).get("audit", {})
    verify_any_record(audit_record, "external_review.audit")
    require(resolve_record_path(audit_record).resolve() == REVIEW_BINDING_AUDIT.resolve(),
            "external review binds an unexpected audit")
    review_audit = load(REVIEW_BINDING_AUDIT)
    require(canonical_sha256(review_audit) == review_audit.get("integrity", {}).get("canonical_sha256"),
            "external review audit canonical hash mismatch")
    require(audit_record.get("canonical_sha256") == review_audit.get("integrity", {}).get("canonical_sha256"),
            "external decision canonical audit binding mismatch")
    require(review_audit.get("contract_id") == CONTRACT, "external review audit contract mismatch")
    require(review_audit.get("status") == "ready_for_exactly_one_fresh_independent_l2_environment_review",
            "external review audit was not review-ready")
    require(review_audit.get("implementation_authorized") is False,
            "external review audit authorizes implementation")
    require(all(review_audit.get("environment_checklist", {}).values()),
            "external review audit checklist incomplete")
    require(review_audit.get("decisive_readiness_check", {}).get("ready_for_independent_l2") is True,
            "external review audit was not ready for independent L2")
    require(review_audit.get("decisive_readiness_check", {}).get("manager_environment_to_rtl_permission") is False,
            "external review audit grants stage permission")

    checker = raw.get("decisive_checker", {})
    require(checker.get("command") == "python tools/run_down_projection_environment_binding_audit.py --check",
            "external decisive checker changed")
    require(checker.get("exit_code") == 0 and checker.get("result") == "pass",
            "external decisive checker did not pass")

    for name in ("immutable_mission", "immutable_handoff"):
        verify_any_record(raw.get("bindings", {}).get(name, {}), f"external_review.{name}")

    if require_live_review_inputs:
        run_check([sys.executable, "tools/run_down_projection_environment_binding_audit.py", "--check"])
    else:
        archives = reviewed_input_archives or []
        by_original = {str(item.get("original_path", "")): item for item in archives}
        required_reviewed_records = [review_audit["first_party_capability_evidence"]["audit"]]
        required_reviewed_records.extend(
            item for item in review_audit.get("bindings", [])
            if item.get("path") == "research/PUBLIC_STATUS.json"
        )
        for record in required_reviewed_records:
            archived = by_original.get(str(record.get("path", "")))
            require(isinstance(archived, dict), f"reviewed snapshot archive missing: {record.get('path')}")
            archive_record = archived.get("archive", {})
            verify_record(archive_record, f"reviewed_snapshot[{record.get('path')}]")
            require(archive_record.get("bytes") == record.get("bytes"),
                    f"reviewed snapshot byte mismatch: {record.get('path')}")
            require(archive_record.get("sha256") == record.get("sha256"),
                    f"reviewed snapshot hash mismatch: {record.get('path')}")
    return review_audit


def archive_external_review_inputs(review_audit: dict[str, Any], reviewed_at: str) -> list[dict[str, Any]]:
    slug = reviewed_at.replace("-", "").replace(":", "").replace(".", "").replace("Z", "Z")
    destination_dir = ARCHIVE / f"fresh_l2_review_inputs_{slug}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    records = [review_audit["first_party_capability_evidence"]["audit"]]
    records.extend(
        item for item in review_audit.get("bindings", [])
        if item.get("path") == "research/PUBLIC_STATUS.json"
    )
    archived: list[dict[str, Any]] = []
    for record in records:
        verify_record(record, f"reviewed_input[{record.get('path')}]")
        source = ROOT / str(record["path"])
        destination = destination_dir / f"{source.stem}-{str(record['sha256'])[:12]}{source.suffix}"
        if not destination.exists():
            shutil.copy2(source, destination)
        archive_record = artifact(destination)
        require(archive_record["bytes"] == record["bytes"] and archive_record["sha256"] == record["sha256"],
                f"failed to preserve reviewed input: {record.get('path')}")
        archived.append({
            "original_path": record["path"],
            "reviewed_bytes": record["bytes"],
            "reviewed_sha256": record["sha256"],
            "archive": archive_record,
        })
    dump(destination_dir / "MANIFEST.json", {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "reviewed_at_utc": reviewed_at,
        "reason": "Preserve mutable live inputs exactly as inspected by the fresh independent L2 environment review.",
        "artifacts": archived,
    })
    return archived


def normalized_external_payload(
    raw: dict[str, Any],
    review_audit: dict[str, Any],
    reviewed_input_archives: list[dict[str, Any]],
) -> dict[str, Any]:
    by_original = {str(item["original_path"]): item["archive"] for item in reviewed_input_archives}
    reviewed_inputs: dict[str, Any] = {
        "environment_binding_audit": artifact(REVIEW_BINDING_AUDIT),
        "audit": by_original["research/ENVIRONMENT_AUDIT.json"],
        "public_status_reviewed": by_original["research/PUBLIC_STATUS.json"],
    }
    selected_paths = {
        "architecture_packet": PACKET.relative_to(ROOT).as_posix(),
        "architecture_review": ARCH_REVIEW.relative_to(ROOT).as_posix(),
        "pipeline": PIPELINE.relative_to(ROOT).as_posix(),
    }
    bindings = {str(item.get("path")): item for item in review_audit.get("bindings", [])}
    for name, path in selected_paths.items():
        require(path in bindings, f"external review audit missing {name}")
        verify_record(bindings[path], f"external_review.{name}")
        reviewed_inputs[name] = bindings[path]
    reviewed_inputs["toolchain"] = artifact(TOOLCHAIN)
    reviewed_inputs["ip_reuse"] = artifact(IP_REUSE)
    reviewed_inputs["delta_probe"] = artifact(PROBE)
    return {
        "reviewed_at_utc": raw["reviewed_at_utc"],
        "evidence": {"artifacts": reviewed_inputs},
        "decision": raw["decision"],
    }


def update_audit_after_review(raw_reviewed_audit: dict[str, Any], reviewed_at: str) -> None:
    audit = load(AUDIT)
    audit["contract_binding"]["environment_review_status"] = "done"
    audit["readiness_summary"]["fresh_independent_environment_acceptance"] = True
    audit["readiness_summary"]["environment_evidence_ready_for_independent_review"] = True
    audit["independent_review"] = {
        "status": "done",
        "reviewed_at_utc": reviewed_at,
        "reviewed_audit": raw_reviewed_audit,
        "raw_decision": artifact(RAW_DECISION),
        "implementation_authorized": False,
        "stage_transition_owner": "Manager",
    }
    audit["generated_at_utc"] = reviewed_at
    audit["integrity"] = {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None}
    audit["integrity"]["canonical_sha256"] = canonical_sha256(audit)
    dump(AUDIT, audit)
    AUDIT_MD.write_text(
        "# ACE-2 Environment Audit\n\n"
        f"- Contract: `{CONTRACT}`\n"
        "- Manager-owned stage observed: `environment`\n"
        f"- Architecture packet: `{PACKET.relative_to(ROOT).as_posix()}`\n"
        f"- Architecture packet SHA-256: `{sha256(PACKET)}`\n"
        "- EDA capability gate: `pass`\n"
        "- Tool/IP selection gate: `pass`\n"
        "- Bootstrap raw capability artifacts: `27`, hash-verified and not rerun\n"
        f"- Contract delta probe: `{PROBE.relative_to(ROOT).as_posix()}` (`pass`)\n"
        "- Fresh independent environment acceptance: `done`\n"
        "- Implementation authorized: `false`\n\n"
        "The independent review accepted both environment checklist items. The delta "
        "probe is synthetic environment evidence, not successor RTL. No verification, "
        "PPA, prototype, benchmark, signoff, GDS, tapeout, or silicon claim is made.\n",
        encoding="utf-8",
    )


def update_public_after_review(reviewed_at: str) -> None:
    public = load(PUBLIC)
    status = "environment_independently_accepted_manager_rtl_entry_pending"
    decision = "down_projection_residual_fusion_environment_independently_accepted"
    manager_action = "manager_may_advance_environment_to_rtl_in_order"
    latest_environment = {
        "status": "independent_review_done_manager_stage_transition_pending",
        "contract_binding": CONTRACT,
        "generated_at_utc": reviewed_at,
        "checklist": {
            "environment.eda-capabilities": True,
            "environment.tool-ip-selection": True,
        },
        "bootstrap_raw_artifact_count": 27,
        "contract_delta_probe": artifact(PROBE),
        "raw_bootstrap_probes_rerun": False,
        "reviewer_verdict": "done",
        "implementation_authorized": False,
        "stage_transition_owner": "Manager",
    }
    latest_environment["contract_delta_probe"].pop("exists", None)
    public["latest_environment_stage"] = copy.deepcopy(latest_environment)
    public["latest_decision"] = decision
    public["routing_status"] = "manager_stage_advance_pending"
    public["routing_authorized"] = "manager_may_advance_environment_to_rtl_only"
    public["generated_at_utc"] = reviewed_at
    public["last_updated_utc"] = reviewed_at

    selected = public["selected_replacement_contract"]
    selected["status"] = status
    selected["environment_review_status"] = "done"
    selected["implementation_authorized"] = False
    selected["required_manager_action"] = manager_action

    for name in ("dashboard_fields", "implementation_frontier"):
        container = public[name]
        container["current_stage"] = "environment"
        container["current_mode"] = "ADVANCE"
        container["latest_decision"] = decision
        container["routing_status"] = "manager_stage_advance_pending"
        container["required_manager_action"] = manager_action
        container["required_operator_action"] = "none"
        container["latest_environment_stage"] = copy.deepcopy(latest_environment)
        candidate = container["candidate_mechanism"]
        candidate["status"] = status
        candidate["environment_review_status"] = "done"
        candidate["implementation_authorized"] = False

    stage = public["stage"]
    stage["current_stage"] = "environment"
    stage["current_stage_status"] = "independent_review_done_manager_stage_transition_pending"
    stage["completed_prior_stages"] = ["definition", "architecture"]
    stage["current_stage_checklist"] = latest_environment["checklist"]
    stage["planner_may_advance_stage"] = False
    stage["stage_transition_owner"] = "Manager"
    stage["stage_closing"] = True

    public["blockers"] = [
        item for item in public.get("blockers", [])
        if item.get("id") not in {
            "independent_environment_review_pending",
            "environment_independent_review_pending",
            "manager_stage_advance_pending",
        }
    ]
    public["blockers"].append({
        "id": "manager_stage_advance_pending",
        "severity": "routing",
        "status": "open",
        "owner": "Manager",
        "detail": "Independent environment review is done; Manager alone may advance environment to rtl.",
    })

    claims = [
        item for item in public.get("public_claims", [])
        if "current Manager-owned stage is" not in str(item.get("claim", ""))
        and "independent environment reviewer certifies" not in str(item.get("claim", ""))
        and "environment evidence is ready" not in str(item.get("claim", ""))
    ]
    claims.insert(0, {"claim": "current Manager-owned stage is environment", "evidence": ["research/PIPELINE_STATE.json"]})
    claims.insert(1, {
        "claim": f"independent environment reviewer certifies both environment checklist items for {CONTRACT}; implementation remains stage-locked",
        "evidence": ["research/ENVIRONMENT_REVIEWER_VERDICT.json", "research/ENVIRONMENT_L2_CERTIFICATION.md"],
    })
    public["public_claims"] = claims

    paths = {
        str(item.get("path")) for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path")
    }
    paths.update({
        AUDIT.relative_to(ROOT).as_posix(),
        PREFLIGHT.relative_to(ROOT).as_posix(),
        TOOLCHAIN.relative_to(ROOT).as_posix(),
        IP_REUSE.relative_to(ROOT).as_posix(),
        PROBE.relative_to(ROOT).as_posix(),
        PACKET.relative_to(ROOT).as_posix(),
        ARCH_REVIEW.relative_to(ROOT).as_posix(),
        RAW_DECISION.relative_to(ROOT).as_posix(),
        VERDICT.relative_to(ROOT).as_posix(),
        CERT.relative_to(ROOT).as_posix(),
        CHECKPOINT.relative_to(ROOT).as_posix(),
    })
    public["artifact_hashes"] = [artifact(ROOT / path) for path in sorted(paths) if (ROOT / path).is_file()]
    for record in public["artifact_hashes"]:
        record.pop("exists", None)
    public["integrity"] = {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None}
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def publish(
    raw_payload: dict[str, Any],
    reviewed_preflight: dict[str, Any] | None,
    archived: list[dict[str, Any]],
    *,
    reviewed_input_archives: list[dict[str, Any]] | None = None,
    reviewed_binding_audit: dict[str, Any] | None = None,
    review_type: str = "independent_environment_stage_closing_review",
) -> None:
    reviewed_at = raw_payload["reviewed_at_utc"]
    reviewed_audit = raw_payload["evidence"]["artifacts"]["audit"]
    update_audit_after_review(reviewed_audit, reviewed_at)
    update_public_after_review(reviewed_at)

    run_check([sys.executable, "tools/run_environment_preflight.py"])
    audit = load(AUDIT)
    verdict = {
        "schema_version": 5,
        "project": "ACE-2",
        "vertical": "chip_design",
        "stage": "environment",
        "stage_closing": True,
        "contract_id": CONTRACT,
        "verdict": "done",
        "reviewed_at_utc": reviewed_at,
        "reviewer_role": "independent_l2_environment",
        "review_type": review_type,
        "checklist": copy.deepcopy(ENVIRONMENT_CHECKLIST),
        "implementation_authorized": False,
        "manager_stage_transition_owner": "Manager",
        "planner_stage_transition_authorized": False,
        "manager_recommendation": "advance_environment_to_rtl_in_order",
        "required_operator_action": "none_operator_directive_current_manager_transition_required",
        "downstream_work_started": False,
        "required_repairs_before_manager_advance": [],
        "reviewed_inputs": raw_payload["evidence"]["artifacts"],
        "reviewed_preflight": reviewed_preflight,
        "reviewed_environment_binding_audit": reviewed_binding_audit,
        "reviewed_input_archives": reviewed_input_archives or [],
        "current_audit": artifact(AUDIT),
        "current_preflight": artifact(PREFLIGHT),
        "raw_reviewer_decision": artifact(RAW_DECISION),
        "tool_ip_selection": [artifact(TOOLCHAIN), artifact(IP_REUSE)],
        "contract_delta_probe": artifact(PROBE),
        "archived_prior_certifications": archived,
        "reviewer_reason": raw_payload["decision"].get("reason", ""),
        "claim_boundaries_confirmed": audit.get("claim_boundaries", []),
    }
    verdict = apply_top_level_verdict_schema(verdict, load(RAW_DECISION), load(REVIEW_BINDING_AUDIT))
    dump(VERDICT, verdict)

    CERT.write_text(
        f"# ACE-2 environment L2 certification\n\n"
        f"Date: {reviewed_at[:10]}\n\n"
        f"## Certified contract\n\n"
        f"- Contract: `{CONTRACT}`\n"
        "- Stage: `environment`\n"
        "- Stage-closing review: `true`\n"
        "- Independent verdict: `DONE`\n"
        "- Implementation authorized: `false`; Manager transition is still required\n\n"
        "## Evidence\n\n"
        f"- Environment audit: `{sha256(AUDIT)}`\n"
        f"- Consolidated preflight: `{sha256(PREFLIGHT)}`\n"
        f"- Contract delta probe: `{sha256(PROBE)}`\n"
        f"- Raw independent decision: `{sha256(RAW_DECISION)}`\n"
        "- Bootstrap capability artifacts: `27`, all byte/hash verified, not rerun\n\n"
        "Both environment checklist items pass. The synthetic contract-delta probe "
        "exercised signed-96/unsigned-64 divide/remainder, ties-to-even rounding, and "
        "signed-int8 saturation through Icarus, Verilator, and pinned ORFS Yosys. "
        "No successor RTL, verification, PPA, prototype, benchmark, signoff, GDS, "
        "tapeout, or silicon work was run or claimed.\n\n"
        "## Routing boundary\n\n"
        "Manager may advance `environment -> rtl` in order. The 2.0 mm² non-SRAM "
        "cap, 100 MHz floor, accepted prefix, first unsupported operator, and historical "
        "PPA frontier remain unchanged.\n\n"
        f"Reviewer reason: {raw_payload['decision'].get('reason', '')}\n",
        encoding="utf-8",
    )

    CHECKPOINT.write_text(
        f"# Goal\n\nClose the environment gate for `{CONTRACT}` without changing the "
        "Manager-owned stage or operator-owned targets.\n\n"
        "# Current State\n\nIndependent L2 environment review certified both checklist items. "
        "The Manager-owned stage remains `environment`; Manager alone may advance to `rtl`. "
        "`implementation_authorized=false` remains stage-locked.\n\n"
        "The accepted prefix remains through `layer_0.v_proj`; first unsupported remains "
        "`layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap and "
        "100 MHz floor are unchanged.\n\n"
        "# Verified Evidence\n\n"
        "- 27 bootstrap environment artifacts byte/hash verify without rerun.\n"
        "- The contract-specific synthetic arithmetic probe passes Icarus, Verilator, and pinned ORFS Yosys.\n"
        "- `make environment-contract-check environment-preflight` passes.\n"
        f"- Independent decision: `{RAW_DECISION.relative_to(ROOT).as_posix()}`.\n"
        f"- Canonical verdict: `{VERDICT.relative_to(ROOT).as_posix()}`.\n\n"
        "# Boundaries\n\nNo successor RTL, dataset discriminator, verification, PPA, prototype, "
        "benchmark, signoff, GDS, tapeout, or silicon work was run in this stage.\n\n"
        "# Required Action\n\nManager may advance `environment -> rtl` in order.\n",
        encoding="utf-8",
    )
    update_public_after_review(reviewed_at)


def check_existing() -> None:
    pipeline = load(PIPELINE)
    audit = load(AUDIT)
    preflight_packet = load(PREFLIGHT)
    raw = load(RAW_DECISION)
    verdict = load(VERDICT)
    public = load(PUBLIC)
    require(pipeline.get("current_stage") == "environment", "Manager stage changed")
    require(raw.get("contract_id") == CONTRACT, "raw decision contract mismatch")
    require(raw.get("decision", {}).get("status") == "done", "independent reviewer is not done")
    if raw.get("scope") == "fresh_independent_l2_environment_stage_closing_review_only":
        review_audit = validate_external_review(
            raw,
            reviewed_input_archives=verdict.get("reviewed_input_archives", []),
            require_live_review_inputs=False,
        )
    else:
        review_audit = load(REVIEW_BINDING_AUDIT)
    require(verdict.get("reviewer_status") == "done", "top-level reviewer status is not done")
    require(verdict.get("status") == "accepted", "top-level environment decision is not accepted")
    require(verdict.get("verdict") == "done" and verdict.get("contract_id") == CONTRACT,
            "canonical verdict is not done")
    require(verdict.get("scope") == TOP_LEVEL_SCOPE, "top-level environment verdict scope changed")
    require(verdict.get("stage_closing") is True, "verdict is not stage-closing")
    require(verdict.get("implementation_authorized") is False, "verdict authorizes implementation")
    require(verdict.get("checklist") == ENVIRONMENT_CHECKLIST,
            "top-level environment checklist is not explicitly true")
    require(verdict.get("environment_checklist") == ENVIRONMENT_CHECKLIST,
            "top-level environment checklist alias is not explicitly true")
    require(verdict.get("required_repairs_before_manager_advance") == [], "verdict has open repairs")
    verify_record(verdict.get("audit", {}), "verdict.audit")
    require(verdict.get("audit") == artifact(REVIEW_BINDING_AUDIT),
            "top-level audit binding is not exact")
    verify_record(verdict.get("architecture_decision", {}), "verdict.architecture_decision")
    require(verdict.get("architecture_decision") == artifact(ARCH_REVIEW),
            "top-level architecture decision binding is not exact")
    require(verdict.get("public_projection_hash_algorithm") ==
            "sha256-json-sort-keys-compact-utf8-v1", "public projection hash algorithm changed")
    require(verdict.get("public_projection_hashes") == current_public_projection_hashes(public),
            "top-level public projection hashes are not current")
    require(verdict.get("public_projection_hashes") ==
            review_audit.get("public_projection_binding", {}).get("accepted_architecture_hashes"),
            "top-level public projection hashes do not match accepted architecture")
    require(verdict.get("preserved_contracts") == review_audit.get("preserved_operator_contract"),
            "top-level preserved contracts do not match the reviewed audit")
    require(audit.get("readiness_summary", {}).get("fresh_independent_environment_acceptance") is True,
            "audit acceptance status stale")
    require(audit.get("contract_binding", {}).get("environment_review_status") == "done",
            "audit review status stale")
    require(canonical_sha256(audit) == audit.get("integrity", {}).get("canonical_sha256"),
            "audit canonical hash mismatch")
    require(preflight_packet.get("contract_id") == CONTRACT and all(preflight_packet.get("checklist", {}).values()),
            "current preflight incomplete")
    require(public.get("routing_status") == "manager_stage_advance_pending", "public routing stale")
    require(public.get("selected_replacement_contract", {}).get("environment_review_status") == "done",
            "public environment review status stale")
    require(public.get("selected_replacement_contract", {}).get("implementation_authorized") is False,
            "public authorizes implementation")
    require(public.get("supported_layer_operator_prefix") == PREFIX, "public prefix changed")
    require(public.get("first_unsupported_layer_operator") == "layer_0.rope_q",
            "public first unsupported operator changed")
    require(canonical_sha256(public) == public.get("integrity", {}).get("canonical_sha256"),
            "PUBLIC_STATUS canonical hash mismatch")
    require("/home/" not in json.dumps(public, sort_keys=True), "PUBLIC_STATUS exposes a private path")
    for record in verdict.get("current_audit", {}), verdict.get("current_preflight", {}), verdict.get("raw_reviewer_decision", {}):
        verify_record(record, "verdict_artifact")
    for index, record in enumerate(public.get("artifact_hashes", [])):
        verify_record(record, f"public_artifact[{index}]")
    require(CONTRACT in CERT.read_text(encoding="utf-8"), "certification contract missing")
    require(CONTRACT in CHECKPOINT.read_text(encoding="utf-8"), "checkpoint contract missing")
    print(
        "ACE2_DOWN_PROJECTION_ENVIRONMENT_STAGE_REVIEW_PASS "
        f"contract={CONTRACT} reviewer=done checklist=2/2 "
        "implementation_authorized=false manager_transition_owner=Manager"
    )


def main() -> None:
    if "--check" in sys.argv[1:]:
        check_existing()
        return
    if "--normalize-existing-schema" in sys.argv[1:]:
        raw = load(RAW_DECISION)
        existing_verdict = load(VERDICT)
        require(raw.get("decision", {}).get("status") == "done",
                "existing independent decision is not done")
        review_audit = validate_external_review(
            raw,
            reviewed_input_archives=existing_verdict.get("reviewed_input_archives", []),
            require_live_review_inputs=False,
        )
        dump(VERDICT, apply_top_level_verdict_schema(existing_verdict, raw, review_audit))
        update_public_after_review(raw["reviewed_at_utc"])
        check_existing()
        return
    if "--publish-existing" in sys.argv[1:]:
        payload = load(RAW_DECISION)
        require(payload.get("decision", {}).get("status") == "done",
                "existing independent decision is not done")
        existing_verdict = load(VERDICT) if VERDICT.is_file() else {}
        archived = merge_artifact_records(
            existing_verdict.get("archived_prior_certifications", [])
            if existing_verdict.get("contract_id") == CONTRACT else [],
            archive_prior(),
        )
        if payload.get("scope") == "fresh_independent_l2_environment_stage_closing_review_only":
            review_audit = validate_external_review(payload, require_live_review_inputs=True)
            reviewed_input_archives = archive_external_review_inputs(review_audit, payload["reviewed_at_utc"])
            normalized = normalized_external_payload(payload, review_audit, reviewed_input_archives)
            publish(
                normalized,
                None,
                archived,
                reviewed_input_archives=reviewed_input_archives,
                reviewed_binding_audit=artifact(REVIEW_BINDING_AUDIT),
                review_type="fresh_independent_l2_environment_stage_closing_review",
            )
        else:
            reviewed_preflight = payload.get("evidence", {}).get("artifacts", {}).get("preflight")
            require(isinstance(reviewed_preflight, dict), "existing decision lacks preflight binding")
            publish(payload, reviewed_preflight, archived)
        check_existing()
        return

    verified = preflight()
    archived = archive_prior()
    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(
        backend=backend_name,
        runner_bin=resolve_runner_bin_setting("reviewer") or None,
    )
    sid = session_id()
    runner.set_usage_context(
        project_root=global_root() / "projects" / sid,
        global_root=global_root(),
        mission_id=MISSION_ID,
    )
    package_root = Path(argus_skill.__file__).resolve().parent
    reviewer_skill = (
        package_root / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md"
    ).read_text(encoding="utf-8")

    objective = (
        "Independently decide whether both environment checklist items are satisfied for "
        "shared_down_projection_residual_fusion_v1 and whether Manager may advance the "
        "environment stage to rtl in order."
    )
    instruction = (
        "Inspect design/CHIP_SCOPE.json first, then research/ENVIRONMENT_AUDIT.json, "
        "research/TOOLCHAIN_CANDIDATES.md, research/IP_REUSE_PLAN.md, the consolidated "
        "environment preflight packet, the accepted architecture packet/review, the synthetic "
        "wide-arithmetic probe and its logs, research/PUBLIC_STATUS.json, and CHECKPOINT.md. "
        "You may run only `make environment-contract-check environment-preflight`, "
        "`python tools/run_down_projection_environment_probe.py --check`, and short read-only "
        "hash/content checks. Edit only CHECKPOINT.md if the reviewer workflow requires a handoff. "
        "Do not run or edit successor RTL, datasets, verification, synthesis/PPA, prototype, "
        "benchmark, signoff, GDS, tapeout, or silicon work. Return done only if the 27 bootstrap "
        "artifacts hash-verify, the contract-specific signed-96/unsigned-64 probe genuinely passes, "
        "maintained tool/IP selections remain current, no dependency/license class changed, both "
        "environment checklist items pass, the public prefix/frontier/targets are preserved, and "
        "implementation_authorized remains false. A done verdict is stage-closing evidence only; "
        "Manager alone changes current_stage. Otherwise return continue with exact repairs."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary=(
            "The active architecture review is accepted. The environment binder now targets the "
            "same contract, preserves 27 hash-verified bootstrap artifacts, adds a passing synthetic "
            "Icarus/Verilator/ORFS-Yosys wide-arithmetic probe, updates maintained tool/IP selections, "
            "and keeps implementation and all downstream stages locked."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="environment_stage_closing_review_only",
        checkpoint_path=str(CHECKPOINT),
        escalate_hint=(
            "If accepted, recommend only the Manager-owned environment-to-rtl transition. "
            "Do not authorize implementation or relax any operator-owned target."
        ),
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort(
                "ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"
            ),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="workspace-write",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    reviewed_at = utc_now()
    raw = asdict(review)
    payload = {
        "schema_version": 1,
        "reviewed_at_utc": reviewed_at,
        "reviewer_role": "independent_l2_environment",
        "reviewer_status": raw.get("status", "continue"),
        "stage": "environment",
        "stage_closing": True,
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "scope": "environment_stage_closing_review_only",
        "evidence": verified,
        "decision": raw,
    }
    dump(RAW_DECISION, payload)
    require(raw.get("status") == "done", raw.get("next_action") or raw.get("reason") or "independent review did not pass")
    publish(payload, verified["artifacts"]["preflight"], archived)
    check_existing()


if __name__ == "__main__":
    main()
