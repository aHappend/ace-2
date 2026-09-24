#!/usr/bin/env python3
"""Run one independent RTL-checklist review for the DPRF candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
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
from validate_down_projection_residual_fusion_evidence import validate_artifact_tree


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
RTL = ROOT / "rtl/ace2_down_projection_residual_fusion_core.sv"
PACKET = ROOT / "evidence/shared_down_projection_residual_fusion_v1/latest/PRECHECK.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
OUT = ROOT / "evidence/review/rtl_checklist_shared_down_projection_residual_fusion_v1/decision.json"
ARCHIVE = OUT.parent / "archive"
MISSION_ID = "rtl-checklist-shared-down-projection-residual-fusion-v1"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def scoped_artifact(path: Path, *, scope: str, recursive: bool = False) -> dict[str, Any]:
    record = artifact(path)
    record["binding_scope"] = scope
    record["content_validation"] = "recursive_json" if recursive else "whole_file_only"
    return record


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def aggregate_hash(records: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(records):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(records[path].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_record(record: dict[str, Any], label: str) -> dict[str, Any]:
    relative = str(record.get("path") or "")
    path = ROOT / relative
    require(relative and path.is_file(), f"{label}: missing {relative}")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label}: bytes differ")
    require(sha256_file(path) == record.get("sha256"), f"{label}: hash differs")
    return artifact(path)


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


def archive_prior_standalone(packet: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    prior: dict[str, Any] | None = None
    if OUT.is_file():
        candidate = load(OUT)
        if candidate.get("stage_closing") is True:
            raise RuntimeError("fresh stage-closing RTL review already exists")
        prior = candidate
        prior_sha = sha256_file(OUT)
        archive_path = ARCHIVE / f"standalone_decision.{prior_sha}.json"
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            shutil.copy2(OUT, archive_path)
        require(sha256_file(archive_path) == prior_sha, "standalone review archive hash differs")
    else:
        archives = sorted(ARCHIVE.glob("standalone_decision.*.json"))
        require(len(archives) == 1, "exactly one archived standalone RTL review is required")
        archive_path = archives[0]
        prior = load(archive_path)

    require(prior.get("reviewer_status") == "done", "prior standalone review is not done")
    require(prior.get("stage_closing") is False, "prior review is not standalone")
    require(prior.get("scope") == "standalone_candidate_rtl_checklist_only",
            "prior review scope differs")
    transition_record = packet.get("transition_binding", {})
    transition_path = ROOT / str(transition_record.get("path") or "")
    require(transition_path.is_file(), "immutable transition binding is missing")
    transition = load(transition_path)
    candidate_transition = transition.get("candidate_transition", {})
    require(
        prior.get("candidate_rtl_hash") == candidate_transition.get("previous_candidate_rtl_hash"),
        "prior review does not match the transition predecessor candidate",
    )
    require(
        packet.get("candidate_rtl_hash") == candidate_transition.get("successor_candidate_rtl_hash"),
        "PRECHECK does not match the transition successor candidate",
    )
    require(prior.get("live_rtl_source_sha256") == sha256_file(RTL),
            "prior review RTL source hash differs")
    require(prior.get("checklist") == CHECKLIST, "prior review checklist differs")
    require(bool(prior.get("reviewer_thread_id")), "prior reviewer identity is missing")
    return prior, archive_path


def preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    packet = load(PACKET)
    manifest = load(MANIFEST)
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(packet.get("contract_id") == CONTRACT, "preflight contract mismatch")
    require(packet.get("status") == "pass_ready_for_independent_rtl_review", "preflight not ready")
    require(packet.get("stage_closing") is False, "preflight unexpectedly closes stage")
    require(packet.get("checklist") == CHECKLIST, "preflight checklist differs")
    require(not any(packet.get("prohibited_runs", {}).values()), "downstream run recorded")
    require(canonical_sha256(packet) == packet.get("integrity", {}).get("canonical_sha256"),
            "preflight canonical hash mismatch")
    recursive = validate_artifact_tree(packet, label=PACKET.relative_to(ROOT).as_posix())
    source_hashes = packet.get("source_hashes", {})
    require(isinstance(source_hashes, dict) and source_hashes, "source hashes missing")
    for relative, expected in source_hashes.items():
        path = ROOT / relative
        require(path.is_file(), f"missing bound source: {relative}")
        require(sha256_file(path) == expected, f"bound source changed: {relative}")
    control_hashes = packet.get("control_source_hashes", {})
    require(isinstance(control_hashes, dict) and control_hashes, "control source hashes missing")
    for relative, expected in control_hashes.items():
        path = ROOT / relative
        require(path.is_file(), f"missing bound control source: {relative}")
        require(sha256_file(path) == expected, f"bound control source changed: {relative}")
    require(aggregate_hash(source_hashes) == packet.get("candidate_rtl_hash"),
            "candidate aggregate hash mismatch")
    require(manifest.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
            "manifest candidate hash mismatch")
    require(manifest.get("candidate_interface", {}).get("ports"), "manifest ports missing")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST,
            "manifest checklist differs")
    prior, prior_path = archive_prior_standalone(packet)
    verified = {
        "precheck": artifact(PACKET),
        "manifest": artifact(MANIFEST),
        "pipeline_state": artifact(PIPELINE),
        "rtl_source": artifact(RTL),
        "traceability": artifact(ROOT / "design/RTL_TRACEABILITY.md"),
        "metadata_manifest": artifact(ROOT / "reference/generated/down_projection_residual_fusion_metadata.json"),
        "metadata_binary": artifact(ROOT / "reference/generated/down_projection_residual_fusion_metadata.bin"),
        "logs": {name: verify_record(record, f"logs.{name}") for name, record in sorted(packet["logs"].items())},
        "interface_xml": {
            name: verify_record(record, f"interface_xml.{name}")
            for name, record in sorted(packet["interface_xml"].items())
        },
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "source_hashes": source_hashes,
        "control_source_hashes": control_hashes,
        "recursive_evidence_validation": asdict(recursive),
        "stage_checklist": CHECKLIST,
        "prohibited_runs": packet["prohibited_runs"],
        "prior_standalone_review": artifact(prior_path),
        "transition_binding": verify_record(packet["transition_binding"], "transition_binding"),
    }
    return packet, verified, prior


def candidate_projection(packet: dict[str, Any], status: str, review: dict[str, Any] | str) -> dict[str, Any]:
    return {
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "candidate_rtl_hash_scope": "standalone_not_shell_admitted_no_candidate_ppa",
        "contract_id": CONTRACT,
        "derived_latency_cycles_per_lane": packet["schedule"]["valid_cycles_per_output_lane"],
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "rtl_review_stage_closing": status == "done",
        "stage_closing": False,
        "status": (
            "rtl_stage_closing_review_accepted_waiting_manager_advance"
            if status == "done"
            else "fresh_independent_rtl_stage_closing_review_pending"
        ),
        "review": review,
    }


def restore_public_probe_binding(public: dict[str, Any]) -> None:
    probe = copy.deepcopy(public["selected_replacement_contract"]["probe"])
    probe.update(
        {
            "classification": "exploratory_float64_proxy_not_scale32_construct_evidence",
            "permitted_use": "historical_context_only",
        }
    )
    public["selected_replacement_contract"]["probe"] = copy.deepcopy(probe)
    public["architecture_proposal_gate"]["probe"] = copy.deepcopy(probe)
    claims = [
        claim
        for claim in public.get("public_claims", [])
        if "retained float64 probe is exploratory historical context only"
        not in str(claim.get("claim", ""))
    ]
    claims.append(
        {
            "claim": "the retained float64 probe is exploratory historical context only; no exact Scale32 two-dataset discriminator has run",
            "evidence": [
                probe["path"],
                "evidence/shared_down_projection_residual_fusion_v1/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json",
            ],
        }
    )
    public["public_claims"] = claims


def prepare_stage_closing_review() -> None:
    packet, _verified, _prior = preflight()
    manifest = load(MANIFEST)
    public = load(PUBLIC)
    require(
        manifest.get("independent_reviewer_acceptance", {}).get("status")
        == "pending_fresh_independent_rtl_stage_closing_review",
        "manifest is not in the final pending-review state",
    )
    require(
        public.get("routing_status") == "fresh_independent_rtl_stage_closing_review_pending",
        "public status is not in the final pending-review state",
    )
    print(f"ACE2_DPRF_RTL_STAGE_CLOSING_REVIEW_READY candidate={packet['candidate_id']}")
    return

    packet, _verified, prior = preflight()
    prior_sha = sha256_file(OUT) if OUT.is_file() else ""
    prior_archive = ARCHIVE / f"standalone_decision.{prior_sha}.json" if prior_sha else sorted(ARCHIVE.glob("standalone_decision.*.json"))[0]

    manifest = load(MANIFEST)
    manifest["candidate_status"] = "fresh_independent_rtl_stage_closing_review_pending"
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "fresh_independent_rtl_stage_closing_review_pending",
        "scope": "rtl_stage_close_for_verification_entry_only",
        "stage_closing": False,
        "evidence": PACKET.relative_to(ROOT).as_posix(),
        "prior_standalone_review": artifact(prior_archive),
        "reviewer_status": "pending",
    }
    manifest["independent_reviewer_acceptance"] = {
        "architecture_review": "accepted",
        "environment_review": "accepted_manager_advanced_to_rtl",
        "rtl_review": "pending_fresh_independent_stage_closing_review",
        "status": "pending_fresh_independent_rtl_stage_closing_review",
    }
    manifest["traceability"]["architecture_contract_gap"] = {
        "status": "candidate_implemented_prior_standalone_review_historical_fresh_stage_closing_review_pending",
        "resolution_owner": "fresh_independent_l2_reviewer",
    }
    dump(MANIFEST, manifest)

    public = load(PUBLIC)
    review_path = prior_archive.relative_to(ROOT).as_posix()
    projection = candidate_projection(packet, "pending", review_path)
    public["latest_decision"] = "down_projection_residual_fusion_fresh_rtl_stage_closing_review_pending"
    restore_public_probe_binding(public)
    public["stage"]["current_stage_status"] = "fresh_independent_rtl_stage_closing_review_pending"
    public["stage"]["stage_closing"] = False
    public["stage"]["current_stage_evidence"] = [
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        PACKET.relative_to(ROOT).as_posix(),
        review_path,
    ]
    public["latest_rtl_candidate"] = copy.deepcopy(projection)
    for name in ("dashboard_fields", "implementation_frontier"):
        public[name]["latest_decision"] = public["latest_decision"]
        public[name]["routing_status"] = "fresh_independent_rtl_stage_closing_review_pending"
        public[name]["latest_rtl_candidate"] = copy.deepcopy(projection)
    public["routing_status"] = "fresh_independent_rtl_stage_closing_review_pending"
    public["routing_authorized"] = "none_pending_fresh_independent_rtl_stage_closing_review"
    public["blockers"] = [{
        "id": "fresh_independent_rtl_stage_closing_review_pending",
        "stage": "rtl",
        "status": "active",
        "reason": "The prior standalone RTL review is historical and stage_closing=false; one distinct fresh independent L2 stage-closing review is required.",
        "required_resolution": "Run the maintained DPRF RTL review runner once and require stage_closing=true with all three RTL checklist items true.",
        "evidence": review_path,
    }]
    public.setdefault("integrity", {})["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)

    trace_path = ROOT / "design/RTL_TRACEABILITY.md"
    trace = trace_path.read_text(encoding="utf-8")
    old = (
        "- The three RTL checklist fields are true for the standalone candidate; independent\n"
        "  review remains pending. No focused quality, shell, PPA, or downstream claim exists.\n"
    )
    new = (
        "- A prior independent standalone review accepted the three RTL checklist fields, but\n"
        "  its `stage_closing=false` verdict is historical. One distinct fresh independent L2\n"
        "  stage-closing review remains pending. No focused quality, shell, PPA, or downstream\n"
        "  claim exists.\n"
    )
    require(old in trace or new in trace, "RTL traceability pending-review text is not bindable")
    trace_path.write_text(trace.replace(old, new), encoding="utf-8")

    checkpoint = f"""# Goal

Close the RTL checklist for `{CONTRACT}` without entering verification or PPA.

# Current State

Candidate `{packet['candidate_id']}` remains at Manager-owned stage `rtl`. The exact
source aggregate is `{packet['candidate_rtl_hash']}`. The earlier independent
standalone verdict is preserved as historical because it has `stage_closing=false`.
One distinct fresh independent L2 stage-closing review is pending.

# Verified Done

- The hash-bound PRECHECK reports all three RTL checklist items true and records no downstream run.
- The prior standalone review binds the same candidate and live RTL source hashes.
- The maintained review path now requires a distinct reviewer identity and emits stage closing only when all three RTL gates pass.

# Open Questions / Blockers

- A fresh Reviewer must issue the stage-closing verdict. This Engineer turn did not run or impersonate that review.

# Relevant Files and Evidence

- `{PACKET.relative_to(ROOT).as_posix()}`
- `{prior_archive.relative_to(ROOT).as_posix()}`
- `tools/run_down_projection_residual_fusion_rtl_review.py`
"""
    CHECKPOINT.write_text(checkpoint, encoding="utf-8")
    print(f"ACE2_DPRF_RTL_STAGE_CLOSING_REVIEW_READY candidate={packet['candidate_id']}")


def update_state(payload: dict[str, Any]) -> None:
    status = payload["reviewer_status"]
    manifest = load(MANIFEST)
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "independent_rtl_checklist_accepted" if status == "done" else "independent_rtl_checklist_repairs_required",
        "scope": "rtl_stage_close_for_verification_entry_only",
        "stage_closing": status == "done",
        "evidence": OUT.relative_to(ROOT).as_posix(),
        "reviewer_status": status,
    }
    if status == "done":
        manifest["candidate_status"] = "rtl_stage_closing_review_accepted_waiting_manager_advance"
        manifest["traceability"]["architecture_contract_gap"] = {
            "status": "closed_by_hash_bound_successor_rtl_and_fresh_stage_closing_review",
            "resolution_owner": "independent_l2_reviewer",
        }
        manifest["independent_reviewer_acceptance"] = {
            "architecture_review": "accepted",
            "environment_review": "accepted_manager_advanced_to_rtl",
            "rtl_review": "accepted_exact_candidate_hash_stage_closing",
            "status": "rtl_stage_closing_review_accepted_waiting_manager_advance",
        }
    else:
        manifest["candidate_status"] = "rtl_review_repairs_required"
        manifest["traceability"]["stage_checklist"] = {name: False for name in CHECKLIST}
    dump(MANIFEST, manifest)

    trace_path = ROOT / "design/RTL_TRACEABILITY.md"
    trace = trace_path.read_text(encoding="utf-8")
    pending = (
        "- A prior independent standalone review accepted the three RTL checklist fields, but\n"
        "  its `stage_closing=false` verdict is historical. One distinct fresh independent L2\n"
        "  stage-closing review remains pending. No focused quality, shell, PPA, or downstream\n"
        "  claim exists.\n"
    )
    accepted = (
        "- A fresh independent L2 review accepted all three RTL checklist fields for the exact\n"
        "  successor aggregate with `stage_closing=true`. The Manager-owned stage remains `rtl`;\n"
        "  no focused quality, shell, PPA, or downstream claim exists.\n"
    )
    require(pending in trace or accepted in trace, "RTL traceability review state is not bindable")
    trace_path.write_text(trace.replace(pending, accepted if status == "done" else pending), encoding="utf-8")

    public = load(PUBLIC)
    if status == "done":
        public["latest_decision"] = "down_projection_residual_fusion_rtl_checklist_independently_accepted"
        restore_public_probe_binding(public)
        public["stage"]["current_stage_status"] = "rtl_stage_closing_review_accepted_waiting_manager_advance"
        public["stage"]["stage_closing"] = False
        public["stage"]["current_stage_evidence"] = [
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            PACKET.relative_to(ROOT).as_posix(),
            OUT.relative_to(ROOT).as_posix(),
        ]
        public["blockers"] = [
            {
                "id": "manager_owned_downstream_stage_lock",
                "stage": "rtl",
                "status": "active",
                "reason": "A fresh independent L2 stage-closing review accepted all three RTL checklist items; verification and PPA remain locked until a Manager transition.",
                "required_resolution": "Manager may advance rtl to verification for the frozen focused discriminator.",
                "evidence": OUT.relative_to(ROOT).as_posix(),
            }
        ]
        for name in ("dashboard_fields", "implementation_frontier"):
            public[name]["latest_decision"] = public["latest_decision"]
            public[name]["routing_status"] = "manager_rtl_to_verification_transition_pending"
    else:
        public["latest_decision"] = "down_projection_residual_fusion_rtl_review_repairs_required"
        public["stage"]["current_stage_status"] = "rtl_review_repairs_required"
        public["blockers"] = [
            {
                "id": "independent_rtl_review_repairs_required",
                "stage": "rtl",
                "status": "active",
                "reason": payload.get("reason", "review did not pass"),
                "required_resolution": "Repair and rerun bounded RTL preflight/review.",
                "evidence": OUT.relative_to(ROOT).as_posix(),
            }
        ]
    public["generated_at_utc"] = payload["review_bound_at_utc"]
    packet = load(PACKET)
    projection = candidate_projection(packet, status, artifact(OUT))
    public["latest_rtl_candidate"] = copy.deepcopy(projection)
    for name in ("dashboard_fields", "implementation_frontier"):
        public[name]["latest_rtl_candidate"] = copy.deepcopy(projection)
    public["routing_status"] = (
        "manager_rtl_to_verification_transition_pending"
        if status == "done" else "rtl_review_repairs_required"
    )
    public["routing_authorized"] = (
        "manager_advance_rtl_to_verification_pending"
        if status == "done" else "none"
    )
    public.setdefault("integrity", {})["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)

    checkpoint = f"""# Goal

Implement and independently inspect the bounded RTL for `{CONTRACT}` without entering downstream stages.

# Current State

The Manager-owned stage remains `rtl`. Candidate `{payload['candidate_id']}` has fresh independent reviewer status `{status}` for the three RTL checklist items. The reviewer verdict has `stage_closing={str(status == 'done').lower()}`; only Manager may advance the project stage.

# Reviewer Reason

{payload.get('reason', '')}

# Boundaries

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains `layer_0.rope_q`; the 2.0 mm2 cap and 100 MHz floor are unchanged. Focused quality, shell regression, SKY130 PPA, prototype, benchmark, signoff, tapeout, and silicon were not run.
"""
    CHECKPOINT.write_text(checkpoint, encoding="utf-8")


def main() -> None:
    if "--prepare-stage-closing" in sys.argv[1:]:
        prepare_stage_closing_review()
        return
    if "--check-ready" in sys.argv[1:]:
        packet, _verified, prior = preflight()
        manifest = load(MANIFEST)
        public = load(PUBLIC)
        require(
            manifest.get("independent_reviewer_acceptance", {}).get("status")
            == "pending_fresh_independent_rtl_stage_closing_review",
            "manifest fresh stage-closing review status differs",
        )
        require(
            manifest.get("candidate_review_binding", {}).get("reviewer_status") == "pending",
            "manifest reviewer status is not pending",
        )
        expected_id = packet["candidate_id"]
        for label, projection in (
            ("latest_rtl_candidate", public.get("latest_rtl_candidate", {})),
            ("dashboard_fields.latest_rtl_candidate", public.get("dashboard_fields", {}).get("latest_rtl_candidate", {})),
            ("implementation_frontier.latest_rtl_candidate", public.get("implementation_frontier", {}).get("latest_rtl_candidate", {})),
        ):
            require(projection.get("candidate_id") == expected_id, f"{label} candidate differs")
            require(
                projection.get("status") == "fresh_independent_rtl_stage_closing_review_pending",
                f"{label} review status differs",
            )
        require(
            canonical_sha256(public) == public.get("integrity", {}).get("canonical_sha256"),
            "public canonical hash mismatch",
        )
        print(
            "ACE2_DPRF_RTL_STAGE_CLOSING_READY_CHECK_PASS "
            f"candidate={packet['candidate_id']} prior_thread={prior['reviewer_thread_id']}"
        )
        return
    if "--rebind-existing" in sys.argv[1:]:
        payload = load(OUT)
        require(payload.get("stage_closing") is True, "existing review is not stage-closing")
        require(
            payload.get("freshness", {}).get("distinct_reviewer_execution") is True,
            "existing review is not a distinct fresh execution",
        )
        update_state(payload)
        print(f"ACE2_DPRF_RTL_REVIEW_REBIND_PASS candidate={payload['candidate_id']}")
        return
    if "--check" in sys.argv[1:]:
        payload = load(OUT)
        packet = load(PACKET)
        manifest = load(MANIFEST)
        public = load(PUBLIC)
        require(payload.get("reviewer_status") == "done", "reviewer status is not done")
        require(
            payload.get("integrity", {}).get("canonical_sha256") == canonical_sha256(payload),
            "review decision canonical hash mismatch",
        )
        require(payload.get("contract_id") == CONTRACT, "review contract mismatch")
        require(payload.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
                "review candidate hash mismatch")
        require(payload.get("live_rtl_source_sha256") == sha256_file(RTL),
                "review live RTL hash mismatch")
        require(payload.get("checklist") == CHECKLIST, "review checklist incomplete")
        require(payload.get("stage_closing") is True, "review is not stage-closing")
        reviewed_precheck = payload.get("reviewed_precheck", {})
        verify_record(reviewed_precheck, "reviewed_precheck")
        transition_record = payload.get("reviewed_transition_binding", {})
        transition_path = ROOT / str(transition_record.get("path") or "")
        verify_record(transition_record, "reviewed_transition_binding")
        transition_summary = validate_artifact_tree(
            load(transition_path),
            label=transition_path.relative_to(ROOT).as_posix(),
        )
        require(transition_summary.current_records > 0, "transition binding has no live records")
        source_hashes = payload.get("live_source_hashes", {})
        require(isinstance(source_hashes, dict) and source_hashes, "review live source hashes missing")
        for relative, expected in source_hashes.items():
            require(sha256_file(ROOT / relative) == expected, f"review-bound source changed: {relative}")
        validate_artifact_tree(payload, label=OUT.relative_to(ROOT).as_posix())
        require(manifest.get("candidate_review_binding", {}).get("reviewer_status") == "done",
                "manifest review binding is not done")
        require(manifest.get("candidate_review_binding", {}).get("stage_closing") is True,
                "manifest review binding is not stage-closing")
        require(public.get("stage", {}).get("current_stage") == "rtl",
                "public stage is not rtl")
        require(canonical_sha256(public) == public.get("integrity", {}).get("canonical_sha256"),
                "public canonical hash mismatch")
        print(f"ACE2_DPRF_RTL_REVIEW_CHECK_PASS candidate={payload['candidate_id']}")
        return
    packet, verified, prior = preflight()
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
        "Independently inspect the exact hash-bound RTL preflight for "
        "shared_down_projection_residual_fusion_v1 and decide whether all three RTL "
        "checklist items support stage-closing acceptance for Manager-controlled verification entry."
    )
    instruction = (
        "Read design/ARCHITECTURE.md, design/NUMERICAL_REPLACEMENT_PROPOSAL.md, "
        "design/RTL_MANIFEST.json, design/RTL_TRACEABILITY.md, "
        "rtl/ace2_down_projection_residual_fusion_core.sv, "
        "tools/ace2_down_projection_residual_fusion_reference.py, and "
        "evidence/shared_down_projection_residual_fusion_v1/latest/PRECHECK.json directly. "
        "Inspect signedness, widths, complete assignments, reset/clear/backpressure, exact "
        "single-round arithmetic, asymmetric saturation thresholds, the 10-cycle schedule, "
        "interfaces, generated metadata provenance, and regeneration commands. Use only short "
        "read-only hash/content checks and the retained logs/XML. Do not modify any file except "
        "CHECKPOINT.md and do not rerun generation, simulation, quality, synthesis, STA, or PPA. "
        f"Return done only if all three gates hold for candidate hash {packet['candidate_rtl_hash']}. "
        "This must be a distinct fresh review from the archived standalone verdict. Return done "
        "only if all three gates hold; that done verdict is stage_closing=true for RTL checklist "
        "closure only. Do not run or authorize verification, quality, shell, PPA, or downstream work."
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
            "Planner preflight verified the canonical packet hash, aggregate source hash, "
            "all retained logs/XML, generated metadata size/identity, the Manager-owned RTL "
            "stage, and the absence of downstream runs. Reviewer must inspect independently."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="rtl_stage_close_for_verification_entry_only",
        checkpoint_path=str(CHECKPOINT),
        escalate_hint="If done, leave the project in rtl with downstream stages locked.",
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="workspace-write",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    raw = asdict(review)
    status = raw.get("status", "continue")
    prior_thread = prior.get("reviewer_thread_id")
    prior_fingerprint = prior.get("raw_decision", {}).get("static_fingerprint")
    current_thread = raw.get("thread_id", "")
    current_fingerprint = raw.get("static_fingerprint")
    fresh = bool(current_thread) and current_thread != prior_thread
    if prior_fingerprint and current_fingerprint:
        fresh = fresh and current_fingerprint != prior_fingerprint
    if status == "done" and not fresh:
        status = "continue"
        raw["reason"] = "Reviewer execution is not fresh relative to the archived standalone verdict."
    payload = {
        "schema_version": 1,
        "review_bound_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": status,
        "contract_id": CONTRACT,
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "live_rtl_source_sha256": sha256_file(RTL),
        "scope": "rtl_stage_close_for_verification_entry_only",
        "stage_closing": status == "done",
        "checklist": CHECKLIST if status == "done" else {name: False for name in CHECKLIST},
        "reason": raw.get("reason", ""),
        "required_repairs": [] if status == "done" else [raw.get("reason", "review did not pass")],
        "candidate_capability_accepted": False,
        "quality_discriminator_complete": False,
        "shell_admitted": False,
        "ppa_run": False,
        "reviewer_thread_id": raw.get("thread_id", ""),
        "reviewed_precheck": artifact(PACKET),
        "reviewed_transition_binding": packet["transition_binding"],
        "prior_standalone_decision": {
            **verified["prior_standalone_review"],
            "binding_scope": "sealed_whole_file",
            "content_validation": "whole_file_only",
        },
        "live_source_hashes": packet["source_hashes"],
        "control_source_hashes": packet["control_source_hashes"],
        "reviewed_artifact_bindings": {
            "logs": packet["logs"],
            "interface_xml": packet["interface_xml"],
            "metadata": packet["metadata"],
            "rtl": packet["evidence"]["rtl"],
            "reference": packet["evidence"]["reference"],
            "architecture_hook": packet["evidence"]["architecture_hook"],
        },
        "freshness": {
            "prior_reviewer_thread_id": prior_thread,
            "current_reviewer_thread_id": current_thread,
            "distinct_reviewer_execution": fresh,
        },
        "raw_decision": raw,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    payload["reviewed_precheck"].update(
        {"binding_scope": "sealed_whole_file", "content_validation": "whole_file_only"}
    )
    payload["reviewed_transition_binding"].update(
        {"binding_scope": "immutable_snapshot", "content_validation": "recursive_json"}
    )
    payload["integrity"]["canonical_sha256"] = canonical_sha256(payload)
    dump(OUT, payload)
    update_state(payload)
    print(f"ACE2_DPRF_RTL_REVIEW_RESULT status={status} candidate={packet['candidate_id']}")


if __name__ == "__main__":
    main()
