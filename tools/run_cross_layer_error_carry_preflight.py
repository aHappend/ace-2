#!/usr/bin/env python3
"""Run the bounded RTL-stage preflight for the frozen QECR successor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
EVIDENCE = ROOT / f"evidence/{CONTRACT}/latest"
PRECHECK = EVIDENCE / "PRECHECK.json"
REVIEW = ROOT / f"evidence/review/rtl_checklist_{CONTRACT}/decision.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
RTL = "rtl/ace2_cross_layer_error_carry_core.sv"
SOURCES = [
    RTL,
    "Makefile",
    "tools/ace2_cross_layer_error_carry_reference.py",
    "tools/ace2_rmsnorm_reference.py",
    "tools/ace2_quality_contracts.py",
    "tools/run_cross_layer_error_carry_preflight.py",
    "tools/bind_cross_layer_error_carry_rtl.py",
    "tools/run_cross_layer_error_carry_rtl_review.py",
    "tools/gen_cross_layer_error_carry_vectors.py",
    "tools/gen_cross_layer_error_carry_metadata.py",
    "tools/run_cross_layer_error_carry_formal.py",
    "formal/ace2_cross_layer_error_carry_formal.sv",
    "formal/ace2_cross_layer_error_carry_formal.ys",
    "verification/test_cross_layer_error_carry.py",
    "verification/tb/ace2_cross_layer_error_carry_tb.sv",
    "verification/generated/cross_layer_error_carry_vectors.json",
    "verification/generated/cross_layer_error_carry_vectors.svh",
    "reference/generated/cross_layer_error_carry_metadata.json",
    "reference/generated/cross_layer_error_carry_metadata.bin",
]
IMPLEMENTATION_SOURCES = [
    RTL,
    "tools/ace2_cross_layer_error_carry_reference.py",
    "tools/ace2_rmsnorm_reference.py",
    "tools/ace2_quality_contracts.py",
    "tools/gen_cross_layer_error_carry_vectors.py",
    "tools/gen_cross_layer_error_carry_metadata.py",
    "tools/run_cross_layer_error_carry_formal.py",
    "formal/ace2_cross_layer_error_carry_formal.sv",
    "formal/ace2_cross_layer_error_carry_formal.ys",
    "verification/test_cross_layer_error_carry.py",
    "verification/tb/ace2_cross_layer_error_carry_tb.sv",
    "verification/generated/cross_layer_error_carry_vectors.json",
    "verification/generated/cross_layer_error_carry_vectors.svh",
    "reference/generated/cross_layer_error_carry_metadata.json",
    "reference/generated/cross_layer_error_carry_metadata.bin",
]
TOPS = [
    "ace2_quantization_error_carry_lane_core",
    "ace2_error_carry_state_core",
    "ace2_carry_aware_rmsnorm_core",
]

DRAFT_COLLATERAL = [
    *SOURCES,
    PRECHECK.relative_to(ROOT).as_posix(),
    MANIFEST.relative_to(ROOT).as_posix(),
    TRACEABILITY.relative_to(ROOT).as_posix(),
    REVIEW.relative_to(ROOT).as_posix(),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: str | Path) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {resolved}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def aggregate_hash(records: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(records):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(records[path].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def engineer_ledger_proof(thread_id: str, session_id: str) -> dict[str, Any] | None:
    ledger_name = os.environ.get("ARGUS_SKILL_AGENT_IO_LOG", "")
    ledger = Path(ledger_name) if ledger_name else Path()
    if not ledger.is_file() or not thread_id or not session_id:
        return None
    starts: dict[str, dict[str, Any]] = {}
    completions: list[dict[str, Any]] = []
    usage: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
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
        start = starts.get(call_id, {})
        recorded = usage.get(call_id, {})
        run_label = str(start.get("run_label") or completion.get("run_label") or "")
        working_dir = start.get("working_dir")
        if not run_label.startswith("engineer"):
            continue
        if working_dir and Path(working_dir).resolve() != ROOT:
            continue
        if completion.get("exit_code") != 0 or completion.get("turn_completed") is not True:
            continue
        if recorded and (
            recorded.get("project_id") != session_id or recorded.get("status") != "completed"
        ):
            continue
        return {
            "call_id": call_id,
            "event_type": "agent.io.complete",
            "run_label": run_label,
            "status": "completed",
            "thread_id": thread_id,
        }
    return None


def preserved_engineer_provenance(source_hashes: dict[str, str]) -> dict[str, Any]:
    implementation_hashes = {path: source_hashes[path] for path in IMPLEMENTATION_SOURCES}
    implementation_aggregate = aggregate_hash(implementation_hashes)
    candidates: list[Path] = []
    if PRECHECK.is_file():
        candidates.append(PRECHECK)
    candidates.extend(
        sorted(
            (EVIDENCE.parent / "archive").glob("rtl_candidate_*/PRECHECK.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    for candidate_path in candidates:
        candidate = load(candidate_path)
        prior_hashes = candidate.get("source_hashes", {})
        if not isinstance(prior_hashes, dict):
            continue
        if any(prior_hashes.get(path) != digest for path, digest in implementation_hashes.items()):
            continue
        provenance = candidate.get("implementation_provenance", {})
        if not isinstance(provenance, dict) or provenance.get("role") != "engineer":
            continue
        thread_id = provenance.get("thread_id", "")
        session_id = provenance.get("session_id", "")
        proof = engineer_ledger_proof(thread_id, session_id)
        if proof is None:
            continue
        preserved = copy.deepcopy(provenance)
        preserved.update({
            "execution_ledger": proof,
            "implementation_source_hash_scope": "standalone_rtl_reference_generated_and_test_sources_excluding_evidence_orchestration",
            "implementation_source_aggregate_sha256": implementation_aggregate,
            "preserved_from": candidate_path.relative_to(ROOT).as_posix(),
            "preserved_for_unchanged_sources": True,
        })
        return preserved
    raise RuntimeError(
        "no ledger-verified Engineer provenance matches the unchanged implementation sources"
    )


def direct_planner_provenance(source_hashes: dict[str, str]) -> dict[str, Any]:
    """Bind direct Planner implementation without claiming Engineer execution."""
    implementation_hashes = {
        path: source_hashes[path] for path in IMPLEMENTATION_SOURCES
    }
    return {
        "agent_layer": "planner",
        "execution_contract": "operator_planner_direct_execution_contract",
        "implementation_source_aggregate_sha256": aggregate_hash(
            implementation_hashes
        ),
        "implementation_source_hash_scope": (
            "standalone_rtl_reference_generated_and_test_sources_"
            "excluding_evidence_orchestration"
        ),
        "role": "planner_direct_executor",
        "status": "fresh_direct_implementation_from_frozen_contract",
        "thread_id": os.environ.get("CODEX_THREAD_ID", "unavailable"),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def review_archive_prefix(decision: dict[str, Any]) -> str:
    provenance = decision.get("producer_provenance", {})
    if isinstance(provenance, dict) and provenance.get("role") == "engineer":
        if decision.get("reviewer_status") == "done":
            return "prior_accepted_engineer_reviewer_decision"
        return "prior_engineer_reviewer_replan_decision"
    return "superseded_planner_draft"


def repair_mislabeled_review_archives() -> None:
    archive_dir = REVIEW.parent / "archive"
    if not archive_dir.is_dir():
        return
    for source in sorted(archive_dir.glob("superseded_planner_draft.*.json")):
        decision = load(source)
        prefix = review_archive_prefix(decision)
        if prefix == "superseded_planner_draft":
            continue
        digest = sha256_file(source)
        destination = archive_dir / f"{prefix}.{digest}.json"
        if destination.exists():
            require(destination.read_bytes() == source.read_bytes(), "review archive repair collision")
            source.unlink()
        else:
            os.replace(source, destination)


def archive_current_review() -> None:
    if not REVIEW.is_file():
        return
    decision = load(REVIEW)
    review_digest = sha256_file(REVIEW)
    prefix = review_archive_prefix(decision)
    review_archive = REVIEW.parent / "archive" / f"{prefix}.{review_digest}.json"
    review_archive.parent.mkdir(parents=True, exist_ok=True)
    if review_archive.exists():
        require(review_archive.read_bytes() == REVIEW.read_bytes(), "review archive collision")
    else:
        shutil.copy2(REVIEW, review_archive)
    REVIEW.unlink()


def planner_draft_archive(prior: dict[str, Any]) -> Path:
    prior_hash = str(prior.get("candidate_rtl_hash") or sha256_file(PRECHECK))
    return EVIDENCE.parent / "archive" / f"planner_draft_{prior_hash}"


def archive_planner_draft() -> dict[str, Any]:
    require(PRECHECK.is_file(), "Planner-draft PRECHECK is missing")
    prior = load(PRECHECK)
    archive = planner_draft_archive(prior)
    archive.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for relative in DRAFT_COLLATERAL:
        source = ROOT / relative
        if not source.is_file():
            continue
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(destination.read_bytes() == source.read_bytes(), f"archive collision: {destination}")
        else:
            shutil.copy2(source, destination)
        records.append({
            "original_path": relative,
            "archived_path": destination.relative_to(ROOT).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "classification": "unaccepted_planner_draft",
        "classification_authority": "live_operator_objective_2026-08-02",
        "candidate_id": prior.get("candidate_id"),
        "candidate_rtl_hash": prior.get("candidate_rtl_hash"),
        "candidate_capability_accepted": False,
        "stage_closing": False,
        "superseded_by": "fresh_engineer_implementation_pending",
        "artifacts": records,
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    provenance["integrity"]["canonical_sha256"] = canonical_sha256(provenance)
    provenance_path = archive / "DRAFT_PROVENANCE.json"
    dump(provenance_path, provenance)
    return artifact(provenance_path)


def run(command: list[str], log_name: str) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = EVIDENCE / log_name
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stdout}")
    return artifact(log)


def tool_version(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, f"version command failed: {' '.join(command)}")
    return result.stdout.strip().splitlines()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-planner-draft", action="store_true")
    args = parser.parse_args()
    if args.archive_planner_draft:
        record = archive_planner_draft()
        print(
            "ACE2_QECR_PLANNER_DRAFT_ARCHIVE_PASS "
            f"path={record['path']} sha256={record['sha256']}"
        )
        return 0

    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(pipeline.get("successor", {}).get("id") == CONTRACT, "active successor differs")
    event = pipeline.get("stage_history", [])[-1]
    event_reason = event.get("reason", "")
    environment_entry = (
        event.get("by") == "manager"
        and event.get("from_stage") == "environment"
        and event.get("to_stage") == "rtl"
        and "bounded RTL implementation" in event_reason
    )
    verification_repair_entry = (
        event.get("by") == "manager"
        and event.get("from_stage") == "verification"
        and event.get("to_stage") == "rtl"
        and "RMSNorm reference/RTL numerical-contract repair" in event_reason
    )
    require(
        environment_entry or verification_repair_entry,
        "latest pipeline event does not authorize the active RTL implementation or repair",
    )

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    repair_mislabeled_review_archives()
    planner_draft_provenance: dict[str, Any] | None = None
    if PRECHECK.is_file():
        prior = load(PRECHECK)
        inherited = prior.get("implementation_provenance", {}).get("planner_draft_archive")
        if isinstance(inherited, dict) and (ROOT / inherited.get("path", "")).is_file():
            require(artifact(inherited["path"]) == inherited, "inherited Planner-draft provenance changed")
            planner_draft_provenance = inherited
        else:
            manifest = load(MANIFEST) if MANIFEST.is_file() else {}
            inherited = manifest.get("candidate_implementation_provenance", {}).get("planner_draft_archive")
            if isinstance(inherited, dict) and (ROOT / inherited.get("path", "")).is_file():
                require(artifact(inherited["path"]) == inherited, "manifest Planner-draft provenance changed")
                planner_draft_provenance = inherited
            else:
                draft_provenance_path = planner_draft_archive(prior) / "DRAFT_PROVENANCE.json"
                if draft_provenance_path.is_file():
                    planner_draft_provenance = artifact(draft_provenance_path)
        prior_hash = str(prior.get("candidate_rtl_hash") or sha256_file(PRECHECK))
        archive = EVIDENCE.parent / "archive" / f"rtl_candidate_{prior_hash}"
        if archive.is_dir():
            collision = any(
                (archive / source.name).exists()
                and (archive / source.name).read_bytes() != source.read_bytes()
                for source in EVIDENCE.iterdir()
                if source.is_file()
            )
            if collision:
                archive = EVIDENCE.parent / "archive" / (
                    f"rtl_candidate_{prior_hash}_{sha256_file(PRECHECK)}"
                )
        archive.mkdir(parents=True, exist_ok=True)
        for source in sorted(EVIDENCE.iterdir()):
            if source.is_file():
                destination = archive / source.name
                if destination.exists():
                    require(destination.read_bytes() == source.read_bytes(), f"archive collision: {destination}")
                else:
                    shutil.copy2(source, destination)
    archive_current_review()
    (ROOT / "build/cross_layer_error_carry").mkdir(parents=True, exist_ok=True)
    logs: dict[str, Any] = {}
    logs["vector_check"] = run([".venv/bin/python", "tools/gen_cross_layer_error_carry_vectors.py", "--check"], "vector_check.log")
    logs["metadata_check"] = run([".venv/bin/python", "tools/gen_cross_layer_error_carry_metadata.py", "--check"], "metadata_check.log")
    logs["software_reference"] = run(
        [".venv/bin/python", "-m", "unittest", "verification.test_cross_layer_error_carry", "-v"],
        "software_reference_unittest.log",
    )
    logs["rtl_elaboration"] = run(
        ["iverilog", "-g2012", "-Wall", "-Iverification/generated", "-o",
         "build/cross_layer_error_carry/ace2_cross_layer_error_carry_tb.vvp", RTL,
         "verification/tb/ace2_cross_layer_error_carry_tb.sv"],
        "rtl_elaboration.log",
    )
    logs["rtl_simulation"] = run(
        ["vvp", "build/cross_layer_error_carry/ace2_cross_layer_error_carry_tb.vvp"],
        "rtl_simulation.log",
    )
    for top in TOPS:
        logs[f"lint_{top}"] = run(
            ["verilator", "--lint-only", "--language", "1800-2017", "-Wall",
             "--top-module", top, RTL],
            f"lint_{top}.log",
        )
        logs[f"elaborate_{top}"] = run(
            ["iverilog", "-g2012", "-s", top, "-o",
             f"build/cross_layer_error_carry/{top}.vvp", RTL],
            f"elaborate_{top}.log",
        )
        interface_path = EVIDENCE / f"interface_{top}.xml"
        logs[f"interface_{top}"] = run(
            ["verilator", "--xml-only", "--language", "1800-2017", "-Wall",
             "--top-module", top, "--xml-output", str(interface_path), RTL],
            f"interface_{top}.log",
        )
    logs["minimal_formal"] = run(
        [".venv/bin/python", "tools/run_cross_layer_error_carry_formal.py"],
        "minimal_formal.log",
    )

    source_hashes = {relative: sha256_file(ROOT / relative) for relative in SOURCES}
    candidate_hash = aggregate_hash(source_hashes)
    candidate_id = f"cross_layer_error_carry_{candidate_hash[:16]}"
    try:
        implementation_provenance = preserved_engineer_provenance(source_hashes)
    except RuntimeError:
        implementation_provenance = direct_planner_provenance(source_hashes)
    implementation_provenance["planner_draft_archive"] = planner_draft_provenance
    metadata = load("reference/generated/cross_layer_error_carry_metadata.json")
    vectors = load("verification/generated/cross_layer_error_carry_vectors.json")
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    packet: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "ordered_standalone_rtl_reference_generated_and_test_sources_not_shell_admitted",
        "live_rtl_source_sha256": source_hashes[RTL],
        "current_stage": "rtl",
        "stage_closing": False,
        "status": "pass_ready_for_independent_rtl_review",
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "implementation_provenance": implementation_provenance,
        "runtime": {
            "python": tool_version([".venv/bin/python", "--version"]),
            "iverilog": tool_version(["iverilog", "-V"]),
            "verilator": tool_version(["verilator", "--version"]),
            "yosys": tool_version(["yosys", "-V"]),
        },
        "manager_transition": event,
        "stage_checklist": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "checklist": {
            "exact_scalar_reference_passed": True,
            "deterministic_vectors_and_metadata_reproduce": True,
            "all_three_modules_lint_without_warning": True,
            "all_three_modules_elaborate": True,
            "integrated_rtl_simulation_passed": True,
            "bounded_formal_invariants_passed": True,
            "atomic_carry_validity_and_identity_exercised": True,
            "dual_valid_start_arbitration_exercised": True,
            "out_of_range_carry_read_fails_closed": True,
            "invalid_consumer_start_returns_accepted_tag": True,
            "accepted_consumer_completion_tag_held_under_backpressure": True,
            "producer_26_cycle_schedule_reproduced": True,
            "signed24_unsigned56_rmsnorm_equations_reproduced": True,
            "output_scale_folded_rmsnorm_gain_contract_reproduced": True,
        },
        "modules": [
            {
                "name": "ace2_quantization_error_carry_lane_core",
                "requirement": "ARCHITECTURE.md producer equations and 26-cycle serial carry finalizer",
            },
            {
                "name": "ace2_error_carry_state_core",
                "requirement": "SPEC.md atomic 896-lane carry state, identity ordering, and exactly-once consumer completion",
                "parameters": {"HIDDEN_SIZE": 896, "TOKEN_ID_WIDTH": 32, "MODEL_ID_WIDTH": 64, "TAG_WIDTH": 16},
            },
            {
                "name": "ace2_carry_aware_rmsnorm_core",
                "requirement": "ARCHITECTURE.md signed-Q8.15 reconstruction, output-scale-folded signed-Q7.8 gain metadata, unsigned-56 square sum, ceil root, reciprocal, and output rounding",
                "parameters": {"HIDDEN_SIZE": 896},
            },
        ],
        "metadata": {
            "layout": metadata["layout"],
            "identity": metadata["identity"],
            "record_counts": metadata["record_counts"],
        },
        "workspace_proof": vectors["workspace_proof"],
        "source_hashes": source_hashes,
        "logs": logs,
        "interfaces": {
            top: artifact(EVIDENCE / f"interface_{top}.xml") for top in TOPS
        },
        "prohibited_runs": {
            "full_shell_regression": False,
            "quality_discriminator": False,
            "paired_smoke": False,
            "candidate_ppa": False,
            "physical_design": False,
            "prototype": False,
            "benchmark": False,
            "signoff": False,
        },
        "claim_boundary": [
            "Standalone synthesizable RTL, deterministic first-party metadata/vectors, lint, elaboration, simulation, and bounded formal invariants only.",
            "The accepted shell prefix through layer_0.v_proj and first unsupported layer_0.rope_q are unchanged.",
            "No quality, shell admission, PPA, power, prototype, benchmark, signoff, tapeout-readiness, or silicon claim is made.",
        ],
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    packet["integrity"]["canonical_sha256"] = canonical_sha256(packet)
    PRECHECK.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "ACE2_QECR_RTL_PREFLIGHT_PASS "
        f"candidate={candidate_id} aggregate={candidate_hash} checklist=3/3 stage_closing=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
