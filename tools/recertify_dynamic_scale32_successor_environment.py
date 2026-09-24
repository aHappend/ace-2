#!/usr/bin/env python3
"""Compose a fresh successor-aware Dynamic Scale32 environment recertification.

The original 35-gate recertifier remains sealed to the version-identity recovery
handoff.  A later policy migration changed only the runner.  This tool reruns
the original capability probes, requires that their sole failure is that
historical runner binding, and replaces exactly that gate with the current
hash-bound successor-policy migration pending fresh independent review.

No runner preflight/run/recover/verify action or executor is invoked.  No model,
dataset, authority, reservation, run, candidate task, or pipeline transition is
created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline"
DEFAULT_OUTPUT = BASELINE / "environment_recertification_successor_v2"
FRESH_BASELINE_SUBDIR = "fresh_baseline_probe"

BASELINE_RECERTIFIER = ROOT / "tools/recertify_dynamic_scale32_baseline_environment.py"
POLICY_AUDITOR = ROOT / "tools/audit_dynamic_scale32_successor_policy.py"
RUNNER = ROOT / "tools/run_dynamic_scale32_verification_baseline.py"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
SUCCESSOR_AUTHORITY_BUNDLE = BASELINE / "recovery/successor_authority_generation_1_v1"
LEGACY_ROOT_SUCCESSOR_AUTHORITY = BASELINE / "SUCCESSOR_EXECUTION_AUTHORITY.json"
LEGACY_ROOT_SUCCESSOR_COMPANION = BASELINE / "SUCCESSOR_EXECUTION_AUTHORITY.sha256"
INDEPENDENT_POLICY_DECISION = ROOT / "evidence/review/verification_baseline_successor_policy_scale32_v1/decision.json"
PUBLISHED_RUNS = BASELINE / "runs"
V1_STATE = BASELINE / "state/dynamic-scale32-baseline-official-v1"

ARTIFACTS = {
    "accepted_recovery_handoff": (
        BASELINE / "recovery/version_identity_remediation_v1/RECOVERY_HANDOFF.json",
        "ca69abebe6880b6034d8413018b87b84e43db4e13faa43a78ec630b8de5eb7a2",
    ),
    "accepted_environment_audit": (
        BASELINE / "environment_recertification_v1/ENVIRONMENT_AUDIT.json",
        "4c3050acec5142018e7815257ce589fefe3caf4cd327d9df5aab7d299e179eb0",
    ),
    "accepted_environment_reproduce": (
        BASELINE / "environment_recertification_v1/REPRODUCE.json",
        "4179106c36cd638e58b58a7938dac51e8cd2dc413b152e625a2895fbd77ab4a8",
    ),
    "manager_successor_decision": (
        BASELINE / "recovery/successor_manager_decision_v1/SUCCESSOR_DECISION.json",
        "e6abadc30f1ad97012e012b7eb8a843d46ae390c1ea397d538d8677f2686846b",
    ),
    "policy_migration_handoff": (
        BASELINE / "recovery/successor_policy_v1/POLICY_MIGRATION_HANDOFF.json",
        "a1df0fc4ad85b2c1125009e9d915f7c6746aa0562b90886612bd363be35a1f33",
    ),
    "post_policy_environment_audit": (
        BASELINE / "recovery/successor_policy_v1/POST_POLICY_ENVIRONMENT_AUDIT.json",
        "e8c47c3966e16e3ecf9e5efdb84442cc2ed376ffcac24da4aeffbf63da60cf5d",
    ),
    "successor_v2_ledger": (
        BASELINE / "state/RUN_LEDGER.successor-v2.json",
        "b343e3f1acb45e912d620f45155b42752319dc90da3e5c503e53c2ec784f629c",
    ),
    "sealed_v1_ledger": (
        BASELINE / "state/RUN_LEDGER.json",
        "4945a5cb78661bb2289428d5baa6e0c3b0d72bd20842c1dca6810710a928464f",
    ),
    "sealed_v1_terminal": (
        V1_STATE / "terminal.bundle/TERMINAL.json",
        "742cd3c2af41773534dd8b3010657ea5595bda0d421a098f7d0ec0e63c890ffe",
    ),
    "recovery_reentry_attestation": (
        BASELINE / "recovery/rtl_verification_reentry_attestation_v1/RECOVERY_REENTRY_ATTESTATION.json",
        "559adeb4db582afd966691fe2a06a786aea1c813e06d5c47ffc64df9a7785bc9",
    ),
}

EXPECTED_RUNNER_SHA256 = "410c5bcebf7edc34f55802e0796620ea068fbac6b692ad2c4a3fca873ee7eb65"
HISTORICAL_RUNNER_SHA256 = "10fc7749aa0540d37d2e01e1cde738db46a47bc41310d68ec74b6350c3c78a24"
SOURCE_GATE = "current_executor_runner_helper_test_hashes"


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def path_label(path: Path) -> str:
    try:
        return relative(path)
    except ValueError:
        return path.resolve().as_posix()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {relative(path)}")
    return value


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"


def canonical_bytes(value: dict[str, Any]) -> bytes:
    payload = dict(value)
    payload.pop("integrity", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def add_integrity(value: dict[str, Any]) -> None:
    value["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": sha256_bytes(canonical_bytes(value)),
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write_pair(directory: Path, name: str, value: dict[str, Any]) -> dict[str, Any]:
    content = json_bytes(value)
    path = directory / name
    digest = sha256_bytes(content)
    atomic_write(path, content)
    atomic_write(path.with_suffix(".sha256"), f"{digest}  {path.name}\n".encode("ascii"))
    return {"path": relative(path), "bytes": len(content), "sha256": digest}


def verify_pair(path: Path) -> str:
    fields = path.with_suffix(".sha256").read_text(encoding="ascii").strip().split()
    digest = sha256_file(path)
    if fields != [digest, path.name]:
        raise RuntimeError(f"companion mismatch: {relative(path)}")
    return digest


def expected_review_bindings() -> dict[str, str]:
    return {
        "post_policy_environment_audit_sha256": ARTIFACTS["post_policy_environment_audit"][1],
        "runner_sha256": EXPECTED_RUNNER_SHA256,
    }


def independent_review_state(decision_path: Path = INDEPENDENT_POLICY_DECISION) -> dict[str, Any]:
    companion_path = decision_path.with_suffix(".sha256")
    detail: dict[str, Any] = {
        "required": True,
        "decision_path": path_label(decision_path),
        "expected_bindings": expected_review_bindings(),
        "present": decision_path.is_file(),
        "companion_present": companion_path.is_file(),
    }
    if not detail["present"] and not detail["companion_present"]:
        return {**detail, "status": "pending_fresh_reviewer"}

    decision: dict[str, Any] = {}
    decision_sha256: str | None = None
    companion_valid = False
    if detail["present"] and detail["companion_present"]:
        try:
            decision_sha256 = verify_pair(decision_path)
            decision = load_json(decision_path)
            companion_valid = True
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass
    gates = {
        "decision_and_companion_present": detail["present"] and detail["companion_present"],
        "decision_companion_valid": companion_valid,
        "review_contract": decision.get("contract_id") == "shared_token_group_dynamic_scale32_v1",
        "reviewer_completed": decision.get("reviewer_status") == "done",
        "review_accepted": decision.get("status") == "accepted" and decision.get("verdict") == "go",
        "review_independent": decision.get("independent_reviewer") is True,
        "review_does_not_close_stage": decision.get("stage_closing") is False,
        "exact_top_level_audit_and_runner_bindings": decision.get("bindings")
        == expected_review_bindings(),
    }
    accepted = all(gates.values())
    return {
        **detail,
        "status": "accepted" if accepted else "invalid",
        "decision_sha256": decision_sha256,
        "gates": gates,
        "accepted": accepted,
    }


def immutable_policy_evidence(policy_and_review: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in policy_and_review.items()
        if key != "independent_review"
    }


def valid_review_transition(
    sealed: dict[str, Any],
    live: dict[str, Any],
    decision_path: Path = INDEPENDENT_POLICY_DECISION,
) -> bool:
    expected_path = path_label(decision_path)
    common_contract = (
        sealed.get("required") is True
        and live.get("required") is True
        and sealed.get("decision_path") == expected_path
        and live.get("decision_path") == expected_path
        and sealed.get("expected_bindings") == expected_review_bindings()
        and live.get("expected_bindings") == expected_review_bindings()
    )
    sealed_pending = (
        sealed.get("present") is False
        and sealed.get("companion_present") is False
        and sealed.get("status") == "pending_fresh_reviewer"
    )
    live_pending = (
        live.get("present") is False
        and live.get("companion_present") is False
        and live.get("status") == "pending_fresh_reviewer"
    )
    live_accepted = (
        live.get("present") is True
        and live.get("companion_present") is True
        and live.get("status") == "accepted"
        and live.get("accepted") is True
        and all(live.get("gates", {}).values())
    )
    return common_contract and sealed_pending and (live_pending or live_accepted)


def tree_manifest(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "path": relative(root)}
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    payload = {"exists": True, "path": relative(root), "files": files}
    payload["manifest_sha256"] = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return payload


def protected_snapshot() -> dict[str, Any]:
    return {
        "pipeline_state_sha256": sha256_file(PIPELINE_STATE),
        "sealed_v1_ledger_sha256": sha256_file(ARTIFACTS["sealed_v1_ledger"][0]),
        "sealed_v1_terminal_sha256": sha256_file(ARTIFACTS["sealed_v1_terminal"][0]),
        "successor_v2_ledger_sha256": sha256_file(ARTIFACTS["successor_v2_ledger"][0]),
        "sealed_v1_state_tree": tree_manifest(V1_STATE),
        "published_runs_tree": tree_manifest(PUBLISHED_RUNS),
        "successor_authority_bundle_exists": SUCCESSOR_AUTHORITY_BUNDLE.exists(),
        "legacy_root_successor_authority_exists": LEGACY_ROOT_SUCCESSOR_AUTHORITY.exists()
        or LEGACY_ROOT_SUCCESSOR_COMPANION.exists(),
    }


def artifact_bindings() -> tuple[dict[str, Any], bool]:
    records: dict[str, Any] = {}
    passed = True
    for name, (path, expected) in sorted(ARTIFACTS.items()):
        observed = sha256_file(path)
        companion = path.with_suffix(".sha256")
        companion_valid = True
        if companion.exists():
            try:
                companion_valid = verify_pair(path) == expected
            except RuntimeError:
                companion_valid = False
        match = observed == expected and companion_valid
        records[name] = {
            "path": relative(path),
            "sha256": observed,
            "expected_sha256": expected,
            "companion_checked": companion.exists(),
            "gate": match,
        }
        passed = passed and match
    runner_sha = sha256_file(RUNNER)
    records["runner"] = {
        "path": relative(RUNNER),
        "sha256": runner_sha,
        "expected_sha256": EXPECTED_RUNNER_SHA256,
        "companion_checked": False,
        "gate": runner_sha == EXPECTED_RUNNER_SHA256,
    }
    return records, passed and runner_sha == EXPECTED_RUNNER_SHA256


def run_checked(command: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"},
    )


def run_fresh_baseline_probe(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    completed = run_checked(
        [sys.executable, relative(BASELINE_RECERTIFIER), "--output-dir", str(output_dir)],
        timeout=600,
    )
    audit_path = output_dir / "ENVIRONMENT_AUDIT.json"
    reproduce_path = output_dir / "REPRODUCE.json"
    if completed.returncode != 0 or not audit_path.exists() or not reproduce_path.exists():
        raise RuntimeError(f"baseline recertifier failed: rc={completed.returncode} output={completed.stdout[-2000:]}")
    verify_pair(audit_path)
    verify_pair(reproduce_path)
    audit = load_json(audit_path)
    gates = audit.get("required_gates", {})
    source_identity = audit.get("source_identity", {})
    mismatches = sorted(
        role for role, record in source_identity.items() if record.get("matches_accepted_recovery") is not True
    )
    summary = {
        "required_gate_count": len(gates),
        "passing_gate_count": sum(value is True for value in gates.values()),
        "failed_required_gate_ids": sorted(key for key, value in gates.items() if value is not True),
        "source_mismatch_roles": mismatches,
        "runner": source_identity.get("runner"),
        "no_execution_counts": audit.get("no_execution_counts"),
        "policy": audit.get("policy"),
        "status": audit.get("status"),
    }
    return audit, summary


def runner_only_historical_mismatch(summary: dict[str, Any]) -> bool:
    runner = summary.get("runner") or {}
    counts = summary.get("no_execution_counts") or {}
    return (
        summary.get("required_gate_count") == 35
        and summary.get("passing_gate_count") == 34
        and summary.get("failed_required_gate_ids") == [SOURCE_GATE]
        and summary.get("source_mismatch_roles") == ["runner"]
        and runner.get("accepted_recovery_sha256") == HISTORICAL_RUNNER_SHA256
        and runner.get("sha256") == EXPECTED_RUNNER_SHA256
        and counts.get("before") == counts.get("after")
        and counts.get("runner_or_executor_invoked") is False
        and counts.get("model_or_data_loaded") is False
        and counts.get("model_or_data_payload_downloaded") is False
    )


def compose_required_gates(baseline_gates: dict[str, bool], replacement_gate: bool) -> dict[str, bool]:
    if len(baseline_gates) != 35 or SOURCE_GATE not in baseline_gates:
        raise RuntimeError("baseline gate set is not the exact 35-gate contract")
    composed = dict(baseline_gates)
    composed[SOURCE_GATE] = replacement_gate
    return composed


def policy_and_review_gates(bindings: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    policy_dir = BASELINE / "recovery/successor_policy_v1"
    completed = run_checked(
        [sys.executable, relative(POLICY_AUDITOR), "--verify-only", "--output-dir", relative(policy_dir)],
        timeout=180,
    )
    policy_audit = load_json(ARTIFACTS["post_policy_environment_audit"][0])
    handoff = load_json(ARTIFACTS["policy_migration_handoff"][0])
    ledger = load_json(ARTIFACTS["successor_v2_ledger"][0])
    bundle_contract = policy_audit.get("successor_authority_bundle_contract", {})
    no_execution = policy_audit.get("no_execution_state", {})
    review_state = independent_review_state()
    gates = {
        "all_cited_artifact_hashes": all(record["gate"] for record in bindings.values()),
        "policy_verify_only_pass": completed.returncode == 0
        and "DYNAMIC_SCALE32_SUCCESSOR_POLICY_VERIFY_PASS" in completed.stdout,
        "post_policy_audit_pass": policy_audit.get("status") == "pass"
        and all(policy_audit.get("gates", {}).values()),
        "policy_runner_binding": policy_audit.get("source_hashes", {}).get("runner")
        == EXPECTED_RUNNER_SHA256
        and handoff.get("source_and_test_hashes", {}).get("runner") == EXPECTED_RUNNER_SHA256,
        "policy_ledger_binding": handoff.get("successor_ledger", {}).get("sha256")
        == ARTIFACTS["successor_v2_ledger"][1]
        and policy_audit.get("migration", {}).get("sha256") == ARTIFACTS["successor_v2_ledger"][1],
        "policy_audit_binding": handoff.get("policy_audit", {}).get("sha256")
        == ARTIFACTS["post_policy_environment_audit"][1],
        "canonical_directory_atomic_bundle_contract": bundle_contract
        == {
            "final_directory": "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/recovery/successor_authority_generation_1_v1",
            "files": [
                "MANAGER_ISSUANCE.json",
                "SUCCESSOR_EXECUTION_AUTHORITY.json",
                "SUCCESSOR_EXECUTION_AUTHORITY.sha256",
            ],
            "publication": "fsync_complete_sibling_staging_then_single_directory_rename",
            "root_or_split_bundle_accepted": False,
            "preflight_reserves_or_consumes_slot": False,
            "reservation_requires_revalidated_final_bundle": True,
            "reservation_revalidation_lock": "successor_ledger_lock",
            "reservation_revalidation_boundary": "immediately_before_slot_consumption_and_reservation_publish",
            "reservation_revalidation_comparison": "complete_three_file_manifest_authority_and_bytes",
        },
        "exact_no_execution_invariants": no_execution.get("authority_created") is False
        and no_execution.get("reservation_created") is False
        and no_execution.get("run_created") is False
        and no_execution.get("candidate_task_created") is False
        and no_execution.get("process_invoked") is False
        and no_execution.get("model_or_data_loaded_or_downloaded") is False,
        "live_ledger_invariants": ledger.get("cumulative_baseline_process_attempts") == 1
        and ledger.get("cumulative_model_counts") == {"baseline_model": 1, "candidate_model": 0}
        and ledger.get("successor_reservations") == 0
        and sorted(ledger.get("runs", {})) == ["dynamic-scale32-baseline-official-v1"]
        and ledger.get("recovery_generation_slots")
        == [{
            "recovery_generation": 1,
            "status": "available",
            "consumed_by_run_id": None,
            "reserved_at_utc": None,
        }],
        "independent_review_absent_or_exactly_bound_accepted": review_state.get("status")
        in {"pending_fresh_reviewer", "accepted"},
    }
    detail = {
        "policy_verify_command": [
            ".venv/bin/python",
            relative(POLICY_AUDITOR),
            "--verify-only",
            "--output-dir",
            relative(policy_dir),
        ],
        "policy_verify_returncode": completed.returncode,
        "policy_verify_output_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "independent_review": review_state,
    }
    return gates, detail


def recovery_state_gates() -> tuple[dict[str, bool], dict[str, Any]]:
    state = load_json(PIPELINE_STATE)
    transitions = state.get("stage_history", [])
    penultimate = transitions[-2] if len(transitions) >= 2 else {}
    final = transitions[-1] if transitions else {}
    gates = {
        "current_stage_verification": state.get("current_stage") == "verification",
        "legal_environment_to_rtl_recovery_step": penultimate.get("by") == "manager"
        and penultimate.get("from_stage") == "environment"
        and penultimate.get("to_stage") == "rtl"
        and penultimate.get("at") == "2026-08-02T12:40:52.671561Z",
        "legal_rtl_to_verification_recovery_step": final.get("by") == "manager"
        and final.get("from_stage") == "rtl"
        and final.get("to_stage") == "verification"
        and final.get("at") == "2026-08-02T12:45:39.777407Z"
        and final.get("evidence", {}).get("sha256") == ARTIFACTS["recovery_reentry_attestation"][1],
        "verification_not_claimed_complete": state.get("stages", {}).get("verification", {}).get("status")
        != "done",
        "no_successor_authority_bundle": SUCCESSOR_AUTHORITY_BUNDLE.exists() is False,
        "no_legacy_root_successor_authority": LEGACY_ROOT_SUCCESSOR_AUTHORITY.exists() is False
        and LEGACY_ROOT_SUCCESSOR_COMPANION.exists() is False,
        "no_published_successor_run": not PUBLISHED_RUNS.exists()
        or not any(path.is_dir() for path in PUBLISHED_RUNS.iterdir()),
    }
    detail = {
        "pipeline_state": {"path": relative(PIPELINE_STATE), "sha256": sha256_file(PIPELINE_STATE)},
        "penultimate_transition": penultimate,
        "final_transition": final,
    }
    return gates, detail


def collect(fresh_output: Path) -> dict[str, Any]:
    before = protected_snapshot()
    bindings, bindings_gate = artifact_bindings()
    accepted_audit = load_json(ARTIFACTS["accepted_environment_audit"][0])
    fresh_audit, fresh_summary = run_fresh_baseline_probe(fresh_output)
    policy_gates, policy_detail = policy_and_review_gates(bindings)
    state_gates, state_detail = recovery_state_gates()
    after = protected_snapshot()

    accepted_gate_map = accepted_audit.get("required_gates", {})
    accepted_gate = len(accepted_gate_map) == 35 and all(accepted_gate_map.values())
    fresh_runner_only = runner_only_historical_mismatch(fresh_summary)
    replacement_gate = bindings_gate and fresh_runner_only and all(policy_gates.values())
    composed = compose_required_gates(fresh_audit.get("required_gates", {}), replacement_gate)
    meta_gates = {
        "accepted_original_35_gate_packet": accepted_gate,
        "fresh_original_probe_is_34_of_35_runner_only": fresh_runner_only,
        "successor_policy_replacement_gate": replacement_gate,
        "protected_state_unchanged_during_recertification": before == after,
        **state_gates,
    }
    return {
        "artifact_bindings": bindings,
        "composed_required_gates": composed,
        "fresh_baseline_probe": fresh_summary,
        "meta_gates": meta_gates,
        "policy_and_review": {"gates": policy_gates, **policy_detail},
        "protected_state": {"before": before, "after": after, "unchanged": before == after},
        "recovery_state": state_detail,
    }


def generate(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    persisted_probe = output_dir / FRESH_BASELINE_SUBDIR
    if persisted_probe.exists():
        shutil.rmtree(persisted_probe)
    with tempfile.TemporaryDirectory(prefix="ace2-successor-env-generate-") as temporary:
        result = collect(Path(temporary) / FRESH_BASELINE_SUBDIR)
    required_gates = result["composed_required_gates"]
    all_pass = (
        len(required_gates) == 35
        and all(required_gates.values())
        and all(result["meta_gates"].values())
        and all(result["policy_and_review"]["gates"].values())
    )
    audit = {
        "schema_version": 1,
        "kind": "dynamic_scale32_successor_aware_environment_recertification_v2",
        "contract_id": "shared_token_group_dynamic_scale32_v1",
        "created_at_utc": now_utc(),
        "status": "all_35_successor_aware_gates_passed_pending_independent_review" if all_pass else "blocked",
        "authority": False,
        "execution": False,
        "stage_closing": False,
        "claim_boundary": "Read-only successor-aware environment recertification only. The historical recovery runner binding is replaced solely by the independently accepted successor-policy migration. No authority, reservation, run, candidate, model/data action, pipeline transition, verification completion, PPA, benchmark, signoff, tapeout, or silicon claim is made.",
        "required_gate_count": len(required_gates),
        "passing_gate_count": sum(value is True for value in required_gates.values()),
        "required_gates": required_gates,
        "required_gates_all_pass": all_pass,
        "replacement": {
            "gate_id": SOURCE_GATE,
            "historical_runner_sha256": HISTORICAL_RUNNER_SHA256,
            "successor_runner_sha256": EXPECTED_RUNNER_SHA256,
            "basis": "hash-bound directory-atomic successor-policy migration pending fresh independent Reviewer verdict",
        },
        **result,
        "review": {"required": True, "status": "pending_fresh_reviewer", "stage_closing": False},
    }
    add_integrity(audit)
    audit_record = write_pair(output_dir, "SUCCESSOR_ENVIRONMENT_AUDIT.json", audit)
    reproduce = {
        "schema_version": 1,
        "kind": "dynamic_scale32_successor_aware_environment_reproduction_v2",
        "contract_id": "shared_token_group_dynamic_scale32_v1",
        "created_at_utc": now_utc(),
        "audit": audit_record,
        "command": [
            ".venv/bin/python",
            relative(Path(__file__).resolve()),
            "--verify-existing",
            "--output-dir",
            relative(output_dir),
        ],
        "read_only": True,
        "authority": False,
        "execution": False,
        "stage_closing": False,
    }
    add_integrity(reproduce)
    reproduce_record = write_pair(output_dir, "REPRODUCE.json", reproduce)
    print(
        "DYNAMIC_SCALE32_SUCCESSOR_ENVIRONMENT_RECERTIFICATION_"
        f"{'PASS' if all_pass else 'FAIL'} audit_sha256={audit_record['sha256']} "
        f"reproduce_sha256={reproduce_record['sha256']}"
    )
    return 0 if all_pass else 2


def verify_existing(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    audit_path = output_dir / "SUCCESSOR_ENVIRONMENT_AUDIT.json"
    reproduce_path = output_dir / "REPRODUCE.json"
    audit_sha = verify_pair(audit_path)
    reproduce_sha = verify_pair(reproduce_path)
    audit = load_json(audit_path)
    reproduce = load_json(reproduce_path)
    if audit.get("integrity", {}).get("canonical_sha256") != sha256_bytes(canonical_bytes(audit)):
        raise RuntimeError("audit canonical integrity mismatch")
    if reproduce.get("integrity", {}).get("canonical_sha256") != sha256_bytes(canonical_bytes(reproduce)):
        raise RuntimeError("reproduce canonical integrity mismatch")
    if reproduce.get("audit", {}).get("sha256") != audit_sha:
        raise RuntimeError("reproduce artifact does not bind audit bytes")
    with tempfile.TemporaryDirectory(prefix="ace2-successor-env-review-") as temporary:
        live = collect(Path(temporary) / FRESH_BASELINE_SUBDIR)
    stable_fields = (
        "artifact_bindings",
        "composed_required_gates",
        "fresh_baseline_probe",
        "meta_gates",
        "recovery_state",
    )
    sealed_policy_and_review = audit.get("policy_and_review", {})
    live_policy_and_review = live.get("policy_and_review", {})
    gates = {
        "sealed_audit_status": audit.get("status")
        == "all_35_successor_aware_gates_passed_pending_independent_review",
        "sealed_audit_scope": audit.get("authority") is False
        and audit.get("execution") is False
        and audit.get("stage_closing") is False,
        "sealed_35_of_35": audit.get("required_gate_count") == 35
        and audit.get("passing_gate_count") == 35
        and audit.get("required_gates_all_pass") is True
        and all(audit.get("required_gates", {}).values()),
        "live_protected_state_unchanged": live.get("protected_state", {}).get("unchanged") is True,
        "live_policy_evidence": immutable_policy_evidence(sealed_policy_and_review)
        == immutable_policy_evidence(live_policy_and_review),
        "live_review_transition": valid_review_transition(
            sealed_policy_and_review.get("independent_review", {}),
            live_policy_and_review.get("independent_review", {}),
        ),
        **{f"live_{field}": audit.get(field) == live.get(field) for field in stable_fields},
    }
    passed = all(gates.values())
    print(
        "DYNAMIC_SCALE32_SUCCESSOR_ENVIRONMENT_VERIFY_"
        f"{'PASS' if passed else 'FAIL'} audit_sha256={audit_sha} reproduce_sha256={reproduce_sha}"
    )
    if not passed:
        print(json.dumps({key: value for key, value in gates.items() if not value}, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    return verify_existing(output_dir) if args.verify_existing else generate(output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
