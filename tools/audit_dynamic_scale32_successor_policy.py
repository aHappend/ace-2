#!/usr/bin/env python3
"""Read-only/capability audit for the Dynamic Scale32 successor policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline"
POLICY_DIR = BASELINE / "recovery/successor_policy_v1"
DEFAULT_OUTPUT = POLICY_DIR
PREDECESSOR_RUN_ID = "dynamic-scale32-baseline-official-v1"
PREDECESSOR_LEDGER = BASELINE / "state/RUN_LEDGER.json"
SUCCESSOR_LEDGER = BASELINE / "state/RUN_LEDGER.successor-v2.json"
PREDECESSOR_STATE = BASELINE / "state" / PREDECESSOR_RUN_ID
PREDECESSOR_TERMINAL = PREDECESSOR_STATE / "terminal.bundle/TERMINAL.json"
V1_AUTHORITY = BASELINE / "EXECUTION_AUTHORITY.json"
V1_AUTHORITY_COMPANION = BASELINE / "EXECUTION_AUTHORITY.sha256"
V1_MANAGER_ISSUANCE = BASELINE / "MANAGER_ISSUANCE.json"
V1_ARCHIVE = POLICY_DIR / "v1_authority_archive/EXECUTION_AUTHORITY.json"
V1_ARCHIVE_COMPANION = POLICY_DIR / "v1_authority_archive/EXECUTION_AUTHORITY.sha256"
LEGACY_ROOT_SUCCESSOR_AUTHORITY = BASELINE / "SUCCESSOR_EXECUTION_AUTHORITY.json"
LEGACY_ROOT_SUCCESSOR_COMPANION = BASELINE / "SUCCESSOR_EXECUTION_AUTHORITY.sha256"
SUCCESSOR_BUNDLE = BASELINE / "recovery/successor_authority_generation_1_v1"
FAILED_PUBLICATION_ARCHIVE = BASELINE / "recovery/failed_publication_successor_authority_generation_1_v1"
FAILED_PUBLICATION_RECORD = FAILED_PUBLICATION_ARCHIVE / "FAILED_PUBLICATION.json"
PUBLISHED_RUNS = BASELINE / "runs"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
RUNNER = ROOT / "tools/run_dynamic_scale32_verification_baseline.py"
EXECUTOR = ROOT / "tools/ace2_dynamic_scale32_baseline.py"
HELPER = ROOT / "tools/ace2_software_identity.py"
TEST = ROOT / "tools/test_dynamic_scale32_successor_policy.py"
AUDITOR = Path(__file__).resolve()

EXPECTED = {
    "predecessor_ledger": "4945a5cb78661bb2289428d5baa6e0c3b0d72bd20842c1dca6810710a928464f",
    "predecessor_terminal": "742cd3c2af41773534dd8b3010657ea5595bda0d421a098f7d0ec0e63c890ffe",
    "v1_authority": "194ec47d40615891d433bdf9dfacf8982db346e36ff5ceebde55f5079e041c75",
    "v1_authority_companion": "e8a6c8bbd11a5bafc0da273650fdf0b22198f035df165f386c3bfa30ec1f8646",
    "v1_manager_issuance": "29b05ed86cb5e19d805283548b581c17e89cf37f8ad8aed8908e58af489344c1",
    "successor_v2_ledger": "b343e3f1acb45e912d620f45155b42752319dc90da3e5c503e53c2ec784f629c",
    "executor": "f950d6e9b8973a9fda5839e2471f4fc3aac38eba71afce6ab38970b9ce4e4620",
    "software_identity_helper": "516825a3bf8229495408328ab339373f8ebefdfdca079930886b82d57be46b94",
}


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


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {relative(path)}")
    return value


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


def verify_pair(path: Path) -> None:
    companion = path.with_suffix(".sha256")
    fields = companion.read_text(encoding="ascii").strip().split()
    if fields != [sha256_file(path), path.name]:
        raise RuntimeError(f"companion mismatch: {relative(path)}")


def tree_manifest(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    payload = {"root": relative(root), "files": files}
    payload["manifest_sha256"] = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return payload


def active_processes() -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    markers = ("run_dynamic_scale32_verification_baseline.py", "ace2_dynamic_scale32_baseline.py")
    proc = Path("/proc")
    if not proc.is_dir():
        return active
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            arguments = [
                item.decode("utf-8", "replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except OSError:
            continue
        matched = sorted(
            marker
            for marker in markers
            if any(Path(argument).name == marker for argument in arguments)
        )
        if matched:
            active.append({"pid": int(entry.name), "matched_entrypoints": matched})
    return active


def run_tests() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(TEST)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=120,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )
    passed = completed.returncode == 0 and "Ran 23 tests" in completed.stdout and "OK" in completed.stdout
    return {
        "command": [".venv/bin/python", relative(TEST)],
        "count": 23,
        "status": "pass" if passed else "fail",
        "returncode": completed.returncode,
        "output_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
    }


def filesystem_capability() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ace2-successor-audit-") as temporary:
        root = Path(temporary)
        staging = root / "staging"
        final = root / "final"
        staging.mkdir()
        (staging / "probe").write_bytes(b"successor-policy-capability\n")
        os.replace(staging, final)
        return {
            "atomic_directory_replace": final.is_dir() and not staging.exists(),
            "temporary_only": True,
            "probe_sha256": sha256_file(final / "probe"),
        }


def current_observations() -> dict[str, Any]:
    source_hashes = {
        "runner": sha256_file(RUNNER),
        "executor": sha256_file(EXECUTOR),
        "software_identity_helper": sha256_file(HELPER),
    }
    source_and_test_hashes = {
        **source_hashes,
        "successor_policy_test": sha256_file(TEST),
        "post_policy_auditor": sha256_file(AUDITOR),
    }
    ledger = load_json(SUCCESSOR_LEDGER)
    run_ids = sorted(ledger.get("runs", {}))
    state_dirs = sorted(
        path.name
        for path in (BASELINE / "state").iterdir()
        if path.is_dir() and path.name.startswith("dynamic-scale32-baseline-")
    )
    published = sorted(path.name for path in PUBLISHED_RUNS.iterdir()) if PUBLISHED_RUNS.is_dir() else []
    recovery_root = BASELINE / "recovery"
    staging = sorted(
        path.name
        for path in recovery_root.iterdir()
        if path.is_dir() and path.name.startswith(".staging-successor-authority-")
    )
    failed_publication = load_json(FAILED_PUBLICATION_RECORD)
    return {
        "source_hashes": source_hashes,
        "source_and_test_hashes": source_and_test_hashes,
        "successor_ledger": {
            "path": relative(SUCCESSOR_LEDGER),
            "sha256": sha256_file(SUCCESSOR_LEDGER),
            "schema_version": ledger.get("schema_version"),
            "kind": ledger.get("kind"),
            "run_ids": run_ids,
            "slot": ledger.get("recovery_generation_slots"),
            "successor_reservations": ledger.get("successor_reservations"),
            "cumulative_baseline_process_attempts": ledger.get("cumulative_baseline_process_attempts"),
            "cumulative_model_counts": ledger.get("cumulative_model_counts"),
            "publication": ledger.get("publication"),
            "migration_binding_sha256": ledger.get("migration_binding_sha256"),
        },
        "state_dirs": state_dirs,
        "published_runs": published,
        "successor_bundle_exists": SUCCESSOR_BUNDLE.exists(),
        "legacy_root_successor_authority_exists": LEGACY_ROOT_SUCCESSOR_AUTHORITY.exists()
        or LEGACY_ROOT_SUCCESSOR_COMPANION.exists(),
        "staging_directories": staging,
        "failed_publication": {
            "record": {
                "path": relative(FAILED_PUBLICATION_RECORD),
                "sha256": sha256_file(FAILED_PUBLICATION_RECORD),
            },
            "status": failed_publication.get("status"),
            "authority": failed_publication.get("authority"),
            "execution": failed_publication.get("execution"),
            "files": failed_publication.get("files"),
        },
        "active_processes": active_processes(),
        "v1_tree": tree_manifest(PREDECESSOR_STATE),
        "pipeline_state_sha256": sha256_file(PIPELINE_STATE),
    }


def gate_results(observations: dict[str, Any], tests: dict[str, Any]) -> dict[str, bool]:
    ledger = observations["successor_ledger"]
    return {
        "sealed_predecessor_ledger_hash": sha256_file(PREDECESSOR_LEDGER) == EXPECTED["predecessor_ledger"],
        "sealed_predecessor_terminal_hash": sha256_file(PREDECESSOR_TERMINAL) == EXPECTED["predecessor_terminal"],
        "sealed_v1_authority_hash": sha256_file(V1_AUTHORITY) == EXPECTED["v1_authority"],
        "sealed_v1_authority_companion_hash": sha256_file(V1_AUTHORITY_COMPANION) == EXPECTED["v1_authority_companion"],
        "sealed_v1_manager_issuance_hash": sha256_file(V1_MANAGER_ISSUANCE) == EXPECTED["v1_manager_issuance"],
        "v1_archive_byte_identity": V1_AUTHORITY.read_bytes() == V1_ARCHIVE.read_bytes()
        and V1_AUTHORITY_COMPANION.read_bytes() == V1_ARCHIVE_COMPANION.read_bytes(),
        "changed_runner_bound": ledger["migration_binding_sha256"] is not None
        and ledger["schema_version"] == 2
        and ledger["kind"] == "dynamic_scale32_baseline_successor_ledger_v2"
        and ledger["sha256"] == EXPECTED["successor_v2_ledger"],
        "executor_hash": observations["source_hashes"]["executor"] == EXPECTED["executor"],
        "helper_hash": observations["source_hashes"]["software_identity_helper"]
        == EXPECTED["software_identity_helper"],
        "only_sealed_v1_record": ledger["run_ids"] == [PREDECESSOR_RUN_ID],
        "one_available_generation_1_slot": ledger["slot"]
        == [
            {
                "recovery_generation": 1,
                "status": "available",
                "consumed_by_run_id": None,
                "reserved_at_utc": None,
            }
        ],
        "counts_unchanged": ledger["cumulative_baseline_process_attempts"] == 1
        and ledger["cumulative_model_counts"] == {"baseline_model": 1, "candidate_model": 0}
        and ledger["successor_reservations"] == 0,
        "publication_closed": ledger["publication"]
        == {"successor_slot_reopened": False, "baseline_l2_accepted": False},
        "no_final_successor_bundle": observations["successor_bundle_exists"] is False,
        "no_legacy_root_successor_authority": observations["legacy_root_successor_authority_exists"] is False,
        "no_stale_successor_staging": observations["staging_directories"] == [],
        "failed_publication_archived_non_authoritative": observations["failed_publication"]["status"]
        == "archived_non_authoritative_failed_publication"
        and observations["failed_publication"]["authority"] is False
        and observations["failed_publication"]["execution"] is False
        and observations["failed_publication"]["files"]
        == [
            {
                "bytes": 3953,
                "name": "MANAGER_ISSUANCE.json",
                "sha256": "0fe3e9c3766a1480f4e28cc2e3487e1553cbe345fc5b8cfc86c2c56bc5660951",
            },
            {
                "bytes": 3837,
                "name": "SUCCESSOR_EXECUTION_AUTHORITY.json",
                "sha256": "69076c942ce47041ae856996c95c8af28abc9f18c6492806228c1a52fa2af559",
            },
            {
                "bytes": 101,
                "name": "SUCCESSOR_EXECUTION_AUTHORITY.sha256",
                "sha256": "04b9cb026723d8224722f67796db1f6d57c9c729e070be86eba3d76e0ff69d12",
            },
        ],
        "no_successor_state": observations["state_dirs"] == [PREDECESSOR_RUN_ID],
        "no_published_successor": observations["published_runs"] == [],
        "no_active_execution": observations["active_processes"] == [],
        "adversarial_tests": tests["status"] == "pass" and tests["count"] == 23,
    }


def generate(output_dir: Path) -> int:
    before_v1 = tree_manifest(PREDECESSOR_STATE)
    before_pipeline = sha256_file(PIPELINE_STATE)
    tests = run_tests()
    capability = filesystem_capability()
    observations = current_observations()
    gates = gate_results(observations, tests)
    after_v1 = tree_manifest(PREDECESSOR_STATE)
    after_pipeline = sha256_file(PIPELINE_STATE)
    gates["v1_tree_unchanged_during_audit"] = before_v1 == after_v1
    gates["pipeline_unchanged_during_audit"] = before_pipeline == after_pipeline
    gates["synthetic_atomic_filesystem"] = capability["atomic_directory_replace"] is True
    status = "pass" if all(gates.values()) else "fail"
    created = now_utc()
    audit = {
        "schema_version": 1,
        "kind": "dynamic_scale32_successor_post_policy_environment_audit_v1",
        "contract_id": "shared_token_group_dynamic_scale32_v1",
        "created_at_utc": created,
        "status": status,
        "authority": False,
        "execution": False,
        "stage_closing": False,
        "claim_boundary": "Capability-only/read-only successor-policy audit with synthetic temporary tests; no runner preflight/run/recover/verify, executor, model/data load or download, authority, reservation, run, candidate task, PIPELINE_STATE edit, or stage transition.",
        "source_hashes": observations["source_hashes"],
        "source_and_test_hashes": observations["source_and_test_hashes"],
        "observed_counts": {"baseline_model": 1, "candidate_model": 0},
        "no_execution_state": {
            "authority_created": False,
            "reservation_created": False,
            "run_created": False,
            "candidate_task_created": False,
            "process_invoked": False,
            "model_or_data_loaded_or_downloaded": False,
        },
        "immutable_v1_tree_proof": after_v1,
        "migration": observations["successor_ledger"],
        "successor_authority_bundle_contract": {
            "final_directory": relative(SUCCESSOR_BUNDLE),
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
        "failed_publication_archive": observations["failed_publication"],
        "capability_probes": {
            "filesystem": capability,
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "platform": platform.platform(),
            },
        },
        "tests": tests,
        "gates": gates,
        "independent_review": {
            "required": True,
            "status": "pending_fresh_reviewer",
            "mode": "read_only",
            "stage_closing": False,
        },
    }
    audit_record = write_pair(output_dir, "POST_POLICY_ENVIRONMENT_AUDIT.json", audit)
    reproduce = {
        "schema_version": 1,
        "kind": "dynamic_scale32_successor_policy_reproduction_v1",
        "contract_id": "shared_token_group_dynamic_scale32_v1",
        "created_at_utc": created,
        "audit": audit_record,
        "command": [
            ".venv/bin/python",
            relative(AUDITOR),
            "--verify-only",
            "--output-dir",
            relative(output_dir),
        ],
        "test_command": [".venv/bin/python", relative(TEST)],
        "read_only": True,
        "stage_closing": False,
    }
    reproduce_record = write_pair(output_dir, "REPRODUCE.json", reproduce)
    handoff = {
        "schema_version": 1,
        "kind": "dynamic_scale32_successor_policy_migration_handoff_v1",
        "contract_id": "shared_token_group_dynamic_scale32_v1",
        "created_at_utc": created,
        "status": "implementation_migration_audit_complete_pending_independent_review" if status == "pass" else "blocked",
        "authority": False,
        "execution": False,
        "stage_closing": False,
        "policy_audit": audit_record,
        "reproduce": reproduce_record,
        "successor_ledger": observations["successor_ledger"],
        "source_and_test_hashes": observations["source_and_test_hashes"],
        "immutable_v1_tree_proof": after_v1,
        "observed_counts": {"baseline_model": 1, "candidate_model": 0},
        "no_execution_state": audit["no_execution_state"],
        "review": {
            "required": True,
            "status": "pending_fresh_reviewer",
            "mode": "read_only",
            "stage_closing": False,
        },
    }
    handoff_record = write_pair(output_dir, "POLICY_MIGRATION_HANDOFF.json", handoff)
    print(
        "DYNAMIC_SCALE32_SUCCESSOR_POLICY_AUDIT_"
        f"{status.upper()} audit_sha256={audit_record['sha256']} handoff_sha256={handoff_record['sha256']}"
    )
    return 0 if status == "pass" else 2


def verify(output_dir: Path) -> int:
    audit_path = output_dir / "POST_POLICY_ENVIRONMENT_AUDIT.json"
    reproduce_path = output_dir / "REPRODUCE.json"
    handoff_path = output_dir / "POLICY_MIGRATION_HANDOFF.json"
    for path in (audit_path, reproduce_path, handoff_path):
        verify_pair(path)
    tests = run_tests()
    observations = current_observations()
    gates = gate_results(observations, tests)
    audit = load_json(audit_path)
    handoff = load_json(handoff_path)
    gates["audit_status"] = audit.get("status") == "pass"
    gates["audit_source_hashes"] = audit.get("source_and_test_hashes") == observations["source_and_test_hashes"]
    gates["audit_ledger_hash"] = audit.get("migration", {}).get("sha256") == observations["successor_ledger"]["sha256"]
    gates["audit_v1_tree"] = audit.get("immutable_v1_tree_proof") == observations["v1_tree"]
    gates["handoff_audit_binding"] = handoff.get("policy_audit", {}).get("sha256") == sha256_file(audit_path)
    gates["handoff_counts"] = handoff.get("observed_counts") == {"baseline_model": 1, "candidate_model": 0}
    gates["handoff_scope"] = handoff.get("authority") is False and handoff.get("execution") is False and handoff.get("stage_closing") is False
    passed = all(gates.values())
    print(
        "DYNAMIC_SCALE32_SUCCESSOR_POLICY_VERIFY_"
        f"{'PASS' if passed else 'FAIL'} audit_sha256={sha256_file(audit_path)}"
    )
    if not passed:
        print(json.dumps({key: value for key, value in gates.items() if not value}, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return verify(output_dir) if args.verify_only else generate(output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
