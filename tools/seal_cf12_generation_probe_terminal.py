#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-generation-probe-cf12-attempt-0001"
)
STATE = PACKAGE.parent / f"{PACKAGE.name}-authority-state"
ATTEMPT = STATE / "attempt-0001"
PREDECESSOR = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-reference-process-predecessor-cf12-0001"
)
SEAL = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-generation-probe-cf12-terminal-seal-0001"
)
REPAIR = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-generation-probe-product-host-repair-cf13-0001"
)
TASK_ID = "ace2-r6-cf12-generation-probe-once-20260825-0001"
RUN_ID = f"{TASK_ID}-1787642540998248323"
REGISTRY = ROOT / ".argus_subagents" / f"{TASK_ID}.json"
REGISTRY_LOGS = ROOT / ".argus_subagents" / f"{TASK_ID}_logs"
EXIT_RECEIPT = REGISTRY_LOGS / f"exit_code.{RUN_ID}"
EXPECTED_ELAPSED_SECONDS = 7210.4
EXPECTED_PRODUCT_LAST_ORDINAL = 63099
EXPECTED_RTL_COMMANDS = 1306104

SOURCE_PATHS = {
    "authority": PACKAGE / "authority.json",
    "authority_runner": PACKAGE / "authority_runner.py",
    "bindings": PACKAGE / "bindings.json",
    "input": PACKAGE / "input.txt",
    "product_probe": PACKAGE / "product_probe.py",
    "authorization_consumed": STATE / "authorization-consumed.json",
    "execution_journal": ATTEMPT / "evidence-journal.json",
    "reference_artifact": ATTEMPT / "reference/independent-reference.json",
    "reference_process_receipt": ATTEMPT / "reference/process-receipt.json",
    "reference_stdout": ATTEMPT / "reference/stdout.bin",
    "reference_stderr": ATTEMPT / "reference/stderr.bin",
    "product_evidence": ATTEMPT / "product/product-rtl-evidence.json",
    "product_process_receipt": ATTEMPT / "product/process-receipt.json",
    "product_stdout": ATTEMPT / "product/stdout.bin",
    "product_stderr": ATTEMPT / "product/stderr.bin",
    "rtl_commands": ATTEMPT / "product/rtl-completion/commands.jsonl",
    "rtl_progress_journal": ATTEMPT / "product/rtl-completion/progress.journal",
    "rtl_progress": ATTEMPT / "product/rtl-completion/progress.json",
    "task_registry": REGISTRY,
    "task_stdout_log": REGISTRY_LOGS / "stdout.log",
    "task_stderr_log": REGISTRY_LOGS / "stderr.log",
    "task_exit_receipt": EXIT_RECEIPT,
    "seal_constructor": Path(__file__).resolve(),
}

OUTPUT_FILES = {
    "construction-report.json",
    "hash-inventory.json",
    "process-death-evidence.json",
    "repair-route.json",
    "review-request.json",
    "seal-manifest.json",
    "seal-manifest.sha256",
    "terminal-seal.json",
    "tests/test_cf12_terminal_seal.py",
    "validate_terminal_seal.py",
}


class SealError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SealError(f"JSON root is not an object: {path}")
    return value


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes(path: Path, raw: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)
    fsync_directory(path.parent)


def write_json(path: Path, value: object) -> None:
    write_bytes(path, canonical_bytes(value))


def load_validator() -> Any:
    path = PREDECESSOR / "receipt_validator.py"
    specification = importlib.util.spec_from_file_location(
        "_cf12_terminal_receipt_validator", path
    )
    if specification is None or specification.loader is None:
        raise SealError("cannot import the reviewed CF12 receipt validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def exact_registry_files() -> list[Path]:
    matches = sorted(
        path
        for path in (ROOT / ".argus_subagents").glob(f"{TASK_ID}*")
        if path == REGISTRY or path == REGISTRY_LOGS
    )
    if matches != [REGISTRY, REGISTRY_LOGS]:
        raise SealError("durable task registry cardinality is not exactly one")
    log_files = sorted(path.name for path in REGISTRY_LOGS.iterdir())
    expected = sorted(["stderr.log", "stdout.log", EXIT_RECEIPT.name])
    if log_files != expected:
        raise SealError("durable task log/exit receipt inventory differs")
    return [REGISTRY, *sorted(REGISTRY_LOGS.iterdir())]


def last_jsonl_object(path: Path) -> tuple[int, dict[str, Any]]:
    count = 0
    last: dict[str, Any] | None = None
    with path.open("rb") as handle:
        for raw in handle:
            count += 1
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise SealError(f"JSONL record is not an object: {path}:{count}")
            last = value
    if last is None:
        raise SealError(f"JSONL file is empty: {path}")
    return count, last


def validate_review_gate(authority: dict[str, Any]) -> dict[str, object]:
    gate = authority.get("activation_gate")
    if not isinstance(gate, dict):
        raise SealError("authority activation gate is absent")
    review_root = Path(str(gate.get("review_root", "")))
    latest_path = review_root / "latest.json"
    latest = load_object(latest_path)
    if latest.get("kind") != "handoff_ref":
        raise SealError("CF12 authority review index is not a handoff")
    handoff_path = Path(str(latest.get("handoff", {}).get("path", "")))
    handoff = load_object(handoff_path)
    if (
        handoff.get("mission_id") != gate.get("mission_id")
        or handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise SealError("CF12 launch authority lacks independent L2 acceptance")
    return {
        "mission_id": gate["mission_id"],
        "producer_role": "reviewer",
        "review_status": "done",
        "latest_sha256": sha256_file(latest_path),
        "handoff_sha256": sha256_file(handoff_path),
    }


def collect_terminal_facts() -> dict[str, Any]:
    for label, path in SOURCE_PATHS.items():
        if not path.is_file() or path.is_symlink():
            raise SealError(f"required {label} file is absent or linked: {path}")
    exact_registry_files()

    authority = load_object(SOURCE_PATHS["authority"])
    consumed = load_object(SOURCE_PATHS["authorization_consumed"])
    registry = load_object(REGISTRY)
    journal = load_object(SOURCE_PATHS["execution_journal"])
    reference = load_object(SOURCE_PATHS["reference_artifact"])
    reference_receipt = load_object(SOURCE_PATHS["reference_process_receipt"])
    product = load_object(SOURCE_PATHS["product_evidence"])
    product_receipt = load_object(SOURCE_PATHS["product_process_receipt"])
    progress = load_object(SOURCE_PATHS["rtl_progress"])

    if (
        authority.get("identity") != PACKAGE.name
        or authority.get("execution_limit") != 1
        or authority.get("durable_task_id") != TASK_ID
        or authority.get("authority_consumption") != "BEFORE_ANY_PROCESS"
        or authority.get("retry_replay_resume_relaunch") != "FORBIDDEN"
        or authority.get("stage1_state") != "OPEN"
        or authority.get("stage2") != "FORBIDDEN"
        or authority.get("stage_transition") != "DISABLED"
    ):
        raise SealError("CF12 authority contract differs")
    review_gate = validate_review_gate(authority)
    if (
        consumed.get("status") != "CONSUMED_BEFORE_REFERENCE_OR_PRODUCT_PROCESS"
        or consumed.get("authority_sha256") != sha256_file(SOURCE_PATHS["authority"])
        or consumed.get("identity") != authority["identity"]
        or consumed.get("durable_task_id") != TASK_ID
        or consumed.get("transaction_nonce") != authority["transaction_nonce"]
        or consumed.get("challenge_nonce") != authority["challenge_nonce"]
        or {
            consumed.get("retry"),
            consumed.get("replay"),
            consumed.get("resume"),
            consumed.get("relaunch"),
        }
        != {"PERMANENTLY_FORBIDDEN"}
    ):
        raise SealError("CF12 consumption record differs")
    if sorted(path.name for path in STATE.iterdir()) != [
        "attempt-0001",
        "authorization-consumed.json",
    ]:
        raise SealError("CF12 authority state has ambiguous attempt cardinality")
    if sorted(path.name for path in ATTEMPT.iterdir()) != [
        "evidence-journal.json",
        "product",
        "reference",
    ]:
        raise SealError("CF12 attempt namespace differs")

    launch = str(PACKAGE / "launch.sh")
    if (
        registry.get("task_id") != TASK_ID
        or registry.get("run_id") != RUN_ID
        or registry.get("command") != launch
        or registry.get("cwd") != str(ROOT)
        or registry.get("mode") != "direct"
        or registry.get("state") != "error"
        or registry.get("exit_code") != 2
        or registry.get("elapsed_seconds") != EXPECTED_ELAPSED_SECONDS
        or type(registry.get("pid")) is not int
        or type(registry.get("worker_pid")) is not int
        or EXIT_RECEIPT.read_bytes() != b"2\n"
    ):
        raise SealError("CF12 durable task terminal record differs")

    contract = load_object(PREDECESSOR / "contract.json")
    validator = load_validator()
    validator.validate_reference_receipt(
        reference_receipt,
        reference,
        SOURCE_PATHS["reference_artifact"],
        contract,
        {
            "source_artifact_sha256": sha256_file(SOURCE_PATHS["product_evidence"]),
            "producer": product.get("producer", {}),
        },
    )
    positions = reference.get("positions")
    if (
        not isinstance(positions, list)
        or len(positions) != 4
        or any(
            position.get("prompt_bytes_sha256") != authority["prompt"]["sha256"]
            or position.get("chat_template_token_ids_sha256")
            != authority["prompt"]["chat_template_token_ids_sha256"]
            for position in positions
        )
    ):
        raise SealError("reference prompt/template binding differs")

    if (
        product_receipt.get("schema")
        != "ace2-r6-cf12-product-process-receipt-v1"
        or product_receipt.get("pid") == reference_receipt["process"]["pid"]
        or product_receipt.get("fresh_separate_process") is not True
        or product_receipt.get("started_after_reference_receipt_fsync") is not True
        or product_receipt.get("exit_code") != 124
        or product_receipt.get("argv_sha256")
        != authority["product_invocation"]["argv_sha256"]
        or product_receipt.get("stdout_sha256")
        != sha256_file(SOURCE_PATHS["product_stdout"])
        or product_receipt.get("stderr_sha256")
        != sha256_file(SOURCE_PATHS["product_stderr"])
    ):
        raise SealError("product timeout receipt differs")
    if (
        product.get("status")
        != "HOST_PARTIAL_EVIDENCE_FSYNCED_ENDPOINT_STILL_REQUIRED"
        or product.get("positions") != []
        or product.get("rtl_positions") != []
        or product.get("decode")
        != "FORBIDDEN_UNTIL_AUTHORITY_COMBINES_FSYNCED_EVIDENCE"
        or product.get("failures")
        != [
            {
                "boundary": "PRODUCT_HOST_TRAJECTORY",
                "error_type": "TypeError",
                "positions_persisted": 0,
            }
        ]
    ):
        raise SealError("product partial-evidence frontier differs")
    if (
        journal.get("status") != "TERMINAL_PARTIAL_EVIDENCE_FSYNCED_NO_RETRY"
        or journal.get("reference") != "RECEIPT_FSYNCED"
        or journal.get("product") != "RECEIPT_FSYNCED"
        or journal.get("decode") != "NOT_STARTED"
        or journal.get("failures")
        != [
            {
                "boundary": "PRODUCT_RTL",
                "classification": "PRODUCT_OR_RTL_PROCESS_FAILED",
            }
        ]
    ):
        raise SealError("attempt terminal journal differs")
    if (ATTEMPT / "combined-predecode-evidence.json").exists():
        raise SealError("decode barrier unexpectedly exists")

    command_count, last_command = last_jsonl_object(SOURCE_PATHS["rtl_commands"])
    if (
        progress.get("status") != "RUNNING"
        or progress.get("total_commands") != EXPECTED_RTL_COMMANDS
        or progress.get("last_committed_ordinal") != EXPECTED_PRODUCT_LAST_ORDINAL
        or progress.get("next_ordinal") != EXPECTED_PRODUCT_LAST_ORDINAL + 1
        or progress.get("generated_token_ids") != []
        or progress.get("terminated") is not False
        or progress.get("resume_count") != 0
        or command_count != EXPECTED_PRODUCT_LAST_ORDINAL + 1
        or last_command.get("ordinal") != EXPECTED_PRODUCT_LAST_ORDINAL
        or last_command.get("generated_token_ids_after") != []
        or last_command.get("generated_token_after") is not None
        or last_command.get("terminated_after") is not False
    ):
        raise SealError("RTL committed frontier differs")

    known_pids = {
        "durable_worker": registry["worker_pid"],
        "durable_command": registry["pid"],
        "reference": reference_receipt["process"]["pid"],
        "product": product_receipt["pid"],
    }
    if len(set(known_pids.values())) != len(known_pids):
        raise SealError("recorded process lineage is not distinct")

    return {
        "authority": authority,
        "consumed": consumed,
        "registry": registry,
        "journal": journal,
        "reference": reference,
        "reference_receipt": reference_receipt,
        "product": product,
        "product_receipt": product_receipt,
        "progress": progress,
        "last_command": last_command,
        "rtl_command_count": command_count,
        "known_pids": known_pids,
        "review_gate": review_gate,
    }


def process_scan(known_pids: dict[str, int]) -> dict[str, Any]:
    pid_records = []
    for role, pid in known_pids.items():
        proc_path = Path("/proc") / str(pid)
        pid_records.append(
            {
                "role": role,
                "pid": pid,
                "proc_entry_exists": proc_path.exists(),
                "status": "LIVE" if proc_path.exists() else "DEAD",
            }
        )
    if any(item["proc_entry_exists"] for item in pid_records):
        raise SealError("a recorded CF12 process PID is still live")

    markers = (
        str(PACKAGE / "launch.sh"),
        str(PACKAGE / "product_probe.py"),
        str(ATTEMPT / "reference/independent-reference.json"),
        str(ATTEMPT / "product/product-rtl-evidence.json"),
        str(ATTEMPT / "product/rtl-completion"),
    )
    matches = []
    for proc in Path("/proc").glob("[0-9]*"):
        cmdline_path = proc / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if command and any(marker in command for marker in markers):
            matches.append({"pid": int(proc.name), "command_sha256": sha256_bytes(raw)})
    if matches:
        raise SealError("a process still references the bound CF12 execution namespace")

    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    boot_id = boot_id_path.read_bytes() if boot_id_path.is_file() else None
    return {
        "schema": "ace2-r6-cf12-process-death-evidence-v1",
        "status": "PASS_ALL_RECORDED_PIDS_DEAD_AND_NO_BOUND_COMMAND_LINE",
        "checked_at_unix_ns": time.time_ns(),
        "procfs_boot_id_sha256": (
            sha256_bytes(boot_id) if boot_id is not None else None
        ),
        "recorded_pids": pid_records,
        "bound_command_line_markers_sha256": [
            sha256_bytes(marker.encode("utf-8")) for marker in markers
        ],
        "matching_processes": [],
        "endpoint_pid": "NOT_RECORDED_IN_FSYNCED_EVIDENCE",
        "endpoint_process_conclusion": (
            "NO_LIVE_BOUND_COMMAND_LINE_AT_SEAL_TIME; INDIVIDUAL_ENDPOINT_PID_UNAVAILABLE"
        ),
    }


def hash_inventory() -> dict[str, Any]:
    records = {}
    for label, path in sorted(SOURCE_PATHS.items()):
        file_stat = path.stat()
        records[label] = {
            "path": relative(path),
            "size": file_stat.st_size,
            "mode": stat.S_IMODE(file_stat.st_mode),
            "sha256": sha256_file(path),
        }
    return {
        "schema": "ace2-r6-cf12-terminal-hash-inventory-v1",
        "status": "SEALED_SOURCE_BYTES",
        "hash_algorithm": "SHA-256",
        "records": records,
    }


def terminal_seal(
    facts: dict[str, Any],
    inventory: dict[str, Any],
    death_evidence: dict[str, Any],
) -> dict[str, Any]:
    authority = facts["authority"]
    registry = facts["registry"]
    progress = facts["progress"]
    reference_receipt = facts["reference_receipt"]
    product_receipt = facts["product_receipt"]
    return {
        "schema": "ace2-r6-cf12-terminal-seal-v1",
        "status": "SEALED_TERMINAL_PARTIAL_EVIDENCE_PENDING_INDEPENDENT_L2",
        "identity": PACKAGE.name,
        "sole_attempt": "attempt-0001",
        "authority": {
            "execution_limit": 1,
            "consumption": "AUTHENTICATED_CONSUMED_BEFORE_ANY_PROCESS",
            "authority_sha256": facts["consumed"]["authority_sha256"],
            "transaction_nonce_sha256": sha256_bytes(
                authority["transaction_nonce"].encode("ascii")
            ),
            "challenge_nonce_sha256": sha256_bytes(
                authority["challenge_nonce"].encode("ascii")
            ),
            "prior_independent_l2": facts["review_gate"],
            "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        },
        "launch_lineage": {
            "authenticated_start_count": 1,
            "durable_task_id": TASK_ID,
            "run_id": RUN_ID,
            "registry_state": registry["state"],
            "registry_exit_code": registry["exit_code"],
            "registry_elapsed_seconds": registry["elapsed_seconds"],
            "durable_worker_pid": registry["worker_pid"],
            "durable_command_pid": registry["pid"],
            "reference_pid": reference_receipt["process"]["pid"],
            "reference_exit_code": reference_receipt["process"]["exit_code"],
            "product_pid": product_receipt["pid"],
            "product_exit_code": product_receipt["exit_code"],
            "terminal": "NATURAL_AUTHORITY_RUNNER_EXIT_2_AFTER_PRODUCT_TIMEOUT_124",
        },
        "process_death_evidence_sha256": sha256_bytes(
            canonical_bytes(death_evidence)
        ),
        "hash_inventory_sha256": sha256_bytes(canonical_bytes(inventory)),
        "fsynced_frontier": {
            "authorization_consumption": "FSYNCED",
            "independent_reference": {
                "receipt": "FSYNCED",
                "status": "PASS_AUTHORITY_LAUNCHED_REFERENCE",
                "exit_code": 0,
                "position_count": len(facts["reference"]["positions"]),
                "full_logit_count": facts["reference"]["full_logit_count"],
            },
            "product": {
                "receipt": "FSYNCED",
                "exit_code": 124,
                "evidence": "HOST_PARTIAL_EVIDENCE_FSYNCED_ENDPOINT_STILL_REQUIRED",
                "host_positions": 0,
                "failure_type": "TypeError",
                "failure_message": "NOT_CAPTURED",
                "traceback": "NOT_CAPTURED",
            },
            "rtl": {
                "progress_snapshot_status": progress["status"],
                "last_committed_ordinal": progress["last_committed_ordinal"],
                "next_ordinal": progress["next_ordinal"],
                "total_commands": progress["total_commands"],
                "committed_command_count": facts["rtl_command_count"],
                "last_operator": facts["last_command"].get("operator"),
                "last_token_step": facts["last_command"].get("token_step"),
                "completed_logit_positions": 0,
                "generated_token_ids": [],
                "endpoint_receipt": "NOT_FSYNCED",
                "terminal_marker": "NOT_FSYNCED",
            },
            "combined_predecode_barrier": "NOT_CREATED",
            "decode": "NOT_STARTED",
        },
        "classification": {
            "status": "EARLIEST_SUPPORTED_FAILURE_NOT_SCIENTIFICALLY_LOCALIZED",
            "earliest_supported_boundary": "PRODUCT_HOST_TRAJECTORY_BEFORE_POSITION_0",
            "observed_failure": "TypeError",
            "classification": (
                "UNLOCALIZED_PRODUCT_HOST_TRAJECTORY_TYPE_ERROR_BEFORE_POSITION_0"
            ),
            "cause": "UNCLEAR_ERROR_MESSAGE_AND_TRACEBACK_NOT_CAPTURED",
            "ordered_boundaries": {
                "prompt_template_cross_comparison": (
                    "UNAVAILABLE_NO_PRODUCT_POSITION; AUTHORITY_REFERENCE_BINDING_ONLY"
                ),
                "independent_reference": "PASS_FSYNCED_FOUR_POSITIONS",
                "cache_behavior": "UNAVAILABLE_NO_PRODUCT_POSITION",
                "per_layer_kv_state": "UNAVAILABLE_NO_PRODUCT_POSITION",
                "rtl_endpoint_logits": (
                    "UNAVAILABLE_NO_COMPLETED_RTL_LOGIT_POSITION_OR_ENDPOINT_RECEIPT"
                ),
                "lm_head_rank_selection": "UNAVAILABLE_NO_PRODUCT_OR_RTL_POSITION",
                "decode": "NOT_STARTED",
            },
            "performance_root_cause": (
                "NOT_CLAIMED_NO_PHASE_TIMING_PROFILE_OR_CONTROLLED_AB"
            ),
        },
        "output_policy": {
            "generated_output": "NONE",
            "punctuation_only_output": "REJECTED_BY_UNCHANGED_POLICY",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
            "stage_transition": "DISABLED",
        },
        "review": {
            "required_level": "INDEPENDENT_L2",
            "status": "PENDING",
            "repair_activation_before_acceptance": "FORBIDDEN",
        },
    }


def repair_route(seal: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ace2-r6-cf12-additive-repair-route-v1",
        "status": "ROUTED_PENDING_INDEPENDENT_L2_TERMINAL_SEAL_ACCEPTANCE",
        "source_terminal_seal_sha256": sha256_bytes(canonical_bytes(seal)),
        "source_attempt": PACKAGE.name,
        "target_component": "PRODUCT_HOST_RUNTIME_BOUNDARY",
        "fresh_namespace": relative(REPAIR),
        "activation_gate": {
            "required_level": "INDEPENDENT_L2",
            "required_verdict": "ACCEPT_CF12_TERMINAL_SEAL",
            "authority_before_acceptance": "NONE",
            "execution_limit_before_acceptance": 0,
        },
        "required_change": {
            "scope": "ADDITIVE_PRODUCT_HOST_RUNTIME_REPAIR",
            "capture_before_endpoint": [
                "EXCEPTION_TYPE",
                "EXCEPTION_MESSAGE",
                "TRACEBACK",
                "BOUNDARY_LOCAL_CONTEXT_WITHOUT_PRIVATE_PROMPT_BYTES",
            ],
            "persistence": "ATOMIC_FILE_AND_DIRECTORY_FSYNC",
            "diagnostic_goal": (
                "LOCALIZE_OR_REPAIR_THE_CF12_PRODUCT_HOST_TYPE_ERROR_BEFORE_POSITION_0"
            ),
        },
        "preserved_contracts": {
            "cf12_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "cf12_attempt_and_seal": "READ_ONLY",
            "public_rtl_contract": "UNCHANGED",
            "streaming_memory_boundary": "ABSTRACT",
            "non_sram_area_limit_mm2": 2.0,
            "minimum_frequency_mhz": 100,
            "operator_frontier": "ORDERED",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
        },
        "fresh_acceptance_requirements": [
            "TASK_NATIVE_BUILD_AND_TEST",
            "EXECUTABLE_PRODUCT_HOST_BOUNDARY_VERIFICATION",
            "SOURCE_AND_CONSTRAINT_SHA256",
            "CANONICAL_SKY130_SYNTHESIS_AND_OPENSTA_IF_RTL_CHANGES",
            "INDEPENDENT_L2_REVIEW",
        ],
        "not_authorized": [
            "CF12_REPLAY",
            "PUNCTUATION_ONLY_SUCCESS",
            "STAGE1_CLOSURE",
            "STAGE2_START",
        ],
    }


VALIDATOR_SOURCE = """\
#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
raise SystemExit(
    subprocess.call(
        [
            sys.executable,
            str(ROOT / "tools/seal_cf12_generation_probe_terminal.py"),
            "--validate-only",
        ]
    )
)
"""


TEST_SOURCE = """\
from __future__ import annotations

import json
import stat
import subprocess
import sys
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


class Cf12TerminalSealTests(unittest.TestCase):
    def test_full_seal_validation(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PACKAGE / "validate_terminal_seal.py")],
            check=False,
        )
        self.assertEqual(completed.returncode, 0)

    def test_partial_frontier_and_gates(self) -> None:
        seal = json.loads((PACKAGE / "terminal-seal.json").read_text())
        route = json.loads((PACKAGE / "repair-route.json").read_text())
        self.assertEqual(seal["launch_lineage"]["authenticated_start_count"], 1)
        self.assertEqual(seal["fsynced_frontier"]["product"]["host_positions"], 0)
        self.assertEqual(
            seal["classification"]["classification"],
            "UNLOCALIZED_PRODUCT_HOST_TRAJECTORY_TYPE_ERROR_BEFORE_POSITION_0",
        )
        self.assertEqual(seal["fsynced_frontier"]["decode"], "NOT_STARTED")
        self.assertEqual(seal["output_policy"]["stage1"], "OPEN")
        self.assertEqual(seal["output_policy"]["stage2"], "FORBIDDEN")
        self.assertEqual(route["target_component"], "PRODUCT_HOST_RUNTIME_BOUNDARY")
        self.assertEqual(route["activation_gate"]["execution_limit_before_acceptance"], 0)

    def test_namespace_is_read_only(self) -> None:
        for path in PACKAGE.rglob("*"):
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertFalse(mode & stat.S_IWUSR, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
"""


def package_manifest(path: Path) -> dict[str, Any]:
    records = {}
    for entry in sorted(path.rglob("*")):
        if not entry.is_file() or entry.name in {"seal-manifest.json", "seal-manifest.sha256"}:
            continue
        records[entry.relative_to(path).as_posix()] = {
            "size": entry.stat().st_size,
            "sha256": sha256_file(entry),
        }
    return {
        "schema": "ace2-r6-cf12-read-only-terminal-seal-manifest-v1",
        "status": "SEALED_PENDING_INDEPENDENT_L2",
        "hash_algorithm": "SHA-256",
        "files": records,
    }


def validate_seal_namespace(facts: dict[str, Any]) -> None:
    if not SEAL.is_dir() or SEAL.is_symlink():
        raise SealError("CF12 terminal seal namespace is absent or linked")
    actual_files = {
        path.relative_to(SEAL).as_posix()
        for path in SEAL.rglob("*")
        if path.is_file()
    }
    if actual_files != OUTPUT_FILES:
        raise SealError("CF12 terminal seal file inventory differs")
    for path in SEAL.rglob("*"):
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & stat.S_IWUSR:
            raise SealError(f"terminal seal entry is linked or owner-writable: {path}")

    inventory = load_object(SEAL / "hash-inventory.json")
    for label, record in inventory.get("records", {}).items():
        expected_path = SOURCE_PATHS.get(label)
        if (
            expected_path is None
            or record.get("path") != relative(expected_path)
            or record.get("size") != expected_path.stat().st_size
            or record.get("sha256") != sha256_file(expected_path)
        ):
            raise SealError(f"sealed source hash differs: {label}")

    death = load_object(SEAL / "process-death-evidence.json")
    if (
        death.get("status")
        != "PASS_ALL_RECORDED_PIDS_DEAD_AND_NO_BOUND_COMMAND_LINE"
        or death.get("matching_processes") != []
        or any(
            item.get("status") != "DEAD"
            or item.get("proc_entry_exists") is not False
            for item in death.get("recorded_pids", [])
        )
    ):
        raise SealError("sealed process-death evidence differs")
    process_scan(facts["known_pids"])

    seal = load_object(SEAL / "terminal-seal.json")
    route = load_object(SEAL / "repair-route.json")
    if (
        seal.get("hash_inventory_sha256")
        != sha256_file(SEAL / "hash-inventory.json")
        or seal.get("process_death_evidence_sha256")
        != sha256_file(SEAL / "process-death-evidence.json")
        or seal.get("launch_lineage", {}).get("authenticated_start_count") != 1
        or seal.get("launch_lineage", {}).get("registry_elapsed_seconds")
        != EXPECTED_ELAPSED_SECONDS
        or seal.get("classification", {}).get("classification")
        != "UNLOCALIZED_PRODUCT_HOST_TRAJECTORY_TYPE_ERROR_BEFORE_POSITION_0"
        or seal.get("review", {}).get("status") != "PENDING"
        or seal.get("output_policy", {}).get("stage1") != "OPEN"
        or seal.get("output_policy", {}).get("stage2") != "FORBIDDEN"
        or route.get("source_terminal_seal_sha256")
        != sha256_file(SEAL / "terminal-seal.json")
        or route.get("target_component") != "PRODUCT_HOST_RUNTIME_BOUNDARY"
        or route.get("activation_gate", {}).get("execution_limit_before_acceptance")
        != 0
        or Path(ROOT / route.get("fresh_namespace", "")).exists()
    ):
        raise SealError("terminal seal or additive repair route differs")

    request = load_object(SEAL / "review-request.json")
    if (
        request.get("required_level") != "INDEPENDENT_L2"
        or request.get("status") != "PENDING"
        or request.get("stage_transition") != "DISABLED"
    ):
        raise SealError("independent review request differs")
    for target in request.get("targets", []):
        target_path = ROOT / target["path"]
        if target.get("sha256") != sha256_file(target_path):
            raise SealError(f"review target hash differs: {target_path}")

    manifest = load_object(SEAL / "seal-manifest.json")
    expected_manifest = package_manifest(SEAL)
    if manifest != expected_manifest:
        raise SealError("terminal package manifest differs")
    expected_manifest_digest = sha256_file(SEAL / "seal-manifest.json")
    if (SEAL / "seal-manifest.sha256").read_text(encoding="ascii") != (
        f"{expected_manifest_digest}  seal-manifest.json\n"
    ):
        raise SealError("terminal package manifest digest differs")


def create_seal() -> None:
    if os.path.lexists(SEAL):
        raise SealError("CF12 terminal seal namespace already exists")
    if os.path.lexists(REPAIR):
        raise SealError("fresh additive repair namespace is not fresh")

    facts = collect_terminal_facts()
    death = process_scan(facts["known_pids"])
    inventory = hash_inventory()
    seal = terminal_seal(facts, inventory, death)
    route = repair_route(seal)
    parent = SEAL.parent
    temporary = Path(tempfile.mkdtemp(prefix=f".{SEAL.name}.", dir=parent))
    try:
        write_json(temporary / "hash-inventory.json", inventory)
        write_json(temporary / "process-death-evidence.json", death)
        write_json(temporary / "terminal-seal.json", seal)
        write_json(temporary / "repair-route.json", route)
        targets = []
        for name in (
            "terminal-seal.json",
            "hash-inventory.json",
            "process-death-evidence.json",
            "repair-route.json",
        ):
            targets.append(
                {
                    "path": f"{relative(SEAL)}/{name}",
                    "sha256": sha256_file(temporary / name),
                }
            )
        write_json(
            temporary / "review-request.json",
            {
                "schema": "ace2-r6-cf12-terminal-seal-review-request-v1",
                "status": "PENDING",
                "required_level": "INDEPENDENT_L2",
                "review_scope": (
                    "EXACTLY_ONCE_TERMINAL_AUTHENTICATION_HASHED_FSYNC_FRONTIER_"
                    "CONSTRAINED_CLASSIFICATION_AND_ADDITIVE_REPAIR_ROUTE"
                ),
                "required_verdicts": [
                    "ACCEPT_CF12_TERMINAL_SEAL",
                    "REJECT_CF12_TERMINAL_SEAL",
                ],
                "stage_transition": "DISABLED",
                "targets": targets,
            },
        )
        write_json(
            temporary / "construction-report.json",
            {
                "schema": "ace2-r6-cf12-terminal-seal-construction-report-v1",
                "status": "PASS_READ_ONLY_SEAL_CREATED_WITHOUT_ATTEMPT_MUTATION",
                "source_hash_inventory_sha256": sha256_file(
                    temporary / "hash-inventory.json"
                ),
                "authenticated_start_count": 1,
                "terminal": seal["launch_lineage"]["terminal"],
                "classification": seal["classification"]["classification"],
                "independent_l2": "REQUIRED_PENDING",
                "repair_route": "PENDING_L2_ACCEPTANCE",
            },
        )
        write_bytes(
            temporary / "validate_terminal_seal.py",
            VALIDATOR_SOURCE.encode("utf-8"),
            mode=0o555,
        )
        write_bytes(
            temporary / "tests/test_cf12_terminal_seal.py",
            TEST_SOURCE.encode("utf-8"),
        )
        manifest = package_manifest(temporary)
        write_json(temporary / "seal-manifest.json", manifest)
        manifest_digest = sha256_file(temporary / "seal-manifest.json")
        write_bytes(
            temporary / "seal-manifest.sha256",
            f"{manifest_digest}  seal-manifest.json\n".encode("ascii"),
        )
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()), reverse=True
        ):
            directory.chmod(0o555)
            fsync_directory(directory)
        temporary.chmod(0o555)
        fsync_directory(temporary)
        os.replace(temporary, SEAL)
        fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    facts = collect_terminal_facts()
    validate_seal_namespace(facts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        facts = collect_terminal_facts()
        validate_seal_namespace(facts)
    else:
        create_seal()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SealError as error:
        raise SystemExit(f"CF12 terminal seal error: {error}") from error
