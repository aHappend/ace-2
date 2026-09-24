#!/usr/bin/env python3
"""Package-native validation for the Stage-1 attempt-0013 candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
SOURCE_PACKAGE = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "attempt-0012-preparation"
)
SOURCE_STATE = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "attempt-0012-6f2c8d4e-authority-state"
)
REPAIR = ROOT / "build/stage1-output-persistence-successor-v1/attempt-0001"
FAILED_PACKAGE = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260831T094450Z-"
    "attempt-0013-preparation"
)
FAILED_QUALIFICATION = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260831T094450Z-"
    "attempt-0013-qualification"
)
EXCLUDED = {"SHA256SUMS", "TREE_ROOT.sha256"}
BACKEND_REL = "tools/rtl_arbitrary_text_generation_backend.py"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"object required: {path}")
    return value


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"invalid checksum record: {line!r}")
        assert match is not None
        require(match[2] not in records, f"duplicate checksum record: {match[2]}")
        records[match[2]] = match[1]
    return records


def directory_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item.relative_to(path).as_posix(): {
            "bytes": item.stat(follow_symlinks=False).st_size,
            "mode": stat.S_IMODE(item.stat(follow_symlinks=False).st_mode),
            "sha256": digest(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }


def validate_package_closure() -> None:
    sums = parse_sums(PACKAGE / "SHA256SUMS")
    actual = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and path.relative_to(PACKAGE).as_posix() not in EXCLUDED
        and "__pycache__" not in path.parts
    }
    require(actual == set(sums), "candidate package file set changed")
    for relative, expected in sums.items():
        path = PACKAGE / relative
        require(not path.is_symlink(), f"candidate package symlink: {relative}")
        require(digest(path) == expected, f"candidate package member changed: {relative}")
        expected_mode = 0o555 if path.suffix in {".py", ".sh"} else 0o444
        require(
            stat.S_IMODE(path.stat().st_mode) == expected_mode,
            f"candidate package mode changed: {relative}",
        )
    root = digest(PACKAGE / "SHA256SUMS")
    declared = (PACKAGE / "TREE_ROOT.sha256").read_text(encoding="ascii").strip()
    require(declared == f"{root}  SHA256SUMS", "candidate package root changed")
    for path in [PACKAGE, *(item for item in PACKAGE.rglob("*") if item.is_dir())]:
        if "__pycache__" in path.parts:
            continue
        require(
            not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o555,
            f"candidate package directory mode changed: {path}",
        )


def validate_predecessor() -> None:
    preservation = load(PACKAGE / "preservation.json")
    require(
        preservation.get("schema")
        == "ace2-attempt-0013-byte-exact-preservation-v1"
        and preservation.get("status") == "PASS",
        "preservation contract changed",
    )
    source = preservation["attempt_0012_package"]
    source_sums = parse_sums(SOURCE_PACKAGE / "SHA256SUMS")
    source_actual = {
        path.relative_to(SOURCE_PACKAGE).as_posix()
        for path in SOURCE_PACKAGE.rglob("*")
        if path.is_file()
        and path.relative_to(SOURCE_PACKAGE).as_posix() not in EXCLUDED
    }
    require(source_actual == set(source_sums), "attempt-0012 package file set changed")
    require(
        digest(SOURCE_PACKAGE / "SHA256SUMS") == source["tree_root"],
        "attempt-0012 package tree root changed",
    )
    require(
        directory_manifest(SOURCE_PACKAGE) == source["files"],
        "attempt-0012 package bytes changed",
    )
    state = preservation["attempt_0012_authority_state"]
    terminal = load(SOURCE_STATE / "terminal-status.json")
    require(
        terminal.get("status") == "SEALED_TERMINAL_NO_RETRY"
        and terminal.get("classification") == "MODEL_DISK_GROWTH_LIMIT"
        and terminal.get("attempt_0012_retry_replay_resume_relaunch")
        == "FORBIDDEN",
        "attempt-0012 terminal changed",
    )
    require(
        digest(SOURCE_STATE / "terminal-status.json")
        == state["terminal_status_sha256"],
        "attempt-0012 terminal bytes changed",
    )
    require(
        directory_manifest(SOURCE_STATE) == state["files"],
        "attempt-0012 authority-state bytes changed",
    )
    exit_record = preservation["attempt_0012_exit_sidecar"]
    exit_path = ROOT / exit_record["path"]
    require(
        exit_path.is_file()
        and exit_path.stat().st_size == exit_record["bytes"]
        and digest(exit_path) == exit_record["sha256"],
        "attempt-0012 exit sidecar changed",
    )
    repair = preservation["transient_pruning_repair"]
    require(
        directory_manifest(REPAIR) == repair["files"],
        "transient-pruning evidence bytes changed",
    )
    require(
        digest(REPAIR / "manifest.json") == repair["manifest_sha256"],
        "transient-pruning manifest changed",
    )
    require(
        all(value is False for value in preservation["mutation"].values()),
        "predecessor mutation claim changed",
    )


def validate_identity_and_contract() -> None:
    candidate = load(PACKAGE / "package.json")
    commands = load(PACKAGE / "commands.json")
    bindings = load(PACKAGE / "execution-bindings.json")
    old_candidate = load(SOURCE_PACKAGE / "package.json")
    old_bindings = load(SOURCE_PACKAGE / "execution-bindings.json")
    require(
        candidate.get("schema") == "ace2-stage1-attempt-preparation-v12"
        and candidate.get("attempt") == "0013"
        and candidate.get("status") == "PREPARED_AUTHORITY_ABSENT_NOT_EXECUTED"
        and candidate.get("current_authority_cardinality") == 0
        and candidate.get("execution_occurred") is False,
        "candidate preparation state changed",
    )
    fresh_values = (
        candidate["identity"],
        candidate["nonce"],
        candidate["package_run_identity"],
        candidate["future_task_id"],
        candidate["future_output"],
        candidate["future_authority_state"],
        candidate["future_review_receipt_namespace"],
    )
    predecessor_values = (
        old_candidate["identity"],
        old_candidate["nonce"],
        old_candidate["package_run_identity"],
        old_candidate["future_task_id"],
        old_candidate["future_output"],
        old_candidate["future_authority_state"],
        old_candidate["future_review_receipt_namespace"],
    )
    failed_candidate = load(FAILED_PACKAGE / "candidate.json")
    failed_values = (
        failed_candidate["identity"],
        failed_candidate["nonce"],
        failed_candidate["package_run_identity"],
        failed_candidate["future_task_id"],
        failed_candidate["future_output"],
        failed_candidate["future_authority_state"],
        failed_candidate["future_review_receipt_namespace"],
    )
    require(
        len(set(fresh_values)) == len(fresh_values)
        and set(fresh_values).isdisjoint(predecessor_values)
        and set(fresh_values).isdisjoint(failed_values),
        "attempt-0013 identity is not wholly fresh",
    )
    repair = load(PACKAGE / "preparation-repair-0001.json")
    failed_status = FAILED_QUALIFICATION / "02-focused-validation.status.json"
    failed_stderr = FAILED_QUALIFICATION / "02-focused-validation.stderr.log"
    require(
        repair.get("failure_taxonomy") == "package_integration"
        and repair.get("identity_reuse") is False
        and repair.get("failed_package_mutated") is False
        and repair["failed_package"]["tree_root"]
        == (FAILED_PACKAGE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()[0]
        and repair["failed_validation"]["status_sha256"] == digest(failed_status)
        and repair["failed_validation"]["stderr_sha256"] == digest(failed_stderr),
        "failed preparation ancestry changed",
    )
    require(
        re.fullmatch(candidate["future_runner_run_id_template"], candidate["future_task_id"] + "-1")
        is not None,
        "fresh run-id pattern changed",
    )
    require(
        commands.get("submitted") is False
        and commands.get("run_id_assigned") is None
        and commands["model_argv"]
        == [
            "python3",
            "-B",
            "tools/ace2_chat_demo.py",
            "--prompt-file",
            f"{PACKAGE.relative_to(ROOT).as_posix()}/prompt.txt",
            "--max-new-tokens",
            "4",
            "--output",
            candidate["future_output"],
        ],
        "future model command changed",
    )
    require(
        "ACE2_ATTEMPT_0012_SUBMIT" in commands["submit_mode_environment_removed"]
        and "ACE2_ATTEMPT_0013_SUBMIT" in commands["submit_mode_environment_removed"],
        "submit-mode isolation changed",
    )
    require(
        (PACKAGE / "prompt.txt").read_bytes()
        == (SOURCE_PACKAGE / "prompt.txt").read_bytes(),
        "accepted prompt bytes changed",
    )
    for field in ("backend", "evaluator", "model", "quantization", "tokenizer"):
        require(bindings[field] == old_bindings[field], f"accepted {field} binding changed")
    require(
        bindings["prompt"]["bytes"] == old_bindings["prompt"]["bytes"]
        and bindings["prompt"]["sha256"] == old_bindings["prompt"]["sha256"]
        and bindings["prompt"]["max_new_tokens"]
        == old_bindings["prompt"]["max_new_tokens"]
        and bindings["backend"]["context_bound"]
        == old_bindings["backend"]["context_bound"],
        "accepted prompt or generation bound changed",
    )
    persistence = bindings.get("host_output_persistence")
    repair_contract = load(REPAIR / "contract.json")
    require(
        isinstance(persistence, dict)
        and persistence.get("backend_sha256")
        == repair_contract["source_bindings"]["backend"]["sha256"]
        and persistence.get("transient_directories")
        == ["execution_sources", "sim", "tensors", "vectors"],
        "transient-pruning binding changed",
    )
    for relative in (
        candidate["future_output"],
        candidate["future_authority_state"],
        candidate["future_review_receipt_namespace"],
        f".argus_subagents/{candidate['future_task_id']}.json",
    ):
        require(not os.path.lexists(ROOT / relative), f"fresh namespace is nonzero: {relative}")
    require(
        not any("authority" in path.name.lower() for path in PACKAGE.iterdir()),
        "candidate package contains execution authority",
    )


def validate_sources() -> None:
    old = parse_sums(SOURCE_PACKAGE / "source-bindings.sha256")
    current = parse_sums(PACKAGE / "source-bindings.sha256")
    closure = load(PACKAGE / "source-closure.json")
    repair = load(REPAIR / "contract.json")
    require(set(current) == set(old), "accepted source member set changed")
    require(
        closure.get("member_count") == len(current)
        and closure.get("source_bindings_sha256")
        == digest(PACKAGE / "source-bindings.sha256")
        and closure.get("total_bytes")
        == sum((ROOT / relative).stat().st_size for relative in current),
        "source closure metadata changed",
    )
    for relative, expected in current.items():
        path = ROOT / relative
        require(
            path.is_file() and not path.is_symlink() and digest(path) == expected,
            f"candidate source changed: {relative}",
        )
        if relative != BACKEND_REL:
            require(expected == old[relative], f"unapproved source drift: {relative}")
    require(
        current[BACKEND_REL] == repair["source_bindings"]["backend"]["sha256"]
        and current[BACKEND_REL] != old[BACKEND_REL],
        "backend is not the measured transient-pruning repair",
    )


def validate_capacity(proof: dict[str, Any]) -> None:
    repair = load(REPAIR / "capacity.json")
    terms = proof["complete_run_bound_terms"]
    calculated_bound = (
        terms["layer_units"] * terms["per_layer_retained_allocated_bytes"]
        + terms["rank1_sidecar_units"]
        * terms["per_rank1_sidecar_retained_allocated_bytes"]
        + terms["head_units"] * terms["per_head_retained_allowance_bytes"]
        + terms["shared_live_support_bytes"]
        + terms["top_level_retained_allowance_bytes"]
        + terms["filesystem_metadata_allowance_bytes"]
    )
    subtotal = (
        proof["runtime_output_max_bytes"]
        + proof["temporary_write_max_bytes"]
        + proof["package_max_bytes"]
    )
    expected_margin = math.ceil(subtotal * 25 / 100)
    require(
        proof.get("schema") == "ace2-attempt-0013-capacity-proof-v1"
        and proof.get("status") == "PASS"
        and proof.get("same_device") is True
        and proof.get("safety_margin_percent") == 25
        and proof["runtime_output_max_bytes"] == calculated_bound
        and proof["runtime_output_max_bytes"] == repair["runtime_output_bound_bytes"]
        and proof["temporary_write_max_bytes"]
        == repair["temporary_write_allowance_bytes"]
        and proof["package_max_bytes"] == repair["package_allowance_bytes"]
        and proof["safety_margin_bytes"] == expected_margin
        and proof["required_available_bytes"] == subtotal + expected_margin,
        "corrected complete-run capacity arithmetic changed",
    )
    candidate = load(PACKAGE / "package.json")
    output_parent = (ROOT / candidate["future_output"]).parent
    devices = {
        PACKAGE.parent.stat().st_dev,
        output_parent.stat().st_dev,
        REPAIR.parent.stat().st_dev,
    }
    require(
        len(devices) == 1 and proof["filesystem_device"] in devices,
        "capacity gate is not bound to the measured output device",
    )
    filesystem = os.statvfs(output_parent)
    available = filesystem.f_bavail * filesystem.f_frsize
    require(
        available >= proof["required_available_bytes"],
        "live same-device capacity is below the complete-run requirement",
    )


def validate_runner_and_review() -> None:
    runner = (PACKAGE / "terminal_runner.py").read_text(encoding="utf-8")
    require(
        "ace2-attempt-0013-consumption-v1" in runner
        and "ace2-attempt-0013-model-invocation-v1" in runner
        and '"attempt_0013_retry_replay_resume_relaunch": "FORBIDDEN"' in runner
        and "ACE2_ATTEMPT_0012_SUBMIT" in runner
        and "ACE2_ATTEMPT_0013_SUBMIT" in runner
        and "ATTEMPT_0013_EXTERNAL_GATES_VALID" in runner,
        "terminal-accountability runner was not rebound to attempt-0013",
    )
    require(
        "ace2-attempt-0012-consumption-v3" not in runner
        and '"attempt_0012_retry_replay_resume_relaunch": "FORBIDDEN"' not in runner,
        "terminal runner retains the consumed attempt-0012 identity",
    )
    review = load(PACKAGE / "review-contract.json")
    require(
        review.get("status") == "PENDING_FRESH_REVIEW"
        and review.get("authority_granted_by_review") is False
        and review.get("execution_permitted_during_review") is False,
        "fresh review boundary changed",
    )


def validate() -> None:
    validate_package_closure()
    validate_predecessor()
    validate_identity_and_contract()
    validate_sources()
    validate_capacity(load(PACKAGE / "capacity.json"))
    validate_runner_and_review()


def main() -> int:
    validate()
    candidate = load(PACKAGE / "package.json")
    capacity = load(PACKAGE / "capacity.json")
    print(
        "ACE2_STAGE1_ATTEMPT_0013_PACKAGE_VALID "
        f"identity={candidate['package_run_identity']} "
        f"runtime_bound_bytes={capacity['runtime_output_max_bytes']} "
        f"required_available_bytes={capacity['required_available_bytes']} "
        "model_executed=false rtl_executed=false authority_cardinality=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
