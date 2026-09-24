#!/usr/bin/env python3
"""Construct the non-executing Stage-1 attempt-0013 candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/ace2_chat_demo"
SOURCE_PACKAGE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "attempt-0012-preparation"
)
SOURCE_STATE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "attempt-0012-6f2c8d4e-authority-state"
)
SOURCE_RUN_ID = (
    "ace2-stage1-official-rtl-chat-20260829-0012-6f2c8d4e-"
    "1788032167386441628"
)
SOURCE_EXIT = ROOT / (
    ".argus_subagents/"
    "ace2-stage1-official-rtl-chat-20260829-0012-6f2c8d4e_logs/"
    f"exit_code.{SOURCE_RUN_ID}"
)
REPAIR = ROOT / "build/stage1-output-persistence-successor-v1/attempt-0001"

STAMP = "20260831T095000Z"
DATE = "20260831"
ATTEMPT = "0013"
NONCE = "c4a91d6e8f305b27a2437cd91e50b684"
SHORT_NONCE = NONCE[:8]
DESTINATION = BUILD / (
    f"stage1-official-rtl-backed-chat-demo-{STAMP}-"
    f"attempt-{ATTEMPT}-preparation-r1"
)
DESTINATION_REL = DESTINATION.relative_to(ROOT).as_posix()
OUTPUT_REL = (
    "build/ace2_chat_demo/"
    f"stage1-official-rtl-backed-chat-demo-{STAMP}-"
    f"runtime-output-{ATTEMPT}-{SHORT_NONCE}"
)
STATE_REL = (
    "build/ace2_chat_demo/"
    f"stage1-official-rtl-backed-chat-demo-{STAMP}-"
    f"attempt-{ATTEMPT}-{SHORT_NONCE}-authority-state"
)
REVIEW_REL = (
    "build/ace2_chat_demo/"
    f"stage1-official-rtl-backed-chat-demo-{STAMP}-"
    f"attempt-{ATTEMPT}-{SHORT_NONCE}-l2-review"
)
TASK_ID = (
    f"ace2-stage1-official-rtl-chat-{DATE}-{ATTEMPT}-{SHORT_NONCE}"
)
PACKAGE_RUN_IDENTITY = f"ace2-stage1-attempt-{ATTEMPT}-{NONCE}"
RUN_PATTERN = rf"^{re.escape(TASK_ID)}-[0-9]+$"

SOURCE_PACKAGE_REL = SOURCE_PACKAGE.relative_to(ROOT).as_posix()
SOURCE_STATE_REL = SOURCE_STATE.relative_to(ROOT).as_posix()
REPAIR_REL = REPAIR.relative_to(ROOT).as_posix()
BACKEND_REL = "tools/rtl_arbitrary_text_generation_backend.py"
REGRESSION_REL = "tools/run_stage1_output_persistence_regression.py"
VALIDATOR_SOURCE = ROOT / "tools/validate_stage1_attempt_0013.py"
TEST_SOURCE = ROOT / "tools/test_stage1_attempt_0013.py"
CREATED_AT = "2026-08-31T09:50:00+00:00"
FAILED_PACKAGE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T094450Z-"
    "attempt-0013-preparation"
)
FAILED_QUALIFICATION = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T094450Z-"
    "attempt-0013-qualification"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def write_bytes(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_bytes(value), 0o444)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object required: {path}")
    return value


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in records:
            raise RuntimeError(f"invalid checksum record: {line!r}")
        records[match[2]] = match[1]
    return records


def directory_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item.relative_to(path).as_posix(): {
            "bytes": item.stat(follow_symlinks=False).st_size,
            "mode": stat.S_IMODE(item.stat(follow_symlinks=False).st_mode),
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }


def verify_source_package() -> dict[str, str]:
    sums = parse_sums(SOURCE_PACKAGE / "SHA256SUMS")
    actual = {
        path.relative_to(SOURCE_PACKAGE).as_posix()
        for path in SOURCE_PACKAGE.rglob("*")
        if path.is_file()
        and path.relative_to(SOURCE_PACKAGE).as_posix()
        not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    if actual != set(sums):
        raise RuntimeError("attempt-0012 package file set changed")
    for relative, expected in sums.items():
        if sha256_file(SOURCE_PACKAGE / relative) != expected:
            raise RuntimeError(f"attempt-0012 package member changed: {relative}")
    root = sha256_file(SOURCE_PACKAGE / "SHA256SUMS")
    expected_root = (
        SOURCE_PACKAGE / "TREE_ROOT.sha256"
    ).read_text(encoding="ascii").split()[0]
    if root != expected_root:
        raise RuntimeError("attempt-0012 package root changed")
    return sums


def verify_repair() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load(REPAIR / "manifest.json")
    for name, record in manifest.items():
        path = REPAIR / name
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"output-persistence evidence changed: {name}")
    controlled = load(REPAIR / "controlled-regression.json")
    capacity = load(REPAIR / "capacity.json")
    contract = load(REPAIR / "contract.json")
    if (
        controlled.get("status") != "PASS"
        or controlled.get("official") is not False
        or controlled.get("model_executed") is not False
        or controlled.get("rtl_executed") is not False
        or controlled.get("retained_sha256_exact") is not True
        or capacity.get("status") != "PASS"
        or capacity.get("same_device") is not True
        or capacity.get("safety_margin_percent") != 25
    ):
        raise RuntimeError("output-persistence repair evidence is not acceptable")
    for label, relative in (
        ("backend", BACKEND_REL),
        ("regression", REGRESSION_REL),
    ):
        binding = contract["source_bindings"][label]
        if (
            binding["path"] != relative
            or sha256_file(ROOT / relative) != binding["sha256"]
        ):
            raise RuntimeError(f"repair {label} source binding changed")
    return controlled, capacity, contract


def updated_source_bindings(repair_contract: dict[str, Any]) -> dict[str, str]:
    records = parse_sums(SOURCE_PACKAGE / "source-bindings.sha256")
    repair_backend_sha = repair_contract["source_bindings"]["backend"]["sha256"]
    if records.get(BACKEND_REL) == repair_backend_sha:
        raise RuntimeError("attempt-0013 backend binding is not distinct from 0012")
    records[BACKEND_REL] = repair_backend_sha
    for relative, expected in records.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise RuntimeError(f"accepted execution source drifted: {relative}")
    return records


def map_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
    if isinstance(value, list):
        return [map_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: map_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def build_candidate(source_root: str, terminal_sha: str) -> dict[str, Any]:
    candidate = copy.deepcopy(load(SOURCE_PACKAGE / "package.json"))
    old = load(SOURCE_PACKAGE / "package.json")
    replacements = {
        old["identity"]: DESTINATION.name,
        old["future_output"]: OUTPUT_REL,
        old["future_authority_state"]: STATE_REL,
        old["future_review_receipt_namespace"]: REVIEW_REL,
        old["future_task_id"]: TASK_ID,
        old["package_run_identity"]: PACKAGE_RUN_IDENTITY,
        old["nonce"]: NONCE,
    }
    candidate = map_strings(candidate, replacements)
    candidate.update(
        {
            "schema": "ace2-stage1-attempt-preparation-v12",
            "attempt": ATTEMPT,
            "identity": DESTINATION.name,
            "nonce": NONCE,
            "package_run_identity": PACKAGE_RUN_IDENTITY,
            "created_at_utc": CREATED_AT,
            "preparation_revision": "transient-pruning-repair-0001",
            "current_authority_cardinality": 0,
            "future_authority_cardinality": 1,
            "future_authority_state": STATE_REL,
            "future_output": OUTPUT_REL,
            "future_review_receipt_namespace": REVIEW_REL,
            "future_task_id": TASK_ID,
            "future_runner_run_id": None,
            "future_runner_run_id_template": RUN_PATTERN,
            "execution_occurred": False,
            "status": "PREPARED_AUTHORITY_ABSENT_NOT_EXECUTED",
            "successor_reason": (
                "FRESH_IDENTITY_AFTER_ATTEMPT_0012_MODEL_DISK_GROWTH_LIMIT_"
                "WITH_MEASURED_TRANSIENT_PRUNING_REPAIR"
            ),
            "attempt_0012_status": "SEALED_TERMINAL_NO_RETRY",
            "attempt_0012_classification": "MODEL_DISK_GROWTH_LIMIT",
            "attempt_0012_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "preserved_attempt_0012_package_tree_root": source_root,
            "preserved_attempt_0012_terminal_sha256": terminal_sha,
        }
    )
    candidate["prohibitions"] = [
        item
        for item in candidate["prohibitions"]
        if item != "attempt-0012 execution or submission during preparation"
    ] + [
        "attempt-0012 mutation, retry, replay, resume, relaunch, watcher, "
        "identity reuse, output reuse, or reinterpretation",
        "attempt-0013 model execution, RTL workload, submission, authority "
        "creation, or authority consumption during preparation",
        "Stage-2 or U280 work",
    ]
    return candidate


def build_execution_bindings(
    repair_contract: dict[str, Any],
) -> dict[str, Any]:
    bindings = copy.deepcopy(load(SOURCE_PACKAGE / "execution-bindings.json"))
    old_package = load(SOURCE_PACKAGE / "package.json")
    replacements = {
        SOURCE_PACKAGE_REL: DESTINATION_REL,
        old_package["future_task_id"]: TASK_ID,
    }
    bindings = map_strings(bindings, replacements)
    bindings["schema"] = "ace2-attempt-0013-execution-bindings-v1"
    removed = bindings["process_mode_contract"]["submit_environment_removed"]
    if "ACE2_ATTEMPT_0013_SUBMIT" not in removed:
        removed.append("ACE2_ATTEMPT_0013_SUBMIT")
    bindings["host_output_persistence"] = {
        "policy": "PRUNE_REGENERABLE_WORKING_SET_AFTER_DURABLE_UNIT_RESULT",
        "transient_directories": [
            "execution_sources",
            "sim",
            "tensors",
            "vectors",
        ],
        "backend_path": BACKEND_REL,
        "backend_sha256": repair_contract["source_bindings"]["backend"]["sha256"],
        "repair_evidence": REPAIR_REL,
        "repair_manifest_sha256": sha256_file(REPAIR / "manifest.json"),
    }
    return bindings


def build_commands() -> dict[str, Any]:
    launch_rel = f"{DESTINATION_REL}/launch.sh"
    runner_rel = f"{DESTINATION_REL}/terminal_runner.py"
    prompt_rel = f"{DESTINATION_REL}/prompt.txt"
    model_argv = [
        "python3",
        "-B",
        "tools/ace2_chat_demo.py",
        "--prompt-file",
        prompt_rel,
        "--max-new-tokens",
        "4",
        "--output",
        OUTPUT_REL,
    ]
    return {
        "schema": "ace2-attempt-0013-future-command-v1",
        "submitted": False,
        "run_id_assigned": None,
        "model_argv": model_argv,
        "package_launch_argv": ["bash", launch_rel],
        "worker_entry_mode": "lifecycle",
        "worker_entrypoint_argv": ["bash", launch_rel],
        "submitter_entrypoint_argv": [
            "python3",
            "-B",
            runner_rel,
            "--submit",
        ],
        "host_loss_recovery_argv": [
            "python3",
            "-B",
            runner_rel,
            "--recover-host-loss",
        ],
        "submission_entrypoint": f"bash {DESTINATION_REL}/submit_and_capture.sh",
        "check_with": (
            '"${ARGUS_SKILL_PYTHON:-python3}" -m '
            f"argus_skill.tools.subagent status --task-id {TASK_ID}"
        ),
        "durable_submit_argv": [
            "${ARGUS_SKILL_PYTHON:-python3}",
            "-m",
            "argus_skill.tools.subagent",
            "submit",
            "--task-id",
            TASK_ID,
            "--mode",
            "direct",
            "--timeout",
            "172800",
            "--command",
            f"bash {launch_rel}",
        ],
        "durable_submit_shell": (
            '"${ARGUS_SKILL_PYTHON:-python3}" -m '
            "argus_skill.tools.subagent submit "
            f"--task-id {TASK_ID} --mode direct --timeout 172800 "
            f"--command 'bash {launch_rel}'"
        ),
        "submit_mode_environment_removed": [
            "ACE2_ATTEMPT_0006_SUBMIT",
            "ACE2_ATTEMPT_0007_SUBMIT",
            "ACE2_ATTEMPT_0008_SUBMIT",
            "ACE2_ATTEMPT_0009_SUBMIT",
            "ACE2_ATTEMPT_0010_SUBMIT",
            "ACE2_ATTEMPT_0012_SUBMIT",
            "ACE2_ATTEMPT_0013_SUBMIT",
        ],
    }


def build_capacity(repair_capacity: dict[str, Any]) -> dict[str, Any]:
    bound = repair_capacity["runtime_output_bound_bytes"]
    temporary = repair_capacity["temporary_write_allowance_bytes"]
    package = repair_capacity["package_allowance_bytes"]
    subtotal = bound + temporary + package
    margin_percent = 25
    margin = math.ceil(subtotal * margin_percent / 100)
    required = subtotal + margin
    output_parent = (ROOT / OUTPUT_REL).parent
    devices = {
        DESTINATION.parent.stat().st_dev,
        output_parent.stat().st_dev,
        REPAIR.parent.stat().st_dev,
    }
    if len(devices) != 1:
        raise RuntimeError("candidate, output, and measured repair are not on one device")
    filesystem = os.statvfs(output_parent)
    available = filesystem.f_bavail * filesystem.f_frsize
    if available < required:
        raise RuntimeError(f"insufficient candidate capacity: {available} < {required}")
    if (
        required != repair_capacity["required_available_bytes"]
        or margin != repair_capacity["safety_margin_bytes"]
    ):
        raise RuntimeError("corrected capacity arithmetic differs from measured proof")
    return {
        "schema": "ace2-attempt-0013-capacity-proof-v1",
        "status": "PASS",
        "launch_policy": "FAIL_CLOSED_BELOW_REQUIRED_OR_ON_DEVICE_CHANGE",
        "projection_basis": (
            "MEASURED_COMPLETE_RUN_RETAINED_BOUND_PLUS_TEMPORARY_AND_"
            "PACKAGE_ALLOWANCES_AND_25_PERCENT_MARGIN"
        ),
        "repair_evidence": REPAIR_REL,
        "repair_capacity_sha256": sha256_file(REPAIR / "capacity.json"),
        "filesystem_device": next(iter(devices)),
        "same_device": True,
        "available_bytes_at_preparation": available,
        "runtime_output_max_bytes": bound,
        "complete_run_bound_terms": repair_capacity["bound_terms"],
        "temporary_write_max_bytes": temporary,
        "package_max_bytes": package,
        "safety_margin_percent": margin_percent,
        "safety_margin_bytes": margin,
        "required_available_bytes": required,
        "headroom_after_reserve_bytes": available - required,
        "measured_at_utc": CREATED_AT,
    }


def adjusted_runner() -> bytes:
    source = (SOURCE_PACKAGE / "terminal_runner.py").read_text(encoding="utf-8")
    source = source.replace(
        '"ACE2_ATTEMPT_0012_SUBMIT",',
        '"ACE2_ATTEMPT_0012_SUBMIT",\n    "ACE2_ATTEMPT_0013_SUBMIT",',
        1,
    )
    source = source.replace(
        "ace2-attempt-0012-consumption-v3",
        "ace2-attempt-0013-consumption-v1",
    )
    source = source.replace(
        "ace2-attempt-0012-model-invocation-v2",
        "ace2-attempt-0013-model-invocation-v1",
    )
    source = source.replace(
        '"attempt_0012_retry_replay_resume_relaunch": "FORBIDDEN"',
        '"attempt_0013_retry_replay_resume_relaunch": "FORBIDDEN"',
    )
    source = source.replace(
        'print("ATTEMPT_0012_EXTERNAL_GATES_VALID")',
        'print("ATTEMPT_0013_EXTERNAL_GATES_VALID")',
    )
    forbidden = (
        "ace2-attempt-0012-consumption-v3",
        "ace2-attempt-0012-model-invocation-v2",
        '"attempt_0012_retry_replay_resume_relaunch": "FORBIDDEN"',
        "ATTEMPT_0012_EXTERNAL_GATES_VALID",
    )
    if any(token in source for token in forbidden):
        raise RuntimeError("attempt-0012 current-identity marker survived runner rebinding")
    return source.encode("utf-8")


def seal_package() -> str:
    excluded = {"SHA256SUMS", "TREE_ROOT.sha256"}
    files = sorted(
        path
        for path in DESTINATION.rglob("*")
        if path.is_file()
        and path.relative_to(DESTINATION).as_posix() not in excluded
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(DESTINATION).as_posix()}"
        for path in files
    ]
    write_bytes(
        DESTINATION / "SHA256SUMS",
        ("\n".join(lines) + "\n").encode("ascii"),
        0o444,
    )
    root = sha256_file(DESTINATION / "SHA256SUMS")
    write_bytes(
        DESTINATION / "TREE_ROOT.sha256",
        f"{root}  SHA256SUMS\n".encode("ascii"),
        0o444,
    )
    for directory in sorted(
        [DESTINATION, *(path for path in DESTINATION.rglob("*") if path.is_dir())],
        reverse=True,
    ):
        directory.chmod(0o555)
    return root


def main() -> int:
    if DESTINATION.exists():
        raise RuntimeError(f"fresh destination already exists: {DESTINATION}")
    if any(
        os.path.lexists(ROOT / path)
        for path in (OUTPUT_REL, STATE_REL, REVIEW_REL)
    ):
        raise RuntimeError("attempt-0013 output, authority, or review namespace is not fresh")
    source_sums = verify_source_package()
    controlled, repair_capacity, repair_contract = verify_repair()
    terminal = load(SOURCE_STATE / "terminal-status.json")
    if (
        terminal.get("status") != "SEALED_TERMINAL_NO_RETRY"
        or terminal.get("classification") != "MODEL_DISK_GROWTH_LIMIT"
        or terminal.get("attempt_0012_retry_replay_resume_relaunch") != "FORBIDDEN"
        or not SOURCE_EXIT.is_file()
    ):
        raise RuntimeError("attempt-0012 is not the expected sealed terminal predecessor")

    source_package_before = directory_manifest(SOURCE_PACKAGE)
    source_state_before = directory_manifest(SOURCE_STATE)
    repair_before = directory_manifest(REPAIR)
    source_root = sha256_file(SOURCE_PACKAGE / "SHA256SUMS")
    terminal_sha = sha256_file(SOURCE_STATE / "terminal-status.json")
    exit_sha = sha256_file(SOURCE_EXIT)
    bindings = updated_source_bindings(repair_contract)

    DESTINATION.mkdir(parents=True)
    write_bytes(
        DESTINATION / "prompt.txt",
        (SOURCE_PACKAGE / "prompt.txt").read_bytes(),
        0o444,
    )
    candidate = build_candidate(source_root, terminal_sha)
    write_json(DESTINATION / "package.json", candidate)
    write_json(
        DESTINATION / "execution-bindings.json",
        build_execution_bindings(repair_contract),
    )
    write_json(DESTINATION / "commands.json", build_commands())
    capacity = build_capacity(repair_capacity)
    write_json(DESTINATION / "capacity.json", capacity)
    write_json(
        DESTINATION / "storage-contract.json",
        {
            "schema": "ace2-attempt-0013-storage-v1",
            "growth_policy": (
                "PRUNE_AFTER_DURABLE_UNIT_RESULT_AND_TERMINATE_CHILD_GROUP_"
                "ABOVE_COMPLETE_RUN_BOUND"
            ),
            "runtime_output_max_bytes": capacity["runtime_output_max_bytes"],
            "temporary_write_max_bytes": capacity["temporary_write_max_bytes"],
            "package_max_bytes": capacity["package_max_bytes"],
            "fresh_output_create_only": True,
            "retained": repair_contract["retention_policy"]["retained"],
            "regenerable_after_durable_result": repair_contract[
                "retention_policy"
            ]["regenerable_after_durable_result"],
            "immutable_shared_objects": load(
                SOURCE_PACKAGE / "storage-contract.json"
            )["immutable_shared_objects"],
        },
    )

    source_lines = [
        f"{digest}  {relative}" for relative, digest in bindings.items()
    ]
    write_bytes(
        DESTINATION / "source-bindings.sha256",
        ("\n".join(source_lines) + "\n").encode("ascii"),
        0o444,
    )
    write_json(
        DESTINATION / "source-closure.json",
        {
            "schema": "ace2-attempt-0013-source-closure-v1",
            "member_count": len(bindings),
            "source_bindings_sha256": sha256_file(
                DESTINATION / "source-bindings.sha256"
            ),
            "total_bytes": sum((ROOT / path).stat().st_size for path in bindings),
            "changed_from_attempt_0012": {
                "path": BACKEND_REL,
                "predecessor_sha256": parse_sums(
                    SOURCE_PACKAGE / "source-bindings.sha256"
                )[BACKEND_REL],
                "candidate_sha256": bindings[BACKEND_REL],
                "reason": "MEASURED_TRANSIENT_PRUNING_REPAIR",
            },
        },
    )

    write_bytes(DESTINATION / "terminal_runner.py", adjusted_runner(), 0o555)
    shutil.copyfile(VALIDATOR_SOURCE, DESTINATION / "validate_package.py")
    (DESTINATION / "validate_package.py").chmod(0o555)
    tests = DESTINATION / "tests"
    tests.mkdir()
    shutil.copyfile(TEST_SOURCE, tests / "test_candidate.py")
    (tests / "test_candidate.py").chmod(0o555)
    launch = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {ROOT}\n"
        f"exec python3 -B {DESTINATION_REL}/terminal_runner.py\n"
    )
    submit = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {ROOT}\n"
        f"exec python3 -B {DESTINATION_REL}/terminal_runner.py --submit\n"
    )
    write_bytes(DESTINATION / "launch.sh", launch.encode("ascii"), 0o555)
    write_bytes(
        DESTINATION / "submit_and_capture.sh",
        submit.encode("ascii"),
        0o555,
    )

    source_package_after = directory_manifest(SOURCE_PACKAGE)
    source_state_after = directory_manifest(SOURCE_STATE)
    repair_after = directory_manifest(REPAIR)
    if (
        source_package_before != source_package_after
        or source_state_before != source_state_after
        or repair_before != repair_after
    ):
        raise RuntimeError("claim-bearing predecessor evidence changed during construction")
    write_json(
        DESTINATION / "preservation.json",
        {
            "schema": "ace2-attempt-0013-byte-exact-preservation-v1",
            "status": "PASS",
            "attempt_0012_package": {
                "path": SOURCE_PACKAGE_REL,
                "tree_root": source_root,
                "member_count": len(source_sums),
                "files": source_package_after,
                "before_after_byte_exact": True,
            },
            "attempt_0012_authority_state": {
                "path": SOURCE_STATE_REL,
                "terminal_status_sha256": terminal_sha,
                "terminal_classification": terminal["classification"],
                "files": source_state_after,
                "before_after_byte_exact": True,
            },
            "attempt_0012_exit_sidecar": {
                "path": SOURCE_EXIT.relative_to(ROOT).as_posix(),
                "sha256": exit_sha,
                "bytes": SOURCE_EXIT.stat().st_size,
            },
            "transient_pruning_repair": {
                "path": REPAIR_REL,
                "manifest_sha256": sha256_file(REPAIR / "manifest.json"),
                "files": repair_after,
                "before_after_byte_exact": True,
                "controlled_regression_status": controlled["status"],
                "retained_sha256_exact": controlled["retained_sha256_exact"],
            },
            "mutation": {
                "attempt_0012_package": False,
                "attempt_0012_authority_state": False,
                "attempt_0012_runtime_output": False,
                "repair_evidence": False,
            },
        },
    )
    write_json(
        DESTINATION / "review-contract.json",
        {
            "schema": "ace2-attempt-0013-fresh-review-request-v1",
            "status": "PENDING_FRESH_REVIEW",
            "reviewer": "FRESH_INDEPENDENT_READ_ONLY",
            "authority_granted_by_review": False,
            "execution_permitted_during_review": False,
            "package": DESTINATION_REL,
            "checks": [
                "candidate package closure and source closure are exact",
                "attempt-0013 identity and all output namespaces are fresh",
                "attempt-0012 and repair evidence remain byte-exact",
                "accepted prompt, tokenizer, model, export, generation, KV, "
                "decode, RTL/reference, latency, and terminal behavior remain bound",
                "complete-run output bound includes temporary and package "
                "allowances plus a 25 percent margin on the same device",
                "validation executes neither the model nor the RTL workload",
                "later execution requires separate exact-once authority",
            ],
        },
    )
    failed_status = load(FAILED_QUALIFICATION / "02-focused-validation.status.json")
    if (
        failed_status.get("focused_validation_exit") != 1
        or not (FAILED_PACKAGE / "TREE_ROOT.sha256").is_file()
    ):
        raise RuntimeError("failed preparation ancestry is absent or changed")
    write_json(
        DESTINATION / "preparation-repair-0001.json",
        {
            "schema": "ace2-attempt-0013-preparation-repair-v1",
            "status": "REPAIRED_IN_DISJOINT_PREPARATION_NAMESPACE",
            "failed_package": {
                "path": FAILED_PACKAGE.relative_to(ROOT).as_posix(),
                "tree_root": (
                    FAILED_PACKAGE / "TREE_ROOT.sha256"
                ).read_text(encoding="ascii").split()[0],
            },
            "failed_validation": {
                "status_path": (
                    FAILED_QUALIFICATION / "02-focused-validation.status.json"
                ).relative_to(ROOT).as_posix(),
                "status_sha256": sha256_file(
                    FAILED_QUALIFICATION / "02-focused-validation.status.json"
                ),
                "stderr_path": (
                    FAILED_QUALIFICATION / "02-focused-validation.stderr.log"
                ).relative_to(ROOT).as_posix(),
                "stderr_sha256": sha256_file(
                    FAILED_QUALIFICATION / "02-focused-validation.stderr.log"
                ),
            },
            "failure_taxonomy": "package_integration",
            "root_cause_hypothesis": (
                "The inherited terminal runner requires package.json, but the "
                "first overlay emitted the same contract as candidate.json."
            ),
            "regression": (
                "Import the sealed package terminal_runner.py before accepting "
                "the preparation closure."
            ),
            "identity_reuse": False,
            "failed_package_mutated": False,
        },
    )
    root = seal_package()
    print(
        "ACE2_STAGE1_ATTEMPT_0013_CONSTRUCTED "
        f"package={DESTINATION_REL} tree_root={root} "
        f"runtime_bound_bytes={capacity['runtime_output_max_bytes']} "
        f"required_available_bytes={capacity['required_available_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
