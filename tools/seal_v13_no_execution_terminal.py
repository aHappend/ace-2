#!/usr/bin/env python3
"""Create the immutable V13 no-execution terminal for mission 531ffdcbd930."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_action_root"
PACKAGE = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V13_SHELLFREE_PACKAGE.json"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
LIVE_ROOT = ACTION_ROOT / "live"
MISSION = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/531ffdcbd930/mission.json")
HANDOFF = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/531ffdcbd930/initial.json")
OUTPUT_ROOT = PROJECT_ROOT / "build/v13-execution-mission-531ffdcbd930-no-execution-terminal"
TERMINAL_NAME = "TERMINAL_NO_EXECUTION.json"
DIGEST_NAME = f"{TERMINAL_NAME}.sha256"
VERIFIER = PROJECT_ROOT / "tools/verify_v13_no_execution_terminal.py"
ACTION_ID = "ace2:qk-gbfp8-base-v13:execute-once:81a8edc1:20260814T075838Z"
TARGET_MARKERS = (
    "qk_gbfp8_head64_granularity_sweep_transport_shellfree_v13.py",
    "qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v13.py",
)
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}


class SealError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SealError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_object(path: Path, *, canonical: bool = True) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    require(type(value) is dict, f"non-object JSON: {path.name}")
    if canonical:
        require(compact_bytes(value) == raw, f"noncanonical JSON: {path.name}")
    return value, raw


def verify_self_checksum(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    require(type(observed) is str and len(observed) == 64, f"invalid {field}")
    payload = dict(value)
    payload.pop(field)
    require(sha256_bytes(compact_bytes(payload)) == observed, f"mismatched {field}")


def verify_invocation(value: dict[str, Any]) -> None:
    require(set(value) == {"argv", "command_representation", "cwd", "environment", "invocation_sha256", "shell"}, "invocation fields")
    require(value["command_representation"] == "ARGV_VECTOR_ONLY", "invocation representation")
    require(value["shell"] is False, "shell must be false")
    require(value["environment"] == EXACT_ENVIRONMENT, "frozen environment mismatch")
    require(type(value["argv"]) is list and all(type(item) is str for item in value["argv"]), "argv vector")
    require(type(value["cwd"]) is str and value["cwd"], "cwd")
    payload = dict(value)
    observed = payload.pop("invocation_sha256")
    require(sha256_bytes(compact_bytes(payload)) == observed, "invocation checksum")


def target_processes() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    self_pid = os.getpid()
    for child in Path("/proc").iterdir():
        if not child.name.isdigit() or int(child.name) == self_pid:
            continue
        try:
            cmdline = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            start_time = (child / "stat").read_text(encoding="ascii").split()[21]
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
            continue
        observed = sorted(marker for marker in TARGET_MARKERS if marker in cmdline)
        if observed:
            matches.append({"markers": observed, "pid": int(child.name), "start_time_ticks": start_time})
    return sorted(matches, key=lambda item: item["pid"])


def write_create_only(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed = os.lstat(path)
    require(stat.S_ISREG(observed.st_mode) and stat.S_IMODE(observed.st_mode) == 0o400, "sealed file mode")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    require(not os.path.lexists(OUTPUT_ROOT), "terminal already exists; replay prohibited")
    require(not os.path.lexists(LIVE_ROOT), "V13 live namespace exists")
    processes = target_processes()
    require(not processes, f"live V13 target process detected: {processes}")

    package, package_raw = load_object(PACKAGE)
    acceptance, acceptance_raw = load_object(ACCEPTANCE)
    mission, mission_raw = load_object(MISSION, canonical=False)
    handoff, handoff_raw = load_object(HANDOFF, canonical=False)
    verify_self_checksum(package, "package_content_sha256")
    verify_self_checksum(acceptance, "acceptance_sha256")
    require(package["action_identity"]["future_action_id"] == ACTION_ID, "action id")
    require(package["selection"]["candidate_order"] == ["G8", "G4", "G2", "G1"], "candidate order")
    require(package["selection"]["every_hard_gate_required"] is True, "hard-gate policy")
    verify_invocation(package["future_invocation"])
    verify_invocation(package["launcher_invocation"])
    require(package["transport_contract"]["shell"] is False, "transport shell contract")
    require(package["transport_contract"]["environment_inheritance_permitted"] is False, "environment inheritance")
    require(package["claim_boundary"]["execution_authorized"] is False, "package authority boundary")
    require(acceptance["action_id"] == ACTION_ID, "acceptance action id")
    require(acceptance["decision"] == "ACCEPT_STATIC_PACKAGE", "static acceptance decision")
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "static acceptance authority boundary")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_raw), "acceptance package binding")
    require(mission["mission_id"] == "531ffdcbd930", "mission id")
    require(mission.get("context_refs") == [] and mission.get("deps") == [], "unexpected mission authority reference")
    require(mission["objective"].startswith("After fresh operator authorization,"), "mission authorization condition")
    require(handoff["kind"] == "mission_started_handoff" and handoff["mission_id"] == "531ffdcbd930", "handoff identity")

    local_bindings: list[dict[str, Any]] = []
    for binding in package["local_artifact_bindings"]:
        path = Path(binding["path"])
        raw = path.read_bytes()
        require(sha256_bytes(raw) == binding["sha256"], f"local binding drift: {binding['id']}")
        local_bindings.append({"id": binding["id"], "sha256": binding["sha256"]})

    source_bindings = {
        "sealer_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "verifier_sha256": sha256_bytes(VERIFIER.read_bytes()),
    }
    terminal: dict[str, Any] = {
        "action_id": ACTION_ID,
        "artifact_kind": "v13_execution_mission_no_execution_terminal",
        "authorization": {
            "classification": "NO_FRESH_EXPLICIT_EXTERNAL_OPERATOR_AUTHORIZATION",
            "consumed": False,
            "mission_context_refs_empty": True,
            "mission_deps_empty": True,
            "operator_instruction_is_conditional_not_a_grant": True,
            "static_acceptance_grants_execution_authority": False,
        },
        "execution": {
            "candidate_sequence_performed": [],
            "classification": "NO_EXECUTION",
            "credential_consumed": False,
            "evaluator_invocation_count_performed": 0,
            "launcher_invocation_count_performed": 0,
            "matching_target_processes_at_seal": processes,
            "payload_open_count_performed": 0,
            "transport_invocation_count_performed": 0,
            "v13_live_namespace_present_at_seal": False,
        },
        "frozen_contract": {
            "candidate_order": package["selection"]["candidate_order"],
            "every_hard_gate_required": package["selection"]["every_hard_gate_required"],
            "future_invocation": package["future_invocation"],
            "launcher_invocation": package["launcher_invocation"],
            "local_artifact_bindings": local_bindings,
            "transport_contract": package["transport_contract"],
        },
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provenance": {
            "handoff_raw_sha256": sha256_bytes(handoff_raw),
            "mission_raw_sha256": sha256_bytes(mission_raw),
            "package_content_sha256": package["package_content_sha256"],
            "package_raw_sha256": sha256_bytes(package_raw),
            "source_bindings": source_bindings,
            "static_acceptance_raw_sha256": sha256_bytes(acceptance_raw),
            "static_acceptance_self_sha256": acceptance["acceptance_sha256"],
        },
        "review": {
            "fresh_l2_execution_review_status": "PENDING",
            "self_review_is_acceptance": False,
        },
        "schema_version": 1,
        "terminal_policy": {
            "first_record_immutable": True,
            "same_mission_retry_replay_resume_permitted": False,
            "v13_action_retired": False,
            "v13_action_unconsumed": True,
            "future_execution_requires_new_mission_and_fresh_explicit_operator_authorization": True,
        },
    }
    terminal["terminal_sha256"] = sha256_bytes(compact_bytes(terminal))
    terminal_raw = compact_bytes(terminal)
    terminal_raw_sha256 = sha256_bytes(terminal_raw)

    staging = OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}.staging-{os.getpid()}")
    os.mkdir(staging, 0o700)
    try:
        write_create_only(staging / TERMINAL_NAME, terminal_raw)
        write_create_only(staging / DIGEST_NAME, f"{terminal_raw_sha256}  {TERMINAL_NAME}\n".encode("ascii"))
        fsync_directory(staging)
        os.chmod(staging, 0o500)
        os.rename(staging, OUTPUT_ROOT)
        fsync_directory(OUTPUT_ROOT.parent)
    except Exception:
        if staging.exists():
            os.chmod(staging, 0o700)
        raise

    print(compact_bytes({
        "action_id": ACTION_ID,
        "classification": "NO_EXECUTION",
        "status": "SEALED_V13_NO_EXECUTION_TERMINAL",
        "terminal_raw_sha256": terminal_raw_sha256,
        "terminal_self_sha256": terminal["terminal_sha256"],
    }).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
