#!/usr/bin/env python3
"""Verify the immutable V13 no-execution terminal and current no-live state."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
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
TERMINAL = OUTPUT_ROOT / "TERMINAL_NO_EXECUTION.json"
DIGEST = OUTPUT_ROOT / "TERMINAL_NO_EXECUTION.json.sha256"
SEALER = PROJECT_ROOT / "tools/seal_v13_no_execution_terminal.py"
STATIC_VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree.py"
ACTION_ID = "ace2:qk-gbfp8-base-v13:execute-once:81a8edc1:20260814T075838Z"
TARGET_MARKERS = (
    "qk_gbfp8_head64_granularity_sweep_transport_shellfree_v13.py",
    "qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v13.py",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    require(type(value) is dict and compact_bytes(value) == raw, f"noncanonical object: {path.name}")
    return value, raw


def mode_octal(path: Path) -> str:
    return f"{stat.S_IMODE(os.lstat(path).st_mode):04o}"


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


def main() -> int:
    terminal, terminal_raw = load_object(TERMINAL)
    observed_self = terminal.get("terminal_sha256")
    payload = dict(terminal)
    payload.pop("terminal_sha256", None)
    require(sha256_bytes(compact_bytes(payload)) == observed_self, "terminal self checksum")
    terminal_raw_sha256 = sha256_bytes(terminal_raw)
    require(DIGEST.read_text(encoding="ascii") == f"{terminal_raw_sha256}  {TERMINAL.name}\n", "terminal companion")
    require(mode_octal(OUTPUT_ROOT) == "0500", "terminal directory mode")
    require(mode_octal(TERMINAL) == "0400" and mode_octal(DIGEST) == "0400", "terminal file mode")
    require(terminal["action_id"] == ACTION_ID, "terminal action id")
    require(terminal["authorization"]["classification"] == "NO_FRESH_EXPLICIT_EXTERNAL_OPERATOR_AUTHORIZATION", "authorization classification")
    require(terminal["authorization"]["consumed"] is False, "authorization consumption")
    require(terminal["execution"]["classification"] == "NO_EXECUTION", "execution classification")
    require(terminal["execution"]["candidate_sequence_performed"] == [], "candidate sequence must be empty")
    require(terminal["execution"]["transport_invocation_count_performed"] == 0, "transport count")
    require(terminal["execution"]["launcher_invocation_count_performed"] == 0, "launcher count")
    require(terminal["execution"]["evaluator_invocation_count_performed"] == 0, "evaluator count")
    require(terminal["terminal_policy"]["same_mission_retry_replay_resume_permitted"] is False, "replay policy")
    require(terminal["terminal_policy"]["v13_action_unconsumed"] is True, "V13 action state")
    require(not os.path.lexists(LIVE_ROOT), "V13 live namespace appeared")
    processes = target_processes()
    require(not processes, f"live V13 target process detected: {processes}")

    current_hashes = {
        "handoff_raw_sha256": sha256_bytes(HANDOFF.read_bytes()),
        "mission_raw_sha256": sha256_bytes(MISSION.read_bytes()),
        "package_raw_sha256": sha256_bytes(PACKAGE.read_bytes()),
        "static_acceptance_raw_sha256": sha256_bytes(ACCEPTANCE.read_bytes()),
    }
    for field, observed in current_hashes.items():
        require(terminal["provenance"][field] == observed, f"provenance drift: {field}")
    require(terminal["provenance"]["source_bindings"]["sealer_sha256"] == sha256_bytes(SEALER.read_bytes()), "sealer source drift")
    require(terminal["provenance"]["source_bindings"]["verifier_sha256"] == sha256_bytes(Path(__file__).read_bytes()), "verifier source drift")
    sealer_source = SEALER.read_text(encoding="utf-8")
    for prohibited in ("posix_spawn", "subprocess", "Popen(", "os.system", "os.popen"):
        require(prohibited not in sealer_source, f"sealer contains process-launch primitive: {prohibited}")

    acceptance, _ = load_object(ACCEPTANCE)
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "static acceptance authority boundary")
    command = ["/usr/bin/python3", str(STATIC_VERIFIER)]
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
    completed = subprocess.run(command, cwd=ACTION_ROOT, env=environment, shell=False, capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"official static verifier exit {completed.returncode}: {completed.stderr.strip()}")
    official = json.loads(completed.stdout)
    require(official["status"] == "PASS_V13_CANONICAL_PREFLIGHT_SUCCESSOR_INDEPENDENT_STATIC_VERIFICATION", "official static verifier status")
    require(official["action_id"] == ACTION_ID and official["target_process_starts"] == 0, "official verifier action/start count")
    require(official["acceptance"]["raw_sha256"] == current_hashes["static_acceptance_raw_sha256"], "official acceptance binding")
    require(not target_processes(), "target process appeared after decisive verification")

    print(compact_bytes({
        "action_id": ACTION_ID,
        "adversarial_cases": official["adversarial_cases"],
        "classification": "NO_EXECUTION",
        "official_static_status": official["status"],
        "status": "PASS_V13_NO_EXECUTION_TERMINAL_VERIFICATION",
        "target_process_starts": official["target_process_starts"],
        "terminal_raw_sha256": terminal_raw_sha256,
        "terminal_self_sha256": observed_self,
    }).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
