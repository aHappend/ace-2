#!/usr/bin/env python3
"""Native ownership and terminal consumer for the reviewed prompt recovery."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus_skill.adapters.agent_cli_backend import AgentCliBackend
from argus_skill.agent_cli.runner_backend import resolve_available_runner
from argus_skill.core.event_catalog import EventType
from argus_skill.core.knobs import resolve_role_model
from argus_skill.life.event_log import JsonlEventSink
from argus_skill.life.memory import Backlog, LifeMemory
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path(__file__).resolve().parents[1]
LIFE_DIR = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b"
)
AUTHORITY_PATH = LIFE_DIR / ".argus/prompt-recovery-native-consumer-20260915-01-authority.json"
STATE_DIR = LIFE_DIR / ".argus/prompt-recovery-native-consumer-20260915-01"
CONTINUATION_STATE = LIFE_DIR / ".argus/prompt-recovery-continuation-20260915-01"
ORIGINAL_STATE = LIFE_DIR / ".argus/ace2-case2-retry-20260915-01"
CONTINUATION_AUTHORITY = LIFE_DIR / ".argus/prompt-recovery-continuation-20260915-01-authority.json"
ACTIVE_DIRECTIVE = LIFE_DIR / "active_manager_directive.json"
SESSION_EVENTS = Path(
    "/home/argustest/.argus-skill-ace2/copilot-home/session-state/"
    "4a4daf2b-d5c5-4ca0-b465-377d7b4c3cdb/events.jsonl"
)
TASK_ID = "2843f71abcdd"
OLD_OWNER = "ace2-prompt-recovery-coordinator:ace2-case2-retry-20260915-01"
NEW_OWNER = "ace2-prompt-recovery-continuation:prompt-recovery-continuation-20260915-01"
EXPECTED_CONTINUATION_AUTHORITY_SHA256 = (
    "85f974e930e2476e95fe3aa495b3f965f42b7c4667bc9ba8d9e63d572cd31058"
)
EXPECTED_RELEASE_SHA256 = (
    "50973a6cebf875af273efdf51fe7026341df1e7d0452ae4c5c52dc4340e721b2"
)
EXPECTED_IMPLEMENTATION_REVIEW_SHA256 = (
    "6e16b531f4d03665d5f682c3b00e25b7460a1b45b9dd6a0e5cd6ea3ea4946a84"
)
EXPECTED_DIRECTIVE_REVISION = "ec73d2169990409bbfd80b4bc9459291"
EXPECTED_ORIGINAL_OWNER = {"pid": 387342, "start_ticks": 147848075}


class ConsumerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConsumerError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_atomic_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ConsumerError(f"append-only record already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def process_identity(pid: int) -> dict[str, Any]:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = stat.rfind(")")
    fields = stat[close + 2 :].split()
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "session_id": int(fields[3]),
        "start_ticks": int(fields[19]),
        "argv": [
            part.decode("utf-8", errors="surrogateescape")
            for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if part
        ],
        "cwd": str(Path(f"/proc/{pid}/cwd").resolve()),
    }


def identity_live(record: dict[str, Any]) -> bool:
    try:
        current = process_identity(int(record["pid"]))
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return (
        current["state"] != "Z"
        and current["start_ticks"] == int(record["start_ticks"])
        and current["session_id"] == int(record["session_id"])
    )


def lock_holder_pid(path: Path) -> int | None:
    stat = path.stat()
    identity = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    for line in Path("/proc/locks").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "FLOCK" and fields[3] == "WRITE":
            if fields[5] == identity:
                return int(fields[4])
    return None


def emit(event: dict[str, Any]) -> None:
    accepted = JsonlEventSink(None, life_dir=LIFE_DIR).handle_event(event)
    require(accepted is not False, "canonical event log rejected event")


def review_prompt(native_bindings: dict[str, str]) -> str:
    return (
        "Independent final review of the frozen ACE2 native ownership/terminal "
        "consumer in /home/argustest/ace-2. Read-only except the exact fixture "
        "command below.\n"
        f"consumer-sha256={sha256_file(Path(__file__).resolve())}\n"
        "test-sha256="
        f"{sha256_file(ROOT / 'tests/test_prompt_recovery_native_consumer.py')}\n"
        "native-bindings-sha256="
        f"{hashlib.sha256(json.dumps(native_bindings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}\n"
        "Review only scripts/run_prompt_recovery_native_consumer.py and "
        "tests/test_prompt_recovery_native_consumer.py against the already approved "
        "continuation artifacts. Verify it cannot run science; waits for exact "
        "adoption; proves the continuation PID holds original owner.lock; changes "
        "only the same running task's owner under the native backlog lock with an "
        "exact old-owner CAS and append-only audit; routes pre-adoption or "
        "committed-owner blocked terminals to native failed settlement and records "
        "an explicit technical block rather than bypassing a missed ownership "
        "handoff; routes success only through the standard "
        "read-only Reviewer.evaluate L2; and emits canonical mission completion "
        "before declaring its own settlement. Do not modify files, launch the "
        "consumer, run science, or settle backlog state.\n"
        "Run exactly: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 "
        "tests/test_prompt_recovery_native_consumer.py\n"
        "If and only if no actionable issue remains, respond exactly:\n"
        "APPROVED"
    )


def read_jsonl_line(path: Path, line_number: int) -> tuple[dict[str, Any], str]:
    require(line_number > 0, "review event line is invalid")
    with path.open("rb") as stream:
        for observed, raw in enumerate(stream, 1):
            if observed == line_number:
                return json.loads(raw), hashlib.sha256(raw).hexdigest()
    raise ConsumerError(f"review event line is absent: {line_number}")


def validate_review(record: dict[str, Any], native_bindings: dict[str, str]) -> None:
    review_path = Path(record["independent_review"]["path"])
    require(sha256_file(review_path) == record["independent_review"]["sha256"], "consumer review binding differs")
    review = load_json(review_path)
    require(review.get("schema") == "ace2-prompt-recovery-native-consumer-review-v1", "consumer review schema differs")
    require(review.get("status") == "APPROVED" and review.get("independent") is True, "consumer review is not approved")
    require(review.get("consumer_sha256") == sha256_file(Path(__file__).resolve()), "reviewed consumer bytes differ")
    require(review.get("test_sha256") == sha256_file(ROOT / "tests/test_prompt_recovery_native_consumer.py"), "reviewed consumer tests differ")
    require(review.get("native_bindings") == native_bindings, "reviewed native bindings differ")
    raw = review["raw_approval_call"]
    require(Path(raw["event_path"]) == SESSION_EVENTS, "consumer review event source differs")
    request, request_sha = read_jsonl_line(SESSION_EVENTS, int(raw["request_event_line"]))
    start, start_sha = read_jsonl_line(SESSION_EVENTS, int(raw["start_event_line"]))
    verdict, verdict_sha = read_jsonl_line(SESSION_EVENTS, int(raw["verdict_event_line"]))
    require(request_sha == raw["request_event_raw_sha256"], "consumer review request changed")
    require(start_sha == raw["start_event_raw_sha256"], "consumer review start changed")
    require(verdict_sha == raw["verdict_event_raw_sha256"], "consumer review verdict changed")
    call_id = raw["tool_call_id"]
    agent_id = raw["agent_id"]
    request_data = request.get("data", {})
    arguments = request_data.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    require(
        request.get("type") == "tool.execution_start"
        and request_data.get("toolName") == "task"
        and request_data.get("toolCallId") == call_id
        and isinstance(arguments, dict)
        and arguments.get("agent_type") == "code-review"
        and arguments.get("prompt") == review_prompt(native_bindings),
        "consumer review request does not exactly bind the release",
    )
    require(
        start.get("type") == "subagent.started"
        and start.get("agentId") == agent_id
        and start.get("data", {}).get("agentType") == "code-review"
        and start.get("data", {}).get("toolCallId") == call_id,
        "consumer review start identity differs",
    )
    require(
        verdict.get("type") == "assistant.message"
        and verdict.get("agentId") == agent_id
        and verdict.get("data", {}).get("parentToolCallId") == call_id
        and verdict.get("data", {}).get("content") == "APPROVED",
        "consumer review verdict is not exact approval",
    )


def expected_worker_argv() -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        str(Path(__file__).resolve()),
        "_consume",
    ]


def authority() -> dict[str, Any]:
    record = load_json(AUTHORITY_PATH)
    require(record.get("schema") == "ace2-prompt-recovery-native-consumer-authority-v1", "consumer authority schema differs")
    require(record.get("status") == "GRANTED_AFTER_INDEPENDENT_REVIEW", "consumer authority status differs")
    require(record.get("task_id") == TASK_ID, "consumer task differs")
    require(record.get("old_owner") == OLD_OWNER and record.get("new_owner") == NEW_OWNER, "consumer owner route differs")
    require(Path(record["state_dir"]) == STATE_DIR, "consumer state path differs")
    require(record.get("continuation_authority_sha256") == EXPECTED_CONTINUATION_AUTHORITY_SHA256, "continuation authority binding differs")
    require(record.get("continuation_release_sha256") == EXPECTED_RELEASE_SHA256, "continuation release binding differs")
    require(record.get("continuation_review_sha256") == EXPECTED_IMPLEMENTATION_REVIEW_SHA256, "continuation review binding differs")
    require(record.get("directive_revision") == EXPECTED_DIRECTIVE_REVISION, "directive revision differs")
    require(record.get("worker_argv") == expected_worker_argv(), "consumer worker argv is not trusted")
    require(record.get("consumer_sha256") == sha256_file(Path(__file__).resolve()), "consumer bytes changed")
    require(record.get("test_sha256") == sha256_file(ROOT / "tests/test_prompt_recovery_native_consumer.py"), "consumer tests changed")
    require(sha256_file(CONTINUATION_AUTHORITY) == EXPECTED_CONTINUATION_AUTHORITY_SHA256, "continuation authority changed")
    directive = load_json(ACTIVE_DIRECTIVE)
    require(directive.get("revision") == EXPECTED_DIRECTIVE_REVISION, "active Manager directive changed")
    native_bindings = record.get("native_bindings", {})
    require(isinstance(native_bindings, dict) and native_bindings, "native lifecycle bindings are absent")
    for raw_path, expected in native_bindings.items():
        require(sha256_file(Path(raw_path)) == expected, f"native lifecycle binding changed: {raw_path}")
    validate_review(record, native_bindings)
    return record


def observer_identity() -> dict[str, Any]:
    process = load_json(CONTINUATION_STATE / "observer-process.json")
    ready = load_json(CONTINUATION_STATE / "observer-ready.json")
    claim = load_json(CONTINUATION_STATE / "worker-claim.json")
    for field in ("pid", "start_ticks", "session_id", "argv", "cwd"):
        require(process.get(field) == ready.get(field) == claim.get(field), f"observer {field} receipts differ")
    require(ready.get("authority_sha256") == EXPECTED_CONTINUATION_AUTHORITY_SHA256, "observer authority differs")
    require(
        claim.get("observer_process_sha256")
        == sha256_file(CONTINUATION_STATE / "observer-process.json"),
        "observer worker claim binding differs",
    )
    return process


def validate_adoption_records() -> tuple[dict[str, Any], dict[str, Any]]:
    adoption = load_json(CONTINUATION_STATE / "adoption.json")
    owner = load_json(CONTINUATION_STATE / "continuation-owner.json")
    observer = observer_identity()
    require(adoption.get("status") == "ADOPTED_AFTER_ORIGINAL_OWNER_LOCK_RELEASE_AND_FAIL_CLOSED_TERMINAL", "adoption status differs")
    require(adoption.get("case2_rtl_replayed") is False, "adoption claims case2 replay")
    require(adoption.get("original_owner", {}).get("pid") == EXPECTED_ORIGINAL_OWNER["pid"], "original owner PID differs")
    require(adoption.get("original_owner", {}).get("start_ticks") == EXPECTED_ORIGINAL_OWNER["start_ticks"], "original owner start differs")
    require(owner.get("status") == "OWNS_ORIGINAL_OWNER_LOCK_AFTER_ATOMIC_ADOPTION", "continuation owner status differs")
    require(owner.get("adoption_sha256") == sha256_file(CONTINUATION_STATE / "adoption.json"), "owner adoption binding differs")
    for field in ("pid", "start_ticks", "session_id"):
        require(owner.get(field) == observer.get(field), f"continuation owner {field} differs")
    require(not identity_live({**EXPECTED_ORIGINAL_OWNER, "session_id": EXPECTED_ORIGINAL_OWNER["pid"]}), "original owner is still live")
    terminal_path = Path(adoption["original_terminal_path"])
    require(terminal_path == ORIGINAL_STATE / "terminal.json", "original terminal path differs")
    require(sha256_file(terminal_path) == adoption["original_terminal_sha256"], "original terminal binding differs")
    terminal = load_json(terminal_path)
    require(terminal.get("status") == "FAILED_NO_AUTOMATIC_RETRY", "original handoff terminal differs")
    require(terminal.get("error") == "reviewed bytes changed before scientific exec gate", "original handoff reason differs")
    return adoption, owner


def validate_adoption() -> tuple[dict[str, Any], dict[str, Any]]:
    adoption, owner = validate_adoption_records()
    require(identity_live(owner), "continuation owner identity is not live")
    require(lock_holder_pid(ORIGINAL_STATE / "owner.lock") == owner["pid"], "original owner.lock is not held by continuation owner")
    return adoption, owner


def transfer_owner(backlog: Backlog, adoption: dict[str, Any], owner: dict[str, Any]) -> None:
    intent_path = STATE_DIR / "owner-transfer-intent.json"
    intent_preexisting = intent_path.exists()
    if not intent_preexisting:
        write_atomic_create(
            intent_path,
            {
                "schema": "ace2-native-owner-transfer-intent-v1",
                "status": "VALIDATED_FOR_EXACT_CAS",
                "task_id": TASK_ID,
                "old_owner": OLD_OWNER,
                "new_owner": NEW_OWNER,
                "adoption_sha256": sha256_file(CONTINUATION_STATE / "adoption.json"),
                "continuation_owner_sha256": sha256_file(CONTINUATION_STATE / "continuation-owner.json"),
                "owner_process": {field: owner[field] for field in ("pid", "start_ticks", "session_id")},
                "recorded_at_utc": utc_now(),
            },
        )
    with backlog._locked():
        items = backlog._load()
        item = next((candidate for candidate in items if candidate.id == TASK_ID), None)
        require(item is not None, "canonical task is absent")
        require(item.status == "running", "canonical task is not running")
        require(identity_live(owner), "continuation owner exited before native owner CAS")
        require(
            lock_holder_pid(ORIGINAL_STATE / "owner.lock") == owner["pid"],
            "continuation released original owner.lock before native owner CAS",
        )
        recovered = item.running_owner == NEW_OWNER
        if recovered:
            require(intent_preexisting, "new owner has no prior native CAS intent")
            intent = load_json(intent_path)
            require(intent.get("old_owner") == OLD_OWNER and intent.get("new_owner") == NEW_OWNER, "recovery transfer intent route differs")
            require(intent.get("adoption_sha256") == sha256_file(CONTINUATION_STATE / "adoption.json"), "recovery transfer intent adoption differs")
        else:
            require(item.running_owner == OLD_OWNER, "canonical task owner is neither expected owner")
            item.running_owner = NEW_OWNER
            backlog._save(items)
        committed = STATE_DIR / "owner-transfer-committed.json"
        require(not committed.exists(), "owner transfer commit already exists")
        write_atomic_create(
            committed,
            {
                "schema": "ace2-native-owner-transfer-v1",
                "status": (
                    "RECOVERED_AFTER_NATIVE_CAS_BEFORE_COMMIT"
                    if recovered
                    else "COMMITTED_UNDER_NATIVE_BACKLOG_LOCK"
                ),
                "task_id": TASK_ID,
                "old_owner": OLD_OWNER,
                "new_owner": NEW_OWNER,
                "intent_sha256": sha256_file(intent_path),
                "adoption_sha256": sha256_file(CONTINUATION_STATE / "adoption.json"),
                "recorded_at_utc": utc_now(),
            },
        )
    if committed.exists():
        emit(
            {
                "type": "life.mission.owner_transferred",
                "item_id": TASK_ID,
                "old_owner": OLD_OWNER,
                "running_owner": NEW_OWNER,
                "adoption_sha256": sha256_file(CONTINUATION_STATE / "adoption.json"),
                "consumer_authority_sha256": sha256_file(AUTHORITY_PATH),
            }
        )


def validate_committed_transfer(backlog: Backlog) -> None:
    intent_path = STATE_DIR / "owner-transfer-intent.json"
    committed_path = STATE_DIR / "owner-transfer-committed.json"
    intent = load_json(intent_path)
    committed = load_json(committed_path)
    require(intent.get("status") == "VALIDATED_FOR_EXACT_CAS", "owner transfer intent differs")
    require(
        committed.get("status")
        in {
            "COMMITTED_UNDER_NATIVE_BACKLOG_LOCK",
            "RECOVERED_AFTER_NATIVE_CAS_BEFORE_COMMIT",
        },
        "owner transfer commit differs",
    )
    require(committed.get("task_id") == TASK_ID, "owner transfer task differs")
    require(committed.get("old_owner") == OLD_OWNER and committed.get("new_owner") == NEW_OWNER, "owner transfer route differs")
    require(committed.get("intent_sha256") == sha256_file(intent_path), "owner transfer intent binding differs")
    require(committed.get("adoption_sha256") == sha256_file(CONTINUATION_STATE / "adoption.json"), "owner transfer adoption binding differs")
    item = task(backlog)
    require(item.status == "running" and item.running_owner == NEW_OWNER, "committed native owner is not active")


def observe_owner_transition(backlog: Backlog, transferred: bool) -> bool:
    adoption_path = CONTINUATION_STATE / "adoption.json"
    owner_path = CONTINUATION_STATE / "continuation-owner.json"
    if transferred:
        validate_committed_transfer(backlog)
        return True
    if adoption_path.exists() and owner_path.exists():
        adoption, owner = validate_adoption()
        transfer_owner(backlog, adoption, owner)
        validate_committed_transfer(backlog)
        return True
    if adoption_path.exists() or owner_path.exists():
        observer = observer_identity()
        require(identity_live(observer), "partial continuation ownership publication")
    return False


def task(backlog: Backlog) -> Any:
    item = next((candidate for candidate in backlog.all() if candidate.id == TASK_ID), None)
    require(item is not None, "canonical task is absent")
    return item


def reviewer_decision(terminal: dict[str, Any], item: Any) -> Any:
    result_path = Path(terminal["result_path"])
    require(result_path.is_file() and sha256_file(result_path) == terminal["result_sha256"], "continuation result binding differs")
    backend_name, runner_bin = resolve_available_runner("copilot", None)
    require(backend_name == "copilot", "native Copilot reviewer backend is unavailable")
    backend = AgentCliBackend(
        backend=backend_name,
        runner_bin=runner_bin,
    )
    reviewer = Reviewer(backend, memory_maintenance_enabled=False)
    decision = reviewer.evaluate(
        objective=(
            "Perform the final independent L2 review for canonical task 2843f71abcdd. "
            "Read only the preserved first-case evidence, adopted case2 evidence, new case3 "
            "evidence, combined result, continuation terminal, adoption, authorities, reviews, "
            "and custody receipts. Return done only if the combined scoped result is authentic "
            "and complete, no case1/case2 RTL replay occurred, all required RTL/oracle predicates "
            "pass, and the continuation stopped at PASS_PENDING_FINAL_INDEPENDENT_L2. Never run "
            "science, modify files, or treat this as a wholly fresh three-case suite."
        ),
        original_objective=item.original_objective or item.objective,
        operator_messages=[],
        round_index=1,
        round_max=1,
        session_id=None,
        main_summary=f"Continuation terminal: {CONTINUATION_STATE / 'continuation-terminal.json'}",
        main_error=None,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=os.environ.get("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", "high"),
            active_vertical="digital_circuit",
            skip_git_repo_check=True,
            working_dir=str(ROOT),
            vertical_state_root=str(LIFE_DIR),
        ),
        raw_evidence=json.dumps(
            {
                "terminal_path": str(CONTINUATION_STATE / "continuation-terminal.json"),
                "terminal_sha256": sha256_file(CONTINUATION_STATE / "continuation-terminal.json"),
                "result_path": str(result_path),
                "result_sha256": terminal["result_sha256"],
                "adoption_path": str(CONTINUATION_STATE / "adoption.json"),
                "authority_path": str(CONTINUATION_AUTHORITY),
                "authority_sha256": EXPECTED_CONTINUATION_AUTHORITY_SHA256,
                "release_sha256": EXPECTED_RELEASE_SHA256,
                "implementation_review_sha256": EXPECTED_IMPLEMENTATION_REVIEW_SHA256,
            },
            sort_keys=True,
        ),
        scope="final_submission",
        preselected_skill_block="",
    )
    record = decision.to_event_payload(
        item_id=TASK_ID,
        role="reviewer",
        level="L2",
        independent=True,
        terminal_sha256=sha256_file(CONTINUATION_STATE / "continuation-terminal.json"),
    )
    write_atomic_create(STATE_DIR / "final-l2-review.json", record)
    emit(record)
    return decision


def settlement_event(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": EventType.LIFE_MISSION_COMPLETED,
        "settlement_id": intent["settlement_id"],
        "item_id": TASK_ID,
        "title": intent["title"],
        "objective": intent["objective"],
        "success": intent["success"],
        "status": intent["event_status"],
        "summary": intent["summary"],
        "review_status": intent["review_status"],
        "final_submission_certified": intent["success"],
        "overall_complete": intent["success"],
        "running_owner": intent["expected_owner"],
        "continuation_terminal_sha256": intent["continuation_terminal_sha256"],
        "consumer_authority_sha256": sha256_file(AUTHORITY_PATH),
    }


def event_already_emitted(settlement_id: str) -> bool:
    path = LIFE_DIR / "events.jsonl"
    if not path.exists():
        return False
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 1024 * 1024))
        return settlement_id.encode("ascii") in stream.read()


def commit_settlement(backlog: Backlog, intent: dict[str, Any]) -> None:
    with backlog._locked():
        items = backlog._load()
        item = next((candidate for candidate in items if candidate.id == TASK_ID), None)
        require(item is not None, "canonical task is absent")
        require(item.status == "running", "canonical task changed during final settlement")
        require(item.running_owner == intent["expected_owner"], "canonical task owner changed during final settlement")
        item.status = intent["backlog_status"]
        item.finished_ts = float(intent["finished_ts"])
        item.last_error = str(intent["last_error"])
        item.outcome = dict(intent["outcome"])
        backlog._save(items)


def finalize_settlement(intent: dict[str, Any]) -> None:
    if not event_already_emitted(intent["settlement_id"]):
        emit(settlement_event(intent))
    receipt = STATE_DIR / "native-settlement.json"
    if not receipt.exists():
        write_atomic_create(
            receipt,
            {
                "schema": "ace2-prompt-recovery-native-settlement-v1",
                "recorded_at_utc": utc_now(),
                **settlement_event(intent),
            },
        )


def recover_pending_settlement(backlog: Backlog) -> bool:
    intent_path = STATE_DIR / "settlement-intent.json"
    if not intent_path.exists() or (STATE_DIR / "native-settlement.json").exists():
        return False
    intent = load_json(intent_path)
    item = task(backlog)
    require(item.status == intent["backlog_status"], "partial settlement backlog status differs")
    require(item.running_owner == intent["expected_owner"], "partial settlement owner differs")
    require(item.outcome == intent["outcome"], "partial settlement outcome differs")
    require(item.last_error == intent["last_error"], "partial settlement error differs")
    finalize_settlement(intent)
    return True


def settle(backlog: Backlog, terminal: dict[str, Any]) -> None:
    item = task(backlog)
    require(item.status == "running", "canonical task cannot be settled from current status")
    if terminal.get("status") == "BLOCKED_NO_FURTHER_LAUNCH":
        require(terminal.get("case2_rtl_execution_count") == 0 and terminal.get("case2_replay") is False, "blocked terminal replay fields differ")
        if (STATE_DIR / "owner-transfer-committed.json").exists():
            validate_committed_transfer(backlog)
            expected_owner = NEW_OWNER
        else:
            require(
                not (CONTINUATION_STATE / "adoption.json").exists()
                and not (CONTINUATION_STATE / "continuation-owner.json").exists(),
                "blocked settlement lacks committed post-adoption owner transfer",
            )
            require(item.running_owner == OLD_OWNER, "pre-adoption blocked task owner differs")
            expected_owner = OLD_OWNER
        outcome = {
            "native_consumer": True,
            "continuation_terminal_sha256": sha256_file(CONTINUATION_STATE / "continuation-terminal.json"),
            "technical_blocker": terminal.get("error", "continuation blocked"),
            "final_l2_performed": False,
        }
        success = False
        backlog_status = "failed"
        event_status = "blocked"
        summary = str(outcome["technical_blocker"])
        review_status = "not_run_blocked_terminal"
        last_error = summary
    else:
        require(terminal.get("status") == "PASS_PENDING_FINAL_INDEPENDENT_L2", "continuation terminal status differs")
        require(terminal.get("case2_rtl_execution_count") == 0, "case2 RTL execution count differs")
        require(terminal.get("case2_oracle_execution_count") == 1, "case2 oracle count differs")
        require(terminal.get("case3_rtl_execution_count") == 1, "case3 RTL count differs")
        require(terminal.get("case3_oracle_execution_count") == 1, "case3 oracle count differs")
        decision = reviewer_decision(terminal, item)
        outcome = {
            "native_consumer": True,
            "continuation_terminal_sha256": sha256_file(CONTINUATION_STATE / "continuation-terminal.json"),
            "final_l2_review_sha256": sha256_file(STATE_DIR / "final-l2-review.json"),
            "review_status": decision.status,
            "review_reason": decision.reason,
            "final_submission_certified": decision.status == "done",
        }
        if decision.status == "done":
            success = True
            backlog_status = "done"
            event_status = "done"
            summary = decision.reason
            last_error = ""
        else:
            success = False
            backlog_status = "failed"
            event_status = "blocked"
            summary = decision.reason
            last_error = decision.reason
        review_status = decision.status
    if terminal.get("status") == "PASS_PENDING_FINAL_INDEPENDENT_L2":
        expected_owner = NEW_OWNER
    require(expected_owner in {OLD_OWNER, NEW_OWNER}, "terminal settlement owner is not trusted")
    intent = {
        "schema": "ace2-prompt-recovery-native-settlement-intent-v1",
        "settlement_id": hashlib.sha256(
            (
                TASK_ID
                + sha256_file(CONTINUATION_STATE / "continuation-terminal.json")
                + review_status
            ).encode("ascii")
        ).hexdigest(),
        "task_id": TASK_ID,
        "title": item.title,
        "objective": item.objective,
        "success": success,
        "backlog_status": backlog_status,
        "event_status": event_status,
        "summary": summary,
        "review_status": review_status,
        "expected_owner": expected_owner,
        "finished_ts": time.time(),
        "last_error": last_error,
        "outcome": outcome,
        "continuation_terminal_sha256": sha256_file(CONTINUATION_STATE / "continuation-terminal.json"),
    }
    write_atomic_create(STATE_DIR / "settlement-intent.json", intent)
    commit_settlement(backlog, intent)
    finalize_settlement(intent)


def settle_post_exit_blocked(backlog: Backlog, terminal_path: Path) -> bool:
    if not terminal_path.exists():
        return False
    terminal = load_json(terminal_path)
    if terminal.get("status") != "BLOCKED_NO_FURTHER_LAUNCH":
        return False
    observer = observer_identity()
    if identity_live(observer):
        return False
    adoption_path = CONTINUATION_STATE / "adoption.json"
    owner_path = CONTINUATION_STATE / "continuation-owner.json"
    require(adoption_path.exists() == owner_path.exists(), "partial continuation ownership publication")
    if adoption_path.exists():
        validate_adoption_records()
        if not (STATE_DIR / "owner-transfer-committed.json").exists():
            write_atomic_create(
                STATE_DIR / "consumer-blocked.json",
                {
                    "schema": "ace2-prompt-recovery-native-consumer-blocked-v1",
                    "status": "BLOCKED_MISSED_LIVE_OWNER_TRANSFER",
                    "task_id": TASK_ID,
                    "technical_blocker": (
                        "continuation exited after adoption before native owner "
                        "transfer could verify the live owner.lock holder"
                    ),
                    "terminal_sha256": sha256_file(terminal_path),
                    "recorded_at_utc": utc_now(),
                },
            )
            emit(
                {
                    "type": "life.mission.external_consumer_blocked",
                    "item_id": TASK_ID,
                    "status": "BLOCKED_MISSED_LIVE_OWNER_TRANSFER",
                    "terminal_sha256": sha256_file(terminal_path),
                }
            )
            return True
        validate_committed_transfer(backlog)
    else:
        if identity_live(
            {
                **EXPECTED_ORIGINAL_OWNER,
                "session_id": EXPECTED_ORIGINAL_OWNER["pid"],
            }
        ):
            return False
    if not (adoption_path.exists() and not (STATE_DIR / "owner-transfer-committed.json").exists()):
        settle(backlog, terminal)
    return True


def wait_for_parent_receipt(path: Path, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    require(path.is_file(), "parent consumer process receipt is absent")


def consume() -> int:
    auth = authority()
    expected_argv = auth["worker_argv"]
    current = process_identity(os.getpid())
    require(current["argv"] == expected_argv, "consumer worker argv differs")
    require(current["session_id"] == current["pid"], "consumer is not detached")
    receipt_path = STATE_DIR / "consumer-process.json"
    wait_for_parent_receipt(receipt_path)
    receipt = load_json(receipt_path)
    for field in ("pid", "start_ticks", "session_id", "argv", "cwd"):
        require(receipt.get(field) == current.get(field), f"parent consumer {field} differs")
    write_atomic_create(
        STATE_DIR / "consumer-ready.json",
        {
            "schema": "ace2-prompt-recovery-native-consumer-v1",
            "status": "WAITING_FOR_ATOMIC_ADOPTION_OR_TERMINAL",
            "authority_sha256": sha256_file(AUTHORITY_PATH),
            "recorded_at_utc": utc_now(),
            **current,
        },
    )
    backlog = LifeMemory.open(LIFE_DIR).backlog
    if recover_pending_settlement(backlog):
        return 0
    next_heartbeat = 0.0
    transferred = (STATE_DIR / "owner-transfer-committed.json").exists()
    while True:
        authority()
        terminal_path = CONTINUATION_STATE / "continuation-terminal.json"
        adoption_path = CONTINUATION_STATE / "adoption.json"
        if settle_post_exit_blocked(backlog, terminal_path):
            return 0
        transferred = observe_owner_transition(backlog, transferred)
        if terminal_path.exists():
            terminal = load_json(terminal_path)
            if terminal.get("status") != "BLOCKED_NO_FURTHER_LAUNCH":
                require(
                    terminal.get("status") == "PASS_PENDING_FINAL_INDEPENDENT_L2",
                    "unknown continuation terminal status",
                )
                require(transferred, "success terminal arrived before native owner transfer")
                observer = observer_identity()
                if not identity_live(observer):
                    settle(backlog, terminal)
                    return 0
        now = time.monotonic()
        if now >= next_heartbeat:
            with (STATE_DIR / "consumer-heartbeats.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "status": "WAITING_FOR_ATOMIC_ADOPTION_OR_TERMINAL",
                            "recorded_at_utc": utc_now(),
                            "task_owner": task(backlog).running_owner,
                            "adoption_present": adoption_path.exists(),
                            "terminal_present": terminal_path.exists(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            next_heartbeat = now + 60
        time.sleep(2)


def launch() -> dict[str, Any]:
    auth = authority()
    require(not STATE_DIR.exists(), "consumer state already exists")
    STATE_DIR.mkdir(parents=True, mode=0o700)
    command = expected_worker_argv()
    console = (STATE_DIR / "consumer-console.log").open("xb")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        deadline = time.monotonic() + 10
        identity: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                identity = process_identity(process.pid)
                if identity["session_id"] == process.pid:
                    break
            except OSError:
                pass
            time.sleep(0.02)
        require(identity is not None and identity["session_id"] == process.pid, "consumer identity unavailable")
        write_atomic_create(
            STATE_DIR / "consumer-process.json",
            {
                "schema": "ace2-prompt-recovery-native-consumer-process-v1",
                "authority_sha256": sha256_file(AUTHORITY_PATH),
                "recorded_at_utc": utc_now(),
                **identity,
            },
        )
        ready = STATE_DIR / "consumer-ready.json"
        while time.monotonic() < deadline and not ready.exists() and process.poll() is None:
            time.sleep(0.05)
        require(ready.exists() and process.poll() is None, "consumer did not become ready")
        return load_json(ready)
    finally:
        console.close()


def status() -> dict[str, Any]:
    settlement = STATE_DIR / "native-settlement.json"
    if settlement.exists():
        return {"state": "TERMINAL", "settlement": load_json(settlement)}
    ready = STATE_DIR / "consumer-ready.json"
    if not ready.exists():
        return {"state": "ABSENT"}
    record = load_json(ready)
    return {
        "state": "WAITING" if identity_live(record) else "CONSUMER_NOT_LIVE",
        "consumer": record,
        "owner_transfer": (STATE_DIR / "owner-transfer-committed.json").exists(),
        "continuation_terminal": (CONTINUATION_STATE / "continuation-terminal.json").exists(),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("launch-consumer")
    sub.add_parser("_consume")
    sub.add_parser("consumer-status")
    return root


def main() -> int:
    command = parser().parse_args().command
    if command == "launch-consumer":
        print(json.dumps(launch(), indent=2, sort_keys=True))
        return 0
    if command == "_consume":
        return consume()
    print(json.dumps(status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
