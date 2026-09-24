#!/usr/bin/env python3
"""Inspect a live Argus project or relay one explicit message to its Manager.

`inspect` is read-only and is the default command. `relay` performs one provider
call only after explicit confirmation and conservative lifecycle admission.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ACTIVE_STATES = {"running", "in_progress", "reviewing", "executing"}
TERMINAL_STATES = {
    "done",
    "failed",
    "blocked",
    "paused_operator",
    "superseded",
    "cancelled",
}
SAFE_DAEMON_ENV_KEYS = {
    "ARGUS_SKILL_BACKEND",
    "ARGUS_SKILL_COPILOT_ACP_MANAGER_TIMEOUT_S",
    "ARGUS_SKILL_ENGINEER_BACKEND",
    "ARGUS_SKILL_ENGINEER_MODEL",
    "ARGUS_SKILL_GLOBAL_DAILY_CAP_USD",
    "ARGUS_SKILL_HOME",
    "ARGUS_SKILL_LIFE_BACKEND",
    "ARGUS_SKILL_MANAGER_BACKEND",
    "ARGUS_SKILL_MANAGER_MODEL",
    "ARGUS_SKILL_PLANNER_BACKEND",
    "ARGUS_SKILL_PLANNER_MODEL",
    "ARGUS_SKILL_PLAN_BACKEND",
    "ARGUS_SKILL_PLAN_MODEL",
    "ARGUS_SKILL_PROJECT_ROOT",
    "ARGUS_SKILL_REQUIRE_INDEPENDENT_REVIEW",
    "ARGUS_SKILL_REVIEWER_BACKEND",
    "ARGUS_SKILL_REVIEWER_MODEL",
    "ARGUS_SKILL_RUNNER_BACKEND",
    "ARGUS_SKILL_SELF_HARD_IDLE_SECONDS",
    "ARGUS_SKILL_SELF_MAINTENANCE",
    "ARGUS_SKILL_SELF_MANAGED_SOURCE",
    "ARGUS_SKILL_SOURCE_ROOT",
    "COPILOT_HOME",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(
    path: Path, *, tolerate_malformed: bool = False
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    if not path.exists():
        return rows, malformed
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_malformed:
                malformed += 1
                continue
            raise ValueError(f"{path}:{number} is malformed JSON")
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        rows.append(value)
    return rows, malformed


def process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, IndexError, ValueError):
        return None


def latest_backlog(state_root: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    rows, _ = read_jsonl(state_root / "backlog.jsonl")
    for row in rows:
        mission_id = row.get("id")
        if isinstance(mission_id, str) and mission_id:
            latest[mission_id] = row
    return sorted(
        latest.values(),
        key=lambda row: float(
            row.get("updated_ts")
            or row.get("finished_ts")
            or row.get("started_ts")
            or row.get("ts")
            or row.get("created_at")
            or 0
        ),
    )


def reviewer_handoff(
    state_root: Path, mission_id: str | None
) -> dict[str, Any] | None:
    if not mission_id:
        return None
    candidates: list[tuple[float, dict[str, Any], Path]] = []
    handoff_root = state_root / "handoffs" / mission_id
    if not handoff_root.is_dir():
        return None
    for path in handoff_root.glob("round-*.json"):
        if path.name.endswith("-engineer.json"):
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if value.get("producer_role") != "reviewer" and not isinstance(
            value.get("review"), dict
        ):
            continue
        candidates.append((float(value.get("created_at") or 0), value, path))
    if not candidates:
        return None
    _, value, path = max(candidates, key=lambda item: item[0])
    review = value.get("review") if isinstance(value.get("review"), dict) else {}
    return {
        "path": str(path),
        "round": value.get("round"),
        "producer_role": value.get("producer_role"),
        "status": review.get("status"),
        "reason": review.get("reason"),
        "created_at": value.get("created_at"),
    }


def registry_summary(workdir: Path) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    live_members: list[dict[str, Any]] = []
    for marker_path in sorted((workdir / ".argus/team").glob("*.json")):
        try:
            marker = read_json(marker_path)
            team_root = Path(marker["team_root"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        pool = read_json(team_root / "pool.json") if (team_root / "pool.json").exists() else {}
        roster = (
            read_json(team_root / "roster.json")
            if (team_root / "roster.json").exists()
            else {}
        )
        members: list[dict[str, Any]] = []
        for raw in roster.get("members", []):
            if not isinstance(raw, dict):
                continue
            pid = raw.get("pid")
            alive = isinstance(pid, int) and process_start_ticks(pid) is not None
            member = {
                "id": raw.get("id"),
                "task_id": raw.get("task_id"),
                "status": raw.get("status"),
                "pid_alive": alive,
            }
            members.append(member)
            if alive and raw.get("status") not in {
                "exited",
                "done",
                "failed",
                "stopped",
                "cancelled",
            }:
                live_members.append({"team_id": marker.get("team_id"), **member})
        markers.append(
            {
                "team_id": marker.get("team_id"),
                "team_root": str(team_root),
                "created_ts": marker.get("created_ts"),
                "pool_state": pool.get("state"),
                "roster_state": roster.get("state"),
                "members": members,
            }
        )
    return {"markers": markers, "live_members": live_members}


def safe_event(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "type",
        "ts",
        "mission_id",
        "status",
        "role",
        "round",
        "run_label",
        "exit_code",
        "turn_completed",
        "turn_failed",
    )
    return {key: row.get(key) for key in keys if key in row}


def resolve_config(path: Path | None) -> tuple[Path, dict[str, Any]]:
    config_path = (
        path.resolve()
        if path
        else (Path(__file__).resolve().parents[1] / "docs/continuation/argus-project.json")
    )
    return config_path, read_json(config_path)


def discover(config: dict[str, Any]) -> dict[str, Any]:
    state_root = Path(config["state_root"])
    status_path = state_root / "daemon.status.json"
    status = read_json(status_path)
    pid = int(status["pid"])
    live_ticks = process_start_ticks(pid)
    runtime = Path(status["runtime"]["source_root"]).resolve()
    workdir = Path(status["project_workdir"]).resolve()
    configured_workdir = Path(config["expected_workdir"]).resolve()
    if workdir != configured_workdir:
        raise RuntimeError(
            f"daemon workdir {workdir} does not match configured {configured_workdir}"
        )
    if not runtime.is_dir():
        raise RuntimeError(f"discovered runtime does not exist: {runtime}")
    backlog = latest_backlog(state_root)
    latest = backlog[-1] if backlog else None
    active = [
        {
            "id": row.get("id"),
            "status": row.get("status"),
            "title": row.get("title"),
        }
        for row in backlog
        if row.get("status") in ACTIVE_STATES
    ]
    events, malformed_events = read_jsonl(
        state_root / "events.jsonl", tolerate_malformed=True
    )
    mission_view_path = state_root / "mission-view.json"
    mission_view = read_json(mission_view_path) if mission_view_path.exists() else {}
    mission = (
        mission_view.get("mission")
        if isinstance(mission_view.get("mission"), dict)
        else {}
    )
    mission_review = (
        mission_view.get("review")
        if isinstance(mission_view.get("review"), dict)
        else {}
    )
    mission_outcome = (
        mission_view.get("outcome")
        if isinstance(mission_view.get("outcome"), dict)
        else {}
    )
    latest_id = latest.get("id") if latest else None
    handoff = reviewer_handoff(state_root, latest_id)
    registry = registry_summary(workdir)
    consistency_errors: list[str] = []
    if latest:
        if mission.get("id") != latest_id:
            consistency_errors.append("mission-view and backlog mission IDs differ")
        if latest.get("status") in ACTIVE_STATES and mission.get("status") not in {
            "working",
            "running",
            "in_progress",
            "reviewing",
            "executing",
        }:
            consistency_errors.append("active backlog disagrees with mission-view")
        if latest.get("status") in TERMINAL_STATES and mission.get("status") in {
            "working",
            "running",
            "in_progress",
            "reviewing",
            "executing",
        }:
            consistency_errors.append("terminal backlog disagrees with mission-view")
        if latest.get("status") == "done" and (
            not handoff or handoff.get("status") != "done"
        ):
            consistency_errors.append("done backlog lacks a normal Reviewer DONE")
        if latest.get("status") == "failed" and not handoff:
            consistency_errors.append("failed backlog lacks a normal Reviewer handoff")
    if registry["live_members"]:
        consistency_errors.append("registered team has a live non-terminal member")
    continuous_path = state_root / "continuous.json"
    continuous = read_json(continuous_path) if continuous_path.exists() else {}
    session_path = workdir / ".manager_session.json"
    session_pointer = read_json(session_path) if session_path.exists() else {}
    return {
        "state_root": state_root,
        "status": status,
        "pid": pid,
        "pid_alive": live_ticks is not None,
        "pid_start_ticks": live_ticks,
        "runtime": runtime,
        "workdir": workdir,
        "backlog": backlog,
        "latest": latest,
        "active": active,
        "events": events,
        "malformed_event_lines": malformed_events,
        "mission": mission,
        "mission_review": mission_review,
        "mission_outcome": mission_outcome,
        "reviewer_handoff": handoff,
        "registry": registry,
        "lifecycle_consistency_errors": consistency_errors,
        "continuous": continuous,
        "manager_session_pointer_present": bool(session_pointer.get("thread_id")),
    }


def inspection(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    found = discover(config)
    status = found["status"]
    backlog = found["backlog"]
    latest = found["latest"]
    latest_status = latest.get("status") if latest else None
    return {
        "schema": "argus-continuation-inspection/v1",
        "read_only": True,
        "observed_at_unix": time.time(),
        "project": config["project"],
        "config": str(config_path),
        "daemon": {
            "pid": found["pid"],
            "pid_alive": found["pid_alive"],
            "pid_start_ticks": found["pid_start_ticks"],
            "started_at_iso": status.get("started_at_iso"),
            "backend": status.get("backend"),
            "life_backend": status.get("life_backend"),
            "runtime_source": str(found["runtime"]),
            "runtime_release": status.get("runtime", {}).get("release_id"),
            "runtime_source_matches_config": status.get("runtime", {}).get(
                "source_root_matches_config"
            ),
        },
        "continuous": {
            key: found["continuous"].get(key)
            for key in ("enabled", "generation", "updated_at", "reason")
            if key in found["continuous"]
        },
        "manager_session_pointer_present": found["manager_session_pointer_present"],
        "current_claim": (
            {
                "mission_id": latest.get("id"),
                "attempt": latest.get("attempt"),
                "status": latest_status,
                "running_owner": latest.get("running_owner"),
                "started_ts": latest.get("started_ts"),
                "finished_ts": latest.get("finished_ts"),
                "authorization_id": latest.get("authorization_id"),
                "active": latest_status in ACTIVE_STATES,
            }
            if latest
            else None
        ),
        "mission_view": {
            "mission": {
                key: found["mission"].get(key)
                for key in ("id", "title", "status", "started_at", "completed_at")
                if key in found["mission"]
            },
            "review": {
                key: found["mission_review"].get(key)
                for key in ("status", "reason", "source", "rejected_attempts")
                if key in found["mission_review"]
            },
            "outcome": found["mission_outcome"],
        },
        "normal_reviewer_handoff": found["reviewer_handoff"],
        "native_terminal": (
            {
                "mission_id": latest.get("id"),
                "status": latest_status,
                "finished_ts": latest.get("finished_ts"),
                "last_error": latest.get("last_error"),
                "mission_view_outcome": found["mission_outcome"],
            }
            if latest_status in TERMINAL_STATES
            else None
        ),
        "team_registry": found["registry"],
        "lifecycle_consistent": not found["lifecycle_consistency_errors"],
        "lifecycle_consistency_errors": found["lifecycle_consistency_errors"],
        "active_missions": found["active"],
        "recent_backlog": [
            {
                key: row.get(key)
                for key in ("id", "status", "title", "finished_ts", "last_error")
                if key in row
            }
            for row in backlog[-8:]
        ],
        "recent_native_events": [safe_event(row) for row in found["events"][-12:]],
        "malformed_event_lines_ignored": found["malformed_event_lines"],
    }


def daemon_environment(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key_raw, value_raw = item.split(b"=", 1)
        key = key_raw.decode("utf-8", errors="strict")
        if key in SAFE_DAEMON_ENV_KEYS:
            values[key] = value_raw.decode("utf-8", errors="strict")
    missing = {
        "ARGUS_SKILL_HOME",
        "ARGUS_SKILL_SOURCE_ROOT",
        "ARGUS_SKILL_PROJECT_ROOT",
        "ARGUS_SKILL_MANAGER_BACKEND",
        "ARGUS_SKILL_MANAGER_MODEL",
        "COPILOT_HOME",
    } - values.keys()
    if missing:
        raise RuntimeError(f"daemon environment lacks required keys: {sorted(missing)}")
    return values


def relay(
    config: dict[str, Any],
    message_file: Path,
    message_id: str,
    manager_url: str,
    authorization_env: str,
    timeout: float,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", message_id):
        raise ValueError("message ID must be unique-looking and 8-128 safe characters")
    body = message_file.read_text(encoding="utf-8").strip()
    if not body:
        raise ValueError("message file is empty")

    found = discover(config)
    if not found["pid_alive"]:
        raise RuntimeError("daemon PID is not alive; do not create a competing Manager")
    if found["active"]:
        raise RuntimeError(
            "a mission is active; wait for its legitimate terminal before relaying: "
            + json.dumps(found["active"], ensure_ascii=False)
        )
    if found["lifecycle_consistency_errors"]:
        raise RuntimeError(
            "native lifecycle sources disagree; relay is fail-closed: "
            + json.dumps(found["lifecycle_consistency_errors"], ensure_ascii=False)
        )

    env = daemon_environment(found["pid"])
    runtime = found["runtime"]
    if Path(env["ARGUS_SKILL_SOURCE_ROOT"]).resolve() != runtime:
        raise RuntimeError("daemon status and daemon environment disagree on runtime")
    if Path(env["ARGUS_SKILL_PROJECT_ROOT"]).resolve() != found["workdir"]:
        raise RuntimeError("daemon status and daemon environment disagree on workdir")
    if env["ARGUS_SKILL_MANAGER_BACKEND"] != "copilot":
        raise RuntimeError("this utility only admits the verified local Copilot Manager")

    state_root: Path = found["state_root"]
    sid = state_root.name
    marker = f"[continuation-message-id: {message_id}]"
    transcript_path = state_root / "transcript.jsonl"
    if transcript_path.exists() and marker in transcript_path.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("duplicate continuation message ID")

    parsed = urllib.parse.urlsplit(manager_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Manager URL must be a credential-free localhost HTTP(S) origin")
    endpoint = manager_url.rstrip("/") + (
        f"/api/projects/{urllib.parse.quote(sid, safe='')}/message"
    )
    payload = json.dumps({"text": marker + "\n\n" + body}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    authorization = os.environ.get(authorization_env)
    if authorization:
        headers["Authorization"] = authorization

    events_path = state_root / "events.jsonl"
    offset = events_path.stat().st_size
    request = urllib.request.Request(
        endpoint, data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"normal Manager WebAPI rejected relay: HTTP {error.code}") from error
    result_body = json.loads(response_body)
    if not isinstance(result_body, dict):
        raise RuntimeError("normal Manager WebAPI returned a non-object response")

    starts: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    with events_path.open("rb") as stream:
        stream.seek(offset)
        for raw in stream:
            event = json.loads(raw)
            if event.get("type") == "agent.io.start":
                starts.append(event)
            elif event.get("type") == "agent.io.complete":
                completions.append(event)
    if len(starts) != 1:
        raise RuntimeError(
            f"expected exactly one Manager provider start; observed {len(starts)}; "
            "reply was not persisted as success"
        )
    call_id = starts[0].get("call_id")
    matched = [row for row in completions if row.get("call_id") == call_id]
    if len(matched) != 1:
        raise RuntimeError(
            "missing or ambiguous matching provider completion; "
            "reply was not persisted as success"
        )
    result = matched[0]
    route_kind = result_body.get("kind")
    route_completed = (
        route_kind == "task"
        or (
            route_kind == "chat"
            and isinstance(result_body.get("reply"), str)
            and bool(result_body["reply"].strip())
        )
    )
    successful = (
        result.get("backend") == env["ARGUS_SKILL_MANAGER_BACKEND"]
        and result.get("model") == env["ARGUS_SKILL_MANAGER_MODEL"]
        and result.get("exit_code") == 0
        and result.get("turn_completed") is True
        and result.get("turn_failed") is False
        and not result.get("fatal_error")
        and route_completed
    )
    if not successful:
        raise RuntimeError(
            "Manager provider turn was incomplete; response is not relay success"
        )
    marker_count = transcript_path.read_text(encoding="utf-8").count(marker)
    if marker_count != 1:
        raise RuntimeError(
            f"expected one persisted continuation marker; observed {marker_count}"
        )
    item = result_body.get("item") if isinstance(result_body.get("item"), dict) else {}
    return {
        "schema": "argus-manager-relay-result/v1",
        "message_id": message_id,
        "call_id": call_id,
        "backend": result.get("backend"),
        "model": result.get("model"),
        "turn_completed": True,
        "route_kind": route_kind,
        "reply": result_body.get("reply"),
        "dispatch_state": result_body.get("dispatch_state"),
        "item": {
            key: item.get(key)
            for key in ("id", "title", "status")
            if key in item
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="project continuation config")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("inspect", help="read current daemon/backlog/event state")
    relay_parser = subparsers.add_parser(
        "relay", help="send exactly one explicit normal Manager message"
    )
    relay_parser.add_argument("--message-file", type=Path, required=True)
    relay_parser.add_argument("--message-id", required=True)
    relay_parser.add_argument(
        "--manager-url",
        required=True,
        help="credential-free localhost origin of the active Argus WebAPI",
    )
    relay_parser.add_argument(
        "--authorization-env",
        default="ARGUS_WEBAPI_AUTHORIZATION",
        help="environment variable containing the WebAPI Authorization value",
    )
    relay_parser.add_argument("--timeout", type=float, default=900.0)
    relay_parser.add_argument(
        "--confirm-provider-call",
        action="store_true",
        help="required acknowledgement that relay performs one provider call",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command or "inspect"
    config_path, config = resolve_config(args.config)
    if command == "inspect":
        result = inspection(config_path, config)
    else:
        if not args.confirm_provider_call:
            raise SystemExit("relay requires --confirm-provider-call")
        result = relay(
            config,
            args.message_file,
            args.message_id,
            args.manager_url,
            args.authorization_env,
            args.timeout,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
