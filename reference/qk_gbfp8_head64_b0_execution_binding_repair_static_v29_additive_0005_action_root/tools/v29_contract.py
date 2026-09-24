#!/usr/bin/env python3
"""Shared static V29 execution-binding contract. No protected payload access."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_execution_binding_repair_static_v29_additive_0005_action_root"
PACKAGE_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
ACCEPTANCE_PATH = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
EVENT_LOG = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/events.jsonl")
FRAMEWORK_STREAM_LOG = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/agent_io.jsonl")
MISSION_ID = "b0executionbindingrepairv29"
STATIC_ACTION_ID = "ace2:qk-gbfp8-base-v29:b0-execution-binding-repair:c4a9f2d1:additive-0005"
EXECUTION_ACTION_ID = "ace2:qk-gbfp8-base-v29:b0-source-oracle-identity-control-execute-once:c4a9f2d1:additive-0005"
CLAIM_BOUNDARY = "STATIC_ONLY_NO_EXECUTION_AUTHORITY"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
RUNTIME_NAMESPACE = PROJECT_ROOT / "runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v29_c4a9f2d1_additive_0005_once"
FAILURE_SEAL_NAMESPACE = Path(str(RUNTIME_NAMESPACE) + ".failed-start")
AUTHORITY_NAMESPACE = PROJECT_ROOT / "external/qk_gbfp8_head64_b0_source_oracle_identity_control_v29_c4a9f2d1_additive_0005_authority"
MANAGER_CAPSULE = AUTHORITY_NAMESPACE / "manager-admission.json"
OPERATOR_CAPSULE = AUTHORITY_NAMESPACE / "operator-authority.json"
CANDIDATE_REPORT = PROJECT_ROOT / "build/v29-b0-execution-binding-repair-static-0005/fresh-l2-sealed-0005/V29_CANDIDATE_REPORT.json"
REVIEWER_PREFLIGHT_SCRIPT = ACTION_ROOT / "tools/reviewer_preflight_v29.py"
V28_STATIC_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0008_action_root"
V28_STATIC_PACKAGE = V28_STATIC_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_PACKAGE.json"
V28_STATIC_ACCEPTANCE = V28_STATIC_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V28_REJECTED_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_source_oracle_identity_control_execution_v28_additive_0008_action_root"
V28_REJECTED_ARCHIVE = PROJECT_ROOT / "research/archive/rtl/v28-b0-execution-package-rejected-provenance-0001"
V29_REJECTED_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_execution_binding_repair_static_v29_additive_0001_action_root"
V29_REJECTED_PACKAGE = V29_REJECTED_ROOT / "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
V29_REJECTED_0002_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_execution_binding_repair_static_v29_additive_0002_action_root"
V29_REJECTED_0002_PACKAGE = V29_REJECTED_0002_ROOT / "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
V29_REJECTED_0003_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_execution_binding_repair_static_v29_additive_0003_action_root"
V29_REJECTED_0003_PACKAGE = V29_REJECTED_0003_ROOT / "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
V29_REJECTED_0004_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_execution_binding_repair_static_v29_additive_0004_action_root"
V29_REJECTED_0004_PACKAGE = V29_REJECTED_0004_ROOT / "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
V28_STATIC_PACKAGE_SHA256 = "c3052c9d72733a36fa83993aa8fb22e037e00f1f0b52c26eebd6c9664fe78237"
V28_STATIC_CONTENT_SHA256 = "21b763f0f64c0a832781860729d906f8af594afff7c107008b41aebae65377f9"
V28_STATIC_ACCEPTANCE_SHA256 = "4e47a94ee3f6faed3ebba23d5cea499171c51303b4cfbe1d18884ddcf19e46c2"
V28_STATIC_ACCEPTANCE_SELF_SHA256 = "9077821d69bfcb6e63d6c602ee99be14fcae1c9f3ef0a688ccd8f8ea3fe4059a"
V28_REVIEWER_PROVENANCE_SHA256 = "3358f7e154dff28f2403f365e0860d2c37bda08b7157d68fe05b20907c21b667"
V28_REJECTED_PACKAGE_ORIGINAL_SHA256 = "ad4735802b3b5a61bd158f43c28ff7c217f6155aafa544f363864a41c618e15b"
V28_REJECTED_CONTENT_ORIGINAL_SHA256 = "5b01e4e2bd3dc5355817dc527e171b8eda19f82860535d30d98b2f11a36baafc"
V28_REJECTED_TREE_ORIGINAL_SHA256 = "f515aea8f140cb3ba12cfe47954e46e40ab372e0d226fb1a6feaf25d495edb19"
V29_REJECTED_PACKAGE_SHA256 = "a1117d532e632db6eba8891ca4877e812a3b0fa35f377694d6bbbb269419b5b9"
V29_REJECTED_CONTENT_SHA256 = "54dff557a45d74748b066a252442f121fb50548d2c9932990793168831c0103f"
V29_REJECTED_TREE_SHA256 = "476f06bad6332d2a35c25a3bd8b96f1b9eb39e1b903632c52012caa46fba7ed9"
V29_REJECTED_0002_PACKAGE_SHA256 = "45cd17b76db35cc45866fdff59491b02ec5d92c44d2696b5ef1f89d9d2ac7f59"
V29_REJECTED_0002_CONTENT_SHA256 = "418bdf48de50906378bd42919a2c772969638edff3eda96b85e1f0d00d9ed143"
V29_REJECTED_0002_TREE_SHA256 = "892d5cecaa408a38ee93d5010f2ca0aea907713257056490805ae1112b4049ca"
V29_REJECTED_0002_EXECUTION_BINDING_SHA256 = "becbd8b2428b87715a6bf71342b0d7096963f2670e1df4adb53d45d75240e4e0"
V29_REJECTED_0003_PACKAGE_SHA256 = "ea2f8db867a55607bf31c4ad056f84e634e1d5a1365a10a1ade1d145fc2c0f65"
V29_REJECTED_0003_CONTENT_SHA256 = "236e58326da29e46b319b6bad6b42cbf2ba5095ad16408759b87a65534fdd0c6"
V29_REJECTED_0003_TREE_SHA256 = "4fb33ece9cec8493f427027e8079e1be33fbbb04f5e26dee482e09a5efb55f03"
V29_REJECTED_0003_EXECUTION_BINDING_SHA256 = "6c4751fa37092ef8588dc2a0d8224e27dfa9ce371333e772670470004368066b"
V29_REJECTED_0004_PACKAGE_SHA256 = "187f14cdc6f179dabf4ebefc2c674b731033babc4ef7dfd39f9b378fdc07daf1"
V29_REJECTED_0004_CONTENT_SHA256 = "e909315f27603c2313165887254ced0e5d5eea9b1dc49221606045ec5a9dc6a5"
V29_REJECTED_0004_TREE_SHA256 = "a0398ab4e6ad971fd8769a732af97ebcd932b99e416c06f568c8092b313e1643"
V29_REJECTED_0004_EXECUTION_BINDING_SHA256 = "6c21d64b872dbeedb2d2edbff8713aa251c759d19beebae34240329b5bd59ba5"
CODEX_EXECUTABLE = Path("/home/argustest/.vscode-server/extensions/openai.chatgpt-26.810.50856-linux-x64/bin/linux-x86_64/codex")
CODEX_EXECUTABLE_SHA256 = "86a4286b2451c69d4c00ad1772fcee5170013f2924e7d257ad2d4f19290c2a20"
ARGUS_LAUNCHER = Path("/home/argustest/argustest2/argus-skill-latest/.venv/bin/argus-skill")
ARGUS_LAUNCHER_SHA256 = "1644f5693db08b8fd5a39e49a0ff62513203d1b48538c57840b3481cbb9b0a39"
BASH_EXECUTABLE = Path("/usr/bin/bash")
BASH_EXECUTABLE_SHA256 = "59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4"
EXACT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
PACKAGE_NAME = "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
ACCEPTANCE_RELATIVE = "review/FRESH_L2_STATIC_ACCEPTANCE.json"
AUTHORITY_MARKER = "ACE2_B0_V29_AUTHORITY_V1 "
MAX_AUTHORITY_LIFETIME_SECONDS = 900.0
CONCURRENT_ACTIVE_WINDOW_SECONDS = 3600.0
CALL_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
CODEX_ID_PATTERN = re.compile(r"^01[0-9a-f-]{34}$")


class ContractError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ContractError(code, detail)


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path: Path) -> str:
    return sha256_bytes((str(path) + "\n").encode("utf-8"))


def reviewer_preflight_command() -> str:
    environment = [f"{key}={value}" for key, value in EXACT_ENVIRONMENT.items()]
    argv = [
        "env",
        "-i",
        f"ARGUS_SKILL_AGENT_IO_LOG={EVENT_LOG}",
        *environment,
        str(INTERPRETER),
        "-B",
        str(REVIEWER_PREFLIGHT_SCRIPT),
    ]
    return shlex.join(argv) + "; status=$?; exit $status"


def reviewer_acceptance_creator_command() -> str:
    environment = [f"{key}={value}" for key, value in EXACT_ENVIRONMENT.items()]
    argv = [
        "env",
        "-i",
        f"ARGUS_SKILL_AGENT_IO_LOG={EVENT_LOG}",
        *environment,
        str(INTERPRETER),
        "-B",
        str(ACTION_ROOT / "tools/create_fresh_l2_acceptance_v29.py"),
        "--candidate-report",
        str(CANDIDATE_REPORT),
    ]
    return shlex.join(argv) + "; status=$?; exit $status"


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, "NONCANONICAL_JSON", str(path))
    return value, raw


def add_self_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(value)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    require(type(observed) is str, "SELF_HASH_ABSENT", field)
    unhashed = dict(value)
    unhashed.pop(field)
    require(observed == sha256_bytes(compact_bytes(unhashed)), "SELF_HASH_MISMATCH", field)


def execution_binding() -> dict[str, Any]:
    wrapper = ACTION_ROOT / "tools/future_execute_once_wrapper_v29.py"
    argv = [
        str(INTERPRETER),
        "-B",
        str(wrapper),
        "--mode",
        "production",
        "--package",
        str(PACKAGE_PATH),
        "--acceptance",
        str(ACCEPTANCE_PATH),
        "--manager-capsule",
        str(MANAGER_CAPSULE),
        "--operator-capsule",
        str(OPERATOR_CAPSULE),
        "--runtime",
        str(RUNTIME_NAMESPACE),
        "--irreversible-action-id",
        EXECUTION_ACTION_ID,
    ]
    value = {
        "argv": argv,
        "cwd": str(ACTION_ROOT),
        "environment": EXACT_ENVIRONMENT,
        "external_authority_namespace": str(AUTHORITY_NAMESPACE),
        "failure_seal_namespace": str(FAILURE_SEAL_NAMESPACE),
        "future_execution_action_id": EXECUTION_ACTION_ID,
        "interpreter": {
            "path": str(INTERPRETER),
            "sha256": INTERPRETER_SHA256,
            "version": "3.13.5",
        },
        "manager_capsule_path": str(MANAGER_CAPSULE),
        "operator_capsule_path": str(OPERATOR_CAPSULE),
        "runtime_namespace": str(RUNTIME_NAMESPACE),
        "shell": False,
    }
    value["framework_command"] = shlex.join(argv) + "; status=$?; exit $status"
    value["argv_sha256"] = sha256_bytes(compact_bytes(argv))
    value["cwd_sha256"] = sha256_bytes((value["cwd"] + "\n").encode("utf-8"))
    value["environment_sha256"] = sha256_bytes(compact_bytes(EXACT_ENVIRONMENT))
    value["runtime_namespace_sha256"] = path_sha256(RUNTIME_NAMESPACE)
    value["authority_namespace_sha256"] = path_sha256(AUTHORITY_NAMESPACE)
    value["execution_binding_sha256"] = sha256_bytes(compact_bytes(value))
    return value


def future_execution_framework_command() -> str:
    return execution_binding()["framework_command"]


def expected_claim(package: dict[str, Any], role: str, nonce: str, issued_at_ts: float, fresh_until_ts: float) -> dict[str, Any]:
    binding = package["future_execution_binding"]
    decision = "ADMIT_B0_ONCE" if role == "MANAGER" else "AUTHORIZE_B0_ONCE"
    value = {
        "argv_sha256": binding["argv_sha256"],
        "authority_namespace_sha256": binding["authority_namespace_sha256"],
        "cwd_sha256": binding["cwd_sha256"],
        "decision": decision,
        "environment_sha256": binding["environment_sha256"],
        "execution_action_id": EXECUTION_ACTION_ID,
        "execution_binding_sha256": binding["execution_binding_sha256"],
        "fresh_until_ts": fresh_until_ts,
        "interpreter_path": str(INTERPRETER),
        "interpreter_sha256": INTERPRETER_SHA256,
        "issued_at_ts": issued_at_ts,
        "mission_id": MISSION_ID,
        "nonce": nonce,
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_file(PACKAGE_PATH),
        "runtime_namespace_sha256": binding["runtime_namespace_sha256"],
        "static_action_id": STATIC_ACTION_ID,
    }
    value["claim_sha256"] = sha256_bytes(compact_bytes(value))
    return value


def authority_event_text(claim: dict[str, Any]) -> str:
    return AUTHORITY_MARKER + compact_bytes(claim).decode("ascii").rstrip("\n")


def _event_records(raw: bytes) -> list[dict[str, Any]]:
    require(raw.endswith(b"\n"), "EVENT_LOG_TRUNCATED", "missing final newline")
    records: list[dict[str, Any]] = []
    offset = 0
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        require(line not in {b"", b"\n"}, "EVENT_LOG_BLANK_LINE", str(line_number))
        try:
            event = json.loads(line.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("EVENT_LOG_INVALID_JSON", f"line {line_number}: {error}") from error
        require(type(event) is dict, "EVENT_LOG_NON_OBJECT", str(line_number))
        records.append(
            {
                "end": offset + len(line),
                "event": event,
                "line": line,
                "line_number": line_number,
                "start": offset,
            }
        )
        offset += len(line)
    return records


def _proc_stat(pid: int) -> tuple[int, int]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as error:
        raise ContractError("FRAMEWORK_PROCESS_UNAVAILABLE", str(pid)) from error
    try:
        fields = raw.rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[19])
    except (IndexError, ValueError) as error:
        raise ContractError("FRAMEWORK_PROCESS_STAT", str(pid)) from error


def _proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as error:
        raise ContractError("FRAMEWORK_PROCESS_CMDLINE", str(pid)) from error
    return [item.decode("utf-8", "strict") for item in raw.split(b"\0") if item]


def _proc_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as error:
        raise ContractError("FRAMEWORK_PROCESS_ENVIRON", str(pid)) from error
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode("utf-8", "strict")] = value.decode("utf-8", "strict")
    return result


def _proc_executable(pid: int) -> Path:
    try:
        return Path(os.readlink(f"/proc/{pid}/exe"))
    except OSError as error:
        raise ContractError("FRAMEWORK_PROCESS_EXECUTABLE", str(pid)) from error


def _process_start_epoch(start_ticks: int) -> float:
    try:
        boot_line = next(line for line in Path("/proc/stat").read_text(encoding="ascii").splitlines() if line.startswith("btime "))
        boot_time = int(boot_line.split()[1])
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    except (OSError, StopIteration, ValueError) as error:
        raise ContractError("FRAMEWORK_BOOT_TIME", str(error)) from error
    return boot_time + (start_ticks / ticks_per_second)


def _ancestor_chain(start_pid: int | None = None) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    pid = os.getpid() if start_pid is None else start_pid
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        parent_pid, start_ticks = _proc_stat(pid)
        chain.append(
            {
                "cmdline": _proc_cmdline(pid),
                "environment": _proc_environ(pid),
                "executable": _proc_executable(pid),
                "parent_pid": parent_pid,
                "pid": pid,
                "start_ticks": start_ticks,
            }
        )
        pid = parent_pid
    return chain


def _record_fields(prefix: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_event_end_byte": record["end"],
        f"{prefix}_event_line_number": record["line_number"],
        f"{prefix}_event_sha256": sha256_bytes(record["line"]),
        f"{prefix}_event_start_byte": record["start"],
    }


def _nested_stream_event(record: dict[str, Any]) -> dict[str, Any] | None:
    outer = record["event"]
    if outer.get("type") != "agent.io.stream" or outer.get("stream") != "stdout":
        return None
    line = outer.get("line")
    if type(line) is not str:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if type(value) is dict else None


def attest_live_framework_call(
    expected_run_label: str | None,
    expected_shell_payload: str | None,
) -> dict[str, Any]:
    chain = _ancestor_chain()
    shell_index = next(
        (
            index
            for index, process in enumerate(chain)
            if process["executable"].resolve() == BASH_EXECUTABLE.resolve()
            and CODEX_ID_PATTERN.fullmatch(process["environment"].get("CODEX_THREAD_ID", "")) is not None
            and CODEX_ID_PATTERN.fullmatch(process["environment"].get("CODEX_SESSION_ID", "")) is not None
        ),
        None,
    )
    require(shell_index is not None, "FRAMEWORK_TOOL_SHELL_ABSENT", str(os.getpid()))
    shell = chain[shell_index]
    require(sha256_file(BASH_EXECUTABLE) == BASH_EXECUTABLE_SHA256, "FRAMEWORK_BASH_HASH", str(BASH_EXECUTABLE))
    shell_cmdline = shell["cmdline"]
    require(len(shell_cmdline) >= 3 and shell_cmdline[1] in {"-c", "-lc"}, "FRAMEWORK_TOOL_SHELL_CMDLINE", repr(shell_cmdline))
    shell_payload = shell_cmdline[2]
    if expected_shell_payload is not None:
        require(shell_payload == expected_shell_payload, "FRAMEWORK_TOOL_SHELL_PAYLOAD", shell_payload)

    require(shell_index + 2 < len(chain), "FRAMEWORK_PROCESS_ANCESTRY_SHORT", str(shell["pid"]))
    codex = chain[shell_index + 1]
    argus = chain[shell_index + 2]
    require(codex["pid"] == shell["parent_pid"], "FRAMEWORK_CODEX_NOT_DIRECT_PARENT", str(codex["pid"]))
    require(codex["executable"] == CODEX_EXECUTABLE, "FRAMEWORK_CODEX_EXECUTABLE", str(codex["executable"]))
    require(sha256_file(codex["executable"]) == CODEX_EXECUTABLE_SHA256, "FRAMEWORK_CODEX_HASH", str(codex["executable"]))
    require(len(codex["cmdline"]) >= 3 and codex["cmdline"][1:3] == ["exec", "--json"], "FRAMEWORK_CODEX_CMDLINE", repr(codex["cmdline"][:4]))
    require(argus["pid"] == codex["parent_pid"], "FRAMEWORK_ARGUS_NOT_DIRECT_PARENT", str(argus["pid"]))
    require(argus["executable"].resolve() == INTERPRETER.resolve(), "FRAMEWORK_ARGUS_INTERPRETER", str(argus["executable"]))
    require(sha256_file(argus["executable"]) == INTERPRETER_SHA256, "FRAMEWORK_ARGUS_INTERPRETER_HASH", str(argus["executable"]))
    require(len(argus["cmdline"]) >= 2 and Path(argus["cmdline"][1]) == ARGUS_LAUNCHER, "FRAMEWORK_ARGUS_LAUNCHER", repr(argus["cmdline"][:2]))
    require(sha256_file(ARGUS_LAUNCHER) == ARGUS_LAUNCHER_SHA256, "FRAMEWORK_ARGUS_LAUNCHER_HASH", str(ARGUS_LAUNCHER))
    require("--resume" in argus["cmdline"] and "s-c8ae985b" in argus["cmdline"], "FRAMEWORK_ARGUS_PROJECT", repr(argus["cmdline"]))

    thread_id = shell["environment"]["CODEX_THREAD_ID"]
    session_id = shell["environment"]["CODEX_SESSION_ID"]
    event_raw, event_info = _read_event_log_stable(EVENT_LOG)
    stream_raw, stream_info = read_stable_regular_file(FRAMEWORK_STREAM_LOG, "FRAMEWORK_STREAM_LOG")
    event_records = _event_records(event_raw)
    stream_records = _event_records(stream_raw)
    thread_records = [
        record
        for record in stream_records
        if (nested := _nested_stream_event(record)) is not None
        and nested.get("type") == "thread.started"
        and nested.get("thread_id") == thread_id
    ]
    require(len(thread_records) == 1, "FRAMEWORK_THREAD_START_CARDINALITY", str(len(thread_records)))
    thread_record = thread_records[0]
    thread_outer = thread_record["event"]
    call_id = thread_outer.get("call_id")
    run_label = thread_outer.get("run_label")
    require(type(call_id) is str and CALL_ID_PATTERN.fullmatch(call_id) is not None, "FRAMEWORK_CALL_ID", str(call_id))
    require(type(run_label) is str and run_label, "FRAMEWORK_RUN_LABEL", str(run_label))
    if expected_run_label == "manager*":
        require(run_label.startswith("manager"), "FRAMEWORK_RUN_LABEL", run_label)
    elif expected_run_label is not None:
        require(run_label == expected_run_label, "FRAMEWORK_RUN_LABEL", run_label)

    providers = [record for record in event_records if record["event"].get("type") == "provider.request.started" and record["event"].get("call_id") == call_id]
    agents = [record for record in event_records if record["event"].get("type") == "agent.io.start" and record["event"].get("call_id") == call_id]
    completed = [record for record in event_records if record["event"].get("type") in {"provider.request.completed", "agent.io.complete"} and record["event"].get("call_id") == call_id]
    require(len(providers) == 1 and len(agents) == 1, "FRAMEWORK_START_CARDINALITY", f"provider={len(providers)},agent={len(agents)}")
    require(not completed, "FRAMEWORK_COMPLETED_CALL_REPLAY", call_id)
    provider_record = providers[0]
    agent_record = agents[0]
    provider_event = provider_record["event"]
    agent_event = agent_record["event"]
    require(provider_event.get("run_label") == run_label and agent_event.get("run_label") == run_label, "FRAMEWORK_START_RUN_LABEL", call_id)
    require(agent_event.get("working_dir") == str(PROJECT_ROOT), "FRAMEWORK_WORKING_DIRECTORY", str(agent_event.get("working_dir")))
    prompt = agent_event.get("prompt")
    require(type(prompt) is str and MISSION_ID in prompt, "FRAMEWORK_MISSION_PROMPT", call_id)
    start_epoch = _process_start_epoch(codex["start_ticks"])
    for label, event in (("provider", provider_event), ("agent", agent_event), ("thread", thread_outer)):
        timestamp = event.get("ts")
        require(type(timestamp) in {int, float} and abs(float(timestamp) - start_epoch) <= 15.0, "FRAMEWORK_START_TIME", label)
    require(provider_record["line_number"] < agent_record["line_number"], "FRAMEWORK_START_ORDER", call_id)
    require(agent_record["end"] <= len(event_raw) and thread_record["end"] <= len(stream_raw), "FRAMEWORK_RECORD_FRONTIER", call_id)

    call_identity: dict[str, Any] = {
        "agent_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "argus_interpreter_sha256": INTERPRETER_SHA256,
        "argus_launcher_sha256": ARGUS_LAUNCHER_SHA256,
        "argus_pid": argus["pid"],
        "argus_start_ticks": argus["start_ticks"],
        "call_id": call_id,
        "codex_executable_sha256": CODEX_EXECUTABLE_SHA256,
        "codex_pid": codex["pid"],
        "codex_session_id": session_id,
        "codex_start_ticks": codex["start_ticks"],
        "codex_thread_id": thread_id,
        "event_log_device": event_info.st_dev,
        "event_log_inode": event_info.st_ino,
        "event_log_path": str(EVENT_LOG),
        "framework_stream_log_device": stream_info.st_dev,
        "framework_stream_log_inode": stream_info.st_ino,
        "framework_stream_log_path": str(FRAMEWORK_STREAM_LOG),
        "mission_id": MISSION_ID,
        "origin_kind": "LIVE_ARGUS_CODEX_PROCESS_AND_RAW_STREAM_V1",
        "run_label": run_label,
        "schema_version": 1,
        **_record_fields("provider_start", provider_record),
        **_record_fields("agent_start", agent_record),
        **_record_fields("thread_start", thread_record),
    }
    call_identity["call_identity_sha256"] = sha256_bytes(compact_bytes(call_identity))
    command_identity: dict[str, Any] = {
        "bash_executable_sha256": BASH_EXECUTABLE_SHA256,
        "call_identity_sha256": call_identity["call_identity_sha256"],
        "codex_session_id": session_id,
        "codex_thread_id": thread_id,
        "cwd_sha256": path_sha256(Path.cwd()),
        "schema_version": 1,
        "shell_payload_sha256": sha256_bytes((shell_payload + "\n").encode("utf-8")),
        "shell_pid": shell["pid"],
        "shell_start_ticks": shell["start_ticks"],
    }
    command_identity["command_identity_sha256"] = sha256_bytes(compact_bytes(command_identity))
    return {"call": call_identity, "command": command_identity, "prompt": prompt}


def read_stable_regular_file(
    path: Any,
    label: str,
    max_bytes: int | None = None,
) -> tuple[bytes, os.stat_result]:
    display = str(path)
    try:
        path_info = os.lstat(path)
    except OSError as error:
        raise ContractError(f"{label}_PATH_UNAVAILABLE", display) from error
    require(stat.S_ISREG(path_info.st_mode), f"{label}_NOT_REGULAR", display)
    require(not stat.S_ISLNK(path_info.st_mode), f"{label}_SYMLINK", display)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"{label}_OPEN_FAILED", display) from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label}_DESCRIPTOR_NOT_REGULAR", display)
        require(
            (path_info.st_dev, path_info.st_ino) == (before.st_dev, before.st_ino),
            f"{label}_REPLACED_BEFORE_OPEN",
            display,
        )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            require(max_bytes is None or total <= max_bytes, f"{label}_TOO_LARGE", display)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            path_after = os.lstat(path)
        except OSError as error:
            raise ContractError(f"{label}_REPLACED_AFTER_READ", display) from error
        final = os.fstat(descriptor)
        require(stat.S_ISREG(path_after.st_mode), f"{label}_PATH_NOT_REGULAR_AFTER_READ", display)
        require(not stat.S_ISLNK(path_after.st_mode), f"{label}_PATH_SYMLINK_AFTER_READ", display)
        require(
            (path_after.st_dev, path_after.st_ino) == (final.st_dev, final.st_ino),
            f"{label}_REPLACED_AFTER_READ",
            display,
        )
    finally:
        os.close(descriptor)
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        == (path_after.st_dev, path_after.st_ino, path_after.st_size, path_after.st_mtime_ns),
        f"{label}_CONCURRENT_CONTENT_DRIFT",
        display,
    )
    return b"".join(chunks), final


def _read_event_log_stable(path: Path = EVENT_LOG) -> tuple[bytes, os.stat_result]:
    return read_stable_regular_file(path, "EVENT_LOG")


def _verify_bound_source(
    source: dict[str, Any],
    raw_override: bytes | None = None,
    info_override: Any | None = None,
    *,
    exact_live_frontier: bool = False,
) -> tuple[bytes, list[dict[str, Any]]]:
    require(source["source_id"] == "ARGUS_PROJECT_EVENTS_JSONL", "WRONG_EVENT_SOURCE_ID", str(source.get("source_id")))
    require(source["source_path"] == str(EVENT_LOG), "WRONG_EVENT_SOURCE_PATH", str(source.get("source_path")))
    if raw_override is None:
        raw, info = _read_event_log_stable(EVENT_LOG)
    else:
        raw = raw_override
        info = info_override
    require(info is not None, "EVENT_SOURCE_INFO_ABSENT", "source stat")
    require(source["source_device"] == info.st_dev and source["source_inode"] == info.st_ino, "EVENT_LOG_IDENTITY_MISMATCH", "device/inode")
    frontier = source["frontier_byte_count"]
    require(0 < frontier <= len(raw), "EVENT_LOG_TRUNCATED_OR_FRONTIER", str(frontier))
    if exact_live_frontier:
        require(frontier == len(raw), "EVENT_LIVE_FRONTIER_DRIFT", f"bound={frontier},live={len(raw)}")
    prefix = raw[:frontier]
    require(prefix.endswith(b"\n"), "EVENT_FRONTIER_PARTIAL_LINE", str(frontier))
    require(sha256_bytes(prefix) == source["frontier_sha256"], "EVENT_FRONTIER_HASH_MISMATCH", str(frontier))
    prefix_records = _event_records(prefix)
    require(len(prefix_records) == source["frontier_line_count"], "EVENT_FRONTIER_LINE_MISMATCH", str(frontier))
    all_records = _event_records(raw)
    return raw, all_records


def recheck_authority_event_sources(
    manager_source: dict[str, Any],
    operator_source: dict[str, Any],
    live_origin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shared_fields = (
        "source_id",
        "source_path",
        "source_device",
        "source_inode",
        "frontier_byte_count",
        "frontier_line_count",
        "frontier_sha256",
    )
    for field in shared_fields:
        require(
            manager_source.get(field) == operator_source.get(field),
            "AUTHORITY_SOURCE_SNAPSHOT_DISAGREEMENT",
            field,
        )
    if live_origin is None:
        live_origin = attest_live_framework_call("manager*", future_execution_framework_command())
    require(manager_source.get("framework_call") == live_origin["call"], "MANAGER_FRAMEWORK_ORIGIN_MISMATCH", "recheck")
    require(operator_source.get("framework_call") == live_origin["call"], "OPERATOR_FRAMEWORK_ORIGIN_MISMATCH", "recheck")
    raw, info = _read_event_log_stable(EVENT_LOG)
    _verify_bound_source(manager_source, raw, info, exact_live_frontier=True)
    _verify_bound_source(operator_source, raw, info, exact_live_frontier=True)
    return {
        "frontier_byte_count": len(raw),
        "frontier_line_count": len(_event_records(raw)),
        "frontier_sha256": sha256_bytes(raw),
        "source_device": info.st_dev,
        "source_inode": info.st_ino,
    }


def validate_authority_capsule(
    capsule: dict[str, Any],
    role: str,
    package: dict[str, Any],
    now_ts: float,
    consumed_nonces: set[str] | None = None,
    raw_override: bytes | None = None,
    info_override: Any | None = None,
    live_origin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require(role in {"MANAGER", "OPERATOR"}, "AUTHORITY_ROLE", role)
    verify_self_hash(capsule, "capsule_sha256")
    require(capsule.get("role") == role, "AUTHORITY_ROLE_MISMATCH", str(capsule.get("role")))
    claim = capsule.get("claim")
    require(type(claim) is dict, "AUTHORITY_CLAIM_ABSENT", role)
    verify_self_hash(claim, "claim_sha256")
    nonce = claim.get("nonce")
    require(type(nonce) is str and re.fullmatch(r"[0-9a-f]{64}", nonce) is not None, "AUTHORITY_NONCE", str(nonce))
    expected = expected_claim(package, role, nonce, claim["issued_at_ts"], claim["fresh_until_ts"])
    require(claim == expected, "AUTHORITY_BINDING_MISMATCH", role)
    lifetime = claim["fresh_until_ts"] - claim["issued_at_ts"]
    require(0.0 < lifetime <= MAX_AUTHORITY_LIFETIME_SECONDS, "AUTHORITY_FRESHNESS_WINDOW", str(lifetime))
    require(claim["issued_at_ts"] <= now_ts <= claim["fresh_until_ts"], "AUTHORITY_STALE_OR_NOT_YET_VALID", str(now_ts))
    require(consumed_nonces is None or nonce not in consumed_nonces, "AUTHORITY_REPLAY", nonce)
    source = capsule.get("source")
    require(type(source) is dict, "AUTHORITY_SOURCE_ABSENT", role)
    if live_origin is None:
        live_origin = attest_live_framework_call("manager*", future_execution_framework_command())
    framework_call = source.get("framework_call")
    require(type(framework_call) is dict, "AUTHORITY_FRAMEWORK_ORIGIN_REQUIRED", role)
    require(framework_call == live_origin["call"], "AUTHORITY_FRAMEWORK_ORIGIN_MISMATCH", role)
    require(framework_call.get("run_label", "").startswith("manager"), "AUTHORITY_FRAMEWORK_MANAGER_CALL", role)
    raw, records = _verify_bound_source(
        source,
        raw_override,
        info_override,
        exact_live_frontier=True,
    )
    index = source["event_line_number"] - 1
    require(0 <= index < len(records), "AUTHORITY_EVENT_LINE", str(source["event_line_number"]))
    record = records[index]
    require(record["start"] == source["event_start_byte"] and record["end"] == source["event_end_byte"], "AUTHORITY_EVENT_FRONTIER", role)
    require(record["end"] <= source["frontier_byte_count"], "AUTHORITY_EVENT_AFTER_FRONTIER", role)
    require(sha256_bytes(record["line"]) == source["event_hash"], "AUTHORITY_EVENT_HASH_MISMATCH", role)
    event = record["event"]
    expected_type = "life.manager.intent.started" if role == "MANAGER" else "ui.operator"
    require(event.get("type") == source["event_type"] == expected_type, "AUTHORITY_EVENT_TYPE", str(event.get("type")))
    event_ts = event.get("ts")
    require(type(event_ts) in {int, float} and float(event_ts) == float(claim["issued_at_ts"]), "AUTHORITY_EVENT_TIMESTAMP", role)
    expected_text = authority_event_text(claim)
    if role == "MANAGER":
        require(event.get("agent_layer") == source["origin_agent_layer"] == "manager", "MANAGER_ORIGIN_LAYER", str(event.get("agent_layer")))
        require(event.get("source") == source["origin_source"] == "user", "MANAGER_ORIGIN_SOURCE", str(event.get("source")))
        require(event.get("intent_id") == source["intent_id"], "MANAGER_INTENT_ID", str(event.get("intent_id")))
        require(event.get("text") == expected_text and event.get("objective") == expected_text, "MANAGER_EVENT_CLAIM", "text/objective")
    else:
        require(event.get("agent_layer") == source["origin_agent_layer"] == "operator", "OPERATOR_ORIGIN_LAYER", str(event.get("agent_layer")))
        require(event.get("message_id") == source["message_id"], "OPERATOR_MESSAGE_ID", str(event.get("message_id")))
        require(event.get("text") == expected_text, "OPERATOR_EVENT_CLAIM", "text")
    agent_line = framework_call.get("agent_start_event_line_number")
    require(type(agent_line) is int and 1 <= agent_line <= len(records), "AUTHORITY_FRAMEWORK_AGENT_START", role)
    require(record["line_number"] < agent_line, "AUTHORITY_EVENT_NOT_PRECALL", role)
    agent_record = records[agent_line - 1]
    require(sha256_bytes(agent_record["line"]) == framework_call.get("agent_start_event_sha256"), "AUTHORITY_FRAMEWORK_AGENT_HASH", role)
    agent_event = agent_record["event"]
    require(agent_event.get("type") == "agent.io.start" and agent_event.get("call_id") == framework_call.get("call_id"), "AUTHORITY_FRAMEWORK_AGENT_CALL", role)
    prompt = agent_event.get("prompt")
    require(type(prompt) is str and expected_text in prompt, "AUTHORITY_EVENT_NOT_IN_MANAGER_PROMPT", role)
    occurrences = 0
    for later in records:
        later_event = later["event"]
        serialized = json.dumps(later_event, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        if nonce in serialized and later_event.get("type") == expected_type:
            occurrences += 1
            if later["line_number"] > source["event_line_number"]:
                raise ContractError("AUTHORITY_COMPLETED_REPLAY_OR_FRONTIER_DRIFT", f"nonce at line {later['line_number']}")
        if (
            nonce in serialized
            and later["line_number"] > source["event_line_number"]
            and later_event.get("type") in {"life.operator.execution.completed", "life.manager.execution.completed", "life.operator.execution.revoked"}
        ):
            raise ContractError("AUTHORITY_COMPLETED_REPLAY_OR_FRONTIER_DRIFT", f"terminal nonce at line {later['line_number']}")
    require(occurrences == 1, "AUTHORITY_NONCE_OCCURRENCE", str(occurrences))
    return {
        "event_hash": source["event_hash"],
        "event_line_number": source["event_line_number"],
        "frontier_sha256": source["frontier_sha256"],
        "nonce": nonce,
        "role": role,
        "source_byte_count": len(raw),
    }


def attest_current_reviewer_start() -> dict[str, Any]:
    return attest_live_framework_call("reviewer", reviewer_preflight_command())


def reviewer_preflight_commitment(
    package: dict[str, Any],
    report: dict[str, Any],
    report_raw: bytes,
    framework_call: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "action_id": STATIC_ACTION_ID,
        "active_call_id": framework_call["call_id"],
        "artifact_kind": "qk_gbfp8_head64_b0_execution_binding_repair_static_v29_reviewer_preflight_commitment",
        "candidate_action_tree_sha256": report["preacceptance_tree_sha256"],
        "candidate_report_file_sha256": sha256_bytes(report_raw),
        "candidate_report_path_sha256": path_sha256(CANDIDATE_REPORT),
        "candidate_report_sha256": report["report_sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_binding_sha256": package["future_execution_binding"]["execution_binding_sha256"],
        "framework_call_identity_sha256": framework_call["call_identity_sha256"],
        "framework_thread_id": framework_call["codex_thread_id"],
        "mission_id": MISSION_ID,
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_file(PACKAGE_PATH),
        "preflight_command_sha256": sha256_bytes((reviewer_preflight_command() + "\n").encode("utf-8")),
        "preflight_cwd_sha256": path_sha256(ACTION_ROOT),
        "preflight_script_sha256": sha256_file(REVIEWER_PREFLIGHT_SCRIPT),
        "schema_version": 1,
        "status": "PASS_V29_REVIEWER_PREFLIGHT",
    }
    return add_self_hash(value, "commitment_sha256")


def _logged_shell_payload(text: Any) -> str:
    require(type(text) is str, "REVIEWER_PREFLIGHT_COMMAND_TEXT", repr(text))
    try:
        words = shlex.split(text)
    except ValueError as error:
        raise ContractError("REVIEWER_PREFLIGHT_COMMAND_PARSE", str(error)) from error
    require(len(words) == 3 and words[:2] == ["/bin/bash", "-lc"], "REVIEWER_PREFLIGHT_COMMAND_WRAPPER", text)
    return words[2]


def _reviewer_preflight_stream_record(
    records: list[dict[str, Any]],
    framework_call: dict[str, Any],
    commitment: dict[str, Any],
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for record in records:
        outer = record["event"]
        if outer.get("call_id") != framework_call["call_id"] or outer.get("run_label") != "reviewer":
            continue
        nested = _nested_stream_event(record)
        if nested is None or nested.get("type") != "item.completed":
            continue
        item = nested.get("item")
        if type(item) is not dict or item.get("type") != "command_execution":
            continue
        command = item.get("command")
        if type(command) is not str:
            continue
        try:
            payload = _logged_shell_payload(command)
        except ContractError:
            continue
        if payload != reviewer_preflight_command():
            continue
        require(item.get("status") == "completed" and item.get("exit_code") == 0, "REVIEWER_PREFLIGHT_COMMAND_STATUS", framework_call["call_id"])
        output = item.get("aggregated_output")
        require(type(output) is str, "REVIEWER_PREFLIGHT_OUTPUT_ABSENT", framework_call["call_id"])
        try:
            observed_commitment = json.loads(output.strip())
        except json.JSONDecodeError as error:
            raise ContractError("REVIEWER_PREFLIGHT_OUTPUT_JSON", str(error)) from error
        require(type(observed_commitment) is dict and compact_bytes(observed_commitment).decode("ascii").strip() == output.strip(), "REVIEWER_PREFLIGHT_OUTPUT_CANONICAL", framework_call["call_id"])
        require(observed_commitment == commitment, "REVIEWER_PREFLIGHT_OUTPUT_BINDING", framework_call["call_id"])
        matches.append(record)
    require(len(matches) == 1, "REVIEWER_PREFLIGHT_ACTIVE_CALL_CARDINALITY", str(len(matches)))
    return matches[0]


def derive_reviewer_provenance(
    package: dict[str, Any],
    report: dict[str, Any],
    report_raw: bytes,
    live_origin: dict[str, Any],
) -> dict[str, Any]:
    framework_call = live_origin["call"]
    require(framework_call.get("run_label") == "reviewer", "REVIEWER_RUN_LABEL", str(framework_call.get("run_label")))
    commitment = reviewer_preflight_commitment(package, report, report_raw, framework_call)
    event_raw, event_info = _read_event_log_stable(EVENT_LOG)
    stream_raw, stream_info = read_stable_regular_file(FRAMEWORK_STREAM_LOG, "FRAMEWORK_STREAM_LOG")
    event_records = _event_records(event_raw)
    stream_records = _event_records(stream_raw)
    command_record = _reviewer_preflight_stream_record(stream_records, framework_call, commitment)
    call_id = framework_call["call_id"]
    try:
        verify_self_hash(framework_call, "call_identity_sha256")
        verify_self_hash(live_origin["command"], "command_identity_sha256")
    except KeyError as error:
        raise ContractError("REVIEWER_FRAMEWORK_IDENTITY", str(error)) from error

    identity: dict[str, Any] = {
        "action_id": STATIC_ACTION_ID,
        "active_call_id": call_id,
        "actor": "reviewer",
        "agent_layer": "reviewer",
        "acceptance_creator_command": live_origin["command"],
        "authoritative_event_source": "ARGUS_PROJECT_EVENTS_AND_RAW_AGENT_IO_PLUS_LIVE_PROCESS_LINEAGE",
        "event_log_device": event_info.st_dev,
        "event_log_frontier_byte_count": len(event_raw),
        "event_log_frontier_line_count": len(event_records),
        "event_log_frontier_sha256": sha256_bytes(event_raw),
        "event_log_inode": event_info.st_ino,
        "event_log_path": str(EVENT_LOG),
        "framework_call": framework_call,
        "framework_stream_log_device": stream_info.st_dev,
        "framework_stream_log_frontier_byte_count": len(stream_raw),
        "framework_stream_log_frontier_line_count": len(stream_records),
        "framework_stream_log_frontier_sha256": sha256_bytes(stream_raw),
        "framework_stream_log_inode": stream_info.st_ino,
        "framework_stream_log_path": str(FRAMEWORK_STREAM_LOG),
        "mission_id": MISSION_ID,
        "run_label": "reviewer",
        "working_dir_sha256": path_sha256(PROJECT_ROOT),
        "candidate_action_tree_sha256": commitment["candidate_action_tree_sha256"],
        "candidate_report_file_sha256": commitment["candidate_report_file_sha256"],
        "candidate_report_path_sha256": commitment["candidate_report_path_sha256"],
        "candidate_report_sha256": commitment["candidate_report_sha256"],
        "claim_boundary": commitment["claim_boundary"],
        "execution_binding_sha256": commitment["execution_binding_sha256"],
        "package_content_sha256": commitment["package_content_sha256"],
        "package_file_sha256": commitment["package_file_sha256"],
        "reviewer_preflight_command_sha256": commitment["preflight_command_sha256"],
        "reviewer_preflight_commitment_sha256": commitment["commitment_sha256"],
        "reviewer_preflight_cwd_sha256": commitment["preflight_cwd_sha256"],
        "reviewer_preflight_script_sha256": commitment["preflight_script_sha256"],
        **_record_fields("reviewer_preflight_stream", command_record),
    }
    identity["provenance_identity_sha256"] = sha256_bytes(compact_bytes(identity))
    return identity


def attest_current_reviewer(
    package: dict[str, Any],
    report: dict[str, Any],
    report_raw: bytes,
) -> dict[str, Any]:
    live_origin = attest_live_framework_call("reviewer", reviewer_acceptance_creator_command())
    return derive_reviewer_provenance(package, report, report_raw, live_origin)


def verify_reviewer_provenance(
    identity: dict[str, Any],
    package: dict[str, Any],
    report: dict[str, Any],
    report_raw: bytes,
    live_origin: dict[str, Any] | None = None,
) -> None:
    verify_self_hash(identity, "provenance_identity_sha256")
    require(identity.get("event_log_path") == str(EVENT_LOG), "REVIEWER_EVENT_LOG_PATH", str(identity.get("event_log_path")))
    framework_call = identity.get("framework_call")
    creator_command = identity.get("acceptance_creator_command")
    require(type(framework_call) is dict and type(creator_command) is dict, "REVIEWER_FRAMEWORK_IDENTITY_ABSENT", identity.get("active_call_id", ""))
    verify_self_hash(framework_call, "call_identity_sha256")
    verify_self_hash(creator_command, "command_identity_sha256")
    require(identity.get("active_call_id") == framework_call.get("call_id"), "REVIEWER_ACTIVE_CALL_BINDING", str(identity.get("active_call_id")))
    require(framework_call.get("run_label") == "reviewer", "REVIEWER_RUN_LABEL", str(framework_call.get("run_label")))
    commitment = reviewer_preflight_commitment(package, report, report_raw, framework_call)
    bindings = {
        "candidate_action_tree_sha256": "candidate_action_tree_sha256",
        "candidate_report_file_sha256": "candidate_report_file_sha256",
        "candidate_report_path_sha256": "candidate_report_path_sha256",
        "candidate_report_sha256": "candidate_report_sha256",
        "claim_boundary": "claim_boundary",
        "execution_binding_sha256": "execution_binding_sha256",
        "package_content_sha256": "package_content_sha256",
        "package_file_sha256": "package_file_sha256",
        "reviewer_preflight_command_sha256": "preflight_command_sha256",
        "reviewer_preflight_commitment_sha256": "commitment_sha256",
        "reviewer_preflight_cwd_sha256": "preflight_cwd_sha256",
        "reviewer_preflight_script_sha256": "preflight_script_sha256",
    }
    for identity_field, commitment_field in bindings.items():
        require(
            identity.get(identity_field) == commitment[commitment_field],
            "REVIEWER_PREFLIGHT_IDENTITY_BINDING",
            identity_field,
        )
    event_raw, event_info = _read_event_log_stable(EVENT_LOG)
    stream_raw, stream_info = read_stable_regular_file(FRAMEWORK_STREAM_LOG, "FRAMEWORK_STREAM_LOG")
    for label, raw, info, path_prefix in (
        ("EVENT", event_raw, event_info, "event_log"),
        ("STREAM", stream_raw, stream_info, "framework_stream_log"),
    ):
        require(info.st_dev == identity[f"{path_prefix}_device"] and info.st_ino == identity[f"{path_prefix}_inode"], f"REVIEWER_{label}_IDENTITY", path_prefix)
        frontier = identity[f"{path_prefix}_frontier_byte_count"]
        require(0 < frontier <= len(raw), f"REVIEWER_{label}_FRONTIER", str(frontier))
        prefix = raw[:frontier]
        require(prefix.endswith(b"\n") and sha256_bytes(prefix) == identity[f"{path_prefix}_frontier_sha256"], f"REVIEWER_{label}_PREFIX_HASH", str(frontier))
        require(len(_event_records(prefix)) == identity[f"{path_prefix}_frontier_line_count"], f"REVIEWER_{label}_PREFIX_LINES", str(frontier))
    stream_prefix = stream_raw[: identity["framework_stream_log_frontier_byte_count"]]
    command_record = _reviewer_preflight_stream_record(_event_records(stream_prefix), framework_call, commitment)
    for suffix in ("event_line_number", "event_start_byte", "event_end_byte", "event_sha256"):
        observed = _record_fields("reviewer_preflight_stream", command_record)[f"reviewer_preflight_stream_{suffix}"]
        require(observed == identity[f"reviewer_preflight_stream_{suffix}"], "REVIEWER_PREFLIGHT_STREAM_BINDING", suffix)
    require(creator_command.get("call_identity_sha256") == framework_call["call_identity_sha256"], "REVIEWER_CREATOR_CALL_BINDING", framework_call["call_id"])
    require(creator_command.get("shell_payload_sha256") == sha256_bytes((reviewer_acceptance_creator_command() + "\n").encode("utf-8")), "REVIEWER_CREATOR_COMMAND_BINDING", framework_call["call_id"])
    if live_origin is not None:
        require(live_origin.get("call") == framework_call, "REVIEWER_LIVE_CALL_MISMATCH", framework_call["call_id"])


def inventory(action_root: Path, include_acceptance: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(action_root.rglob("*"), key=lambda item: item.relative_to(action_root).as_posix()):
        relative = path.relative_to(action_root).as_posix()
        if relative == ACCEPTANCE_RELATIVE and not include_acceptance:
            continue
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), "INVENTORY_SYMLINK", relative)
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "dir", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append(
                {
                    "kind": "file",
                    "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size": info.st_size,
                }
            )
        else:
            raise ContractError("INVENTORY_SPECIAL_FILE", relative)
    return records


def normalized_preacceptance_inventory(action_root: Path) -> list[dict[str, Any]]:
    records = inventory(action_root, include_acceptance=False)
    for record in records:
        if record["kind"] == "dir" and record["path"] == "review":
            record["mode"] = "0755"
    return records


def preacceptance_tree_sha256(action_root: Path = ACTION_ROOT) -> str:
    return sha256_bytes(compact_bytes(normalized_preacceptance_inventory(action_root)))


def verify_package(package: dict[str, Any], package_raw: bytes) -> None:
    verify_self_hash(package, "package_content_sha256")
    require(package.get("action_identity", {}).get("action_id") == STATIC_ACTION_ID, "PACKAGE_ACTION_ID", "static")
    require(package.get("action_identity", {}).get("root_id") == ACTION_ROOT.name, "PACKAGE_ROOT_ID", ACTION_ROOT.name)
    require(package.get("claim_boundary", {}).get("claim") == CLAIM_BOUNDARY, "PACKAGE_CLAIM_BOUNDARY", "static")
    require(package.get("future_execution_binding") == execution_binding(), "PACKAGE_EXECUTION_BINDING", "exact binding")
    require(package["predecessor_preservation"]["accepted_v28_static"]["package_file_sha256"] == V28_STATIC_PACKAGE_SHA256, "PACKAGE_V28_STATIC_HASH", "package")
    require(package["predecessor_preservation"]["rejected_v28_execution"]["accepted_lineage"] is False, "PACKAGE_REJECTED_LINEAGE", "V28 execution")
    require(package["predecessor_preservation"]["rejected_v28_execution"]["binding_status"] == "EXPLICITLY_RETIRED_PERMANENTLY_REJECTED_PLANNER_MUTATED_FAILURE_EVIDENCE_ONLY", "PACKAGE_REJECTED_RETIREMENT", "V28 execution")
    rejected_v29 = package["predecessor_preservation"]["rejected_v29_additive_0001"]
    require(rejected_v29["accepted_lineage"] is False, "PACKAGE_V29_REJECTED_LINEAGE", "additive-0001")
    require(rejected_v29["package_file_sha256"] == V29_REJECTED_PACKAGE_SHA256, "PACKAGE_V29_REJECTED_PACKAGE_HASH", "additive-0001")
    require(rejected_v29["preacceptance_tree_sha256"] == V29_REJECTED_TREE_SHA256, "PACKAGE_V29_REJECTED_TREE_HASH", "additive-0001")
    rejected_v29_0002 = package["predecessor_preservation"]["rejected_v29_additive_0002"]
    require(rejected_v29_0002["accepted_lineage"] is False, "PACKAGE_V29_0002_REJECTED_LINEAGE", "additive-0002")
    require(rejected_v29_0002["package_file_sha256"] == V29_REJECTED_0002_PACKAGE_SHA256, "PACKAGE_V29_0002_REJECTED_PACKAGE_HASH", "additive-0002")
    require(rejected_v29_0002["preacceptance_tree_sha256"] == V29_REJECTED_0002_TREE_SHA256, "PACKAGE_V29_0002_REJECTED_TREE_HASH", "additive-0002")
    require(rejected_v29_0002["execution_binding_sha256"] == V29_REJECTED_0002_EXECUTION_BINDING_SHA256, "PACKAGE_V29_0002_REJECTED_EXECUTION_BINDING", "additive-0002")
    rejected_v29_0003 = package["predecessor_preservation"]["rejected_v29_additive_0003"]
    require(rejected_v29_0003["accepted_lineage"] is False, "PACKAGE_V29_0003_REJECTED_LINEAGE", "additive-0003")
    require(rejected_v29_0003["package_file_sha256"] == V29_REJECTED_0003_PACKAGE_SHA256, "PACKAGE_V29_0003_REJECTED_PACKAGE_HASH", "additive-0003")
    require(rejected_v29_0003["preacceptance_tree_sha256"] == V29_REJECTED_0003_TREE_SHA256, "PACKAGE_V29_0003_REJECTED_TREE_HASH", "additive-0003")
    require(rejected_v29_0003["execution_binding_sha256"] == V29_REJECTED_0003_EXECUTION_BINDING_SHA256, "PACKAGE_V29_0003_REJECTED_EXECUTION_BINDING", "additive-0003")
    rejected_v29_0004 = package["predecessor_preservation"]["rejected_v29_additive_0004"]
    require(rejected_v29_0004["accepted_lineage"] is False, "PACKAGE_V29_0004_REJECTED_LINEAGE", "additive-0004")
    require(rejected_v29_0004["package_file_sha256"] == V29_REJECTED_0004_PACKAGE_SHA256, "PACKAGE_V29_0004_REJECTED_PACKAGE_HASH", "additive-0004")
    require(rejected_v29_0004["preacceptance_tree_sha256"] == V29_REJECTED_0004_TREE_SHA256, "PACKAGE_V29_0004_REJECTED_TREE_HASH", "additive-0004")
    require(rejected_v29_0004["execution_binding_sha256"] == V29_REJECTED_0004_EXECUTION_BINDING_SHA256, "PACKAGE_V29_0004_REJECTED_EXECUTION_BINDING", "additive-0004")
    require(package["authority_contract"]["event_source"].get("exact_live_frontier_equality_required") is True, "PACKAGE_AUTHORITY_EXACT_LIVE_FRONTIER", "required")
    require(package["authority_contract"]["event_source"].get("post_consumption_recheck_before_envelope_and_protected_start_required") is True, "PACKAGE_AUTHORITY_FINAL_FRONTIER_RECHECK", "required")
    require(package["authority_contract"]["event_source"].get("single_stable_read_shared_by_both_capsule_sources") is True, "PACKAGE_AUTHORITY_SHARED_FINAL_READ", "required")
    require(package["runtime_ordering"].get("claim_liveness_rechecked_before_every_mutation_and_protected_start") is True, "PACKAGE_DUPLICATE_START_GUARD", "required")
    require(package["runtime_ordering"].get("duplicate_start_terminal_materialized_inside_claimed_runtime_and_sibling") is True, "PACKAGE_DUPLICATE_START_TERMINALS", "required")
    require(package["runtime_ordering"].get("consumed_capsule_snapshot_rechecked_before_envelope_and_protected_start") is True, "PACKAGE_CONSUMED_CAPSULE_RECHECK", "required")
    require(package["runtime_ordering"].get("event_log_identity_and_exact_frontier_rechecked_before_envelope_and_protected_start") is True, "PACKAGE_EVENT_LOG_BOUNDARY_RECHECK", "required")
    require(package["b0_control"]["permanently_nonselecting"] is True and package["b0_control"]["quantization_applied"] is False, "PACKAGE_B0_CONTROL", "nonselecting/nonquantizing")
    require(package["fresh_l2_review"]["status"] == "PENDING_INDEPENDENT_REVIEW", "PACKAGE_REVIEW_STATUS", str(package["fresh_l2_review"].get("status")))
    require(package["fresh_l2_review"]["acceptance_materialized_by_engineer"] is False, "PACKAGE_ENGINEER_ACCEPTANCE", "forbidden")
    require(package["fresh_l2_review"]["scheduler_start_binding"] == "IMMUTABLE_MISSION_ONLY", "PACKAGE_REVIEWER_START_BINDING", "scheduler")
    require(package["fresh_l2_review"]["reviewer_preflight_command"] == reviewer_preflight_command(), "PACKAGE_REVIEWER_PREFLIGHT_COMMAND", "exact")
    require(package["fresh_l2_review"]["reviewer_preflight_working_directory"] == str(ACTION_ROOT), "PACKAGE_REVIEWER_PREFLIGHT_CWD", "exact")
    require(package["fresh_l2_review"]["acceptance_creator_command"] == reviewer_acceptance_creator_command(), "PACKAGE_REVIEWER_CREATOR_COMMAND", "exact")
    require(package["fresh_l2_review"]["acceptance_creator_working_directory"] == str(ACTION_ROOT), "PACKAGE_REVIEWER_CREATOR_CWD", "exact")
    require(package["fresh_l2_review"]["origin_lineage"] == "LIVE_ARGUS_CODEX_PROCESS_AND_RAW_STREAM_V1", "PACKAGE_REVIEWER_ORIGIN_LINEAGE", "framework")
    require(package["future_execution_binding"]["framework_command"] == future_execution_framework_command(), "PACKAGE_FUTURE_FRAMEWORK_COMMAND", "exact")
    require(package["package_file_policy"]["package_relative_path"] == PACKAGE_NAME, "PACKAGE_RELATIVE_PATH", PACKAGE_NAME)
    require(package_raw == compact_bytes(package), "PACKAGE_CANONICAL_BYTES", PACKAGE_NAME)


def verify_predecessor_preservation() -> dict[str, Any]:
    require(sha256_file(V28_STATIC_PACKAGE) == V28_STATIC_PACKAGE_SHA256, "V28_STATIC_PACKAGE_DRIFT", str(V28_STATIC_PACKAGE))
    accepted, _ = load_canonical(V28_STATIC_PACKAGE)
    require(accepted.get("package_content_sha256") == V28_STATIC_CONTENT_SHA256, "V28_STATIC_CONTENT_DRIFT", str(V28_STATIC_PACKAGE))
    require(sha256_file(V28_STATIC_ACCEPTANCE) == V28_STATIC_ACCEPTANCE_SHA256, "V28_STATIC_ACCEPTANCE_DRIFT", str(V28_STATIC_ACCEPTANCE))
    acceptance, _ = load_canonical(V28_STATIC_ACCEPTANCE)
    require(acceptance.get("acceptance_sha256") == V28_STATIC_ACCEPTANCE_SELF_SHA256, "V28_STATIC_ACCEPTANCE_SELF_DRIFT", str(V28_STATIC_ACCEPTANCE))
    require(acceptance.get("provenance_identity_sha256") == V28_REVIEWER_PROVENANCE_SHA256, "V28_STATIC_REVIEWER_PROVENANCE_DRIFT", str(V28_STATIC_ACCEPTANCE))
    require(V28_REJECTED_ROOT.is_dir(), "V28_REJECTED_ROOT_ABSENT", str(V28_REJECTED_ROOT))
    require(V28_REJECTED_ARCHIVE.is_dir(), "V28_REJECTED_ARCHIVE_ABSENT", str(V28_REJECTED_ARCHIVE))
    require(V29_REJECTED_ROOT.is_dir(), "V29_REJECTED_ROOT_ABSENT", str(V29_REJECTED_ROOT))
    require(sha256_file(V29_REJECTED_PACKAGE) == V29_REJECTED_PACKAGE_SHA256, "V29_REJECTED_PACKAGE_DRIFT", str(V29_REJECTED_PACKAGE))
    rejected_v29, _ = load_canonical(V29_REJECTED_PACKAGE)
    require(rejected_v29.get("package_content_sha256") == V29_REJECTED_CONTENT_SHA256, "V29_REJECTED_CONTENT_DRIFT", str(V29_REJECTED_PACKAGE))
    require(preacceptance_tree_sha256(V29_REJECTED_ROOT) == V29_REJECTED_TREE_SHA256, "V29_REJECTED_TREE_DRIFT", str(V29_REJECTED_ROOT))
    require(not os.path.lexists(V29_REJECTED_ROOT / ACCEPTANCE_RELATIVE), "V29_REJECTED_ACCEPTANCE_PRESENT", str(V29_REJECTED_ROOT))
    require(V29_REJECTED_0002_ROOT.is_dir(), "V29_REJECTED_0002_ROOT_ABSENT", str(V29_REJECTED_0002_ROOT))
    require(sha256_file(V29_REJECTED_0002_PACKAGE) == V29_REJECTED_0002_PACKAGE_SHA256, "V29_REJECTED_0002_PACKAGE_DRIFT", str(V29_REJECTED_0002_PACKAGE))
    rejected_v29_0002, _ = load_canonical(V29_REJECTED_0002_PACKAGE)
    require(rejected_v29_0002.get("package_content_sha256") == V29_REJECTED_0002_CONTENT_SHA256, "V29_REJECTED_0002_CONTENT_DRIFT", str(V29_REJECTED_0002_PACKAGE))
    require(rejected_v29_0002.get("future_execution_binding", {}).get("execution_binding_sha256") == V29_REJECTED_0002_EXECUTION_BINDING_SHA256, "V29_REJECTED_0002_EXECUTION_BINDING_DRIFT", str(V29_REJECTED_0002_PACKAGE))
    require(preacceptance_tree_sha256(V29_REJECTED_0002_ROOT) == V29_REJECTED_0002_TREE_SHA256, "V29_REJECTED_0002_TREE_DRIFT", str(V29_REJECTED_0002_ROOT))
    require(not os.path.lexists(V29_REJECTED_0002_ROOT / ACCEPTANCE_RELATIVE), "V29_REJECTED_0002_ACCEPTANCE_PRESENT", str(V29_REJECTED_0002_ROOT))
    require(V29_REJECTED_0003_ROOT.is_dir(), "V29_REJECTED_0003_ROOT_ABSENT", str(V29_REJECTED_0003_ROOT))
    require(sha256_file(V29_REJECTED_0003_PACKAGE) == V29_REJECTED_0003_PACKAGE_SHA256, "V29_REJECTED_0003_PACKAGE_DRIFT", str(V29_REJECTED_0003_PACKAGE))
    rejected_v29_0003, _ = load_canonical(V29_REJECTED_0003_PACKAGE)
    require(rejected_v29_0003.get("package_content_sha256") == V29_REJECTED_0003_CONTENT_SHA256, "V29_REJECTED_0003_CONTENT_DRIFT", str(V29_REJECTED_0003_PACKAGE))
    require(rejected_v29_0003.get("future_execution_binding", {}).get("execution_binding_sha256") == V29_REJECTED_0003_EXECUTION_BINDING_SHA256, "V29_REJECTED_0003_EXECUTION_BINDING_DRIFT", str(V29_REJECTED_0003_PACKAGE))
    require(preacceptance_tree_sha256(V29_REJECTED_0003_ROOT) == V29_REJECTED_0003_TREE_SHA256, "V29_REJECTED_0003_TREE_DRIFT", str(V29_REJECTED_0003_ROOT))
    require(not os.path.lexists(V29_REJECTED_0003_ROOT / ACCEPTANCE_RELATIVE), "V29_REJECTED_0003_ACCEPTANCE_PRESENT", str(V29_REJECTED_0003_ROOT))
    require(V29_REJECTED_0004_ROOT.is_dir(), "V29_REJECTED_0004_ROOT_ABSENT", str(V29_REJECTED_0004_ROOT))
    require(sha256_file(V29_REJECTED_0004_PACKAGE) == V29_REJECTED_0004_PACKAGE_SHA256, "V29_REJECTED_0004_PACKAGE_DRIFT", str(V29_REJECTED_0004_PACKAGE))
    rejected_v29_0004, _ = load_canonical(V29_REJECTED_0004_PACKAGE)
    require(rejected_v29_0004.get("package_content_sha256") == V29_REJECTED_0004_CONTENT_SHA256, "V29_REJECTED_0004_CONTENT_DRIFT", str(V29_REJECTED_0004_PACKAGE))
    require(rejected_v29_0004.get("future_execution_binding", {}).get("execution_binding_sha256") == V29_REJECTED_0004_EXECUTION_BINDING_SHA256, "V29_REJECTED_0004_EXECUTION_BINDING_DRIFT", str(V29_REJECTED_0004_PACKAGE))
    require(preacceptance_tree_sha256(V29_REJECTED_0004_ROOT) == V29_REJECTED_0004_TREE_SHA256, "V29_REJECTED_0004_TREE_DRIFT", str(V29_REJECTED_0004_ROOT))
    require(not os.path.lexists(V29_REJECTED_0004_ROOT / ACCEPTANCE_RELATIVE), "V29_REJECTED_0004_ACCEPTANCE_PRESENT", str(V29_REJECTED_0004_ROOT))
    return {
        "accepted_v28_static_package_sha256": V28_STATIC_PACKAGE_SHA256,
        "accepted_v28_static_acceptance_sha256": V28_STATIC_ACCEPTANCE_SHA256,
        "rejected_v28_archive_present": True,
        "rejected_v28_mutated_root_present": True,
        "rejected_v29_additive_0001_package_sha256": V29_REJECTED_PACKAGE_SHA256,
        "rejected_v29_additive_0001_tree_sha256": V29_REJECTED_TREE_SHA256,
        "rejected_v29_additive_0002_package_sha256": V29_REJECTED_0002_PACKAGE_SHA256,
        "rejected_v29_additive_0002_tree_sha256": V29_REJECTED_0002_TREE_SHA256,
        "rejected_v29_additive_0003_package_sha256": V29_REJECTED_0003_PACKAGE_SHA256,
        "rejected_v29_additive_0003_tree_sha256": V29_REJECTED_0003_TREE_SHA256,
        "rejected_v29_additive_0004_package_sha256": V29_REJECTED_0004_PACKAGE_SHA256,
        "rejected_v29_additive_0004_tree_sha256": V29_REJECTED_0004_TREE_SHA256,
    }


def verify_exact_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    expected_files = set(package["package_file_policy"]["allowed_relative_files"])
    preacceptance_file_count = len(expected_files) - 1
    if not allow_acceptance:
        expected_files.remove(ACCEPTANCE_RELATIVE)
        require(not os.path.lexists(ACCEPTANCE_PATH), "ENGINEER_ACCEPTANCE_MATERIALIZED", str(ACCEPTANCE_PATH))
    else:
        require(ACCEPTANCE_PATH.is_file(), "ACCEPTANCE_ABSENT", str(ACCEPTANCE_PATH))
    records = inventory(ACTION_ROOT, include_acceptance=allow_acceptance)
    observed_files = {record["path"] for record in records if record["kind"] == "file"}
    observed_dirs = {record["path"] for record in records if record["kind"] == "dir"}
    require(observed_files == expected_files, "INVENTORY_FILES", str(sorted(observed_files ^ expected_files)))
    require(observed_dirs == set(package["package_file_policy"]["allowed_relative_directories"]), "INVENTORY_DIRECTORIES", str(sorted(observed_dirs)))
    for record in records:
        path = record["path"]
        require("__pycache__" not in path and not path.endswith((".pyc", ".pyo", ".tmp", ".swp", "~")), "INVENTORY_FORBIDDEN", path)
        if record["kind"] == "file":
            require(record["mode"] == "0444", "INVENTORY_FILE_MODE", path)
        elif path == "review" and not allow_acceptance:
            require(record["mode"] == "0755", "INVENTORY_REVIEW_MODE", path)
        else:
            require(record["mode"] == "0555", "INVENTORY_DIRECTORY_MODE", path)
    return {
        "file_count": preacceptance_file_count,
        "directory_count": len(observed_dirs),
        "preacceptance_tree_sha256": preacceptance_tree_sha256(),
    }


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, expected in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file(), "GENERATED_FILE_ABSENT", relative)
        require(path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"], "GENERATED_FILE_DRIFT", relative)
    return {"generated_file_count": len(package["generated_files"])}


def synthetic_rows() -> list[tuple[int, int, int]]:
    return [(index, (index * 17) - 4096, (index % 13) - 6) for index in range(574)]


def classify_rows(source: list[tuple[int, int, int]], oracle: list[tuple[int, int, int]]) -> dict[str, Any]:
    require(len(source) == len(oracle) == 574, "ROW_COUNT", f"{len(source)}/{len(oracle)}")
    mismatches = [index for index, (left, right) in enumerate(zip(source, oracle, strict=True)) if left != right]
    result = {
        "classification": "SOURCE_ORACLE_MATCH" if not mismatches else "SOURCE_ORACLE_MISMATCH",
        "mismatch_count": len(mismatches),
        "mismatch_membership_sha256": sha256_bytes(compact_bytes(mismatches)),
        "row_count": 574,
        "safe_output_boundary": "AGGREGATE_ONLY_NO_RAW_OR_RECONSTRUCTABLE_DATA",
    }
    require(set(result) == {"classification", "mismatch_count", "mismatch_membership_sha256", "row_count", "safe_output_boundary"}, "SAFE_AGGREGATE_OUTPUT", "unexpected fields")
    return result


def transition_model(runtime_preexists: bool, failure_seal_preexists: bool, starts: int, capsules_valid: bool) -> str:
    if failure_seal_preexists:
        return "REJECT_TERMINALLY_SEALED_PRIOR_FAILURE"
    if runtime_preexists:
        return "REJECT_RUNTIME_PREEXISTENCE_AND_SEAL_SIBLING"
    if starts != 1:
        return "REJECT_DUPLICATE_START"
    if not capsules_valid:
        return "CLAIM_RUNTIME_THEN_FAILED_TERMINAL_NO_RETRY"
    return "CLAIM_CONSUME_FREEZE_BEFORE_PROTECTED_BOUNDARY"


def adapter_boundary_model(case: str) -> str:
    mapping = {
        "spawn": "SPAWN",
        "abi": "ABI",
        "return_code": "RETURN_CODE",
        "stdout": "STDOUT",
        "stderr": "STDERR",
        "decode": "DECODE",
        "result_schema": "RESULT_SCHEMA",
        "result_record": "RESULT_RECORD",
        "exception": "EXCEPTION",
        "publication": "PUBLICATION",
    }
    require(case in mapping, "ADAPTER_CASE", case)
    return mapping[case]


def acceptance_unhashed(package: dict[str, Any], report: dict[str, Any], report_raw: bytes, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "action_id": STATIC_ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_b0_execution_binding_repair_static_v29_fresh_l2_acceptance",
        "candidate_action_tree_sha256": report["preacceptance_tree_sha256"],
        "candidate_report_file_sha256": sha256_bytes(report_raw),
        "candidate_report_sha256": report["report_sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
        "creator_provenance": provenance,
        "decision": "ACCEPT_STATIC_PACKAGE",
        "execution_binding_sha256": package["future_execution_binding"]["execution_binding_sha256"],
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_file(PACKAGE_PATH),
        "regression_case_count": report["regression_case_count"],
        "schema_version": 1,
        "static_acceptance_grants_execution_authority": False,
    }


def verify_acceptance(
    package: dict[str, Any],
    acceptance: dict[str, Any],
    acceptance_raw: bytes,
    report: dict[str, Any],
    live_origin: dict[str, Any] | None = None,
) -> None:
    verify_self_hash(acceptance, "acceptance_sha256")
    report_raw = compact_bytes(report)
    expected = add_self_hash(acceptance_unhashed(package, report, report_raw, acceptance["creator_provenance"]), "acceptance_sha256")
    require(acceptance == expected, "ACCEPTANCE_BINDING_MISMATCH", "package/tree/report/provenance")
    require(acceptance_raw == compact_bytes(acceptance), "ACCEPTANCE_CANONICAL_BYTES", str(ACCEPTANCE_PATH))
    verify_reviewer_provenance(acceptance["creator_provenance"], package, report, report_raw, live_origin)


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"}:
        raise ContractError("STATIC_VERIFIER_PROCESS_START", event)
    if event == "open" and args:
        path = args[0]
        if isinstance(path, (str, bytes)):
            text = path.decode("utf-8", "replace") if isinstance(path, bytes) else path
            lowered = text.lower()
            if lowered.endswith((".safetensors", ".npy", ".npz", ".pt", ".pth", ".bin")) or "/protected-payload/" in lowered:
                raise ContractError("STATIC_VERIFIER_PROTECTED_OPEN", text)


def install_audit_hook() -> None:
    sys.addaudithook(audit_hook)
