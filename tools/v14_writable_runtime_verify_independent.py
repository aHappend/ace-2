#!/usr/bin/env python3
"""Inert independent verifier for the Base V14 writable-runtime successor.

This verifier never imports or starts the transport, launcher, controller,
evaluator, or C02 parser.  It executes only AST-extracted canonical JSON reader
functions and transient create-only probes in the empty pre-authority runtime
directories.  Every probe is removed before completion.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v14_shellfree_action_root"
PACKAGE_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V14_SHELLFREE_PACKAGE.json"
LAUNCHER_PATH = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v14.py"
TRANSPORT_PATH = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v14.py"
ACCEPTANCE_PATH = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
ACCEPTANCE_SCHEMA_PATH = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V14_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json"
RUNTIME_CONTAINER = PROJECT_ROOT / "runtime"
RUNTIME_ROOT = RUNTIME_CONTAINER / "qk_gbfp8_head64_granularity_sweep_execution_v14_f1eb0abd"
PRIMARY_ROOT = RUNTIME_ROOT / "primary"
FALLBACK_ROOT = RUNTIME_ROOT / "fallback"
PRIMARY_TERMINAL = PRIMARY_ROOT / "authority/base/first-terminal.json"
FALLBACK_TERMINAL = FALLBACK_ROOT / "first-terminal.json"
RUNTIME_DIRECTORIES = (
    RUNTIME_CONTAINER,
    RUNTIME_ROOT,
    PRIMARY_ROOT,
    PRIMARY_ROOT / "authority",
    PRIMARY_ROOT / "authority/base",
    PRIMARY_ROOT / "result",
    PRIMARY_ROOT / "result/base",
    FALLBACK_ROOT,
)
RUNTIME_FILES = {
    "authority": PRIMARY_ROOT / "authority/base/authority.json",
    "credential": PRIMARY_ROOT / "authority/base/credential.json",
    "ledger": PRIMARY_ROOT / "authority/base/authority-ledger.json",
    "result": PRIMARY_ROOT / "result/base/result.json",
    "first_terminal": PRIMARY_TERMINAL,
    "fallback_terminal": FALLBACK_TERMINAL,
}
V13_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_action_root"
V13_PACKAGE = V13_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V13_SHELLFREE_PACKAGE.json"
V13_ACCEPTANCE = V13_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V13_FAILURE_RECORD = PROJECT_ROOT / ".autors/ace-2/wiki/sources/runs/531ffdcbd930-r002.md"
V9_TERMINAL = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root/live/authority/base/first-terminal.json"
V8_PACKAGE_RAW = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json"
V8_RESULT_SCHEMA_RAW = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json"
CANONICAL_PACKAGE = ACTION_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
CANONICAL_RESULT_SCHEMA = ACTION_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"

ACTION_ID = "ace2:qk-gbfp8-base-v14:execute-once:f1eb0abd:20260814T083636Z"
V13_ACTION_ID = "ace2:qk-gbfp8-base-v13:execute-once:81a8edc1:20260814T075838Z"
V9_ACTION_ID = "ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z"
EXPECTED_PACKAGE_KIND = "qk_gbfp8_head64_granularity_sweep_execution_v14_shellfree_package"
EXPECTED_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXPECTED_UID = os.getuid()
EXPECTED_GID = os.getgid()
EXPECTED_BINDING_IDS = [
    "accepted_v8_handoff",
    "accepted_r2_handoff",
    "accepted_v8_manifest",
    "runtime_package",
    "package_predecessor_raw",
    "result_schema_predecessor_raw",
    "package",
    "result_schema",
    "controller",
    "evaluator",
    "c02_parser",
    "decisive_verifier",
    "v9_first_terminal",
    "v8_authority",
    "v8_consumed_ledger",
    "v8_first_terminal",
    "v7_consumed_ledger",
    "v7_first_terminal",
    "v13_package",
    "v13_static_acceptance",
    "v13_failure_record",
]
EXPECTED_LOCAL_IDS = [
    "authority_schema",
    "credential_schema",
    "ledger_schema",
    "first_terminal_schema",
    "fresh_l2_acceptance_schema",
    "transport",
    "launcher",
    "static_verifier",
    "forensic_report",
]
TARGET_MARKERS = (
    str(TRANSPORT_PATH),
    str(LAUNCHER_PATH),
    "qk_gbfp8_head64_granularity_sweep_controller_static_v8.py",
    "qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py",
    "qk_gbfp8_head64_granularity_sweep_c02_static_v8.py",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_object(raw: bytes, *, canonical: bool) -> dict[str, Any]:
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(VerificationError(token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError("invalid JSON") from error
    require(type(value) is dict and not duplicate, "duplicate key or non-object")
    if canonical:
        require(compact_bytes(value) == raw, "noncanonical bytes")
    return value


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return strict_object(raw, canonical=True), raw


def verify_self_checksum(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    require(type(observed) is str and len(observed) == 64, f"{field} syntax")
    payload = dict(value)
    payload.pop(field)
    require(sha256_bytes(compact_bytes(payload)) == observed, f"{field} mismatch")


def mode_octal(path: Path) -> str:
    return f"{stat.S_IMODE(os.lstat(path).st_mode):04o}"


def process_snapshot() -> dict[str, set[tuple[int, str]]]:
    observed = {marker: set() for marker in TARGET_MARKERS}
    proc = Path("/proc")
    if not proc.is_dir():
        return observed
    for child in proc.iterdir():
        if not child.name.isdigit():
            continue
        try:
            cmdline = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            start_time = (child / "stat").read_text(encoding="ascii").split()[21]
        except (FileNotFoundError, PermissionError, IndexError, OSError):
            continue
        for marker in TARGET_MARKERS:
            if marker in cmdline:
                observed[marker].add((int(child.name), start_time))
    return observed


def reader_namespace(source: str) -> tuple[dict[str, Any], str]:
    tree = ast.parse(source, filename=str(LAUNCHER_PATH))
    wanted_classes = {"ExecutionError"}
    wanted_functions = {"require", "compact_bytes", "read_bytes", "decode_canonical_json", "read_canonical_json"}
    selected: list[ast.stmt] = []
    segments: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted_classes:
            selected.append(copy.deepcopy(node))
            segments.append(ast.get_source_segment(source, node) or "")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            selected.append(copy.deepcopy(node))
            segments.append(ast.get_source_segment(source, node) or "")
    names = {node.name for node in selected if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    require(names == wanted_classes | wanted_functions, "production reader extraction incomplete")
    prefix = ast.parse(
        "from __future__ import annotations\n"
        "import hashlib\n"
        "import json\n"
        "from pathlib import Path\n"
        "from typing import Any\n"
    ).body
    module = ast.Module(body=[*prefix, *selected], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(LAUNCHER_PATH), "exec"), namespace)
    exact_source = "\n\n".join(segments) + "\n"
    return namespace, sha256_bytes(exact_source.encode("utf-8"))


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise VerificationError(f"missing function: {name}")


def verify_preflight_order(source: str) -> dict[str, int]:
    tree = ast.parse(source, filename=str(LAUNCHER_PATH))
    main = function_node(tree, "main")
    preflight = function_node(tree, "_preflight")
    accepted_reader = function_node(tree, "_read_accepted_package_and_result_schema")
    runtime_proof = function_node(tree, "_verify_runtime_namespace_pre_authority")

    preflight_calls = [node for node in ast.walk(preflight) if isinstance(node, ast.Call)]
    require(any(call_name(node) == "_read_accepted_package_and_result_schema" for node in preflight_calls), "preflight omits accepted-byte reader")
    require(any(call_name(node) == "_verify_runtime_namespace_pre_authority" for node in preflight_calls), "preflight omits runtime proof")

    reader_calls = [node for node in ast.walk(accepted_reader) if isinstance(node, ast.Call)]
    canonical_calls = [node for node in reader_calls if call_name(node) == "read_canonical_json"]
    require(len(canonical_calls) == 2, "accepted-byte reader must perform exactly two canonical reads")
    contexts = []
    for node in canonical_calls:
        require(len(node.args) == 2 and isinstance(node.args[1], ast.Constant), "canonical read context shape")
        contexts.append(node.args[1].value)
    require(contexts == ["accepted V8 package", "accepted V8 result schema"], "production canonical read contexts differ")

    probe_calls = [node for node in ast.walk(runtime_proof) if isinstance(node, ast.Call) and call_name(node) == "_probe_create_once"]
    require(len(probe_calls) == 2, "runtime proof must probe primary and fallback terminal parents")
    main_calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    preflight_lines = [node.lineno for node in main_calls if call_name(node) == "_preflight"]
    durable_lines = [node.lineno for node in main_calls if call_name(node) == "_durable_create"]
    spawn_lines = [node.lineno for node in main_calls if call_name(node) in {"posix_spawn", "Popen", "run"}]
    require(len(preflight_lines) == 1 and durable_lines, "main preflight/live-publication structure differs")
    require(preflight_lines[0] < min(durable_lines), "authority creation precedes preflight")
    require(not spawn_lines, "launcher main contains a process start")
    return {
        "accepted_reader_line": accepted_reader.lineno,
        "first_authority_create_line": min(durable_lines),
        "main_preflight_line": preflight_lines[0],
        "runtime_proof_line": runtime_proof.lineno,
    }


def expect_reader_rejects(decode: Any, raw: bytes, label: str) -> None:
    try:
        decode(raw, label)
    except Exception:
        return
    raise VerificationError(f"production reader accepted adversarial bytes: {label}")


def reader_adversarial_checks(namespace: dict[str, Any], package_value: dict[str, Any], package_raw: bytes) -> int:
    decode = namespace["decode_canonical_json"]
    require(decode(package_raw, "accepted V8 package") == package_value, "production decoder changed accepted package")
    reversed_value = {key: package_value[key] for key in reversed(list(package_value))}
    cases = [
        (package_raw[:-1], "missing-final-newline"),
        (package_raw + b"\n", "extra-newline"),
        (b" " + package_raw, "leading-space"),
        (package_raw[:-1] + b" \n", "trailing-space"),
        (package_raw.replace(b"\n", b"\r\n"), "crlf"),
        ((json.dumps(package_value, indent=2, sort_keys=True) + "\n").encode("ascii"), "pretty-print"),
        ((json.dumps(reversed_value, separators=(",", ":"), sort_keys=False) + "\n").encode("ascii"), "unsorted-keys"),
        (b'{"artifact_kind":"x","artifact_kind":"y"}\n', "duplicate-key"),
        (b'{"x":NaN}\n', "nan"),
        (b'{"x":Infinity}\n', "infinity"),
        ('{"x":"caf\u00e9"}\n'.encode("utf-8"), "non-ascii-input"),
        (b'[]\n', "non-object"),
        (package_raw + b"\0", "nul-suffix"),
    ]
    for raw, label in cases:
        expect_reader_rejects(decode, raw, label)
    return len(cases)


def binding_map(package: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    records = package.get(key)
    require(type(records) is list, f"{key} is not a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        require(type(record) is dict and type(record.get("id")) is str, f"{key} record shape")
        require(record["id"] not in result, f"duplicate {key} id")
        result[record["id"]] = record
    return result


def verify_invocation(record: dict[str, Any], expected_program: Path) -> None:
    require(set(record) == {"argv", "command_representation", "cwd", "environment", "invocation_sha256", "shell"}, "invocation fields")
    require(record["command_representation"] == "ARGV_VECTOR_ONLY", "invocation representation")
    require(record["cwd"] == str(ACTION_ROOT), "invocation cwd")
    require(record["environment"] == EXPECTED_ENVIRONMENT, "invocation environment")
    require(record["shell"] is False, "invocation shell")
    argv = record["argv"]
    require(type(argv) is list and len(argv) == 8 and all(type(item) is str for item in argv), "invocation argv vector")
    require(argv[1] == str(expected_program) and argv[-1] == ACTION_ID, "invocation program/action")
    require(not any(item in {"/bin/bash", "/bin/sh", "-c"} for item in argv), "shell wrapper token")
    payload = dict(record)
    observed = payload.pop("invocation_sha256")
    require(sha256_bytes(compact_bytes(payload)) == observed, "invocation checksum")


def expected_runtime_contract() -> dict[str, Any]:
    return {
        "directory_mode_octal": "0700",
        "fallback_terminal": str(FALLBACK_TERMINAL),
        "file_mode_octal": "0400",
        "owner_gid": EXPECTED_GID,
        "owner_uid": EXPECTED_UID,
        "paths": {key: str(path) for key, path in RUNTIME_FILES.items() if key != "fallback_terminal"},
        "precreated_directories": [str(path) for path in RUNTIME_DIRECTORIES],
        "primary_root": str(PRIMARY_ROOT),
        "publication": "O_CREAT|O_EXCL then file fsync then parent-directory fsync",
        "runtime_root": str(RUNTIME_ROOT),
        "static_preparation_only": True,
        "terminal_publication_order": ["primary", "fallback"],
        "writability_proof": "TRANSIENT_O_EXCL_CREATE_SECOND_CREATE_REJECT_UNLINK_FSYNC",
    }


def verify_manifest_contract(package: dict[str, Any], *, files: bool) -> None:
    require(package.get("artifact_kind") == EXPECTED_PACKAGE_KIND, "package kind")
    require(package.get("action_identity") == {
        "future_action_id": ACTION_ID,
        "prior_attempt_action_id": V13_ACTION_ID,
        "prior_terminal_action_id": V9_ACTION_ID,
        "retired_action_ids": [
            "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1610Z",
            "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z",
            "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z",
            V9_ACTION_ID,
            V13_ACTION_ID,
        ],
        "reuse_permitted": False,
    }, "action identity")
    require(package.get("prior_attempt_state") == {
        "action_id": V13_ACTION_ID,
        "evaluator_invocation_count": 0,
        "exit_status": 2,
        "first_terminal_materialized": False,
        "invocation_count_performed": 1,
        "launcher_contract_matched": False,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "reason_code": "IMMUTABLE_ROOT_RUNTIME_NAMESPACE_UNWRITABLE",
        "shell_wrapper_observed": "/bin/bash -c",
    }, "prior V13 attempt state")
    require([item["id"] for item in package.get("canonical_bindings", [])] == EXPECTED_BINDING_IDS, "canonical binding order")
    require([item["id"] for item in package.get("local_artifact_bindings", [])] == EXPECTED_LOCAL_IDS, "local binding order")
    require(package.get("runtime_namespace") == expected_runtime_contract(), "runtime namespace contract")
    require(package.get("production_canonical_preflight_proof", {}).get("runtime_proof_before_authority") is True, "runtime proof ordering declaration")
    require(package.get("production_canonical_preflight_proof", {}).get("target_process_starts_required") == 0, "target-start proof boundary")
    require(package.get("claim_boundary", {}).get("execution_authorized") is False, "execution authority boundary")
    require(package.get("claim_boundary", {}).get("runtime_namespace_precreated_empty") is True, "runtime preparation boundary")
    verify_invocation(package["future_invocation"], TRANSPORT_PATH)
    verify_invocation(package["launcher_invocation"], LAUNCHER_PATH)
    require(package.get("transport_contract", {}).get("direct_exec_api") == "os.posix_spawn", "direct exec API")
    require(package.get("transport_contract", {}).get("shell") is False, "transport shell")
    require(package.get("transport_contract", {}).get("environment_inheritance_permitted") is False, "transport environment inheritance")

    derivation = package.get("accepted_byte_derivation", {})
    require(set(derivation) == {"package", "result_schema"}, "accepted derivation keys")
    expected_derivations = {
        "package": (V8_PACKAGE_RAW, CANONICAL_PACKAGE, "accepted V8 package"),
        "result_schema": (V8_RESULT_SCHEMA_RAW, CANONICAL_RESULT_SCHEMA, "accepted V8 result schema"),
    }
    for key, (source, canonical, context) in expected_derivations.items():
        record = derivation[key]
        require(record.get("source_path") == str(source), f"{key} source path")
        require(record.get("canonical_path") == str(canonical), f"{key} canonical path")
        require(record.get("production_reader_context") == context, f"{key} reader context")
        require(record.get("semantic_json_equal") is True, f"{key} semantic equality")
        if files:
            source_raw = source.read_bytes()
            canonical_raw = canonical.read_bytes()
            require(strict_object(source_raw, canonical=False) == strict_object(canonical_raw, canonical=True), f"{key} semantic derivation")
            require(compact_bytes(strict_object(source_raw, canonical=False)) == canonical_raw, f"{key} canonical derivation")
            require(record.get("source_raw_sha256") == sha256_bytes(source_raw), f"{key} source hash")
            require(record.get("canonical_sha256") == sha256_bytes(canonical_raw), f"{key} canonical hash")

    canonical_bindings = binding_map(package, "canonical_bindings")
    local_bindings = binding_map(package, "local_artifact_bindings")
    require(canonical_bindings["package"]["path"] == str(CANONICAL_PACKAGE), "canonical package binding path")
    require(canonical_bindings["package"]["sha256"] == derivation["package"]["canonical_sha256"], "canonical package binding hash")
    require(canonical_bindings["result_schema"]["path"] == str(CANONICAL_RESULT_SCHEMA), "canonical result-schema binding path")
    require(canonical_bindings["result_schema"]["sha256"] == derivation["result_schema"]["canonical_sha256"], "canonical result-schema binding hash")
    require(canonical_bindings["package_predecessor_raw"]["path"] == str(V8_PACKAGE_RAW), "package predecessor binding path")
    require(canonical_bindings["package_predecessor_raw"]["sha256"] == derivation["package"]["source_raw_sha256"], "package predecessor binding hash")
    require(canonical_bindings["result_schema_predecessor_raw"]["path"] == str(V8_RESULT_SCHEMA_RAW), "result-schema predecessor binding path")
    require(canonical_bindings["result_schema_predecessor_raw"]["sha256"] == derivation["result_schema"]["source_raw_sha256"], "result-schema predecessor binding hash")
    require(canonical_bindings["v13_package"]["path"] == str(V13_PACKAGE), "V13 package binding")
    require(canonical_bindings["v13_static_acceptance"]["path"] == str(V13_ACCEPTANCE), "V13 acceptance binding")
    require(canonical_bindings["v13_failure_record"]["path"] == str(V13_FAILURE_RECORD), "V13 failure binding")
    for record in [*canonical_bindings.values(), *local_bindings.values()]:
        path = Path(record["path"])
        require(path.is_file() and not path.is_symlink(), f"bound file absent: {path}")
        require(record.get("sha256") == sha256_file(path), f"bound hash differs: {path}")
    policy = package.get("static_file_policy", {})
    require("live" in policy.get("forbidden_directories", []), "package live directory not forbidden")
    require(policy.get("static_files_only") is True and policy.get("generated_python_artifacts_permitted") is False, "static policy")


def reseal(package: dict[str, Any]) -> None:
    package.pop("package_content_sha256", None)
    package["package_content_sha256"] = sha256_bytes(compact_bytes(package))


def manifest_adversarial_checks(package: dict[str, Any]) -> int:
    mutations = []

    def add(mutator: Any) -> None:
        candidate = copy.deepcopy(package)
        mutator(candidate)
        reseal(candidate)
        mutations.append(candidate)

    add(lambda value: value["action_identity"].__setitem__("future_action_id", V13_ACTION_ID))
    add(lambda value: value["action_identity"].__setitem__("reuse_permitted", True))
    add(lambda value: value["action_identity"].__setitem__("prior_attempt_action_id", V9_ACTION_ID))
    add(lambda value: value["prior_attempt_state"].__setitem__("exit_status", 0))
    add(lambda value: value["prior_attempt_state"].__setitem__("first_terminal_materialized", True))
    add(lambda value: value["prior_attempt_state"].__setitem__("payload_open_count", 1))
    add(lambda value: value["prior_attempt_state"].__setitem__("launcher_contract_matched", True))
    add(lambda value: value["canonical_bindings"].reverse())
    add(lambda value: value["canonical_bindings"].pop())
    add(lambda value: value["local_artifact_bindings"][6].__setitem__("sha256", "0" * 64))
    add(lambda value: value["accepted_byte_derivation"]["package"].__setitem__("semantic_json_equal", False))
    add(lambda value: value["accepted_byte_derivation"]["package"].__setitem__("canonical_sha256", "0" * 64))
    add(lambda value: value["production_canonical_preflight_proof"].__setitem__("runtime_proof_before_authority", False))
    add(lambda value: value["production_canonical_preflight_proof"].__setitem__("target_process_starts_required", 1))
    add(lambda value: value["future_invocation"].__setitem__("shell", True))
    add(lambda value: value["future_invocation"].__setitem__("cwd", str(PROJECT_ROOT)))
    add(lambda value: value["future_invocation"].__setitem__("environment", {}))
    add(lambda value: value["future_invocation"]["argv"].__setitem__(1, "/bin/bash"))
    add(lambda value: value["future_invocation"]["argv"].__setitem__(2, "-c"))
    add(lambda value: value["launcher_invocation"].__setitem__("shell", True))
    add(lambda value: value["transport_contract"].__setitem__("direct_exec_api", "subprocess.run"))
    add(lambda value: value["transport_contract"].__setitem__("environment_inheritance_permitted", True))
    add(lambda value: value["runtime_namespace"].__setitem__("runtime_root", str(ACTION_ROOT / "live")))
    add(lambda value: value["runtime_namespace"].__setitem__("primary_root", str(ACTION_ROOT / "live")))
    add(lambda value: value["runtime_namespace"].__setitem__("fallback_terminal", str(PRIMARY_TERMINAL)))
    add(lambda value: value["runtime_namespace"].__setitem__("directory_mode_octal", "0755"))
    add(lambda value: value["runtime_namespace"].__setitem__("owner_uid", EXPECTED_UID + 1))
    add(lambda value: value["runtime_namespace"].__setitem__("owner_gid", EXPECTED_GID + 1))
    add(lambda value: value["runtime_namespace"].__setitem__("terminal_publication_order", ["fallback", "primary"]))
    add(lambda value: value["runtime_namespace"].__setitem__("writability_proof", "ACCESS_W_ONLY"))
    add(lambda value: value["runtime_namespace"]["paths"].__setitem__("authority", str(ACTION_ROOT / "authority.json")))
    add(lambda value: value["runtime_namespace"]["precreated_directories"].pop())
    add(lambda value: value["claim_boundary"].__setitem__("execution_authorized", True))
    add(lambda value: value["claim_boundary"].__setitem__("runtime_namespace_precreated_empty", False))
    add(lambda value: value["static_file_policy"]["forbidden_directories"].remove("live"))
    add(lambda value: value.__setitem__("artifact_kind", "qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_package"))

    for index, candidate in enumerate(mutations, 1):
        try:
            verify_manifest_contract(candidate, files=False)
        except VerificationError:
            continue
        raise VerificationError(f"adversarial manifest mutation accepted: {index}")
    return len(mutations)


def verify_package_inventory(package: dict[str, Any]) -> dict[str, int]:
    root_info = os.lstat(ACTION_ROOT)
    require(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode), "action root absent or symlink")
    require(mode_octal(ACTION_ROOT) == "0555", "action root mode")
    allowed = set(package["static_file_policy"]["allowed_relative_files"])
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in sorted(ACTION_ROOT.rglob("*")):
        relative = path.relative_to(ACTION_ROOT).as_posix()
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink in action root: {relative}")
        if stat.S_ISDIR(info.st_mode):
            observed_directories.add(relative)
            require(mode_octal(path) == "0555", f"directory mode: {relative}")
        else:
            require(stat.S_ISREG(info.st_mode), f"non-regular entry: {relative}")
            observed_files.add(relative)
            require(mode_octal(path) == "0444", f"file mode: {relative}")
            require(not relative.endswith((".pyc", ".pyo")), f"generated Python artifact: {relative}")
    expected = set(allowed)
    if not ACCEPTANCE_PATH.exists():
        expected.remove("review/FRESH_L2_STATIC_ACCEPTANCE.json")
        require("review" not in observed_directories, "pre-review directory unexpectedly exists")
    require(observed_files == expected, f"static file set differs: expected={sorted(expected)} observed={sorted(observed_files)}")
    require("live" not in observed_directories and not os.path.lexists(ACTION_ROOT / "live"), "package-local live namespace exists")
    return {"directory_count": 1 + len(observed_directories), "file_count": len(observed_files)}


def verify_runtime_inventory() -> dict[str, Any]:
    observed_directories: set[Path] = set()
    observed_files: set[Path] = set()
    for path in RUNTIME_ROOT.rglob("*"):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink in runtime tree: {path}")
        if stat.S_ISDIR(info.st_mode):
            observed_directories.add(path)
        else:
            observed_files.add(path)
    require(observed_directories == set(RUNTIME_DIRECTORIES[2:]), "runtime directory inventory")
    require(not observed_files, f"runtime tree is not empty: {sorted(str(path) for path in observed_files)}")
    for path in RUNTIME_DIRECTORIES:
        info = os.lstat(path)
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"runtime directory type: {path}")
        require(stat.S_IMODE(info.st_mode) == 0o700, f"runtime directory mode: {path}")
        require(info.st_uid == EXPECTED_UID and info.st_gid == EXPECTED_GID, f"runtime directory owner: {path}")
    for path in RUNTIME_FILES.values():
        require(not os.path.lexists(path), f"pre-authority runtime file exists: {path}")
    require(ACTION_ROOT not in RUNTIME_ROOT.parents and RUNTIME_ROOT not in ACTION_ROOT.parents, "runtime/package roots overlap")
    return {"directory_count": len(RUNTIME_DIRECTORIES), "file_count": 0, "runtime_root": str(RUNTIME_ROOT)}


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prove_create_once(parent: Path, label: str) -> dict[str, Any]:
    probe = parent / f".v14-independent-{label}-create-once-probe"
    require(not os.path.lexists(probe), f"stale probe: {probe}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    created = False
    second_create_rejected = False
    try:
        descriptor = os.open(probe, flags, 0o400)
        created = True
        try:
            raw = f"{ACTION_ID}:{label}\n".encode("ascii")
            require(os.write(descriptor, raw) == len(raw), f"short probe write: {label}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        info = os.lstat(probe)
        require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o400, f"probe mode: {label}")
        require(info.st_uid == EXPECTED_UID and info.st_gid == EXPECTED_GID, f"probe owner: {label}")
        try:
            duplicate = os.open(probe, flags, 0o400)
        except FileExistsError:
            second_create_rejected = True
        else:
            os.close(duplicate)
            raise VerificationError(f"second create unexpectedly succeeded: {label}")
    finally:
        if created and os.path.lexists(probe):
            os.unlink(probe)
            fsync_directory(parent)
    require(second_create_rejected and not os.path.lexists(probe), f"create-once probe cleanup: {label}")
    return {"parent": str(parent), "second_create_rejected": True, "writable": True}


def verify_v13_retirement() -> dict[str, Any]:
    require(mode_octal(V13_ROOT) == "0555", "V13 root mode")
    require(not os.path.lexists(V13_ROOT / "live"), "V13 live namespace appeared")
    record = V13_FAILURE_RECORD.read_text(encoding="utf-8")
    for text in ("mode-0555", "/bin/bash", "No immutable", "no G8/G4/G2/G1 evaluation ran"):
        require(text in record, f"V13 failure record missing: {text}")
    return {
        "action_id": V13_ACTION_ID,
        "failure_record_sha256": sha256_file(V13_FAILURE_RECORD),
        "package_sha256": sha256_file(V13_PACKAGE),
        "static_acceptance_sha256": sha256_file(V13_ACCEPTANCE),
    }


def verify_acceptance_if_present(package_raw: bytes) -> dict[str, Any] | None:
    schema, _ = load_canonical(ACCEPTANCE_SCHEMA_PATH)
    required_review_fields = {"independent_verifier_rerun_sha256", "reviewer_decision_sha256", "reviewer_thread_id", "target_process_starts"}
    require(required_review_fields <= set(schema.get("required", [])), "acceptance schema omits review bindings")
    require(schema.get("properties", {}).get("target_process_starts") == {"const": 0}, "acceptance schema target-start contract")
    if not ACCEPTANCE_PATH.exists():
        return None
    acceptance, raw = load_canonical(ACCEPTANCE_PATH)
    verify_self_checksum(acceptance, "acceptance_sha256")
    require(acceptance.get("artifact_kind") == "qk_gbfp8_head64_granularity_sweep_execution_v14_shellfree_fresh_l2_static_acceptance", "acceptance kind")
    require(acceptance.get("action_id") == ACTION_ID, "acceptance action")
    require(acceptance.get("decision") == "ACCEPT_STATIC_PACKAGE" and acceptance.get("reviewer_role") == "Fresh-L2", "acceptance decision/reviewer")
    require(acceptance.get("static_acceptance_grants_execution_authority") is False, "acceptance authority boundary")
    require(acceptance.get("execution_package_file_sha256") == sha256_bytes(package_raw), "acceptance package binding")
    require(acceptance.get("target_process_starts") == 0, "acceptance target starts")
    return {"raw_sha256": sha256_bytes(raw), "self_sha256": acceptance["acceptance_sha256"]}


def main() -> int:
    before = process_snapshot()
    package, package_raw = load_canonical(PACKAGE_PATH)
    verify_self_checksum(package, "package_content_sha256")
    verify_manifest_contract(package, files=True)
    package_inventory = verify_package_inventory(package)
    runtime_inventory_before = verify_runtime_inventory()
    v13_retirement = verify_v13_retirement()

    launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
    namespace, reader_source_sha256 = reader_namespace(launcher_source)
    order = verify_preflight_order(launcher_source)
    production_read = namespace["read_canonical_json"]
    accepted_package, accepted_package_raw = production_read(CANONICAL_PACKAGE, "accepted V8 package")
    accepted_schema, accepted_schema_raw = production_read(CANONICAL_RESULT_SCHEMA, "accepted V8 result schema")
    require(accepted_package == strict_object(V8_PACKAGE_RAW.read_bytes(), canonical=False), "production package semantic drift")
    require(accepted_schema == strict_object(V8_RESULT_SCHEMA_RAW.read_bytes(), canonical=False), "production schema semantic drift")

    reader_cases = reader_adversarial_checks(namespace, accepted_package, accepted_package_raw)
    manifest_cases = manifest_adversarial_checks(package)
    primary_probe = prove_create_once(PRIMARY_TERMINAL.parent, "primary-terminal")
    fallback_probe = prove_create_once(FALLBACK_TERMINAL.parent, "fallback-terminal")
    runtime_inventory_after = verify_runtime_inventory()
    acceptance = verify_acceptance_if_present(package_raw)
    after = process_snapshot()
    starts = {marker: sorted(after[marker] - before[marker]) for marker in TARGET_MARKERS if after[marker] - before[marker]}
    require(not starts, f"target process started during verification: {starts}")

    result = {
        "acceptance": acceptance,
        "action_id": ACTION_ID,
        "adversarial_cases": reader_cases + manifest_cases,
        "manifest_adversarial_cases": manifest_cases,
        "manifest_raw_sha256": sha256_bytes(package_raw),
        "package_content_sha256": package["package_content_sha256"],
        "package_inventory": package_inventory,
        "preflight_order": order,
        "production_reader_exact_source_sha256": reader_source_sha256,
        "reader_adversarial_cases": reader_cases,
        "runtime_inventory_after": runtime_inventory_after,
        "runtime_inventory_before": runtime_inventory_before,
        "runtime_probes": {"fallback": fallback_probe, "primary": primary_probe},
        "status": "PASS_V14_WRITABLE_RUNTIME_SUCCESSOR_INDEPENDENT_STATIC_VERIFICATION",
        "target_process_starts": 0,
        "v13_retirement": v13_retirement,
    }
    require(result["adversarial_cases"] >= 35, "insufficient adversarial cases")
    sys.stdout.buffer.write(compact_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
