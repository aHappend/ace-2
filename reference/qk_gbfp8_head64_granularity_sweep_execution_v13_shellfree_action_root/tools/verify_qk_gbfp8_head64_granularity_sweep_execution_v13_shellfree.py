#!/usr/bin/env python3
"""Inert independent verifier for the Base V13 canonical-byte successor.

The verifier never imports or starts the transport, launcher, controller,
evaluator, or C02 parser.  It extracts only the production launcher's exact
canonical-reader function definitions with AST, executes those definitions on
the exact package-bound accepted JSON files, and checks that the read is in the
read-only preflight region before any durable live-state creation.
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
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_action_root"
PACKAGE_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V13_SHELLFREE_PACKAGE.json"
LAUNCHER_PATH = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v13.py"
TRANSPORT_PATH = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v13.py"
ACCEPTANCE_PATH = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
ACCEPTANCE_SCHEMA_PATH = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V13_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json"
V9_TERMINAL = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root/live/authority/base/first-terminal.json"
V8_PACKAGE_RAW = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json"
V8_RESULT_SCHEMA_RAW = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json"
CANONICAL_PACKAGE = ACTION_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
CANONICAL_RESULT_SCHEMA = ACTION_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"

ACTION_ID = "ace2:qk-gbfp8-base-v13:execute-once:81a8edc1:20260814T075838Z"
V9_ACTION_ID = "ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z"
EXPECTED_PACKAGE_KIND = "qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_package"
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
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


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
            stat_fields = (child / "stat").read_text(encoding="ascii").split()
            start_time = stat_fields[21]
        except (FileNotFoundError, PermissionError, IndexError, OSError):
            continue
        for marker in TARGET_MARKERS:
            if marker in cmdline:
                observed[marker].add((int(child.name), start_time))
    return observed


def reader_namespace(source: str) -> tuple[dict[str, Any], str]:
    tree = ast.parse(source, filename=str(LAUNCHER_PATH))
    wanted_classes = {"ExecutionError"}
    wanted_functions = {
        "require",
        "compact_bytes",
        "read_bytes",
        "decode_canonical_json",
        "read_canonical_json",
    }
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

    preflight_calls = [node for node in ast.walk(preflight) if isinstance(node, ast.Call)]
    require(any(call_name(node) == "_read_accepted_package_and_result_schema" for node in preflight_calls), "preflight omits accepted-byte reader")

    reader_calls = [node for node in ast.walk(accepted_reader) if isinstance(node, ast.Call)]
    canonical_calls = [node for node in reader_calls if call_name(node) == "read_canonical_json"]
    require(len(canonical_calls) == 2, "accepted-byte reader must perform exactly two canonical reads")
    contexts = []
    for node in canonical_calls:
        require(len(node.args) == 2 and isinstance(node.args[1], ast.Constant), "canonical read context shape")
        contexts.append(node.args[1].value)
    require(contexts == ["accepted V8 package", "accepted V8 result schema"], "production canonical read contexts differ")

    main_calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    preflight_lines = [node.lineno for node in main_calls if call_name(node) == "_preflight"]
    durable_lines = [node.lineno for node in main_calls if call_name(node) == "_durable_create"]
    spawn_lines = [node.lineno for node in main_calls if call_name(node) in {"posix_spawn", "Popen", "run"}]
    require(len(preflight_lines) == 1 and durable_lines, "main preflight/live-publication structure differs")
    require(preflight_lines[0] < min(durable_lines), "durable live-state creation precedes read-only preflight")
    require(not spawn_lines, "launcher main contains a process start")
    require(source.index("context = _preflight(") < source.index("_durable_create(LIVE_PATHS[\"authority\"]"), "source order differs")
    return {
        "main_preflight_line": preflight_lines[0],
        "first_durable_create_line": min(durable_lines),
        "accepted_reader_line": accepted_reader.lineno,
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


def verify_manifest_contract(package: dict[str, Any], *, files: bool) -> None:
    require(package.get("artifact_kind") == EXPECTED_PACKAGE_KIND, "package kind")
    require(package.get("action_identity") == {
        "future_action_id": ACTION_ID,
        "prior_terminal_action_id": V9_ACTION_ID,
        "retired_action_ids": [
            "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1610Z",
            "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z",
            "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z",
            V9_ACTION_ID,
        ],
        "reuse_permitted": False,
    }, "action identity")
    require([item["id"] for item in package.get("canonical_bindings", [])] == EXPECTED_BINDING_IDS, "canonical binding order")
    require([item["id"] for item in package.get("local_artifact_bindings", [])] == EXPECTED_LOCAL_IDS, "local binding order")
    require(package.get("prior_terminal_state") == {
        "action_id": V9_ACTION_ID,
        "invocation_count_performed": 0,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "reason_code": "READ_ONLY_PREFLIGHT_FAILED",
        "result_file_sha256": None,
        "status": "PREFLIGHT_FAILED_TERMINAL",
    }, "prior V9 terminal declaration")
    proof = package.get("production_canonical_preflight_proof", {})
    require(proof.get("reader_functions") == ["decode_canonical_json", "read_canonical_json"], "reader proof functions")
    require(proof.get("accepted_reader_function") == "_read_accepted_package_and_result_schema", "accepted reader proof")
    require(proof.get("preflight_function") == "_preflight", "preflight proof")
    require(proof.get("live_creation_after_preflight") is True, "preflight/live ordering proof")
    require(proof.get("target_process_starts_required") == 0, "target-start proof boundary")
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
            source_value = strict_object(source_raw, canonical=False)
            canonical_value = strict_object(canonical_raw, canonical=True)
            require(source_value == canonical_value, f"{key} semantic derivation")
            require(compact_bytes(source_value) == canonical_raw, f"{key} canonical derivation")
            require(record.get("source_raw_sha256") == sha256_bytes(source_raw), f"{key} source hash")
            require(record.get("source_byte_count") == len(source_raw), f"{key} source bytes")
            require(record.get("canonical_sha256") == sha256_bytes(canonical_raw), f"{key} canonical hash")
            require(record.get("canonical_byte_count") == len(canonical_raw), f"{key} canonical bytes")
    canonical_bindings = binding_map(package, "canonical_bindings")
    local_bindings = binding_map(package, "local_artifact_bindings")
    require(canonical_bindings["package"]["path"] == str(CANONICAL_PACKAGE), "canonical package binding path")
    require(canonical_bindings["package"]["sha256"] == derivation["package"]["canonical_sha256"], "canonical package binding hash")
    require(canonical_bindings["result_schema"]["path"] == str(CANONICAL_RESULT_SCHEMA), "canonical schema binding path")
    require(canonical_bindings["result_schema"]["sha256"] == derivation["result_schema"]["canonical_sha256"], "canonical schema binding hash")
    require(canonical_bindings["package_predecessor_raw"]["path"] == str(V8_PACKAGE_RAW), "predecessor package binding")
    require(canonical_bindings["package_predecessor_raw"]["sha256"] == derivation["package"]["source_raw_sha256"], "predecessor package hash")
    require(canonical_bindings["result_schema_predecessor_raw"]["path"] == str(V8_RESULT_SCHEMA_RAW), "predecessor schema binding")
    require(canonical_bindings["result_schema_predecessor_raw"]["sha256"] == derivation["result_schema"]["source_raw_sha256"], "predecessor schema hash")
    require(canonical_bindings["v9_first_terminal"]["path"] == str(V9_TERMINAL), "V9 terminal binding")
    require(canonical_bindings["v9_first_terminal"]["sha256"] == "53303e4db595fa9f472344ec8b29fff4d7e974b3878428a187ade5a23f2efc95", "V9 terminal binding hash")
    require(local_bindings["launcher"]["path"] == str(LAUNCHER_PATH), "launcher local path")
    require(local_bindings["transport"]["path"] == str(TRANSPORT_PATH), "transport local path")
    for record in [*canonical_bindings.values(), *local_bindings.values()]:
        path = Path(record["path"])
        require(path.is_file() and not path.is_symlink(), f"bound file absent: {path}")
        require(record.get("sha256") == sha256_file(path), f"bound hash differs: {path}")
    require(package.get("future_invocation", {}).get("shell") is False, "transport shell")
    require(package.get("launcher_invocation", {}).get("shell") is False, "launcher shell")
    require(package.get("future_invocation", {}).get("argv", [])[-1:] == [ACTION_ID], "transport action argv")
    require(package.get("launcher_invocation", {}).get("argv", [])[-1:] == [ACTION_ID], "launcher action argv")
    require(package.get("claim_boundary", {}).get("execution_authorized") is False, "execution authority boundary")
    policy = package.get("static_file_policy", {})
    require("live" in policy.get("forbidden_directories", []), "live directory not forbidden")
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

    add(lambda value: value["action_identity"].__setitem__("future_action_id", V9_ACTION_ID))
    add(lambda value: value["action_identity"].__setitem__("reuse_permitted", True))
    add(lambda value: value["action_identity"].__setitem__("prior_terminal_action_id", value["action_identity"]["retired_action_ids"][2]))
    add(lambda value: value["prior_terminal_state"].__setitem__("status", "CONSUMED_ORPHAN"))
    add(lambda value: value["prior_terminal_state"].__setitem__("reason_code", "POST_CONSUMPTION_AMBIGUITY"))
    add(lambda value: value["canonical_bindings"].reverse())
    add(lambda value: value["canonical_bindings"][6].__setitem__("path", str(V8_PACKAGE_RAW)))
    add(lambda value: value["canonical_bindings"][6].__setitem__("sha256", sha256_file(V8_PACKAGE_RAW)))
    add(lambda value: value["canonical_bindings"][7].__setitem__("path", str(V8_RESULT_SCHEMA_RAW)))
    add(lambda value: value["canonical_bindings"].pop(12))
    add(lambda value: value["local_artifact_bindings"][6].__setitem__("sha256", "0" * 64))
    add(lambda value: value["accepted_byte_derivation"]["package"].__setitem__("semantic_json_equal", False))
    add(lambda value: value["accepted_byte_derivation"]["package"].__setitem__("canonical_sha256", "0" * 64))
    add(lambda value: value["accepted_byte_derivation"]["result_schema"].__setitem__("source_raw_sha256", "0" * 64))
    add(lambda value: value["production_canonical_preflight_proof"].__setitem__("accepted_reader_function", "_read_json_permissive"))
    add(lambda value: value["production_canonical_preflight_proof"].__setitem__("live_creation_after_preflight", False))
    add(lambda value: value["production_canonical_preflight_proof"].__setitem__("target_process_starts_required", 1))
    add(lambda value: value["future_invocation"].__setitem__("shell", True))
    add(lambda value: value["future_invocation"]["argv"].__setitem__(-1, V9_ACTION_ID))
    add(lambda value: value["claim_boundary"].__setitem__("execution_authorized", True))
    add(lambda value: value["static_file_policy"]["forbidden_directories"].remove("live"))
    add(lambda value: value.__setitem__("artifact_kind", "qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_package"))

    for index, candidate in enumerate(mutations, 1):
        try:
            verify_manifest_contract(candidate, files=False)
        except VerificationError:
            continue
        raise VerificationError(f"adversarial manifest mutation accepted: {index}")
    return len(mutations)


def verify_modes_and_inventory(package: dict[str, Any]) -> dict[str, Any]:
    require(ACTION_ROOT.is_dir() and not ACTION_ROOT.is_symlink(), "action root absent")
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
    require("live" not in observed_directories and not os.path.lexists(ACTION_ROOT / "live"), "live namespace exists")
    return {"file_count": len(observed_files), "directory_count": 1 + len(observed_directories)}


def verify_v9_terminal() -> dict[str, Any]:
    terminal, raw = load_canonical(V9_TERMINAL)
    verify_self_checksum(terminal, "first_terminal_sha256")
    require(sha256_bytes(raw) == "53303e4db595fa9f472344ec8b29fff4d7e974b3878428a187ade5a23f2efc95", "V9 terminal raw hash")
    require(terminal.get("action_id") == V9_ACTION_ID, "V9 terminal action")
    require(terminal.get("status") == "PREFLIGHT_FAILED_TERMINAL", "V9 terminal status")
    require(terminal.get("reason_code") == "READ_ONLY_PREFLIGHT_FAILED", "V9 terminal reason")
    require(terminal.get("failure_detail_sha256") == "7fe1602a1cd89139fc712070f5961dce30d1bc5be2f13153130498de18661373", "V9 terminal failure detail")
    require(terminal.get("invocation_count_performed") == 0 and terminal.get("payload_open_count") == 0, "V9 no-execution counters")
    require(terminal.get("retry_replay_resume_repair_replacement_permitted") is False, "V9 no-replay boundary")
    return {"raw_sha256": sha256_bytes(raw), "self_sha256": terminal["first_terminal_sha256"]}


def verify_acceptance_if_present(package_raw: bytes) -> dict[str, Any] | None:
    schema, _ = load_canonical(ACCEPTANCE_SCHEMA_PATH)
    required_review_fields = {
        "independent_verifier_rerun_sha256",
        "reviewer_decision_sha256",
        "reviewer_thread_id",
        "target_process_starts",
    }
    require(required_review_fields <= set(schema.get("required", [])), "acceptance schema omits review bindings")
    require(schema.get("properties", {}).get("target_process_starts") == {"const": 0}, "acceptance schema target-start contract")
    if not ACCEPTANCE_PATH.exists():
        return None
    acceptance, raw = load_canonical(ACCEPTANCE_PATH)
    verify_self_checksum(acceptance, "acceptance_sha256")
    require(acceptance.get("artifact_kind") == "qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_fresh_l2_static_acceptance", "acceptance kind")
    require(acceptance.get("action_id") == ACTION_ID, "acceptance action")
    require(acceptance.get("decision") == "ACCEPT_STATIC_PACKAGE", "acceptance decision")
    require(acceptance.get("reviewer_role") == "Fresh-L2", "acceptance reviewer")
    require(acceptance.get("static_acceptance_grants_execution_authority") is False, "acceptance authority boundary")
    require(acceptance.get("execution_package_file_sha256") == sha256_bytes(package_raw), "acceptance package binding")
    require(acceptance.get("target_process_starts") == 0, "acceptance target starts")
    require(type(acceptance.get("reviewer_thread_id")) is str and acceptance["reviewer_thread_id"], "acceptance reviewer thread")
    require(type(acceptance.get("reviewer_decision_sha256")) is str and len(acceptance["reviewer_decision_sha256"]) == 64, "acceptance reviewer decision binding")
    require(type(acceptance.get("independent_verifier_rerun_sha256")) is str and len(acceptance["independent_verifier_rerun_sha256"]) == 64, "acceptance verifier rerun binding")
    return {"raw_sha256": sha256_bytes(raw), "self_sha256": acceptance["acceptance_sha256"]}


def main() -> int:
    before = process_snapshot()
    package, package_raw = load_canonical(PACKAGE_PATH)
    verify_self_checksum(package, "package_content_sha256")
    verify_manifest_contract(package, files=True)
    v9_terminal = verify_v9_terminal()

    launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
    namespace, reader_source_sha256 = reader_namespace(launcher_source)
    order = verify_preflight_order(launcher_source)
    production_read = namespace["read_canonical_json"]
    accepted_package, accepted_package_raw = production_read(CANONICAL_PACKAGE, "accepted V8 package")
    accepted_schema, accepted_schema_raw = production_read(CANONICAL_RESULT_SCHEMA, "accepted V8 result schema")
    require(accepted_package_raw == CANONICAL_PACKAGE.read_bytes(), "production package read byte drift")
    require(accepted_schema_raw == CANONICAL_RESULT_SCHEMA.read_bytes(), "production schema read byte drift")
    require(accepted_package == strict_object(V8_PACKAGE_RAW.read_bytes(), canonical=False), "production package semantic drift")
    require(accepted_schema == strict_object(V8_RESULT_SCHEMA_RAW.read_bytes(), canonical=False), "production schema semantic drift")

    reader_cases = reader_adversarial_checks(namespace, accepted_package, accepted_package_raw)
    manifest_cases = manifest_adversarial_checks(package)
    inventory = verify_modes_and_inventory(package)
    acceptance = verify_acceptance_if_present(package_raw)
    after = process_snapshot()
    starts = {
        marker: sorted(after[marker] - before[marker])
        for marker in TARGET_MARKERS
        if after[marker] - before[marker]
    }
    require(not starts, f"target process started during verification: {starts}")

    result = {
        "accepted_package_canonical_sha256": sha256_bytes(accepted_package_raw),
        "accepted_result_schema_canonical_sha256": sha256_bytes(accepted_schema_raw),
        "acceptance": acceptance,
        "action_id": ACTION_ID,
        "adversarial_cases": reader_cases + manifest_cases,
        "inventory": inventory,
        "manifest_raw_sha256": sha256_bytes(package_raw),
        "package_content_sha256": package["package_content_sha256"],
        "preflight_order": order,
        "production_reader_exact_source_sha256": reader_source_sha256,
        "reader_adversarial_cases": reader_cases,
        "manifest_adversarial_cases": manifest_cases,
        "status": "PASS_V13_CANONICAL_PREFLIGHT_SUCCESSOR_INDEPENDENT_STATIC_VERIFICATION",
        "target_process_starts": 0,
        "v9_terminal": v9_terminal,
    }
    sys.stdout.buffer.write(compact_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
