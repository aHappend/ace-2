#!/usr/bin/env python3
"""Independent inert verifier for the marker-free V16 preauthority package.

This verifier reads static package material, executes only the synthetic
fixture in-process, and never invokes the transport, launcher, controller,
evaluator, or sealed tensor path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
PRESERVED_V16_CANDIDATE_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_shellfree_action_root"
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_e4c05735"
MANIFEST = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_PACKAGE.json"
LAUNCHER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v16.py"
TRANSPORT = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v16.py"
SHARED = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16.py"
RESULT_VALIDATION = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_result_validation_v16.py"
FIXTURE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_marker_free_fixture_v16.py"
FIXTURE_REPORT = ACTION_ROOT / "evidence/MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE_REPORT.json"
RESULT_SCHEMA = ACTION_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"
TERMINAL_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_FIRST_TERMINAL_SCHEMA.json"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
SEALED_TENSOR = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
ACTION_ID = "ace2:qk-gbfp8-base-v16:execute-once:e4c05735:20260814T093500Z"
DISPOSITION = "V16_EVALUATOR_RETURN_PATH_SUCCESSOR_STATIC_PACKAGE_READY_NO_EXECUTION_AUTHORITY"
FAILURE_STAGES = {
    "EVALUATOR_CALL",
    "CANONICAL_DECODE",
    "RESULT_SCHEMA",
    "RESULT_RECORD",
    "RESULT_PUBLICATION",
    "TERMINAL_BUILD",
    "TERMINAL_SCHEMA",
    "TERMINAL_PUBLICATION",
}
TARGET_MARKERS = (
    str(TRANSPORT),
    str(LAUNCHER),
    "qk_gbfp8_head64_granularity_sweep_controller_static_v8.py",
    "qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py",
)
AUDIT = {"official_payload_opens": 0, "official_target_starts": 0}


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


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    value = json.loads(raw.decode("ascii", "strict"), object_pairs_hook=pairs)
    require(type(value) is dict and not duplicate, f"non-object or duplicate key: {path}")
    require(compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def verify_self_checksum(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    require(type(observed) is str and len(observed) == 64, f"checksum syntax: {field}")
    payload = dict(value)
    payload.pop(field)
    require(sha256_bytes(compact_bytes(payload)) == observed, f"checksum mismatch: {field}")


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            observed = Path(os.fspath(args[0])).resolve()
        except (TypeError, ValueError, OSError):
            observed = None
        if observed == SEALED_TENSOR:
            AUDIT["official_payload_opens"] += 1
    if event in {"os.posix_spawn", "os.posix_spawnp", "subprocess.Popen", "os.exec", "os.execve"}:
        rendered = repr(args)
        if any(marker in rendered for marker in TARGET_MARKERS):
            AUDIT["official_target_starts"] += 1


def call_names(function: ast.FunctionDef) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            result.append((node.lineno, node.func.id))
        elif isinstance(node.func, ast.Attribute):
            result.append((node.lineno, node.func.attr))
    return sorted(result)


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise VerificationError(f"missing function: {name}")


def verify_shared_source() -> dict[str, Any]:
    source = SHARED.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SHARED))
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    require(not (imports & {"os", "pathlib", "subprocess", "importlib", "socket"}), "shared path has I/O-capable import")
    process = function_node(tree, "process_evaluator_return")
    names = [name for _, name in call_names(process)]
    ordered = ["_decode_canonical_json", "schema_validate", "record_validate", "compact_bytes"]
    positions = [names.index(name) for name in ordered]
    require(positions == sorted(positions), "shared evaluator-return operation order differs")
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    require(FAILURE_STAGES <= constants, "shared source omits required failure stage")
    return {"operation_order": ordered, "sha256": sha256_file(SHARED)}


def load_result_validation_module() -> Any:
    spec = importlib.util.spec_from_file_location("v16_exact_result_validation_inert", RESULT_VALIDATION)
    require(spec is not None and spec.loader is not None, "result validation import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_result_validation_source() -> dict[str, Any]:
    source = RESULT_VALIDATION.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RESULT_VALIDATION))
    imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    require(not (imports & {"os", "pathlib", "subprocess", "importlib", "socket", "runpy"}), "result validation module has I/O-capable import")
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({"construct_result_schema_validator", "validate_result_schema_exact", "validate_result_record_exact", "bind_result_validators"} <= function_names, "exact result validator function missing")
    require("Draft202012Validator" in source and "check_schema" in source, "production Draft 2020-12 schema validator absent")
    module = load_result_validation_module()
    schema, schema_raw = load_canonical(RESULT_SCHEMA)
    require(module.FROZEN_RESULT_SCHEMA_BYTES == schema_raw, "embedded schema bytes differ from production schema")
    require(module.RESULT_SCHEMA_SHA256 == sha256_bytes(schema_raw), "embedded schema hash differs")
    require(module.frozen_result_schema() == schema, "embedded schema object differs")
    validator = module.construct_result_schema_validator(schema)
    require(type(validator).__module__ == "jsonschema.validators" and type(validator).__qualname__ == "Draft202012Validator", "production schema validator class")
    return {
        "record_validator": "qk_gbfp8_head64_granularity_sweep_result_validation_v16.validate_result_record_exact",
        "schema_raw_sha256": sha256_bytes(schema_raw),
        "schema_validator_class": "jsonschema.validators.Draft202012Validator",
        "sha256": sha256_file(RESULT_VALIDATION),
    }


def verify_launcher_source() -> dict[str, Any]:
    source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(LAUNCHER))
    run_consumed = function_node(tree, "_run_consumed")
    calls = call_names(run_consumed)
    by_name: dict[str, list[int]] = {}
    for line, name in calls:
        by_name.setdefault(name, []).append(line)
    for name in ("call_evaluator_once", "process_evaluator_return", "complete_result_path"):
        require(name in by_name, f"production launcher omits shared call: {name}")
    require(
        min(by_name["call_evaluator_once"]) < min(by_name["process_evaluator_return"]) < min(by_name["complete_result_path"]),
        "production evaluator-return call order differs",
    )
    require("qk_gbfp8_head64_granularity_sweep_result_validation_v16" in source, "production launcher omits exact result-validation import")
    require(not any(isinstance(node, ast.FunctionDef) and node.name == "_validate_result_record" for node in tree.body), "production launcher retains a substitute result-record validator")
    freeze = function_node(tree, "_freeze_execution_plan")
    freeze_calls = {name for _, name in call_names(freeze)}
    require("bind_result_validators" in freeze_calls, "production launcher does not bind exact result validators")
    require("validate_result_record=result_validation.validate_result_record" in source, "production record validator binding differs")
    require("validate_result_schema=result_validation.validate_result_schema" in source, "production schema validator binding differs")
    main = function_node(tree, "main")
    main_calls = call_names(main)
    authority_lines = [line for line, name in main_calls if name == "_durable_create"]
    preflight_lines = [line for line, name in main_calls if name == "_preflight"]
    require(preflight_lines and authority_lines and min(preflight_lines) < min(authority_lines), "authority precedes preflight")
    require("failure_stage=None" in source, "successful terminal does not use null failure_stage")
    return {
        "evaluate_line": min(by_name["call_evaluator_once"]),
        "process_line": min(by_name["process_evaluator_return"]),
        "complete_line": min(by_name["complete_result_path"]),
        "result_validator_factory": "bind_result_validators",
        "sha256": sha256_file(LAUNCHER),
    }


def verify_fixture_source() -> dict[str, Any]:
    source = FIXTURE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FIXTURE))
    forbidden_imports = {"os", "pathlib", "subprocess", "importlib", "socket", "runpy"}
    imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    require(not (imports & forbidden_imports), "fixture imports an authority-capable module")
    require("qk_gbfp8_head64_granularity_sweep_result_validation_v16" in source, "fixture omits exact result-validation module")
    require(not any(isinstance(node, ast.FunctionDef) and node.name in {"schema_validate", "record_validate"} for node in tree.body), "fixture defines substitute result validator")
    forbidden_calls = {"open", "read_bytes", "read_text", "write_bytes", "write_text", "posix_spawn", "Popen", "run", "execve"}
    observed_calls = {name for _, name in call_names(function_node(tree, "run_fixture"))}
    require(not (observed_calls & forbidden_calls), "fixture contains filesystem/process call")
    for marker in (
        str(SEALED_TENSOR),
        str(TRANSPORT),
        str(LAUNCHER),
        "authority.json",
        "credential.json",
        "authority-ledger.json",
        "first-terminal.json",
        "qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py",
    ):
        require(marker not in source, f"fixture contains forbidden reachability marker: {marker}")
    require("process_evaluator_return" in source and "complete_result_path" in source, "fixture does not use shared production path")
    require("bind_result_validators" in source, "fixture does not bind exact result validators")
    require("RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record" in source, "fixture exact validator call binding differs")
    return {"imports": sorted(imports), "result_validation_sha256": sha256_file(RESULT_VALIDATION), "sha256": sha256_file(FIXTURE)}


def load_fixture_module() -> Any:
    tools_dir = str(FIXTURE.parent)
    sys.path.insert(0, tools_dir)
    try:
        spec = importlib.util.spec_from_file_location("v16_marker_free_fixture_inert", FIXTURE)
        require(spec is not None and spec.loader is not None, "fixture import spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(tools_dir)


def verify_fixture_report() -> dict[str, Any]:
    module = load_fixture_module()
    observed = module.run_fixture()
    bound, raw = load_canonical(FIXTURE_REPORT)
    require(observed == bound, "bound fixture report differs from fresh fixture run")
    require(bound.get("status") == "PASS_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE", "fixture status")
    require(bound.get("official_payload_open_count") == 0, "fixture payload opens")
    require(bound.get("official_target_process_starts") == 0, "fixture target starts")
    require(bound.get("nominal_schema_conforming_case_count") == 2, "fixture schema-conforming nominal count")
    require(bound.get("nominal_record_valid_case_count") == 2, "fixture record-valid nominal count")
    require(bound.get("result_schema_sha256") == sha256_file(RESULT_SCHEMA), "fixture result schema binding")
    require(bound.get("schema_validator_class") == "jsonschema.validators.Draft202012Validator", "fixture schema validator class")
    require(bound.get("schema_validator_callable") == "qk_gbfp8_head64_granularity_sweep_result_validation_v16.validate_result_schema_exact", "fixture schema validator callable")
    require(bound.get("record_validator_callable") == "qk_gbfp8_head64_granularity_sweep_result_validation_v16.validate_result_record_exact", "fixture record validator callable")
    require(bound.get("validator_factory") == "qk_gbfp8_head64_granularity_sweep_result_validation_v16.bind_result_validators", "fixture validator factory")
    cases = bound.get("cases")
    require(type(cases) is list and len(cases) >= 12, "fixture case count")
    require(all(case.get("passed") is True for case in cases), "fixture case failure")
    by_name = {case.get("name"): case for case in cases}
    require(by_name.get("exact_key_mismatch", {}).get("observed_stage") == "RESULT_SCHEMA", "exact-key mismatch did not reach production schema validator")
    require(by_name.get("binding_mismatch", {}).get("observed_stage") == "RESULT_RECORD", "binding mismatch did not reach exact record validator")
    require(by_name.get("self_checksum_mismatch", {}).get("observed_stage") == "RESULT_RECORD", "self-checksum mismatch did not reach exact record validator")
    observed_stages = {case.get("observed_stage") for case in cases if case.get("observed_stage") is not None}
    require(FAILURE_STAGES <= observed_stages, "fixture does not cover every required failure stage")
    outcomes = {case.get("name") for case in cases if case.get("observed_stage") is None}
    require({"all_hard_gates_fail", "first_passing_candidate"} <= outcomes, "fixture omits structural outcome")
    return {
        "case_count": len(cases),
        "nominal_record_valid_case_count": 2,
        "nominal_schema_conforming_case_count": 2,
        "raw_sha256": sha256_bytes(raw),
        "result_schema_sha256": bound["result_schema_sha256"],
        "stages": sorted(observed_stages),
    }


def verify_terminal_schema() -> dict[str, Any]:
    schema, _ = load_canonical(TERMINAL_SCHEMA)
    required = set(schema.get("required", []))
    require("failure_stage" in required, "terminal schema does not require failure_stage")
    property_value = schema.get("properties", {}).get("failure_stage", {})
    enum_values: set[str] = set()
    for branch in property_value.get("oneOf", []):
        enum_values.update(branch.get("enum", []))
    require(FAILURE_STAGES <= enum_values, "terminal schema omits failure stage")
    return {"failure_stages": sorted(enum_values), "sha256": sha256_file(TERMINAL_SCHEMA)}


def verify_manifest() -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    package, raw = load_canonical(MANIFEST)
    verify_self_checksum(package, "package_content_sha256")
    require(package.get("artifact_kind") == "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_package", "package kind")
    require(package.get("root_id") == "qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_shellfree_action_root", "repair root identity")
    require(package.get("action_identity", {}).get("future_action_id") == ACTION_ID, "action identity")
    require(package.get("required_disposition") == DISPOSITION, "required disposition")
    claim = package.get("claim_boundary", {})
    require(claim == {
        "acceptance_materialized": False,
        "authority_materialized": False,
        "controller_or_evaluator_invoked": False,
        "credential_materialized": False,
        "execution_authorized": False,
        "first_terminal_materialized": False,
        "ledger_materialized": False,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "result_materialized": False,
        "runtime_namespace_file_count": 0,
        "transport_or_launcher_invoked": False,
    }, "claim boundary")
    for group in ("canonical_bindings", "local_artifact_bindings"):
        records = package.get(group)
        require(type(records) is list and records, f"binding group: {group}")
        seen: set[str] = set()
        for record in records:
            require(set(record) == {"id", "path", "sha256"}, f"binding shape: {group}")
            require(record["id"] not in seen, f"duplicate binding: {record['id']}")
            seen.add(record["id"])
            path = Path(record["path"])
            require(path.is_file() and not path.is_symlink(), f"bound file absent: {path}")
            require(sha256_file(path) == record["sha256"], f"bound hash differs: {path}")
    future = package.get("future_invocation", {})
    require(future.get("command_representation") == "ARGV_VECTOR_ONLY" and future.get("shell") is False, "future invocation representation")
    require(future.get("cwd") == str(ACTION_ROOT), "future invocation cwd")
    require(future.get("environment") == {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}, "future invocation environment")
    require(future.get("argv", [None, None])[1] == str(TRANSPORT), "future transport argv")
    evaluator_return_path = package.get("evaluator_return_path", {})
    require(evaluator_return_path.get("production_and_fixture_bind_exact_result_validators") is True, "manifest exact result-validator claim")
    require(evaluator_return_path.get("result_validation_module") == str(RESULT_VALIDATION), "manifest result-validation path")
    require(evaluator_return_path.get("result_validator_factory") == "bind_result_validators", "manifest result-validator factory")
    return package, raw, {"manifest_raw_sha256": sha256_bytes(raw), "binding_count": sum(len(package[key]) for key in ("canonical_bindings", "local_artifact_bindings"))}


def verify_package_inventory(package: dict[str, Any]) -> dict[str, Any]:
    require(ACTION_ROOT.is_dir() and not ACTION_ROOT.is_symlink(), "action root absent")
    require(stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode) == 0o555, "action root mode")
    allowed = set(package["static_file_policy"]["allowed_relative_files"])
    optional = set(package["static_file_policy"].get("reviewer_optional_relative_files", []))
    observed_files: set[str] = set()
    for path in ACTION_ROOT.rglob("*"):
        relative = path.relative_to(ACTION_ROOT).as_posix()
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink in action root: {relative}")
        if path.is_dir():
            require(stat.S_IMODE(info.st_mode) == 0o555, f"directory mode: {relative}")
        else:
            observed_files.add(relative)
            require(stat.S_IMODE(info.st_mode) == 0o444, f"file mode: {relative}")
    require(observed_files <= allowed | optional, "unexpected package file")
    require(allowed <= observed_files, "required package file absent")
    require(not (ACTION_ROOT / "live").exists(), "package-local live namespace exists")
    require(ACCEPTANCE.exists() is False, "Fresh-L2 acceptance was materialized by Engineer")
    return {"file_count": len(observed_files), "acceptance_present": False}


def verify_runtime_inventory(package: dict[str, Any]) -> dict[str, Any]:
    runtime = package["runtime_namespace"]
    require(runtime["runtime_root"] == str(RUNTIME_ROOT), "runtime root binding")
    expected_directories = {Path(path) for path in runtime["precreated_directories"]}
    observed_directories: set[Path] = set()
    observed_files: set[Path] = set()
    for path in RUNTIME_ROOT.rglob("*"):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"runtime symlink: {path}")
        if path.is_dir():
            observed_directories.add(path)
            require(stat.S_IMODE(info.st_mode) == 0o700, f"runtime directory mode: {path}")
        else:
            observed_files.add(path)
    require(RUNTIME_ROOT in expected_directories, "runtime root absent from declaration")
    require(observed_directories | {RUNTIME_ROOT} == expected_directories, "runtime directory inventory")
    require(not observed_files, "runtime namespace is not empty")
    for path in runtime["paths"].values():
        require(not os.path.lexists(path), f"runtime lifecycle file exists: {path}")
    require(not os.path.lexists(runtime["fallback_terminal"]), "fallback terminal exists")
    return {"directory_count": len(expected_directories), "file_count": 0, "runtime_root": str(RUNTIME_ROOT)}


def verify_v15_preserved(package: dict[str, Any]) -> dict[str, Any]:
    records = package.get("v15_retirement", {}).get("static_file_sha256")
    require(type(records) is dict and records, "V15 preservation map absent")
    root = Path(package["v15_retirement"]["root"])
    for relative, expected in records.items():
        path = root / relative
        require(path.is_file() and sha256_file(path) == expected, f"retired V15 changed: {relative}")
    return {"file_count": len(records), "root": str(root)}


def verify_rejected_v16_candidate_preserved(package: dict[str, Any]) -> dict[str, Any]:
    declaration = package.get("preserved_rejected_v16_candidate", {})
    records = declaration.get("static_file_sha256")
    require(type(records) is dict and records, "rejected V16 candidate preservation map absent")
    root = Path(declaration.get("root", ""))
    require(root == PRESERVED_V16_CANDIDATE_ROOT, "rejected V16 candidate root binding")
    require(declaration.get("mutation_permitted") is False, "rejected V16 candidate mutation declaration")
    require(declaration.get("status") == "REJECTED_CANDIDATE_PRESERVED_UNMODIFIED", "rejected V16 candidate status")
    require(root.is_dir() and not root.is_symlink(), "rejected V16 candidate root absent")
    require(stat.S_IMODE(os.lstat(root).st_mode) == 0o555, "rejected V16 candidate root mode")
    for relative, expected in records.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"rejected V16 candidate file absent: {relative}")
        require(sha256_file(path) == expected, f"rejected V16 candidate changed: {relative}")
    require(not (root / "review/FRESH_L2_STATIC_ACCEPTANCE.json").exists(), "rejected V16 candidate unexpectedly accepted")
    return {"file_count": len(records), "root": str(root)}


def write_report_once(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite verifier report: {path}")
    raw = compact_bytes(report)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "short verifier report write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    package, package_raw, manifest_evidence = verify_manifest()
    inventory = verify_package_inventory(package)
    runtime = verify_runtime_inventory(package)
    shared = verify_shared_source()
    result_validation = verify_result_validation_source()
    launcher = verify_launcher_source()
    fixture_source = verify_fixture_source()
    fixture = verify_fixture_report()
    terminal_schema = verify_terminal_schema()
    v15 = verify_v15_preserved(package)
    rejected_v16_candidate = verify_rejected_v16_candidate_preserved(package)
    require(AUDIT["official_payload_opens"] == 0, "official payload was opened")
    require(AUDIT["official_target_starts"] == 0, "official target was started")
    report = {
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_independent_static_verifier_report",
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "fixture": fixture,
        "fixture_source": fixture_source,
        "fresh_l2_acceptance_present": False,
        "launcher": launcher,
        "manifest": manifest_evidence,
        "official_payload_open_count": AUDIT["official_payload_opens"],
        "official_target_process_starts": AUDIT["official_target_starts"],
        "package_inventory": inventory,
        "required_disposition": DISPOSITION,
        "rejected_v16_candidate_preservation": rejected_v16_candidate,
        "result_validation": result_validation,
        "runtime_inventory": runtime,
        "shared_result_path": shared,
        "status": "PASS_V16_MARKER_FREE_PREAUTHORITY_INDEPENDENT_STATIC_VERIFICATION",
        "terminal_schema": terminal_schema,
        "v15_preservation": v15,
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    if args.report is not None:
        write_report_once(args.report, report)
    sys.stdout.buffer.write(compact_bytes(report))
    require(sha256_bytes(package_raw) == manifest_evidence["manifest_raw_sha256"], "manifest evidence drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
