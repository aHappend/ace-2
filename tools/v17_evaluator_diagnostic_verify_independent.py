#!/usr/bin/env python3
"""Independent inert verifier for the V17 evaluator-diagnostic package."""

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

import jsonschema


PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ID = "ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z"
V16_ACTION_ID = "ace2:qk-gbfp8-base-v16:execute-once:e4c05735:20260814T093500Z"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31"
BUILD_ROOT = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001"
MANIFEST = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json"
ADAPTER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py"
WORKER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py"
BRIDGE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17.py"
FIXTURE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_fixture_v17.py"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_EVALUATOR_ADAPTER_FIXTURE_REPORT.json"
CAUSE_REPORT = ACTION_ROOT / "evidence/V16_EVALUATOR_CALL_STATIC_DIAGNOSIS.json"
TERMINAL_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_FIRST_TERMINAL_SCHEMA.json"
ACCEPTANCE = BUILD_ROOT / "FRESH_L2_STATIC_ACCEPTANCE.json"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
OFFICIAL_TARGET_MARKER = str(WORKER)
REQUIRED_CASES = {
    "evaluator_nonzero_exit",
    "evaluator_no_output",
    "evaluator_malformed_output",
    "evaluator_exception",
    "result_publication_failure",
}
REQUIRED_FAILURE_CLASSES = {
    "SPAWN",
    "ABI",
    "RETURN_CODE",
    "STDOUT",
    "STDERR",
    "DECODE",
    "RESULT_SCHEMA",
    "RESULT_RECORD",
    "EXCEPTION",
    "PUBLICATION",
}
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


def canonical_object(path: Path) -> tuple[dict[str, Any], bytes]:
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
        if observed == OFFICIAL_PAYLOAD:
            AUDIT["official_payload_opens"] += 1
    if event in {"os.posix_spawn", "os.posix_spawnp", "subprocess.Popen", "os.exec", "os.execve"}:
        rendered = repr(args)
        if OFFICIAL_TARGET_MARKER in rendered and "production" in rendered:
            AUDIT["official_target_starts"] += 1


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"missing inventory root: {root}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink in inventory: {path}")
        if stat.S_ISREG(info.st_mode):
            records[relative] = {
                "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                "sha256": sha256_file(path),
                "size": info.st_size,
            }
    return records


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise VerificationError(f"missing function: {name}")


def verify_adapter_source() -> dict[str, Any]:
    source = ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ADAPTER))
    invoke = function_node(tree, "invoke_evaluator")
    calls = [
        node
        for node in ast.walk(invoke)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "runner") or (isinstance(node.func, ast.Attribute) and node.func.attr == "run"))
    ]
    require(len(calls) == 1, "adapter must have one process runner call")
    keywords = {item.arg: item.value for item in calls[0].keywords if item.arg is not None}
    require(isinstance(keywords.get("shell"), ast.Constant) and keywords["shell"].value is False, "adapter shell must be false")
    require({"cwd", "env", "input", "stdout", "stderr", "check", "shell"} <= set(keywords), "adapter runner ABI incomplete")
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    require(REQUIRED_FAILURE_CLASSES <= constants, "adapter failure-class vocabulary incomplete")
    require("diagnostic_sha256" in constants and "preview_ascii" in constants, "adapter structured diagnostics absent")
    return {"failure_classes": sorted(REQUIRED_FAILURE_CLASSES), "sha256": sha256_file(ADAPTER)}


def verify_bridge_source(package: dict[str, Any]) -> dict[str, Any]:
    source = BRIDGE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BRIDGE))
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
    require("qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17" in imports, "bridge does not import shared adapter")
    production = function_node(tree, "invoke_production_evaluator")
    names = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(production)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    require({"pack_production_input", "invoke_evaluator"} <= names, "production bridge bypasses shared adapter")
    adapter = package["production_evaluator_adapter"]
    argv = adapter["argv"]
    require(argv[2:4] == ["--mode", "production"], "production worker mode")
    require(adapter["shell"] is False, "production shell")
    require(adapter["input_transport"] == "STDIN_BYTES" and adapter["output_transport"] == "STDOUT_BYTES", "production byte transport")
    return {"production_argv_sha256": sha256_bytes(compact_bytes(argv)), "sha256": sha256_file(BRIDGE)}


def verify_worker_source() -> dict[str, Any]:
    source = WORKER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(WORKER))
    require(function_node(tree, "synthetic") is not None, "synthetic worker mode absent")
    production = function_node(tree, "production")
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    require({"production", "synthetic", "nominal", "nonzero", "no-output", "malformed", "stderr"} <= constants, "worker modes incomplete")
    opened_names = {
        node.func.attr
        for node in ast.walk(production)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    require("read_bytes" in opened_names, "production acceptance/package checks absent")
    require(str(OFFICIAL_PAYLOAD) not in source, "worker embeds official payload path")
    return {"sha256": sha256_file(WORKER), "synthetic_mode_present": True}


def load_fixture_module() -> Any:
    sys.path.insert(0, str(FIXTURE.parent))
    try:
        spec = importlib.util.spec_from_file_location("v17_evaluator_adapter_fixture_inert", FIXTURE)
        require(spec is not None and spec.loader is not None, "fixture import spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def verify_fixture() -> dict[str, Any]:
    module = load_fixture_module()
    observed = module.run_fixture()
    bound, _ = canonical_object(FIXTURE_REPORT)
    require(observed == bound, "fresh fixture differs from bound report")
    require(observed["status"] == "PASS_V17_EVALUATOR_ADAPTER_SYNTHETIC_FIXTURE", "fixture status")
    require(observed["official_payload_open_count"] == 0, "fixture payload opens")
    require(observed["official_target_process_starts"] == 0, "fixture official target starts")
    require(observed["production_mode_exercised"] is False and observed["synthetic_bytes_only"] is True, "fixture synthetic boundary")
    require(observed["nominal_frozen_schema_record_count"] == 2, "nominal frozen-schema records")
    cases = observed["cases"]
    require(type(cases) is list and all(case["passed"] for case in cases), "fixture case failure")
    names = {case["name"] for case in cases}
    require(REQUIRED_CASES <= names, "required fail-closed case absent")
    failure_classes = {case["diagnostic"]["failure_class"] for case in cases}
    require({"SPAWN", "ABI", "RETURN_CODE", "STDOUT", "STDERR", "DECODE", "EXCEPTION", "PUBLICATION"} <= failure_classes, "fixture diagnostic classes incomplete")
    for case in cases:
        diagnostic = case["diagnostic"]
        require(diagnostic["diagnostic_sha256"] == sha256_bytes(compact_bytes({key: value for key, value in diagnostic.items() if key != "diagnostic_sha256"})), f"diagnostic checksum: {case['name']}")
        require(len(diagnostic["stdout"]["preview_ascii"]) <= 160 and len(diagnostic["stderr"]["preview_ascii"]) <= 160, f"bounded preview: {case['name']}")
    return {
        "case_count": len(cases),
        "failure_classes": sorted(failure_classes),
        "nominal_frozen_schema_record_count": 2,
        "report_sha256": sha256_file(FIXTURE_REPORT),
    }


def verify_terminal_schema(package: dict[str, Any]) -> dict[str, Any]:
    schema, raw = canonical_object(TERMINAL_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(schema)
    require(package["terminal_diagnostics"]["schema_sha256"] == sha256_bytes(raw), "terminal schema manifest binding")
    diagnostic = schema["$defs"]["diagnostic"]
    required = set(diagnostic["required"])
    require({"failure_class", "invocation", "process", "stdout", "stderr", "exception", "diagnostic_sha256"} <= required, "terminal structured diagnostic fields")
    return {"schema_id": schema["$id"], "sha256": sha256_bytes(raw)}


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    declared = package["generated_files"]
    observed_files = {
        path.relative_to(ACTION_ROOT).as_posix(): path
        for path in ACTION_ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    require(set(observed_files) == set(declared), "generated file inventory differs")
    for relative, record in declared.items():
        path = observed_files[relative]
        info = os.lstat(path)
        require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"generated file type: {relative}")
        require(stat.S_IMODE(info.st_mode) == 0o444, f"generated file mode: {relative}")
        require(info.st_size == record["size"] and sha256_file(path) == record["sha256"], f"generated file binding: {relative}")
    require(stat.S_IMODE(os.lstat(MANIFEST).st_mode) == 0o444, "manifest mode")
    require(stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode) == 0o555, "action root mode")
    return {"file_count": len(observed_files) + 1}


def verify_preservation(package: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, declaration in package["preservation"].items():
        root = Path(declaration["root"])
        observed = inventory(root)
        require(observed == declaration["files"], f"preserved root changed: {label}")
        result[label] = {"file_count": len(observed), "root": str(root)}
    return result


def verify_runtime(package: dict[str, Any]) -> dict[str, Any]:
    declaration = package["runtime_namespace"]
    require(Path(declaration["runtime_root"]) == RUNTIME_ROOT, "runtime root binding")
    expected = {Path(item) for item in declaration["precreated_directories"]}
    observed_dirs = {RUNTIME_ROOT} | {path for path in RUNTIME_ROOT.rglob("*") if path.is_dir()}
    observed_files = [path for path in RUNTIME_ROOT.rglob("*") if path.is_file()]
    require(observed_dirs == expected, "runtime directory inventory")
    require(not observed_files and declaration["file_count"] == 0, "runtime namespace files")
    for path in expected:
        info = os.lstat(path)
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"runtime directory type: {path}")
        require(stat.S_IMODE(info.st_mode) == 0o700, f"runtime directory mode: {path}")
    forbidden = {"authority.json", "credential.json", "authority-ledger.json", "first-terminal.json", "result.json"}
    require(not any(path.name in forbidden for path in RUNTIME_ROOT.rglob("*")), "runtime lifecycle materialized")
    return {"directory_count": len(expected), "file_count": 0}


def verify_diagnosis(package: dict[str, Any]) -> dict[str, Any]:
    report, raw = canonical_object(CAUSE_REPORT)
    require(report["confirmed_adapter_cause_class"] == "EVALUATOR_CALL_OBSERVABILITY_COLLAPSE", "diagnosis adapter cause")
    require(report["evidence_bounded_underlying_cause_class"] == "UNKNOWN_IN_PROCESS_EVALUATOR_EXCEPTION", "diagnosis unknown cause boundary")
    require("TOP_LEVEL_PYTHON_CALLABLE_ARITY_MISMATCH" in report["excluded_by_static_evidence"], "ABI exclusion absent")
    require(package["diagnosis"]["report_sha256"] == sha256_bytes(raw), "diagnosis binding")
    return {
        "confirmed_adapter_cause_class": report["confirmed_adapter_cause_class"],
        "underlying_cause_class": report["evidence_bounded_underlying_cause_class"],
    }


def verify() -> dict[str, Any]:
    require(ACTION_ID != V16_ACTION_ID, "V17 action identity reused")
    require(MANIFEST.is_file() and ACTION_ROOT.is_dir(), "V17 package absent")
    require(not ACCEPTANCE.exists(), "Fresh-L2 acceptance must be external and absent before review")
    package, raw = canonical_object(MANIFEST)
    verify_self_checksum(package, "package_content_sha256")
    require(package["root_id"] == ROOT_ID, "root identity")
    require(package["action_identity"]["future_action_id"] == ACTION_ID, "action identity")
    require(package["action_identity"]["prior_action_id"] == V16_ACTION_ID, "prior action identity")
    require(package["claim_boundary"] == {
        "acceptance_materialized": False,
        "authority_materialized": False,
        "credential_materialized": False,
        "execution_authorized": False,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "runtime_namespace_file_count": 0,
        "v17_executed": False,
        "v16_replayed": False,
    }, "claim boundary")
    require(package["required_disposition"] == "V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY", "disposition")
    generated = verify_generated_files(package)
    adapter = verify_adapter_source()
    bridge = verify_bridge_source(package)
    worker = verify_worker_source()
    diagnosis = verify_diagnosis(package)
    terminal_schema = verify_terminal_schema(package)
    fixture = verify_fixture()
    runtime = verify_runtime(package)
    preservation = verify_preservation(package)
    require(AUDIT["official_payload_opens"] == 0, "official payload was opened")
    require(AUDIT["official_target_starts"] == 0, "official production target was started")
    report = {
        "action_id": ACTION_ID,
        "adapter": adapter,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_independent_inert_verifier_report",
        "bridge": bridge,
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "diagnosis": diagnosis,
        "fixture": fixture,
        "fresh_l2_acceptance_present": False,
        "generated_package": generated,
        "manifest_sha256": sha256_bytes(raw),
        "official_payload_open_count": AUDIT["official_payload_opens"],
        "official_target_process_starts": AUDIT["official_target_starts"],
        "preservation": preservation,
        "runtime": runtime,
        "status": "PASS_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_INDEPENDENT_INERT_VERIFICATION",
        "terminal_schema": terminal_schema,
        "worker": worker,
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    report = verify()
    raw = compact_bytes(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
