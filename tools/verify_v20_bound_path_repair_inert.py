#!/usr/bin/env python3
"""Acceptance-aware inert verifier for the immutable V20 bound-path repair."""

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
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v20:execute-once:a81f2916:additive-0001"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_a81f2916"
MANIFEST_NAME = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json"
MANIFEST = ACTION_ROOT / MANIFEST_NAME
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_FRESH_L2_ACCEPTANCE_SCHEMA.json"
FIXTURE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v20.py"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_BOUND_CORE_MANIFEST_PREFLIGHT_FIXTURE_REPORT.json"
PREFLIGHT = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v20.py"
LAUNCHER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v20.py"
TRANSPORT = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v20.py"
ENGINE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v20.py"
WORKER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v20.py"
BINDING_TABLE = ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
CORE_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root"
CORE_MANIFEST = CORE_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json"
CORE_ACCEPTANCE = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/FRESH_L2_STATIC_ACCEPTANCE.json"
CORE_VERIFIER_REPORT = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/independent-inert-verifier-report.json"
STATIC_V8_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root"
G16_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root"
OFFICIAL_LANE_METADATA = G16_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
OFFICIAL_PAYLOAD = G16_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
OFFICIAL_EVALUATOR = STATIC_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
OFFICIAL_PARSER = STATIC_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
V18_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root"
V18_MANIFEST = V18_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_STATIC_PACKAGE.json"
V18_BINDING = V18_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
V18_ACCEPTANCE = V18_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V18_POST_REPORT = PROJECT_ROOT / "build/v18-c02-binding-repair-static-0001/fresh-l2-post-acceptance-inert-report.json"
V19_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_action_root"
V19_BUILD_ROOT = PROJECT_ROOT / "build/v19-exactly-once-v18-binding-attempt-0001"
V19_RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_fdfc33a1"
V19_TERMINAL = V19_RUNTIME_ROOT / "primary/authority/base/first-terminal.json"
V19_ACTION_ID = "ace2:qk-gbfp8-base-v19:execute-once:fdfc33a1:additive-0001"
V19_TERMINAL_SHA256 = "a81f2916fe3e5f2aa0489caee7d7a52ada1a8760b32ea7faf3a94bad6f4cb862"
V18_BINDING_SHA256 = "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655"
EXPECTED_INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
SELECTED_NAMES = ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"]
EXPECTED_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TZ": "UTC"}
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
    descriptor = os.open(path, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0))
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    candidate = dict(value)
    candidate.pop(field, None)
    require(observed == sha256_bytes(compact_bytes(candidate)), f"self hash: {field}")


def inventory_digest(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"preservation root absent: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"preservation symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative, "sha256": sha256_file(path), "size": info.st_size})
        else:
            raise VerificationError(f"unsupported preservation entry: {path}")
    return {"entry_count": len(records), "file_count": sum(item["kind"] == "file" for item in records), "root": str(root), "tree_sha256": sha256_bytes(compact_bytes(records))}


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path == OFFICIAL_PAYLOAD.resolve():
            AUDIT["official_payload_opens"] += 1
            raise VerificationError("official payload open prohibited")
    if event == "subprocess.Popen":
        rendered = repr(args)
        if "evaluator_worker_v20.py" in rendered or "evaluator_static_v8.py" in rendered:
            AUDIT["official_target_starts"] += 1
            raise VerificationError("official evaluator start prohibited")


def verify_manifest() -> tuple[dict[str, Any], bytes]:
    package, raw = canonical(MANIFEST)
    verify_self_hash(package, "package_content_sha256")
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_package", "package kind")
    require(package["root_id"] == ROOT_ID, "root id")
    identity = package["action_identity"]
    require(identity["action_id"] == ACTION_ID and identity["predecessor_action_id"] == V19_ACTION_ID, "action identity")
    require(identity["retired_predecessor_terminal_sha256"] == V19_TERMINAL_SHA256, "retired terminal binding")
    require(identity["v19_modified_or_replayed"] is False and identity["retry_replay_resume_repair_replacement_permitted"] is False, "predecessor boundary")
    claim = package["claim_boundary"]
    require(claim["claim"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "claim boundary")
    require(all(claim[key] is False for key in ("authority_materialized", "credential_materialized", "execution_authorized", "ledger_materialized", "owner_claim_materialized", "result_materialized", "runtime_namespace_materialized", "runtime_terminal_materialized", "stage_2_activity", "v20_executed")), "static claim booleans")
    require(claim["evaluator_invocations"] == 0 and claim["official_payload_open_count"] == 0 and claim["official_target_process_starts"] == 0, "static claim counters")
    require(package["core_manifest_closure"]["package_key"] == "core_manifest_closure", "canonical package key")
    require(package["interpreter"] == {"path": str(INTERPRETER), "sha256": EXPECTED_INTERPRETER_SHA256, "version": "3.13.5"}, "interpreter contract")
    require(sha256_file(INTERPRETER) == EXPECTED_INTERPRETER_SHA256, "interpreter hash")
    return package, raw


def verify_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    require(ACTION_ROOT.is_dir() and not ACTION_ROOT.is_symlink(), "action root absent")
    require(stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode) == 0o555, "action root mode")
    policy = package["static_file_policy"]
    allowed = set(policy["allowed_relative_files"])
    acceptance_relative = "review/FRESH_L2_STATIC_ACCEPTANCE.json"
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in sorted(ACTION_ROOT.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"action symlink: {path}")
        relative = path.relative_to(ACTION_ROOT).as_posix()
        if stat.S_ISDIR(info.st_mode):
            observed_directories.add(relative)
            require(stat.S_IMODE(info.st_mode) == 0o555, f"directory mode: {relative}")
        else:
            require(stat.S_ISREG(info.st_mode), f"non-regular action entry: {relative}")
            observed_files.add(relative)
            require(stat.S_IMODE(info.st_mode) == 0o444, f"file mode: {relative}")
            require(not relative.endswith((".pyc", ".pyo")), f"Python artifact: {relative}")
    expected = set(allowed)
    if allow_acceptance:
        require(ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    else:
        expected.remove(acceptance_relative)
        require(not os.path.lexists(ACCEPTANCE), "unexpected Fresh-L2 acceptance")
        require("review" not in observed_directories, "pre-acceptance review directory")
    require(observed_files == expected, f"action inventory drift: expected={sorted(expected)} observed={sorted(observed_files)}")
    return {"acceptance_present": allow_acceptance, "directory_count": len(observed_directories) + 1, "file_count": len(observed_files)}


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, record in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file() and sha256_file(path) == record["sha256"] and path.stat().st_size == record["size"], f"generated binding: {relative}")
    own = "tools/verify_qk_gbfp8_head64_granularity_sweep_bound_path_repair_v20.py"
    require(package["generated_files"][own]["sha256"] == sha256_file(Path(__file__)), "verifier self binding")
    return {"generated_file_count": len(package["generated_files"]), "verifier_file_sha256": sha256_file(Path(__file__))}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_expected_path(paths: set[str], path: Path, root: Path | None = None) -> None:
    path = path.resolve()
    paths.add(str(path))
    if root is None:
        return
    root = root.resolve()
    current = path.parent
    while current == root or root in current.parents:
        paths.add(str(current))
        if current == root:
            break
        current = current.parent


def derive_expected_closure_paths() -> set[str]:
    core, _ = canonical(CORE_MANIFEST)
    accepted_path = CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
    accepted, _ = canonical(accepted_path)
    paths: set[str] = set()
    add_expected_path(paths, CORE_ROOT)
    add_expected_path(paths, CORE_MANIFEST)
    for relative in core["generated_files"]:
        add_expected_path(paths, CORE_ROOT / relative, CORE_ROOT)
    for preserved in core["preservation"].values():
        root = Path(preserved["root"])
        add_expected_path(paths, root)
        for relative in preserved.get("files", {}):
            add_expected_path(paths, root / relative, root)
    def absolute_strings(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                absolute_strings(item)
        elif isinstance(value, list):
            for item in value:
                absolute_strings(item)
        elif isinstance(value, str) and value.startswith("/") and os.path.lexists(value):
            add_expected_path(paths, Path(value))
    absolute_strings(core)
    for path in (CORE_ACCEPTANCE, CORE_VERIFIER_REPORT, V18_MANIFEST, V18_BINDING, V18_ACCEPTANCE, V18_POST_REPORT, INTERPRETER, accepted_path):
        add_expected_path(paths, path)
    for label in ("c02_parser", "controller", "evaluator", "result_schema", "verifier"):
        add_expected_path(paths, STATIC_V8_ROOT / accepted["static_bindings"][label]["path"], STATIC_V8_ROOT)
    add_expected_path(paths, OFFICIAL_LANE_METADATA)
    add_expected_path(paths, OFFICIAL_PAYLOAD, G16_ROOT)
    return paths


def verify_closure(package: dict[str, Any]) -> dict[str, Any]:
    module = load_module("v20_core_manifest_preflight_verify", PREFLIGHT)
    observed = module.verify_core_manifest_closure(package["core_manifest_closure"], expected_core_manifest_path=CORE_MANIFEST)
    closure = package["core_manifest_closure"]
    observed_paths = {entry["path"] for entry in closure["entries"]}
    expected_paths = derive_expected_closure_paths()
    require(observed_paths == expected_paths, f"transitive closure path drift: missing={sorted(expected_paths - observed_paths)} extra={sorted(observed_paths - expected_paths)}")
    require(closure["core_manifest_path"] == str(CORE_MANIFEST), "exact core manifest path")
    require(closure["core_manifest_sha256"] == sha256_file(CORE_MANIFEST), "exact core manifest hash")
    require(closure["core_manifest_mode"] == "0444", "exact core manifest mode")
    require(closure["deferred_hash_paths"] == [str(OFFICIAL_PAYLOAD)], "sole deferred official payload")
    require(closure["official_evaluator_sha256"] == sha256_file(OFFICIAL_EVALUATOR), "official evaluator closure")
    require(closure["official_parser_sha256"] == sha256_file(OFFICIAL_PARSER), "official parser closure")
    return {**observed, "entries_sha256": closure["entries_sha256"], "path_count": len(observed_paths)}


def verify_binding(package: dict[str, Any]) -> dict[str, Any]:
    table, raw = canonical(BINDING_TABLE)
    verify_self_hash(table, "binding_table_sha256")
    require(sha256_bytes(raw) == V18_BINDING_SHA256 and raw == V18_BINDING.read_bytes(), "integrated V18 binding bytes")
    require(table["record_count"] == 25 and len(table["records"]) == 25, "binding count")
    require(table["selected_tensor_names"] == SELECTED_NAMES, "three numerical tensors")
    require(package["numerical_selection"] == {"complete_parser_binding_count": 25, "evaluator_receives_complete_binding_table": True, "numerical_tensor_count": 3, "tensor_names": SELECTED_NAMES}, "numerical package contract")
    return {"binding_table_file_sha256": sha256_bytes(raw), "complete_binding_count": 25, "numerical_tensor_count": 3}


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, f"function cardinality: {name}")
    return matches[0]


def package_subscripts(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "package" and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
    return keys


def verify_sources(package: dict[str, Any]) -> dict[str, Any]:
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    engine_source = ENGINE.read_text(encoding="utf-8")
    transport_source = TRANSPORT.read_text(encoding="utf-8")
    fixture_source = FIXTURE.read_text(encoding="utf-8")
    preflight_source = PREFLIGHT.read_text(encoding="utf-8")
    require(str(CORE_MANIFEST) in launcher_source, "launcher exact V17 CORE_MANIFEST path")
    require("EXECUTION_V20_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json" not in launcher_source, "nonexistent V20 core basename")
    require("frozen_core_imports" not in launcher_source, "absent package key retained")
    require("from qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v20 import verify_core_manifest_closure" in launcher_source, "shared preflight import")
    launcher_tree = ast.parse(launcher_source)
    run_source = ast.get_source_segment(launcher_source, function(launcher_tree, "run_launcher")) or ""
    require(run_source.index("read_only_preflight(") < run_source.index("run_execution("), "shared preflight before owner engine")
    read_source = ast.get_source_segment(launcher_source, function(launcher_tree, "read_only_preflight")) or ""
    require("verify_core_manifest_closure(holder_closure" in read_source, "closure reached by production preflight")
    require("not any(os.path.lexists(path) for path in paths)" in read_source, "all lifecycle paths absent before owner")
    require("canonical(paths.owner)" not in read_source, "owner not required before shared preflight")
    engine_tree = ast.parse(engine_source)
    execution_source = ast.get_source_segment(engine_source, function(engine_tree, "run_execution")) or ""
    create_source = ast.get_source_segment(engine_source, function(engine_tree, "durable_create")) or ""
    claim_source = ast.get_source_segment(engine_source, function(engine_tree, "_claim_owner")) or ""
    require("os.O_EXCL" in create_source and "os.O_CREAT" in create_source, "atomic create")
    require(claim_source.count("durable_create(") == 1, "single owner claim")
    require(execution_source.count("payload_reader(") == 1 and execution_source.count("invoke_once(") == 1, "single payload/evaluator boundary")
    require('return Outcome(counts, True, None, "DUPLICATE_REJECTED", None, False)' in execution_source, "loser no-publication")
    require("os.posix_spawn" in transport_source, "shell-free transport")
    require("real_preflight_cases" in fixture_source and "verify_core_manifest_closure" in fixture_source, "fixture reaches shared preflight")
    require("DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN" in preflight_source, "payload hash deferral policy")
    package_keys = set(package)
    checked_files = []
    used_keys: set[str] = set()
    for path in sorted((ACTION_ROOT / "tools").glob("*.py")):
        if path in (FIXTURE, PREFLIGHT) or path.name.startswith("verify_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        keys = package_subscripts(tree)
        require(keys <= package_keys, f"absent constant package key in {path.name}: {sorted(keys - package_keys)}")
        used_keys.update(keys)
        checked_files.append(path.name)
    require("core_manifest_closure" in used_keys and "frozen_core_imports" not in used_keys, "canonical production package key coverage")
    lifecycle = package["lifecycle"]
    require(lifecycle["consumption_order"][0] == "shared_complete_core_manifest_preflight", "preflight lifecycle order")
    require(lifecycle["authority_create_count_maximum"] == 1 and lifecycle["credential_consumption_count_maximum"] == 1, "authority cardinality")
    require(lifecycle["payload_open_count_maximum"] == 1 and lifecycle["invocation_count_maximum"] == 1, "invocation cardinality")
    return {"constant_package_keys": sorted(used_keys), "production_source_count": len(checked_files), "shared_preflight_before_owner": True}


def verify_invocations(package: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "launcher_invocation": ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v20.py",
        "transport_invocation": ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v20.py",
        "production_evaluator_invocation": ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v20.py",
    }
    for key, executable in expected.items():
        record = package[key]
        require(record["argv"][0] == str(INTERPRETER) and record["argv"][1] == str(executable), f"{key} executable")
        require(record["cwd"] == str(ACTION_ROOT) and record["environment"] == EXPECTED_ENVIRONMENT and record["shell"] is False, f"{key} cwd/environment")
        candidate = dict(record)
        observed = candidate.pop("invocation_sha256")
        require(observed == sha256_bytes(compact_bytes(candidate)), f"{key} self binding")
    return {"invocation_contract_count": 3}


def load_fixture() -> Any:
    sys.path.insert(0, str(ACTION_ROOT / "tools"))
    try:
        return load_module("v20_bound_path_fixture_verify", FIXTURE)
    finally:
        sys.path.pop(0)


def verify_fixture(package: dict[str, Any]) -> dict[str, Any]:
    observed = load_fixture().run_fixture()
    bound, raw = canonical(FIXTURE_REPORT)
    require(observed == bound, "fresh fixture differs from sealed report")
    require(bound["status"] == "PASS_V20_BOUND_CORE_MANIFEST_SYNTHETIC_FIXTURE", "fixture status")
    require(bound["case_count"] == 29 and bound["real_preflight_case_count"] == 8, "fixture case cardinality")
    require(all(case["passed"] for case in bound["cases"]), "fixture case failure")
    cases = {case["name"]: case for case in bound["cases"]}
    stages = {"real_preflight_absent": "PATH_ABSENT", "real_preflight_wrong_path": "CORE_MANIFEST_PATH", "real_preflight_wrong_hash": "PATH_HASH", "real_preflight_wrong_mode": "PATH_MODE", "real_preflight_malformed": "JSON_MALFORMED", "real_preflight_inconsistent": "MANIFEST_INCONSISTENT"}
    for name, stage in stages.items():
        require(cases[name]["failure_stage"] == stage and cases[name]["invocation_calls"] == 0 and cases[name]["owner_created"] is False, f"fixture rejection: {name}")
    require(cases["real_preflight_positive"]["invocation_calls"] == 1, "positive real preflight invocation")
    require(cases["real_preflight_concurrent"]["invocation_calls"] == 1 and cases["real_preflight_concurrent"]["loser_published"] is False, "real preflight race")
    require(bound["max_branch_invocation_count"] == 1 and bound["all_branches_at_most_once"] is True, "fixture at-most-once")
    require(bound["official_payload_open_count"] == 0 and bound["official_target_process_starts"] == 0 and bound["production_mode_exercised"] is False, "fixture inert boundary")
    require(package["synthetic_fixture"]["report_file_sha256"] == sha256_bytes(raw), "package fixture binding")
    return {"case_count": bound["case_count"], "real_preflight_case_count": 8, "report_file_sha256": sha256_bytes(raw)}


def verify_acceptance(package: dict[str, Any], package_raw: bytes, allow_acceptance: bool) -> dict[str, Any]:
    if not allow_acceptance:
        return {"present": False, "permitted_count": 1}
    from jsonschema import Draft202012Validator
    schema, _ = canonical(ACCEPTANCE_SCHEMA)
    acceptance, raw = canonical(ACCEPTANCE)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(acceptance)
    verify_self_hash(acceptance, "acceptance_sha256")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_raw), "acceptance package binding")
    require(acceptance["fixture_report_file_sha256"] == sha256_file(FIXTURE_REPORT), "acceptance fixture binding")
    require(acceptance["core_manifest_closure_entries_sha256"] == package["core_manifest_closure"]["entries_sha256"], "acceptance closure binding")
    require(acceptance["claim_boundary"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY" and acceptance["static_acceptance_grants_execution_authority"] is False, "acceptance authority boundary")
    return {"acceptance_file_sha256": sha256_bytes(raw), "acceptance_self_sha256": acceptance["acceptance_sha256"], "present": True}


def verify_runtime_and_preservation(package: dict[str, Any]) -> dict[str, Any]:
    require(not os.path.lexists(RUNTIME_ROOT), "V20 runtime namespace materialized")
    require(sha256_file(V19_TERMINAL) == V19_TERMINAL_SHA256, "V19 terminal drift")
    namespace = package["runtime_namespace"]
    require(namespace["runtime_root"] == str(RUNTIME_ROOT) and namespace["file_count"] == 0 and namespace["static_acceptance_namespace_absent"] is True, "runtime declaration")
    roots = {item["root"] for item in package["preservation"]}
    require({str(V19_ROOT), str(V19_BUILD_ROOT), str(V19_RUNTIME_ROOT)} <= roots, "V19 preservation roots absent")
    for expected in package["preservation"]:
        require(inventory_digest(Path(expected["root"])) == expected, f"preservation drift: {expected['root']}")
    return {"preservation_root_count": len(package["preservation"]), "runtime_namespace_file_count": 0}


def write_once(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = compact_bytes(report)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short verifier report write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify(allow_acceptance: bool = False) -> dict[str, Any]:
    AUDIT.update({"official_payload_opens": 0, "official_target_starts": 0})
    package, package_raw = verify_manifest()
    report = {
        "acceptance": verify_acceptance(package, package_raw, allow_acceptance),
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_inert_verifier_report",
        "binding_and_selection": verify_binding(package),
        "core_manifest_closure": verify_closure(package),
        "fixture": verify_fixture(package),
        "generated_files": verify_generated_files(package),
        "inventory": verify_inventory(package, allow_acceptance),
        "invocations": verify_invocations(package),
        "manifest_file_sha256": sha256_bytes(package_raw),
        "runtime_and_preservation": verify_runtime_and_preservation(package),
        "sources": verify_sources(package),
        "status": "PASS_V20_BOUND_PATH_REPAIR_POST_ACCEPTANCE_INERT" if allow_acceptance else "PASS_V20_BOUND_PATH_REPAIR_CANDIDATE_INERT",
    }
    require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "inert audit boundary")
    report["claim_boundary"] = "STATIC_ONLY_NO_EXECUTION_AUTHORITY"
    report["official_payload_open_count"] = 0
    report["official_target_process_starts"] = 0
    report["v20_executed"] = False
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-acceptance", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    try:
        report = verify(args.allow_acceptance)
        if args.report is not None:
            write_once(args.report, report)
    except Exception as error:
        sys.stderr.write(f"V20_VERIFY_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
