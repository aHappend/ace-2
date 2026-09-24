#!/usr/bin/env python3
"""Acceptance-aware inert verifier for the immutable V21 static package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_289140ba"
MANIFEST = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_FRESH_L2_ACCEPTANCE_SCHEMA.json"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_V21_PRODUCTION_PREPARATION_TERMINAL_COVERAGE_REPORT.json"
WRAPPER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21.py"
FIXTURE_SOURCE = ACTION_ROOT / "tools/test_qk_gbfp8_head64_v21_preowner_terminal_coverage.py"
V18_BINDING = ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
V20_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root"
V20_BUILD_ROOT = PROJECT_ROOT / "build/v20-bound-path-repair-attempt-0001"
V20_MANIFEST = V20_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json"
V20_ACCEPTANCE = V20_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V20_TERMINAL_OBSERVATION = V20_BUILD_ROOT / "sole-v20-transport-terminal-observation.json"
V20_STDERR = V20_BUILD_ROOT / "sole-v20-transport-stderr.log"

V20_PACKAGE_SHA256 = "e1ab2b55dd07b6cfc9f8540d42eb3fd9b387cb57aadf0704933a29fe6c05c314"
V20_ACCEPTANCE_SHA256 = "516f3007e0bedc00b98748d835b5240fb09b08b189ccc87286aef9ea1bfbe963"
V20_TERMINAL_OBSERVATION_SHA256 = "289140ba983b6808ef0a2bb56381579857ecfc50c35349e96c266ac2b0c75d33"
V20_STDERR_SHA256 = "6470257db40ccc24d46277d34818419f11cc1a4752dba8263d6c742d321a9e4d"
V18_BINDING_SHA256 = "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655"
EXPECTED_NAMES = ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"]

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
    return {
        "entry_count": len(records),
        "file_count": sum(item["kind"] == "file" for item in records),
        "root": str(root),
        "tree_sha256": sha256_bytes(compact_bytes(records)),
    }


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path == OFFICIAL_PAYLOAD.resolve():
            AUDIT["official_payload_opens"] += 1
            raise VerificationError("official payload open prohibited")
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.posix_spawnp"}:
        AUDIT["official_target_starts"] += 1
        raise VerificationError("process start prohibited during inert verification")


def verify_manifest() -> tuple[dict[str, Any], bytes]:
    package, raw = canonical(MANIFEST)
    verify_self_hash(package, "package_content_sha256")
    require(package["action_identity"]["action_id"] == ACTION_ID, "action identity")
    require(package["authoritative_stage"] == "Base", "authoritative Base stage")
    require(package["claim_boundary"]["claim"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "claim boundary")
    require(package["claim_boundary"]["v21_executed"] is False, "V21 execution claim")
    require(package["repair_contract"]["stale_design_stage_ignored"] == "rtl", "stale stage treatment")
    return package, raw


def verify_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    observed = sorted(path.relative_to(ACTION_ROOT).as_posix() for path in ACTION_ROOT.rglob("*") if path.is_file())
    expected = set(package["static_file_policy"]["allowed_relative_files"])
    acceptance_relative = package["static_file_policy"]["acceptance_relative_path"]
    if not allow_acceptance:
        expected.remove(acceptance_relative)
        require(not ACCEPTANCE.exists(), "unexpected V21 acceptance before reviewer")
    else:
        require(ACCEPTANCE.is_file(), "accepted verification requested without acceptance")
    require(observed == sorted(expected), "exact static inventory")
    return {"file_count": len(observed), "acceptance_present": allow_acceptance}


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, record in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"generated file: {relative}")
        require(path.stat().st_size == record["size"] and sha256_file(path) == record["sha256"], f"generated identity: {relative}")
    return {"generated_file_count": len(package["generated_files"])}


def verify_binding_and_fixture(package: dict[str, Any]) -> dict[str, Any]:
    require(sha256_file(V18_BINDING) == V18_BINDING_SHA256, "integrated V18 binding bytes")
    binding, _ = canonical(V18_BINDING)
    verify_self_hash(binding, "binding_table_sha256")
    require(binding["record_count"] == 25 and len(binding["records"]) == 25, "25-record binding table")
    require(binding["selected_tensor_names"] == EXPECTED_NAMES, "three canonical tensors")
    fixture, fixture_raw = canonical(FIXTURE_REPORT)
    require(sha256_bytes(fixture_raw) == package["synthetic_fixture"]["report_file_sha256"], "fixture package binding")
    require(fixture["status"] == "PASS_V21_PRODUCTION_PREPARATION_AND_CANONICAL_TERMINAL_COVERAGE", "fixture status")
    require(fixture["permutation_case_count"] == 6 and fixture["preowner_failure_case_count"] == 6, "fixture coverage cardinality")
    require(len(fixture["negative_cases"]) == 5, "negative binding cardinality")
    require({case["failure_stage"] for case in fixture["preowner_failure_cases"]} == {"LAUNCHER_ARGV", "SCHEMA_SETUP", "CORE_BINDING", "V18_BINDING", "CORE_IMPORT", "PREPARE_EXECUTION"}, "pre-owner stage coverage")
    require(sum(case["primary_write_fault"] for case in fixture["preowner_failure_cases"]) == 1, "primary/fallback coverage")
    require(all(case["duplicate_status"] == "DUPLICATE_REJECTED" for case in fixture["preowner_failure_cases"]), "duplicate loser exclusion")
    require(fixture["official_payload_open_count"] == 0 and fixture["official_target_process_starts"] == 0, "fixture inert boundary")
    return {"binding_record_count": 25, "numerical_tensor_count": 3, "fixture_case_count": fixture["case_count"]}


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise VerificationError(f"missing function: {name}")


def verify_sources() -> dict[str, Any]:
    wrapper_source = WRAPPER.read_text(encoding="utf-8")
    wrapper_tree = ast.parse(wrapper_source)
    prepare = function(wrapper_tree, "prepare_execution")
    run_launcher = function(wrapper_tree, "run_launcher")
    prepare_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(prepare)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    require("resolve_production_selection" in prepare_calls, "production identity resolver call")
    require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "values" for node in ast.walk(prepare)), "mapping-order values use in production preparation")
    require(any(isinstance(node, ast.ExceptHandler) and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "retire_preconsumption_failure" for child in ast.walk(node)) for node in ast.walk(run_launcher)), "caught preparation retirement")
    fixture_source = FIXTURE_SOURCE.read_text(encoding="utf-8")
    require("wrapper.prepare_execution" in fixture_source and "wrapper.run_launcher" in fixture_source, "regression production path binding")
    fixture_tree = ast.parse(fixture_source)
    prohibited_process_calls = []
    for node in ast.walk(fixture_tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if (node.func.value.id, node.func.attr) in {
                ("subprocess", "Popen"), ("subprocess", "run"), ("os", "system"),
                ("os", "posix_spawn"), ("os", "posix_spawnp"),
            }:
                prohibited_process_calls.append(f"{node.func.value.id}.{node.func.attr}")
    require(not prohibited_process_calls, "fixture process-call exclusion")
    return {"production_prepare_identity_bound": True, "caught_preowner_retirement_bound": True}


def verify_invocations(package: dict[str, Any]) -> dict[str, Any]:
    for key in ("launcher_invocation", "transport_invocation", "production_evaluator_invocation"):
        invocation = package[key]
        require(invocation["shell"] is False, f"{key} shell")
        require(invocation["cwd"] == str(ACTION_ROOT), f"{key} cwd")
        require(invocation["environment"] == {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TZ": "UTC"}, f"{key} environment")
        require(invocation["argv"][0] == str(INTERPRETER) and ACTION_ID in invocation["argv"], f"{key} identity")
    return {"invocation_contract_count": 3}


def verify_runtime_and_preservation(package: dict[str, Any]) -> dict[str, Any]:
    require(not os.path.lexists(RUNTIME_ROOT), "official V21 runtime namespace materialized")
    require(package["runtime_namespace"]["runtime_root"] == str(RUNTIME_ROOT), "runtime namespace binding")
    require(sha256_file(V20_MANIFEST) == V20_PACKAGE_SHA256, "V20 package drift")
    require(sha256_file(V20_ACCEPTANCE) == V20_ACCEPTANCE_SHA256, "V20 acceptance drift")
    require(sha256_file(V20_TERMINAL_OBSERVATION) == V20_TERMINAL_OBSERVATION_SHA256, "V20 observation drift")
    require(sha256_file(V20_STDERR) == V20_STDERR_SHA256, "V20 stderr drift")
    roots = {item["root"] for item in package["preservation"]}
    require({str(V20_ROOT), str(V20_BUILD_ROOT)} <= roots, "V20 closure absent from preservation")
    for expected in package["preservation"]:
        require(inventory_digest(Path(expected["root"])) == expected, f"preservation drift: {expected['root']}")
    return {"preservation_root_count": len(package["preservation"]), "runtime_namespace_file_count": 0}


def verify_acceptance(package: dict[str, Any], package_raw: bytes, allow_acceptance: bool) -> dict[str, Any]:
    if not allow_acceptance:
        return {"present": False, "decision": None}
    from jsonschema import Draft202012Validator
    schema, _ = canonical(ACCEPTANCE_SCHEMA)
    acceptance, raw = canonical(ACCEPTANCE)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(acceptance)
    verify_self_hash(acceptance, "acceptance_sha256")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_raw), "acceptance package binding")
    require(acceptance["preparation_fixture_report_file_sha256"] == sha256_file(FIXTURE_REPORT), "acceptance fixture binding")
    require(acceptance["decision"] == "ACCEPT_STATIC_PACKAGE" and acceptance["reviewer_role"] == "Fresh-L2", "acceptance decision")
    require(acceptance["claim_boundary"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY" and acceptance["static_acceptance_grants_execution_authority"] is False, "acceptance authority boundary")
    return {"present": True, "decision": acceptance["decision"], "acceptance_file_sha256": sha256_bytes(raw)}


def verify(allow_acceptance: bool = False) -> dict[str, Any]:
    AUDIT.update({"official_payload_opens": 0, "official_target_starts": 0})
    package, package_raw = verify_manifest()
    report = {
        "acceptance": verify_acceptance(package, package_raw, allow_acceptance),
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_inert_verifier_report",
        "binding_and_fixture": verify_binding_and_fixture(package),
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "generated_files": verify_generated_files(package),
        "inventory": verify_inventory(package, allow_acceptance),
        "invocations": verify_invocations(package),
        "manifest_file_sha256": sha256_bytes(package_raw),
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "runtime_and_preservation": verify_runtime_and_preservation(package),
        "sources": verify_sources(),
        "status": "PASS_V21_POST_ACCEPTANCE_INERT" if allow_acceptance else "PASS_V21_CANDIDATE_INERT_READY_FOR_FRESH_L2",
        "v20_replayed_or_mutated": False,
        "v21_executed": False,
    }
    require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "inert audit boundary")
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-acceptance", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    sys.addaudithook(audit_hook)
    try:
        report = verify(arguments.allow_acceptance)
        if arguments.output is not None:
            write_once(arguments.output, report)
    except Exception as error:
        sys.stderr.write(f"V21_INERT_VERIFY_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
