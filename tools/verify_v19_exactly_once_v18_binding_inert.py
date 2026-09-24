#!/usr/bin/env python3
"""Acceptance-aware inert verifier for the additive V19 package."""

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
ACTION_ID = "ace2:qk-gbfp8-base-v19:execute-once:fdfc33a1:additive-0001"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_fdfc33a1"
MANIFEST = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_PACKAGE.json"
BINDING_TABLE = ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
FIXTURE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v19.py"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_EXACTLY_ONCE_V18_BINDING_FIXTURE_REPORT.json"
ENGINE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v19.py"
LAUNCHER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v19.py"
WORKER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v19.py"
TRANSPORT = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v19.py"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_FRESH_L2_ACCEPTANCE_SCHEMA.json"

V18_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root"
V18_MANIFEST = V18_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_STATIC_PACKAGE.json"
V18_BINDING = V18_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
V18_ACCEPTANCE = V18_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V18_POST_REPORT = PROJECT_ROOT / "build/v18-c02-binding-repair-static-0001/fresh-l2-post-acceptance-inert-report.json"
V18_ACTION_ID = "ace2:qk-gbfp8-base-v18:static-c02-binding-repair:additive-0001"
V18_MANIFEST_SHA256 = "fdfc33a1443f237fbaca882b13fb977ae3c2ad7f61e427453297d0621f4445c8"
V18_BINDING_SHA256 = "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655"
V18_ACCEPTANCE_SHA256 = "342ab0130d7e132e4e288b9fa349f7e0c5efb87cb35f29cdbc86e3bc3f5e8cf2"
V18_ACCEPTANCE_SELF_SHA256 = "601980bd0a7af5040c1ad07c9ea88e6aad71619326574436175419637fe74e12"
V18_POST_REPORT_SELF_SHA256 = "19d63c0b6610af00a9f031b7f91665496f0c17cdbfccb8cdc593d5a31ca0b3ae"
EXPECTED_INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
EXPECTED_INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
SELECTED_NAMES = ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"]
DISPOSITION = "V19_EXACTLY_ONCE_V18_BINDING_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
OFFICIAL_EVALUATOR = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
OFFICIAL_PARSER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
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
    records: list[dict[str, Any]] = []
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
        "file_count": sum(record["kind"] == "file" for record in records),
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
    if event == "subprocess.Popen":
        rendered = repr(args)
        if "evaluator_worker_v19.py" in rendered or "evaluator_static_v8.py" in rendered:
            AUDIT["official_target_starts"] += 1
            raise VerificationError("official evaluator start prohibited")


def verify_prerequisites(package: dict[str, Any]) -> dict[str, Any]:
    require(sha256_file(V18_MANIFEST) == V18_MANIFEST_SHA256, "V18 manifest raw hash")
    require(sha256_file(V18_BINDING) == V18_BINDING_SHA256, "V18 binding raw hash")
    require(sha256_file(V18_ACCEPTANCE) == V18_ACCEPTANCE_SHA256, "V18 acceptance raw hash")
    v18_manifest, _ = canonical(V18_MANIFEST)
    v18_binding, _ = canonical(V18_BINDING)
    v18_acceptance, _ = canonical(V18_ACCEPTANCE)
    v18_post, _ = canonical(V18_POST_REPORT)
    verify_self_hash(v18_binding, "binding_table_sha256")
    verify_self_hash(v18_acceptance, "acceptance_sha256")
    verify_self_hash(v18_post, "report_sha256")
    require(v18_manifest["action_identity"]["action_id"] == V18_ACTION_ID, "V18 action id")
    require(v18_acceptance["acceptance_sha256"] == V18_ACCEPTANCE_SELF_SHA256, "V18 acceptance self hash")
    require(v18_post["report_sha256"] == V18_POST_REPORT_SELF_SHA256, "V18 post report self hash")
    require(package["v18_binding"] == {
        "acceptance_file_sha256": V18_ACCEPTANCE_SHA256,
        "acceptance_self_sha256": V18_ACCEPTANCE_SELF_SHA256,
        "action_id": V18_ACTION_ID,
        "binding_table_file_sha256": V18_BINDING_SHA256,
        "integrated_binding_table_path": str(BINDING_TABLE),
        "manifest_file_sha256": V18_MANIFEST_SHA256,
        "post_acceptance_report_self_sha256": V18_POST_REPORT_SELF_SHA256,
        "record_count": 25,
        "source_root": str(V18_ROOT),
    }, "manifest V18 binding")
    return {"action_id": V18_ACTION_ID, "binding_table_file_sha256": V18_BINDING_SHA256, "record_count": 25}


def verify_manifest() -> tuple[dict[str, Any], bytes]:
    package, raw = canonical(MANIFEST)
    verify_self_hash(package, "package_content_sha256")
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_package", "package kind")
    require(package["root_id"] == ROOT_ID, "root id")
    require(package["action_identity"] == {
        "action_id": ACTION_ID,
        "additive_successor": True,
        "future_action_id": ACTION_ID,
        "predecessor_action_id": V18_ACTION_ID,
        "predecessor_static_acceptance_bound": True,
        "retry_replay_resume_repair_replacement_permitted": False,
        "v17_modified_or_replayed": False,
        "v18_modified_or_executed": False,
    }, "action identity")
    claim = package["claim_boundary"]
    require(claim["claim"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "claim boundary")
    require(all(claim[key] is False for key in (
        "authority_materialized", "credential_materialized", "execution_authorized", "ledger_materialized",
        "owner_claim_materialized", "result_materialized", "runtime_namespace_materialized", "runtime_terminal_materialized",
        "stage_2_activity", "v19_executed",
    )), "static claim booleans")
    require(claim["evaluator_invocations"] == 0 and claim["official_payload_open_count"] == 0 and claim["official_target_process_starts"] == 0, "static claim counters")
    require(package["required_disposition"] == DISPOSITION, "required disposition")
    require(package["interpreter"] == {"path": str(EXPECTED_INTERPRETER), "sha256": EXPECTED_INTERPRETER_SHA256, "version": "3.13.5"}, "interpreter contract")
    require(sha256_file(EXPECTED_INTERPRETER) == EXPECTED_INTERPRETER_SHA256, "interpreter file hash")
    return package, raw


def verify_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    require(ACTION_ROOT.is_dir() and not ACTION_ROOT.is_symlink(), "action root absent")
    require(stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode) == 0o555, "action root mode")
    policy = package["static_file_policy"]
    acceptance_relative = "review/FRESH_L2_STATIC_ACCEPTANCE.json"
    allowed = set(policy["allowed_relative_files"])
    require(policy["acceptance_relative_path"] == acceptance_relative and policy["optional_before_acceptance"] == [acceptance_relative], "acceptance inventory policy")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in sorted(ACTION_ROOT.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"action root symlink: {path}")
        relative = path.relative_to(ACTION_ROOT).as_posix()
        if stat.S_ISDIR(info.st_mode):
            observed_directories.add(relative)
            require(stat.S_IMODE(info.st_mode) == 0o555, f"directory mode: {relative}")
        else:
            require(stat.S_ISREG(info.st_mode), f"non-regular action entry: {relative}")
            observed_files.add(relative)
            require(stat.S_IMODE(info.st_mode) == 0o444, f"file mode: {relative}")
            require(not relative.endswith((".pyc", ".pyo")), f"generated Python artifact: {relative}")
    expected = set(allowed)
    if allow_acceptance:
        require(ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    else:
        expected.remove(acceptance_relative)
        require(not os.path.lexists(ACCEPTANCE), "unexpected Fresh-L2 acceptance")
        require("review" not in observed_directories, "pre-acceptance review directory exists")
    require(observed_files == expected, f"action root inventory drift: expected={sorted(expected)} observed={sorted(observed_files)}")
    require(not any(item == "live" or item.startswith("live/") for item in observed_directories), "live directory exists")
    return {"acceptance_present": allow_acceptance, "directory_count": len(observed_directories) + 1, "file_count": len(observed_files)}


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, record in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file() and sha256_file(path) == record["sha256"] and path.stat().st_size == record["size"], f"generated file binding: {relative}")
    own_relative = "tools/verify_qk_gbfp8_head64_granularity_sweep_exactly_once_v18_binding_v19.py"
    require(package["generated_files"][own_relative]["sha256"] == sha256_file(Path(__file__)), "verifier self binding")
    return {"generated_file_count": len(package["generated_files"]), "verifier_file_sha256": sha256_file(Path(__file__))}


def verify_binding_and_selection(package: dict[str, Any]) -> dict[str, Any]:
    table, raw = canonical(BINDING_TABLE)
    verify_self_hash(table, "binding_table_sha256")
    require(sha256_bytes(raw) == V18_BINDING_SHA256 and raw == V18_BINDING.read_bytes(), "integrated V18 table bytes")
    require(table["record_count"] == 25 and len(table["records"]) == 25, "complete binding count")
    require(table["selected_tensor_names"] == SELECTED_NAMES, "selected tensor names")
    require(package["numerical_selection"] == {
        "complete_parser_binding_count": 25,
        "evaluator_receives_complete_binding_table": True,
        "numerical_tensor_count": 3,
        "tensor_names": SELECTED_NAMES,
    }, "manifest numerical selection")
    evaluator = OFFICIAL_EVALUATOR.read_text(encoding="utf-8")
    parser = OFFICIAL_PARSER.read_text(encoding="utf-8")
    require('selected = _validated_selected_records(records)' in evaluator, "frozen evaluator selected-record gate")
    require('query = selected[QUERY_RECORD_NAME]' in evaluator and 'key = selected[KEY_RECORD_NAME]' in evaluator and 'oracle = selected[ORACLE_RECORD_NAME]' in evaluator, "frozen evaluator three tensors")
    require('binding = bindings.get(name)' in parser and 'set(binding) != {"dtype", "sha256", "shape"}' in parser, "frozen parser complete exact binding validation")
    return {"binding_table_file_sha256": sha256_bytes(raw), "complete_binding_count": 25, "numerical_tensor_count": 3}


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, f"function cardinality: {name}")
    return matches[0]


def verify_sources(package: dict[str, Any]) -> dict[str, Any]:
    engine_source = ENGINE.read_text(encoding="utf-8")
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    worker_source = WORKER.read_text(encoding="utf-8")
    transport_source = TRANSPORT.read_text(encoding="utf-8")
    engine_tree = ast.parse(engine_source)
    run_source = ast.get_source_segment(engine_source, function(engine_tree, "run_execution")) or ""
    claim_source = ast.get_source_segment(engine_source, function(engine_tree, "_claim_owner")) or ""
    create_source = ast.get_source_segment(engine_source, function(engine_tree, "durable_create")) or ""
    require("os.O_EXCL" in create_source and "os.O_CREAT" in create_source, "atomic durable create")
    require(claim_source.count("durable_create(") == 1 and "_prepare_runtime_namespace(paths)" in claim_source, "atomic owner claim")
    require(run_source.count("payload_reader(") == 1 and run_source.count("invoke_once(") == 1, "single payload/evaluator boundary")
    require(run_source.index('stage = "CREDENTIAL_CONSUMPTION"') < run_source.index('stage = "PAYLOAD_OPEN"') < run_source.index('stage = "EVALUATOR_CALL"'), "consumption order")
    require('return Outcome(counts, True, None, "DUPLICATE_REJECTED", None, False)' in run_source, "loser no-publication return")
    require('status="PREFLIGHT_FAILED_TERMINAL"' in engine_source and 'status="CONSUMED_ORPHAN"' in engine_source, "pre/post consumption retirement")
    require('holder["tensor_bindings"] = binding_table["records"]' in launcher_source, "launcher full binding table")
    require('binding_table["record_count"] == 25' in launcher_source and 'selected_names == EXPECTED_SELECTED_NAMES' in launcher_source, "launcher binding/selection validation")
    require('holder["bridge"].invoke_production_evaluator' in launcher_source, "launcher production evaluator bridge")
    require('envelope["bindings"] == binding_table["records"]' in worker_source, "worker full binding equality")
    require("os.posix_spawn" in transport_source and "shell=False" not in transport_source, "shell-free transport")
    for path in (ENGINE, LAUNCHER, WORKER, TRANSPORT, FIXTURE):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    lifecycle = package["lifecycle"]
    require(lifecycle["authority_create_count_maximum"] == 1 and lifecycle["credential_consumption_count_maximum"] == 1, "authority credential cardinality")
    require(lifecycle["payload_open_count_maximum"] == 1 and lifecycle["invocation_count_maximum"] == 1, "payload evaluator cardinality")
    require(lifecycle["concurrent_loser_disposition"] == "DUPLICATE_REJECTED_NO_PUBLICATION", "loser disposition")
    return {"atomic_owner_claim": True, "full_binding_worker_gate": True, "single_invoke_callsite": True}


def load_fixture() -> Any:
    sys.path.insert(0, str(ACTION_ROOT / "tools"))
    try:
        spec = importlib.util.spec_from_file_location("v19_exactly_once_fixture_verify", FIXTURE)
        require(spec is not None and spec.loader is not None, "fixture import spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def verify_fixture(package: dict[str, Any]) -> dict[str, Any]:
    module = load_fixture()
    observed = module.run_fixture()
    bound, raw = canonical(FIXTURE_REPORT)
    require(observed == bound, "fresh fixture differs from bound report")
    require(bound["status"] == "PASS_V19_EXACTLY_ONCE_SYNTHETIC_FIXTURE", "fixture status")
    require(bound["case_count"] == 21 and all(case["passed"] for case in bound["cases"]), "fixture cases")
    names = {case["name"] for case in bound["cases"]}
    required_cases = {"concurrent_start", "preflight_failure", "launcher_setup_failure", "transport_spawn_failure", "nominal", "schema_failure", "result_publication_failure", "terminal_total_failure"}
    require(required_cases <= names, "fixture required coverage")
    concurrent = next(case for case in bound["cases"] if case["name"] == "concurrent_start")
    require(concurrent["loser_published"] is False and concurrent["invocation_calls"] == 1 and concurrent["payload_open_calls"] == 1, "owner loser race")
    require(bound["max_branch_invocation_count"] == 1 and bound["all_branches_at_most_once"] is True, "branch invocation cardinality")
    require(bound["owner_loser_publication_exclusion_verified"] is True, "loser publication exclusion")
    require(bound["binding_contract"] == {"binding_table_file_sha256": V18_BINDING_SHA256, "complete_binding_count": 25, "numerical_tensor_names": SELECTED_NAMES}, "fixture binding contract")
    require(bound["official_payload_open_count"] == 0 and bound["official_target_process_starts"] == 0 and bound["production_mode_exercised"] is False, "fixture inert boundary")
    require(package["synthetic_fixture"]["report_file_sha256"] == sha256_bytes(raw), "fixture report package binding")
    return {"case_count": bound["case_count"], "max_branch_invocation_count": 1, "report_file_sha256": sha256_bytes(raw)}


def verify_acceptance(package: dict[str, Any], package_raw: bytes, allow_acceptance: bool) -> dict[str, Any]:
    if not allow_acceptance:
        return {"present": False, "permitted_count": 1}
    from jsonschema import Draft202012Validator

    schema, _ = canonical(ACCEPTANCE_SCHEMA)
    Draft202012Validator.check_schema(schema)
    acceptance, raw = canonical(ACCEPTANCE)
    Draft202012Validator(schema).validate(acceptance)
    verify_self_hash(acceptance, "acceptance_sha256")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_raw), "acceptance package binding")
    require(acceptance["fixture_report_file_sha256"] == sha256_file(FIXTURE_REPORT), "acceptance fixture binding")
    require(acceptance["claim_boundary"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "acceptance claim")
    require(acceptance["static_acceptance_grants_execution_authority"] is False and acceptance["v19_executed"] is False, "acceptance authority boundary")
    require(package["static_acceptance"]["inventory_policy"] == "EXACT_STATIC_FILES_PLUS_ZERO_OR_ONE_BOUND_FRESH_L2_ACCEPTANCE", "acceptance policy")
    return {"acceptance_file_sha256": sha256_bytes(raw), "acceptance_self_sha256": acceptance["acceptance_sha256"], "present": True}


def verify_runtime_and_preservation(package: dict[str, Any]) -> dict[str, Any]:
    require(not os.path.lexists(RUNTIME_ROOT), "V19 runtime namespace materialized")
    namespace = package["runtime_namespace"]
    require(namespace["runtime_root"] == str(RUNTIME_ROOT) and namespace["file_count"] == 0, "runtime namespace declaration")
    require(namespace["static_acceptance_namespace_absent"] is True, "runtime static absence")
    require(namespace["provisioning"] == "LAZY_PRIVATE_DIRECTORY_CREATION_BEFORE_ATOMIC_OWNER_CLAIM", "runtime provisioning")
    for expected in package["preservation"]:
        require(inventory_digest(Path(expected["root"])) == expected, f"preservation drift: {expected['root']}")
    require(package["v17_exactly_once_precedent"]["tree_sha256"] == "3b3666553b71ae081fa4a39202204ff78f97f767c08266f12c12e6ce3bed1545", "V17 precedent tree")
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
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_inert_verifier_report",
        "binding_and_selection": verify_binding_and_selection(package),
        "fixture": verify_fixture(package),
        "generated_files": verify_generated_files(package),
        "inventory": verify_inventory(package, allow_acceptance),
        "manifest_file_sha256": sha256_bytes(package_raw),
        "prerequisites": verify_prerequisites(package),
        "runtime_and_preservation": verify_runtime_and_preservation(package),
        "sources": verify_sources(package),
        "status": "PASS_V19_EXACTLY_ONCE_V18_BINDING_POST_ACCEPTANCE_INERT" if allow_acceptance else "PASS_V19_EXACTLY_ONCE_V18_BINDING_CANDIDATE_INERT",
    }
    require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "inert audit boundary")
    report["official_payload_open_count"] = 0
    report["official_target_process_starts"] = 0
    report["v19_executed"] = False
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
        sys.stderr.write(f"V19_VERIFY_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
