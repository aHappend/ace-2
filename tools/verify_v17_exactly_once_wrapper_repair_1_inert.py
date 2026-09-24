#!/home/argustest/miniconda3/bin/python3.13
"""Independent inert verifier for V17 exactly-once wrapper repair attempt 0002."""

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
ACTION_ID = "ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
ATTEMPT_0001_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_action_root"
ATTEMPT_0001_BUILD = PROJECT_ROOT / "build/v17-exactly-once-wrapper-4c8a6e31-attempt-0001"
ATTEMPT_0001_VERIFIER = PROJECT_ROOT / "tools/verify_v17_exactly_once_wrapper_inert.py"
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31"
CORE_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root"
CORE_MANIFEST = CORE_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json"
CORE_ACCEPTANCE = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/FRESH_L2_STATIC_ACCEPTANCE.json"
CORE_VERIFIER_REPORT = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/independent-inert-verifier-report.json"
MANIFEST = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json"
ENGINE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17.py"
LAUNCHER = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py"
TRANSPORT = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v17.py"
FIXTURE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v17.py"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_EXACTLY_ONCE_FIXTURE_REPORT.json"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
VERIFIER = PROJECT_ROOT / "tools/verify_v17_exactly_once_wrapper_repair_1_inert.py"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
CORE_WORKER = CORE_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py"
DISPOSITION = "V17_EXACTLY_ONCE_WRAPPER_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY"
EXPECTED_CORE_MANIFEST_SHA256 = "9e120e0e831ec736df45bd1b9f825217ce3368c5e22dbf3e4fc67ac537ea61f1"
EXPECTED_CORE_CONTENT_SHA256 = "dabd719fde0df6bc075ea8f5d34e09606bef464bbd104dfdf6581268a463898d"
EXPECTED_CORE_ACCEPTANCE_SHA256 = "9e8eb7a1a7c4340ab8d7b70a2ee97a132169711659f0ca7608d6701ad3225761"
EXPECTED_CORE_ACCEPTANCE_SELF_SHA256 = "dcb09b60b3a9010a8360f0fb5c0004f86c70be7b94646f2031dfd868e601bd17"
EXPECTED_CORE_VERIFIER_REPORT_SHA256 = "36191b9bc6b0b99ad9bd37519d5982e0efdb4ed61572f3cc69ee1c3526c6c018"
EXPECTED_INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
EXPECTED_ATTEMPT_0001_MANIFEST_SHA256 = "277d08665af369aa1bd75677cc48b83c4be54b2e3387eeb18332c2b772037d4e"
EXPECTED_ATTEMPT_0001_VERIFIER_SHA256 = "906872a4b74ddde22a39fd46f0fd5bfabbb3c3e033362e9e14b70e4f85b5c105"
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


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path == OFFICIAL_PAYLOAD:
            AUDIT["official_payload_opens"] += 1
            raise VerificationError("official payload open prohibited")
    if event == "subprocess.Popen":
        text = repr(args)
        if str(CORE_WORKER) in text or "qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py" in text:
            AUDIT["official_target_starts"] += 1
            raise VerificationError("production target start prohibited")


def inventory_digest(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"preservation root absent: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink in preservation root: {path}")
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


def verify_core_bindings(package: dict[str, Any]) -> dict[str, Any]:
    binding = package["core_binding"]
    require(binding == {
        "acceptance_file_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256,
        "acceptance_self_sha256": EXPECTED_CORE_ACCEPTANCE_SELF_SHA256,
        "independent_verifier_report_file_sha256": EXPECTED_CORE_VERIFIER_REPORT_SHA256,
        "manifest_file_sha256": EXPECTED_CORE_MANIFEST_SHA256,
        "package_content_sha256": EXPECTED_CORE_CONTENT_SHA256,
        "root": str(CORE_ROOT),
    }, "core binding record")
    require(sha256_file(CORE_MANIFEST) == EXPECTED_CORE_MANIFEST_SHA256, "core manifest raw hash")
    require(sha256_file(CORE_ACCEPTANCE) == EXPECTED_CORE_ACCEPTANCE_SHA256, "core acceptance raw hash")
    require(sha256_file(CORE_VERIFIER_REPORT) == EXPECTED_CORE_VERIFIER_REPORT_SHA256, "core verifier report raw hash")
    core, _ = canonical(CORE_MANIFEST)
    acceptance, _ = canonical(CORE_ACCEPTANCE)
    require(core["package_content_sha256"] == EXPECTED_CORE_CONTENT_SHA256, "core content self hash")
    require(acceptance["acceptance_sha256"] == EXPECTED_CORE_ACCEPTANCE_SELF_SHA256, "core acceptance self hash")
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "core acceptance authority boundary")
    return {"acceptance_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256, "manifest_sha256": EXPECTED_CORE_MANIFEST_SHA256, "verifier_report_sha256": EXPECTED_CORE_VERIFIER_REPORT_SHA256}


def verify_attempt_0001(package: dict[str, Any]) -> dict[str, Any]:
    binding = package["repair_of_attempt_0001"]
    manifest = ATTEMPT_0001_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json"
    require(binding == {
        "build_root": str(ATTEMPT_0001_BUILD),
        "manifest_file_sha256": EXPECTED_ATTEMPT_0001_MANIFEST_SHA256,
        "root": str(ATTEMPT_0001_ROOT),
        "verifier_file_sha256": EXPECTED_ATTEMPT_0001_VERIFIER_SHA256,
    }, "attempt 0001 repair binding")
    require(sha256_file(manifest) == EXPECTED_ATTEMPT_0001_MANIFEST_SHA256, "attempt 0001 manifest drift")
    require(sha256_file(ATTEMPT_0001_VERIFIER) == EXPECTED_ATTEMPT_0001_VERIFIER_SHA256, "attempt 0001 verifier drift")
    return {"manifest_sha256": EXPECTED_ATTEMPT_0001_MANIFEST_SHA256, "preserved": True, "verifier_sha256": EXPECTED_ATTEMPT_0001_VERIFIER_SHA256}


def verify_manifest() -> tuple[dict[str, Any], bytes]:
    package, raw = canonical(MANIFEST)
    verify_self_hash(package, "package_content_sha256")
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_package", "package kind")
    require(package["root_id"] == ROOT_ID, "root id")
    require(package["action_identity"]["action_id"] == ACTION_ID, "action id")
    require(package["required_disposition"] == DISPOSITION, "required disposition")
    require(package["claim_boundary"] == {
        "authority_materialized": False,
        "credential_materialized": False,
        "execution_authorized": False,
        "invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "owner_claim_materialized": False,
        "result_materialized": False,
        "runtime_namespace_file_count": 0,
        "static_acceptance_materialized": False,
        "v17_executed": False,
    }, "claim boundary")
    require(package["static_acceptance"]["grants_execution_authority"] is False and package["static_acceptance"]["present"] is False, "static acceptance claim")
    require(package["interpreter"] == {
        "path": "/home/argustest/miniconda3/bin/python3.13",
        "sha256": EXPECTED_INTERPRETER_SHA256,
        "version": "3.13.5",
    }, "interpreter binding")
    return package, raw


def verify_package_inventory(package: dict[str, Any]) -> dict[str, Any]:
    require(ACTION_ROOT.is_dir() and not ACTION_ROOT.is_symlink(), "wrapper repair root absent")
    require(stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode) == 0o555, "wrapper repair root mode")
    expected = set(package["generated_files"]) | {MANIFEST.name}
    observed: set[str] = set()
    for path in ACTION_ROOT.rglob("*"):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"wrapper symlink: {path}")
        relative = path.relative_to(ACTION_ROOT).as_posix()
        if path.is_dir():
            require(stat.S_IMODE(info.st_mode) == 0o555, f"wrapper directory mode: {relative}")
        else:
            observed.add(relative)
            require(stat.S_IMODE(info.st_mode) == 0o444, f"wrapper file mode: {relative}")
    require(observed == expected, "wrapper file inventory")
    for relative, record in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file() and sha256_file(path) == record["sha256"] and path.stat().st_size == record["size"], f"generated file binding: {relative}")
    require(not ACCEPTANCE.exists(), "Engineer materialized Fresh-L2 acceptance")
    return {"acceptance_present": False, "file_count": len(observed)}


def verify_runtime(package: dict[str, Any]) -> dict[str, Any]:
    runtime = package["runtime_namespace"]
    require(runtime["runtime_root"] == str(RUNTIME_ROOT), "runtime root binding")
    expected_directories = {Path(path) for path in runtime["precreated_directories"]}
    observed_directories = {RUNTIME_ROOT}
    observed_files: set[Path] = set()
    for path in RUNTIME_ROOT.rglob("*"):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"runtime symlink: {path}")
        if path.is_dir():
            observed_directories.add(path)
            require(stat.S_IMODE(info.st_mode) == 0o700, f"runtime mode: {path}")
        else:
            observed_files.add(path)
    require(observed_directories == expected_directories, "runtime directory inventory")
    require(not observed_files, "runtime files exist")
    for path in runtime["paths"].values():
        require(not os.path.lexists(path), f"live lifecycle path exists: {path}")
    require(not os.path.lexists(runtime["fallback_terminal"]), "fallback terminal exists")
    return {"directory_count": len(observed_directories), "file_count": 0, "live_authority_present": False, "live_credential_present": False, "live_invocation_present": False, "live_owner_claim_present": False}


def verify_preservation(package: dict[str, Any]) -> list[dict[str, Any]]:
    observed = []
    for expected in package["preservation"]:
        current = inventory_digest(Path(expected["root"]))
        require(current == expected, f"preservation drift: {expected['root']}")
        observed.append(current)
    roots = {item["root"] for item in observed}
    require(str(CORE_ROOT) in roots, "accepted core preservation absent")
    require(str(ATTEMPT_0001_ROOT) in roots and str(ATTEMPT_0001_BUILD) in roots, "attempt 0001 preservation absent")
    for expected in package["preserved_files"]:
        path = Path(expected["path"])
        require(path.is_file() and sha256_file(path) == expected["sha256"] and path.stat().st_size == expected["size"], f"preserved file drift: {path}")
    return observed


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise VerificationError(f"function absent: {name}")


def nested_function(parent: ast.FunctionDef, name: str) -> ast.FunctionDef:
    for node in parent.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise VerificationError(f"nested function absent: {parent.name}.{name}")


def calls(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                result.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                result.add(child.func.attr)
    return result


def outer_calls(function_node: ast.FunctionDef) -> set[str]:
    result: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function_node:
                for statement in node.body:
                    self.visit(statement)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                result.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                result.add(node.func.attr)
            self.generic_visit(node)

    Visitor().visit(function_node)
    return result


def call_lines(node: ast.AST, name: str) -> list[int]:
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            called = child.func.id if isinstance(child.func, ast.Name) else child.func.attr if isinstance(child.func, ast.Attribute) else None
            if called == name:
                lines.append(child.lineno)
    return lines


def verify_sources(package: dict[str, Any]) -> dict[str, Any]:
    engine_source = ENGINE.read_text(encoding="ascii")
    launcher_source = LAUNCHER.read_text(encoding="ascii")
    transport_source = TRANSPORT.read_text(encoding="ascii")
    fixture_source = FIXTURE.read_text(encoding="ascii")
    engine_tree = ast.parse(engine_source, filename=str(ENGINE))
    launcher_tree = ast.parse(launcher_source, filename=str(LAUNCHER))
    transport_tree = ast.parse(transport_source, filename=str(TRANSPORT))

    run = function(engine_tree, "run_execution")
    run_calls = calls(run)
    require({"_claim_owner", "durable_create", "durable_unlink", "invoke_once", "payload_reader", "preflight"} <= run_calls, "lifecycle engine call structure")
    require(min(call_lines(run, "_claim_owner")) < min(call_lines(run, "preflight")), "owner claim must precede preflight")
    retire = function(engine_tree, "retire_preconsumption_failure")
    require({"_claim_owner", "_publish_preconsumption_terminal"} <= calls(retire), "retirement helper ownership structure")
    require("subprocess" not in engine_source and "posix_spawn" not in engine_source, "engine contains process implementation")

    launcher_main = function(launcher_tree, "main")
    require("run_launcher" in outer_calls(launcher_main) and "read_only_preflight" not in outer_calls(launcher_main), "launcher main bypasses lifecycle")
    run_launcher_node = function(launcher_tree, "run_launcher")
    require("run_execution" in outer_calls(run_launcher_node), "launcher omits lifecycle engine")
    require("read_only_preflight" not in outer_calls(run_launcher_node) and "prepare_execution" not in outer_calls(run_launcher_node), "launcher performs setup outside lifecycle callback")
    launcher_preflight = nested_function(run_launcher_node, "preflight")
    require({"parse_exact_arguments", "schema_loader", "read_only_preflight", "prepare_execution"} <= calls(launcher_preflight), "launcher preflight integration incomplete")
    run_calls_nodes = [node for node in ast.walk(run_launcher_node) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_execution"]
    require(len(run_calls_nodes) == 1 and len(run_calls_nodes[0].args) >= 4 and isinstance(run_calls_nodes[0].args[3], ast.Name) and run_calls_nodes[0].args[3].id == "preflight", "launcher lifecycle callback ordering")
    require("invoke_production_evaluator" in launcher_source, "launcher does not import accepted bridge behavior")
    require("bind_result_validators" in launcher_source and "frozen_result_schema" in launcher_source, "launcher does not bind exact frozen validators")
    require("evaluate_bundle_once" not in launcher_source, "launcher forks evaluator-call behavior")

    transport_main = function(transport_tree, "main")
    require("run_transport" in outer_calls(transport_main), "transport main bypasses integration")
    run_transport_node = function(transport_tree, "run_transport")
    transport_calls = calls(run_transport_node)
    require({"require", "launcher_hash_provider", "retire_preconsumption_failure", "spawn_process"} <= transport_calls, "transport retirement integration incomplete")
    require("posix_spawn" in transport_source and "system" not in transport_calls and "Popen" not in transport_calls, "transport shell-free contract")
    require("shell=True" not in launcher_source + transport_source, "shell execution present")

    require(str(OFFICIAL_PAYLOAD) not in fixture_source, "fixture reaches official payload")
    require(str(CORE_WORKER) not in fixture_source, "fixture reaches production worker")
    require("runner=lambda" in fixture_source and "adapter.invoke_evaluator" in fixture_source, "fixture does not inject accepted adapter runner")
    require("threading.Barrier" in fixture_source and "loser_published" in fixture_source, "fixture lacks deterministic concurrent start")
    imports = package["frozen_core_imports"]
    for key in ("adapter", "bridge", "result_validation", "worker"):
        require(sha256_file(Path(imports[key]["path"])) == imports[key]["sha256"], f"frozen core import hash: {key}")
    return {
        "engine_sha256": sha256_file(ENGINE),
        "fixture_sha256": sha256_file(FIXTURE),
        "launcher_preflight_is_lifecycle_callback": True,
        "launcher_sha256": sha256_file(LAUNCHER),
        "owner_claim_precedes_preflight": True,
        "transport_retirement_integrated": True,
        "transport_sha256": sha256_file(TRANSPORT),
    }


def load_fixture() -> Any:
    tools = str(ACTION_ROOT / "tools")
    core_tools = str(CORE_ROOT / "tools")
    sys.path.insert(0, tools)
    sys.path.insert(0, core_tools)
    try:
        spec = importlib.util.spec_from_file_location("v17_exactly_once_fixture_repair_inert", FIXTURE)
        require(spec is not None and spec.loader is not None, "fixture module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(core_tools)
        sys.path.remove(tools)


def verify_fixture(package: dict[str, Any]) -> dict[str, Any]:
    observed = load_fixture().run_fixture()
    bound, raw = canonical(FIXTURE_REPORT)
    require(observed == bound, "fresh fixture differs from bound report")
    require(bound["status"] == "PASS_V17_EXACTLY_ONCE_SYNTHETIC_FIXTURE", "fixture status")
    require(bound["case_count"] >= 21 and all(case["passed"] is True for case in bound["cases"]), "fixture cases")
    require(bound["official_payload_open_count"] == 0 and bound["official_target_process_starts"] == 0, "fixture inert counts")
    require(bound["concurrent_start_exercised"] is True and bound["launcher_transport_integration_exercised"] is True, "fixture integration flags")
    names = {case["name"] for case in bound["cases"]}
    require({
        "nominal", "preflight_failure", "authority_failure", "credential_failure", "ledger_failure",
        "credential_consumption_failure", "duplicate_invocation", "concurrent_start", "nonzero_evaluator",
        "decode_failure", "schema_failure", "result_publication_failure", "terminal_primary_fallback",
        "terminal_total_failure", "launcher_cwd_failure", "launcher_setup_failure", "transport_cwd_failure",
        "transport_environment_failure", "transport_argv_failure", "transport_launcher_hash_failure",
        "transport_spawn_failure",
    } <= names, "fixture coverage")
    concurrent = next(case for case in bound["cases"] if case["name"] == "concurrent_start")
    require(concurrent["payload_open_calls"] == 1 and concurrent["invocation_calls"] == 1 and concurrent["loser_published"] is False, "concurrent ownership result")
    require(package["preattempt_synthetic_fixture"]["report_sha256"] == sha256_bytes(raw), "fixture manifest binding")
    return {"case_count": bound["case_count"], "concurrent_loser_published": False, "report_sha256": sha256_bytes(raw), "status": bound["status"]}


def verify_schemas() -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    observed = {}
    for path in sorted((ACTION_ROOT / "reference").glob("*_SCHEMA.json")):
        schema, _ = canonical(path)
        Draft202012Validator.check_schema(schema)
        observed[path.name] = sha256_file(path)
    require(len(observed) == 6, "lifecycle schema count")
    owner, _ = canonical(ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_OWNER_CLAIM_SCHEMA.json")
    require(owner["properties"]["state"]["const"] == "OWNER_CLAIMED" and owner["properties"]["action_retired"]["const"] is True, "owner claim schema")
    terminal, _ = canonical(ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json")
    require("diagnostic" in terminal["$defs"] and terminal["$defs"]["diagnostic"]["properties"]["stdout"]["properties"]["preview_ascii"]["maxLength"] == 160, "bounded diagnostic schema")
    acceptance, _ = canonical(ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FRESH_L2_ACCEPTANCE_SCHEMA.json")
    require(acceptance["properties"]["static_acceptance_grants_execution_authority"]["const"] is False, "acceptance schema authority boundary")
    return observed


def verify_verifier_command(package: dict[str, Any]) -> dict[str, Any]:
    info = os.lstat(VERIFIER)
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), "verifier file identity")
    require(stat.S_IMODE(info.st_mode) == 0o555, "verifier command is not executable immutable")
    require(VERIFIER.read_bytes().splitlines()[0] == b"#!/home/argustest/miniconda3/bin/python3.13", "verifier shebang interpreter")
    record = package["verifier"]
    require(record["path"] == str(VERIFIER) and record["sha256"] == sha256_file(VERIFIER), "verifier file binding")
    require(record["exact_argv"] == [str(VERIFIER)] and record["executable"] is True, "verifier exact command binding")
    return {"exact_argv": record["exact_argv"], "executable": True, "sha256": record["sha256"]}


def write_once(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing verifier report overwrite: {path}")
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


def verify() -> dict[str, Any]:
    package, manifest_raw = verify_manifest()
    attempt_0001 = verify_attempt_0001(package)
    core = verify_core_bindings(package)
    inventory = verify_package_inventory(package)
    runtime = verify_runtime(package)
    preservation = verify_preservation(package)
    schemas = verify_schemas()
    sources = verify_sources(package)
    fixture = verify_fixture(package)
    verifier_command = verify_verifier_command(package)
    require(AUDIT["official_payload_opens"] == 0, "official payload was opened")
    require(AUDIT["official_target_starts"] == 0, "production target was started")
    report = {
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_independent_inert_verifier_report",
        "attempt": "0002_REPAIR_1",
        "attempt_0001": attempt_0001,
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "core_binding": core,
        "fixture": fixture,
        "fresh_l2_acceptance_present": False,
        "invocation_count": 0,
        "manifest_sha256": sha256_bytes(manifest_raw),
        "official_payload_open_count": AUDIT["official_payload_opens"],
        "official_target_process_starts": AUDIT["official_target_starts"],
        "package_inventory": inventory,
        "preservation_root_count": len(preservation),
        "required_disposition": DISPOSITION,
        "runtime": runtime,
        "schemas": schemas,
        "sources": sources,
        "static_acceptance_grants_execution_authority": False,
        "status": "PASS_V17_EXACTLY_ONCE_WRAPPER_REPAIR_INDEPENDENT_INERT_VERIFICATION",
        "v17_executed": False,
        "verifier_command": verifier_command,
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    report = verify()
    if args.report is not None:
        write_once(args.report, report)
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
