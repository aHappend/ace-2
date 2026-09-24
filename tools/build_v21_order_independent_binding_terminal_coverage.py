#!/usr/bin/env python3
"""Build the additive V21 static binding-order and terminal-coverage repair."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v20_bound_path_repair as v20


PROJECT_ROOT = Path("/home/argustest/ace-2")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_289140ba"
BUILD_ROOT = PROJECT_ROOT / "build/v21-order-independent-binding-terminal-coverage-attempt-0001"
MANIFEST_NAME = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json"
DISPOSITION = "V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY"

V20_ROOT = v20.ACTION_ROOT
V20_MANIFEST = V20_ROOT / v20.MANIFEST_NAME
V20_BUILD_ROOT = v20.BUILD_ROOT
V20_ACTION_ID = v20.ACTION_ID
V20_PACKAGE_SHA256 = "e1ab2b55dd07b6cfc9f8540d42eb3fd9b387cb57aadf0704933a29fe6c05c314"
V20_ACCEPTANCE = V20_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V20_ACCEPTANCE_SHA256 = "516f3007e0bedc00b98748d835b5240fb09b08b189ccc87286aef9ea1bfbe963"
V20_TERMINAL_OBSERVATION = V20_BUILD_ROOT / "sole-v20-transport-terminal-observation.json"
V20_TERMINAL_OBSERVATION_SHA256 = "289140ba983b6808ef0a2bb56381579857ecfc50c35349e96c266ac2b0c75d33"
V20_STDERR = V20_BUILD_ROOT / "sole-v20-transport-stderr.log"
V20_STDERR_SHA256 = "6470257db40ccc24d46277d34818419f11cc1a4752dba8263d6c742d321a9e4d"

DESIGN_ARTIFACT = PROJECT_ROOT / "design/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_REPLACEMENT_BATCH.json"
DESIGN_SHA_FILE = DESIGN_ARTIFACT.with_suffix(".sha256")
V18_BINDING = v20.V18_BINDING
V18_BINDING_SHA256 = v20.base.V18_BINDING_SHA256
CORE_PACKAGE = v20.CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
CORE_PACKAGE_SHA256 = "3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832"
RESOLVER_SOURCE = PROJECT_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_v21_binding_resolution.py"
TERMINAL_FIXTURE_SOURCE = PROJECT_ROOT / "tools/test_qk_gbfp8_head64_v21_preowner_terminal_coverage.py"
ROOT_VERIFIER_SOURCE = PROJECT_ROOT / "tools/verify_v21_order_independent_binding_terminal_coverage_inert.py"
FIXTURE_REPORT_NAME = "SYNTHETIC_V21_PRODUCTION_PREPARATION_TERMINAL_COVERAGE_REPORT.json"
LEGACY_FIXTURE_REPORT_NAME = "SYNTHETIC_V21_EXACTLY_ONCE_ENGINE_FIXTURE_REPORT.json"
ACCEPTANCE_SCHEMA_NAME = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_FRESH_L2_ACCEPTANCE_SCHEMA.json"
EXACT_ENVIRONMENT = v20.EXACT_ENVIRONMENT


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


compact_bytes = v20.compact_bytes
sha256_bytes = v20.sha256_bytes
sha256_file = v20.sha256_file
canonical = v20.canonical
sealed = v20.sealed
write_bytes = v20.write_bytes
write_json = v20.write_json
inventory_digest = v20.inventory_digest
invocation_contract = v20.invocation_contract


def rewrite_strings(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [rewrite_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_strings(item, replacements) for key, item in value.items()}
    return value


def replacements() -> list[tuple[str, str]]:
    return [
        (str(V20_ROOT), str(ACTION_ROOT)),
        (str(v20.RUNTIME_ROOT), str(RUNTIME_ROOT)),
        (V20_ACTION_ID, ACTION_ID),
        (v20.MANIFEST_NAME, MANIFEST_NAME),
        ("V20_BOUND_PATH_REPAIR", "V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE"),
        ("v20_bound_path_repair", "v21_order_independent_binding_terminal_coverage"),
        ("EXECUTION_V20_EXACTLY_ONCE", "EXECUTION_V21_EXACTLY_ONCE"),
        ("execution_v20_exactly_once", "execution_v21_exactly_once"),
        ("V20", "V21"),
        ("v20", "v21"),
    ]


def v21ize(source: str) -> str:
    for old, new in replacements():
        source = source.replace(old, new)
    return source


def acceptance_schema() -> dict[str, Any]:
    sha = {"pattern": "^[0-9a-f]{64}$", "type": "string"}
    properties = {
        "acceptance_sha256": sha,
        "accepted": {"const": True},
        "action_id": {"const": ACTION_ID},
        "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_fresh_l2_static_acceptance"},
        "claim_boundary": {"const": "STATIC_ONLY_NO_EXECUTION_AUTHORITY"},
        "decision": {"const": "ACCEPT_STATIC_PACKAGE"},
        "execution_package_file_sha256": sha,
        "official_payload_open_count": {"const": 0},
        "official_target_process_starts": {"const": 0},
        "preparation_fixture_report_file_sha256": sha,
        "required_disposition": {"const": DISPOSITION},
        "reviewer_role": {"const": "Fresh-L2"},
        "runtime_namespace_file_count": {"const": 0},
        "static_acceptance_grants_execution_authority": {"const": False},
        "v18_binding_table_file_sha256": {"const": V18_BINDING_SHA256},
        "v20_acceptance_file_sha256": {"const": V20_ACCEPTANCE_SHA256},
        "v20_execution_package_file_sha256": {"const": V20_PACKAGE_SHA256},
        "v20_terminal_observation_file_sha256": {"const": V20_TERMINAL_OBSERVATION_SHA256},
        "v21_executed": {"const": False},
    }
    return {
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_FRESH_L2_ACCEPTANCE_SCHEMA",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
        "type": "object",
    }


def transformed_schemas(staging: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for source_path in sorted((V20_ROOT / "reference").glob("*.json")):
        if source_path.name.endswith("FRESH_L2_ACCEPTANCE_SCHEMA.json"):
            continue
        name = v21ize(source_path.name)
        value = rewrite_strings(json.loads(source_path.read_text(encoding="ascii")), replacements())
        if name.endswith("EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json"):
            value["$defs"]["wrapper_diagnostic"]["properties"]["failure_stage"]["pattern"] = "^[A-Z0-9_]+$"
        path = staging / "reference" / name
        write_json(path, value)
        hashes[name] = sha256_file(path)
    acceptance_path = staging / "reference" / ACCEPTANCE_SCHEMA_NAME
    write_json(acceptance_path, acceptance_schema())
    hashes[ACCEPTANCE_SCHEMA_NAME] = sha256_file(acceptance_path)
    return hashes


def patch_launcher(source: str, schema_hashes: dict[str, str]) -> str:
    source = v21ize(source)
    source = source.replace(
        "from qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v21 import verify_core_manifest_closure\n",
        "from qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v21 import verify_core_manifest_closure\n"
        "import qk_gbfp8_head64_granularity_sweep_v21_binding_resolution as binding_resolution\n",
    )
    source = source.replace("    run_execution,\n", "    retire_preconsumption_failure,\n    run_execution,\n")
    old_schema = re.search(r"SCHEMA_HASHES = \{.*?\n\}\nPATHS", source, re.S)
    require(old_schema is not None, "V21 launcher schema block")
    source = source[:old_schema.start()] + f"SCHEMA_HASHES = {json.dumps(schema_hashes, sort_keys=True, indent=4)}\nPATHS" + source[old_schema.end():]
    start = source.index("def prepare_execution(")
    end = source.index("\n\ndef read_payload_once", start)
    preparation = '''def load_binding_table() -> dict[str, Any]:
    binding_table, binding_raw = canonical(BINDING_TABLE)
    verify_self_hash(binding_table, "binding_table_sha256")
    require(sha256_bytes(binding_raw) == EXPECTED_V18_BINDING_SHA256, "integrated V18 binding hash", "V18_BINDING")
    return binding_table


def prepare_execution(
    holder: dict[str, Any],
    module_loader: Callable[[str, Path], Any],
    *,
    binding_table_loader: Callable[[], dict[str, Any]] = load_binding_table,
) -> None:
    package = holder["package"]
    accepted = holder["core_package"]
    selected_records = accepted["official_benchmark"]["input_bindings"]["tensor_records"]
    binding_table = binding_table_loader()
    selection = binding_resolution.resolve_production_selection(
        selected_records, binding_table, tuple(EXPECTED_SELECTED_NAMES)
    )
    holder["tensor_bindings"] = selection.binding_records
    holder["numerical_tensor_names"] = list(selection.canonical_names)
    holder["resolved_tensor_records"] = selection.records_by_name
    holder["resolved_source_keys_by_name"] = selection.source_keys_by_name
    holder["input_bindings"] = accepted["official_benchmark"]["input_bindings"]
    holder["official"] = package["official_identities"]
    try:
        holder["adapter"] = module_loader(
            "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v21",
            CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v21.py",
        )
        holder["bridge"] = module_loader(
            "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v21",
            CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v21.py",
        )
        result_validation = module_loader(
            "qk_gbfp8_head64_granularity_sweep_result_validation_v21",
            CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_result_validation_v21.py",
        )
        result_schema = result_validation.frozen_result_schema()
        holder["result_validation"] = result_validation
        holder["result_validator"] = result_validation.construct_result_schema_validator(result_schema)
    except binding_resolution.BindingResolutionError:
        raise
    except BaseException as error:
        raise LauncherError(f"production module import/setup: {type(error).__name__}: {error}", "CORE_IMPORT") from error
'''
    source = source[:start] + preparation + source[end:]
    start = source.index("def run_launcher(\n")
    end = source.index("\n\ndef main()", start)
    run_launcher = '''def run_launcher(
    *,
    paths: RuntimePaths = PATHS,
    argv: list[str] | None = None,
    observed_cwd: Path | None = None,
    observed_environment: dict[str, str] | None = None,
    schema_loader: Callable[[], dict[str, Any]] = load_schemas,
    module_loader: Callable[[str, Path], Any] = load_module,
    preflight_reader: Callable[..., dict[str, Any]] = read_only_preflight,
    preparer: Callable[..., None] = prepare_execution,
    binding_table_loader: Callable[[], dict[str, Any]] = load_binding_table,
    preowner_faults: frozenset[str] = frozenset(),
) -> Outcome:
    validators: dict[str, Any] = {}
    stage = "LAUNCHER_ARGV"
    try:
        raw_argv = list(sys.argv[1:] if argv is None else argv)
        arguments = parse_exact_arguments(raw_argv)
        stage = "SCHEMA_SETUP"
        validators = schema_loader()
        stage = "PREFLIGHT"
        holder = preflight_reader(
            arguments, validators, paths, observed_cwd=observed_cwd, observed_environment=observed_environment
        )
        stage = "PREPARE_EXECUTION"
        preparer(holder, module_loader, binding_table_loader=binding_table_loader)
    except BaseException as error:
        diagnostic_stage = getattr(error, "failure_stage", stage)
        if not isinstance(diagnostic_stage, str) or not 1 <= len(diagnostic_stage) <= 64:
            diagnostic_stage = stage
        return retire_preconsumption_failure(
            paths, ACTION_ID, error, diagnostic_stage,
            terminal_validator=validators.get("terminal"), faults=preowner_faults,
        )

    package = holder["package"]
    bindings = {
        "action_id": ACTION_ID,
        "binding_record_count": 25,
        "numerical_tensor_names": holder["numerical_tensor_names"],
        "v18_acceptance_file_sha256": EXPECTED_V18_ACCEPTANCE_SHA256,
        "v18_acceptance_self_sha256": EXPECTED_V18_ACCEPTANCE_SELF_SHA256,
        "v18_action_id": ''' + repr(v20.base.V18_ACTION_ID) + ''',
        "v18_binding_table_file_sha256": EXPECTED_V18_BINDING_SHA256,
        "v18_manifest_file_sha256": EXPECTED_V18_MANIFEST_SHA256,
        "v18_post_acceptance_report_self_sha256": EXPECTED_V18_POST_REPORT_SELF_SHA256,
        "core_acceptance_file_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256,
        "core_manifest_file_sha256": EXPECTED_CORE_MANIFEST_SHA256,
        "execution_package_file_sha256": holder["package_sha256"],
        "fresh_l2_acceptance_sha256": holder["acceptance_sha256"],
        "interpreter_sha256": EXPECTED_INTERPRETER_SHA256,
        "invocation_sha256": package["production_evaluator_invocation"]["invocation_sha256"],
        "launcher_invocation_sha256": package["launcher_invocation"]["invocation_sha256"],
        "official_identities": holder["official"],
        "transport_attestation": holder["transport_attestation"],
    }

    def payload_reader() -> bytes:
        return read_payload_once(holder["official"]["tensor_bundle_sha256"], package["official_payload"]["byte_count"])

    def invoke_once(payload: bytes, authority_sha256: str, ledger_sha256: str, publish_result: Any) -> Any:
        official = holder["official"]
        result_validation = holder["result_validation"]
        evaluator_sha256 = package["core_manifest_closure"]["official_evaluator_sha256"]
        result_bindings = result_validation.ResultBindings(
            acceptance_sha256=holder["acceptance_sha256"], authority_sha256=authority_sha256,
            evaluator_sha256=evaluator_sha256, invocation_sha256=package["production_evaluator_invocation"]["invocation_sha256"],
            ledger_sha256=ledger_sha256, model_identity_sha256=official["model_sha256"],
            package_sha256=holder["core_package_sha256"], tensor_bundle_sha256=official["tensor_bundle_sha256"],
        )
        bound = result_validation.bind_result_validators(holder["result_validator"], result_bindings, ACTION_ID)
        context = {
            "authority_sha256": authority_sha256, "consumed_ledger_sha256": ledger_sha256,
            "evaluator_sha256": evaluator_sha256, "fresh_l2_acceptance_sha256": holder["acceptance_sha256"],
            "input_bindings": holder["input_bindings"],
            "invocation_sha256": package["production_evaluator_invocation"]["invocation_sha256"],
            "irreversible_action_id": ACTION_ID, "model_identity_sha256": official["model_sha256"],
            "package_sha256": holder["core_package_sha256"],
        }
        return holder["bridge"].invoke_production_evaluator(
            payload, holder["tensor_bindings"], context, bound.validate_result_schema,
            bound.validate_result_record, publish_result,
        )

    return run_execution(paths, bindings, validators, lambda: None, payload_reader, invoke_once)
'''
    return source[:start] + run_launcher + source[end:]


def patch_transport(source: str, launcher_hash: str) -> str:
    source = v21ize(source)
    old_hash = re.search(r"LAUNCHER_SHA256 = '[0-9a-f]{64}'", source)
    require(old_hash is not None, "V21 transport launcher hash")
    return source[:old_hash.start()] + f"LAUNCHER_SHA256 = {launcher_hash!r}" + source[old_hash.end():]


def patch_legacy_fixture(source: str) -> str:
    source = v21ize(source)
    start = source.index("def launcher_cwd_failure(")
    end = source.index("\n\ndef transport_failure", start)
    replacement = '''def launcher_cwd_failure(paths: RuntimePaths) -> dict[str, Any]:
    outcome = launcher.run_launcher(
        paths=paths, argv=launcher.expected_launcher_arguments(), observed_cwd=Path("/synthetic-wrong-cwd"),
        observed_environment=dict(launcher.EXPECTED_ENVIRONMENT), schema_loader=validators,
    )
    if outcome.terminal_path is None:
        return {"passed": False, "failure_stage": "NO_TERMINAL", "status": outcome.status}
    terminal = read_json(Path(outcome.terminal_path))
    return {
        "passed": outcome.status == "PREFLIGHT_FAILED_TERMINAL" and terminal["wrapper_diagnostic"]["failure_stage"] == "LAUNCHER_CWD" and paths.owner.exists() and not paths.authority.exists(),
        "failure_stage": terminal["wrapper_diagnostic"]["failure_stage"],
        "status": outcome.status,
    }


def launcher_setup_failure(paths: RuntimePaths) -> dict[str, Any]:
    def fail_schema_setup() -> dict[str, Any]:
        raise RuntimeError("synthetic schema setup")
    outcome = launcher.run_launcher(
        paths=paths, argv=launcher.expected_launcher_arguments(), observed_cwd=launcher.ROOT,
        observed_environment=dict(launcher.EXPECTED_ENVIRONMENT), schema_loader=fail_schema_setup,
    )
    if outcome.terminal_path is None:
        return {"passed": False, "failure_stage": "NO_TERMINAL", "status": outcome.status}
    terminal = read_json(Path(outcome.terminal_path))
    return {
        "passed": outcome.status == "PREFLIGHT_FAILED_TERMINAL" and terminal["wrapper_diagnostic"]["failure_stage"] == "SCHEMA_SETUP" and paths.owner.exists() and not paths.authority.exists(),
        "failure_stage": terminal["wrapper_diagnostic"]["failure_stage"],
        "status": outcome.status,
    }
'''
    return source[:start] + replacement + source[end:]


def run_fixture(path: Path, cwd: Path, expected_status: str) -> bytes:
    completed = subprocess.run(
        [str(INTERPRETER), str(path)], cwd=cwd, env=dict(EXACT_ENVIRONMENT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    require(
        completed.returncode == 0 and completed.stderr == b"",
        f"fixture failed: {path.name}: rc={completed.returncode} stderr={completed.stderr!r} stdout={completed.stdout!r}",
    )
    report = json.loads(completed.stdout.decode("ascii", "strict"))
    require(compact_bytes(report) == completed.stdout, f"fixture canonical: {path.name}")
    require(report["status"] == expected_status, f"fixture status: {path.name}")
    return completed.stdout


def runtime_directories() -> list[str]:
    return [
        str(RUNTIME_ROOT), str(RUNTIME_ROOT / "primary"), str(RUNTIME_ROOT / "primary/authority"),
        str(RUNTIME_ROOT / "primary/authority/base"), str(RUNTIME_ROOT / "primary/result"),
        str(RUNTIME_ROOT / "primary/result/base"), str(RUNTIME_ROOT / "fallback"),
    ]


def build() -> dict[str, Any]:
    require(not os.path.lexists(ACTION_ROOT), f"immutable V21 action root already exists: {ACTION_ROOT}")
    require(not os.path.lexists(BUILD_ROOT), f"immutable V21 build root already exists: {BUILD_ROOT}")
    require(not os.path.lexists(RUNTIME_ROOT), f"V21 runtime namespace already exists: {RUNTIME_ROOT}")
    require(sha256_file(V20_MANIFEST) == V20_PACKAGE_SHA256, "V20 package identity")
    require(sha256_file(V20_ACCEPTANCE) == V20_ACCEPTANCE_SHA256, "V20 acceptance identity")
    require(sha256_file(V20_TERMINAL_OBSERVATION) == V20_TERMINAL_OBSERVATION_SHA256, "V20 terminal observation identity")
    require(sha256_file(V20_STDERR) == V20_STDERR_SHA256, "V20 stderr identity")
    require(sha256_file(V18_BINDING) == V18_BINDING_SHA256, "V18 binding identity")
    require(sha256_file(CORE_PACKAGE) == CORE_PACKAGE_SHA256, "frozen core package identity")
    design_raw = DESIGN_ARTIFACT.read_bytes()
    require(DESIGN_SHA_FILE.read_text(encoding="ascii").strip().split()[0] == sha256_bytes(design_raw), "V21 design sidecar")
    design = json.loads(design_raw.decode("ascii", "strict"))
    require(design["replacement_nodes"][0]["new_identity"]["action_id"] == ACTION_ID, "V21 design action identity")

    v20_package, _ = canonical(V20_MANIFEST)
    preservation = copy.deepcopy(v20_package["preservation"])
    preservation.extend([inventory_digest(V20_ROOT), inventory_digest(V20_BUILD_ROOT)])
    require(len({item["root"] for item in preservation}) == len(preservation), "preservation roots unique")
    before = {item["root"]: item for item in preservation}

    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{ROOT_ID}.staging-", dir=PROJECT_ROOT / "reference"))
    try:
        assert staging is not None
        (staging / "review").mkdir(parents=True)
        schema_hashes = transformed_schemas(staging)
        for source_path in sorted((V20_ROOT / "tools").glob("*.py")):
            if source_path.name in {
                "qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v20.py",
                "qk_gbfp8_head64_granularity_sweep_transport_wrapper_v20.py",
                "verify_qk_gbfp8_head64_granularity_sweep_bound_path_repair_v20.py",
            }:
                continue
            source = source_path.read_text(encoding="utf-8")
            if source_path.name == "qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v20.py":
                source = patch_legacy_fixture(source)
            else:
                source = v21ize(source)
            name = v21ize(source_path.name)
            compile(source, name, "exec")
            write_bytes(staging / "tools" / name, source.encode("utf-8"))

        write_bytes(staging / "tools/qk_gbfp8_head64_granularity_sweep_v21_binding_resolution.py", RESOLVER_SOURCE.read_bytes())
        launcher = patch_launcher((V20_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v20.py").read_text(encoding="utf-8"), schema_hashes)
        compile(launcher, "launcher_v21", "exec")
        launcher_path = staging / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21.py"
        write_bytes(launcher_path, launcher.encode("utf-8"))
        transport = patch_transport((V20_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v20.py").read_text(encoding="utf-8"), sha256_file(launcher_path))
        compile(transport, "transport_v21", "exec")
        write_bytes(staging / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v21.py", transport.encode("utf-8"))
        write_bytes(staging / "tools/test_qk_gbfp8_head64_v21_preowner_terminal_coverage.py", TERMINAL_FIXTURE_SOURCE.read_bytes())
        write_bytes(staging / "tools/verify_qk_gbfp8_head64_granularity_sweep_order_independent_binding_terminal_coverage_v21.py", ROOT_VERIFIER_SOURCE.read_bytes())
        write_bytes(staging / "bindings/C02_EXACT_BINDINGS_25.json", V18_BINDING.read_bytes())

        legacy_fixture_path = staging / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v21.py"
        legacy_raw = run_fixture(legacy_fixture_path, staging, "PASS_V21_BOUND_CORE_MANIFEST_SYNTHETIC_FIXTURE")
        write_bytes(staging / "evidence" / LEGACY_FIXTURE_REPORT_NAME, legacy_raw)
        preparation_fixture_path = staging / "tools/test_qk_gbfp8_head64_v21_preowner_terminal_coverage.py"
        preparation_raw = run_fixture(
            preparation_fixture_path, staging,
            "PASS_V21_PRODUCTION_PREPARATION_AND_CANONICAL_TERMINAL_COVERAGE",
        )
        preparation_report_path = staging / "evidence" / FIXTURE_REPORT_NAME
        write_bytes(preparation_report_path, preparation_raw)
        preparation_report = json.loads(preparation_raw.decode("ascii", "strict"))
        require(preparation_report["permutation_case_count"] == 6, "V21 permutation case count")
        require(preparation_report["preowner_failure_case_count"] == 6, "V21 pre-owner case count")
        require(preparation_report["official_payload_open_count"] == 0, "V21 fixture payload boundary")
        require(preparation_report["official_target_process_starts"] == 0, "V21 fixture target boundary")

        relative_files = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        generated_files = {
            relative: {"sha256": sha256_file(staging / relative), "size": (staging / relative).stat().st_size}
            for relative in relative_files
        }
        paths = {
            "owner": str(RUNTIME_ROOT / "primary/authority/base/owner-claim.json"),
            "authority": str(RUNTIME_ROOT / "primary/authority/base/authority.json"),
            "credential": str(RUNTIME_ROOT / "primary/authority/base/credential.json"),
            "ledger": str(RUNTIME_ROOT / "primary/authority/base/authority-ledger.json"),
            "result": str(RUNTIME_ROOT / "primary/result/base/result.json"),
            "first_terminal": str(RUNTIME_ROOT / "primary/authority/base/first-terminal.json"),
        }
        launcher_invocation = invocation_contract([
            str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21.py"),
            "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
            "--irreversible-action-id", ACTION_ID,
        ], ACTION_ROOT)
        transport_invocation = invocation_contract([
            str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v21.py"),
            "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
            "--irreversible-action-id", ACTION_ID,
        ], ACTION_ROOT)
        production_invocation = invocation_contract([
            str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v21.py"),
            "--mode", "production", "--package", str(ACTION_ROOT / MANIFEST_NAME),
            "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
            "--irreversible-action-id", ACTION_ID,
        ], ACTION_ROOT)
        manifest = {
            "action_identity": {
                "action_id": ACTION_ID, "additive_successor": True, "future_action_id": ACTION_ID,
                "predecessor_action_id": V20_ACTION_ID, "retired_predecessor_terminal_observation_sha256": V20_TERMINAL_OBSERVATION_SHA256,
                "retry_replay_resume_repair_replacement_permitted": False, "v20_modified_replayed_or_invoked": False,
            },
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_package",
            "attempt": "0001",
            "authoritative_stage": "Base",
            "claim_boundary": {
                "claim": "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "authority_materialized": False,
                "credential_materialized": False, "evaluator_invocations": 0, "execution_authorized": False,
                "ledger_materialized": False, "official_payload_open_count": 0, "official_target_process_starts": 0,
                "owner_claim_materialized": False, "result_materialized": False, "runtime_namespace_materialized": False,
                "runtime_terminal_materialized": False, "v21_executed": False,
            },
            "core_manifest_closure": copy.deepcopy(v20_package["core_manifest_closure"]),
            "frozen_benchmark_contract": copy.deepcopy(design["frozen_benchmark_contract"]),
            "generated_files": generated_files,
            "interpreter": copy.deepcopy(v20_package["interpreter"]),
            "launcher_invocation": launcher_invocation,
            "lifecycle": {
                **copy.deepcopy(v20_package["lifecycle"]),
                "pre_owner_preparation_failure": "OWNER_ONLY_CANONICAL_PRIMARY_CREATE_THEN_FALLBACK_CREATE_WITH_ZERO_AUTHORITY_CREDENTIAL_PAYLOAD_EVALUATOR_OR_RESULT",
            },
            "numerical_selection": copy.deepcopy(v20_package["numerical_selection"]),
            "official_identities": copy.deepcopy(v20_package["official_identities"]),
            "official_payload": copy.deepcopy(v20_package["official_payload"]),
            "preservation": preservation,
            "production_evaluator_invocation": production_invocation,
            "repair_contract": {
                "binding_resolution": "EXACT_TENSOR_NAME_IDENTITY_THEN_CANONICAL_V18_NAME_ORDER",
                "design_artifact_path": str(DESIGN_ARTIFACT),
                "design_artifact_sha256": sha256_bytes(design_raw),
                "preowner_terminal_coverage": ["LAUNCHER_ARGV", "SCHEMA_SETUP", "CORE_BINDING", "V18_BINDING", "CORE_IMPORT", "PREPARE_EXECUTION"],
                "production_and_regression_prepare_execution_same_function": True,
                "stale_design_stage_ignored": "rtl",
            },
            "required_disposition": DISPOSITION,
            "root_id": ROOT_ID,
            "runtime_namespace": {
                "fallback_terminal": str(RUNTIME_ROOT / "fallback/first-terminal.json"), "file_count": 0,
                "paths": paths, "precreated_directories": runtime_directories(),
                "provisioning": "LAZY_PRIVATE_DIRECTORY_CREATION_ONLY_ON_LATER_AUTHORIZED_LAUNCHER_ENTRY",
                "runtime_root": str(RUNTIME_ROOT), "static_acceptance_namespace_absent": True,
            },
            "schema_version": 1,
            "static_acceptance": {
                "acceptance_relative_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json", "grants_execution_authority": False,
                "inventory_policy": "EXACT_STATIC_FILES_PLUS_ZERO_OR_ONE_BOUND_FRESH_L2_ACCEPTANCE", "present": False,
                "schema_path": str(ACTION_ROOT / "reference" / ACCEPTANCE_SCHEMA_NAME),
            },
            "static_file_policy": {
                "acceptance_relative_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json",
                "allowed_relative_files": sorted([MANIFEST_NAME, "review/FRESH_L2_STATIC_ACCEPTANCE.json", *generated_files]),
                "optional_before_acceptance": ["review/FRESH_L2_STATIC_ACCEPTANCE.json"],
            },
            "synthetic_fixture": {
                "case_count": preparation_report["case_count"], "official_payload_open_count": 0,
                "official_target_process_starts": 0, "permutation_case_count": 6,
                "preowner_failure_case_count": 6, "production_mode_exercised": False,
                "report_file_sha256": sha256_file(preparation_report_path),
                "report_path": str(ACTION_ROOT / "evidence" / FIXTURE_REPORT_NAME),
                "temporary_synthetic_namespaces_only": True,
            },
            "transport_invocation": transport_invocation,
            "v18_binding": copy.deepcopy(v20_package["v18_binding"]),
            "v20_retired_predecessor": {
                "acceptance_file_sha256": V20_ACCEPTANCE_SHA256, "action_id": V20_ACTION_ID,
                "build_root": str(V20_BUILD_ROOT), "execution_package_file_sha256": V20_PACKAGE_SHA256,
                "root": str(V20_ROOT), "stderr_file_sha256": V20_STDERR_SHA256,
                "terminal_observation_file_sha256": V20_TERMINAL_OBSERVATION_SHA256,
                "transport_invocation_count": 1,
            },
            "verifier": {
                "acceptance_aware": True,
                "exact_argv": [str(INTERPRETER), str(ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_order_independent_binding_terminal_coverage_v21.py")],
                "inventory_policy": "EXACT_STATIC_FILES_PLUS_ZERO_OR_ONE_BOUND_FRESH_L2_ACCEPTANCE",
                "path": str(ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_order_independent_binding_terminal_coverage_v21.py"),
            },
        }
        manifest["v18_binding"]["integrated_binding_table_path"] = str(ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json")
        manifest = sealed(manifest, "package_content_sha256")
        write_json(staging / MANIFEST_NAME, manifest)
        for path in sorted(staging.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(staging, 0o555)
        os.rename(staging, ACTION_ROOT)
        staging = None

        BUILD_ROOT.mkdir(parents=True, mode=0o755)
        report = sealed({
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_build_report",
            "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
            "manifest_file_sha256": sha256_file(ACTION_ROOT / MANIFEST_NAME),
            "official_payload_open_count": 0,
            "official_target_process_starts": 0,
            "preparation_fixture_report_file_sha256": sha256_file(ACTION_ROOT / "evidence" / FIXTURE_REPORT_NAME),
            "preservation_root_count": len(preservation),
            "required_disposition": DISPOSITION,
            "runtime_namespace_file_count": 0,
            "status": "PASS_V21_STATIC_PACKAGE_BUILT_READY_FOR_INDEPENDENT_FRESH_L2",
            "v20_replayed_or_mutated": False,
            "v21_executed": False,
        }, "report_sha256")
        write_json(BUILD_ROOT / "build-report.json", report)
        os.chmod(BUILD_ROOT / "build-report.json", 0o444)
        require(not os.path.lexists(RUNTIME_ROOT), "builder materialized V21 runtime namespace")
        for root, expected in before.items():
            require(inventory_digest(Path(root)) == expected, f"preservation drift after V21 build: {root}")
        return report
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    try:
        report = build()
    except Exception as error:
        sys.stderr.write(f"V21_BUILD_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
