#!/usr/bin/env python3
"""Package-local, non-consuming V21 preparation and terminal regression."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path("/home/argustest/ace-2")
TOOLS = Path(__file__).resolve().parent
PACKAGE_ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21 as wrapper
import qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v21 as engine
import qk_gbfp8_head64_granularity_sweep_v21_binding_resolution as resolution


CORE_PACKAGE = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
EXPECTED_CORE_PACKAGE_SHA256 = "3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832"
EXPECTED_V18_BINDING_SHA256 = "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"

AUDIT = {"official_payload_open_count": 0, "official_target_process_starts": 0}


class RegressionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        candidate = args[0]
        if isinstance(candidate, (str, bytes, os.PathLike)):
            try:
                if Path(candidate).resolve() == OFFICIAL_PAYLOAD.resolve():
                    AUDIT["official_payload_open_count"] += 1
            except (OSError, RuntimeError, ValueError):
                pass
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.posix_spawnp"}:
        AUDIT["official_target_process_starts"] += 1


def load_canonical(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"frozen hash: {path}")
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}")
    return value


def record_identities(records_by_name: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {name: sha256_bytes(compact_bytes(record)) for name, record in records_by_name.items()}


class SyntheticResultValidation:
    @staticmethod
    def frozen_result_schema() -> dict[str, Any]:
        return {}

    @staticmethod
    def construct_result_schema_validator(_schema: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
        return lambda _record: None


def synthetic_module_loader(name: str, _path: Path) -> Any:
    if name.endswith("result_validation_v21"):
        return SyntheticResultValidation()
    return type("SyntheticModule", (), {})()


def holder_for(core_package: dict[str, Any], tensor_records: dict[str, Any]) -> dict[str, Any]:
    accepted = copy.deepcopy(core_package)
    accepted["official_benchmark"]["input_bindings"]["tensor_records"] = tensor_records
    return {
        "core_package": accepted,
        "package": {
            "official_identities": {
                "input_tokens_sha256": "1" * 64,
                "lane_metadata_sha256": "2" * 64,
                "model_sha256": "3" * 64,
                "tensor_bundle_sha256": "4" * 64,
            }
        },
    }


def expect_binding_failure(name: str, code: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        operation()
    except resolution.BindingResolutionError as error:
        require(error.failure_stage == "V18_BINDING", f"{name}: failure stage")
        require(error.code == code, f"{name}: failure code")
        return {"failure_code": code, "failure_stage": error.failure_stage, "name": name, "passed": True}
    raise RegressionError(f"{name}: not rejected")


def binding_cases(core_package: dict[str, Any], binding_table: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tensor_records = core_package["official_benchmark"]["input_bindings"]["tensor_records"]
    baseline_holder = holder_for(core_package, tensor_records)
    wrapper.prepare_execution(baseline_holder, synthetic_module_loader, binding_table_loader=lambda: binding_table)
    baseline_identities = record_identities(baseline_holder["resolved_tensor_records"])
    permutations = []
    for permutation in itertools.permutations(tensor_records.items()):
        permuted = dict(permutation)
        holder = holder_for(core_package, permuted)
        wrapper.prepare_execution(holder, synthetic_module_loader, binding_table_loader=lambda: binding_table)
        require(holder["numerical_tensor_names"] == list(resolution.CANONICAL_TENSOR_NAMES), "canonical tensor order")
        require(record_identities(holder["resolved_tensor_records"]) == baseline_identities, "record identity preservation")
        permutations.append({
            "canonical_tensor_names": holder["numerical_tensor_names"],
            "record_identities": record_identities(holder["resolved_tensor_records"]),
            "source_key_order": list(permuted),
        })

    duplicate = copy.deepcopy(tensor_records)
    duplicate["bf16_oracle_scores"]["tensor_name"] = "bf16.k_rope"
    missing = copy.deepcopy(tensor_records)
    del missing["realized_query_source"]
    unexpected = copy.deepcopy(tensor_records)
    unexpected["realized_query_source"]["tensor_name"] = "bf16.unexpected"
    malformed = copy.deepcopy(tensor_records)
    malformed["realized_query_source"] = {"dtype": "torch.bfloat16"}
    altered_table = copy.deepcopy(binding_table)
    altered_table["selected_tensor_names"] = list(reversed(altered_table["selected_tensor_names"]))
    negatives = [
        expect_binding_failure("duplicate_tensor_name", "DUPLICATE_TENSOR_NAME", lambda: wrapper.prepare_execution(holder_for(core_package, duplicate), synthetic_module_loader, binding_table_loader=lambda: binding_table)),
        expect_binding_failure("missing_tensor_name", "MISSING_TENSOR_NAME", lambda: wrapper.prepare_execution(holder_for(core_package, missing), synthetic_module_loader, binding_table_loader=lambda: binding_table)),
        expect_binding_failure("unexpected_tensor_name", "UNEXPECTED_TENSOR_NAME", lambda: wrapper.prepare_execution(holder_for(core_package, unexpected), synthetic_module_loader, binding_table_loader=lambda: binding_table)),
        expect_binding_failure("malformed_tensor_record", "TENSOR_RECORD_MALFORMED", lambda: wrapper.prepare_execution(holder_for(core_package, malformed), synthetic_module_loader, binding_table_loader=lambda: binding_table)),
        expect_binding_failure("altered_v18_selected_names", "BINDING_TABLE_SELECTED_NAMES", lambda: wrapper.prepare_execution(holder_for(core_package, tensor_records), synthetic_module_loader, binding_table_loader=lambda: altered_table)),
    ]
    return permutations, negatives


def runtime_paths(root: Path) -> engine.RuntimePaths:
    return engine.RuntimePaths(
        owner=root / "primary/authority/base/owner-claim.json",
        authority=root / "primary/authority/base/authority.json",
        credential=root / "primary/authority/base/credential.json",
        ledger=root / "primary/authority/base/authority-ledger.json",
        result=root / "primary/result/base/result.json",
        first_terminal=root / "primary/authority/base/first-terminal.json",
        fallback_terminal=root / "fallback/first-terminal.json",
    )


def terminal_validator() -> Callable[[dict[str, Any]], None]:
    schema, _ = wrapper.canonical(PACKAGE_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema).validate


def injected_error(stage: str) -> wrapper.LauncherError:
    return wrapper.LauncherError(f"synthetic {stage}", stage)


def terminal_case(
    root: Path,
    name: str,
    expected_stage: str,
    core_package: dict[str, Any],
    binding_table: dict[str, Any],
    *,
    argv: list[str] | None = None,
    schema_loader: Callable[[], dict[str, Any]] | None = None,
    preflight_reader: Callable[..., dict[str, Any]] | None = None,
    module_loader: Callable[[str, Path], Any] = synthetic_module_loader,
    preparer: Callable[..., None] = wrapper.prepare_execution,
    faults: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    paths = runtime_paths(root / name)
    validators = {"terminal": terminal_validator()}
    if argv is None:
        argv = wrapper.expected_launcher_arguments()
    if schema_loader is None:
        schema_loader = lambda: validators
    if preflight_reader is None:
        preflight_reader = lambda *_args, **_kwargs: holder_for(
            core_package, copy.deepcopy(core_package["official_benchmark"]["input_bindings"]["tensor_records"])
        )
    outcome = wrapper.run_launcher(
        paths=paths,
        argv=argv,
        observed_cwd=wrapper.ROOT,
        observed_environment=wrapper.EXPECTED_ENVIRONMENT,
        schema_loader=schema_loader,
        module_loader=module_loader,
        preflight_reader=preflight_reader,
        preparer=preparer,
        binding_table_loader=lambda: binding_table,
        preowner_faults=faults,
    )
    require(outcome.status == "PREFLIGHT_FAILED_TERMINAL", f"{name}: terminal status")
    terminal_path = paths.fallback_terminal if "TERMINAL_PRIMARY" in faults else paths.first_terminal
    require(outcome.terminal_path == str(terminal_path) and terminal_path.is_file(), f"{name}: terminal path")
    raw = terminal_path.read_bytes()
    terminal = json.loads(raw.decode("ascii", "strict"))
    require(compact_bytes(terminal) == raw, f"{name}: canonical terminal")
    validators["terminal"](terminal)
    require(terminal["wrapper_diagnostic"]["failure_stage"] == expected_stage, f"{name}: diagnostic stage")
    require(terminal["authority_sha256"] is None and terminal["consumed_ledger_sha256"] is None, f"{name}: zero authority")
    require(
        terminal["payload_open_count"] == 0
        and terminal["invocation_count_performed"] == 0
        and terminal["official_target_process_starts"] == 0,
        f"{name}: zero effects",
    )
    require(not any(os.path.lexists(path) for path in (paths.authority, paths.credential, paths.ledger, paths.result)), f"{name}: lifecycle exclusion")
    before = sorted(path.relative_to(root / name).as_posix() for path in (root / name).rglob("*") if path.is_file())
    duplicate = wrapper.run_launcher(
        paths=paths,
        argv=argv,
        observed_cwd=wrapper.ROOT,
        observed_environment=wrapper.EXPECTED_ENVIRONMENT,
        schema_loader=schema_loader,
        module_loader=module_loader,
        preflight_reader=preflight_reader,
        preparer=preparer,
        binding_table_loader=lambda: binding_table,
        preowner_faults=faults,
    )
    after = sorted(path.relative_to(root / name).as_posix() for path in (root / name).rglob("*") if path.is_file())
    require(duplicate.status == "DUPLICATE_REJECTED" and duplicate.duplicate_rejected is True, f"{name}: duplicate exclusion")
    require(before == after, f"{name}: duplicate publication")
    return {
        "duplicate_status": duplicate.status,
        "failure_stage": expected_stage,
        "name": name,
        "primary_write_fault": "TERMINAL_PRIMARY" in faults,
        "terminal_relative_path": terminal_path.relative_to(root / name).as_posix(),
    }


def terminal_cases(core_package: dict[str, Any], binding_table: dict[str, Any]) -> list[dict[str, Any]]:
    tensor_records = core_package["official_benchmark"]["input_bindings"]["tensor_records"]
    malformed = copy.deepcopy(tensor_records)
    malformed["realized_query_source"] = {"dtype": "torch.bfloat16"}
    with tempfile.TemporaryDirectory(prefix="v21-preowner-terminal-") as temporary:
        root = Path(temporary)
        cases = [
            terminal_case(root, "argument", "LAUNCHER_ARGV", core_package, binding_table, argv=["--invalid"]),
            terminal_case(root, "schema", "SCHEMA_SETUP", core_package, binding_table, schema_loader=lambda: (_ for _ in ()).throw(injected_error("SCHEMA_SETUP"))),
            terminal_case(root, "core_binding", "CORE_BINDING", core_package, binding_table, preflight_reader=lambda *_args, **_kwargs: (_ for _ in ()).throw(injected_error("CORE_BINDING"))),
            terminal_case(root, "v18_binding", "V18_BINDING", core_package, binding_table, preflight_reader=lambda *_args, **_kwargs: holder_for(core_package, malformed)),
            terminal_case(root, "module_import", "CORE_IMPORT", core_package, binding_table, module_loader=lambda *_args: (_ for _ in ()).throw(ImportError("synthetic import"))),
            terminal_case(root, "prepare", "PREPARE_EXECUTION", core_package, binding_table, preparer=lambda *_args, **_kwargs: (_ for _ in ()).throw(injected_error("PREPARE_EXECUTION")), faults=frozenset({"TERMINAL_PRIMARY"})),
        ]
        require(all(not path.name.startswith("authority") for path in root.rglob("*.json")), "synthetic authority exclusion")
        return cases


def run() -> dict[str, Any]:
    core_package = load_canonical(CORE_PACKAGE, EXPECTED_CORE_PACKAGE_SHA256)
    local_binding_table = PACKAGE_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
    binding_table = load_canonical(local_binding_table, EXPECTED_V18_BINDING_SHA256)
    permutations, negatives = binding_cases(core_package, binding_table)
    preparation_failures = terminal_cases(core_package, binding_table)
    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "non-consuming audit")
    return {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_v21_preowner_terminal_coverage_regression",
        "canonical_tensor_names": list(resolution.CANONICAL_TENSOR_NAMES),
        "case_count": len(permutations) + len(negatives) + len(preparation_failures),
        "frozen_core_package_sha256": EXPECTED_CORE_PACKAGE_SHA256,
        "frozen_v18_binding_table_sha256": EXPECTED_V18_BINDING_SHA256,
        "negative_cases": negatives,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": AUDIT["official_payload_open_count"],
        "official_target_process_starts": AUDIT["official_target_process_starts"],
        "permutation_case_count": len(permutations),
        "permutation_cases": permutations,
        "preowner_failure_case_count": len(preparation_failures),
        "preowner_failure_cases": preparation_failures,
        "schema_version": 1,
        "status": "PASS_V21_PRODUCTION_PREPARATION_AND_CANONICAL_TERMINAL_COVERAGE",
    }


def main() -> int:
    sys.addaudithook(audit_hook)
    try:
        report = run()
    except Exception as error:
        sys.stderr.write(f"V21_PREOWNER_TERMINAL_REGRESSION_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
