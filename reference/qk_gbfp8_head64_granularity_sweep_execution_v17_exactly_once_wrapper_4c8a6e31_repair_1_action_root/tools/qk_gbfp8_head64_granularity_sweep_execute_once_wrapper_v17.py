#!/home/argustest/miniconda3/bin/python3.13
"""Irreversible V17 wrapper launcher; static acceptance alone is not authority."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple

from qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17 import (
    Outcome,
    RuntimePaths,
    compact_bytes,
    run_execution,
    sha256_bytes,
    verify_self_hash,
)


ROOT = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root')
CORE_ROOT = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root')
CORE_TOOLS = CORE_ROOT / "tools"
RUNTIME_ROOT = Path('/home/argustest/ace-2/runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31')
PACKAGE = ROOT / 'QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json'
ACCEPTANCE = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
INTERPRETER = Path('/home/argustest/miniconda3/bin/python3.13')
ACTION_ID = 'ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z'
OFFICIAL_PAYLOAD = Path('/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin')
CORE_MANIFEST = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json')
CORE_ACCEPTANCE = Path('/home/argustest/ace-2/build/v17-evaluator-diagnostic-preauthority-repair-0001/FRESH_L2_STATIC_ACCEPTANCE.json')
CORE_VERIFIER_REPORT = Path('/home/argustest/ace-2/build/v17-evaluator-diagnostic-preauthority-repair-0001/independent-inert-verifier-report.json')
EXPECTED_ENVIRONMENT = {'LANG': 'C', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0', 'TZ': 'UTC'}
EXPECTED_INTERPRETER_SHA256 = 'fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad'
EXPECTED_CORE_MANIFEST_SHA256 = '9e120e0e831ec736df45bd1b9f825217ce3368c5e22dbf3e4fc67ac537ea61f1'
EXPECTED_CORE_ACCEPTANCE_SHA256 = '9e8eb7a1a7c4340ab8d7b70a2ee97a132169711659f0ca7608d6701ad3225761'
EXPECTED_CORE_VERIFIER_REPORT_SHA256 = '36191b9bc6b0b99ad9bd37519d5982e0efdb4ed61572f3cc69ee1c3526c6c018'
SCHEMA_HASHES = {
    'QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_AUTHORITY_SCHEMA.json': 'e8c6b37fa49455dd352e1fb73c710d6ddf5ae08d4443e30162a3d5f45ffde790',
    'QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_CREDENTIAL_SCHEMA.json': '8baca11e88d50264388b23d0519206b64c5daead7646af7cab3c1cbf0f20a8d6',
    'QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_LEDGER_SCHEMA.json': 'e434dfba3f74e4f3e1cba42dbd197487eb8c85713fa56ac9c22e489a92e58eb3',
    'QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json': '0935244bb444e84d12169c1241ec892af02bea4d418f32dc41e06f66bee836c0',
    'QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FRESH_L2_ACCEPTANCE_SCHEMA.json': 'feb07867329dc875d3e78aa86667845136f86c438594045c1e41c876e835705a',
    'QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_OWNER_CLAIM_SCHEMA.json': 'b88db241c4ecc4317cf320bc74a7534a3020aa616630509416661f8dd409b05b',
}
PATHS = RuntimePaths(
    owner=RUNTIME_ROOT / "primary/authority/base/owner-claim.json",
    authority=RUNTIME_ROOT / "primary/authority/base/authority.json",
    credential=RUNTIME_ROOT / "primary/authority/base/credential.json",
    ledger=RUNTIME_ROOT / "primary/authority/base/authority-ledger.json",
    result=RUNTIME_ROOT / "primary/result/base/result.json",
    first_terminal=RUNTIME_ROOT / "primary/authority/base/first-terminal.json",
    fallback_terminal=RUNTIME_ROOT / "fallback/first-terminal.json",
)


class LauncherError(RuntimeError):
    def __init__(self, message: str, failure_stage: str = "PREFLIGHT") -> None:
        super().__init__(message)
        self.failure_stage = failure_stage


class LauncherArguments(NamedTuple):
    package: str
    acceptance: str
    irreversible_action_id: str


def require(condition: bool, message: str, failure_stage: str = "PREFLIGHT") -> None:
    if not condition:
        raise LauncherError(message, failure_stage)


def expected_launcher_arguments() -> list[str]:
    return [
        "--package",
        str(PACKAGE),
        "--acceptance",
        str(ACCEPTANCE),
        "--irreversible-action-id",
        ACTION_ID,
    ]


def parse_exact_arguments(argv: list[str]) -> LauncherArguments:
    require(argv == expected_launcher_arguments(), "launcher argv", "LAUNCHER_ARGV")
    return LauncherArguments(argv[1], argv[3], argv[5])


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
    require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}", "CANONICAL_JSON")
    return value, raw


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {name}", "CORE_IMPORT_SETUP")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_schemas() -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    mapping = {
        "owner": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_OWNER_CLAIM_SCHEMA.json",
        "authority": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_AUTHORITY_SCHEMA.json",
        "credential": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_CREDENTIAL_SCHEMA.json",
        "ledger": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_LEDGER_SCHEMA.json",
        "terminal": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json",
        "acceptance": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    }
    validators = {}
    for key, path in mapping.items():
        require(sha256_file(path) == SCHEMA_HASHES[path.name], f"schema hash: {key}", "SCHEMA_SETUP")
        schema, _ = canonical(path)
        Draft202012Validator.check_schema(schema)
        validators[key] = Draft202012Validator(schema).validate
    return validators


def transport_attestation(package: dict[str, Any]) -> dict[str, Any]:
    parent = os.getppid()
    argv_raw = Path(f"/proc/{parent}/cmdline").read_bytes()
    env_raw = Path(f"/proc/{parent}/environ").read_bytes()
    executable = Path(f"/proc/{parent}/exe").resolve()
    require(executable == INTERPRETER, "transport interpreter", "TRANSPORT_ATTESTATION")
    record = {
        "api": "os.posix_spawn",
        "immediate_parent_argv_sha256": sha256_bytes(argv_raw),
        "immediate_parent_environment_sha256": sha256_bytes(env_raw),
        "immediate_parent_executable_sha256": sha256_file(executable),
        "immediate_parent_pid": parent,
        "immediate_parent_transport_sha256": package["generated_files"]["tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v17.py"]["sha256"],
        "launcher_argv_sha256": package["launcher_invocation"]["invocation_sha256"],
        "launcher_environment_sha256": sha256_bytes(compact_bytes(EXPECTED_ENVIRONMENT)),
        "shell": False,
    }
    record["transport_attestation_sha256"] = sha256_bytes(compact_bytes(record))
    return record


def read_only_preflight(
    args: LauncherArguments,
    validators: dict[str, Any],
    paths: RuntimePaths,
    *,
    observed_cwd: Path | None = None,
    observed_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    cwd = Path.cwd() if observed_cwd is None else observed_cwd
    environment = dict(os.environ) if observed_environment is None else observed_environment
    require(cwd == ROOT, "launcher cwd", "LAUNCHER_CWD")
    require(environment == EXPECTED_ENVIRONMENT, "launcher environment", "LAUNCHER_ENVIRONMENT")
    require(Path(sys.executable).resolve() == INTERPRETER, "interpreter path", "INTERPRETER_IDENTITY")
    require(sha256_file(INTERPRETER) == EXPECTED_INTERPRETER_SHA256, "interpreter hash", "INTERPRETER_IDENTITY")
    require(args == LauncherArguments(str(PACKAGE), str(ACCEPTANCE), ACTION_ID), "launcher argv", "LAUNCHER_ARGV")
    owner, _ = canonical(paths.owner)
    validators["owner"](owner)
    verify_self_hash(owner, "owner_claim_sha256")
    require(not any(os.path.lexists(path) for path in paths if path != paths.owner), "runtime lifecycle already materialized", "RUNTIME_PREFLIGHT")
    package, package_raw = canonical(PACKAGE)
    verify_self_hash(package, "package_content_sha256")
    require(package["action_identity"]["action_id"] == ACTION_ID, "package action", "PACKAGE_BINDING")
    require(package["runtime_namespace"]["paths"]["owner"] == str(paths.owner), "owner path binding", "PACKAGE_BINDING")
    require(sha256_file(CORE_MANIFEST) == EXPECTED_CORE_MANIFEST_SHA256, "core manifest hash", "CORE_BINDING")
    require(sha256_file(CORE_ACCEPTANCE) == EXPECTED_CORE_ACCEPTANCE_SHA256, "core acceptance hash", "CORE_BINDING")
    require(sha256_file(CORE_VERIFIER_REPORT) == EXPECTED_CORE_VERIFIER_REPORT_SHA256, "core verifier report hash", "CORE_BINDING")
    core_acceptance, _ = canonical(CORE_ACCEPTANCE)
    require(core_acceptance["acceptance_sha256"] == 'dcb09b60b3a9010a8360f0fb5c0004f86c70be7b94646f2031dfd868e601bd17', "core acceptance self hash", "CORE_BINDING")
    acceptance, acceptance_raw = canonical(ACCEPTANCE)
    validators["acceptance"](acceptance)
    verify_self_hash(acceptance, "acceptance_sha256")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_raw), "wrapper acceptance package binding", "ACCEPTANCE_BINDING")
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "static acceptance authority boundary", "ACCEPTANCE_BINDING")
    for directory in package["runtime_namespace"]["precreated_directories"]:
        info = os.lstat(directory)
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"runtime directory: {directory}", "RUNTIME_PREFLIGHT")
        require(stat.S_IMODE(info.st_mode) == 0o700, f"runtime mode: {directory}", "RUNTIME_PREFLIGHT")
    payload_info = os.lstat(OFFICIAL_PAYLOAD)
    require(stat.S_ISREG(payload_info.st_mode) and not stat.S_ISLNK(payload_info.st_mode), "official payload lstat", "PAYLOAD_METADATA")
    require(payload_info.st_size == package["official_payload"]["byte_count"], "official payload size", "PAYLOAD_METADATA")
    core_package, core_package_raw = canonical(CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json")
    return {
        "acceptance_sha256": sha256_bytes(acceptance_raw),
        "core_package": core_package,
        "core_package_sha256": sha256_bytes(core_package_raw),
        "package": package,
        "package_sha256": sha256_bytes(package_raw),
        "transport_attestation": transport_attestation(package),
    }


def prepare_execution(holder: dict[str, Any], module_loader: Callable[[str, Path], Any]) -> None:
    package = holder["package"]
    accepted = holder["core_package"]
    records = accepted["official_benchmark"]["input_bindings"]["tensor_records"]
    holder["tensor_bindings"] = {
        record["tensor_name"]: {"dtype": record["dtype"], "sha256": record["sha256"], "shape": list(record["shape"])}
        for record in records.values()
    }
    holder["input_bindings"] = accepted["official_benchmark"]["input_bindings"]
    holder["official"] = package["official_identities"]
    holder["adapter"] = module_loader(
        "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17",
        CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py",
    )
    holder["bridge"] = module_loader(
        "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17",
        CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17.py",
    )
    result_validation = module_loader(
        "qk_gbfp8_head64_granularity_sweep_result_validation_v17",
        CORE_TOOLS / "qk_gbfp8_head64_granularity_sweep_result_validation_v17.py",
    )
    result_schema = result_validation.frozen_result_schema()
    holder["result_validation"] = result_validation
    holder["result_validator"] = result_validation.construct_result_schema_validator(result_schema)


def read_payload_once(expected_sha256: str, expected_size: int) -> bytes:
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(OFFICIAL_PAYLOAD, flags)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_size == expected_size, "official payload fstat", "PAYLOAD_OPEN")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    require(sha256_bytes(raw) == expected_sha256, "official payload hash", "PAYLOAD_OPEN")
    return raw


def run_launcher(
    *,
    paths: RuntimePaths = PATHS,
    argv: list[str] | None = None,
    observed_cwd: Path | None = None,
    observed_environment: dict[str, str] | None = None,
    schema_loader: Callable[[], dict[str, Any]] = load_schemas,
    module_loader: Callable[[str, Path], Any] = load_module,
) -> Outcome:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    validators: dict[str, Any] = {}
    bindings: dict[str, Any] = {"action_id": ACTION_ID}
    holder: dict[str, Any] = {}

    def preflight() -> None:
        arguments = parse_exact_arguments(raw_argv)
        validators.update(schema_loader())
        holder.update(read_only_preflight(
            arguments,
            validators,
            paths,
            observed_cwd=observed_cwd,
            observed_environment=observed_environment,
        ))
        prepare_execution(holder, module_loader)
        package = holder["package"]
        bindings.update({
            "core_acceptance_file_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256,
            "core_manifest_file_sha256": EXPECTED_CORE_MANIFEST_SHA256,
            "execution_package_file_sha256": holder["package_sha256"],
            "fresh_l2_acceptance_sha256": holder["acceptance_sha256"],
            "interpreter_sha256": EXPECTED_INTERPRETER_SHA256,
            "invocation_sha256": package["production_evaluator_invocation"]["invocation_sha256"],
            "launcher_invocation_sha256": package["launcher_invocation"]["invocation_sha256"],
            "official_identities": holder["official"],
            "transport_attestation": holder["transport_attestation"],
        })

    def payload_reader() -> bytes:
        package = holder["package"]
        official = holder["official"]
        return read_payload_once(official["tensor_bundle_sha256"], package["official_payload"]["byte_count"])

    def invoke_once(payload: bytes, authority_sha256: str, ledger_sha256: str, publish_result: Any) -> Any:
        package = holder["package"]
        official = holder["official"]
        result_validation = holder["result_validation"]
        result_bindings = result_validation.ResultBindings(
            acceptance_sha256=holder["acceptance_sha256"],
            authority_sha256=authority_sha256,
            evaluator_sha256=package["frozen_core_imports"]["official_evaluator_sha256"],
            invocation_sha256=package["production_evaluator_invocation"]["invocation_sha256"],
            ledger_sha256=ledger_sha256,
            model_identity_sha256=official["model_sha256"],
            package_sha256=holder["core_package_sha256"],
            tensor_bundle_sha256=official["tensor_bundle_sha256"],
        )
        bound = result_validation.bind_result_validators(holder["result_validator"], result_bindings, ACTION_ID)
        context = {
            "authority_sha256": authority_sha256,
            "consumed_ledger_sha256": ledger_sha256,
            "evaluator_sha256": package["frozen_core_imports"]["official_evaluator_sha256"],
            "fresh_l2_acceptance_sha256": holder["acceptance_sha256"],
            "input_bindings": holder["input_bindings"],
            "invocation_sha256": package["production_evaluator_invocation"]["invocation_sha256"],
            "irreversible_action_id": ACTION_ID,
            "model_identity_sha256": official["model_sha256"],
            "package_sha256": holder["core_package_sha256"],
        }
        return holder["bridge"].invoke_production_evaluator(
            payload,
            holder["tensor_bindings"],
            context,
            bound.validate_result_schema,
            bound.validate_result_record,
            publish_result,
        )

    return run_execution(paths, bindings, validators, preflight, payload_reader, invoke_once)


def main() -> int:
    outcome = run_launcher()
    return 0 if outcome.status == "SUCCEEDED_TERMINAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
