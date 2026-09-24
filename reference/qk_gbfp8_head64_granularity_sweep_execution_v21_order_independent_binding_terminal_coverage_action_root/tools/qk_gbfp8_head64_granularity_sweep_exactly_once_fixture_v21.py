#!/home/argustest/miniconda3/bin/python3.13
"""Marker-free synthetic lifecycle and launcher/transport regression."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
CORE_TOOLS = ROOT / "tools"
ACTION_ID = 'ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001'
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(CORE_TOOLS))

from qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v21 import RuntimePaths, compact_bytes, run_execution, sha256_bytes
import qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v21 as core_preflight
import qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v21 as adapter
import qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21 as launcher
import qk_gbfp8_head64_granularity_sweep_nominal_records_v21 as nominal
import qk_gbfp8_head64_granularity_sweep_result_validation_v21 as result_validation
import qk_gbfp8_head64_granularity_sweep_transport_wrapper_v21 as transport

BRANCH_INVOCATION_COUNTS: list[int] = []


def binding_contract() -> dict[str, Any]:
    raw = (ROOT / "bindings/C02_EXACT_BINDINGS_25.json").read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    if compact_bytes(value) != raw:
        raise ValueError("fixture binding canonical")
    if value["record_count"] != 25 or len(value["records"]) != 25:
        raise ValueError("fixture binding count")
    if value["selected_tensor_names"] != ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"]:
        raise ValueError("fixture numerical selection")
    return {"binding_table_file_sha256": sha256_bytes(raw), "complete_binding_count": 25, "numerical_tensor_names": value["selected_tensor_names"]}



def _seal_synthetic_closure(closure: dict[str, Any]) -> None:
    closure["entries"] = sorted(closure["entries"], key=lambda item: item["path"])
    closure["entries_sha256"] = sha256_bytes(compact_bytes(closure["entries"]))


def _synthetic_closure(root: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = root / "CORE_MANIFEST.json"
    artifact_path = root / "artifact.bin"
    payload_path = root / "payload.bin"
    manifest = {"artifact_kind": "synthetic_core_manifest", "generated_files": {"artifact.bin": {"sha256": sha256_bytes(b"artifact"), "size": 8}}, "package_content_sha256": "1" * 64}
    manifest_raw = compact_bytes(manifest)
    manifest_path.write_bytes(manifest_raw)
    artifact_path.write_bytes(b"artifact")
    payload_path.write_bytes(b"synthetic-payload")
    for path in (manifest_path, artifact_path, payload_path):
        os.chmod(path, 0o444)
    def record(path: Path, *, canonical_json: bool = False, deferred: bool = False) -> dict[str, Any]:
        raw = path.read_bytes()
        semantic = None
        if canonical_json:
            semantic = sha256_bytes(compact_bytes(json.loads(raw.decode("ascii"))))
        return {"canonical_json": canonical_json, "hash_policy": "DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN" if deferred else "FULL_READ_ONLY_PREFLIGHT", "kind": "file", "mode": "0444", "path": str(path), "semantic_sha256": semantic, "sha256": sha256_bytes(raw), "size": len(raw), "sources": ["synthetic"]}
    entries = [
        {"canonical_json": False, "hash_policy": "NOT_APPLICABLE", "kind": "directory", "mode": f"{os.lstat(root).st_mode & 0o7777:04o}", "path": str(root), "semantic_sha256": None, "sha256": None, "size": None, "sources": ["synthetic"]},
        record(manifest_path, canonical_json=True), record(artifact_path), record(payload_path, deferred=True),
    ]
    closure = {
        "core_manifest_mode": "0444", "core_manifest_path": str(manifest_path),
        "core_manifest_semantic_sha256": sha256_bytes(manifest_raw), "core_manifest_sha256": sha256_bytes(manifest_raw),
        "core_manifest_self_sha256": "1" * 64, "deferred_hash_paths": [str(payload_path)],
        "directory_count": 1, "entries": entries, "entries_sha256": "", "file_count": 3,
        "full_hash_file_count": 2, "official_evaluator_sha256": "2" * 64,
        "official_parser_sha256": "3" * 64, "package_key": "core_manifest_closure",
        "path_hash_mode_complete": True, "schema_version": 1,
    }
    _seal_synthetic_closure(closure)
    return closure, manifest_path, artifact_path


def _real_preflight_case(name: str, mutation: str | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v21-real-preflight-") as temporary:
        root = Path(temporary)
        closure, manifest_path, artifact_path = _synthetic_closure(root)
        expected_stage = None
        if mutation == "absent":
            manifest_path.unlink()
            expected_stage = "PATH_ABSENT"
        elif mutation == "wrong_path":
            wrong = root / "WRONG_CORE_MANIFEST.json"
            wrong.write_bytes(manifest_path.read_bytes())
            os.chmod(wrong, 0o444)
            entry = next(item for item in closure["entries"] if item["path"] == str(manifest_path))
            entry["path"] = str(wrong)
            closure["core_manifest_path"] = str(wrong)
            _seal_synthetic_closure(closure)
            expected_stage = "CORE_MANIFEST_PATH"
        elif mutation == "wrong_hash":
            os.chmod(artifact_path, 0o644)
            artifact_path.write_bytes(b"mutated!")
            os.chmod(artifact_path, 0o444)
            expected_stage = "PATH_HASH"
        elif mutation == "wrong_mode":
            os.chmod(artifact_path, 0o644)
            expected_stage = "PATH_MODE"
        elif mutation == "malformed":
            os.chmod(manifest_path, 0o644)
            raw = b"{malformed\n"
            manifest_path.write_bytes(raw)
            os.chmod(manifest_path, 0o444)
            entry = next(item for item in closure["entries"] if item["path"] == str(manifest_path))
            entry["sha256"] = sha256_bytes(raw)
            entry["size"] = len(raw)
            _seal_synthetic_closure(closure)
            expected_stage = "JSON_MALFORMED"
        elif mutation == "inconsistent":
            os.chmod(manifest_path, 0o644)
            value = json.loads(manifest_path.read_text(encoding="ascii"))
            value["generated_files"]["artifact.bin"]["size"] = 9
            raw = compact_bytes(value)
            manifest_path.write_bytes(raw)
            os.chmod(manifest_path, 0o444)
            entry = next(item for item in closure["entries"] if item["path"] == str(manifest_path))
            entry["sha256"] = sha256_bytes(raw)
            entry["size"] = len(raw)
            entry["semantic_sha256"] = sha256_bytes(raw)
            closure["core_manifest_sha256"] = sha256_bytes(raw)
            _seal_synthetic_closure(closure)
            expected_stage = "MANIFEST_INCONSISTENT"
        runtime = root / "runtime"
        try:
            observed = core_preflight.verify_core_manifest_closure(closure, expected_core_manifest_path=manifest_path)
        except core_preflight.CoreManifestPreflightError as error:
            return {"name": name, "passed": mutation is not None and error.stage == expected_stage and not runtime.exists(), "failure_stage": error.stage, "invocation_calls": 0, "owner_created": runtime.exists()}
        if mutation is not None:
            return {"name": name, "passed": False, "failure_stage": "NOT_REJECTED", "invocation_calls": 0, "owner_created": runtime.exists()}
        paths = make_paths(runtime)
        outcome = run_execution(paths, bindings(), validators(), lambda: None, lambda: b"synthetic-payload", invoker("nominal"))
        BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)
        return {"name": name, "passed": observed["status"].startswith("PASS_") and outcome.status == "SUCCEEDED_TERMINAL" and outcome.counts.invocation_count == 1, "failure_stage": None, "invocation_calls": outcome.counts.invocation_count, "owner_created": paths.owner.exists()}


def _real_preflight_concurrent() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v21-real-preflight-race-") as temporary:
        root = Path(temporary)
        closure, manifest_path, _ = _synthetic_closure(root)
        paths = make_paths(root / "runtime")
        barrier = threading.Barrier(3)
        lock = threading.Lock()
        outcomes = []
        invocation_calls = 0
        def worker() -> None:
            nonlocal invocation_calls
            core_preflight.verify_core_manifest_closure(closure, expected_core_manifest_path=manifest_path)
            barrier.wait(timeout=5)
            base_invoker = invoker("nominal")
            def invoke_once(*args: Any) -> Any:
                nonlocal invocation_calls
                with lock:
                    invocation_calls += 1
                return base_invoker(*args)
            outcome = run_execution(paths, bindings(), validators(), lambda: None, lambda: b"synthetic-payload", invoke_once)
            with lock:
                outcomes.append(outcome)
        threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
        for thread in threads: thread.start()
        barrier.wait(timeout=5)
        for thread in threads: thread.join(timeout=10)
        owners = [item for item in outcomes if not item.duplicate_rejected]
        losers = [item for item in outcomes if item.duplicate_rejected]
        loser_published = any(item.terminal_path is not None or item.terminal_publication_failed for item in losers)
        BRANCH_INVOCATION_COUNTS.extend(item.counts.invocation_count for item in outcomes)
        return {"name": "real_preflight_concurrent", "passed": len(owners) == 1 and len(losers) == 1 and invocation_calls == 1 and not loser_published, "invocation_calls": invocation_calls, "loser_published": loser_published}


def real_preflight_cases() -> list[dict[str, Any]]:
    cases = [_real_preflight_case("real_preflight_positive", None)]
    for mutation in ("absent", "wrong_path", "wrong_hash", "wrong_mode", "malformed", "inconsistent"):
        cases.append(_real_preflight_case(f"real_preflight_{mutation}", mutation))
    cases.append(_real_preflight_concurrent())
    return cases


def load_schema(name: str) -> dict[str, Any]:
    raw = (ROOT / "reference" / name).read_bytes()
    value = json.loads(raw.decode("ascii"))
    if compact_bytes(value) != raw:
        raise ValueError("fixture schema canonical")
    return value


def validators() -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    names = {
        "owner": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_EXACTLY_ONCE_OWNER_CLAIM_SCHEMA.json",
        "authority": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_EXACTLY_ONCE_AUTHORITY_SCHEMA.json",
        "credential": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_EXACTLY_ONCE_CREDENTIAL_SCHEMA.json",
        "ledger": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_EXACTLY_ONCE_LEDGER_SCHEMA.json",
        "terminal": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json",
    }
    return {key: Draft202012Validator(load_schema(name)).validate for key, name in names.items()}


def make_paths(root: Path) -> RuntimePaths:
    directories = [
        root,
        root / "primary",
        root / "primary/authority",
        root / "primary/authority/base",
        root / "primary/result",
        root / "primary/result/base",
        root / "fallback",
    ]
    for directory in directories:
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
    return RuntimePaths(
        owner=root / "primary/authority/base/owner-claim.json",
        authority=root / "primary/authority/base/authority.json",
        credential=root / "primary/authority/base/credential.json",
        ledger=root / "primary/authority/base/authority-ledger.json",
        result=root / "primary/result/base/result.json",
        first_terminal=root / "primary/authority/base/first-terminal.json",
        fallback_terminal=root / "fallback/first-terminal.json",
    )


def bindings() -> dict[str, Any]:
    attestation = {
        "api": "os.posix_spawn",
        "immediate_parent_argv_sha256": "1" * 64,
        "immediate_parent_environment_sha256": "2" * 64,
        "immediate_parent_executable_sha256": 'fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad',
        "immediate_parent_pid": 1,
        "immediate_parent_transport_sha256": "3" * 64,
        "launcher_argv_sha256": "4" * 64,
        "launcher_environment_sha256": "5" * 64,
        "shell": False,
    }
    attestation["transport_attestation_sha256"] = sha256_bytes(compact_bytes(attestation))
    return {
        "action_id": ACTION_ID,
        "binding_record_count": 25,
        "numerical_tensor_names": ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"],
        "v18_acceptance_file_sha256": '342ab0130d7e132e4e288b9fa349f7e0c5efb87cb35f29cdbc86e3bc3f5e8cf2',
        "v18_acceptance_self_sha256": '601980bd0a7af5040c1ad07c9ea88e6aad71619326574436175419637fe74e12',
        "v18_action_id": 'ace2:qk-gbfp8-base-v18:static-c02-binding-repair:additive-0001',
        "v18_binding_table_file_sha256": '87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655',
        "v18_manifest_file_sha256": 'fdfc33a1443f237fbaca882b13fb977ae3c2ad7f61e427453297d0621f4445c8',
        "v18_post_acceptance_report_self_sha256": '19d63c0b6610af00a9f031b7f91665496f0c17cdbfccb8cdc593d5a31ca0b3ae',
        "core_acceptance_file_sha256": '9e8eb7a1a7c4340ab8d7b70a2ee97a132169711659f0ca7608d6701ad3225761',
        "core_manifest_file_sha256": '9e120e0e831ec736df45bd1b9f825217ce3368c5e22dbf3e4fc67ac537ea61f1',
        "execution_package_file_sha256": "6" * 64,
        "fresh_l2_acceptance_sha256": "7" * 64,
        "interpreter_sha256": 'fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad',
        "invocation_sha256": "8" * 64,
        "launcher_invocation_sha256": "9" * 64,
        "official_identities": {
            "input_tokens_sha256": "a" * 64,
            "lane_metadata_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "tensor_bundle_sha256": "d" * 64,
        },
        "transport_attestation": attestation,
    }


def result_bytes(authority_sha256: str, ledger_sha256: str, *, invalid_schema: bool = False) -> tuple[bytes, Any]:
    value = nominal.make_result("G4")
    value["authority_sha256"] = authority_sha256
    value["consumed_ledger_sha256"] = ledger_sha256
    value["evaluator_sha256"] = "e" * 64
    value["fresh_l2_acceptance_sha256"] = "7" * 64
    value["invocation_sha256"] = "8" * 64
    value["model_identity_sha256"] = "c" * 64
    value["package_sha256"] = "f" * 64
    value["input_bindings"]["tensor_bundle_sha256"] = "d" * 64
    if invalid_schema:
        value["candidate_results"][0]["metrics"]["top_key"]["row_count"] = 573
    value.pop("result_sha256", None)
    value["result_sha256"] = sha256_bytes(compact_bytes(value))
    schema_validator = result_validation.construct_result_schema_validator(result_validation.frozen_result_schema())
    bound = result_validation.bind_result_validators(
        schema_validator,
        result_validation.ResultBindings(
            acceptance_sha256="7" * 64,
            authority_sha256=authority_sha256,
            evaluator_sha256="e" * 64,
            invocation_sha256="8" * 64,
            ledger_sha256=ledger_sha256,
            model_identity_sha256="c" * 64,
            package_sha256="f" * 64,
            tensor_bundle_sha256="d" * 64,
        ),
        ACTION_ID,
    )
    return compact_bytes(value), bound


def invoker(scenario: str) -> Callable[..., Any]:
    def invoke(payload: bytes, authority_sha256: str, ledger_sha256: str, publish: Any) -> Any:
        invalid = scenario == "schema"
        raw, bound = result_bytes(authority_sha256, ledger_sha256, invalid_schema=invalid)
        if scenario == "decode":
            raw = b"{malformed\n"
        return_code = 7 if scenario == "nonzero" else 0
        stderr = b"synthetic nonzero\n" if scenario == "nonzero" else b""
        completed = subprocess.CompletedProcess(["synthetic"], return_code, stdout=raw, stderr=stderr)
        spec = adapter.InvocationSpec(
            argv=("synthetic-v21-adapter", scenario),
            cwd="/synthetic",
            environment=(("LANG", "C"),),
            max_capture_bytes=1 << 20,
        )
        return adapter.invoke_evaluator(
            spec,
            b"synthetic-envelope",
            bound.validate_result_schema,
            bound.validate_result_record,
            publish,
            runner=lambda *args, **kwargs: completed,
        )

    return invoke


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def run_case(name: str, operation: Callable[[RuntimePaths], dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v21-wrapper-repair-fixture-") as temporary:
        paths = make_paths(Path(temporary) / "runtime")
        try:
            observed = operation(paths)
            passed = bool(observed.pop("passed"))
            return {"name": name, "passed": passed, **observed}
        except BaseException as error:
            return {"name": name, "passed": False, "error_type": type(error).__name__}


def execute(
    paths: RuntimePaths,
    scenario: str = "nominal",
    *,
    faults: frozenset[str] = frozenset(),
    preflight_error: bool = False,
) -> Any:
    def preflight() -> None:
        if preflight_error:
            raise ValueError("synthetic preflight")

    outcome = run_execution(
        paths,
        bindings(),
        validators(),
        preflight,
        lambda: b"synthetic-payload",
        invoker(scenario),
        faults=faults,
    )
    BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)
    return outcome


def concurrent_start(paths: RuntimePaths) -> dict[str, Any]:
    start = threading.Barrier(3)
    release_owner = threading.Event()
    lock = threading.Lock()
    outcomes: list[Any] = []
    payload_open_calls = 0
    invocation_calls = 0

    def worker() -> None:
        nonlocal payload_open_calls, invocation_calls

        def preflight() -> None:
            if not release_owner.wait(timeout=5):
                raise TimeoutError("concurrent loser did not reject")

        def payload_reader() -> bytes:
            nonlocal payload_open_calls
            with lock:
                payload_open_calls += 1
            return b"synthetic-payload"

        base_invoker = invoker("nominal")

        def invoke_once(*args: Any) -> Any:
            nonlocal invocation_calls
            with lock:
                invocation_calls += 1
            return base_invoker(*args)

        start.wait(timeout=5)
        outcome = run_execution(paths, bindings(), validators(), preflight, payload_reader, invoke_once)
        if outcome.duplicate_rejected:
            release_owner.set()
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("concurrent fixture thread")

    BRANCH_INVOCATION_COUNTS.extend(outcome.counts.invocation_count for outcome in outcomes)
    owners = [outcome for outcome in outcomes if not outcome.duplicate_rejected]
    losers = [outcome for outcome in outcomes if outcome.duplicate_rejected]
    terminal = read_json(paths.first_terminal)
    authority = read_json(paths.authority)
    ledger = read_json(paths.ledger)
    owner_claim = read_json(paths.owner)
    loser_published = any(outcome.terminal_path is not None or outcome.terminal_publication_failed for outcome in losers)
    consistent = (
        terminal["authority_sha256"] == authority["authority_sha256"]
        and terminal["consumed_ledger_sha256"] == ledger["consumed_ledger_sha256"]
        and terminal["result_file_sha256"] == sha256_bytes(paths.result.read_bytes())
        and owner_claim["state"] == "OWNER_CLAIMED"
    )
    passed = (
        len(outcomes) == 2
        and len(owners) == 1
        and len(losers) == 1
        and owners[0].status == "SUCCEEDED_TERMINAL"
        and losers[0].status == "DUPLICATE_REJECTED"
        and payload_open_calls == 1
        and invocation_calls == 1
        and not loser_published
        and consistent
    )
    return {
        "passed": passed,
        "invocation_calls": invocation_calls,
        "loser_published": loser_published,
        "owner_status": owners[0].status if len(owners) == 1 else "INVALID",
        "payload_open_calls": payload_open_calls,
    }


def launcher_cwd_failure(paths: RuntimePaths) -> dict[str, Any]:
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


def transport_failure(paths: RuntimePaths, scenario: str) -> dict[str, Any]:
    argv = transport.expected_transport_arguments()
    cwd = transport.ROOT
    environment = dict(transport.EXACT_ENVIRONMENT)
    launcher_hash_provider: Callable[[Path], str] = lambda path: transport.LAUNCHER_SHA256
    spawn: Callable[..., int] | None = None
    expected_stage = {
        "cwd": "TRANSPORT_CWD",
        "environment": "TRANSPORT_ENVIRONMENT",
        "argv": "TRANSPORT_ARGV",
        "launcher_hash": "TRANSPORT_LAUNCHER_HASH",
        "spawn": "TRANSPORT_SPAWN",
    }[scenario]
    if scenario == "cwd":
        cwd = Path("/synthetic-wrong-cwd")
    elif scenario == "environment":
        environment = {**environment, "EXTRA": "1"}
    elif scenario == "argv":
        argv = argv[:-1]
    elif scenario == "launcher_hash":
        launcher_hash_provider = lambda path: "0" * 64
    elif scenario == "spawn":
        def fail_spawn(*args: Any) -> int:
            raise OSError("synthetic spawn")
        spawn = fail_spawn
    result = transport.run_transport(
        paths=paths,
        argv=argv,
        observed_cwd=cwd,
        observed_environment=environment,
        launcher_hash_provider=launcher_hash_provider,
        spawn=spawn,
    )
    if result.outcome is not None:
        BRANCH_INVOCATION_COUNTS.append(result.outcome.counts.invocation_count)
    if result.outcome is None or result.outcome.terminal_path is None:
        return {"passed": False, "status": "NO_RETIREMENT"}
    terminal = read_json(Path(result.outcome.terminal_path))
    return {
        "passed": (
            result.exit_code == 1
            and not result.spawned
            and result.outcome.status == "PREFLIGHT_FAILED_TERMINAL"
            and result.outcome.counts.invocation_count == 0
            and terminal["wrapper_diagnostic"]["failure_stage"] == expected_stage
            and paths.owner.exists()
            and not paths.authority.exists()
        ),
        "failure_stage": terminal["wrapper_diagnostic"]["failure_stage"],
        "status": result.outcome.status,
    }


def run_fixture() -> dict[str, Any]:
    BRANCH_INVOCATION_COUNTS.clear()
    binding = binding_contract()
    cases = []
    cases.append(run_case("nominal", lambda p: (lambda o: {
        "passed": o.status == "SUCCEEDED_TERMINAL" and o.counts.invocation_count == 1 and o.counts.payload_open_count == 1 and p.owner.exists() and p.result.exists() and p.ledger.exists() and not p.credential.exists(),
        "status": o.status,
    })(execute(p))))
    cases.append(run_case("preflight_failure", lambda p: (lambda o: {
        "passed": o.status == "PREFLIGHT_FAILED_TERMINAL" and o.counts.invocation_count == 0 and p.owner.exists() and not p.authority.exists(),
        "status": o.status,
    })(execute(p, preflight_error=True))))
    cases.append(run_case("authority_failure", lambda p: (lambda o: {
        "passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and p.owner.exists() and not p.authority.exists(),
        "status": o.status,
    })(execute(p, faults=frozenset({"AUTHORITY_WRITE"})))))
    cases.append(run_case("credential_failure", lambda p: (lambda o: {
        "passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and p.authority.exists() and not p.credential.exists() and not p.ledger.exists(),
        "status": o.status,
    })(execute(p, faults=frozenset({"CREDENTIAL_WRITE"})))))
    cases.append(run_case("ledger_failure", lambda p: (lambda o: {
        "passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and not p.credential.exists() and not p.ledger.exists(),
        "status": o.status,
    })(execute(p, faults=frozenset({"LEDGER_WRITE"})))))
    cases.append(run_case("credential_consumption_failure", lambda p: (lambda o: {
        "passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and p.credential.exists() and p.ledger.exists(),
        "status": o.status,
    })(execute(p, faults=frozenset({"CREDENTIAL_UNLINK"})))))

    def duplicate(paths: RuntimePaths) -> dict[str, Any]:
        first = execute(paths)
        before = paths.first_terminal.read_bytes()
        second = execute(paths)
        return {
            "passed": first.status == "SUCCEEDED_TERMINAL" and second.duplicate_rejected and second.counts.invocation_count == 0 and second.terminal_path is None and paths.first_terminal.read_bytes() == before,
            "status": second.status,
        }

    cases.append(run_case("duplicate_invocation", duplicate))
    cases.append(run_case("concurrent_start", concurrent_start))
    for name, scenario, failure_class in (
        ("nonzero_evaluator", "nonzero", "RETURN_CODE"),
        ("decode_failure", "decode", "DECODE"),
        ("schema_failure", "schema", "RESULT_SCHEMA"),
    ):
        def operation(paths: RuntimePaths, scenario: str = scenario, failure_class: str = failure_class) -> dict[str, Any]:
            outcome = execute(paths, scenario)
            terminal = read_json(Path(outcome.terminal_path))
            return {
                "passed": outcome.status == "CONSUMED_ORPHAN" and outcome.counts.invocation_count == 1 and terminal["evaluator_diagnostic"]["failure_class"] == failure_class,
                "failure_class": terminal["evaluator_diagnostic"]["failure_class"],
                "status": outcome.status,
            }

        cases.append(run_case(name, operation))
    cases.append(run_case("result_publication_failure", lambda p: (lambda o: {
        "passed": o.status == "CONSUMED_ORPHAN" and o.counts.invocation_count == 1 and read_json(Path(o.terminal_path))["evaluator_diagnostic"]["failure_class"] == "PUBLICATION",
        "status": o.status,
    })(execute(p, faults=frozenset({"RESULT_WRITE"})))))
    cases.append(run_case("terminal_primary_fallback", lambda p: (lambda o: {
        "passed": o.status == "CONSUMED_ORPHAN" and o.terminal_path == str(p.fallback_terminal) and p.result.exists(),
        "status": o.status,
    })(execute(p, faults=frozenset({"TERMINAL_PRIMARY"})))))
    cases.append(run_case("terminal_total_failure", lambda p: (lambda o: {
        "passed": o.status == "CONSUMED_ORPHAN_UNPUBLISHED" and o.terminal_publication_failed and o.counts.invocation_count == 1 and not p.first_terminal.exists() and not p.fallback_terminal.exists(),
        "status": o.status,
    })(execute(p, faults=frozenset({"TERMINAL_PRIMARY", "TERMINAL_FALLBACK"})))))
    cases.append(run_case("launcher_cwd_failure", launcher_cwd_failure))
    cases.append(run_case("launcher_setup_failure", launcher_setup_failure))
    for scenario in ("cwd", "environment", "argv", "launcher_hash", "spawn"):
        cases.append(run_case(f"transport_{scenario}_failure", lambda p, scenario=scenario: transport_failure(p, scenario)))
    preflight_cases = real_preflight_cases()
    cases.extend(preflight_cases)
    passed = all(case["passed"] for case in cases)
    return {
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v21_exactly_once_synthetic_fixture_report",
        "all_branches_at_most_once": max(BRANCH_INVOCATION_COUNTS, default=0) <= 1,
        "binding_contract": binding,
        "branch_invocation_observation_count": len(BRANCH_INVOCATION_COUNTS),
        "case_count": len(cases),
        "cases": cases,
        "concurrent_start_exercised": True,
        "launcher_transport_integration_exercised": True,
        "max_branch_invocation_count": max(BRANCH_INVOCATION_COUNTS, default=0),
        "owner_loser_publication_exclusion_verified": next((case.get("loser_published") is False for case in cases if case["name"] == "concurrent_start"), False),
        "official_payload_open_count": 0,
        "real_preflight_case_count": len(preflight_cases),
        "real_preflight_required_mutations": ["absent", "wrong_path", "wrong_hash", "wrong_mode", "malformed", "inconsistent"],
        "official_target_process_starts": 0,
        "production_mode_exercised": False,
        "shared_adapter_sha256": 'ae79abbd3d9bed60f730698fbcb4b65aae15e641bd92214caa28d1d18a132df1',
        "status": "PASS_V21_BOUND_CORE_MANIFEST_SYNTHETIC_FIXTURE" if passed else "FAIL_V21_BOUND_CORE_MANIFEST_SYNTHETIC_FIXTURE",
        "synthetic_bytes_only": True,
    }


def write_once(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = compact_bytes(report)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short fixture report write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_fixture()
    if args.report is not None:
        write_once(args.report, report)
    sys.stdout.buffer.write(compact_bytes(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
