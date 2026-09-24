#!/usr/bin/env python3
"""Static-only verifier for the c02 QK live-authority transaction core."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import qk_bfp8_e16_head64_v1_c02_live_authority_surface as core


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = Path(__file__).resolve()
IMPLEMENTATION_REL = "tools/qk_bfp8_e16_head64_v1_c02_live_authority_surface.py"
VERIFIER_REL = "tools/verify_qk_bfp8_e16_head64_v1_c02_live_authority_surface.py"
PACKAGE_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_SURFACE_IMPLEMENTATION_PACKAGE.json"
)
EVIDENCE_DIR_REL = (
    "evidence/verification/"
    "qk-bfp8-e16-head64-v1-c02-live-authority-surface-v1"
)
PROOF_REL = EVIDENCE_DIR_REL + "/static_synthetic_proof.json"
CHECKSUMS_REL = EVIDENCE_DIR_REL + "/SHA256SUMS"

AUTHORITY_PACKAGE_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EXTERNAL_EXACTLY_ONCE_AUTHORITY_PACKAGE.json"
)
AUTHORITY_VERIFIER_REL = (
    "tools/verify_qk_bfp8_e16_head64_v1_c02_score_path_external_exactly_once_authority.py"
)
AUTHORITY_CHECKSUMS_REL = (
    "evidence/verification/"
    "qk-bfp8-e16-head64-v1-c02-score-path-external-exactly-once-authority-v1/"
    "SHA256SUMS"
)
QUALIFYING_REVIEW_SOURCE = (
    Path.home()
    / ".argus-skill-ace2/projects/s-c8ae985b/handoffs/8c1b4f7e2a96/round-0001.json"
)
PREFLIGHT_REVIEW_SOURCE = (
    Path.home()
    / ".argus-skill-ace2/projects/s-c8ae985b/handoffs/51a7d3bc902e/round-0001.json"
)

PACKAGE = ROOT / PACKAGE_REL
IMPLEMENTATION = ROOT / IMPLEMENTATION_REL
PROOF = ROOT / PROOF_REL
CHECKSUMS = ROOT / CHECKSUMS_REL


class VerificationError(RuntimeError):
    """Static package verification failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def load_canonical_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON artifact: {path}")
    require(not path.is_symlink(), f"symlinked JSON artifact: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON artifact: {path}: {exc}") from exc
    require(raw == canonical_json_bytes(value), f"noncanonical JSON artifact: {path}")
    return value


def verify_file(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}")
    require(not path.is_symlink(), f"symlinked {label}")
    require(sha256_file(path) == expected, f"{label} SHA-256 mismatch")


def verify_external_bindings() -> None:
    verify_file(
        ROOT / AUTHORITY_PACKAGE_REL,
        core.AUTHORITY_PACKAGE_SHA256,
        "authority package",
    )
    verify_file(
        ROOT / AUTHORITY_VERIFIER_REL,
        core.AUTHORITY_VERIFIER_SHA256,
        "authority verifier",
    )
    verify_file(
        ROOT / AUTHORITY_CHECKSUMS_REL,
        core.CHECKSUM_MANIFEST_SHA256,
        "authority checksum manifest",
    )
    verify_file(
        QUALIFYING_REVIEW_SOURCE,
        core.QUALIFYING_AUTHORITY_REVIEW_SHA256,
        "qualifying authority review",
    )
    verify_file(
        PREFLIGHT_REVIEW_SOURCE,
        core.FAIL_CLOSED_PREFLIGHT_REVIEW_SHA256,
        "fail-closed preflight review",
    )
    qualifying = load_canonical_json(QUALIFYING_REVIEW_SOURCE)
    preflight = load_canonical_json(PREFLIGHT_REVIEW_SOURCE)
    for label, review in (("qualifying", qualifying), ("preflight", preflight)):
        require(review.get("kind") == "round_reviewed_handoff", f"{label} review kind")
        require(review.get("producer_role") == "reviewer", f"{label} review role")
        require(review.get("review", {}).get("status") == "done", f"{label} review status")
    authority = load_canonical_json(ROOT / AUTHORITY_PACKAGE_REL)
    require(authority.get("claim_boundary", {}).get("status") == "NO_EXECUTION_PERFORMED", "authority claim boundary")
    require(authority.get("implementation_identity", {}).get("executable_implementation_included") is False, "authority executable absence")
    lanes = authority.get("lanes")
    require(isinstance(lanes, list) and len(lanes) == 2, "authority lane count")
    for lane in lanes:
        require(lane.get("command", {}).get("descriptor", {}).get("shell_argv_materialized") is False, "authority argv absence")
        require(lane.get("environment") == core.ENVIRONMENT, "authority environment")


def verify_source_boundaries() -> None:
    implementation_source = IMPLEMENTATION.read_text(encoding="utf-8")
    verifier_source = VERIFIER.read_text(encoding="utf-8")
    implementation_tree = ast.parse(implementation_source)
    verifier_tree = ast.parse(verifier_source)
    model_modules = {
        "cocotb",
        "numpy",
        "onnxruntime",
        "tensorflow",
        "torch",
        "transformers",
    }
    verifier_forbidden_modules = model_modules | {"subprocess"}
    verifier_forbidden_calls = {
        "Popen",
        "call",
        "exec",
        "execve",
        "from_pretrained",
        "generate",
        "popen",
        "run",
        "system",
    }

    def imports(tree: ast.AST) -> set[str]:
        result: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                result.add(node.module.split(".")[0])
        return result

    called: set[str] = set()
    for node in ast.walk(verifier_tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    require(not imports(implementation_tree).intersection(model_modules), "implementation imports model module")
    require(not imports(verifier_tree).intersection(verifier_forbidden_modules), "verifier imports model/execution module")
    require(not called.intersection(verifier_forbidden_calls), "verifier contains execution call")
    require("attention-substage-tensors.bin" not in implementation_source, "implementation names sealed tensor bundle")
    functions = {
        node.name: node
        for node in implementation_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    require(
        [argument.arg for argument in functions["materialize_lane"].args.args]
        == ["lane_label", "authority_credential_sha256"],
        "materializer accepts caller-selected contract, root, or bypass",
    )
    require(
        [argument.arg for argument in functions["consume_and_run"].args.args]
        == ["lane_label"],
        "consumer accepts caller-selected contract, root, runner, or bypass",
    )
    require(
        "validate_executable_file" not in implementation_source,
        "executable validation bypass remains",
    )


def expected_package() -> dict[str, Any]:
    lanes = {}
    for lane_label, lane in core.LANES.items():
        lanes[lane_label] = {
            "command_sha256": lane["command_sha256"],
            "environment": core.ENVIRONMENT,
            "invocation_sha256": lane["invocation_sha256"],
            "namespace_label": lane["namespace_label"],
            "output_path": lane["output_path"],
            "record_id": lane["record_id"],
            "shell_argv": None,
            "shell_argv_status": "MISSING_FROM_FROZEN_ACCEPTED_AUTHORITY_SURFACE",
        }
    return {
        "artifact_kind": "qk_bfp8_e16_head64_v1_c02_live_authority_surface_implementation_package",
        "bindings": core.fixed_bindings(),
        "claim_boundary": {
            "authority_consumed": False,
            "authority_materialized": False,
            "declared_execution_namespace_created": False,
            "fixed_input_payload_deserialized": False,
            "invocation_count_performed": 0,
            "live_surface_namespace_created": False,
            "model_loaded_or_executed": False,
            "repair_replay_resume_or_retry_performed": False,
            "score_computation_performed": False,
            "status": "NO_EXECUTION_PERFORMED",
        },
        "crash_semantics": {
            "after_consumption_before_terminal": "the durable CONSUMED record remains authoritative; absent terminal is an orphaned-consumption outcome and may never be retried, replayed, resumed, repaired, or replaced",
            "before_consumption_commit": "no process launch or tensor access is permitted; a canonical READY record may be irreversibly invalidated and failed terminal sealed",
            "terminal_write": "atomic create-if-absent, fsync file, fsync parent; the first terminal record is immutable",
        },
        "implementation": {
            "entrypoint_policy": "LIVE_SURFACE_BLOCKED_NO_EXECUTION_PERFORMED",
            "launch_policy": "checksum-bound exact argv, shell=false, canonical cwd, explicit exact environment",
            "path": IMPLEMENTATION_REL,
            "sha256": sha256_file(IMPLEMENTATION),
        },
        "lanes": lanes,
        "launch_authority_binding": {
            "authority_id": core.LAUNCH_AUTHORITY_ID,
            "path": None,
            "sha256": None,
            "status": "MISSING_FROM_FROZEN_ACCEPTED_AUTHORITY_SURFACE",
        },
        "live_eligibility": {
            "blocking_conditions": [
                "accepted authority package sets shell_argv_materialized=false for both commands",
                "accepted authority package includes no executable evaluator implementation",
                "no checksum-bound executable SHA-256 is present for either command",
                "no checksum-bound launch authority amendment exists",
            ],
            "status": "FAIL_CLOSED_MISSING_FROZEN_SHELL_ARGV_AND_EXECUTABLE_IDENTITY",
        },
        "mutation_closure": [
            "checksum-bound launch authority or launch descriptor mismatch",
            "bound external artifact mutation",
            "hash binding mismatch",
            "lane, invocation, record, namespace, or command identity mismatch",
            "argv, executable hash, or environment mismatch",
            "noncanonical root, output, executable, ledger, terminal, or symlinked path",
            "unexpected state or terminal collision",
            "lane output alias or output collision",
            "checkpoint-176 ordering violation",
            "retry, replay, resume, repair, or terminal replacement",
        ],
        "path_contract": {
            "live_surface_root": core.LIVE_SURFACE_REL,
            "per_lane": "lanes/<namespace_label>/{authority-ledger.json,.transition.lock,first-terminal.json}",
            "permissions": {
                "directories": "0700 owner=current effective uid",
                "ledger_and_terminal": "0400 regular non-symlink",
                "transition_lock": "0600 regular non-symlink",
            },
            "score_output_root": core.OUTPUT_ROOT_REL,
        },
        "schema_version": 1,
        "schemas": {
            "contract": core.CONTRACT_SCHEMA_ID,
            "ready": core.READY_SCHEMA_ID,
            "states": [core.READY_STATE, core.CONSUMED_STATE, core.INVALIDATED_STATE],
            "terminal": core.TERMINAL_SCHEMA_ID,
            "terminal_states": sorted(core.TERMINAL_STATES),
        },
        "static_verifier": {
            "implementation": "CPython",
            "path": VERIFIER_REL,
            "python_version": "3.13.5",
            "sha256": sha256_file(VERIFIER),
        },
    }


def expected_proof() -> dict[str, Any]:
    return {
        "artifact_kind": "qk_bfp8_e16_head64_v1_c02_live_authority_surface_static_synthetic_proof",
        "claim_boundary": expected_package()["claim_boundary"],
        "implementation": {"path": IMPLEMENTATION_REL, "sha256": sha256_file(IMPLEMENTATION)},
        "negative_tests": {
            "count": 19,
            "labels": [
                "caller_selected_contract_and_surface_root_removed",
                "cross_lane_argv_alias",
                "dangling_output_symlink",
                "environment_mutation",
                "executable_validation_bypass_removed",
                "executable_hash_mutation",
                "injected_runner_removed",
                "launch_authority_hash_mutation",
                "launch_descriptor_mutation",
                "missing_executable",
                "missing_shell_argv",
                "noncanonical_surface_root",
                "output_collision",
                "postconsume_artifact_identity_mutation",
                "prior_ledger_identity_mutation",
                "second_lane_before_base_terminal",
                "terminal_overwrite",
                "reconsume_after_terminal",
                "run_after_preflight_invalidation",
            ],
        },
        "package": {"path": PACKAGE_REL, "sha256": sha256_file(PACKAGE)},
        "schema_version": 1,
        "synthetic_observations": {
            "consumption_visible_before_exact_launch": True,
            "exact_launch_argv_verified_without_execution": True,
            "exact_launch_environment_verified_without_execution": True,
            "failed_launch_terminal_sealed": True,
            "first_terminal_create_only": True,
            "isolated_temporary_fixture_count": 10,
            "lane_isolation_verified": True,
            "launch_backend_intercepted_before_process_creation": True,
            "live_entrypoint_fail_closed": True,
            "orphaned_consumption_retry_path_present": False,
            "sealed_tensor_payload_access_count": 0,
        },
        "verifier": {"path": VERIFIER_REL, "sha256": sha256_file(VERIFIER)},
    }


def synthetic_contract(surface_root: Path, executable: Path) -> dict[str, Any]:
    executable_sha256 = sha256_file(executable)
    lanes: dict[str, dict[str, Any]] = {}
    for lane_label, expected in core.LANES.items():
        lane: dict[str, Any] = {
            **expected,
            "environment": copy.deepcopy(core.ENVIRONMENT),
            "executable_sha256": executable_sha256,
            "shell_argv": [
                str(executable),
                "--synthetic-never-executed",
                lane_label,
            ],
        }
        lane["launch_descriptor_sha256"] = core.object_sha256(
            core.launch_descriptor(lane_label, lane)
        )
        lanes[lane_label] = lane
    return {
        "bindings": core.fixed_bindings(),
        "evaluator": {
            "evaluator_id": core.EVALUATOR_ID,
            "evaluator_spec_sha256": core.EVALUATOR_SPEC_SHA256,
        },
        "launch_authority_id": core.LAUNCH_AUTHORITY_ID,
        "lanes": lanes,
        "schema_id": core.CONTRACT_SCHEMA_ID,
        "surface_root": str(surface_root),
    }


@contextlib.contextmanager
def synthetic_authority_fixture(
    prefix: str,
    *,
    private_executable: bool = False,
    mutate_contract: Callable[[dict[str, Any]], None] | None = None,
):
    original = {
        "CANONICAL_ROOT": core.CANONICAL_ROOT,
        "LAUNCH_AUTHORITY_PATH": core.LAUNCH_AUTHORITY_PATH,
        "LAUNCH_AUTHORITY_SHA256": core.LAUNCH_AUTHORITY_SHA256,
        "LIVE_SURFACE_ROOT": core.LIVE_SURFACE_ROOT,
        "OUTPUT_ROOT": core.OUTPUT_ROOT,
        "ROOT": core.ROOT,
    }
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        synthetic_repo = Path(temporary) / "repo"
        synthetic_repo.mkdir(mode=0o700)
        (synthetic_repo / "build").mkdir(mode=0o700)
        core.ROOT = synthetic_repo
        core.CANONICAL_ROOT = synthetic_repo
        core.LIVE_SURFACE_ROOT = synthetic_repo / core.LIVE_SURFACE_REL
        core.OUTPUT_ROOT = synthetic_repo / core.OUTPUT_ROOT_REL
        if private_executable:
            executable = synthetic_repo / "synthetic-executable"
            executable.write_bytes(b"#!/bin/sh\nexit 0\n")
            executable.chmod(0o700)
        else:
            executable = Path(sys.executable).resolve()
        contract = synthetic_contract(core.LIVE_SURFACE_ROOT, executable)
        if mutate_contract is not None:
            mutate_contract(contract)
        authority_path = synthetic_repo / "synthetic-launch-authority.json"
        authority_path.write_bytes(canonical_json_bytes(contract))
        authority_path.chmod(0o400)
        core.LAUNCH_AUTHORITY_PATH = authority_path
        core.LAUNCH_AUTHORITY_SHA256 = sha256_file(authority_path)
        try:
            yield synthetic_repo, contract, executable, authority_path
        finally:
            for name, value in original.items():
                setattr(core, name, value)


@contextlib.contextmanager
def intercepted_launch(callback: Callable[..., Any]):
    original = core.subprocess.run
    core.subprocess.run = callback
    try:
        yield
    finally:
        core.subprocess.run = original


def completed(returncode: int) -> Any:
    return type("SyntheticCompletedProcess", (), {"returncode": returncode})()


def rewrite_canonical_json(path: Path, value: Any) -> None:
    path.chmod(0o600)
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o400)


def expect_failure(action: Callable[[], Any], label: str) -> None:
    try:
        action()
    except (core.SurfaceError, FileExistsError):
        return
    raise VerificationError(f"negative test accepted: {label}")


def verify_synthetic_state_machine() -> None:
    require(not core.LIVE_SURFACE_ROOT.exists(), "live surface namespace exists before tests")
    require(not core.OUTPUT_ROOT.exists(), "declared score output namespace exists before tests")

    with synthetic_authority_fixture("qk-live-surface-synthetic-") as (_repo, contract, _executable, _authority):
        core.materialize_lane("Base", "1" * 64)
        core.materialize_lane("checkpoint-176", "2" * 64)
        expect_failure(
            lambda: core.consume_and_run("checkpoint-176"),
            "second_lane_before_base_terminal",
        )
        launch_count = 0

        def fake_launch(argv: tuple[str, ...], **kwargs: Any) -> Any:
            nonlocal launch_count
            launch_count += 1
            lane_label = argv[-1]
            paths = core.lane_paths(core.LIVE_SURFACE_ROOT, lane_label)
            ledger = core.load_canonical_json(paths["ledger"])
            require(ledger["state"] == core.CONSUMED_STATE, "launch observed READY ledger")
            require(not core.lexical_exists(paths["terminal"]), "terminal existed before launch")
            require(tuple(contract["lanes"][lane_label]["shell_argv"]) == argv, "launch argv")
            require(kwargs == {
                "check": False,
                "close_fds": True,
                "cwd": core.ROOT,
                "env": core.ENVIRONMENT,
                "shell": False,
            }, "launch options")
            return completed(0 if lane_label == "Base" else 7)

        ambient = {key: os.environ.get(key) for key in core.ENVIRONMENT}
        os.environ.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
        os.environ.pop("PYTHONHASHSEED", None)
        os.environ.pop("TZ", None)
        try:
            with intercepted_launch(fake_launch):
                base_terminal = core.consume_and_run("Base")
                require(base_terminal["status"] == "SUCCEEDED_TERMINAL", "Base terminal")
                expect_failure(
                    lambda: core.consume_and_run("Base"),
                    "reconsume_after_terminal",
                )
                base_paths = core.lane_paths(core.LIVE_SURFACE_ROOT, "Base")
                expect_failure(
                    lambda: core.seal_first_terminal(
                        base_paths["terminal"],
                        base_terminal,
                        core.load_canonical_json(base_paths["ledger"]),
                    ),
                    "terminal_overwrite",
                )
                checkpoint_terminal = core.consume_and_run("checkpoint-176")
        finally:
            for key, value in ambient.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        require(launch_count == 2, "launch cardinality")
        require(checkpoint_terminal["status"] == "FAILED_TERMINAL", "checkpoint terminal")
        base_paths = core.lane_paths(core.LIVE_SURFACE_ROOT, "Base")
        checkpoint_paths = core.lane_paths(core.LIVE_SURFACE_ROOT, "checkpoint-176")
        require(base_paths["ledger"] != checkpoint_paths["ledger"], "ledger alias")
        require(base_paths["terminal"] != checkpoint_paths["terminal"], "terminal alias")

    with synthetic_authority_fixture("qk-live-surface-invalidate-"):
        core.materialize_lane("Base", "3" * 64)
        failed = core.invalidate_preflight_failure(
            "Base", "SYNTHETIC_PREFLIGHT_REJECTED"
        )
        require(failed["status"] == "FAILED_TERMINAL", "preflight terminal")
        require(failed["execution_started"] is False, "preflight execution flag")
        expect_failure(
            lambda: core.consume_and_run("Base"),
            "run_after_preflight_invalidation",
        )

    with synthetic_authority_fixture("qk-live-surface-mutations-") as (_repo, contract, _executable, _authority):
        alias = copy.deepcopy(contract)
        alias_lane = alias["lanes"]["checkpoint-176"]
        alias_lane["shell_argv"] = alias["lanes"]["Base"]["shell_argv"]
        alias_lane["launch_descriptor_sha256"] = core.object_sha256(
            core.launch_descriptor("checkpoint-176", alias_lane)
        )
        expect_failure(lambda: core.validate_contract(alias), "cross_lane_argv_alias")
        environment = copy.deepcopy(contract)
        environment["lanes"]["Base"]["environment"]["TZ"] = "Etc/UTC"
        expect_failure(lambda: core.validate_contract(environment), "environment_mutation")
        missing_argv = copy.deepcopy(contract)
        missing_argv["lanes"]["Base"]["shell_argv"] = None
        expect_failure(lambda: core.validate_contract(missing_argv), "missing_shell_argv")
        launch_hash = copy.deepcopy(contract)
        launch_hash["lanes"]["Base"]["shell_argv"][1] = "--mutated"
        expect_failure(
            lambda: core.validate_contract(launch_hash),
            "launch_descriptor_mutation",
        )
        wrong_root = copy.deepcopy(contract)
        wrong_root["surface_root"] = str(core.LIVE_SURFACE_ROOT.parent / "other")
        expect_failure(
            lambda: core.validate_contract(wrong_root),
            "noncanonical_surface_root",
        )

    with synthetic_authority_fixture("qk-live-surface-collision-") as (_repo, contract, _executable, _authority):
        collision = core.ROOT / contract["lanes"]["Base"]["output_path"]
        collision.parent.mkdir(parents=True)
        collision.write_text("synthetic collision\n", encoding="ascii")
        expect_failure(lambda: core.materialize_lane("Base", "4" * 64), "output_collision")
        require(not core.LIVE_SURFACE_ROOT.exists(), "collision test created surface")

    with synthetic_authority_fixture("qk-live-surface-dangling-"):
        core.OUTPUT_ROOT.symlink_to(core.ROOT / "missing-output-target")
        expect_failure(
            lambda: core.materialize_lane("Base", "5" * 64),
            "dangling_output_symlink",
        )
        require(not core.LIVE_SURFACE_ROOT.exists(), "dangling symlink test created surface")

    def missing_executable(contract: dict[str, Any]) -> None:
        missing = str(Path("/synthetic/missing/executable"))
        for lane_label, lane in contract["lanes"].items():
            lane["shell_argv"][0] = missing
            lane["executable_sha256"] = "0" * 64
            lane["launch_descriptor_sha256"] = core.object_sha256(
                core.launch_descriptor(lane_label, lane)
            )

    with synthetic_authority_fixture(
        "qk-live-surface-missing-executable-",
        mutate_contract=missing_executable,
    ):
        expect_failure(
            lambda: core.materialize_lane("Base", "6" * 64),
            "missing_executable",
        )
        require(not core.LIVE_SURFACE_ROOT.exists(), "missing executable created surface")

    with synthetic_authority_fixture(
        "qk-live-surface-executable-mutation-",
        private_executable=True,
    ) as (_repo, _contract, executable, _authority):
        core.materialize_lane("Base", "7" * 64)
        executable.write_bytes(b"#!/bin/sh\nexit 9\n")
        executable.chmod(0o700)
        expect_failure(
            lambda: core.consume_and_run("Base"),
            "executable_hash_mutation",
        )

    with synthetic_authority_fixture("qk-live-surface-prior-ledger-"):
        core.materialize_lane("Base", "8" * 64)
        core.materialize_lane("checkpoint-176", "9" * 64)
        launch_count = 0

        def count_launch(_argv: tuple[str, ...], **_kwargs: Any) -> Any:
            nonlocal launch_count
            launch_count += 1
            return completed(0)

        with intercepted_launch(count_launch):
            core.consume_and_run("Base")
            base_ledger_path = core.lane_paths(core.LIVE_SURFACE_ROOT, "Base")["ledger"]
            tampered = core.load_canonical_json(base_ledger_path)
            tampered["environment"]["LANG"] = "C.UTF-8"
            tampered["ready_record_sha256"] = "0" * 64
            rewrite_canonical_json(base_ledger_path, tampered)
            expect_failure(
                lambda: core.consume_and_run("checkpoint-176"),
                "prior_ledger_identity_mutation",
            )
        require(launch_count == 1, "tampered prior ledger reached launch")

    with synthetic_authority_fixture("qk-live-surface-artifact-closure-"):
        core.materialize_lane("Base", "a" * 64)
        original_validate = core.validate_bound_artifacts
        validation_count = 0
        launch_count = 0

        def stateful_artifact_validation() -> None:
            nonlocal validation_count
            validation_count += 1
            if validation_count == 1:
                original_validate()
            else:
                raise core.SurfaceError("synthetic bound artifact mutation")

        def unreachable_launch(_argv: tuple[str, ...], **_kwargs: Any) -> Any:
            nonlocal launch_count
            launch_count += 1
            return completed(0)

        core.validate_bound_artifacts = stateful_artifact_validation
        try:
            with intercepted_launch(unreachable_launch):
                expect_failure(
                    lambda: core.consume_and_run("Base"),
                    "postconsume_artifact_identity_mutation",
                )
        finally:
            core.validate_bound_artifacts = original_validate
        require(launch_count == 0, "artifact mutation reached launch")
        terminal = core.load_canonical_json(
            core.lane_paths(core.LIVE_SURFACE_ROOT, "Base")["terminal"]
        )
        require(terminal["execution_started"] is False, "artifact closure terminal")

    with synthetic_authority_fixture("qk-live-surface-authority-hash-") as (_repo, contract, _executable, authority_path):
        authority_path.chmod(0o600)
        authority_path.write_bytes(canonical_json_bytes({**contract, "schema_version": 99}))
        authority_path.chmod(0o400)
        expect_failure(
            lambda: core.materialize_lane("Base", "b" * 64),
            "launch_authority_hash_mutation",
        )
        require(not core.LIVE_SURFACE_ROOT.exists(), "authority mutation created surface")

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            core.main()
        except SystemExit as exc:
            require(exc.code == 2, "live entrypoint exit status")
        else:
            raise VerificationError("live entrypoint returned")
    require("LIVE_SURFACE_BLOCKED_NO_EXECUTION_PERFORMED" in stderr.getvalue(), "live entrypoint marker")
    require(not core.LIVE_SURFACE_ROOT.exists(), "live surface namespace exists after tests")
    require(not core.OUTPUT_ROOT.exists(), "declared score output namespace exists after tests")


def verify_package_and_proof() -> None:
    require(load_canonical_json(PACKAGE) == expected_package(), "implementation package mismatch")
    require(load_canonical_json(PROOF) == expected_proof(), "static synthetic proof mismatch")


def verify_checksums() -> None:
    expected = "".join(
        f"{sha256_file(path)}  {relative}\n"
        for relative, path in [
            (PACKAGE_REL, PACKAGE),
            (IMPLEMENTATION_REL, IMPLEMENTATION),
            (VERIFIER_REL, VERIFIER),
            (PROOF_REL, PROOF),
        ]
    )
    require(CHECKSUMS.is_file(), "missing checksum manifest")
    require(not CHECKSUMS.is_symlink(), "symlinked checksum manifest")
    require(CHECKSUMS.read_text(encoding="ascii") == expected, "checksum manifest mismatch")


def main() -> int:
    require(platform.python_implementation() == "CPython", "Python implementation mismatch")
    require(platform.python_version() == "3.13.5", "Python version mismatch")
    verify_external_bindings()
    verify_source_boundaries()
    verify_synthetic_state_machine()
    verify_package_and_proof()
    verify_checksums()
    require(not core.LIVE_SURFACE_ROOT.exists(), "live surface namespace exists at completion")
    require(not core.OUTPUT_ROOT.exists(), "score output namespace exists at completion")
    print(
        "STATIC_VALID_SYNTHETIC_ONLY_NO_EXECUTION_PERFORMED "
        f"package_sha256={sha256_file(PACKAGE)} "
        f"implementation_sha256={sha256_file(IMPLEMENTATION)} "
        "live_surface_namespaces=0 score_output_namespaces=0 sealed_tensor_accesses=0"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, core.SurfaceError) as exc:
        print(f"STATIC_INVALID_NO_EXECUTION_PERFORMED: {exc}", file=sys.stderr)
        raise SystemExit(1)
