#!/usr/bin/env python3
"""Phase-safe fixed-input-scale launcher for nonofficial hybrid attempt 0018."""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0018"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0018.json"
SELECTION = PRIVATE / "candidate-selection-0018.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0018.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0018.json"
INPUT_SCALE_REGRESSION = PRIVATE / "fixed-input-scale-regression-0018.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0018.json"
LIFECYCLE_REGRESSION = PRIVATE / "record-lifecycle-regression-0018.json"
GATE_ORDER_REGRESSION = PRIVATE / "gate-order-regression-0018.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0018.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0018.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0018.json"
ARGV_SHAPE_REGRESSION = PRIVATE / "argv-shape-regression-0018.json"
CLOSURE_CERTIFICATE = PRIVATE / "closure-certificate-0018.json"
PHASE_TRANSITION_REGRESSION = PRIVATE / "phase-transition-regression-0018.json"

PRE_IMPORT_ENVIRONMENT = dict(os.environ)
PRE_IMPORT_SYS_PATH = list(sys.path)
PRE_IMPORT_SYS_PREFIX = sys.prefix
PRE_IMPORT_ARGV = list(sys.argv)
PRE_IMPORT_CWD = str(Path.cwd().resolve())
PRE_IMPORT_MODULES = frozenset(sys.modules)

SEALED_0015_TERMINAL_SHA256 = (
    "6b2c3e97460f37dafe41e7f0bf0c02c937be7326642074f3de9b70433fbd2186"
)
SEALED_0015_LAUNCHER_SHA256 = (
    "163e168bfc9c8f8aff111dfe412a4aa726fcbe74bdc4446bed61211772c307be"
)
SEALED_0016_TERMINAL_SHA256 = (
    "40940183e82ddc86aa74421db4d509d2f4ea2295c05953a41e5ce160ebbaecfe"
)
SEALED_0016_LAUNCHER_SHA256 = (
    "f5155317dd15e61f55bba495757ffdeee3b54ffa9cad15c0554f11a40e7d8aa4"
)
SEALED_0017_TERMINAL_SHA256 = (
    "2761f66b627358a97bd344cc94bc79806613bcdf290098149683835eeaf05ba6"
)
SEALED_0017_LAUNCHER_SHA256 = (
    "594d10c4280251a0d01150b978951fc18b3c52c1f08a6849128130b3b35f5f94"
)
SEALED_0017_CANONICAL_ARGV_SHA256 = (
    "1cd750c1003a2f1d2709b5ebd976aed47f002936c0e088ab39206877d4f1bcc7"
)

CANDIDATES = (
    {"id": "fresh-0018-a", "text": "Why do cattails sway?"},
    {"id": "fresh-0018-b", "text": "How do acorns tumble?"},
    {"id": "fresh-0018-c", "text": "Where do fireflies rest?"},
    {"id": "fresh-0018-d", "text": "Why does basalt weather?"},
    {"id": "fresh-0018-e", "text": "How do cobwebs glisten?"},
    {"id": "fresh-0018-f", "text": "What makes reeds rustle?"},
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _actual_process_argv(pre_import_argv: list[str]) -> list[str]:
    return [str(Path(sys.executable).resolve()), *pre_import_argv]


def _sanitized_traceback(error: BaseException) -> dict[str, Any]:
    frames = []
    rendered = ["Traceback (most recent call last):"]
    for frame in traceback.extract_tb(error.__traceback__):
        path = Path(frame.filename)
        try:
            source_file = str(path.resolve().relative_to(ROOT))
        except (OSError, ValueError):
            source_file = f"<external>/{path.name}"
        frames.append(
            {
                "source_file": source_file,
                "function": frame.name,
                "line": frame.lineno,
                "expression": frame.line or "",
            }
        )
        rendered.append(f'  File "{source_file}", line {frame.lineno}, in {frame.name}')
        if frame.line:
            rendered.append(f"    {frame.line}")
    rendered.append(f"{type(error).__name__}: {error}")
    return {
        "exception_type": type(error).__name__,
        "message": str(error),
        "frames": frames,
        "formatted": "\n".join(rendered) + "\n",
    }


def _seal_preloader_failure(error: BaseException) -> None:
    if TERMINAL.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid18execute",
        "attempt_identity": "nonofficial-hybrid-0018",
        "status": "FAILED_SEALED_NO_EXECUTION",
        "outcome": "pre_execution_failure",
        "sealed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "process_start_count": 0,
        "process_started": False,
        "failure": {
            "failure_taxonomy": "pre_execution_gate_failure",
            "phase": "stdlib_preloader_or_nonconsuming_gate",
            "root_cause_hypothesis": f"{type(error).__name__}: {error}",
            "traceback": _sanitized_traceback(error),
            "regression": "No retry, replay, or resume; preserve all 0018 artifacts.",
        },
    }
    try:
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _patch_inherited_chain(prior: Any, prior_0014: Any) -> None:
    for name, value in {
        "LAUNCHER": LAUNCHER,
        "OUTPUT": OUTPUT,
        "TERMINAL": TERMINAL,
        "PRIVATE": PRIVATE,
        "PROMPTS": PROMPTS,
        "SELECTION": SELECTION,
        "TOKENIZATION_PREFLIGHT": TOKENIZATION_PREFLIGHT,
        "TOKENIZATION_REGRESSION": TOKENIZATION_REGRESSION,
        "INPUT_SCALE_REGRESSION": INPUT_SCALE_REGRESSION,
        "SCALE_REGRESSION": SCALE_REGRESSION,
        "LIFECYCLE_REGRESSION": LIFECYCLE_REGRESSION,
        "GATE_ORDER_REGRESSION": GATE_ORDER_REGRESSION,
        "TEMPLATE_GATE": TEMPLATE_GATE,
        "DEPENDENCY_PROBE": DEPENDENCY_PROBE,
        "LAUNCH_REGRESSION": LAUNCH_REGRESSION,
        "CANDIDATES": CANDIDATES,
        "PRE_IMPORT_ENVIRONMENT": PRE_IMPORT_ENVIRONMENT,
        "PRE_IMPORT_SYS_PATH": PRE_IMPORT_SYS_PATH,
        "PRE_IMPORT_SYS_PREFIX": PRE_IMPORT_SYS_PREFIX,
        "PRE_IMPORT_ARGV": PRE_IMPORT_ARGV,
        "PRE_IMPORT_CWD": PRE_IMPORT_CWD,
        "PRE_IMPORT_MODULES": PRE_IMPORT_MODULES,
    }.items():
        setattr(prior, name, value)
    prior._configure_0014(prior_0014)
    prior_0014.DIAGNOSTIC_PROMPT_SOURCES = (
        *prior_0014.DIAGNOSTIC_PROMPT_SOURCES,
        PRIVATE / "prompts-0015.json",
        PRIVATE / "prompts-0016.json",
        PRIVATE / "prompts-0017.json",
    )
    prior_0014.SEALED_BOUNDARY_SHA256 = {
        **prior_0014.SEALED_BOUNDARY_SHA256,
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0015/terminal-record.json"
        ): SEALED_0015_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0015.py": (
            SEALED_0015_LAUNCHER_SHA256
        ),
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0016/terminal-record.json"
        ): SEALED_0016_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0016.py": (
            SEALED_0016_LAUNCHER_SHA256
        ),
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0017/terminal-record.json"
        ): SEALED_0017_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0017.py": (
            SEALED_0017_LAUNCHER_SHA256
        ),
    }


def _configure_runner(
    runner: Any,
    prior: Any,
    predecessor_modules: tuple[Any, ...],
) -> None:
    runner.MISSION_ID = "stage1rank1hybrid18execute"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0018"
    runner.OUTPUT = OUTPUT
    runner.PREFLIGHT = OUTPUT / "preflight.json"
    runner.FREEZE = OUTPUT / "freeze.json"
    runner.PACKAGE = OUTPUT / "execution-package.json"
    runner.AUTHORITY = OUTPUT / "execution-authority.json"
    runner.LIVE = OUTPUT / "live"
    runner.RESULT = runner.LIVE / "result.json"
    runner.FAILURE = runner.LIVE / "failure.json"
    runner.SUMS = runner.LIVE / "SHA256SUMS"
    runner.STATE = OUTPUT / "supervisor-state"
    runner.STDOUT = OUTPUT / "execution.stdout.log"
    runner.STDERR = OUTPUT / "execution.stderr.log"
    runner.TERMINAL = TERMINAL
    runner.DEFAULT_PROMPTS = PROMPTS
    runner.TEMPLATE_GATE = TEMPLATE_GATE
    runner.DEPENDENCY_PROBE = DEPENDENCY_PROBE
    runner.PREDECESSOR_TERMINAL_SEALS = {
        **runner.PREDECESSOR_TERMINAL_SEALS,
        "nonofficial-hybrid-0015": {
            "sha256": SEALED_0015_TERMINAL_SHA256,
            "process_start_count": 1,
            "process_started": True,
        },
        "nonofficial-hybrid-0016": {
            "sha256": SEALED_0016_TERMINAL_SHA256,
            "process_start_count": 1,
            "process_started": True,
        },
        "nonofficial-hybrid-0017": {
            "sha256": SEALED_0017_TERMINAL_SHA256,
            "process_start_count": 1,
            "process_started": True,
        },
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_package_bindings = runner.package_bindings
    prior_package_template = runner.package_template_invariant
    prior_preflight = runner.preflight
    prior_freeze = runner.freeze
    prior_package = runner.package_execution
    prior_authority = runner.grant_authority
    prior_execute = runner.execute
    prior_verify = runner.verify

    terminal_0017 = (
        ROOT
        / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
        "nonofficial-hybrid-0017/terminal-record.json"
    )
    durable_0017 = terminal_0017.parent / "supervisor-state/terminal_record.json"
    launcher_0017 = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0017.py"
    package_0017 = terminal_0017.parent / "execution-package.json"
    authority_0017 = terminal_0017.parent / "execution-authority.json"
    intent_0017 = terminal_0017.parent / "supervisor-state/pre_start_intent.json"
    started_0017 = terminal_0017.parent / "supervisor-state/process_started.json"

    def verify_0017_seal() -> dict[str, Any]:
        sealed_paths = (
            terminal_0017,
            durable_0017,
            launcher_0017,
            package_0017,
            authority_0017,
            intent_0017,
            started_0017,
        )
        for path in sealed_paths:
            runner.require(
                stat.S_ISREG(os.lstat(path).st_mode),
                f"sealed 0017 artifact is not regular: {runner.public_path(path)}",
            )
        terminal = runner.read_json(terminal_0017)
        package = runner.read_json(package_0017)
        authority = runner.read_json(authority_0017)
        intent = runner.read_json(intent_0017)
        started = runner.read_json(started_0017)
        argv = terminal["process"]["argv"]
        argv_sha256 = runner.supervisor.canonical_value_sha256(argv)
        package_measurement = runner.supervisor.measure_file(package_0017)
        authority_measurement = runner.supervisor.measure_file(authority_0017)
        expected_package_provenance = {
            "path": str(package_0017),
            "byte_count": package_measurement.byte_count,
            "sha256": package_measurement.sha256,
            "package_id": package["package_id"],
        }
        expected_authority_provenance = {
            "path": str(authority_0017),
            "byte_count": authority_measurement.byte_count,
            "sha256": authority_measurement.sha256,
            "authority_id": authority["authority_id"],
        }
        runner.require(
            runner.sha256_file(terminal_0017) == SEALED_0017_TERMINAL_SHA256
            and runner.sha256_file(durable_0017) == SEALED_0017_TERMINAL_SHA256
            and runner.sha256_file(launcher_0017) == SEALED_0017_LAUNCHER_SHA256
            and terminal == runner.read_json(durable_0017)
            and terminal.get("process_start_count") == 1
            and terminal.get("process_started") is True
            and terminal.get("outcome") == "child_nonzero_exit",
            "sealed 0017 terminal or launcher changed",
        )
        runner.require(
            len(argv) == 7
            and argv_sha256 == SEALED_0017_CANONICAL_ARGV_SHA256
            and terminal["process"]["argv_sha256"] == argv_sha256
            and package["bindings"]["argv"] == argv
            and authority["bindings"]["argv"] == argv
            and intent["bindings"]["argv"] == argv
            and package["bindings"]["argv_sha256"] == argv_sha256
            and authority["bindings"]["argv_sha256"] == argv_sha256
            and intent["bindings"]["argv_sha256"] == argv_sha256,
            "sealed 0017 package/authority/supervisor argv changed",
        )
        runner.require(
            authority["package"] == expected_package_provenance
            and intent["provenance"]["package"] == expected_package_provenance
            and terminal["provenance"]["package"] == expected_package_provenance
            and intent["provenance"]["authority"] == expected_authority_provenance
            and started["provenance"] == intent["provenance"]
            and terminal["provenance"] == intent["provenance"]
            and started["process_start_count"] == 1,
            "sealed 0017 provenance chain changed",
        )
        return {
            "terminal": runner.file_record(terminal_0017),
            "launcher": runner.file_record(launcher_0017),
            "package": runner.file_record(package_0017),
            "authority": runner.file_record(authority_0017),
            "pre_start_intent": runner.file_record(intent_0017),
            "process_started": runner.file_record(started_0017),
            "canonical_argv_sha256": argv_sha256,
        }

    def verify_predecessor_seals_0018() -> None:
        prior_verify_predecessor_seals()
        verify_0017_seal()

    runner.verify_predecessor_seals = verify_predecessor_seals_0018

    def module_sources() -> list[dict[str, Any]]:
        return [
            {
                "module": module.__name__,
                "source": runner.file_record(Path(module.__file__).resolve(strict=True)),
            }
            for module in predecessor_modules
        ]

    def expected_argv_shape_regression() -> dict[str, Any]:
        pre_import_argv = [
            str(LAUNCHER),
            "--execute",
            "--prompts",
            str(PROMPTS.resolve()),
            "--freeze-sha256",
            "0" * 64,
        ]
        process_argv = [
            str(Path(sys.executable).resolve()),
            *pre_import_argv,
        ]
        reconstructed = _actual_process_argv(pre_import_argv)
        tampered_interpreter = [str(ROOT / "tampered-python"), *pre_import_argv]
        tampered_script = [
            process_argv[0],
            str(ROOT / "tools/tampered-launcher.py"),
            *pre_import_argv[1:],
        ]
        tampered_argument = [*process_argv[:-1], "f" * 64]
        process_argv_sha256 = runner.supervisor.canonical_value_sha256(process_argv)

        def tamper_fails(candidate: list[str]) -> bool:
            return (
                candidate != process_argv
                and runner.supervisor.canonical_value_sha256(candidate)
                != process_argv_sha256
            )

        return {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_RECONSTRUCTED_PROCESS_ARGV_SHAPE_AND_TAMPER_REJECTION",
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
            "legacy_pre_import_argv_length": len(pre_import_argv),
            "supervisor_process_argv_length": len(process_argv),
            "process_argv_sha256": process_argv_sha256,
            "checks": {
                "legacy_six_vs_seven_direct_comparison_fails": (
                    len(pre_import_argv) == 6
                    and len(process_argv) == 7
                    and pre_import_argv != process_argv
                ),
                "resolved_interpreter_plus_pre_import_argv_passes": (
                    reconstructed == process_argv
                ),
                "interpreter_tampering_fails": tamper_fails(tampered_interpreter),
                "script_tampering_fails": tamper_fails(tampered_script),
                "argument_tampering_fails": tamper_fails(tampered_argument),
                "pre_import_script_resolves_to_exact_launcher": (
                    Path(pre_import_argv[0]).resolve() == LAUNCHER
                ),
                "process_not_started": True,
            },
        }

    def verify_argv_shape_regression() -> dict[str, Any]:
        runner.require(
            stat.S_ISREG(os.lstat(ARGV_SHAPE_REGRESSION).st_mode),
            "0018 argv-shape regression is absent or not regular",
        )
        record = runner.read_json(ARGV_SHAPE_REGRESSION)
        runner.require(
            record == expected_argv_shape_regression()
            and all(record["checks"].values()),
            "0018 argv-shape regression changed",
        )
        return record

    def argv_shape_regression() -> int:
        runner.verify_exact_interpreter()
        runner.verify_predecessor_seals()
        runner.validate_no_execution()
        dependencies = (
            SELECTION,
            LIFECYCLE_REGRESSION,
            DEPENDENCY_PROBE,
            GATE_ORDER_REGRESSION,
            INPUT_SCALE_REGRESSION,
            TOKENIZATION_PREFLIGHT,
            TOKENIZATION_REGRESSION,
            SCALE_REGRESSION,
        )
        runner.require(
            all(path.is_file() for path in dependencies),
            "argv-shape gate requires all dependency-first regressions",
        )
        runner.require(
            not ARGV_SHAPE_REGRESSION.exists(),
            "0018 argv-shape regression already exists",
        )
        runner.write_json(ARGV_SHAPE_REGRESSION, expected_argv_shape_regression())
        record = verify_argv_shape_regression()
        print(
            "ACE2_HYBRID_0018_ARGV_SHAPE_REGRESSION_PASS "
            f"argv_sha256={record['process_argv_sha256']} "
            "legacy_argv_length=6 process_argv_length=7 process_start_count=0",
            flush=True,
        )
        return 0

    runner.argv_shape_regression = argv_shape_regression
    runner.verify_argv_shape_regression = verify_argv_shape_regression

    def execution_envelope() -> dict[str, Any]:
        argv_template = [
            str(Path(sys.executable).resolve()),
            str(LAUNCHER),
            "--execute",
            "--prompts",
            str(PROMPTS.resolve(strict=True)),
            "--freeze-sha256",
            "<sha256-of-freeze.json>",
        ]
        modules = sorted(PRE_IMPORT_MODULES)
        return {
            "argv_template": argv_template,
            "argv_freeze_binding_policy": (
                "replace only the final placeholder with the current freeze.json SHA-256"
            ),
            "environment": PRE_IMPORT_ENVIRONMENT,
            "environment_sha256": runner.supervisor.canonical_value_sha256(
                PRE_IMPORT_ENVIRONMENT
            ),
            "cwd": str(ROOT),
            "pre_import_modules": modules,
            "pre_import_modules_sha256": runner.supervisor.canonical_value_sha256(
                modules
            ),
            "reservation": {
                "state_dir": str(runner.STATE),
                "pre_start_intent": str(runner.STATE / "pre_start_intent.json"),
                "process_started": str(runner.STATE / "process_started.json"),
                "terminal_record": str(runner.STATE / "terminal_record.json"),
                "process_start_count": 1,
            },
        }

    def binding_core() -> dict[str, Any]:
        return {
            "schema": "ace2-phase-safe-binding-core-v1",
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "claim_boundary": (
                "bounded nonofficial hybrid evidence only; not Stage-1 certification"
            ),
            "execution_envelope": execution_envelope(),
            "launcher": runner.file_record(LAUNCHER),
            "private_prompts": runner.file_record(PROMPTS, expose_path=False),
            "supervisor": runner.file_record(
                Path(runner.supervisor.__file__).resolve(strict=True)
            ),
            "predecessor_modules": module_sources(),
            "sealed_predecessor": verify_0017_seal(),
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
        }

    def historical_observation_contract() -> dict[str, Any]:
        return {
            "phase": "pre_freeze_preauthority",
            "authority_present": False,
            "package_present": False,
            "freeze_present": False,
            "pre_start_intent_present": False,
            "process_started_present": False,
            "terminal_present": False,
            "result_present": False,
            "process_start_count": 0,
        }

    def observe_preauthority() -> dict[str, Any]:
        return {
            "phase": "pre_freeze_preauthority",
            "authority_present": runner.AUTHORITY.exists(),
            "package_present": runner.PACKAGE.exists(),
            "freeze_present": runner.FREEZE.exists(),
            "pre_start_intent_present": (
                runner.STATE / "pre_start_intent.json"
            ).exists(),
            "process_started_present": (
                runner.STATE / "process_started.json"
            ).exists(),
            "terminal_present": runner.TERMINAL.exists(),
            "result_present": runner.RESULT.exists(),
            "process_start_count": 0,
        }

    def verify_closure_certificate() -> dict[str, Any]:
        runner.require(
            stat.S_ISREG(os.lstat(CLOSURE_CERTIFICATE).st_mode),
            "0018 closure certificate is absent or not regular",
        )
        certificate = runner.read_json(CLOSURE_CERTIFICATE)
        core = binding_core()
        core_sha256 = runner.supervisor.canonical_value_sha256(core)
        observation = certificate.get("preauthority_observation")
        runner.require(
            set(certificate)
            == {
                "schema_version",
                "status",
                "binding_core",
                "binding_core_sha256",
                "preauthority_observation",
                "preauthority_observation_sha256",
            }
            and certificate.get("schema_version") == 2
            and certificate.get("status") == "PASS_IMMUTABLE_PHASE_SAFE_BINDING"
            and certificate.get("binding_core") == core
            and certificate.get("binding_core_sha256") == core_sha256
            and observation == historical_observation_contract()
            and certificate.get("preauthority_observation_sha256")
            == runner.supervisor.canonical_value_sha256(observation),
            "0018 immutable closure certificate changed",
        )
        return certificate

    def expected_phase_regression(certificate: dict[str, Any]) -> dict[str, Any]:
        original_whole = {
            "binding_core": certificate["binding_core"],
            "preauthority_observation": certificate["preauthority_observation"],
        }
        transitioned_observation = {
            **certificate["preauthority_observation"],
            "phase": "child_entry_after_authority_and_start",
            "authority_present": True,
            "package_present": True,
            "freeze_present": True,
            "pre_start_intent_present": True,
            "process_started_present": True,
            "process_start_count": 1,
        }
        legacy_recomputed = {
            "binding_core": binding_core(),
            "preauthority_observation": transitioned_observation,
        }
        tampered_core = copy.deepcopy(certificate["binding_core"])
        tampered_core["execution_envelope"]["cwd"] = str(ROOT / "tampered")
        original_whole_sha256 = runner.supervisor.canonical_value_sha256(
            original_whole
        )
        legacy_sha256 = runner.supervisor.canonical_value_sha256(legacy_recomputed)
        tampered_sha256 = runner.supervisor.canonical_value_sha256(tampered_core)
        return {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": (
                "PASS_LEGACY_WHOLE_RECORD_FAILS_PHASE_SEPARATED_PASSES_TAMPER_FAILS"
            ),
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "original_whole_record_sha256": original_whole_sha256,
            "legacy_transitioned_whole_record_sha256": legacy_sha256,
            "binding_core_sha256": certificate["binding_core_sha256"],
            "tampered_binding_core_sha256": tampered_sha256,
            "checks": {
                "legacy_whole_record_recomputation_fails_after_authority_start": (
                    legacy_sha256 != original_whole_sha256
                ),
                "historical_preauthority_observation_retained_without_recomputation": (
                    original_whole["preauthority_observation"]
                    == historical_observation_contract()
                ),
                "phase_separated_binding_core_recomputation_passes": (
                    runner.supervisor.canonical_value_sha256(binding_core())
                    == certificate["binding_core_sha256"]
                ),
                "binding_core_tamper_fails_digest_verification": (
                    tampered_sha256 != certificate["binding_core_sha256"]
                ),
                "process_not_started": True,
            },
        }

    def verify_phase_transition_regression() -> dict[str, Any]:
        runner.require(
            PHASE_TRANSITION_REGRESSION.is_file(),
            "0018 phase-transition regression is absent",
        )
        certificate = verify_closure_certificate()
        record = runner.read_json(PHASE_TRANSITION_REGRESSION)
        runner.require(
            record == expected_phase_regression(certificate)
            and all(record["checks"].values()),
            "0018 phase-transition regression changed",
        )
        return record

    def phase_binding_regression() -> int:
        runner.verify_exact_interpreter()
        runner.verify_predecessor_seals()
        runner.validate_no_execution()
        required = (
            SELECTION,
            LIFECYCLE_REGRESSION,
            DEPENDENCY_PROBE,
            GATE_ORDER_REGRESSION,
            INPUT_SCALE_REGRESSION,
            TOKENIZATION_PREFLIGHT,
            TOKENIZATION_REGRESSION,
            SCALE_REGRESSION,
            ARGV_SHAPE_REGRESSION,
        )
        runner.require(
            all(path.is_file() for path in required),
            "phase-binding gate requires all dependency-first regressions",
        )
        runner.require(
            not CLOSURE_CERTIFICATE.exists()
            and not PHASE_TRANSITION_REGRESSION.exists(),
            "0018 phase-binding artifacts already exist",
        )
        verify_argv_shape_regression()
        observation = observe_preauthority()
        runner.require(
            observation == historical_observation_contract(),
            "0018 preauthority observation is not pristine",
        )
        core = binding_core()
        certificate = {
            "schema_version": 2,
            "status": "PASS_IMMUTABLE_PHASE_SAFE_BINDING",
            "binding_core": core,
            "binding_core_sha256": runner.supervisor.canonical_value_sha256(core),
            "preauthority_observation": observation,
            "preauthority_observation_sha256": (
                runner.supervisor.canonical_value_sha256(observation)
            ),
        }
        runner.write_json(CLOSURE_CERTIFICATE, certificate)
        runner.write_json(
            PHASE_TRANSITION_REGRESSION,
            expected_phase_regression(certificate),
        )
        verify_phase_transition_regression()
        print(
            "ACE2_HYBRID_0018_PHASE_BINDING_REGRESSION_PASS "
            f"certificate_sha256={runner.sha256_file(CLOSURE_CERTIFICATE)} "
            f"binding_core_sha256={certificate['binding_core_sha256']} "
            "process_start_count=0",
            flush=True,
        )
        return 0

    runner.phase_binding_regression = phase_binding_regression
    runner.verify_closure_certificate = verify_closure_certificate
    runner.verify_phase_transition_regression = verify_phase_transition_regression

    def phase_binding_manifest() -> dict[str, Any]:
        certificate = verify_closure_certificate()
        verify_phase_transition_regression()
        envelope = certificate["binding_core"]["execution_envelope"]
        return {
            "certificate": runner.supervisor.file_binding(CLOSURE_CERTIFICATE),
            "binding_core_sha256": certificate["binding_core_sha256"],
            "argv_shape_regression": runner.supervisor.file_binding(
                ARGV_SHAPE_REGRESSION
            ),
            "execution_envelope": envelope,
            "execution_envelope_sha256": (
                runner.supervisor.canonical_value_sha256(envelope)
            ),
            "phase_transition_regression": runner.supervisor.file_binding(
                PHASE_TRANSITION_REGRESSION
            ),
        }

    def source_records_0018(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(model, tokenizer_json, tokenizer_config, prompts)
        for path in (
            ARGV_SHAPE_REGRESSION,
            CLOSURE_CERTIFICATE,
            PHASE_TRANSITION_REGRESSION,
        ):
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0018

    def execution_bound_paths_0018(
        prompts_path: Path,
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        *,
        include_frozen_stage_files: bool,
    ) -> list[Path]:
        return sorted(
            {
                *prior_execution_bound_paths(
                    prompts_path,
                    model,
                    tokenizer_json,
                    tokenizer_config,
                    include_frozen_stage_files=include_frozen_stage_files,
                ),
                ARGV_SHAPE_REGRESSION.resolve(strict=True),
                CLOSURE_CERTIFICATE.resolve(strict=True),
                PHASE_TRANSITION_REGRESSION.resolve(strict=True),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0018

    def package_bindings_0018(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bindings = prior_package_bindings(*args, **kwargs)
        bindings["phase_binding"] = phase_binding_manifest()
        return bindings

    runner.package_bindings = package_bindings_0018

    def package_template_0018(prompts_path: Path) -> int:
        verify_phase_transition_regression()
        result = prior_package_template(prompts_path)
        gate = runner.read_json(runner.TEMPLATE_GATE)
        gate["phase_safe_binding"] = {
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": verify_closure_certificate()[
                "binding_core_sha256"
            ],
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
        }
        gate["checks"]["phase_safe_binding_template_passed"] = True
        runner.write_json(runner.TEMPLATE_GATE, gate)
        return result

    runner.package_template_invariant = package_template_0018

    def preflight_0018(prompts_path: Path) -> int:
        verify_phase_transition_regression()
        result = prior_preflight(prompts_path)
        record = runner.read_json(runner.PREFLIGHT)
        record["phase_safe_binding"] = {
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": verify_closure_certificate()[
                "binding_core_sha256"
            ],
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
        }
        record["checks"]["phase_safe_binding_regression_passed"] = True
        runner.write_json(runner.PREFLIGHT, record)
        return result

    runner.preflight = preflight_0018

    def freeze_0018(prompts_path: Path) -> int:
        certificate = verify_closure_certificate()
        verify_phase_transition_regression()
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["sealed_0017"] = verify_0017_seal()
        frozen["phase_safe_binding"] = {
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": certificate["binding_core_sha256"],
            "preauthority_observation_sha256": certificate[
                "preauthority_observation_sha256"
            ],
            "execution_envelope": certificate["binding_core"][
                "execution_envelope"
            ],
            "execution_envelope_sha256": (
                runner.supervisor.canonical_value_sha256(
                    certificate["binding_core"]["execution_envelope"]
                )
            ),
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
        }
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0018

    def verify_frozen_phase_binding() -> dict[str, Any]:
        certificate = verify_closure_certificate()
        regression = verify_phase_transition_regression()
        frozen = runner.read_json(runner.FREEZE)
        expected = {
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": certificate["binding_core_sha256"],
            "preauthority_observation_sha256": certificate[
                "preauthority_observation_sha256"
            ],
            "execution_envelope": certificate["binding_core"][
                "execution_envelope"
            ],
            "execution_envelope_sha256": (
                runner.supervisor.canonical_value_sha256(
                    certificate["binding_core"]["execution_envelope"]
                )
            ),
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
        }
        runner.require(
            frozen.get("phase_safe_binding") == expected
            and regression["binding_core_sha256"]
            == certificate["binding_core_sha256"],
            "frozen phase-safe binding changed",
        )
        return expected

    def package_0018(prompts_path: Path) -> int:
        verify_frozen_phase_binding()
        result = prior_package(prompts_path)
        package = runner.read_json(runner.PACKAGE)
        runner.require(
            package["bindings"].get("phase_binding") == phase_binding_manifest(),
            "execution package lost phase-safe binding",
        )
        return result

    runner.package_execution = package_0018

    def authority_0018(prompts_path: Path) -> int:
        verify_frozen_phase_binding()
        result = prior_authority(prompts_path)
        package = runner.read_json(runner.PACKAGE)
        authority = runner.read_json(runner.AUTHORITY)
        runner.require(
            package["bindings"]["phase_binding"] == phase_binding_manifest()
            and authority["bindings"]["phase_binding"]
            == package["bindings"]["phase_binding"],
            "authority lost package phase-safe binding",
        )
        return result

    runner.grant_authority = authority_0018

    def wait_for_process_started() -> dict[str, Any]:
        started_path = runner.STATE / "process_started.json"
        deadline = time.monotonic() + 5.0
        while not started_path.is_file():
            runner.require(
                time.monotonic() < deadline,
                "supervisor process-start phase handoff timed out",
            )
            time.sleep(0.01)
        return runner.read_json(started_path)

    def child_entry_attestation_0018(
        prompts_path: Path,
        expected_freeze_sha256: str,
    ) -> dict[str, Any]:
        certificate = verify_closure_certificate()
        verify_phase_transition_regression()
        frozen_binding = verify_frozen_phase_binding()
        package_measurement = runner.supervisor.measure_file(runner.PACKAGE)
        authority_measurement = runner.supervisor.measure_file(runner.AUTHORITY)
        package = runner.read_json(runner.PACKAGE)
        authority = runner.read_json(runner.AUTHORITY)
        phase_binding = phase_binding_manifest()
        intent = runner.read_json(runner.STATE / "pre_start_intent.json")
        started = wait_for_process_started()
        expected_package_provenance = {
            "path": str(runner.PACKAGE),
            "byte_count": package_measurement.byte_count,
            "sha256": package_measurement.sha256,
            "package_id": package["package_id"],
        }
        expected_authority_provenance = {
            "path": str(runner.AUTHORITY),
            "byte_count": authority_measurement.byte_count,
            "sha256": authority_measurement.sha256,
            "authority_id": authority["authority_id"],
        }
        expected_argv = runner.execution_argv(
            prompts_path,
            expected_freeze_sha256,
        )
        actual_process_argv = _actual_process_argv(PRE_IMPORT_ARGV)
        actual_process_argv_sha256 = runner.supervisor.canonical_value_sha256(
            actual_process_argv
        )
        expected_modules = sorted(PRE_IMPORT_MODULES)
        runner.require(
            runner.sha256_file(runner.FREEZE) == expected_freeze_sha256,
            "actual freeze differs from child argv binding",
        )
        runner.require(
            package["bindings"]["phase_binding"] == phase_binding
            and authority["bindings"]["phase_binding"] == phase_binding
            and frozen_binding["binding_core_sha256"]
            == certificate["binding_core_sha256"],
            "certificate bytes/hash or binding core lost package/authority binding",
        )
        runner.require(
            intent.get("schema") == runner.supervisor.INTENT_SCHEMA
            and intent.get("process_start_count") == 0
            and intent.get("bindings") == package["bindings"]
            and intent.get("runtime") == authority["runtime"]
            and intent.get("provenance", {}).get("package")
            == expected_package_provenance
            and intent.get("provenance", {}).get("authority")
            == expected_authority_provenance,
            "durable pre-start intent or reservation changed",
        )
        runner.require(
            started.get("schema") == runner.supervisor.STARTED_SCHEMA
            and started.get("process_start_count") == 1
            and started.get("pid") == os.getpid()
            and started.get("provenance") == intent["provenance"]
            and sum(
                path.name == "process_started.json"
                for path in runner.STATE.iterdir()
            )
            == 1,
            "sole process_started=1 handoff changed",
        )
        runner.require(
            Path(PRE_IMPORT_ARGV[0]).resolve() == LAUNCHER,
            "pre-import argv[0] does not resolve to the exact 0018 launcher",
        )
        runner.require(
            actual_process_argv == expected_argv
            and package["bindings"]["argv"] == actual_process_argv
            and authority["bindings"]["argv"] == actual_process_argv
            and intent["bindings"]["argv"] == actual_process_argv
            and package["bindings"]["argv_sha256"]
            == actual_process_argv_sha256
            and authority["bindings"]["argv_sha256"]
            == actual_process_argv_sha256
            and intent["bindings"]["argv_sha256"]
            == actual_process_argv_sha256
            and runner.supervisor.canonical_value_sha256(expected_argv)
            == actual_process_argv_sha256
            and len(PRE_IMPORT_ARGV) == 6
            and len(actual_process_argv) == 7,
            "reconstructed process argv differs from package/authority/intent snapshot",
        )
        runner.require(
            started["provenance"] == intent["provenance"]
            and started["provenance"]["package"] == expected_package_provenance
            and started["provenance"]["authority"] == expected_authority_provenance,
            "process-started provenance does not bind the attested child argv",
        )
        runner.require(
            package["bindings"]["argv"] == expected_argv
            and authority["bindings"]["argv"] == expected_argv
            and intent["bindings"]["argv"] == expected_argv,
            "expected execution argv differs across provenance-bound records",
        )
        runner.require(
            PRE_IMPORT_ENVIRONMENT
            == certificate["binding_core"]["execution_envelope"]["environment"]
            == package["bindings"]["environment"]
            == authority["bindings"]["environment"]
            and PRE_IMPORT_CWD
            == certificate["binding_core"]["execution_envelope"]["cwd"]
            == package["bindings"]["cwd"]
            and expected_modules
            == certificate["binding_core"]["execution_envelope"][
                "pre_import_modules"
            ],
            "actual child environment/cwd/module snapshot changed",
        )
        for module in predecessor_modules:
            runner.require(
                module.PRE_IMPORT_ARGV == PRE_IMPORT_ARGV
                and module.PRE_IMPORT_ENVIRONMENT == PRE_IMPORT_ENVIRONMENT
                and module.PRE_IMPORT_CWD == PRE_IMPORT_CWD
                and sorted(module.PRE_IMPORT_MODULES) == expected_modules,
                f"effective module snapshot changed: {module.__name__}",
            )
        runner.require(
            authority["package"] == expected_package_provenance
            and authority["runtime"]["state_dir"] == str(runner.STATE)
            and authority["runtime"]["terminal_record_path"] == str(runner.TERMINAL),
            "current package/authority hash or reservation binding changed",
        )
        attestation = {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_PHASE_SAFE_RECONSTRUCTED_ARGV_CHILD_ENTRY_ATTESTATION",
            "actual_process_argv": actual_process_argv,
            "actual_process_argv_sha256": actual_process_argv_sha256,
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": certificate["binding_core_sha256"],
            "historical_preauthority_observation_sha256": certificate[
                "preauthority_observation_sha256"
            ],
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
            "package": runner.file_record(runner.PACKAGE),
            "authority": runner.file_record(runner.AUTHORITY),
            "pre_start_intent": runner.file_record(
                runner.STATE / "pre_start_intent.json"
            ),
            "process_started": runner.file_record(
                runner.STATE / "process_started.json"
            ),
            "checks": {
                "original_certificate_bytes_and_hash_verified": True,
                "historical_observation_verified_without_phase_recomputation": True,
                "binding_core_recomputed": True,
                "current_package_and_authority_hashes_verified": True,
                "pre_start_intent_verified": True,
                "sole_process_started_equals_one": True,
                "pre_import_launcher_resolution_verified_separately": True,
                "reconstructed_process_argv_and_canonical_sha_verified": True,
                "process_started_provenance_binds_child_argv": True,
                "actual_environment_cwd_modules_verified": True,
                "reservation_verified": True,
            },
        }
        runner.write_json(runner.STATE / "child_entry_attestation.json", attestation)
        return attestation

    runner.child_entry_attestation = child_entry_attestation_0018

    def execute_0018(prompts_path: Path, expected_freeze_sha256: str) -> int:
        child_entry_attestation_0018(prompts_path, expected_freeze_sha256)
        return prior_execute(prompts_path, expected_freeze_sha256)

    runner.execute = execute_0018

    def verify_0018() -> int:
        result = prior_verify()
        certificate = verify_closure_certificate()
        verify_phase_transition_regression()
        verify_argv_shape_regression()
        verify_frozen_phase_binding()
        attestation = runner.read_json(
            runner.STATE / "child_entry_attestation.json"
        )
        terminal = runner.read_json(runner.TERMINAL)
        package = runner.read_json(runner.PACKAGE)
        authority = runner.read_json(runner.AUTHORITY)
        intent = runner.read_json(runner.STATE / "pre_start_intent.json")
        started = runner.read_json(runner.STATE / "process_started.json")
        execution_result = runner.read_json(runner.RESULT)
        attested_argv = attestation["actual_process_argv"]
        attested_argv_sha256 = runner.supervisor.canonical_value_sha256(
            attested_argv
        )
        runner.require(
            attestation.get("status")
            == "PASS_PHASE_SAFE_RECONSTRUCTED_ARGV_CHILD_ENTRY_ATTESTATION"
            and attestation.get("binding_core_sha256")
            == certificate["binding_core_sha256"]
            and all(attestation.get("checks", {}).values())
            and attestation["actual_process_argv_sha256"] == attested_argv_sha256
            and terminal["process"]["argv"] == attested_argv
            and terminal["process"]["argv_sha256"] == attested_argv_sha256
            and package["bindings"]["argv"] == attested_argv
            and authority["bindings"]["argv"] == attested_argv
            and intent["bindings"]["argv"] == attested_argv
            and package["bindings"]["argv_sha256"] == attested_argv_sha256
            and authority["bindings"]["argv_sha256"] == attested_argv_sha256
            and intent["bindings"]["argv_sha256"] == attested_argv_sha256
            and started["provenance"] == intent["provenance"]
            and terminal["provenance"] == intent["provenance"]
            and terminal.get("outcome") == "success"
            and terminal.get("process_start_count") == 1
            and execution_result["aggregate"]["prompt_count"] == 2
            and execution_result["aggregate"]["generated_tokens_per_prompt"] == 4
            and execution_result["aggregate"]["corrected_v_bytes_compared"]
            == 2 * 4 * 128
            and execution_result["aggregate"][
                "rtl_emitted_corrected_v_consumed_downstream_all_steps"
            ]
            is True
            and execution_result["aggregate"]["downstream_logits_all_steps_equal"]
            is True
            and execution_result["aggregate"]["selected_tokens_all_steps_equal"]
            is True
            and execution_result["aggregate"]["four_token_sequences_equal"] is True
            and execution_result["aggregate"][
                "disabled_sidecar_control_all_steps"
            ]
            is True,
            "0018 terminal argv, phase-safe, or RTL-in-loop evidence changed",
        )
        print(
            "ACE2_HYBRID_0018_DECISIVE_VERIFY_PASS "
            "prompt_count=2 generated_tokens_per_prompt=4 rtl_v_bytes=1024 "
            "process_start_count=1 reconstructed_process_argv=true "
            "phase_safe_binding=true",
            flush=True,
        )
        return result

    runner.verify = verify_0018


def main() -> int:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import run_stage1_layer23_v_rank1_hybrid_0010 as prior_0010
        from tools import run_stage1_layer23_v_rank1_hybrid_0011 as prior_0011
        from tools import run_stage1_layer23_v_rank1_hybrid_0012 as prior_0012
        from tools import run_stage1_layer23_v_rank1_hybrid_0013 as prior_0013
        from tools import run_stage1_layer23_v_rank1_hybrid_0014 as prior_0014
        from tools import run_stage1_layer23_v_rank1_hybrid_0015 as prior
        from tools import run_stage1_layer23_v_rank1_hybrid_0016 as prior_0016
        from tools import run_stage1_layer23_v_rank1_hybrid_0017 as prior_0017

        predecessor_modules = (
            prior_0010,
            prior_0011,
            prior_0012,
            prior_0013,
            prior_0014,
            prior,
            prior_0016,
            prior_0017,
        )
        prior_0016._bind_predecessor_snapshots(
            predecessor_modules,
            argv=PRE_IMPORT_ARGV,
            environment=PRE_IMPORT_ENVIRONMENT,
            cwd=PRE_IMPORT_CWD,
            imported_modules=PRE_IMPORT_MODULES,
        )
        _patch_inherited_chain(prior, prior_0014)
        prior_0014._configure_prior(prior_0011)
        prior_0011._configure_prior(prior_0010)
        prior_0010._verify_pre_import_launch()

        from tools import ace2_layer23_v_input_scale_oracle
        from tools import ace2_layer23_v_rank1_integer_correction_reference
        from tools import run_stage1_layer23_v_rank1_hybrid as runner
        from tools import run_stage1_layer23_v_rank1_hybrid_0008 as base

        prior._configure_runner(
            runner,
            prior_0014,
            prior_0011,
            prior_0010,
            base,
            ace2_layer23_v_rank1_integer_correction_reference,
            ace2_layer23_v_input_scale_oracle,
            dict(os.environ),
        )
        _configure_runner(runner, prior, predecessor_modules)
        if "--select-prompts" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--select-prompts", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.select_prompts()
        if "--check-record-lifecycle" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-record-lifecycle", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.record_lifecycle_regression()
        if "--check-gate-order" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-gate-order", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.gate_order_regression()
        if "--check-input-scale" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-input-scale", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.input_scale_regression()
        if "--check-argv-shape" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-argv-shape", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.argv_shape_regression()
        if "--check-phase-binding" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-phase-binding", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.phase_binding_regression()
        if "--tokenization-preflight" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--tokenization-preflight", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.tokenization_preflight(arguments.prompts.resolve())
        if "--check-tokenization-verifier" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-tokenization-verifier", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.tokenization_verifier_regression(
                arguments.prompts.resolve()
            )
        if "--check-scale-alignment" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-scale-alignment", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.scale_regression(arguments.prompts.resolve())
        return runner.main()
    except BaseException as error:
        _seal_preloader_failure(error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
