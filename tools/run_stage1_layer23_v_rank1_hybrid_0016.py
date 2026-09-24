#!/usr/bin/env python3
"""Recursively rebound fixed-input-scale launcher for nonofficial attempt 0016."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0016"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0016.json"
SELECTION = PRIVATE / "candidate-selection-0016.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0016.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0016.json"
INPUT_SCALE_REGRESSION = PRIVATE / "fixed-input-scale-regression-0016.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0016.json"
LIFECYCLE_REGRESSION = PRIVATE / "record-lifecycle-regression-0016.json"
GATE_ORDER_REGRESSION = PRIVATE / "gate-order-regression-0016.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0016.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0016.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0016.json"
CLOSURE_REGRESSION = PRIVATE / "closure-binding-regression-0016.json"

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

CANDIDATES = (
    {"id": "fresh-0016-a", "text": "How do dunes move?"},
    {"id": "fresh-0016-b", "text": "Why does linen crease?"},
    {"id": "fresh-0016-c", "text": "Where do beetles hide?"},
    {"id": "fresh-0016-d", "text": "How do icicles grow?"},
    {"id": "fresh-0016-e", "text": "Why does copper tarnish?"},
    {"id": "fresh-0016-f", "text": "What makes cork float?"},
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


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
        "mission_id": "stage1rank1hybrid16",
        "attempt_identity": "nonofficial-hybrid-0016",
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
            "regression": "No retry, replay, or resume; preserve all 0016 artifacts.",
        },
    }
    try:
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _bind_predecessor_snapshots(
    modules: tuple[Any, ...],
    *,
    argv: list[str],
    environment: dict[str, str],
    cwd: str,
    imported_modules: frozenset[str],
) -> None:
    values = {
        "PRE_IMPORT_ARGV": list(argv),
        "PRE_IMPORT_ENVIRONMENT": dict(environment),
        "PRE_IMPORT_CWD": cwd,
        "PRE_IMPORT_MODULES": frozenset(imported_modules),
        "PRE_IMPORT_SYS_PATH": list(PRE_IMPORT_SYS_PATH),
        "PRE_IMPORT_SYS_PREFIX": PRE_IMPORT_SYS_PREFIX,
    }
    for module in modules:
        for name, value in values.items():
            if hasattr(module, name):
                setattr(module, name, value)


def _configure_0015(prior: Any, prior_0014: Any) -> None:
    prior.LAUNCHER = LAUNCHER
    prior.OUTPUT = OUTPUT
    prior.TERMINAL = TERMINAL
    prior.PRIVATE = PRIVATE
    prior.PROMPTS = PROMPTS
    prior.SELECTION = SELECTION
    prior.TOKENIZATION_PREFLIGHT = TOKENIZATION_PREFLIGHT
    prior.TOKENIZATION_REGRESSION = TOKENIZATION_REGRESSION
    prior.INPUT_SCALE_REGRESSION = INPUT_SCALE_REGRESSION
    prior.SCALE_REGRESSION = SCALE_REGRESSION
    prior.LIFECYCLE_REGRESSION = LIFECYCLE_REGRESSION
    prior.GATE_ORDER_REGRESSION = GATE_ORDER_REGRESSION
    prior.TEMPLATE_GATE = TEMPLATE_GATE
    prior.DEPENDENCY_PROBE = DEPENDENCY_PROBE
    prior.LAUNCH_REGRESSION = LAUNCH_REGRESSION
    prior.CANDIDATES = CANDIDATES
    prior._configure_0014(prior_0014)
    prior_0014.DIAGNOSTIC_PROMPT_SOURCES = (
        *prior_0014.DIAGNOSTIC_PROMPT_SOURCES,
        PRIVATE / "prompts-0015.json",
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
    }


def _configure_runner(
    runner: Any,
    prior: Any,
    prior_0014: Any,
    prior_0011: Any,
    prior_0010: Any,
    base: Any,
    rank1_reference: Any,
    input_scale_oracle: Any,
    predecessor_modules: tuple[Any, ...],
    post_import_environment: dict[str, str],
) -> None:
    prior._configure_runner(
        runner,
        prior_0014,
        prior_0011,
        prior_0010,
        base,
        rank1_reference,
        input_scale_oracle,
        post_import_environment,
    )
    runner.MISSION_ID = "stage1rank1hybrid16"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0016"
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
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_freeze = runner.freeze
    prior_authority = runner.grant_authority
    prior_child_entry_attestation = runner.child_entry_attestation
    prior_execute = runner.execute
    prior_verify = runner.verify

    terminal_0015_path = (
        ROOT
        / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
        "nonofficial-hybrid-0015/terminal-record.json"
    )
    durable_0015_path = terminal_0015_path.parent / "supervisor-state/terminal_record.json"
    launcher_0015_path = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0015.py"

    def verify_0015_seal() -> dict[str, Any]:
        for path in (terminal_0015_path, durable_0015_path, launcher_0015_path):
            runner.require(
                stat.S_ISREG(os.lstat(path).st_mode),
                f"sealed 0015 artifact is not regular: {runner.public_path(path)}",
            )
        runner.require(
            runner.sha256_file(terminal_0015_path) == SEALED_0015_TERMINAL_SHA256
            and runner.sha256_file(durable_0015_path)
            == SEALED_0015_TERMINAL_SHA256
            and runner.sha256_file(launcher_0015_path) == SEALED_0015_LAUNCHER_SHA256,
            "sealed 0015 terminal or launcher changed",
        )
        terminal = runner.read_json(terminal_0015_path)
        runner.require(
            terminal == runner.read_json(durable_0015_path)
            and terminal.get("process_start_count") == 1
            and terminal.get("process_started") is True
            and terminal.get("outcome") == "child_nonzero_exit",
            "sealed 0015 exactly-once terminal state changed",
        )
        return runner.file_record(terminal_0015_path)

    def verify_predecessor_seals_0016() -> None:
        prior_verify_predecessor_seals()
        verify_0015_seal()

    runner.verify_predecessor_seals = verify_predecessor_seals_0016

    def freeze_0016(prompts_path: Path) -> int:
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["sealed_0015_terminal"] = verify_0015_seal()
        frozen["recursive_predecessor_snapshot_binding"] = {
            "modules": [module.__name__ for module in predecessor_modules],
            "bound_before_inherited_runner_closures": True,
            "required_snapshots": [
                "PRE_IMPORT_ARGV",
                "PRE_IMPORT_ENVIRONMENT",
                "PRE_IMPORT_CWD",
                "PRE_IMPORT_MODULES",
            ],
            "child_entry_argv_attestation_retained": True,
        }
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0016

    def expected_child_argv(prompts_path: Path) -> tuple[list[str], str]:
        argv = runner.execution_argv(
            prompts_path,
            runner.sha256_file(runner.FREEZE),
        )
        return argv, runner.supervisor.canonical_value_sha256(argv)

    def bind_expected_child_snapshot(prompts_path: Path) -> tuple[list[str], str]:
        argv, argv_sha256 = expected_child_argv(prompts_path)
        runner.require(
            PRE_IMPORT_ENVIRONMENT == runner.child_environment()
            and PRE_IMPORT_CWD == str(ROOT),
            "nonconsuming gate snapshot differs from the authorized child environment/cwd",
        )
        _bind_predecessor_snapshots(
            predecessor_modules,
            argv=argv,
            environment=PRE_IMPORT_ENVIRONMENT,
            cwd=PRE_IMPORT_CWD,
            imported_modules=PRE_IMPORT_MODULES,
        )
        return argv, argv_sha256

    def module_binding_records() -> list[dict[str, Any]]:
        records = []
        expected_modules = sorted(PRE_IMPORT_MODULES)
        environment_sha256 = runner.supervisor.canonical_value_sha256(
            PRE_IMPORT_ENVIRONMENT
        )
        modules_sha256 = runner.supervisor.canonical_value_sha256(expected_modules)
        for module in predecessor_modules:
            argv = list(module.PRE_IMPORT_ARGV)
            records.append(
                {
                    "module": module.__name__,
                    "source": runner.file_record(Path(module.__file__).resolve(strict=True)),
                    "effective_pre_import_argv": argv,
                    "effective_pre_import_argv_list_sha256": (
                        runner.supervisor.canonical_value_sha256(argv)
                    ),
                    "effective_pre_import_environment_sha256": (
                        runner.supervisor.canonical_value_sha256(
                            module.PRE_IMPORT_ENVIRONMENT
                        )
                    ),
                    "effective_pre_import_cwd": module.PRE_IMPORT_CWD,
                    "effective_pre_import_modules_list_sha256": (
                        runner.supervisor.canonical_value_sha256(
                            sorted(module.PRE_IMPORT_MODULES)
                        )
                    ),
                    "checks": {
                        "environment_equals_outermost_snapshot": (
                            module.PRE_IMPORT_ENVIRONMENT == PRE_IMPORT_ENVIRONMENT
                            and runner.supervisor.canonical_value_sha256(
                                module.PRE_IMPORT_ENVIRONMENT
                            )
                            == environment_sha256
                        ),
                        "cwd_equals_outermost_snapshot": (
                            module.PRE_IMPORT_CWD == PRE_IMPORT_CWD
                        ),
                        "modules_equal_outermost_snapshot": (
                            sorted(module.PRE_IMPORT_MODULES) == expected_modules
                            and runner.supervisor.canonical_value_sha256(
                                sorted(module.PRE_IMPORT_MODULES)
                            )
                            == modules_sha256
                        ),
                    },
                }
            )
        return records

    def authority_template_bindings(prompts_path: Path) -> dict[str, Any]:
        package = runner.read_json(runner.PACKAGE)
        _snapshot, model, tokenizer_json, tokenizer_config = (
            runner.resolve_runtime_inputs()
        )
        return runner.package_bindings(
            prompts_path,
            runner.sha256_file(runner.FREEZE),
            package["bindings"]["required_absent_outputs"],
            model,
            tokenizer_json,
            tokenizer_config,
            include_frozen_stage_files=True,
        )

    def closure_binding_regression(prompts_path: Path) -> int:
        runner.verify_exact_interpreter()
        runner.verify_dependency_probe()
        runner.verify_predecessor_seals()
        runner.validate_no_execution()
        runner.require(
            runner.PACKAGE.is_file()
            and not runner.AUTHORITY.exists()
            and not LAUNCH_REGRESSION.exists()
            and not CLOSURE_REGRESSION.exists(),
            "closure-binding regression requires package and must precede authority",
        )
        argv, argv_sha256 = bind_expected_child_snapshot(prompts_path)
        modules = module_binding_records()
        package = runner.read_json(runner.PACKAGE)
        authority_bindings = authority_template_bindings(prompts_path)
        runner.require(
            len(modules) == 6
            and [record["module"].rsplit("_", 1)[-1] for record in modules]
            == ["0010", "0011", "0012", "0013", "0014", "0015"]
            and all(
                record["effective_pre_import_argv"] == argv
                and record["effective_pre_import_argv_list_sha256"] == argv_sha256
                and all(record["checks"].values())
                for record in modules
            )
            and package["bindings"]["argv"] == argv
            and package["bindings"]["argv_sha256"] == argv_sha256
            and authority_bindings["argv"] == argv
            and authority_bindings["argv_sha256"] == argv_sha256,
            "pre-authority recursive closure argv binding differs from package templates",
        )
        runner.write_json(
            CLOSURE_REGRESSION,
            {
                "schema_version": 1,
                "mission_id": runner.MISSION_ID,
                "attempt_identity": runner.ATTEMPT_ID,
                "status": "PASS_NONCONSUMING_RECURSIVE_CLOSURE_BINDING",
                "created_at_utc": runner.utc_now(),
                "model_executed": False,
                "simulator_executed": False,
                "process_start_count": 0,
                "predecessor_modules": modules,
                "launch_certificate_template": {
                    "child_argv": argv,
                    "child_argv_list_sha256": argv_sha256,
                },
                "package": {
                    "artifact": runner.file_record(runner.PACKAGE),
                    "argv": package["bindings"]["argv"],
                    "argv_list_sha256": package["bindings"]["argv_sha256"],
                },
                "authority_template": {
                    "argv": authority_bindings["argv"],
                    "argv_list_sha256": authority_bindings["argv_sha256"],
                },
                "checks": {
                    "modules_0010_through_0015_enumerated": True,
                    "all_module_argv_lists_equal_launch_certificate": True,
                    "all_module_argv_list_sha256_equal_launch_certificate": True,
                    "package_argv_list_and_sha256_equal": True,
                    "authority_template_argv_list_and_sha256_equal": True,
                    "outermost_environment_cwd_modules_snapshots_equal": True,
                    "authority_absent": True,
                    "process_not_started": True,
                },
            },
        )
        verify_closure_binding_regression(prompts_path, require_existing=True)
        print(
            "ACE2_HYBRID_0016_CLOSURE_BINDING_REGRESSION_PASS "
            f"sha256={runner.sha256_file(CLOSURE_REGRESSION)} "
            f"argv_sha256={argv_sha256} process_start_count=0",
            flush=True,
        )
        return 0

    def verify_closure_binding_regression(
        prompts_path: Path,
        *,
        require_existing: bool,
    ) -> dict[str, Any]:
        runner.require(
            CLOSURE_REGRESSION.is_file(),
            "0016 closure-binding regression is absent",
        )
        if require_existing:
            argv, argv_sha256 = expected_child_argv(prompts_path)
        else:
            argv, argv_sha256 = bind_expected_child_snapshot(prompts_path)
        records = module_binding_records()
        package = runner.read_json(runner.PACKAGE)
        authority_bindings = authority_template_bindings(prompts_path)
        record = runner.read_json(CLOSURE_REGRESSION)
        runner.require(
            record.get("status") == "PASS_NONCONSUMING_RECURSIVE_CLOSURE_BINDING"
            and record.get("model_executed") is False
            and record.get("simulator_executed") is False
            and record.get("process_start_count") == 0
            and record.get("predecessor_modules") == records
            and record.get("launch_certificate_template")
            == {
                "child_argv": argv,
                "child_argv_list_sha256": argv_sha256,
            }
            and record.get("package")
            == {
                "artifact": runner.file_record(runner.PACKAGE),
                "argv": package["bindings"]["argv"],
                "argv_list_sha256": package["bindings"]["argv_sha256"],
            }
            and record.get("authority_template")
            == {
                "argv": authority_bindings["argv"],
                "argv_list_sha256": authority_bindings["argv_sha256"],
            }
            and all(record.get("checks", {}).values())
            and all(
                item["effective_pre_import_argv"] == argv
                and item["effective_pre_import_argv_list_sha256"] == argv_sha256
                and all(item["checks"].values())
                for item in records
            ),
            "0016 recursive closure-binding regression artifact changed",
        )
        return record

    runner.closure_binding_regression = closure_binding_regression
    runner.verify_closure_binding_regression = verify_closure_binding_regression

    def authority_0016(prompts_path: Path) -> int:
        closure = verify_closure_binding_regression(
            prompts_path,
            require_existing=False,
        )
        result = prior_authority(prompts_path)
        launch = runner.read_json(LAUNCH_REGRESSION)
        package = runner.read_json(runner.PACKAGE)
        authority = runner.read_json(runner.AUTHORITY)
        expected_argv = closure["launch_certificate_template"]["child_argv"]
        expected_sha256 = closure["launch_certificate_template"][
            "child_argv_list_sha256"
        ]
        runner.require(
            launch["child_argv"] == expected_argv
            and launch["child_argv_sha256"] == expected_sha256
            and package["bindings"]["argv"] == expected_argv
            and package["bindings"]["argv_sha256"] == expected_sha256
            and authority["bindings"]["argv"] == expected_argv
            and authority["bindings"]["argv_sha256"] == expected_sha256,
            "materialized launch/package/authority argv differs from closure regression",
        )
        launch["closure_binding_regression"] = runner.file_record(
            CLOSURE_REGRESSION
        )
        launch["predecessor_module_argv_list_sha256"] = {
            item["module"]: item["effective_pre_import_argv_list_sha256"]
            for item in closure["predecessor_modules"]
        }
        runner.write_json(LAUNCH_REGRESSION, launch)
        return result

    runner.grant_authority = authority_0016

    def child_entry_attestation_0016(
        prompts_path: Path,
        expected_freeze_sha256: str,
    ) -> dict[str, Any]:
        closure = verify_closure_binding_regression(
            prompts_path,
            require_existing=True,
        )
        attestation = prior_child_entry_attestation(
            prompts_path,
            expected_freeze_sha256,
        )
        launch = runner.read_json(LAUNCH_REGRESSION)
        runner.require(
            launch.get("closure_binding_regression")
            == runner.file_record(CLOSURE_REGRESSION),
            "launch certificate lost closure-binding regression",
        )
        attestation["closure_binding_regression_sha256"] = runner.sha256_file(
            CLOSURE_REGRESSION
        )
        attestation["predecessor_module_argv_list_sha256"] = {
            item["module"]: item["effective_pre_import_argv_list_sha256"]
            for item in closure["predecessor_modules"]
        }
        return attestation

    runner.child_entry_attestation = child_entry_attestation_0016

    def execute_0016(prompts_path: Path, expected_freeze_sha256: str) -> int:
        child_entry_attestation_0016(prompts_path, expected_freeze_sha256)
        return prior_execute(prompts_path, expected_freeze_sha256)

    runner.execute = execute_0016

    def verify_0016() -> int:
        result = prior_verify()
        closure = verify_closure_binding_regression(
            PROMPTS,
            require_existing=False,
        )
        launch = runner.read_json(LAUNCH_REGRESSION)
        runner.require(
            launch.get("closure_binding_regression")
            == runner.file_record(CLOSURE_REGRESSION)
            and len(closure["predecessor_modules"]) == 6,
            "0016 verification lost recursive closure-binding evidence",
        )
        print(
            "ACE2_HYBRID_0016_DECISIVE_VERIFY_PASS "
            "prompt_count=2 generated_tokens_per_prompt=4 fixed_input_scale_steps=16 "
            "predecessor_module_count=6",
            flush=True,
        )
        return result

    runner.verify = verify_0016


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

        predecessor_modules = (
            prior_0010,
            prior_0011,
            prior_0012,
            prior_0013,
            prior_0014,
            prior,
        )
        _bind_predecessor_snapshots(
            predecessor_modules,
            argv=PRE_IMPORT_ARGV,
            environment=PRE_IMPORT_ENVIRONMENT,
            cwd=PRE_IMPORT_CWD,
            imported_modules=PRE_IMPORT_MODULES,
        )
        _configure_0015(prior, prior_0014)
        prior_0014._configure_prior(prior_0011)
        prior_0011._configure_prior(prior_0010)
        prior_0010._verify_pre_import_launch()

        from tools import ace2_layer23_v_input_scale_oracle
        from tools import ace2_layer23_v_rank1_integer_correction_reference
        from tools import run_stage1_layer23_v_rank1_hybrid as runner
        from tools import run_stage1_layer23_v_rank1_hybrid_0008 as base

        _configure_runner(
            runner,
            prior,
            prior_0014,
            prior_0011,
            prior_0010,
            base,
            ace2_layer23_v_rank1_integer_correction_reference,
            ace2_layer23_v_input_scale_oracle,
            predecessor_modules,
            dict(os.environ),
        )
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
        if "--check-closure-binding" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-closure-binding", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.closure_binding_regression(arguments.prompts.resolve())
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
