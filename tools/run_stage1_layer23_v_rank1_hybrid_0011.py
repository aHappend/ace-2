#!/usr/bin/env python3
"""Pristine-environment launcher for nonofficial hybrid attempt 0011."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
PRIOR_LAUNCHER = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0010.py"
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0011"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0011.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0011.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0011.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0011.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0011.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0011.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0011.json"
SUPERVISOR = ROOT / "tools/ace2_exact_once_runtime_supervisor.py"

PRE_IMPORT_ENVIRONMENT = dict(os.environ)
PRE_IMPORT_SYS_PATH = list(sys.path)
PRE_IMPORT_SYS_PREFIX = sys.prefix
PRE_IMPORT_ARGV = list(sys.argv)
PRE_IMPORT_CWD = str(Path.cwd().resolve())
PRE_IMPORT_MODULES = frozenset(sys.modules)

PREDECESSOR_0010_SHA256 = {
    ".argus_subagents/stage1rank1hybrid07-nonofficial-hybrid-0010-execute.json": (
        "1a6801a68872af55c43deb869d19bb4b31918c63b8eaf9a0bb51a801b6bcbc32"
    ),
    ".argus_subagents/stage1rank1hybrid07-nonofficial-hybrid-0010-execute_logs/"
    "exit_code.stage1rank1hybrid07-nonofficial-hybrid-0010-execute-"
    "1787242535869662781": (
        "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865"
    ),
    ".argus_subagents/stage1rank1hybrid07-nonofficial-hybrid-0010-execute_logs/"
    "stderr.log": "be4449b47d45cd6f19a47638bf9ea4128ed65714ebbb1190e91a2103ff3920af",
    ".argus_subagents/stage1rank1hybrid07-nonofficial-hybrid-0010-execute_logs/"
    "stdout.log": "f0271d0e988419590e6d44ba2fbc943f3d51eb6e774eb2523a0e65f4a5d93aa2",
    "build/stage1-layer23-v-rank1-hybrid-v1/private/dependency-probe-0010.json": (
        "341956ce6a0380a3fe907b57131aad777b0039024c4dd2d3310e5f1524989671"
    ),
    "build/stage1-layer23-v-rank1-hybrid-v1/private/"
    "exact-tokenization-preflight-0010.json": (
        "62c3768584a15b588c93c1c460ab18ff18176694bc4ae12999379cabfdff070f"
    ),
    "build/stage1-layer23-v-rank1-hybrid-v1/private/"
    "package-template-invariant-0010.json": (
        "f9dc078b613ee85769bafb2750d0f735a09536c5f8bb5fa3b333417fdd983f83"
    ),
    "build/stage1-layer23-v-rank1-hybrid-v1/private/prompts-0010.json": (
        "aa7a88d40160711209244faeddbf86f7908687b8ebf2cea922f01f6665c7a205"
    ),
    "build/stage1-layer23-v-rank1-hybrid-v1/private/"
    "retained-scale-regression-0010.json": (
        "f62e1d7172caf3f1166e7f25c804ed1e6b90ba40fe091a975e10367b7f646ced"
    ),
    "build/stage1-layer23-v-rank1-hybrid-v1/private/"
    "tokenization-verifier-regression-0010.json": (
        "5e5e10d8c6e23b4701fac91e72a2fed30ef74d658e467b1d98a280d2d2fbbc7a"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0010/execution-authority.json": (
        "c373993d996f922705c43ec2ae3d345ceb085310edc539fab26f92af24578fe5"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0010/execution-package.json": (
        "4248df7725aa0e909d948bf0427729396ba305e9dd446523c23809a9f97ea352"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0010/freeze.json": (
        "ffe47cc53484cd478474d4820e62cf5224f72937d28fa9bac5ac75d292be1990"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0010/live/failure.json": (
        "4c96110c77b480a5eb70ba673dcce8b54e95196da0727d4c0a9d375efca9fa83"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0010/preflight.json": (
        "5adb55205e7d529a89dc8728371e080c2dedfceb20739f66c887a6935e7a608f"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0010/terminal-record.json": (
        "32de31d59ac78e9dac25eeb6ca06bde0c9035b6f33f0ce27be8b731074b8f65c"
    ),
    "tools/run_stage1_layer23_v_rank1_hybrid_0010.py": (
        "a47c8ca682eacb8a4bffe93fbf61cfb083c300bad359771010e85541fad5d440"
    ),
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal_preloader_failure(error: BaseException) -> None:
    if TERMINAL.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid11",
        "attempt_identity": "nonofficial-hybrid-0011",
        "status": "FAILED_SEALED_NO_EXECUTION",
        "outcome": "pre_execution_failure",
        "sealed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "process_start_count": 0,
        "process_started": False,
        "failure": {
            "failure_taxonomy": "pre_execution_gate_failure",
            "phase": "stdlib_preloader_or_nonconsuming_gate",
            "root_cause_hypothesis": f"{type(error).__name__}: {error}",
            "regression": "No retry, replay, or resume; preserve the failed gate artifacts.",
        },
    }
    try:
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _configure_prior(prior: Any) -> None:
    prior.LAUNCHER = LAUNCHER
    prior.SOURCE_PRIOR_LAUNCHER = PRIOR_LAUNCHER
    prior.OUTPUT = OUTPUT
    prior.TERMINAL = TERMINAL
    prior.PRIVATE = PRIVATE
    prior.TOKENIZATION_PREFLIGHT = TOKENIZATION_PREFLIGHT
    prior.TOKENIZATION_REGRESSION = TOKENIZATION_REGRESSION
    prior.SCALE_REGRESSION = SCALE_REGRESSION
    prior.PRE_IMPORT_ENVIRONMENT = PRE_IMPORT_ENVIRONMENT
    prior.PRE_IMPORT_SYS_PATH = PRE_IMPORT_SYS_PATH
    prior.PRE_IMPORT_SYS_PREFIX = PRE_IMPORT_SYS_PREFIX
    prior.PRE_IMPORT_ARGV = PRE_IMPORT_ARGV
    prior.PRE_IMPORT_CWD = PRE_IMPORT_CWD
    prior.PRE_IMPORT_MODULES = PRE_IMPORT_MODULES


def _configure_runner(
    runner: Any,
    prior: Any,
    base: Any,
    rank1_reference: Any,
    post_import_environment: dict[str, str],
) -> None:
    prior._bind_base_module(base)
    prior._configure_runner(
        runner,
        base,
        rank1_reference,
        post_import_environment,
    )
    runner.MISSION_ID = "stage1rank1hybrid11"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0011"
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
        "nonofficial-hybrid-0010": {
            "sha256": PREDECESSOR_0010_SHA256[
                "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
                "nonofficial-hybrid-0010/terminal-record.json"
            ],
            "process_start_count": 0,
            "process_started": False,
        },
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_freeze = runner.freeze
    prior_child_entry_attestation = runner.child_entry_attestation
    prior_verify = runner.verify

    def predecessor_0010_records() -> dict[str, Any]:
        records: dict[str, Any] = {}
        for relative, expected_sha256 in sorted(PREDECESSOR_0010_SHA256.items()):
            path = ROOT / relative
            path_stat = os.lstat(path)
            runner.require(stat.S_ISREG(path_stat.st_mode), f"0010 evidence is not regular: {relative}")
            runner.require(
                runner.sha256_file(path) == expected_sha256,
                f"sealed 0010 evidence hash changed: {relative}",
            )
            records[relative] = runner.file_record(path)
        terminal = runner.read_json(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0010/terminal-record.json"
        )
        runner.require(
            terminal.get("outcome") == "pre_execution_failure"
            and terminal.get("process_start_count") == 0
            and terminal.get("process_started") is False,
            "sealed 0010 zero-start terminal state changed",
        )
        return records

    def verify_predecessor_seals_0011() -> None:
        prior_verify_predecessor_seals()
        predecessor_0010_records()

    runner.verify_predecessor_seals = verify_predecessor_seals_0011

    def source_records_0011(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(model, tokenizer_json, tokenizer_config, prompts)
        for path in [SUPERVISOR, *(ROOT / relative for relative in PREDECESSOR_0010_SHA256)]:
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0011

    def execution_bound_paths_0011(
        prompts_path: Path,
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        *,
        include_frozen_stage_files: bool,
    ) -> list[Path]:
        paths = prior_execution_bound_paths(
            prompts_path,
            model,
            tokenizer_json,
            tokenizer_config,
            include_frozen_stage_files=include_frozen_stage_files,
        )
        return sorted(
            {
                *paths,
                SUPERVISOR.resolve(strict=True),
                *(ROOT / relative for relative in PREDECESSOR_0010_SHA256),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0011

    def freeze_0011(prompts_path: Path) -> int:
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["predecessor_0010_evidence"] = predecessor_0010_records()
        frozen["exact_once_runtime_supervisor"] = runner.file_record(SUPERVISOR)
        frozen["launch_fidelity_contract"] = {
            "child_argv_constructor": "execution_argv",
            "required_argument_order": [
                "python",
                "launcher",
                "--execute",
                "--prompts",
                "fresh_prompts",
                "--freeze-sha256",
                "freeze_sha256",
            ],
            "package_authority_exact_list_and_sha256": True,
            "authority_requires_pre_materialization_package_argv_reconstruction": True,
            "post_materialization_package_authority_recheck": True,
            "consuming_entrypoint": "ace2_exact_once_runtime_supervisor.py",
        }
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0011

    def reconstructed_child_argv(prompts_path: Path) -> tuple[list[str], str]:
        argv = runner.execution_argv(prompts_path, runner.sha256_file(runner.FREEZE))
        runner.require(
            len(argv) == 7
            and argv[2] == "--execute"
            and argv[3] == "--prompts"
            and argv[4] == str(prompts_path.resolve(strict=True))
            and argv[5] == "--freeze-sha256"
            and argv[6] == runner.sha256_file(runner.FREEZE),
            "reconstructed child argv lost explicit prompts or freeze binding",
        )
        return argv, runner.supervisor.canonical_value_sha256(argv)

    def verify_launch_fidelity(prompts_path: Path) -> dict[str, Any]:
        runner.require(LAUNCH_REGRESSION.is_file(), "launch-fidelity regression is absent")
        record = runner.read_json(LAUNCH_REGRESSION)
        argv, argv_sha256 = reconstructed_child_argv(prompts_path)
        package = runner.read_json(runner.PACKAGE)
        authority = runner.read_json(runner.AUTHORITY)
        supervisor_command = [
            str(Path(sys.executable).resolve()),
            str(SUPERVISOR),
            "--package",
            str(runner.PACKAGE),
            "--authority",
            str(runner.AUTHORITY),
        ]
        expected_checks = {
            "process_not_started": True,
            "pre_start_intent_absent": True,
            "explicit_prompts_argument_present": True,
            "explicit_freeze_sha256_argument_present": True,
            "package_exact_list_equal": True,
            "package_exact_sha256_equal": True,
            "authority_exact_list_equal": True,
            "authority_exact_sha256_equal": True,
            "supervisor_path_exact": True,
        }
        runner.require(
            package["bindings"]["argv"] == argv
            and package["bindings"]["argv_sha256"] == argv_sha256,
            "package child argv differs from reconstruction",
        )
        runner.require(
            authority["bindings"]["argv"] == argv
            and authority["bindings"]["argv_sha256"] == argv_sha256,
            "authority child argv differs from reconstruction",
        )
        runner.require(
            record.get("child_argv") == argv
            and record.get("child_argv_sha256") == argv_sha256
            and record.get("supervisor_command") == supervisor_command
            and record.get("supervisor_command_sha256")
            == runner.supervisor.canonical_value_sha256(supervisor_command)
            and record.get("supervisor") == runner.file_record(SUPERVISOR)
            and record.get("package") == runner.file_record(runner.PACKAGE)
            and record.get("authority") == runner.file_record(runner.AUTHORITY)
            and record.get("checks") == expected_checks,
            "launch-fidelity regression record changed",
        )
        return record

    def grant_authority_0011(prompts_path: Path) -> int:
        runner.verify_exact_interpreter()
        runner.verify_dependency_probe()
        runner.require(runner.PACKAGE.is_file(), "execution package is absent")
        runner.require(not runner.AUTHORITY.exists(), "execution authority already exists")
        runner.require(not LAUNCH_REGRESSION.exists(), "launch-fidelity regression already exists")
        runner.verify_predecessor_seals()
        runner.validate_no_execution()
        frozen = runner.read_json(runner.FREEZE)
        runner.verify_frozen_sources(frozen, prompts_path)
        _snapshot, model, tokenizer_json, tokenizer_config = runner.resolve_runtime_inputs()
        required_outputs, runtime = runner.supervisor.prepare_runtime_bindings(
            source_root=ROOT / "rtl",
            state_dir=runner.STATE,
            stdout_path=runner.STDOUT,
            stderr_path=runner.STDERR,
            terminal_record_path=runner.TERMINAL,
            additional_outputs=(runner.LIVE,),
        )
        expected_bindings = runner.package_bindings(
            prompts_path,
            runner.sha256_file(runner.FREEZE),
            required_outputs,
            model,
            tokenizer_json,
            tokenizer_config,
            include_frozen_stage_files=True,
        )
        reconstructed_argv, reconstructed_sha256 = reconstructed_child_argv(prompts_path)
        package = runner.read_json(runner.PACKAGE)
        runner.require(package["bindings"] == expected_bindings, "packaged bindings changed before authority")
        runner.require(
            package["bindings"]["argv"] == reconstructed_argv
            and package["bindings"]["argv_sha256"] == reconstructed_sha256,
            "pre-authority package argv reconstruction failed",
        )
        package_measurement = runner.supervisor.measure_file(runner.PACKAGE)
        authority = {
            "schema": runner.supervisor.AUTHORITY_SCHEMA,
            "authority_id": (
                f"operator-{runner.MISSION_ID}-{runner.ATTEMPT_ID}-execute-once"
            ),
            "decision": "execute_once",
            "package": {
                "path": str(runner.PACKAGE),
                "byte_count": package_measurement.byte_count,
                "sha256": package_measurement.sha256,
                "package_id": package["package_id"],
            },
            "bindings": expected_bindings,
            "runtime": runtime,
        }
        runner.write_json(runner.AUTHORITY, authority)
        validated = runner.supervisor.validate_run(runner.PACKAGE, runner.AUTHORITY)
        runner.require(
            validated.package["bindings"]["argv"] == reconstructed_argv
            and validated.authority["bindings"]["argv"] == reconstructed_argv
            and validated.package["bindings"]["argv_sha256"] == reconstructed_sha256
            and validated.authority["bindings"]["argv_sha256"] == reconstructed_sha256,
            "post-authority package/authority argv recheck failed",
        )
        supervisor_command = [
            str(Path(sys.executable).resolve()),
            str(SUPERVISOR),
            "--package",
            str(runner.PACKAGE),
            "--authority",
            str(runner.AUTHORITY),
        ]
        record = {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_NONCONSUMING_EXACT_SUPERVISOR_CHILD_ARGV",
            "created_at_utc": runner.utc_now(),
            "child_argv": reconstructed_argv,
            "child_argv_sha256": reconstructed_sha256,
            "supervisor_command": supervisor_command,
            "supervisor_command_sha256": runner.supervisor.canonical_value_sha256(
                supervisor_command
            ),
            "supervisor": runner.file_record(SUPERVISOR),
            "package": runner.file_record(runner.PACKAGE),
            "authority": runner.file_record(runner.AUTHORITY),
            "checks": {
                "process_not_started": True,
                "pre_start_intent_absent": True,
                "explicit_prompts_argument_present": True,
                "explicit_freeze_sha256_argument_present": True,
                "package_exact_list_equal": True,
                "package_exact_sha256_equal": True,
                "authority_exact_list_equal": True,
                "authority_exact_sha256_equal": True,
                "supervisor_path_exact": True,
            },
        }
        runner.write_json(LAUNCH_REGRESSION, record)
        verify_launch_fidelity(prompts_path)
        print(
            "ACE2_HYBRID_AUTHORITY_LAUNCH_FIDELITY_PASS "
            f"authority_sha256={runner.sha256_file(runner.AUTHORITY)} "
            f"argv_sha256={reconstructed_sha256}",
            flush=True,
        )
        return 0

    runner.grant_authority = grant_authority_0011

    def child_entry_attestation_0011(
        prompts_path: Path, expected_freeze_sha256: str
    ) -> dict[str, Any]:
        attestation = prior_child_entry_attestation(
            prompts_path, expected_freeze_sha256
        )
        launch = verify_launch_fidelity(prompts_path)
        runner.require(
            PRE_IMPORT_ARGV == launch["child_argv"],
            "actual supervisor child argv differs from launch-fidelity regression",
        )
        attestation["launch_fidelity_regression_sha256"] = runner.sha256_file(
            LAUNCH_REGRESSION
        )
        return attestation

    runner.child_entry_attestation = child_entry_attestation_0011

    def verify_0011() -> int:
        result = prior_verify()
        verify_launch_fidelity(PROMPTS)
        state_files = {
            "pre_start_intent": runner.STATE / "pre_start_intent.json",
            "process_started": runner.STATE / "process_started.json",
            "terminal_record": runner.STATE / "terminal_record.json",
        }
        runner.require(
            all(path.is_file() for path in state_files.values()),
            "durable supervisor state record is absent",
        )
        durable_terminal = runner.read_json(state_files["terminal_record"])
        primary_terminal = runner.read_json(runner.TERMINAL)
        started = runner.read_json(state_files["process_started"])
        runner.require(
            durable_terminal == primary_terminal
            and started.get("process_start_count") == 1
            and primary_terminal.get("process_start_count") == 1,
            "durable supervisor start/terminal record changed",
        )
        runner.require(
            runner.sha256_file(runner.STDOUT)
            == primary_terminal["captures"]["stdout"]["sha256"]
            and runner.sha256_file(runner.STDERR)
            == primary_terminal["captures"]["stderr"]["sha256"],
            "supervisor stdout/stderr capture hash changed",
        )
        execution_result = runner.read_json(runner.RESULT)
        runner.require(
            execution_result["classification"]
            == "bounded_nonofficial_autoregressive_rtl_sidecar_hybrid"
            and execution_result["claim_boundary"]
            == "bounded nonofficial hybrid evidence only; not Stage-1 completion",
            "nonofficial bridge claim boundary changed",
        )
        print(
            "ACE2_HYBRID_0011_SUPERVISOR_VERIFY_PASS "
            f"launch_regression_sha256={runner.sha256_file(LAUNCH_REGRESSION)} "
            "process_start_count=1",
            flush=True,
        )
        return result

    runner.verify = verify_0011


def main() -> int:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import run_stage1_layer23_v_rank1_hybrid_0010 as prior

        _configure_prior(prior)
        prior._verify_pre_import_launch()

        from tools import ace2_layer23_v_rank1_integer_correction_reference
        from tools import run_stage1_layer23_v_rank1_hybrid as runner
        from tools import run_stage1_layer23_v_rank1_hybrid_0008 as base

        _configure_runner(
            runner,
            prior,
            base,
            ace2_layer23_v_rank1_integer_correction_reference,
            dict(os.environ),
        )
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
