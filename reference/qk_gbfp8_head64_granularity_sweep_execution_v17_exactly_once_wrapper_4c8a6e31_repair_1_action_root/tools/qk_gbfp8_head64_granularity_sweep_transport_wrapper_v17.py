#!/home/argustest/miniconda3/bin/python3.13
"""Shell-free exact transport for the irreversible V17 wrapper launcher."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple

from qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17 import Outcome, RuntimePaths, retire_preconsumption_failure


ROOT = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root')
RUNTIME_ROOT = Path('/home/argustest/ace-2/runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31')
INTERPRETER = '/home/argustest/miniconda3/bin/python3.13'
LAUNCHER = '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root/tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py'
PACKAGE = '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json'
ACCEPTANCE = '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json'
ACTION_ID = 'ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z'
EXACT_ARGV = [INTERPRETER, LAUNCHER, '--package', PACKAGE, '--acceptance', ACCEPTANCE, '--irreversible-action-id', ACTION_ID]
EXACT_ENVIRONMENT = {'LANG': 'C', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0', 'TZ': 'UTC'}
LAUNCHER_SHA256 = '1bf18b6e683a4411f7958bb085e4d688da24d25b12c6e7cacbe9840e15461c74'
PATHS = RuntimePaths(
    owner=RUNTIME_ROOT / 'primary/authority/base/owner-claim.json',
    authority=RUNTIME_ROOT / 'primary/authority/base/authority.json',
    credential=RUNTIME_ROOT / 'primary/authority/base/credential.json',
    ledger=RUNTIME_ROOT / 'primary/authority/base/authority-ledger.json',
    result=RUNTIME_ROOT / 'primary/result/base/result.json',
    first_terminal=RUNTIME_ROOT / 'primary/authority/base/first-terminal.json',
    fallback_terminal=RUNTIME_ROOT / 'fallback/first-terminal.json',
)


class TransportError(RuntimeError):
    def __init__(self, message: str, failure_stage: str) -> None:
        super().__init__(message)
        self.failure_stage = failure_stage


class TransportResult(NamedTuple):
    exit_code: int
    outcome: Outcome | None
    spawned: bool


def expected_transport_arguments() -> list[str]:
    return ['--package', PACKAGE, '--acceptance', ACCEPTANCE, '--irreversible-action-id', ACTION_ID]


def require(condition: bool, message: str, failure_stage: str) -> None:
    if not condition:
        raise TransportError(message, failure_stage)


def sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, 'O_NOFOLLOW') else 0)
    descriptor = os.open(path, flags)
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


def run_transport(
    *,
    paths: RuntimePaths = PATHS,
    argv: list[str] | None = None,
    observed_cwd: Path | None = None,
    observed_environment: dict[str, str] | None = None,
    launcher_hash_provider: Callable[[Path], str] = sha256_file,
    spawn: Callable[..., int] | None = None,
    waitpid: Callable[[int, int], tuple[int, int]] | None = None,
) -> TransportResult:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    cwd = Path.cwd() if observed_cwd is None else observed_cwd
    environment = dict(os.environ) if observed_environment is None else observed_environment
    stage = 'TRANSPORT_CWD'
    try:
        require(cwd == ROOT, 'transport cwd', stage)
        stage = 'TRANSPORT_ENVIRONMENT'
        require(environment == EXACT_ENVIRONMENT, 'transport environment', stage)
        stage = 'TRANSPORT_ARGV'
        require(raw_argv == expected_transport_arguments(), 'transport argv', stage)
        stage = 'TRANSPORT_LAUNCHER_HASH'
        require(launcher_hash_provider(Path(LAUNCHER)) == LAUNCHER_SHA256, 'launcher hash', stage)
        stage = 'TRANSPORT_SPAWN'
        spawn_process = os.posix_spawn if spawn is None else spawn
        pid = spawn_process(INTERPRETER, EXACT_ARGV, EXACT_ENVIRONMENT)
    except BaseException as error:
        failure_stage = getattr(error, 'failure_stage', stage)
        outcome = retire_preconsumption_failure(paths, ACTION_ID, error, failure_stage)
        return TransportResult(1, outcome, False)

    wait_process = os.waitpid if waitpid is None else waitpid
    _, status = wait_process(pid, 0)
    return TransportResult(os.waitstatus_to_exitcode(status), None, True)


def main() -> int:
    return run_transport().exit_code


if __name__ == '__main__':
    raise SystemExit(main())
