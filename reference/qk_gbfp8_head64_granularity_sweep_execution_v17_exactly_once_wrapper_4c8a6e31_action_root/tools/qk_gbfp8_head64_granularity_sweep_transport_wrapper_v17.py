
#!/usr/bin/env python3
"""Shell-free exact transport for the irreversible V17 wrapper launcher."""

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_action_root')
INTERPRETER = '/home/argustest/miniconda3/bin/python3.13'
LAUNCHER = '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_action_root/tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py'
EXACT_ARGV = ['/home/argustest/miniconda3/bin/python3.13', '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_action_root/tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py', '--package', '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json', '--acceptance', '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json', '--irreversible-action-id', 'ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z']
EXACT_ENVIRONMENT = {'LANG': 'C', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0', 'TZ': 'UTC'}
LAUNCHER_SHA256 = '7c9c4afed18db213d4d857a4a2607c9055dd669067265501b1dd9801d78d98cd'

def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()

def main() -> int:
    if Path.cwd() != ROOT or dict(os.environ) != EXACT_ENVIRONMENT:
        return 64
    if sys.argv != [__file__, "--package", EXACT_ARGV[3], "--acceptance", EXACT_ARGV[5], "--irreversible-action-id", EXACT_ARGV[7]]:
        return 65
    if sha256_file(LAUNCHER) != LAUNCHER_SHA256:
        return 66
    pid = os.posix_spawn(INTERPRETER, EXACT_ARGV, EXACT_ENVIRONMENT)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)

if __name__ == "__main__":
    raise SystemExit(main())
