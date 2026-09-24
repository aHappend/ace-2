#!/usr/bin/env python3
"""Fresh exactly-once launcher for nonofficial layer-23 V hybrid attempt 0004."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_stage1_layer23_v_rank1_hybrid as runner

runner.MISSION_ID = "stage1rank1hybrid04"
runner.ATTEMPT_ID = "nonofficial-hybrid-0004"
runner.OUTPUT = (
    runner.ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / runner.ATTEMPT_ID
)
runner.PREFLIGHT = runner.OUTPUT / "preflight.json"
runner.FREEZE = runner.OUTPUT / "freeze.json"
runner.PACKAGE = runner.OUTPUT / "execution-package.json"
runner.AUTHORITY = runner.OUTPUT / "execution-authority.json"
runner.LIVE = runner.OUTPUT / "live"
runner.RESULT = runner.LIVE / "result.json"
runner.FAILURE = runner.LIVE / "failure.json"
runner.SUMS = runner.LIVE / "SHA256SUMS"
runner.STATE = runner.OUTPUT / "supervisor-state"
runner.STDOUT = runner.OUTPUT / "execution.stdout.log"
runner.STDERR = runner.OUTPUT / "execution.stderr.log"
runner.TERMINAL = runner.OUTPUT / "terminal-record.json"
runner.DEFAULT_PROMPTS = (
    runner.ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private/prompts-0004.json"
)
runner.TEMPLATE_GATE = (
    runner.ROOT
    / "build/stage1-layer23-v-rank1-hybrid-v1/private"
    / "package-template-invariant-0004.json"
)
runner.DEPENDENCY_PROBE = (
    runner.ROOT
    / "build/stage1-layer23-v-rank1-hybrid-v1/private"
    / "dependency-probe-0004.json"
)
runner.PREDECESSOR_TERMINAL_SEALS = {
    "nonofficial-hybrid-0001": {
        "sha256": "c97541d51a481812fbf41cb8fccdb035fe560ec97db179ca2a2aa4d6afb2780e",
        "process_start_count": 0,
        "process_started": False,
    },
    "nonofficial-hybrid-0002": {
        "sha256": "c8facc07691df72aab7fed0c6a6c51c15d6bc2c6fa222e622c3ee4c088bf4573",
        "process_start_count": 0,
        "process_started": False,
    },
    "nonofficial-hybrid-0003": {
        "sha256": "defc1c23ce2c85548a8f95971c18f0f2a9c155a8816ce0078060203ec746e50f",
        "process_start_count": 1,
        "process_started": True,
    },
}
runner.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    raise SystemExit(runner.main())
