#!/usr/bin/env python3
"""Run the environment-only wide-arithmetic compatibility probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/raw/environment/down_projection_residual_fusion_compatibility"
RTL = ROOT / "tools/probes/down_projection_residual_fusion_env_probe.sv"
TB = ROOT / "tools/probes/down_projection_residual_fusion_env_probe_tb.sv"
YS = ROOT / "tools/probes/down_projection_residual_fusion_env_probe.ys"
RESULTS = OUT / "RESULTS.json"
IMAGE = "openroad/orfs@sha256:3bc303869d5e4caac8f72c854f2b1614c726b2961bbb372f54bc8fbc0e725e71"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def run(name: str, command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=180,
    )
    log = OUT / f"{name}.log"
    log.write_text(completed.stdout, encoding="utf-8")
    require(completed.returncode == 0, f"{name} failed\n{completed.stdout}")
    return {"command": command, "exit_status": completed.returncode, "log": artifact(log)}


def validate() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    require(payload.get("contract_id") == "shared_down_projection_residual_fusion_v1", "contract mismatch")
    require(payload.get("status") == "pass", "probe status is not pass")
    require(payload.get("claim_boundary") == "synthetic_environment_capability_only", "claim boundary changed")
    for record in payload.get("sources", []):
        path = ROOT / record["path"]
        require(path.is_file(), f"missing source: {record['path']}")
        require(artifact(path) == record, f"source binding stale: {record['path']}")
    for item in payload.get("checks", []):
        require(item.get("exit_status") == 0, "probe command did not exit zero")
        record = item.get("log", {})
        path = ROOT / str(record.get("path", ""))
        require(path.is_file(), f"missing log: {record.get('path')}")
        require(artifact(path) == record, f"log binding stale: {record.get('path')}")
    sim_log = (OUT / "iverilog_run.log").read_text(encoding="utf-8")
    require("ACE2_DOWN_PROJECTION_ENV_ARITHMETIC_SIM_PASS" in sim_log, "simulation pass marker missing")
    yosys_log = (OUT / "orfs_yosys.log").read_text(encoding="utf-8")
    require("cells" in yosys_log and "$div" in yosys_log and "$mod" in yosys_log,
            "Yosys did not retain the required divide/remainder operators")
    print("ACE2_DOWN_PROJECTION_ENVIRONMENT_PROBE_CHECK_PASS contract=shared_down_projection_residual_fusion_v1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    vvp = OUT / "probe.vvp"
    checks = [
        run("iverilog_compile", ["iverilog", "-g2012", "-s", "down_projection_residual_fusion_env_probe_tb", "-o", str(vvp), str(RTL), str(TB)]),
        run("iverilog_run", ["vvp", str(vvp)]),
        run("verilator_lint", ["verilator", "--lint-only", "-Wall", "-Wno-fatal", str(RTL)]),
        run(
            "orfs_yosys",
            [
                "docker", "run", "--rm", "--user", f"{subprocess.check_output(['id', '-u'], text=True).strip()}:{subprocess.check_output(['id', '-g'], text=True).strip()}",
                "-v", f"{ROOT}:/work", IMAGE, "bash", "-lc",
                "source /OpenROAD-flow-scripts/env.sh && cd /work && yosys -s tools/probes/down_projection_residual_fusion_env_probe.ys",
            ],
        ),
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract_id": "shared_down_projection_residual_fusion_v1",
        "status": "pass",
        "claim_boundary": "synthetic_environment_capability_only",
        "requirements_exercised": [
            "signed_96_bit_numerator",
            "unsigned_64_bit_denominator",
            "variable_integer_divide_and_remainder",
            "signed_round_to_nearest_ties_to_even",
            "signed_int8_saturation",
        ],
        "sources": [artifact(RTL), artifact(TB), artifact(YS)],
        "checks": checks,
    }
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate()


if __name__ == "__main__":
    main()
