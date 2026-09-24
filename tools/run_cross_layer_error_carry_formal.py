#!/usr/bin/env python3
"""Run bounded Yosys SAT checks on a parser-compatible live QECR lane core."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/ace2_cross_layer_error_carry_core.sv"
GENERATED = ROOT / "build/cross_layer_error_carry/formal_compatible_lane.sv"
SCRIPT = ROOT / "formal/ace2_cross_layer_error_carry_formal.ys"


def main() -> int:
    source = RTL.read_text(encoding="utf-8")
    marker = "module ace2_error_carry_state_core"
    if source.count(marker) != 1:
        raise RuntimeError("QECR formal source split marker changed")
    lane_source = source.split(marker, 1)[0]
    replacements = {
        "localparam logic [2:0]": "localparam [2:0]",
        "function automatic logic scale32_valid": "function automatic scale32_valid",
        "function automatic logic [5:0] exponent_delta": "function automatic [5:0] exponent_delta",
        "function automatic logic [96:0] multiply_u64_by_255": "function automatic [96:0] multiply_u64_by_255",
        "function automatic logic [96:0] multiply_u64_by_257": "function automatic [96:0] multiply_u64_by_257",
        "always_comb": "always @*",
        "always_ff": "always",
        """            exponent_delta = 6'($unsigned(
                $signed({exponent[7], exponent}) -
                $signed({common_exponent[7], common_exponent})
            ));""": """            exponent_delta = $unsigned(
                $signed({exponent[7], exponent}) -
                $signed({common_exponent[7], common_exponent})
            );""",
    }
    for old, new in replacements.items():
        if old not in lane_source:
            raise RuntimeError(f"QECR formal parser shim no longer matches: {old}")
        lane_source = lane_source.replace(old, new)
    lane_source += "\n`default_nettype wire\n"
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(lane_source, encoding="utf-8")
    result = subprocess.run(
        ["yosys", "-Q", "-q", "-s", str(SCRIPT.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError("bounded QECR Yosys SAT proof failed")
    print(
        "ACE2_QECR_MINIMAL_FORMAL_PASS "
        f"rtl_sha256={hashlib.sha256(RTL.read_bytes()).hexdigest()} "
        "depth=3 properties=exclusive_errors,latency_bound,successful_range,reset_ready,stall_stability"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
