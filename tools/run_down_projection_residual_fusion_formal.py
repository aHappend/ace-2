#!/usr/bin/env python3
"""Run bounded Yosys SAT checks on an exact parser-compatible DPRF RTL copy."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/ace2_down_projection_residual_fusion_core.sv"
GENERATED = ROOT / "build/down_projection_residual_fusion/formal_compatible_core.sv"
SCRIPT = ROOT / "formal/ace2_down_projection_residual_fusion_formal.ys"

ENUM_SOURCE = """    typedef enum logic [2:0] {
        DPRF_IDLE,
        DPRF_PREPARE,
        DPRF_DIVIDE,
        DPRF_FINALIZE
    } state_t;

    state_t state_q;
"""
ENUM_REPLACEMENT = """    localparam [2:0] DPRF_IDLE = 3'd0;
    localparam [2:0] DPRF_PREPARE = 3'd1;
    localparam [2:0] DPRF_DIVIDE = 3'd2;
    localparam [2:0] DPRF_FINALIZE = 3'd3;

    logic [2:0] state_q;
"""
PARSER_REPLACEMENTS = {
    "function automatic logic scale32_valid": "function automatic scale32_valid",
    "function automatic logic [96:0] multiply_u64_by_255": "function automatic [96:0] multiply_u64_by_255",
    "function automatic logic [96:0] multiply_u64_by_257": "function automatic [96:0] multiply_u64_by_257",
    "function automatic logic [5:0] exponent_delta": "function automatic [5:0] exponent_delta",
    """            exponent_delta = 6'($unsigned(
                $signed({exponent[7], exponent}) -
                $signed({common_exponent[7], common_exponent})
            ));""": """            exponent_delta = $unsigned(
                $signed({exponent[7], exponent}) -
                $signed({common_exponent[7], common_exponent})
            );""",
}


def main() -> int:
    source = RTL.read_text(encoding="utf-8")
    if source.count(ENUM_SOURCE) != 1:
        raise RuntimeError("DPRF formal parser shim no longer matches the live RTL enum")
    compatible = source.replace(ENUM_SOURCE, ENUM_REPLACEMENT)
    for old, new in PARSER_REPLACEMENTS.items():
        if compatible.count(old) != 1:
            raise RuntimeError(f"DPRF formal parser shim no longer matches: {old}")
        compatible = compatible.replace(old, new)
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(compatible, encoding="utf-8")
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
        raise RuntimeError("bounded DPRF Yosys SAT proof failed")
    print(
        "ACE2_DPRF_MINIMAL_FORMAL_PASS "
        f"rtl_sha256={hashlib.sha256(RTL.read_bytes()).hexdigest()} "
        "properties=mutual_exclusion,saturation_values,latency_bound,reset_ready,stall_stability"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
