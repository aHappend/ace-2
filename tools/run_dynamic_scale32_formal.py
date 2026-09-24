#!/usr/bin/env python3
"""Run a bounded Yosys proof on the live dynamic Scale32 accumulator."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/ace2_dynamic_scale32_core.sv"
GENERATED = ROOT / "build/dynamic_scale32/formal_compatible_accumulator.sv"
SCRIPT = ROOT / "formal/ace2_dynamic_scale32_formal.ys"


def main() -> int:
    source = RTL.read_text(encoding="utf-8")
    marker = "module ace2_scale32_tagged_accumulator_core"
    if source.count(marker) != 1:
        raise RuntimeError("dynamic Scale32 accumulator split marker changed")
    accumulator = marker + source.split(marker, 1)[1]
    accumulator = accumulator.split("/* verilator lint_on DECLFILENAME */", 1)[0]
    replacements = {
        "localparam logic [6:0] MAX_EVENTS_U7 = 7'(MAX_EVENTS);":
            "localparam [6:0] MAX_EVENTS_U7 = MAX_EVENTS;",
        "function automatic logic scale32_valid": "function automatic scale32_valid",
        "always_ff": "always",
    }
    if accumulator.count("always @*") != 1 or "always_comb" in accumulator:
        raise RuntimeError("formal parser shim requires one diagnostic-clean always @* process")
    for old, new in replacements.items():
        if old not in accumulator:
            raise RuntimeError(f"formal parser shim no longer matches: {old}")
        accumulator = accumulator.replace(old, new)
    accumulator += "\n`default_nettype wire\n"
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(accumulator, encoding="utf-8")

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
        raise RuntimeError("bounded dynamic Scale32 Yosys SAT proof failed")
    print(
        "ACE2_DYNAMIC_SCALE32_MINIMAL_FORMAL_PASS "
        f"rtl_sha256={hashlib.sha256(RTL.read_bytes()).hexdigest()} "
        "depth=4 properties=exclusive_errors,canonical_exponent,reset_ready,stall_stability"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
