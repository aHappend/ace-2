#!/usr/bin/env python3
"""Decisive bounded verifier for the attention max-predicate retime."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/rtl_attn_max_predicate_retime_v1"
RTL = ROOT / "rtl/ace2_shell.sv"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(path: Path) -> str:
    require(path.is_file(), f"missing required artifact: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def trace_lines(path: Path) -> list[str]:
    return [
        line
        for line in read(path).splitlines()
        if line.startswith("ATTN_RETIME_TRACE ")
    ]


def command_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip().splitlines()[0]


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    source = read(RTL)

    required_source_fragments = [
        "reg attn_score_update_max_q;",
        "wire attn_score_update_max_w =",
        "(attn_token_idx_q == {ATTN_TOKEN_INDEX_WIDTH{1'b0}}) ||",
        "(attn_rounded_abs_w < attn_score_max_magnitude_q)",
        "(attn_rounded_abs_w > attn_score_max_magnitude_q)",
        "attn_score_update_max_q <=\n                            attn_score_update_max_w;",
        "if (attn_score_update_max_q) begin\n                            attn_score_max_magnitude_q <= attn_rounded_abs_q;\n                            attn_score_max_negative_q <= attn_score_negative_q;",
        "`ifdef ACE2_ATTN_MAX_RETIME_FORMAL",
    ]
    for fragment in required_source_fragments:
        require(fragment in source, f"required RTL fragment missing: {fragment}")
    require("attn_score_gt_max_w" not in source, "legacy late predicate remains")
    require(
        source.count("attn_score_update_max_q <= 1'b0;") == 3,
        "predicate must clear only on asynchronous reset, watchdog, and ST_IDLE",
    )
    require(
        source.count("if (attn_score_update_max_q) begin") == 1,
        "registered predicate must have one WAIT_OUT consumer",
    )

    lint_log = read(EVIDENCE / "rtl_lint.log")
    focused_log = read(EVIDENCE / "focused_simulation.log")
    pre_log = read(EVIDENCE / "prechange_attention_score_shell.log")
    post_log = read(EVIDENCE / "postchange_attention_score_shell.log")
    require("ACE2_ATTN_MAX_RETIME_LINT_PASS" in lint_log, "fresh lint did not pass")
    require(
        "ACE2_ATTN_MAX_PREDICATE_TB_PASS checks=126" in focused_log,
        "focused predicate simulation did not pass",
    )
    require(
        "ACE2_SHELL_ATTN_SCORE_TB_PASS" in pre_log,
        "pre-change attention shell trace did not pass",
    )
    require(
        "ACE2_SHELL_ATTN_SCORE_TB_PASS" in post_log,
        "post-change attention shell trace did not pass",
    )

    pre_trace = trace_lines(EVIDENCE / "prechange_attention_score_shell.log")
    post_trace = trace_lines(EVIDENCE / "postchange_attention_score_shell.log")
    require(pre_trace, "pre-change cycle trace is empty")
    require(pre_trace == post_trace, "pre/post attention cycle traces differ")

    formal = subprocess.run(
        ["yosys", "-q", "-s", "formal/ace2_attn_max_predicate_retime.ys"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    formal_log = formal.stdout
    if formal.returncode == 0:
        formal_log += "ACE2_ATTN_MAX_RETIME_FORMAL_PASS\n"
    else:
        formal_log += f"ACE2_ATTN_MAX_RETIME_FORMAL_FAIL status={formal.returncode}\n"
    (EVIDENCE / "formal_equivalence.log").write_text(formal_log, encoding="utf-8")
    require(formal.returncode == 0, "bounded formal equivalence failed")

    rtl_files = sorted((ROOT / "rtl").rglob("*.sv"))
    manifest_lines = []
    for path in rtl_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}\n")
    manifest = "".join(manifest_lines).encode("utf-8")
    (EVIDENCE / "rtl_tree_manifest.sha256").write_bytes(manifest)
    tree_hash = hashlib.sha256(manifest).hexdigest()
    (EVIDENCE / "rtl_tree_hash.txt").write_text(
        f"{tree_hash}  rtl_tree_manifest.sha256\n", encoding="utf-8"
    )

    summary = {
        "status": "PASS",
        "rtl_tree_hash": tree_hash,
        "rtl_tree_hash_method": (
            "sha256 of the UTF-8 sorted per-file SHA256 manifest for rtl/**/*.sv"
        ),
        "cycle_trace_lines": len(pre_trace),
        "cycle_trace_identical": True,
        "focused_assertion_checks": 126,
        "formal_equivalence": "PASS",
        "lint": "PASS",
        "attention_shell_simulation": "PASS_pre_and_post",
        "tools": {
            "iverilog": command_version(["iverilog", "-V"]),
            "verilator": command_version(["verilator", "--version"]),
            "yosys": command_version(["yosys", "-V"]),
            "z3": command_version(["z3", "--version"]),
        },
    }
    (EVIDENCE / "verification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = f"""# Attention max-predicate retime implementation report

## Result

PASS for the bounded RTL repair. `attn_score_update_max_q` is captured in
`ST_ATTN_ROUND` from `attn_rounded_abs_w`, token-zero, score sign, and the
current signed-magnitude maximum. `ST_ATTN_WAIT_OUT` uses that registered bit
as the sole enable for both maximum magnitude and sign updates.

## Verification

- Fresh Verilator lint: PASS (`rtl_lint.log`).
- Focused directed simulation: PASS, 126 checks covering token zero, sign
  precedence, smaller/equal/larger magnitudes, rounding increment/no-increment,
  last-token and non-last sequencing, response-fault hold, watchdog clear,
  soft reset, and asynchronous reset (`focused_simulation.log`).
- Full existing attention-score shell simulation: PASS before and after the
  RTL change. All {len(pre_trace)} retained cycle-trace lines for the internal
  maximum state and externally visible request/write/completion behavior are
  byte-identical (`prechange_attention_score_shell.log`,
  `postchange_attention_score_shell.log`).
- Bounded symbolic equivalence of the legacy and registered predicates plus
  reset/watchdog/fault/idle update priority: PASS (`formal_equivalence.log`).

## Deterministic RTL-tree hash

`{tree_hash}`

This is SHA256 over `rtl_tree_manifest.sha256`, whose lines are sorted by path
and contain SHA256 for every `rtl/**/*.sv` file.

No synthesis, OpenSTA, canonical PPA, model, baseline, candidate, benchmark,
Scale32, or protected frontier/ledger/seal update was run or claimed here.
"""
    (EVIDENCE / "IMPLEMENTATION_REPORT.md").write_text(report, encoding="utf-8")

    print("ACE2_ATTN_MAX_RETIME_DECISIVE_VERIFY_PASS")
    print(f"rtl_tree_hash={tree_hash}")
    print(f"cycle_trace_lines={len(pre_trace)}")


if __name__ == "__main__":
    main()
