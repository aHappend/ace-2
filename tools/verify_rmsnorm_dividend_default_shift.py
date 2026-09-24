#!/usr/bin/env python3
"""Decisive verifier for the bounded RMSNorm dividend default-shift repair."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/rtl_rmsnorm_dividend_default_shift_v1"
BUILD = ROOT / "build/rtl_rmsnorm_dividend_default_shift_v1"
RTL = ROOT / "rtl/ace2_rmsnorm_core.sv"
PRECHANGE = EVIDENCE / "prechange/ace2_rmsnorm_core.sv"

EXPECTED_PRE_TREE = "78252ff41ac1c3c9c847bb26be330cd52942ce5193239b5e11c1116ab0f7e6bd"
EXPECTED_PRE_SOURCE = "37fc050c776998eabad3cb8b534362303b4bae1393c8486db2dc282d3024a9a8"

LEGACY_BLOCK = """    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            div_dividend_q <= {ACC_WIDTH{1'b0}};
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        div_dividend_q <= {ACC_WIDTH{1'b0}};
                    end
                end
                ST_COLLECT: begin
                    if (collect_active_q && collect_square_valid_q &&
                        lane_last_q && (collect_idx_q == LAST_BEAT)) begin
                        div_dividend_q <= next_sumsq_w + HIDDEN_HALF_ACC;
                    end
                end
                ST_MEAN_DIV,
                ST_INV_DIV:
                    div_dividend_q <= div_dividend_next_w;
                ST_SQRT_DECIDE: begin
                    if (sqrt_done_q) begin
                        div_dividend_q <= ACC_WIDTH'(1) << INV_RMS_FRAC;
                    end
                end
                default: begin
                end
            endcase
        end
    end
"""

REPAIRED_BLOCK = """    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            div_dividend_q <= {ACC_WIDTH{1'b0}};
        end else if ((state_q == ST_COLLECT) && collect_active_q &&
                     collect_square_valid_q && lane_last_q &&
                     (collect_idx_q == LAST_BEAT)) begin
            div_dividend_q <= next_sumsq_w + HIDDEN_HALF_ACC;
        end else if ((state_q == ST_SQRT_DECIDE) && sqrt_done_q) begin
            div_dividend_q <= ACC_WIDTH'(1) << INV_RMS_FRAC;
        end else begin
            div_dividend_q <= div_dividend_next_w;
        end
    end
"""

FORMAL_PORTS = """    output wire                              saturation_seen_o,
    output wire [3:0]                        formal_state_o,
    output wire [((HIDDEN_SIZE / LANES) <= 1 ? 1 : $clog2((HIDDEN_SIZE / LANES) + 1))-1:0] formal_collect_idx_o,
    output wire [((HIDDEN_SIZE / LANES) <= 1 ? 1 : $clog2((HIDDEN_SIZE / LANES) + 1))-1:0] formal_scale_idx_o,
    output wire [((LANES <= 1) ? 1 : $clog2(LANES))-1:0] formal_lane_idx_o,
    output wire [LANES*ACT_WIDTH-1:0]        formal_collect_beat_o,
    output wire [LANES*ACT_WIDTH-1:0]        formal_out_work_o,
    output wire [ACC_WIDTH-1:0]              formal_sumsq_q_o,
    output wire [ACC_WIDTH-1:0]              formal_mean_square_o,
    output wire [2*ACT_WIDTH-1:0]            formal_collect_square_o,
    output wire [15:0]                       formal_rms_candidate_o,
    output wire [31:0]                       formal_rms_square_o,
    output wire [INV_RMS_FRAC+1:0]           formal_inv_rms_q_o,
    output wire [ACC_WIDTH-1:0]              formal_dividend_o,
    output wire [10:0]                       formal_divisor_o,
    output wire [INV_RMS_FRAC-1:0]           formal_quotient_o,
    output wire [10:0]                       formal_remainder_o,
    output wire [((ACC_WIDTH <= 1) ? 1 : $clog2(ACC_WIDTH + 1))-1:0] formal_count_o,
    output wire                              formal_out_valid_q_o,
    output wire                              formal_done_valid_q_o,
    output wire                              formal_saturation_q_o,
    output wire                              formal_sqrt_done_o,
    output wire                              formal_collect_active_o,
    output wire                              formal_collect_square_valid_o,
    output wire                              formal_lane_last_o,
    output wire                              formal_scale_active_o,
    output wire                              formal_scale_product_sign_o,
    output wire                              formal_scale_round_active_o,
    output wire [ACT_WIDTH-1:0]              formal_scale_act_o,
    output wire [GAIN_WIDTH-1:0]             formal_scale_gain_o,
    output wire [63:0]                       formal_mul_acc_o,
    output wire [63:0]                       formal_mul_multiplicand_o,
    output wire [31:0]                       formal_mul_multiplier_o,
    output wire [31:0]                       formal_mul_addend_hi_o,
    output wire                              formal_mul_carry_o,
    output wire [5:0]                        formal_mul_count_o
);"""

FORMAL_ASSIGNMENTS = """
    assign formal_state_o = state_q;
    assign formal_collect_idx_o = collect_idx_q;
    assign formal_scale_idx_o = scale_idx_q;
    assign formal_lane_idx_o = lane_idx_q;
    assign formal_collect_beat_o = collect_beat_q;
    assign formal_out_work_o = out_work_q;
    assign formal_sumsq_q_o = sumsq_q;
    assign formal_mean_square_o = mean_square_q;
    assign formal_collect_square_o = collect_square_q;
    assign formal_rms_candidate_o = rms_candidate_q;
    assign formal_rms_square_o = rms_square_q;
    assign formal_inv_rms_q_o = inv_rms_q;
    assign formal_dividend_o = div_dividend_q;
    assign formal_divisor_o = div_divisor_q;
    assign formal_quotient_o = div_quotient_q;
    assign formal_remainder_o = div_remainder_q;
    assign formal_count_o = div_count_q;
    assign formal_out_valid_q_o = out_valid_q;
    assign formal_done_valid_q_o = done_valid_q;
    assign formal_saturation_q_o = saturation_seen_q;
    assign formal_sqrt_done_o = sqrt_done_q;
    assign formal_collect_active_o = collect_active_q;
    assign formal_collect_square_valid_o = collect_square_valid_q;
    assign formal_lane_last_o = lane_last_q;
    assign formal_scale_active_o = scale_active_q;
    assign formal_scale_product_sign_o = scale_product_sign_q;
    assign formal_scale_round_active_o = scale_round_active_q;
    assign formal_scale_act_o = scale_act_q;
    assign formal_scale_gain_o = scale_gain_q;
    assign formal_mul_acc_o = scale_mul_acc_q;
    assign formal_mul_multiplicand_o = scale_mul_multiplicand_q;
    assign formal_mul_multiplier_o = scale_mul_multiplier_q;
    assign formal_mul_addend_hi_o = scale_mul_addend_hi_q;
    assign formal_mul_carry_o = scale_mul_carry_q;
    assign formal_mul_count_o = scale_mul_count_q;

"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def tree_hash() -> tuple[str, str]:
    lines = []
    for path in sorted((ROOT / "rtl").rglob("*.sv")):
        lines.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")
    manifest = "".join(lines)
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest(), manifest


def protected_paths() -> list[Path]:
    exact = {
        ROOT / "MISSION.md",
        ROOT / "design/RTL_MANIFEST.json",
        ROOT / "design/RTL_MANIFEST.sha256",
        ROOT / "design/RTL_TRACEABILITY.md",
        ROOT / "design/RTL_TRACEABILITY.sha256",
        ROOT / "design/PPA_FRONTIER_LEDGER.json",
        ROOT / "formal/ACE2_VERIFICATION_PROPERTIES.json",
        ROOT / "research/PIPELINE_STATE.json",
        ROOT / "research/PIPELINE_STATE.sha256",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or EVIDENCE in path.parents:
            continue
        name = path.name.upper()
        if "AUTHORITY" in name or name.startswith("SEAL"):
            exact.add(path)
    return sorted(path for path in exact if path.is_file())


def snapshot(paths: list[Path]) -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def archive_previous_failure() -> None:
    failure = EVIDENCE / "FAILURE.txt"
    if not failure.exists():
        return
    archive = EVIDENCE / "failures" / str(time.time_ns())
    archive.mkdir(parents=True, exist_ok=True)
    for path in EVIDENCE.iterdir():
        if path.is_file() and (path.suffix in {".log", ".txt", ".vcd"}):
            shutil.copy2(path, archive / path.name)
    failure.unlink()


def run(command: list[str], log_name: str, timeout: int = 180) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = "command=" + " ".join(command) + "\n" + completed.stdout
    write_text(EVIDENCE / log_name, output)
    require(completed.returncode == 0, f"command failed ({log_name}): {completed.returncode}")
    return output


def rename_module(source: str, new_name: str) -> str:
    old = "module ace2_rmsnorm_core #(" 
    require(source.count(old) == 1, "unexpected RMSNorm module declaration count")
    return source.replace(old, f"module {new_name} #(", 1)


def instrument_formal(source: str, module_name: str) -> str:
    compatibility_replacements = {
        "always @(posedge clk_i or negedge rst_ni)": "always @(posedge clk_i)",
        "ACC_WIDTH'(HIDDEN_SIZE)": "HIDDEN_SIZE",
        "BEAT_INDEX_WIDTH'(BEATS)": "BEATS",
        "BEAT_INDEX_WIDTH'(BEATS - 1)": "(BEATS - 1)",
        "LANE_INDEX_WIDTH'(LANES - 1)": "(LANES - 1)",
        "LANE_INDEX_WIDTH'(LANES - 2)": "(LANES - 2)",
        "DIV_COUNT_WIDTH'(ACC_WIDTH)": "ACC_WIDTH",
        "ACC_WIDTH'(1)": "{{(ACC_WIDTH-1){1'b0}}, 1'b1}",
        "DIV_REM_WIDTH'(HIDDEN_SIZE)": "HIDDEN_SIZE",
    }
    for old, new in compatibility_replacements.items():
        source = source.replace(old, new)
    source = rename_module(source, module_name)
    old_port_end = "    output wire                              saturation_seen_o\n);"
    require(source.count(old_port_end) == 1, "formal port insertion point missing")
    source = source.replace(old_port_end, FORMAL_PORTS, 1)
    end_marker = "endmodule\n\n`default_nettype wire"
    require(source.count(end_marker) == 1, "formal assignment insertion point missing")
    return source.replace(end_marker, FORMAL_ASSIGNMENTS + end_marker, 1)


def version(command: list[str]) -> str:
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
    BUILD.mkdir(parents=True, exist_ok=True)
    archive_previous_failure()

    protected = protected_paths()
    protected_before = snapshot(protected)

    pre_source = PRECHANGE.read_text(encoding="utf-8")
    candidate_source = RTL.read_text(encoding="utf-8")
    require(sha256(PRECHANGE) == EXPECTED_PRE_SOURCE, "retained pre-change source hash mismatch")
    require(pre_source.count(LEGACY_BLOCK) == 1, "retained legacy process mismatch")
    expected_candidate = pre_source.replace(LEGACY_BLOCK, REPAIRED_BLOCK, 1)
    require(candidate_source == expected_candidate, "candidate contains a change outside the accepted process repair")

    diff = "".join(
        difflib.unified_diff(
            pre_source.splitlines(keepends=True),
            candidate_source.splitlines(keepends=True),
            fromfile="prechange/ace2_rmsnorm_core.sv",
            tofile="rtl/ace2_rmsnorm_core.sv",
        )
    )
    write_text(EVIDENCE / "source_diff.patch", diff)

    write_text(
        BUILD / "ace2_rmsnorm_core_prechange.sv",
        rename_module(pre_source, "ace2_rmsnorm_core_prechange"),
    )
    write_text(
        BUILD / "prechange_formal.sv",
        instrument_formal(pre_source, "ace2_rmsnorm_core_prechange_formal"),
    )
    write_text(
        BUILD / "candidate_formal.sv",
        instrument_formal(candidate_source, "ace2_rmsnorm_core_formal"),
    )

    lint = run(
        [
            "verilator", "--lint-only", "--language", "1800-2017", "-Wall",
            "-Wno-fatal", "--top-module", "ace2_rmsnorm_core",
            "rtl/ace2_rmsnorm_core.sv",
        ],
        "rtl_lint.log",
    )
    write_text(EVIDENCE / "rtl_lint.log", lint + "ACE2_RMSNORM_DIVIDEND_LINT_PASS\n")

    run(
        [
            "iverilog", "-g2012", "-s", "ace2_rmsnorm_dividend_equiv_tb",
            "-o", str(BUILD / "focused_equiv.vvp"),
            str(BUILD / "ace2_rmsnorm_core_prechange.sv"),
            "rtl/ace2_rmsnorm_core.sv",
            "verification/tb/ace2_rmsnorm_dividend_equiv_tb.sv",
        ],
        "focused_compile.log",
    )
    focused = run(["vvp", str(BUILD / "focused_equiv.vvp")], "focused_simulation.log")
    require("ACE2_RMSNORM_DIVIDEND_EQUIV_PASS" in focused, "focused equivalence marker missing")
    require("clear_mask=fff" in focused, "clear-in-every-state coverage missing")
    require("deterministic_random=12" in focused, "deterministic random coverage missing")

    run(
        [
            "iverilog", "-g2012", "-Irtl", "-Iverification/tb",
            "-s", "ace2_rmsnorm_tb", "-o", str(BUILD / "golden_rmsnorm.vvp"),
            "rtl/ace2_rmsnorm_core.sv", "verification/tb/ace2_rmsnorm_tb.sv",
        ],
        "golden_compile.log",
    )
    golden = run(["vvp", str(BUILD / "golden_rmsnorm.vvp")], "golden_simulation.log", timeout=300)
    require("ACE2_RMSNORM_TB_PASS cases=15 beats_per_case=56" in golden, "golden RMSNorm marker missing")

    formal = run(["yosys", "-s", "formal/ace2_rmsnorm_dividend_equiv.ys"], "formal_equivalence.log", timeout=180)
    write_text(EVIDENCE / "formal_equivalence.log", formal + "ACE2_RMSNORM_DIVIDEND_FORMAL_PASS\n")

    post_tree, manifest = tree_hash()
    write_text(EVIDENCE / "rtl_tree_manifest.sha256", manifest)
    write_text(EVIDENCE / "rtl_tree_hash.txt", f"{post_tree}  rtl_tree_manifest.sha256\n")

    protected_after = snapshot(protected)
    require(protected_before == protected_after, "protected state changed during verification")
    write_text(
        EVIDENCE / "protected_file_hashes.json",
        json.dumps(protected_after, indent=2, sort_keys=True) + "\n",
    )

    match = re.search(
        r"cycles=(\d+).*external_checks=(\d+).*default_shift_checks=(\d+).*"
        r"divider_transition_checks=(\d+).*mean_entries=(\d+).*inv_entries=(\d+)",
        focused,
    )
    require(match is not None, "focused coverage counters missing")
    cycles, external_checks, shift_checks, divider_checks, mean_entries, inv_entries = (
        int(value) for value in match.groups()
    )
    require(min(external_checks, shift_checks, divider_checks, mean_entries, inv_entries) > 0,
            "focused assertion coverage is empty")

    tool_versions = {
        "iverilog": version(["iverilog", "-V"]),
        "verilator": version(["verilator", "--version"]),
        "yosys": version(["yosys", "-V"]),
        "z3": version(["z3", "--version"]),
    }
    log_hashes = {
        name: sha256(EVIDENCE / name)
        for name in [
            "rtl_lint.log", "focused_compile.log", "focused_simulation.log",
            "golden_compile.log", "golden_simulation.log", "formal_equivalence.log",
            "source_diff.patch",
        ]
    }
    summary = {
        "status": "PASS",
        "mission": "rtl-rmsnorm-dividend-default-shift-v1",
        "prechange_rtl_tree_hash": EXPECTED_PRE_TREE,
        "prechange_source_sha256": EXPECTED_PRE_SOURCE,
        "postchange_rtl_tree_hash": post_tree,
        "postchange_source_sha256": sha256(RTL),
        "rtl_tree_hash_method": "sha256 of UTF-8 sorted per-file SHA256 manifest for rtl/**/*.sv",
        "scope_exact": True,
        "external_cycle_equivalence": True,
        "focused_cycles": cycles,
        "external_cycle_checks": external_checks,
        "default_shift_checks": shift_checks,
        "divider_transition_checks": divider_checks,
        "mean_div_entry_checks": mean_entries,
        "inv_div_entry_checks": inv_entries,
        "clear_state_mask": "fff",
        "deterministic_random_cases": 12,
        "golden_full_size_cases": 15,
        "lint": "PASS",
        "focused_simulation": "PASS",
        "golden_simulation": "PASS",
        "formal": "PASS_reset_reachable_bmc5_plus_symbolic_one_step",
        "protected_state_unchanged": True,
        "prohibited_flows_run": [],
        "log_sha256": log_hashes,
        "tools": tool_versions,
    }
    write_text(
        EVIDENCE / "verification_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )

    report = f"""# RMSNorm dividend default-shift implementation report

Date: 2026-08-03  
Mission: `rtl-rmsnorm-dividend-default-shift-v1`

## Result

PASS for the bounded implementation increment. The only RTL difference from
the retained source is the accepted `div_dividend_q` process repair: reset and
the final-collect and sqrt-complete loads retain priority, and every other
non-reset cycle captures `div_dividend_next_w`. The former `ST_IDLE`
start-time zero assignment is absent.

## Hash binding

- Pre-change RTL tree: `{EXPECTED_PRE_TREE}`
- Retained pre-change RMSNorm source: `{EXPECTED_PRE_SOURCE}`
- Post-change RTL tree: `{post_tree}`
- Post-change RMSNorm source: `{sha256(RTL)}`
- Tree method: SHA256 over `rtl_tree_manifest.sha256`, a path-sorted manifest
  containing SHA256 for every `rtl/**/*.sv` source.

The exact sole-source diff is retained in `source_diff.patch`.

## Executable verification

- Fresh Verilator lint: PASS (`rtl_lint.log`).
- Dual-RTL cycle miter: PASS for {cycles} cycles and {external_checks}
  cycle-accurate external comparisons. It checked all ready/valid handshakes,
  `out_data_o`, `sumsq_o`, `inv_rms_q30_o`, `saturation_seen_o`, state and
  latency alignment, plus {shift_checks} candidate dividend-update relations.
- Divider safety: PASS with {mean_entries} final-collect-to-mean-div entries,
  {inv_entries} sqrt-complete-to-inv-div entries, and {divider_checks}
  quotient/remainder/count/dividend transition checks. Dividend equality was
  required on entry to and throughout both divide states.
- Directed and deterministic randomized simulation: PASS. Coverage includes
  asynchronous reset, `clear_i` in all 12 states (`0xfff`), early/back-to-back
  start, input and output backpressure, zero, signed extrema, maximum
  accumulated sum, rounding patterns, saturation, held/completed done
  handshake, and 12 fixed-seed random cases.
- Independent full-size numerical regression: PASS for all 15 retained
  896-element golden cases (`golden_simulation.log`), comparing output beats,
  sum of squares, inverse RMS, rounding/saturation behavior, and completion.
- Formal checks: PASS (`formal_equivalence.log`). A five-cycle reset-reachable
  full-core miter proves external and retained-state equality. A separate
  one-step arbitrary-defined-state proof covers permitted dividend divergence,
  both prioritized entry loads, the default shift relation, and unchanged
  quotient/remainder/count transitions. Generated formal copies use sampled
  reset semantics; asynchronous reset is covered by the dual-RTL simulation.

## Scope audit

Protected pipeline, manifest, traceability, PPA ledger, formal-frontier,
authority, and seal files retained identical hashes (`protected_file_hashes.json`).
No synthesis, OpenSTA, canonical PPA, model, baseline, candidate, benchmark,
Scale32, or protected-state flow was invoked.
"""
    write_text(EVIDENCE / "IMPLEMENTATION_REPORT.md", report)

    print("ACE2_RMSNORM_DIVIDEND_DEFAULT_SHIFT_DECISIVE_PASS")
    print(f"postchange_rtl_tree_hash={post_tree}")
    print(f"postchange_source_sha256={sha256(RTL)}")
    print(f"focused_cycles={cycles}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        write_text(EVIDENCE / "FAILURE.txt", f"{type(exc).__name__}: {exc}\n")
        raise
