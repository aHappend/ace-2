#!/usr/bin/env python3
"""Run and bind the ACE-2 environment-stage readiness audit.

This script deliberately uses a tiny synthetic design.  It proves tool and
platform executability without running ACE-2 RTL, verification, PPA, prototype,
benchmark, or sign-off stages.  It binds the active architecture packet but
does not grant implementation authority or perform an independent review.
"""

from __future__ import annotations

import argparse
import copy
import gmpy2
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
AUDIT = RESEARCH / "ENVIRONMENT_AUDIT.json"
AUDIT_MD = RESEARCH / "ENVIRONMENT_AUDIT.md"
PIPELINE = RESEARCH / "PIPELINE_STATE.json"
PUBLIC_STATUS = RESEARCH / "PUBLIC_STATUS.json"
TOOLCHAIN = RESEARCH / "TOOLCHAIN_CANDIDATES.md"
IP_REUSE = RESEARCH / "IP_REUSE_PLAN.md"
SCRIPT = ROOT / "tools" / "run_environment_audit.py"
REVIEW_VERDICT = RESEARCH / "ENVIRONMENT_REVIEWER_VERDICT.json"
REVIEW_CERTIFICATION = RESEARCH / "ENVIRONMENT_L2_CERTIFICATION.md"
REVIEW_RAW_DECISION = (
    ROOT
    / "evidence/review/"
    "environment_stage_closing_layer0_projection_shadow_staged_attention_v1/decision.json"
)
ARCHITECTURE_PACKET = (
    ROOT
    / "evidence/shared_qk_residual_cross_term_attention_v1/latest/"
    "ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
)

ORFS_IMAGE = (
    "openroad/orfs@sha256:"
    "3bc303869d5e4caac8f72c854f2b1614c726b2961bbb372f54bc8fbc0e725e71"
)
ORFS_DIGEST = "sha256:3bc303869d5e4caac8f72c854f2b1614c726b2961bbb372f54bc8fbc0e725e71"
ORFS_ROOT = "/OpenROAD-flow-scripts"
SKY130 = f"{ORFS_ROOT}/flow/platforms/sky130hd"
SKY130_LIB = f"{SKY130}/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
SKY130_TECH_LEF = f"{SKY130}/lef/sky130_fd_sc_hd.tlef"
SKY130_CELL_LEF = f"{SKY130}/lef/sky130_fd_sc_hd_merged.lef"

ARCHITECTURE_INPUTS = [
    "design/ARCHITECTURE.md",
    "design/CHIP_SCOPE.json",
    "design/MEMORY_MODEL.json",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    "design/RTL_MANIFEST.json",
    "design/SPEC.md",
    "evidence/shared_qk_residual_cross_term_attention_v1/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json",
    "research/PUBLIC_STATUS.json",
]

ACTIVE_CONTRACT_ID = "shared_v_residual_value_correction_attention_v1"
ACTIVE_PROPOSAL_SHA256 = (
    "dcd2fdbb7c75cefa9f7ace497aff38d4fa77faf225e846fb37ea662011d34cfe"
)

REMOTE_CANDIDATES = {
    "openroad_flow_scripts": "https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git",
    "skywater_pdk": "https://github.com/google/skywater-pdk.git",
    "open_pdks": "https://github.com/RTimothyEdwards/open_pdks.git",
    "gemmini": "https://github.com/ucb-bar/gemmini.git",
    "vta": "https://github.com/apache/tvm-vta.git",
    "nvdla": "https://github.com/nvdla/hw.git",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_slug(value: str) -> str:
    return value.replace("-", "").replace(":", "")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path | str) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    return {
        "path": relative(resolved),
        "exists": resolved.is_file(),
        "bytes": resolved.stat().st_size if resolved.is_file() else 0,
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def canonical_sha256(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def command_version(command: list[str], first_matching: str | None = None) -> str:
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    require(process.returncode == 0, f"version probe failed: {' '.join(command)}")
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if first_matching:
        for line in lines:
            if first_matching in line:
                return line
    require(bool(lines), f"empty version output: {' '.join(command)}")
    return lines[0]


def run_logged(
    command: list[str],
    log: Path,
    *,
    summary: str,
    timeout: int = 300,
    cwd: Path = ROOT,
) -> dict[str, Any]:
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    log.write_text(process.stdout, encoding="utf-8")
    require(process.returncode == 0, f"{summary} failed; inspect {relative(log)}")
    return {
        "command_summary": summary,
        "exit_status": process.returncode,
        "log": artifact(log),
    }


def docker_command(script: str, *, bind_z3: bool = False) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/work",
        "-w",
        "/work",
    ]
    if bind_z3:
        z3 = shutil.which("z3")
        require(z3 is not None, "Z3 is required for the selected formal flow")
        command.extend(["-v", f"{z3}:/usr/local/bin/z3:ro"])
    command.extend([ORFS_IMAGE, "bash", "-lc", script])
    return command


def write_probe_sources(run_dir: Path) -> None:
    (run_dir / "env_probe.sv").write_text(
        """module env_probe(
  input  logic       clk,
  input  logic       rst_n,
  input  logic [7:0] d,
  output logic [7:0] q,
  output logic [7:0] y
);
  logic signed [31:0]  shadow_word;
  logic signed [63:0]  limb_product;
  logic signed [73:0]  score_accumulator;
  logic signed [105:0] scale_combine;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q                 <= 8'h00;
      shadow_word       <= '0;
      limb_product      <= '0;
      score_accumulator <= '0;
      scale_combine     <= '0;
    end else begin
      q                 <= d + 8'h01;
      shadow_word       <= {{24{d[7]}}, d};
      limb_product      <= shadow_word * 32'sd257;
      score_accumulator <= {{10{limb_product[63]}}, limb_product};
      scale_combine     <= {{32{score_accumulator[73]}}, score_accumulator}
                           + ({{32{score_accumulator[73]}}, score_accumulator} <<< 16);
    end
  end
  assign y = (q ^ d) ^ {8{^scale_combine}};
endmodule
""",
        encoding="utf-8",
    )
    (run_dir / "env_probe_tb.sv").write_text(
        """module env_probe_tb;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic [7:0] d = 8'h00;
  logic [7:0] q;
  logic [7:0] y;
  env_probe dut(.clk, .rst_n, .d, .q, .y);
  always #5 clk = ~clk;
  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    d = 8'h29;
    @(posedge clk);
    #1;
    if (q !== 8'h2a) $fatal(1, "q=%0h expected=2a", q);
    $display("ACE2_ENV_IVERILOG_SMOKE_PASS q=%0h y=%0h", q, y);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    (run_dir / "formal_probe.sv").write_text(
        """module formal_probe(input wire [73:0] a, input wire [73:0] b);
  wire [74:0] sum = {1'b0, a} + {1'b0, b};
  wire signed [105:0] sign_extended = {{32{a[73]}}, a};
  always @* begin
    assert(sum >= a);
    assert(sum >= b);
    assert(sign_extended[105:74] == {32{a[73]}});
  end
endmodule
""",
        encoding="utf-8",
    )
    (run_dir / "formal.sby").write_text(
        """[options]
mode prove
depth 1

[engines]
smtbmc z3

[script]
read -formal formal_probe.sv
prep -top formal_probe

[files]
formal_probe.sv
""",
        encoding="utf-8",
    )
    (run_dir / "compiler_probe.cpp").write_text(
        """#include <iostream>
int main() { std::cout << "ACE2_ENV_CPP_RUNTIME_PASS\\n"; return 0; }
""",
        encoding="utf-8",
    )
    (run_dir / "exact_math_probe.py").write_text(
        """import gmpy2

def ratio(value):
    return tuple(map(int, value.as_integer_ratio()))

with gmpy2.context(gmpy2.get_context(), precision=24, round=gmpy2.RoundToNearest):
    alpha = gmpy2.mpfr(14) / gmpy2.mpfr(64)
    omega = gmpy2.mpfr(1) / (gmpy2.mpfr(1000000) ** alpha)
    angle = gmpy2.mpfr(32767) * omega
    cos_f32 = gmpy2.cos(angle)
    sin_f32 = gmpy2.sin(angle)
    assert ratio(alpha) == (7, 32)
    assert ratio(omega) == (13071935, 268435456)
    assert ratio(cos_f32) == (8055901, 8388608)
    assert ratio(sin_f32) == (-4678121, 16777216)

with gmpy2.context(gmpy2.get_context(), precision=8, round=gmpy2.RoundToNearest):
    cos_bf16 = gmpy2.mpfr(cos_f32)
    sin_bf16 = gmpy2.mpfr(sin_f32)
    assert ratio(cos_bf16) == (123, 128)
    assert ratio(sin_bf16) == (-143, 512)

with gmpy2.context(gmpy2.get_context(), precision=53, round=gmpy2.RoundToNearest):
    exp_step = gmpy2.exp(gmpy2.mpfr(-1) / 16)
    assert ratio(exp_step) == (
        2115370159816873,
        2251799813685248,
    )

print(
    "ACE2_ENV_EXACT_MATH_PASS "
    f"gmpy2={gmpy2.version()} mpfr={gmpy2.mpfr_version()} "
    f"cos_f32={ratio(cos_f32)} sin_f32={ratio(sin_f32)} "
    f"cos_bf16={ratio(cos_bf16)} sin_bf16={ratio(sin_bf16)} "
    f"exp_f64={ratio(exp_step)}"
)
""",
        encoding="utf-8",
    )
    run_rel = relative(run_dir)
    (run_dir / "sky130_probe.ys").write_text(
        f"""read_verilog -sv /work/{run_rel}/env_probe.sv
hierarchy -check -top env_probe
proc
opt
techmap
opt
dfflibmap -liberty {SKY130_LIB}
abc -liberty {SKY130_LIB}
clean
stat -liberty {SKY130_LIB}
write_verilog -noattr /work/{run_rel}/env_probe_mapped.v
""",
        encoding="utf-8",
    )
    (run_dir / "sky130_probe.sdc").write_text(
        """create_clock -name clk -period 10.000 [get_ports clk]
set_input_delay 0.100 -clock clk [get_ports d]
set_output_delay 0.100 -clock clk [get_ports {q y}]
""",
        encoding="utf-8",
    )
    (run_dir / "sky130_sta.tcl").write_text(
        f"""read_liberty {SKY130_LIB}
read_verilog /work/{run_rel}/env_probe_mapped_sta.v
link_design env_probe
read_sdc /work/{run_rel}/sky130_probe.sdc
report_checks -path_delay max -group_count 5
report_checks -path_delay min -group_count 5
report_wns
report_tns
exit
""",
        encoding="utf-8",
    )
    (run_dir / "sky130_openroad.tcl").write_text(
        f"""read_lef {SKY130_TECH_LEF}
read_lef {SKY130_CELL_LEF}
read_liberty {SKY130_LIB}
read_verilog /work/{run_rel}/env_probe_mapped_sta.v
link_design env_probe
initialize_floorplan -die_area {{0 0 120 120}} -core_area {{10 10 110 110}} -site unithd
make_tracks
place_pins -hor_layers met3 -ver_layers met2
global_placement -skip_io -density 0.40
detailed_placement
check_placement -verbose
write_def /work/{run_rel}/env_probe_placed.def
exit
""",
        encoding="utf-8",
    )


def probe_remote_head(url: str) -> dict[str, Any]:
    process = subprocess.run(
        ["git", "ls-remote", "--symref", url, "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    branch = None
    revision = None
    for line in lines:
        if line.startswith("ref: refs/heads/") and line.endswith("\tHEAD"):
            branch = line.split("refs/heads/", 1)[1].split("\t", 1)[0]
        elif line.endswith("\tHEAD") and re.fullmatch(r"[0-9a-f]{40}\tHEAD", line):
            revision = line.split("\t", 1)[0]
    return {
        "status": "reachable" if process.returncode == 0 and revision else "unreachable",
        "branch": branch,
        "revision": revision,
        "query": "git ls-remote --symref <repository> HEAD",
    }


def validate_current_authority_state() -> None:
    packet = load_json(ARCHITECTURE_PACKET)
    require(packet.get("contract_id") == ACTIVE_CONTRACT_ID, "architecture packet contract is stale")
    require(packet.get("current_stage") == "architecture", "architecture packet stage changed")
    require(packet.get("implementation_authorized") is False, "architecture packet authorizes implementation")
    require(
        packet.get("acceptance", {}).get("architecture", {}).get("status")
        == "pending_fresh_independent_reviewer",
        "architecture packet self-claims independent acceptance",
    )
    require(
        packet.get("acceptance", {}).get("environment", {}).get("status")
        == "pending_after_architecture_acceptance",
        "architecture packet self-claims environment acceptance",
    )
    preserved = packet.get("preserved_contracts", {})
    require(preserved.get("area_cap_non_sram_mm2") == 2.0, "area cap changed")
    require(preserved.get("frequency_floor_mhz") == 100.0, "frequency floor changed")
    require(preserved.get("abstract_streaming_memory_boundary_bits") == 128, "stream boundary changed")
    require(preserved.get("first_unsupported_layer_operator") == "layer_0.rope_q", "unsupported frontier changed")
    require(
        preserved.get("ordered_supported_layer_operator_prefix")
        == ["layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"],
        "supported prefix changed",
    )

    scope = load_json(ROOT / "design" / "CHIP_SCOPE.json")
    active = scope.get("operator_owned_execution_policy", {}).get("active_successor_contract", {})
    require(active.get("contract_id") == ACTIVE_CONTRACT_ID, "CHIP_SCOPE active successor is stale")
    require(active.get("proposal_sha256") == ACTIVE_PROPOSAL_SHA256, "CHIP_SCOPE proposal hash is stale")
    require(active.get("implementation_authorized") is False, "CHIP_SCOPE authorizes implementation")
    require(active.get("rtl_started") is False, "CHIP_SCOPE claims successor RTL started")
    require(scope.get("stage", {}).get("current_stage") == "architecture", "CHIP_SCOPE stage changed")
    primary = scope.get("technology_targets", {}).get("primary", {})
    require(primary.get("maximum_non_sram_standard_cell_area_mm2") == 2.0, "CHIP_SCOPE area cap changed")
    require(primary.get("minimum_frequency_hz") == 100_000_000, "CHIP_SCOPE frequency floor changed")

    public = load_json(PUBLIC_STATUS)
    selected = public.get("selected_replacement_contract", {})
    require(selected.get("contract_id") == ACTIVE_CONTRACT_ID, "PUBLIC_STATUS successor is stale")
    require(selected.get("implementation_authorized") is False, "PUBLIC_STATUS authorizes implementation")
    require(public.get("stage", {}).get("current_stage") == "architecture", "public architecture hold changed")
    require(
        sha256_file(ROOT / "design" / "NUMERICAL_REPLACEMENT_PROPOSAL.md")
        == ACTIVE_PROPOSAL_SHA256,
        "active proposal content hash changed after architecture acceptance",
    )


def run_audit() -> dict[str, Any]:
    pipeline = load_json(PIPELINE)
    require(pipeline.get("current_stage") == "environment", "Manager-owned stage is not environment")
    validate_current_authority_state()
    latest_transition = pipeline.get("stage_history", [])[-1]
    require(latest_transition.get("from_stage") == "architecture", "latest Manager transition is not from architecture")
    require(latest_transition.get("to_stage") == "environment", "latest Manager transition is not to environment")
    require(latest_transition.get("by") == "manager", "environment entry is not Manager-owned")

    generated = utc_now()
    run_dir = RESEARCH / "raw" / "environment" / timestamp_slug(generated)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_probe_sources(run_dir)

    python_modules: dict[str, str] = {}
    for module in ("numpy", "pytest", "hypothesis", "torch", "transformers", "datasets", "gmpy2"):
        python_modules[module] = importlib.metadata.version(module)

    host_tools = {}
    for tool in ("python3", "iverilog", "vvp", "verilator", "docker", "git", "make", "gcc", "g++", "z3"):
        located = shutil.which(tool)
        require(located is not None, f"required host tool missing: {tool}")
        host_tools[tool] = {"status": "ready", "executable": f"$PATH/{tool}"}

    host_tools["python3"]["version"] = command_version([sys.executable, "--version"])
    host_tools["iverilog"]["version"] = command_version(["iverilog", "-V"], "Icarus Verilog version")
    host_tools["vvp"]["version"] = command_version(["vvp", "-V"], "Icarus Verilog runtime version")
    host_tools["verilator"]["version"] = command_version(["verilator", "--version"])
    host_tools["docker"]["version"] = command_version(["docker", "--version"])
    host_tools["git"]["version"] = command_version(["git", "--version"])
    host_tools["make"]["version"] = command_version(["make", "--version"], "GNU Make")
    host_tools["gcc"]["version"] = command_version(["gcc", "--version"])
    host_tools["g++"]["version"] = command_version(["g++", "--version"])
    host_tools["z3"]["version"] = command_version(["z3", "--version"])

    iverilog_compile = run_logged(
        [
            "iverilog",
            "-g2012",
            "-s",
            "env_probe_tb",
            "-o",
            str(run_dir / "env_probe.vvp"),
            str(run_dir / "env_probe.sv"),
            str(run_dir / "env_probe_tb.sv"),
        ],
        run_dir / "iverilog_compile.log",
        summary="compile synthetic SystemVerilog smoke with Icarus",
    )
    iverilog_run = run_logged(
        ["vvp", str(run_dir / "env_probe.vvp")],
        run_dir / "iverilog_run.log",
        summary="execute synthetic SystemVerilog smoke with vvp",
    )
    verilator_lint = run_logged(
        [
            "verilator",
            "--lint-only",
            "--language",
            "1800-2017",
            "-Wall",
            "-Wno-fatal",
            str(run_dir / "env_probe.sv"),
        ],
        run_dir / "verilator_lint.log",
        summary="lint synthetic SystemVerilog smoke with Verilator",
    )
    cpp_compile = run_logged(
        [
            "g++",
            "-std=c++17",
            str(run_dir / "compiler_probe.cpp"),
            "-o",
            str(run_dir / "compiler_probe"),
        ],
        run_dir / "compiler_compile.log",
        summary="compile C++17 runtime smoke",
    )
    cpp_run = run_logged(
        [str(run_dir / "compiler_probe")],
        run_dir / "compiler_run.log",
        summary="execute C++17 runtime smoke",
    )
    exact_math = run_logged(
        [sys.executable, str(run_dir / "exact_math_probe.py")],
        run_dir / "exact_math_probe.log",
        summary="execute gmpy2/MPFR correctly rounded transcendental smoke",
    )

    image_process = subprocess.run(
        ["docker", "image", "inspect", ORFS_IMAGE],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    require(image_process.returncode == 0, "pinned ORFS image is unavailable")
    image_data = json.loads(image_process.stdout)[0]
    require(image_data["Id"] == ORFS_DIGEST, "ORFS image ID does not match pinned digest")

    run_rel = relative(run_dir)
    orfs_probe_script = f"""set -euo pipefail
source {ORFS_ROOT}/env.sh
yosys -V
sby --version
openroad -version
sta -version
klayout -v
test -x {ORFS_ROOT}/tools/install/yosys/bin/yosys-smtbmc
test -f {SKY130}/config.mk
test -f {SKY130_TECH_LEF}
test -f {SKY130_CELL_LEF}
test -f {SKY130_LIB}
test -f {SKY130}/gds/sky130_fd_sc_hd.gds
test -f {SKY130}/drc/sky130hd.lydrc
test -f {SKY130}/lvs/sky130hd.lylvs
test -f {ORFS_ROOT}/flow/platforms/nangate45/config.mk
test -f {ORFS_ROOT}/flow/platforms/asap7/config.mk
sha256sum {SKY130}/config.mk {SKY130_TECH_LEF} {SKY130_LIB} {SKY130}/drc/sky130hd.lydrc {SKY130}/lvs/sky130hd.lylvs
openroad -exit <<'TCL'
help scan_replace
help analyze_power_grid
help check_power_grid
help check_antennas
exit
TCL
"""
    orfs_probe = run_logged(
        docker_command(orfs_probe_script),
        run_dir / "orfs_tool_platform_probe.log",
        summary="probe pinned ORFS tools, SKY130/sign-off collateral, DFT/power commands, and comparison platforms",
    )
    formal = run_logged(
        docker_command(
            f"set -euo pipefail; source {ORFS_ROOT}/env.sh; cd /work/{run_rel}; sby -f formal.sby",
            bind_z3=True,
        ),
        run_dir / "formal.log",
        summary="prove synthetic property with SymbiYosys, Yosys SMTBMC, and read-only Z3",
    )
    synthesis = run_logged(
        docker_command(
            f"set -euo pipefail; source {ORFS_ROOT}/env.sh; yosys -l /work/{run_rel}/sky130_yosys.log -s /work/{run_rel}/sky130_probe.ys"
        ),
        run_dir / "synthesis_driver.log",
        summary="map synthetic sequential design with pinned ORFS Yosys and SKY130 Liberty",
    )
    mapped = run_dir / "env_probe_mapped.v"
    require(mapped.is_file() and mapped.stat().st_size > 0, "SKY130 mapped netlist is missing")
    mapped_text = mapped.read_text(encoding="utf-8")
    require("sky130_fd_sc_hd__" in mapped_text, "mapped netlist has no SKY130 standard cells")
    mapped_sta = run_dir / "env_probe_mapped_sta.v"
    mapped_sta.write_text(
        re.sub(
            r"\b(wire|reg|input|output) signed\b",
            r"\1",
            mapped_text,
        ),
        encoding="utf-8",
    )
    sta = run_logged(
        docker_command(
            f"set -euo pipefail; source {ORFS_ROOT}/env.sh; sta -exit /work/{run_rel}/sky130_sta.tcl"
        ),
        run_dir / "sky130_sta.log",
        summary="link mapped SKY130 netlist and execute OpenSTA max/min timing reports",
    )
    physical = run_logged(
        docker_command(
            f"set -euo pipefail; source {ORFS_ROOT}/env.sh; openroad -exit /work/{run_rel}/sky130_openroad.tcl"
        ),
        run_dir / "sky130_openroad.log",
        summary="execute SKY130 floorplan, global placement, detailed placement, and placement checks",
    )
    placed_def = run_dir / "env_probe_placed.def"
    require(placed_def.is_file() and placed_def.stat().st_size > 0, "placed DEF is missing")

    remote_queries = {name: probe_remote_head(url) for name, url in REMOTE_CANDIDATES.items()}
    require(remote_queries["openroad_flow_scripts"]["status"] == "reachable", "ORFS upstream query failed")
    require(remote_queries["skywater_pdk"]["status"] == "reachable", "SKY130 upstream query failed")
    require(remote_queries["gemmini"]["status"] == "reachable", "Gemmini baseline query failed")

    evidence_files = [
        path
        for path in run_dir.iterdir()
        if path.is_file()
    ]
    evidence = [artifact(path) for path in sorted(evidence_files)]

    architecture_bindings = [artifact(path) for path in ARCHITECTURE_INPUTS]
    document_bindings = [artifact(TOOLCHAIN), artifact(IP_REUSE), artifact(SCRIPT)]

    capability_matrix = {
        "compiler_runtime": {
            "required_for_current_delivery": True,
            "status": "ready",
            "selected": [
                host_tools["python3"]["version"],
                host_tools["g++"]["version"],
                f"gmpy2 {python_modules['gmpy2']} with {gmpy2.mpfr_version()}",
            ],
            "python_modules": python_modules,
            "evidence": [cpp_compile, cpp_run, exact_math],
        },
        "exact_transcendental_table_generation": {
            "required_for_current_delivery": True,
            "status": "ready",
            "selected": f"gmpy2 {python_modules['gmpy2']} backed by {gmpy2.mpfr_version()}",
            "rounding_mode": "round_to_nearest_ties_to_even_at_explicit_binary_precision",
            "purpose": "Preserve executable exact-math support for Scale32 conversion and existing table-generation paths without native-libm dependence.",
            "evidence": [exact_math],
        },
        "rtl_simulator": {
            "required_for_current_delivery": True,
            "status": "ready",
            "selected": [host_tools["iverilog"]["version"], host_tools["vvp"]["version"]],
            "evidence": [iverilog_compile, iverilog_run],
        },
        "lint": {
            "required_for_current_delivery": True,
            "status": "ready_with_version_constraint",
            "selected": host_tools["verilator"]["version"],
            "constraint": "Host Verilator is older; use only proven 1800-2017 constructs or record an upgrade need.",
            "evidence": [verilator_lint],
        },
        "formal": {
            "required_for_current_delivery": True,
            "status": "ready",
            "selected": ["SBY v0.67", "ORFS Yosys SMTBMC", host_tools["z3"]["version"]],
            "evidence": [formal],
        },
        "synthesis": {
            "required_for_current_delivery": True,
            "status": "ready",
            "selected": "ORFS Yosys 0.67+post with sky130hd Liberty",
            "evidence": [synthesis, artifact(run_dir / "sky130_yosys.log"), artifact(mapped)],
        },
        "fpga": {
            "required_for_current_delivery": False,
            "status": "not_required",
            "reason": "The frozen delivery level is pre_tapeout ASIC evidence; no FPGA prototype claim is authorized.",
            "reopen_condition": "Select a board, vendor/open toolchain, constraints, memory model, and resource/timing gate before any FPGA claim.",
        },
        "physical_design": {
            "required_for_current_delivery": True,
            "status": "ready",
            "selected": "OpenROAD 26Q3-771-g7cfb2105c9 in pinned ORFS",
            "evidence": [physical, artifact(placed_def)],
        },
        "pdk_platforms": {
            "required_for_current_delivery": True,
            "status": "ready",
            "primary": "ORFS sky130hd public platform",
            "comparative": ["ORFS nangate45 academic platform", "ORFS asap7 predictive platform"],
            "evidence": [orfs_probe],
        },
        "sta_power": {
            "required_for_current_delivery": True,
            "status": "ready_for_flow_execution",
            "selected": "OpenSTA 3.1.0 and OpenROAD activity/power-grid commands",
            "evidence": [sta, orfs_probe],
            "claim_boundary": "No ACE-2 timing or power result is produced by this environment smoke.",
        },
        "dft": {
            "required_for_current_delivery": True,
            "status": "ready_with_public_tool_limitation",
            "selected": "OpenROAD scan_replace",
            "limitation": "No standalone fault/atpg executable is available; final DFT evidence must report that limitation and may not claim ATPG coverage.",
            "evidence": [orfs_probe],
        },
        "drc_lvs_antenna": {
            "required_for_current_delivery": True,
            "status": "ready_with_selected_public_path",
            "selected": "KLayout 0.30.7 SKY130 DRC/LVS decks plus OpenROAD antenna checks",
            "limitation": "Magic/Netgen redundancy is unavailable and is not claimed.",
            "evidence": [orfs_probe],
        },
        "public_baseline_access": {
            "required_for_current_environment_to_rtl_gate": False,
            "status": "selected_with_deferred_runtime",
            "selected": "Gemmini required fair-open benchmark candidate",
            "remote_queries": remote_queries,
            "limitation": "Java/SBT is deferred to the benchmark stage and no baseline result is claimed here.",
        },
    }

    audit: dict[str, Any] = {
        "schema_version": 5,
        "project": "ACE-2",
        "vertical": "chip_design",
        "stage": "environment",
        "generated_at_utc": generated,
        "audit_scope": {
            "delivery_level": "pre_tapeout",
            "current_stage_source": "research/PIPELINE_STATE.json",
            "primary_fast_loop_platform": "sky130hd",
            "comparative_platforms": ["nangate45", "asap7"],
            "probe_design": "synthetic_environment_capability_smoke_not_ace2_rtl",
            "forbidden_downstream_work_respected": True,
        },
        "contract_binding": {
            "current_stage": pipeline["current_stage"],
            "pipeline_state": artifact(PIPELINE),
            "architecture_packet": artifact(ARCHITECTURE_PACKET),
            "manager_architecture_acceptance_transition": copy.deepcopy(latest_transition),
            "review_trigger": "Manager advanced the fresh independently accepted value-residual architecture packet to environment compatibility; implementation remains unauthorized and fresh environment review is pending.",
            "architecture_inputs": architecture_bindings,
            "active_contract_id": ACTIVE_CONTRACT_ID,
            "active_repair_scope": "baseline_attention_distribution_plus_signed4_v_residual_attention_value_correction",
            "architecture_review_status": "manager_advanced_after_fresh_independent_architecture_acceptance",
            "environment_review_status": "pending_fresh_independent_reviewer",
            "implementation_authorized": False,
            "implementation_authority_status": "locked_pending_environment_acceptance_and_manager_rtl_advance",
            "preserved_contracts": copy.deepcopy(load_json(ARCHITECTURE_PACKET)["preserved_contracts"]),
            "dependency_delta": "signed4_residual_v_probability_mac_checked_lane_accumulation_and_scale32_conversion_fit_existing_python_rtl_formal_orfs_sky130_stack_no_new_eda_pdk_rtl_ip_license_board_or_compiler_class",
            "reason": "The frozen value-path successor uses integer arithmetic, checked accumulation, existing Scale32 conversion support, the selected DMA/SRAM interfaces, and the maintained simulation, formal, ORFS, and SKY130 stack. No tool, PDK, board, licensed IP, API, GPU, or model-runtime dependency changed.",
        },
        "selected_environment": {
            "host": {
                "python_modules": python_modules,
                "tools": host_tools,
            },
            "canonical_container": {
                "status": "ready",
                "image": ORFS_IMAGE,
                "image_id": image_data["Id"],
                "created_at_utc": image_data["Created"],
                "size_bytes": image_data["Size"],
                "user_mapping_required": True,
                "z3_bind_mode": "read_only",
            },
            "capabilities": capability_matrix,
        },
        "tool_ip_selection": {
            "status": "selected_before_shared_v_residual_value_correction_attention_v1_rtl",
            "selection_documents": document_bindings,
            "maintained_flow": "Pinned OpenROAD-flow-scripts container; no custom flow replacement selected.",
            "reused_ip_and_collateral": [
                "ORFS sky130hd platform and public standard-cell collateral",
                "ORFS nangate45 and asap7 comparison platforms",
                "OpenROAD/Yosys/SymbiYosys/OpenSTA/KLayout/Z3 tool stack",
                "gmpy2/MPFR exact arithmetic support for deterministic Scale32 and table generation",
                "Python hashlib SHA-256 support for immutable packet and generated-artifact validation",
                "existing ACE-2 projection lanes, DMA, SRAM, softmax, attention-value, round/saturate, vector, and command-control interfaces plus standard-cell arithmetic mapping",
            ],
            "third_party_rtl_vendored": False,
            "proprietary_or_credentialed_dependency": False,
            "public_remote_queries": remote_queries,
        },
        "stage_gate_items": {
            "environment.eda-capabilities": True,
            "environment.tool-ip-selection": True,
        },
        "readiness_summary": {
            "primary_rtl_sky130_fast_loop_ready": True,
            "current_active_contract_compatible": True,
            "implementation_authorized": False,
            "environment_evidence_ready_for_independent_review": True,
            "fresh_independent_environment_acceptance": False,
            "manager_stage_transition_owner": "Manager",
            "planner_stage_transition_authorized": False,
        },
        "non_gating_future_limitations": [
            {
                "id": "standalone_atpg_unavailable",
                "impact": "No ATPG coverage claim; report the limitation during final DFT/sign-off evidence.",
            },
            {
                "id": "magic_netgen_redundancy_unavailable",
                "impact": "KLayout/OpenROAD is the selected public path; no redundant Magic/Netgen claim.",
            },
            {
                "id": "gemmini_java_sbt_deferred",
                "impact": "Benchmark-stage runtime preparation only; no baseline result is claimed here.",
            },
            {
                "id": "fpga_not_selected",
                "impact": "No FPGA claim is in the frozen pre_tapeout delivery level.",
            },
        ],
        "claim_boundaries": [
            "no ACE-2 RTL was compiled, simulated, linted, proved, synthesized, timed, or placed by this environment audit",
            "no ACE-2 verification, PPA, FPGA, benchmark, sign-off, GDS, tapeout-readiness, or silicon claim is made",
            f"the 2.0 mm^2 non-SRAM cap and 100 MHz floor remain unchanged and unproven for {ACTIVE_CONTRACT_ID}",
            "environment readiness does not grant RTL entry or implementation authority",
            "prior environment Reviewer and L2 certifications remain historical and are not presented as review of this recertification",
        ],
        "evidence": {
            "run_directory": relative(run_dir),
            "artifacts": evidence,
            "reproduction_command": "./tools/run_environment_audit.py",
            "validation_command": "./tools/run_environment_audit.py --check",
        },
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    audit["integrity"]["canonical_sha256"] = canonical_sha256(audit)
    write_json(AUDIT, audit)
    write_current_audit_markdown(audit)
    validate_current_environment_evidence()
    return audit


def write_audit_markdown(audit: dict[str, Any]) -> None:
    caps = audit["selected_environment"]["capabilities"]
    rows = []
    for key in (
        "compiler_runtime",
        "exact_transcendental_table_generation",
        "rtl_simulator",
        "lint",
        "formal",
        "synthesis",
        "fpga",
        "physical_design",
        "pdk_platforms",
        "sta_power",
        "dft",
        "drc_lvs_antenna",
        "public_baseline_access",
    ):
        item = caps[key]
        rows.append(
            f"| `{key}` | `{item['status']}` | "
            f"{'yes' if item.get('required_for_current_delivery') else 'no/deferred'} |"
        )
    limitations = "\n".join(
        f"- `{item['id']}`: {item['impact']}"
        for item in audit["non_gating_future_limitations"]
    )
    text = f"""# ACE-2 environment audit

Generated: {audit['generated_at_utc']}

## Result

Fresh project-local executable probes pass for the current
`{ACTIVE_CONTRACT_ID}` architecture contract. Both environment checklist items are satisfied in
`research/ENVIRONMENT_AUDIT.json`; the evidence is ready for independent review
and a Manager-owned transition. This Planner audit does not author or claim a
new Reviewer verdict.

The probe used a tiny synthetic design, not ACE-2 RTL. No downstream RTL,
verification, PPA, prototype, benchmark, sign-off, GDS, tapeout-readiness, or
silicon claim is made.

## Current dependency review

`{ACTIVE_CONTRACT_ID}` reuses the selected projection lanes, DMA, SRAM,
softmax, attention-value, vector, round/saturate, command-control, and public
ORFS/SKY130 capabilities. Its signed-32 shadows, signed-34 products, signed-74
accumulators, signed-106 combine, staged external scratch, and shared 32x32
arithmetic add no external RTL IP, PDK, licensed dependency, FPGA board,
compiler class, or EDA flow. The audit exercises wide SystemVerilog arithmetic
through simulation, lint, formal, synthesis, STA, and placement, and retains
the installed gmpy2/MPFR plus Python hashlib runtime for deterministic
coefficient-image generation and software-side SHA-256 validation.

## Capability matrix

| Capability | Status | Required for frozen pre-tapeout delivery/current gate |
| --- | --- | --- |
{os.linesep.join(rows)}

## Immutable evidence

- Run directory: `{audit['evidence']['run_directory']}`
- Reproduce: `python tools/run_environment_audit.py`
- Validate without rerunning tools: `python tools/run_environment_audit.py --check`
- Canonical ORFS image: `{audit['selected_environment']['canonical_container']['image']}`

## Explicit future limitations

{limitations}

These limitations do not substitute for evidence and do not authorize any
target relaxation. The 2.0 mm2 non-SRAM cap, 100 MHz floor, and 1.05x quality
target remain operator-owned and unchanged.

## Review packet

Use `research/ENVIRONMENT_AUDIT.json` as the machine-readable source of truth,
`research/TOOLCHAIN_CANDIDATES.md` for maintained tool/flow selection, and
`research/IP_REUSE_PLAN.md` for IP and collateral reuse boundaries. The existing
`research/ENVIRONMENT_REVIEWER_VERDICT.json` and
`research/ENVIRONMENT_L2_CERTIFICATION.md` are historical evidence for prior
contracts, not verdicts on this `{ACTIVE_CONTRACT_ID}` refresh. Architecture is
accepted for environment entry, but fresh operator implementation approval is
still absent and this audit does not grant it.
"""
    AUDIT_MD.write_text(text, encoding="utf-8")


def update_public_status(audit: dict[str, Any]) -> None:
    status = load_json(PUBLIC_STATUS)
    pipeline = load_json(PIPELINE)
    architecture_advance = next(
        item
        for item in reversed(pipeline.get("stage_history", []))
        if item.get("from_stage") == "architecture"
        and item.get("to_stage") == "environment"
        and item.get("direction") == "advance"
    )
    architecture_accepted_at = architecture_advance["at"]
    generated = audit["generated_at_utc"]
    decision = (
        "environment_recertified_for_layer0_projection_shadow_staged_attention_v1_"
        "independent_review_and_implementation_approval_pending"
    )

    blockers = [
        item
        for item in status.get("blockers", [])
        if item.get("stage") != "architecture"
        and item.get("id") not in {
            "architecture_independent_l2_acceptance_pending",
            "architecture_successor_contract_independent_l2_review_pending",
            "environment_independent_review_pending",
            "manager_stage_advance_pending",
            "operator_implementation_approval_pending",
            "relative_rope_score_fusion_failed_both_dataset_smoke_gate",
        }
    ]
    blockers.insert(
        0,
        {
            "id": "environment_independent_review_pending",
            "evidence": [
                "research/ENVIRONMENT_AUDIT.json",
                "research/TOOLCHAIN_CANDIDATES.md",
                "research/IP_REUSE_PLAN.md",
            ],
            "resolution": f"Fresh Planner-owned environment probes pass for {ACTIVE_CONTRACT_ID}; independent environment review and a Manager-owned stage transition remain required.",
        },
    )
    blockers.insert(
        1,
        {
            "id": "operator_implementation_approval_pending",
            "stage": "rtl",
            "status": "active",
            "evidence": [
                "research/PIPELINE_STATE.json",
                "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            ],
            "resolution": f"Architecture is accepted and the environment is compatible, but no RTL implementation may begin until the operator grants fresh explicit approval for {ACTIVE_CONTRACT_ID}.",
        },
    )
    blockers[0]["resolution"] = (
        "Fresh Planner-owned environment probes pass for "
        f"{ACTIVE_CONTRACT_ID}; independent environment "
        "review and a Manager-owned stage transition remain required."
    )
    status["blockers"] = blockers

    latest_environment = {
        "status": "planner_evidence_ready_for_independent_review",
        "contract_binding": ACTIVE_CONTRACT_ID,
        "generated_at_utc": generated,
        "checklist": audit["stage_gate_items"],
        "primary_fast_loop_ready": True,
        "dependency_delta": audit["contract_binding"]["dependency_delta"],
        "evidence": [
            "research/ENVIRONMENT_AUDIT.json",
            "research/ENVIRONMENT_AUDIT.md",
            "research/TOOLCHAIN_CANDIDATES.md",
            "research/IP_REUSE_PLAN.md",
        ],
        "reviewer_verdict": "pending",
        "stage_transition_owner": "Manager",
    }
    historical_frontier = copy.deepcopy(
        status.get("dashboard_fields", {}).get("latest_ppa_frontier")
    )
    for container_name in ("dashboard_fields", "implementation_frontier"):
        container = status.setdefault(container_name, {})
        container["current_stage"] = "environment"
        container["current_mode"] = "ADVANCE"
        container["latest_decision"] = decision
        container["required_manager_action"] = (
            "independent_environment_review_then_manager_owned_stage_transition"
        )
        container["required_operator_action"] = (
            "fresh_explicit_implementation_approval_before_any_rtl_change"
        )
        container["latest_ppa_frontier_status"] = "historical_pre_replacement_contract_no_new_ppa"
        container["latest_environment_stage"] = latest_environment
        if not isinstance(container.get("latest_ppa_frontier"), dict) and isinstance(historical_frontier, dict):
            container["latest_ppa_frontier"] = copy.deepcopy(historical_frontier)
        if isinstance(container.get("latest_ppa_frontier"), dict):
            frontier = container["latest_ppa_frontier"]
            frontier["remaining_frequency_reserve_mhz"] = (
                float(frontier.get("fmax_mhz", 0.0)) - 100.0
            )
            frontier["remaining_frequency_reserve_interpretation"] = (
                "derived from the verified 100 MHz operating point minus the 100 MHz floor; not a maximum-frequency search"
            )
        if isinstance(container.get("latest_ppa_stage"), dict):
            container["latest_ppa_stage"]["status"] = "historical_pre_replacement_contract_no_new_ppa"
        if isinstance(container.get("candidate_mechanism"), dict):
            container["candidate_mechanism"].update(
                {
                    "contract_id": ACTIVE_CONTRACT_ID,
                    "implementation_authorized": False,
                    "status": "architecture_accepted_environment_recertified_implementation_approval_pending",
                    "review": {
                        "candidate_capability_accepted": True,
                        "decision": "done",
                        "evidence": "research/PIPELINE_STATE.json",
                        "evidence_sha256": sha256_file(PIPELINE),
                        "independent_l2_acceptance_claim": True,
                        "reviewed_at_utc": architecture_accepted_at,
                        "scope": "architecture_stage_close_for_environment_entry_only",
                        "stage_closing": True,
                    },
                }
            )
        if isinstance(container.get("current_architecture_performance_model"), dict):
            container["current_architecture_performance_model"]["status"] = (
                "architecture_accepted_estimate_pending_implementation_evidence"
            )

    status["current_mode"] = "ADVANCE"
    status["latest_decision"] = decision
    status["latest_ppa_frontier_status"] = "historical_pre_replacement_contract_no_new_ppa"
    status["supported_layer_operator_prefix"] = [
        "layer_0.input_rmsnorm",
        "layer_0.q_proj",
        "layer_0.k_proj",
        "layer_0.v_proj",
    ]
    status["first_unsupported_layer_operator"] = "layer_0.rope_q"

    architecture_review = {
        "decision": "done",
        "evidence": "research/PIPELINE_STATE.json",
        "evidence_sha256": sha256_file(PIPELINE),
        "reviewed_at_utc": architecture_accepted_at,
        "scope": "architecture_stage_close_for_environment_entry_only",
        "stage_closing": True,
    }
    if isinstance(status.get("architecture_proposal_gate"), dict):
        status["architecture_proposal_gate"].update(
            {
                "contract_id": ACTIVE_CONTRACT_ID,
                "implementation_authorized": False,
                "review": architecture_review,
                "status": "architecture_accepted_environment_recertified_implementation_approval_pending",
            }
        )

    status["stage"] = {
        "completed_prior_stages": ["definition", "architecture"],
        "current_stage": "environment",
        "current_stage_status": "planner_evidence_ready_for_independent_review",
        "current_stage_checklist": audit["stage_gate_items"],
        "current_stage_evidence": latest_environment["evidence"],
        "current_stage_source": "research/PIPELINE_STATE.json",
        "downstream_locked_until_manager_advance": [
            "rtl",
            "verification",
            "ppa",
            "prototype",
            "benchmark",
            "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }

    claims = [
        item
        for item in status.get("public_claims", [])
        if not str(item.get("claim", "")).startswith("current Manager-owned stage is")
        and "environment" not in str(item.get("claim", "")).lower()
        and not str(item.get("claim", "")).startswith("fresh project-local tool probes satisfy")
        and not str(item.get("claim", "")).startswith("the exact-transcendental runtime requirement")
        and not str(item.get("claim", "")).startswith("layer0_absolute_rope_online_attention_v1 remains")
    ]
    status["public_claims"] = [
        {
            "claim": "current Manager-owned stage is environment",
            "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"],
        },
        {
            "claim": f"fresh project-local tool probes satisfy both environment checklist items for {ACTIVE_CONTRACT_ID}",
            "evidence": [
                "research/ENVIRONMENT_AUDIT.json",
                "research/TOOLCHAIN_CANDIDATES.md",
                "research/IP_REUSE_PLAN.md",
            ],
        },
        {
            "claim": "the projection-shadow wide-integer and software image-generation requirements are satisfied by the selected simulation/formal/ORFS/SKY130 and Python gmpy2/MPFR/hashlib stack without adding a new EDA, PDK, external RTL-IP, licensed, board, or compiler dependency",
            "evidence": [
                "design/ARCHITECTURE.md",
                "research/ENVIRONMENT_AUDIT.json",
            ],
        },
        {
            "claim": f"{ACTIVE_CONTRACT_ID} remains implementation-unauthorized pending fresh explicit operator approval",
            "evidence": [
                "research/PIPELINE_STATE.json",
                "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            ],
        },
        *claims,
    ]

    metrics = status.setdefault("reviewer_certified_metrics", [])
    for metric in metrics:
        if metric.get("name") == "environment_stage_close_after_attention_value_dependency_refresh":
            metric["status"] = "historical_certification_for_prior_attention_value_contract"
            metric["contract_binding_status"] = "historical_not_current_per_head_qk_environment_evidence"
            metric["reviewed_environment_audit_sha256"] = "65b7f41ce7572514948c25c7cd28f3e201bb5d8e3fed8d31606038e9bdcdcfca"
            metric["current_environment_audit_sha256"] = audit["integrity"]["canonical_sha256"]
            metric["evidence"] = ["research/PIPELINE_STATE.json"]
        if metric.get("name") == "architecture_stage_reclose_per_head_qk_contract":
            metric["status"] = "historical_superseded_by_dynamic_rope_head_scale_v1"
            metric["contract_binding_status"] = "historical_per_head_qk_contract"
        if metric.get("name") == "architecture_stage_reclose_dynamic_rope_head_scale_v1":
            metric["status"] = "historical_rejected_contract"
            metric["contract_binding_status"] = "historical_dynamic_rope_head_scale_v1"
    if not any(
        metric.get("name") == "architecture_stage_reclose_dynamic_rope_head_scale_v1"
        for metric in metrics
    ):
        metrics.append(
            {
                "certified_at_utc": "2026-07-31T14:04:19Z",
                "name": "architecture_stage_reclose_dynamic_rope_head_scale_v1",
                "status": "certified_for_environment_entry",
                "contract_binding_status": ACTIVE_CONTRACT_ID,
                "evidence": ["research/PIPELINE_STATE.json"],
            }
        )
    if not any(
        metric.get("name")
        == "architecture_stage_reclose_layer0_projection_shadow_staged_attention_v1"
        for metric in metrics
    ):
        metrics.append(
            {
                "certified_at_utc": architecture_accepted_at,
                "name": "architecture_stage_reclose_layer0_projection_shadow_staged_attention_v1",
                "status": "certified_for_environment_entry",
                "contract_binding_status": ACTIVE_CONTRACT_ID,
                "evidence": ["research/PIPELINE_STATE.json"],
                "implementation_authorized": False,
            }
        )

    if isinstance(status.get("selected_replacement_contract"), dict):
        status["selected_replacement_contract"].update(
            {
                "contract_id": ACTIVE_CONTRACT_ID,
                "implementation_authorized": False,
                "review": architecture_review,
                "status": "architecture_accepted_environment_recertified_implementation_approval_pending",
            }
        )

    dashboard = status.get("dashboard_fields", {})
    operator_policy = dashboard.get("operator_policy", {}) if isinstance(dashboard, dict) else {}
    if isinstance(operator_policy.get("selected_replacement_contract"), dict):
        operator_policy["selected_replacement_contract"].update(
            {
                "authority": "independently_accepted_architecture_contract",
                "contract_id": ACTIVE_CONTRACT_ID,
                "implementation_authorized": False,
                "proposal_sha256": sha256_file(ROOT / "design" / "NUMERICAL_REPLACEMENT_PROPOSAL.md"),
                "required_manager_action": "complete_environment_review_and_stage_transition_without_authorizing_rtl_implementation",
                "required_operator_action": "fresh_explicit_implementation_approval_before_any_rtl_change",
                "status": "architecture_accepted_environment_recertified_operator_approval_pending",
            }
        )
    if isinstance(dashboard, dict):
        dashboard["required_manager_action"] = (
            "independent_environment_review_then_manager_owned_stage_transition"
        )
        dashboard["required_operator_action"] = (
            "fresh_explicit_implementation_approval_before_any_rtl_change"
        )
        dashboard["routing_status"] = (
            "environment_recertified_pending_independent_review_and_operator_implementation_approval"
        )

    status["generated_at_utc"] = generated
    status["last_updated_utc"] = generated

    paths = {item["path"] for item in status.get("artifact_hashes", []) if isinstance(item, dict) and item.get("path")}
    paths.update(
        {
            "MISSION.md",
            "design/ARCHITECTURE.md",
            "design/FAST_LOOP_POLICY.json",
            "design/MEMORY_MODEL.json",
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "design/RTL_MANIFEST.json",
            "design/SPEC.md",
            "design/TARGET.json",
            "reference/ORACLE_MANIFEST.json",
            "research/ENVIRONMENT_AUDIT.json",
            "research/ENVIRONMENT_AUDIT.md",
            "research/IP_REUSE_PLAN.md",
            "research/PIPELINE_STATE.json",
            "research/TOOLCHAIN_CANDIDATES.md",
            "tools/run_environment_audit.py",
        }
    )
    status["artifact_hashes"] = [artifact(path) for path in sorted(paths) if (ROOT / path).is_file()]
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": "UTF-8, sorted keys, two-space indentation, trailing newline, with integrity.canonical_sha256 set to null",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    write_json(PUBLIC_STATUS, status)


def validate_artifact_binding(item: dict[str, Any]) -> None:
    path = ROOT / item["path"]
    require(path.is_file(), f"bound artifact missing: {item['path']}")
    require(path.stat().st_size == item["bytes"], f"bound artifact size mismatch: {item['path']}")
    require(sha256_file(path) == item["sha256"], f"bound artifact hash mismatch: {item['path']}")


def record_matches_path(item: dict[str, Any], path: Path) -> bool:
    return (
        item.get("path") == relative(path)
        and path.is_file()
        and item.get("bytes") == path.stat().st_size
        and item.get("sha256") == sha256_file(path)
    )


def load_current_environment_review(audit: dict[str, Any]) -> dict[str, Any] | None:
    if not (
        REVIEW_VERDICT.is_file()
        and REVIEW_CERTIFICATION.is_file()
        and REVIEW_RAW_DECISION.is_file()
    ):
        return None

    verdict = load_json(REVIEW_VERDICT)
    if not (
        verdict.get("contract_id") == ACTIVE_CONTRACT_ID
        and verdict.get("stage") == "environment"
        and verdict.get("verdict") == "done"
        and verdict.get("stage_closing") is True
        and verdict.get("implementation_authorized") is False
        and verdict.get("audit_canonical_sha256")
        == audit.get("integrity", {}).get("canonical_sha256")
    ):
        return None

    audit_record = next(
        (
            item
            for item in verdict.get("inputs_reviewed", [])
            if item.get("path") == relative(AUDIT)
        ),
        None,
    )
    if not isinstance(audit_record, dict) or not record_matches_path(audit_record, AUDIT):
        return None

    validate_artifact_binding(verdict["raw_reviewer_decision"])
    raw = load_json(REVIEW_RAW_DECISION)
    require(raw.get("contract_id") == ACTIVE_CONTRACT_ID, "raw environment review contract mismatch")
    require(raw.get("stage") == "environment", "raw environment review stage mismatch")
    require(raw.get("stage_closing") is True, "raw environment review is not stage-closing")
    require(raw.get("decision", {}).get("status") == "done", "raw environment review is not done")
    require(raw.get("evidence", {}).get("unchanged_audit_was_not_rerun") is True,
            "environment review does not preserve the no-rerun boundary")
    require(
        raw.get("evidence", {}).get("audit", {}).get("canonical_sha256")
        == audit.get("integrity", {}).get("canonical_sha256"),
        "raw environment review audit binding mismatch",
    )
    return {"verdict": verdict, "raw": raw}


def validate_current_evidence() -> None:
    validate_current_authority_state()
    audit = load_json(AUDIT)
    pipeline = load_json(PIPELINE)
    require(pipeline.get("current_stage") == "environment", "Manager-owned stage changed from environment")
    require(audit.get("stage") == "environment", "audit stage mismatch")
    require(audit["contract_binding"].get("active_contract_id") == ACTIVE_CONTRACT_ID, "audit contract binding is stale")
    require(audit["contract_binding"].get("architecture_review_status") == "accepted_for_environment_entry", "architecture acceptance is not bound")
    require(audit["contract_binding"].get("implementation_authorized") is False, "environment audit must not authorize implementation")
    require(audit["readiness_summary"].get("current_active_contract_compatible") is True, "active contract compatibility is not established")
    require(audit["readiness_summary"].get("implementation_authorized") is False, "readiness summary overstates implementation authority")
    require(audit["integrity"]["canonical_sha256"] == canonical_sha256(audit), "audit canonical hash mismatch")
    require(audit["contract_binding"]["pipeline_state"]["sha256"] == sha256_file(PIPELINE), "audit pipeline binding is stale")
    current_review = load_current_environment_review(audit)
    for item in audit["contract_binding"]["architecture_inputs"]:
        validate_artifact_binding(item)
    for item in audit["tool_ip_selection"]["selection_documents"]:
        if current_review is not None and item.get("path") == relative(SCRIPT):
            reviewed_script = next(
                (
                    record
                    for record in current_review["raw"].get("evidence", {}).get(
                        "selection_documents", []
                    )
                    if record.get("path") == relative(SCRIPT)
                ),
                None,
            )
            require(
                isinstance(reviewed_script, dict)
                and reviewed_script.get("bytes") == item.get("bytes")
                and reviewed_script.get("sha256") == item.get("sha256"),
                "reviewed environment-audit script binding is missing",
            )
        else:
            validate_artifact_binding(item)
    for item in audit["evidence"]["artifacts"]:
        validate_artifact_binding(item)
    require(
        audit["stage_gate_items"]
        == {
            "environment.eda-capabilities": True,
            "environment.tool-ip-selection": True,
        },
        "environment checklist is not fully satisfied",
    )
    acceptable = {
        "ready",
        "ready_with_version_constraint",
        "ready_for_flow_execution",
        "ready_with_public_tool_limitation",
        "ready_with_selected_public_path",
        "not_required",
        "selected_with_deferred_runtime",
    }
    for name, item in audit["selected_environment"]["capabilities"].items():
        require(item["status"] in acceptable, f"capability is not ready/deferred by contract: {name}")

    status = load_json(PUBLIC_STATUS)
    require(status["stage"]["current_stage"] == "environment", "public stage is stale")
    blocker_ids = {item.get("id") for item in status.get("blockers", [])}
    require(
        not any(item.get("stage") == "architecture" for item in status.get("blockers", [])),
        "public blockers retain a stale architecture-stage gate",
    )
    require(
        "operator_implementation_approval_pending" in blocker_ids,
        "public blockers omit the operator implementation-approval gate",
    )
    require(status["current_mode"] == "ADVANCE", "top-level mode is stale")
    if current_review is None:
        require("manager_stage_advance_pending" not in blocker_ids,
                "public blockers claim a Manager advance before independent review")
        require("environment_independent_review_pending" in blocker_ids,
                "public blockers omit the independent-review gate")
        require(status["latest_decision"].startswith(
            "environment_recertified_for_layer0_projection_shadow_staged_attention_v1"
        ), "top-level decision is stale")
        expected_manager_action = "independent_environment_review_then_manager_owned_stage_transition"
        expected_environment_status = "planner_evidence_ready_for_independent_review"
        expected_reviewer_verdict = "pending"
    else:
        require("manager_stage_advance_pending" in blocker_ids,
                "public blockers omit the Manager stage-advance gate")
        require("environment_independent_review_pending" not in blocker_ids,
                "public blockers retain the completed independent-review gate")
        require(status["latest_decision"].startswith(
            "environment_independently_certified_for_layer0_projection_shadow_staged_attention_v1"
        ), "top-level post-review decision is stale")
        expected_manager_action = "advance_environment_to_rtl_preserve_implementation_authorized_false"
        expected_environment_status = "independent_review_done_manager_stage_transition_pending"
        expected_reviewer_verdict = "done"
    require(status["latest_ppa_frontier_status"] == "historical_pre_replacement_contract_no_new_ppa", "top-level PPA status is ambiguous")
    require(status["dashboard_fields"]["current_stage"] == "environment", "dashboard stage is stale")
    require(status["implementation_frontier"]["current_stage"] == "environment", "implementation stage is stale")
    expected_prefix = [
        "layer_0.input_rmsnorm",
        "layer_0.q_proj",
        "layer_0.k_proj",
        "layer_0.v_proj",
    ]
    required_frontier_fields = load_json(ROOT / "design" / "FAST_LOOP_POLICY.json")["frontier_required_fields"]
    for container_name in ("dashboard_fields", "implementation_frontier"):
        container = status[container_name]
        require(container["current_mode"] == "ADVANCE", f"{container_name} mode is stale")
        require(
            container["required_manager_action"]
            == expected_manager_action,
            f"{container_name} Manager action is stale",
        )
        require(
            container["required_operator_action"]
            == "fresh_explicit_implementation_approval_before_any_rtl_change",
            f"{container_name} operator action is stale",
        )
        require(container["ordered_supported_layer_operator_prefix"] == expected_prefix, f"{container_name} prefix is stale")
        require(container["first_unsupported_layer_operator"] == "layer_0.rope_q", f"{container_name} first unsupported operator is stale")
        require(container["latest_ppa_frontier_status"] == "historical_pre_replacement_contract_no_new_ppa", f"{container_name} PPA status is ambiguous")
        require(container["latest_environment_stage"]["contract_binding"] == ACTIVE_CONTRACT_ID, f"{container_name} environment contract is stale")
        require(container["latest_environment_stage"]["status"] == expected_environment_status,
                f"{container_name} environment review status is stale")
        require(container["latest_environment_stage"]["reviewer_verdict"] == expected_reviewer_verdict,
                f"{container_name} environment reviewer verdict is stale")
        if current_review is not None:
            require(container["latest_environment_stage"].get("implementation_authorized") is False,
                    f"{container_name} environment review overstates implementation authority")
            require(
                container["latest_environment_stage"].get("review", {}).get("evidence_sha256")
                == sha256_file(REVIEW_VERDICT),
                f"{container_name} environment review hash is stale",
            )
        if isinstance(container.get("candidate_mechanism"), dict):
            require(container["candidate_mechanism"].get("implementation_authorized") is False, f"{container_name} candidate overstates implementation authority")
            require(container["candidate_mechanism"].get("review", {}).get("candidate_capability_accepted") is True, f"{container_name} architecture review is stale")
        if isinstance(container.get("latest_ppa_stage"), dict):
            require(container["latest_ppa_stage"]["status"] == "historical_pre_replacement_contract_no_new_ppa", f"{container_name} PPA stage status is stale")
        frontier = container["latest_ppa_frontier"]
        missing = [field for field in required_frontier_fields if field not in frontier]
        require(not missing, f"{container_name} PPA frontier missing fields: {missing}")
    require(status["selected_replacement_contract"].get("contract_id") == ACTIVE_CONTRACT_ID, "selected replacement contract is stale")
    require(status["selected_replacement_contract"].get("implementation_authorized") is False, "selected replacement contract overstates authority")
    require(status["stage"].get("completed_prior_stages") == ["definition", "architecture"], "completed prior stages are stale")
    require(status["integrity"]["canonical_sha256"] == canonical_sha256(status), "PUBLIC_STATUS canonical hash mismatch")
    require(status["privacy_policy"]["public_safe"] is True, "PUBLIC_STATUS privacy gate is false")
    require("/home/" not in json.dumps(status, sort_keys=True), "PUBLIC_STATUS exposes a private host path")
    for item in status["artifact_hashes"]:
        validate_artifact_binding(item)


def write_current_audit_markdown(audit: dict[str, Any]) -> None:
    binding = audit["contract_binding"]
    packet = binding["architecture_packet"]
    text = f"""# ACE-2 Environment Audit

- Contract: `{binding['active_contract_id']}`
- Manager-owned stage observed: `{binding['current_stage']}`
- Architecture packet: `{packet['path']}`
- Architecture packet SHA-256: `{packet['sha256']}`
- EDA capability gate: `pass`
- Tool/IP selection gate: `pass`
- Fresh independent environment acceptance: `pending`
- Implementation authorized: `false`

This audit executed only a synthetic tool/platform capability smoke. It did not
compile, simulate, prove, synthesize, time, place, benchmark, or sign off ACE-2
RTL. The 2.0 mm^2 non-SRAM cap, 100 MHz floor, preserved operator prefix, and
historical PPA frontier remain unchanged and unproven for the successor.

Raw evidence: `{audit['evidence']['run_directory']}`
"""
    AUDIT_MD.write_text(text, encoding="utf-8")


def validate_current_environment_evidence() -> None:
    require(os.access(SCRIPT, os.X_OK), "environment audit entrypoint is not executable")
    validate_current_authority_state()
    pipeline = load_json(PIPELINE)
    require(pipeline.get("current_stage") == "environment", "Manager-owned stage changed from environment")

    audit = load_json(AUDIT)
    require(audit.get("stage") == "environment", "audit stage mismatch")
    require(audit.get("schema_version") == 5, "audit schema is stale")
    binding = audit.get("contract_binding", {})
    require(binding.get("active_contract_id") == ACTIVE_CONTRACT_ID, "audit contract binding is stale")
    require(binding.get("current_stage") == "environment", "audit Manager stage binding is stale")
    require(binding.get("implementation_authorized") is False, "audit authorizes implementation")
    require(
        binding.get("environment_review_status") == "pending_fresh_independent_reviewer",
        "audit self-claims environment acceptance",
    )
    require(
        binding.get("architecture_review_status")
        == "manager_advanced_after_fresh_independent_architecture_acceptance",
        "audit omits the Manager-bound architecture acceptance",
    )
    require(record_matches_path(binding.get("architecture_packet", {}), ARCHITECTURE_PACKET),
            "audit architecture packet binding is stale")
    require(record_matches_path(binding.get("pipeline_state", {}), PIPELINE),
            "audit pipeline binding is stale")
    packet = load_json(ARCHITECTURE_PACKET)
    require(binding.get("preserved_contracts") == packet.get("preserved_contracts"),
            "audit changed the preserved frontier")
    require(audit.get("integrity", {}).get("canonical_sha256") == canonical_sha256(audit),
            "audit canonical hash mismatch")

    for item in binding.get("architecture_inputs", []):
        validate_artifact_binding(item)
    for item in audit.get("tool_ip_selection", {}).get("selection_documents", []):
        validate_artifact_binding(item)
    for item in audit.get("evidence", {}).get("artifacts", []):
        validate_artifact_binding(item)

    require(
        audit.get("stage_gate_items")
        == {"environment.eda-capabilities": True, "environment.tool-ip-selection": True},
        "environment checklist is incomplete",
    )
    readiness = audit.get("readiness_summary", {})
    require(readiness.get("current_active_contract_compatible") is True,
            "active contract compatibility is not established")
    require(readiness.get("environment_evidence_ready_for_independent_review") is True,
            "environment evidence is not review-ready")
    require(readiness.get("fresh_independent_environment_acceptance") is False,
            "audit self-claims independent environment acceptance")
    require(readiness.get("implementation_authorized") is False,
            "readiness summary authorizes implementation")

    acceptable = {
        "ready",
        "ready_with_version_constraint",
        "ready_for_flow_execution",
        "ready_with_public_tool_limitation",
        "ready_with_selected_public_path",
        "not_required",
        "selected_with_deferred_runtime",
    }
    for name, item in audit.get("selected_environment", {}).get("capabilities", {}).items():
        require(item.get("status") in acceptable, f"capability is not ready/deferred: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate the bound audit without rerunning tools")
    args = parser.parse_args()
    if args.check:
        validate_current_environment_evidence()
        audit = load_json(AUDIT)
    else:
        audit = run_audit()
    print(
        "ACE2_ENVIRONMENT_AUDIT_PASS "
        f"generated={audit['generated_at_utc']} "
        "eda_capabilities=true tool_ip_selection=true "
        "manager_transition_owner=Manager"
    )


if __name__ == "__main__":
    main()
