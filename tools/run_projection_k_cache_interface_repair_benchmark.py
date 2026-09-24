#!/usr/bin/env python3
"""Freeze and run the mission-local projection K-cache interface repair."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_projection_k_cache_benchmark as v3

ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = "35b17ba3b226"
NAME = "projection-k-cache-interface-repair-v1"
BENCHMARK_DIR = ROOT / "benchmark/projection_k_cache_interface_repair_v1"
BUILD_DIR = ROOT / "build/projection-k-cache-interface-repair-v1"
MANIFEST = BENCHMARK_DIR / "interface_manifest.json"
CONTRACT = BENCHMARK_DIR / "contract.json"
PUBLIC_RESULTS = BENCHMARK_DIR / "results.json"
BASELINE_SHELL = BUILD_DIR / "frozen-baseline/rtl/ace2_shell.sv"
CANDIDATE_SHELL = BUILD_DIR / "frozen-candidate/rtl/ace2_shell.sv"
BASELINE_OUTPUT = BUILD_DIR / "round-0001-baseline"
CANDIDATE_OUTPUT = BUILD_DIR / "round-0001-candidate"
V3_CONTRACT = ROOT / "benchmark/projection_k_cache_v3/contract.json"
V3_RESULTS = ROOT / "benchmark/projection_k_cache_v3/results.json"
V3_BASELINE_SHELL = (
    ROOT / "build/projection-k-cache-v3/frozen-baseline/rtl/ace2_shell.sv"
)
LIVE_CANDIDATE_SHELL = ROOT / "rtl/ace2_shell.sv"
TB = ROOT / "verification/tb/ace2_projection_k_cache_benchmark_tb.sv"

EXPECTED_IDENTITIES = {
    "v3_contract": "6fa062985fbf1c1ca2adda6b5980ddeddf6618c3d1cf633504fc44d46b6e2ba5",
    "v3_results": "64c3b93b7835c2dcee9898375b059f32f2b63d81467f6c1dc092019ca5adc06e",
    "v3_baseline": "febccf8762a43149eb1964796ff0c842a791cfcdeca106150efcd6d6147ad216",
    "candidate": "6a3db778c0c5e8b410bd4a8c19fbdac738c5cd3dbbbfa7d4ca9b4853a5b51d84",
    "testbench": "7e537236772cc42cdbbd9fd735f858b922607a88f2d24b80e4f578e09ef0b66d",
}

PARAMETERS = [
    {
        "declaration": "parameter integer HIDDEN_SIZE = ACE2_HIDDEN_SIZE",
        "effective_value": 896,
        "value_source": "rtl/ace2_pkg.sv:ACE2_HIDDEN_SIZE",
    },
    {
        "declaration": "parameter integer LANES = ACE2_VECTOR_LANES",
        "effective_value": 16,
        "value_source": "rtl/ace2_pkg.sv:ACE2_VECTOR_LANES",
    },
    {
        "declaration": "parameter integer ACT_WIDTH = 8",
        "effective_value": 8,
        "value_source": "ace2_shell default",
    },
    {
        "declaration": "parameter integer GAIN_WIDTH = 16",
        "effective_value": 16,
        "value_source": "ace2_shell default",
    },
    {
        "declaration": "parameter integer ACC_WIDTH = 48",
        "effective_value": 48,
        "value_source": "ace2_shell default",
    },
    {
        "declaration": "parameter integer INV_RMS_FRAC = 30",
        "effective_value": 30,
        "value_source": "ace2_shell default",
    },
    {
        "declaration": "parameter integer GAIN_FRAC = 8",
        "effective_value": 8,
        "value_source": "ace2_shell default",
    },
    {
        "declaration": "parameter integer PROJ_MAC_LANES = 4",
        "effective_value": 4,
        "value_source": "ace2_shell default",
    },
    {
        "declaration": "parameter integer PROJ_ACC_WIDTH = 32",
        "effective_value": 32,
        "value_source": "ace2_shell default",
    },
    {
        "declaration": "parameter integer PROJ_M_MAX = 16",
        "effective_value": 16,
        "value_source": "ace2_shell default",
    },
    {
        "declaration": "parameter integer SRAM_BANKS = ACE2_SRAM_BANKS",
        "effective_value": 8,
        "value_source": "rtl/ace2_pkg.sv:ACE2_SRAM_BANKS",
    },
    {
        "declaration": "parameter integer SRAM_ADDR_WIDTH = ACE2_SRAM_ADDR_WIDTH",
        "effective_value": 12,
        "value_source": "rtl/ace2_pkg.sv:ACE2_SRAM_ADDR_WIDTH",
    },
]

PORTS = [
    ("input wire clk_i", 1),
    ("input wire rst_ni", 1),
    ("input wire csr_valid_i", 1),
    ("output wire csr_ready_o", 1),
    ("input wire csr_write_i", 1),
    ("input wire [31:0] csr_addr_i", 32),
    ("input wire [63:0] csr_wdata_i", 64),
    ("input wire [7:0] csr_wstrb_i", 8),
    ("output reg csr_rvalid_o", 1),
    ("input wire csr_rready_i", 1),
    ("output reg [63:0] csr_rdata_o", 64),
    ("output reg csr_error_o", 1),
    ("output wire irq_o", 1),
    ("input wire cmd_valid_i", 1),
    ("output wire cmd_ready_o", 1),
    ("input wire [7:0] cmd_opcode_i", 8),
    ("input wire [7:0] cmd_flags_i", 8),
    ("input wire [7:0] cmd_layer_id_i", 8),
    ("input wire [15:0] cmd_m_i", 16),
    ("input wire [15:0] cmd_n_i", 16),
    ("input wire [15:0] cmd_k_i", 16),
    ("input wire [15:0] cmd_sequence_position_i", 16),
    ("input wire [15:0] cmd_completion_tag_i", 16),
    ("input wire [63:0] cmd_src0_addr_i", 64),
    ("input wire [63:0] cmd_src1_addr_i", 64),
    ("input wire [63:0] cmd_dst_addr_i", 64),
    ("input wire [63:0] cmd_scale_addr_i", 64),
    ("input wire [63:0] cmd_scratch_addr_i", 64),
    ("output wire mem_req_valid_o", 1),
    ("input wire mem_req_ready_i", 1),
    ("output wire mem_req_write_o", 1),
    ("output wire [63:0] mem_req_addr_o", 64),
    ("output wire [15:0] mem_req_len_o", 16),
    ("output wire [7:0] mem_req_tag_o", 8),
    ("output wire mem_wvalid_o", 1),
    ("input wire mem_wready_i", 1),
    ("output wire [LANES*ACT_WIDTH-1:0] mem_wdata_o", 128),
    ("output wire [15:0] mem_wstrb_o", 16),
    ("output wire [7:0] mem_wtag_o", 8),
    ("input wire mem_rvalid_i", 1),
    ("output wire mem_rready_o", 1),
    ("input wire [LANES*ACT_WIDTH-1:0] mem_rdata_i", 128),
    ("input wire [7:0] mem_rtag_i", 8),
    ("input wire mem_rerror_i", 1),
    ("input wire mem_bvalid_i", 1),
    ("output wire mem_bready_o", 1),
    ("input wire [7:0] mem_btag_i", 8),
    ("input wire mem_berror_i", 1),
    ("output wire [SRAM_BANKS-1:0] sram_req_valid_o", 8),
    ("input wire [SRAM_BANKS-1:0] sram_req_ready_i", 8),
    ("output wire [SRAM_BANKS-1:0] sram_write_o", 8),
    ("output wire [SRAM_BANKS*SRAM_ADDR_WIDTH-1:0] sram_addr_o", 96),
    ("output wire [SRAM_BANKS*LANES*ACT_WIDTH-1:0] sram_wdata_o", 1024),
    ("output wire [SRAM_BANKS*16-1:0] sram_wstrb_o", 128),
    ("input wire [SRAM_BANKS*LANES*ACT_WIDTH-1:0] sram_rdata_i", 1024),
    ("input wire [SRAM_BANKS-1:0] sram_rvalid_i", 8),
    ("output wire busy_o", 1),
    ("output wire cmd_done_valid_o", 1),
    ("input wire cmd_done_ready_i", 1),
    ("output wire [15:0] cmd_done_tag_o", 16),
    ("output wire cmd_done_error_o", 1),
    ("output wire [ACC_WIDTH-1:0] cmd_done_sumsq_o", 48),
    ("output wire [INV_RMS_FRAC+1:0] cmd_done_inv_rms_q30_o", 32),
    ("output wire cmd_done_saturation_seen_o", 1),
]


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def file_record(path: Path) -> dict[str, Any]:
    return {"path": relative(path), **v3.file_record(path)}


def normalize_declaration(value: str) -> str:
    return " ".join(value.split())


def shell_interface(path: Path) -> tuple[list[str], list[str]]:
    source = path.read_text(encoding="utf-8")
    match = re.search(
        r"module\s+ace2_shell\s*#\((?P<parameters>.*?)\)\s*"
        r"\((?P<ports>.*?)\)\s*;",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"ace2_shell header not found in {relative(path)}")
    parameters = [
        normalize_declaration(item)
        for item in match.group("parameters").split(",")
        if item.strip()
    ]
    ports = [
        normalize_declaration(item)
        for item in match.group("ports").split(",")
        if item.strip()
    ]
    return parameters, ports


def source_records() -> dict[str, dict[str, Any]]:
    paths = [ROOT / path for path in v3.RTL_SOURCES]
    paths.extend(
        [
            ROOT / "rtl/ace2_pkg.sv",
            TB,
            Path(__file__).resolve(),
            ROOT / "constraints/ace2_rmsnorm_core.sdc",
        ]
    )
    return {relative(path): v3.file_record(path) for path in dict.fromkeys(paths)}


def identity(name: str, origin: Path, snapshot: Path) -> dict[str, Any]:
    return {
        "identity": name,
        "origin": file_record(origin),
        "immutable_snapshot": file_record(snapshot),
    }


def output_paths() -> dict[str, Any]:
    return {
        "interface_manifest": relative(MANIFEST),
        "contract": relative(CONTRACT),
        "baseline_source_snapshot": relative(BASELINE_SHELL),
        "candidate_source_snapshot": relative(CANDIDATE_SHELL),
        "baseline_measurement_directory": relative(BASELINE_OUTPUT),
        "baseline_result": relative(BASELINE_OUTPUT / "result.json"),
        "candidate_measurement_directory": relative(CANDIDATE_OUTPUT),
        "candidate_result": relative(CANDIDATE_OUTPUT / "result.json"),
        "comparison_result": relative(PUBLIC_RESULTS),
        "measurement_files": [
            "ace2_projection_k_cache_benchmark.vvp",
            "compile.stdout.log",
            "compile.stderr.log",
            "expected.hex",
            "warmup.stdout.log",
            "warmup.stderr.log",
            "measured-01.stdout.log",
            "measured-01.stderr.log",
            "measured-02.stdout.log",
            "measured-02.stderr.log",
            "measured-03.stdout.log",
            "measured-03.stderr.log",
            "result.json",
        ],
    }


def create_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "benchmark": NAME,
        "top_module": "ace2_shell",
        "interface_authority": (
            "This mission-local manifest binds both evaluated source snapshots; "
            "the stale design/BENCHMARK_INTERFACE.json source hash is not used."
        ),
        "parameters": PARAMETERS,
        "parameter_count": len(PARAMETERS),
        "ports": [
            {"declaration": declaration, "effective_width_bits": width}
            for declaration, width in PORTS
        ],
        "port_count": len(PORTS),
        "clock_semantics": {
            "port": "clk_i",
            "period_ns": 10,
            "half_period_ns": 5,
            "accepting_edge": "rising",
            "testbench_initial_value": 0,
        },
        "reset_semantics": {
            "port": "rst_ni",
            "polarity": "active_low",
            "rtl_assertion": "asynchronous",
            "initial_value": 0,
            "asserted_rising_edges": 5,
            "deassertion_edge": "falling edge after the fifth rising edge",
            "post_deassert_settle_rising_edges": 2,
            "mid_command_reset_tested": False,
        },
        "protocol_semantics": {
            "csr_enable": {
                "transfer": "rising edge with csr_valid_i && csr_ready_o",
                "write": {
                    "csr_write_i": 1,
                    "csr_addr_i": "0x00000018",
                    "csr_wdata_i": "0x0000000000000001",
                    "csr_wstrb_i": "0xff",
                },
                "source_rule": (
                    "assert on a falling edge after reset settling, hold through "
                    "the accepting rising edge, deassert on the following falling edge"
                ),
            },
            "command": {
                "transfer": "rising edge with cmd_valid_i && cmd_ready_o",
                "source_rule": (
                    "wait for cmd_ready_o, assert cmd_valid_i on a falling edge "
                    "with all descriptor fields stable through the accepting rising edge"
                ),
                "descriptor": {
                    "cmd_opcode_i": 1,
                    "cmd_flags_i": 0,
                    "cmd_layer_id_i": 0,
                    "cmd_m_i": 1,
                    "cmd_n_i": 128,
                    "cmd_k_i": 896,
                    "cmd_sequence_position_i": 0,
                    "cmd_completion_tag_i": "0x4b01",
                    "cmd_src0_addr_i": "0x0000000000001000",
                    "cmd_src1_addr_i": "0x0000000000002000",
                    "cmd_dst_addr_i": "0x0000000000011000",
                    "cmd_scale_addr_i": "0x0000000000010000",
                    "cmd_scratch_addr_i": "0x0000000000000000",
                },
                "accepted_descriptor_count": 1,
            },
            "completion": {
                "transfer": "rising edge with cmd_done_valid_o && cmd_done_ready_i",
                "sink_rule": (
                    "hold cmd_done_ready_i low until completion and output checks "
                    "finish, then assert it for the accepting rising edge"
                ),
                "required_tag": "0x4b01",
                "required_error_relation": (
                    "cmd_done_error_o == cmd_done_saturation_seen_o"
                ),
                "timeout_cycles_after_command_acceptance": 2_000_000,
            },
            "memory_request": {
                "transfer": "rising edge with mem_req_valid_o && mem_req_ready_i",
                "required_mem_req_len_o": 1,
                "ready_rule": "rst_ni && ((cycle_count % 17) != 3)",
                "one_request_may_be_outstanding_per_response_channel": True,
            },
            "memory_read_response": {
                "latency": "one cycle after accepted read request",
                "payload": "deterministic benchmark memory_beat(mem_req_addr_o)",
                "tag": "echo accepted mem_req_tag_o",
                "error": 0,
                "hold_rule": "hold mem_rvalid_i and payload until mem_rready_o",
            },
            "memory_write_data": {
                "transfer": "rising edge with mem_wvalid_o && mem_wready_i",
                "ready_rule": "rst_ni && ((cycle_count % 19) != 5)",
                "required_strobe": "0xffff",
                "required_tag": "match the pending write-request tag",
                "destination_range": "0x0000000000011000..0x000000000001107f",
            },
            "memory_write_response": {
                "latency": "assert after accepted matching write data",
                "tag": "echo accepted mem_wtag_o",
                "error": 0,
                "hold_rule": "hold mem_bvalid_i and tag until mem_bready_o",
            },
            "sram": {
                "sram_req_ready_i": "0xff",
                "sram_rvalid_i": "0x00",
                "sram_rdata_i": "1024'b0",
                "outputs_scored": False,
            },
        },
        "output_paths": output_paths(),
        "source_identities": {
            "baseline": identity(
                "frozen-v3-pre-cache-shell", V3_BASELINE_SHELL, BASELINE_SHELL
            ),
            "candidate": identity(
                "activation-and-weight-beat-cache-shell",
                LIVE_CANDIDATE_SHELL,
                CANDIDATE_SHELL,
            ),
        },
        "immutability": (
            "freeze and measurement require absent output paths; existing "
            "manifest, contract, snapshots, result directories, or comparison "
            "result are never overwritten"
        ),
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    expected_parameters = [item["declaration"] for item in PARAMETERS]
    expected_ports = [declaration for declaration, _ in PORTS]
    if manifest["parameter_count"] != 12 or manifest["port_count"] != 64:
        raise RuntimeError("interface manifest count mismatch")
    if [item["declaration"] for item in manifest["parameters"]] != expected_parameters:
        raise RuntimeError("interface manifest parameter declarations changed")
    if [item["declaration"] for item in manifest["ports"]] != expected_ports:
        raise RuntimeError("interface manifest port declarations changed")
    if manifest["output_paths"] != output_paths():
        raise RuntimeError("interface manifest output paths changed")
    for label, shell in (
        ("baseline", BASELINE_SHELL),
        ("candidate", CANDIDATE_SHELL),
    ):
        parameters, ports = shell_interface(shell)
        if parameters != expected_parameters or ports != expected_ports:
            raise RuntimeError(f"{label} shell does not match the mission interface")
        snapshot = manifest["source_identities"][label]["immutable_snapshot"]
        if file_record(shell) != snapshot:
            raise RuntimeError(f"{label} source identity changed")
    if shell_interface(BASELINE_SHELL) != shell_interface(CANDIDATE_SHELL):
        raise RuntimeError("baseline and candidate public interfaces differ")


def require_sha(path: Path, expected: str) -> None:
    actual = v3.sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"identity mismatch for {relative(path)}: expected {expected}, got {actual}"
        )


def freeze() -> None:
    immutable_paths = [
        MANIFEST,
        CONTRACT,
        BASELINE_SHELL,
        CANDIDATE_SHELL,
        BASELINE_OUTPUT,
        CANDIDATE_OUTPUT,
        PUBLIC_RESULTS,
    ]
    existing = [relative(path) for path in immutable_paths if path.exists()]
    if existing:
        raise RuntimeError(f"repair namespace is not fresh: {', '.join(existing)}")

    require_sha(V3_CONTRACT, EXPECTED_IDENTITIES["v3_contract"])
    require_sha(V3_RESULTS, EXPECTED_IDENTITIES["v3_results"])
    require_sha(V3_BASELINE_SHELL, EXPECTED_IDENTITIES["v3_baseline"])
    require_sha(LIVE_CANDIDATE_SHELL, EXPECTED_IDENTITIES["candidate"])
    require_sha(TB, EXPECTED_IDENTITIES["testbench"])

    BASELINE_SHELL.parent.mkdir(parents=True)
    CANDIDATE_SHELL.parent.mkdir(parents=True)
    shutil.copy2(V3_BASELINE_SHELL, BASELINE_SHELL)
    shutil.copy2(LIVE_CANDIDATE_SHELL, CANDIDATE_SHELL)

    manifest = create_manifest()
    validate_manifest(manifest)
    v3.write_atomic(MANIFEST, v3.canonical_bytes(manifest))

    expected = v3.reference_outputs()
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "name": NAME,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fixed_external_benchmark": False,
        "protected_or_sealed_inputs_used": False,
        "official_full_chat_invoked": False,
        "interface_manifest": file_record(MANIFEST),
        "frozen_sources": manifest["source_identities"],
        "predecessor_v3_unchanged": {
            "contract": file_record(V3_CONTRACT),
            "results": file_record(V3_RESULTS),
        },
        "workload": {
            "operator": "layer-0 K projection",
            "host_descriptor_commands": 1,
            "outputs_checked": v3.OUTPUTS,
            "expected_output": {
                "bytes": len(expected),
                "sha256": v3.sha256_bytes(expected),
            },
        },
        "score_policy": {
            "primary": "candidate simulator cycles <= 75% of frozen baseline",
            "correctness": (
                "all 128 RTL output bytes exactly equal the Python fixed-point reference"
            ),
            "accounting": (
                "host descriptors and memory writes unchanged; memory reads reduced"
            ),
            "wall_time": "median of three measured trials; reported but not gated",
        },
        "sources_and_constraint_hashes": source_records(),
        "limitations": [
            "This is a deterministic synthetic layer-0 K projection, not a full model or chat attempt.",
            "The benchmark measures the abstract streaming-memory boundary, not a physical DRAM controller.",
            "Activation-beat reuse is disabled for dynamic-sidecar projections; weight-beat reuse remains enabled.",
            "Full-shell 100 MHz closure remains limited by the pre-existing shared control-to-SILU path.",
        ],
    }
    v3.write_atomic(CONTRACT, v3.canonical_bytes(contract))
    print(
        "ACE2_PROJECTION_K_CACHE_INTERFACE_REPAIR_FREEZE_PASS "
        f"manifest_sha256={v3.sha256_file(MANIFEST)} "
        f"baseline_sha256={v3.sha256_file(BASELINE_SHELL)} "
        f"candidate_sha256={v3.sha256_file(CANDIDATE_SHELL)}"
    )


def read_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    if not CONTRACT.is_file() or not MANIFEST.is_file():
        raise RuntimeError("missing repair contract or interface manifest")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if file_record(MANIFEST) != contract["interface_manifest"]:
        raise RuntimeError("interface manifest changed after freeze")
    validate_manifest(manifest)
    for key, path in (("contract", V3_CONTRACT), ("results", V3_RESULTS)):
        if file_record(path) != contract["predecessor_v3_unchanged"][key]:
            raise RuntimeError(f"predecessor v3 {key} changed")
    if source_records() != contract["sources_and_constraint_hashes"]:
        raise RuntimeError("repair benchmark source or constraint changed")
    expected = v3.reference_outputs()
    if {
        "bytes": len(expected),
        "sha256": v3.sha256_bytes(expected),
    } != contract["workload"]["expected_output"]:
        raise RuntimeError("reference workload changed after freeze")
    return contract, manifest


def measure(label: str) -> None:
    contract, manifest = read_contract()
    shell = BASELINE_SHELL if label == "baseline" else CANDIDATE_SHELL
    output = BASELINE_OUTPUT if label == "baseline" else CANDIDATE_OUTPUT
    if output.exists():
        raise RuntimeError(f"immutable measurement namespace exists: {relative(output)}")
    output.mkdir(parents=True)

    expected_path = output / "expected.hex"
    expected_path.write_text(
        "".join(f"{value:02x}\n" for value in v3.reference_outputs()),
        encoding="ascii",
    )
    binary = v3.compile_benchmark(shell, output)
    trials: list[dict[str, Any]] = []
    for trial_index in range(v3.MEASURED_TRIALS + 1):
        start = time.perf_counter()
        completed = subprocess.run(
            ["vvp", str(binary), f"+EXPECTED={expected_path}"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        wall_seconds = time.perf_counter() - start
        trial = "warmup" if trial_index == 0 else f"measured-{trial_index:02d}"
        (output / f"{trial}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output / f"{trial}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        match = v3.PASS_RE.search(completed.stdout)
        if completed.returncode != 0 or match is None:
            raise RuntimeError(
                f"{label} {trial} failed:\n{completed.stdout}\n{completed.stderr}"
            )
        metrics = {key: int(value) for key, value in match.groupdict().items()}
        if metrics["outputs_checked"] != v3.OUTPUTS:
            raise RuntimeError("benchmark output coverage changed")
        if metrics["host_commands"] != 1:
            raise RuntimeError("benchmark descriptor count changed")
        trials.append(
            {
                "trial": trial,
                "measured": trial_index != 0,
                "wall_seconds": wall_seconds,
                **metrics,
            }
        )

    measured = [trial for trial in trials if trial["measured"]]
    deterministic_keys = [
        "host_commands",
        "outputs_checked",
        "simulator_cycles",
        "memory_read_requests",
        "memory_write_requests",
        "saturation",
    ]
    if any(
        {key: trial[key] for key in deterministic_keys}
        != {key: measured[0][key] for key in deterministic_keys}
        for trial in measured[1:]
    ):
        raise RuntimeError("cycle or command metrics were not deterministic")

    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "benchmark": NAME,
        "label": label,
        "status": "PASS_BIT_EXACT",
        "contract": file_record(CONTRACT),
        "interface_manifest": file_record(MANIFEST),
        "source_identity": manifest["source_identities"][label],
        "expected": file_record(expected_path),
        "binary": file_record(binary),
        "metrics": {
            **{key: measured[0][key] for key in deterministic_keys},
            "median_wall_seconds": statistics.median(
                trial["wall_seconds"] for trial in measured
            ),
        },
        "trials": trials,
    }
    v3.write_atomic(output / "result.json", v3.canonical_bytes(result))
    print(
        f"ACE2_PROJECTION_K_CACHE_INTERFACE_REPAIR_{label.upper()}_PASS "
        f"cycles={result['metrics']['simulator_cycles']} "
        f"reads={result['metrics']['memory_read_requests']} "
        f"wall_seconds={result['metrics']['median_wall_seconds']:.6f}"
    )


def compare() -> None:
    contract, manifest = read_contract()
    if PUBLIC_RESULTS.exists():
        raise RuntimeError(
            f"immutable comparison namespace exists: {relative(PUBLIC_RESULTS)}"
        )
    baseline_path = BASELINE_OUTPUT / "result.json"
    candidate_path = CANDIDATE_OUTPUT / "result.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if result["status"] != "PASS_BIT_EXACT":
            raise RuntimeError(f"{label} lacks exact reference agreement")
        if result["contract"] != file_record(CONTRACT):
            raise RuntimeError(f"{label} result uses a different contract")
        if result["interface_manifest"] != file_record(MANIFEST):
            raise RuntimeError(f"{label} result uses a different interface manifest")
        if result["source_identity"] != manifest["source_identities"][label]:
            raise RuntimeError(f"{label} result uses a different source identity")

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    cycle_ratio = (
        candidate_metrics["simulator_cycles"] / baseline_metrics["simulator_cycles"]
    )
    read_ratio = (
        candidate_metrics["memory_read_requests"]
        / baseline_metrics["memory_read_requests"]
    )
    passed = (
        baseline_metrics["host_commands"]
        == candidate_metrics["host_commands"]
        == 1
        and baseline_metrics["memory_write_requests"]
        == candidate_metrics["memory_write_requests"]
        and cycle_ratio <= 0.75
        and candidate_metrics["memory_read_requests"]
        < baseline_metrics["memory_read_requests"]
    )
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "benchmark": NAME,
        "status": "PASS" if passed else "FAIL",
        "contract": file_record(CONTRACT),
        "interface_manifest": file_record(MANIFEST),
        "source_identities": manifest["source_identities"],
        "measurement_results": {
            "baseline": file_record(baseline_path),
            "candidate": file_record(candidate_path),
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "improvement": {
            "simulator_cycles_percent": (1.0 - cycle_ratio) * 100.0,
            "memory_read_requests_percent": (1.0 - read_ratio) * 100.0,
            "median_wall_time_percent": (
                1.0
                - candidate_metrics["median_wall_seconds"]
                / baseline_metrics["median_wall_seconds"]
            )
            * 100.0,
        },
        "equivalence": {
            "rtl_reference": "PASS_BIT_EXACT",
            "bytes_compared_per_trial": v3.OUTPUTS,
            "public_interface": "IDENTICAL_12_PARAMETERS_64_PORTS",
            "host_descriptor_commands_unchanged": True,
            "memory_write_requests_unchanged": (
                baseline_metrics["memory_write_requests"]
                == candidate_metrics["memory_write_requests"]
            ),
        },
        "rtl_scope": (
            "One 128-bit activation beat register and one 128-bit W4 weight beat "
            "register in ace2_shell; reuse is internal to a projection output and "
            "does not change ports, descriptors, W4A8 arithmetic, or memory layout."
        ),
        "limitations": contract["limitations"],
    }
    v3.write_atomic(PUBLIC_RESULTS, v3.canonical_bytes(result))
    print(
        f"ACE2_PROJECTION_K_CACHE_INTERFACE_REPAIR_COMPARE_{result['status']} "
        f"cycle_improvement_percent={result['improvement']['simulator_cycles_percent']:.3f} "
        f"read_improvement_percent={result['improvement']['memory_read_requests_percent']:.3f} "
        f"wall_improvement_percent={result['improvement']['median_wall_time_percent']:.3f}"
    )
    if not passed:
        raise RuntimeError("candidate did not meet the repair score policy")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("freeze")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument(
        "--label", choices=("baseline", "candidate"), required=True
    )
    subparsers.add_parser("compare")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "measure":
        measure(args.label)
    else:
        compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
