#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = (
    ROOT
    / "evidence/verification/repair-full-qwen-final-lm-head-tile-boundary-v1"
)
BASELINE_SHELL = BASELINE_ROOT / "source_snapshot/rtl/ace2_shell.sv"
BASELINE_TB = (
    BASELINE_ROOT / "source_snapshot/verification/tb/ace2_shell_tb.sv"
)
BASELINE_VECTORS = (
    BASELINE_ROOT
    / "source_snapshot/verification/generated/projection_vectors.svh"
)
BASELINE_LOG = BASELINE_ROOT / "simulation/rtl_lm_head_sim.log"
BASELINE_SUMS = BASELINE_ROOT / "SHA256SUMS"
SUCCESSOR_SHELL = ROOT / "rtl/ace2_shell.sv"
SUCCESSOR_TB = ROOT / "verification/tb/ace2_shell_tb.sv"
SUCCESSOR_VECTORS = ROOT / "verification/generated/projection_vectors.svh"
MONITOR = ROOT / "verification/tb/ace2_lm_head_equiv_monitor.sv"

EXPECTED_BASELINE_HASHES = {
    BASELINE_SHELL: "9617340685a4245532d2eccf95c54f6775e7f6ea3be4ef6fb30cd94fa1cbcb04",
    BASELINE_TB: "0a086feb925d2301f3df83b7f5bc3a1ba49ba99705cb67d357ccc4244b198547",
    BASELINE_VECTORS: "0ae7f28674c1c3e40cc4bcf662ce67376320a9fba0b589bad0ac06716ba8fec7",
    BASELINE_LOG: "4c1198565dd41ae1005c367111d88c55d5edb92f85efb8d9d4bea2adc0537d48",
    BASELINE_SUMS: "7894b0931170f79141a306ed4be6ef1d41ced813beea70121d0d7062724b1777",
}

RTL_SOURCES = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_dynamic_scale32_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_dynamic_rope_head_core.sv",
    "rtl/ace2_fixed_q7_rope_score_core.sv",
    "rtl/ace2_relative_rope_score_fusion_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
]

ACT_BASE = 0x0010_0000
WEIGHT_BASE = 0x0020_0000
META_BASE = 0x0030_0000
CASES = 2
OUTPUTS = 32
HIDDEN_SIZE = 896
MAC_LANES = 4
GROUPS = HIDDEN_SIZE // MAC_LANES
ACT_BEATS = HIDDEN_SIZE // 16
WEIGHT_BEATS_PER_OUTPUT = (HIDDEN_SIZE // 2) // 16
BASELINE_COUNTS = {"40": 7168, "41": 0, "42": 7168, "43": 32}
SUCCESSOR_COUNTS = {"40": 1792, "41": 0, "42": 896, "43": 32}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tool_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    require(completed.returncode == 0, f"tool version failed: {' '.join(command)}")
    return completed.stdout.strip()


def module_header(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"module\s+ace2_shell\s*#\((.*?)\)\s*\((.*?)\);",
        text,
        flags=re.DOTALL,
    )
    require(match is not None, f"ace2_shell public header not found in {path}")
    header = match.group(0)
    header = re.sub(r"//[^\n]*", "", header)
    return re.sub(r"\s+", " ", header).strip()


def normalized_vectors(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(
        line
        for line in lines
        if line.strip()
        != "localparam integer PROJ_GROUPS_PER_INPUT_BEAT = 4;"
    )


def compile_and_run(
    *,
    name: str,
    shell: Path,
    testbench: Path,
    vector_dir: Path,
    output_dir: Path,
    iverilog: str,
    vvp: str,
) -> tuple[Path, Path]:
    sim_dir = output_dir / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    image = sim_dir / f"{name}.vvp"
    compile_log = output_dir / f"{name}.compile.log"
    trace_log = output_dir / f"{name}.trace.log"
    sources = [str(ROOT / source) for source in RTL_SOURCES]
    sources.extend([str(shell), str(testbench), str(MONITOR)])
    command = [
        iverilog,
        "-g2012",
        f"-I{ROOT / 'rtl'}",
        f"-I{vector_dir}",
        f"-I{testbench.parent}",
        f"-I{ROOT / 'verification/tb'}",
        "-s",
        "ace2_shell_tb",
        "-s",
        "ace2_lm_head_equiv_monitor",
        "-o",
        str(image),
        *sources,
    ]
    compiled = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    compile_log.write_text(compiled.stdout, encoding="utf-8")
    require(compiled.returncode == 0, f"{name} Icarus compilation failed")
    simulated = subprocess.run(
        [vvp, str(image), "+LM_HEAD_ONLY"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    trace_log.write_text(simulated.stdout, encoding="utf-8")
    require(simulated.returncode == 0, f"{name} LM-head simulation failed")
    require("ACE2_SHELL_LM_HEAD_TB_PASS" in simulated.stdout, f"{name} PASS missing")
    return compile_log, trace_log


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        key, separator, value = token.partition("=")
        require(bool(separator), f"malformed trace token: {token}")
        fields[key] = value
    return fields


def parse_trace(path: Path) -> dict[str, Any]:
    event_prefixes = {
        "EQUIV_CMD": "commands",
        "EQUIV_READ": "reads",
        "EQUIV_PAIR": "pairs",
        "EQUIV_LOGIT": "logits",
        "EQUIV_DONE": "done",
        "EQUIV_VIOLATION": "violations",
        "EQUIV_PROTOCOL": "protocol",
    }
    parsed: dict[str, Any] = {
        "commands": [],
        "reads": [],
        "pairs": [],
        "logits": [],
        "done": [],
        "violations": [],
        "protocol": [],
    }
    pass_line = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ACE2_SHELL_LM_HEAD_TB_PASS"):
            pass_line = line
        prefix = line.split(maxsplit=1)[0] if line else ""
        if prefix in event_prefixes:
            parsed[event_prefixes[prefix]].append(parse_fields(line))
    require(bool(pass_line), f"LM-head pass line missing from {path}")
    parsed["pass"] = parse_fields(pass_line)
    return parsed


def integer(field: str, base: int = 10) -> int:
    require(not re.search(r"[xXzZ]", field), f"unknown value in trace: {field}")
    return int(field, base)


def signed(value: int, width: int) -> int:
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def compare_protocol(
    baseline: dict[str, Any],
    successor: dict[str, Any],
) -> dict[str, Any]:
    expected_tags = [f"{tag:04x}" for tag in range(0x6200, 0x6206)]
    for name, trace in (("baseline", baseline), ("successor", successor)):
        require(not trace["violations"], f"{name} protocol monitor reported violations")
        require(len(trace["protocol"]) == 1, f"{name} protocol summary count")
        protocol = trace["protocol"][0]
        require(integer(protocol["accepted"]) == 6, f"{name} accepted command count")
        require(integer(protocol["completed"]) == 6, f"{name} completion count")
        require(integer(protocol["violations"]) == 0, f"{name} protocol violation count")
        require(integer(protocol["in_flight"]) == 0, f"{name} command left in flight")
        require(
            [event["tag"] for event in trace["commands"]] == expected_tags,
            f"{name} command tags or ordering",
        )
        require(
            [event["tag"] for event in trace["done"]] == expected_tags,
            f"{name} completion tags or ordering",
        )
        for event in trace["commands"]:
            for key in ("layer", "m", "n", "k", "busy", "ready"):
                integer(event[key])
            for key in (
                "opcode",
                "flags",
                "src0",
                "src1",
                "dst",
                "scale",
                "scratch",
                "tag",
            ):
                integer(event[key], 16)
        for event in trace["done"]:
            for key in ("error", "saturation", "busy", "ready"):
                integer(event[key])
            for key in ("tag", "sumsq", "inv_rms"):
                integer(event[key], 16)

    command_fields = [
        "opcode",
        "flags",
        "layer",
        "m",
        "n",
        "k",
        "src0",
        "src1",
        "dst",
        "scale",
        "scratch",
        "tag",
        "busy",
        "ready",
    ]
    done_fields = [
        "tag",
        "error",
        "saturation",
        "sumsq",
        "inv_rms",
        "busy",
        "ready",
    ]
    require(
        [[event[key] for key in command_fields] for event in baseline["commands"]]
        == [[event[key] for key in command_fields] for event in successor["commands"]],
        "public command acceptance semantics differ",
    )
    require(
        [[event[key] for key in done_fields] for event in baseline["done"]]
        == [[event[key] for key in done_fields] for event in successor["done"]],
        "public completion semantics differ",
    )

    cycle_results: dict[str, Any] = {}
    baseline_cycles = {
        event["tag"]: integer(done["cycle"]) - integer(event["cycle"])
        for event, done in zip(baseline["commands"], baseline["done"], strict=True)
    }
    successor_cycles = {
        event["tag"]: integer(done["cycle"]) - integer(event["cycle"])
        for event, done in zip(successor["commands"], successor["done"], strict=True)
    }
    for tag in expected_tags:
        cycle_results[tag] = {
            "baseline": baseline_cycles[tag],
            "successor": successor_cycles[tag],
            "delta": successor_cycles[tag] - baseline_cycles[tag],
        }
    return {
        "command_count": 6,
        "completion_count": 6,
        "ordered_tags": expected_tags,
        "events_identical_excluding_cycle": True,
        "protocol_violations": 0,
        "cycles_by_tag": cycle_results,
    }


def expected_read_addresses(kind: str, case_id: int) -> dict[str, list[int]]:
    del case_id
    if kind == "baseline":
        activation = [
            ACT_BASE + (group // 4) * 16
            for _output in range(OUTPUTS)
            for group in range(GROUPS)
        ]
        weight = [
            WEIGHT_BASE + output * 448 + (group // 8) * 16
            for output in range(OUTPUTS)
            for group in range(GROUPS)
        ]
    else:
        activation = [
            ACT_BASE + beat * 16
            for _output in range(OUTPUTS)
            for beat in range(ACT_BEATS)
        ]
        weight = [
            WEIGHT_BASE + output * 448 + beat * 16
            for output in range(OUTPUTS)
            for beat in range(WEIGHT_BEATS_PER_OUTPUT)
        ]
    metadata = [META_BASE + output * 16 for output in range(OUTPUTS)]
    return {"40": activation, "41": [], "42": weight, "43": metadata}


def check_reads(name: str, trace: dict[str, Any]) -> dict[str, Any]:
    expected_counts = BASELINE_COUNTS if name == "baseline" else SUCCESSOR_COUNTS
    result: dict[str, Any] = {}
    for case_id in range(CASES):
        by_tag: dict[str, list[int]] = {"40": [], "41": [], "42": [], "43": []}
        for event in trace["reads"]:
            event_case = integer(event["case"])
            integer(event["tag"], 16)
            event_address = integer(event["addr"], 16)
            if event_case == case_id:
                by_tag[event["tag"]].append(event_address)
        expected = expected_read_addresses(name, case_id)
        for tag, count in expected_counts.items():
            require(len(by_tag[tag]) == count, f"{name} case {case_id} tag {tag} count")
            require(
                by_tag[tag] == expected[tag],
                f"{name} case {case_id} tag {tag} address order",
            )
        result[str(case_id)] = {
            "activation_reads": len(by_tag["40"]) + len(by_tag["41"]),
            "weight_reads": len(by_tag["42"]),
            "metadata_reads": len(by_tag["43"]),
            "activation_span": [hex(min(by_tag["40"])), hex(max(by_tag["40"]))],
            "weight_span": [hex(min(by_tag["42"])), hex(max(by_tag["42"]))],
            "metadata_span": [hex(min(by_tag["43"])), hex(max(by_tag["43"]))],
            "address_order_exact": True,
        }
    return result


def compare_pairs(
    baseline: dict[str, Any],
    successor: dict[str, Any],
) -> dict[str, Any]:
    require(len(baseline["pairs"]) == CASES * OUTPUTS * GROUPS, "baseline pair count")
    require(len(successor["pairs"]) == len(baseline["pairs"]), "successor pair count")
    canonical = hashlib.sha256()
    activation_keys: set[tuple[int, int, int]] = set()
    weight_nibbles: set[tuple[int, int]] = set()
    weight_byte_positions: Counter[tuple[int, int]] = Counter()
    signed_lane_min = 127
    signed_lane_max = -128
    signed_nibble_min = 7
    signed_nibble_max = -8

    for ordinal, (old, new) in enumerate(
        zip(baseline["pairs"], successor["pairs"], strict=True)
    ):
        expected_case = ordinal // (OUTPUTS * GROUPS)
        within_case = ordinal % (OUTPUTS * GROUPS)
        expected_out = within_case // GROUPS
        expected_group = within_case % GROUPS
        expected_key = (expected_case, 0, expected_out, expected_group)
        old_key = (
            integer(old["case"]),
            integer(old["row"]),
            integer(old["out"]),
            integer(old["group"]),
        )
        new_key = (
            integer(new["case"]),
            integer(new["row"]),
            integer(new["out"]),
            integer(new["group"]),
        )
        require(old_key == expected_key, f"baseline logical pair order at {ordinal}")
        require(new_key == expected_key, f"successor logical pair order at {ordinal}")
        require(old["act"] == new["act"], f"activation lane packing at {ordinal}")
        require(old["weight"] == new["weight"], f"weight nibble packing at {ordinal}")

        act_raw = integer(old["act"], 16)
        weight_raw = integer(old["weight"], 16)
        act_values = [signed((act_raw >> (lane * 8)) & 0xFF, 8) for lane in range(4)]
        weight_values = [
            signed((weight_raw >> (lane * 4)) & 0xF, 4) for lane in range(4)
        ]
        signed_lane_min = min(signed_lane_min, *act_values)
        signed_lane_max = max(signed_lane_max, *act_values)
        signed_nibble_min = min(signed_nibble_min, *weight_values)
        signed_nibble_max = max(signed_nibble_max, *weight_values)

        for lane, value in enumerate(act_values):
            address = ACT_BASE + expected_group * MAC_LANES + lane
            key = (expected_case, expected_out, address)
            require(key not in activation_keys, f"duplicate activation logical byte {key}")
            activation_keys.add(key)
            canonical.update(
                f"A,{expected_case},{expected_out},{address:016x},{value}\n".encode()
            )
        for lane, value in enumerate(weight_values):
            nibble_index = expected_out * HIDDEN_SIZE + expected_group * MAC_LANES + lane
            key = (expected_case, nibble_index)
            require(key not in weight_nibbles, f"duplicate weight logical nibble {key}")
            weight_nibbles.add(key)
            byte_address = WEIGHT_BASE + nibble_index // 2
            nibble_position = nibble_index & 1
            weight_byte_positions[(expected_case, byte_address)] += 1
            canonical.update(
                f"W,{expected_case},{nibble_index},{byte_address:016x},"
                f"{nibble_position},{value}\n".encode()
            )

    require(
        len(activation_keys) == CASES * OUTPUTS * HIDDEN_SIZE,
        "activation logical coverage",
    )
    require(
        len(weight_nibbles) == CASES * OUTPUTS * HIDDEN_SIZE,
        "weight logical nibble coverage",
    )
    require(
        len(weight_byte_positions) == CASES * OUTPUTS * HIDDEN_SIZE // 2,
        "weight logical byte coverage",
    )
    require(
        set(weight_byte_positions.values()) == {2},
        "weight bytes do not contain exactly low/high nibbles",
    )
    return {
        "pair_count": len(baseline["pairs"]),
        "activation_logical_byte_occurrences": len(activation_keys),
        "activation_unique_bytes_per_output": HIDDEN_SIZE,
        "weight_logical_nibble_occurrences": len(weight_nibbles),
        "weight_logical_byte_occurrences": len(weight_byte_positions),
        "weight_span_bytes_per_case": OUTPUTS * HIDDEN_SIZE // 2,
        "no_gaps_or_duplicates": True,
        "lane_order": "lane 0 is bits [7:0], then ascending little-endian byte lanes",
        "nibble_order": "input channel even is low nibble, odd is high nibble",
        "signed_activation_range_observed": [signed_lane_min, signed_lane_max],
        "signed_weight_range_observed": [signed_nibble_min, signed_nibble_max],
        "canonical_signed_stream_sha256": canonical.hexdigest(),
    }


def compare_logits(
    baseline: dict[str, Any],
    successor: dict[str, Any],
) -> dict[str, Any]:
    require(len(baseline["logits"]) == CASES * OUTPUTS, "baseline logit count")
    require(len(successor["logits"]) == len(baseline["logits"]), "successor logit count")
    result: dict[str, Any] = {}
    for case_id in range(CASES):
        old_case = [event for event in baseline["logits"] if integer(event["case"]) == case_id]
        new_case = [event for event in successor["logits"] if integer(event["case"]) == case_id]
        require(len(old_case) == OUTPUTS, f"baseline case {case_id} logit count")
        require(len(new_case) == OUTPUTS, f"successor case {case_id} logit count")
        old_values: list[int] = []
        new_values: list[int] = []
        for output, (old, new) in enumerate(zip(old_case, new_case, strict=True)):
            require(integer(old["out"]) == output, "baseline logit order")
            require(integer(new["out"]) == output, "successor logit order")
            integer(old["acc"], 16)
            integer(new["acc"], 16)
            integer(old["saturation"])
            integer(new["saturation"])
            require(
                [old[key] for key in ("data", "acc", "saturation")]
                == [new[key] for key in ("data", "acc", "saturation")],
                f"logit or accumulator mismatch case {case_id} output {output}",
            )
            old_values.append(signed(integer(old["data"], 16), 8))
            new_values.append(signed(integer(new["data"], 16), 8))
        require(old_values == new_values, f"case {case_id} checked logits")
        selected = max(range(OUTPUTS), key=lambda index: (old_values[index], -index))
        digest = hashlib.sha256(bytes(value & 0xFF for value in old_values)).hexdigest()
        result[str(case_id)] = {
            "checked_logits": OUTPUTS,
            "logit_s8_sha256": digest,
            "selected_tile_index": selected,
            "selected_logit_s8": old_values[selected],
        }
    return {
        "cases": result,
        "all_available_checked_logits_identical": True,
        "selected_tile_argmax_identical": True,
        "full_vocabulary_token_claim": False,
    }


def write_sums(output_dir: Path) -> Path:
    sums_path = output_dir / "SHA256SUMS"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != sums_path
    )
    lines = [f"{sha256(path)}  {path.relative_to(ROOT)}" for path in files]
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sums_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "evidence/verification/"
            "lm-head-descriptor-traffic-equivalence-v1/engineer-repair-0005"
        ),
    )
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    require(not output_dir.exists(), f"refusing to overwrite evidence: {output_dir}")
    output_dir.mkdir(parents=True)

    for path, expected in EXPECTED_BASELINE_HASHES.items():
        require(path.is_file(), f"missing accepted baseline artifact: {path}")
        require(sha256(path) == expected, f"accepted baseline hash mismatch: {path}")
    require(
        normalized_vectors(BASELINE_VECTORS) == normalized_vectors(SUCCESSOR_VECTORS),
        "baseline and successor projection vectors differ beyond a derived localparam",
    )
    baseline_header = module_header(BASELINE_SHELL)
    successor_header = module_header(SUCCESSOR_SHELL)
    require(baseline_header == successor_header, "public ace2_shell contract differs")

    input_paths = [
        *EXPECTED_BASELINE_HASHES,
        SUCCESSOR_SHELL,
        SUCCESSOR_TB,
        SUCCESSOR_VECTORS,
        MONITOR,
        Path(__file__).resolve(),
        *(ROOT / source for source in RTL_SOURCES),
        ROOT / "rtl/generated/ace2_model_parameters.svh",
    ]
    input_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in input_paths
    }

    compile_and_run(
        name="baseline",
        shell=BASELINE_SHELL,
        testbench=BASELINE_TB,
        vector_dir=BASELINE_VECTORS.parent,
        output_dir=output_dir,
        iverilog=args.iverilog,
        vvp=args.vvp,
    )
    compile_and_run(
        name="successor",
        shell=SUCCESSOR_SHELL,
        testbench=SUCCESSOR_TB,
        vector_dir=SUCCESSOR_VECTORS.parent,
        output_dir=output_dir,
        iverilog=args.iverilog,
        vvp=args.vvp,
    )

    baseline = parse_trace(output_dir / "baseline.trace.log")
    successor = parse_trace(output_dir / "successor.trace.log")
    protocol = compare_protocol(baseline, successor)
    physical_reads = {
        "baseline": check_reads("baseline", baseline),
        "successor": check_reads("successor", successor),
    }
    logical_streams = compare_pairs(baseline, successor)
    logits = compare_logits(baseline, successor)

    baseline_cycles = integer(baseline["pass"]["cycles"])
    successor_cycles = integer(successor["pass"]["cycles"])
    success_tag_cycles = protocol["cycles_by_tag"]
    require(
        sum(success_tag_cycles[tag]["baseline"] for tag in ("6200", "6201"))
        == baseline_cycles,
        "baseline measured success-cycle total",
    )
    require(
        sum(success_tag_cycles[tag]["successor"] for tag in ("6200", "6201"))
        == successor_cycles,
        "successor measured success-cycle total",
    )
    baseline_primary_reads = BASELINE_COUNTS["40"] + BASELINE_COUNTS["42"]
    successor_primary_reads = SUCCESSOR_COUNTS["40"] + SUCCESSOR_COUNTS["42"]
    baseline_total_reads = baseline_primary_reads + BASELINE_COUNTS["43"]
    successor_total_reads = successor_primary_reads + SUCCESSOR_COUNTS["43"]

    result = {
        "schema_version": 1,
        "mission": "bounded descriptor-bound 0.5B LM-head traffic equivalence",
        "status": "PASS_AWAITING_INDEPENDENT_REVIEW",
        "publication_authorized": False,
        "official_attempt": False,
        "sealed_evidence_modified": False,
        "synthesis_or_ppa_executed": False,
        "stage_2_executed": False,
        "frozen_inputs": {
            "input_sha256": input_hashes,
            "baseline_tree_sha256_from_accepted_contract": "35da3c7ca116eb3031b9d86cdbdda3b6faab5369d694f438aff116c08dad1947",
            "baseline_public_shell_header_sha256": hashlib.sha256(
                baseline_header.encode()
            ).hexdigest(),
            "successor_public_shell_header_sha256": hashlib.sha256(
                successor_header.encode()
            ).hexdigest(),
            "visible_vector_payloads_identical": True,
            "vector_file_drift": (
                "successor adds only the derived "
                "PROJ_GROUPS_PER_INPUT_BEAT=4 localparam"
            ),
            "hidden_harness_or_golden_outputs_accessed": False,
        },
        "tool_versions": {
            "python": platform.python_version(),
            "iverilog": tool_version([args.iverilog, "-V"]),
            "vvp": tool_version([args.vvp, "-V"]),
        },
        "score_policy": {
            "all_invariants_required": True,
            "mismatch_tolerance": 0,
            "x_or_z_tolerance": 0,
        },
        "public_shell": {
            "module_header_identical": True,
            "protocol": protocol,
        },
        "physical_reads_per_case": physical_reads,
        "logical_streams": logical_streams,
        "observable_outputs": logits,
        "measured_delta": {
            "baseline_activation_reads": BASELINE_COUNTS["40"],
            "successor_activation_reads": SUCCESSOR_COUNTS["40"],
            "baseline_weight_reads": BASELINE_COUNTS["42"],
            "successor_weight_reads": SUCCESSOR_COUNTS["42"],
            "baseline_primary_reads": baseline_primary_reads,
            "successor_primary_reads": successor_primary_reads,
            "primary_read_delta": successor_primary_reads - baseline_primary_reads,
            "primary_read_reduction_percent": (
                (baseline_primary_reads - successor_primary_reads)
                * 100.0
                / baseline_primary_reads
            ),
            "baseline_total_reads_including_metadata": baseline_total_reads,
            "successor_total_reads_including_metadata": successor_total_reads,
            "total_read_delta": successor_total_reads - baseline_total_reads,
            "baseline_success_cycles": baseline_cycles,
            "successor_success_cycles": successor_cycles,
            "success_cycle_delta": successor_cycles - baseline_cycles,
            "success_cycle_reduction_percent": (
                (baseline_cycles - successor_cycles) * 100.0 / baseline_cycles
            ),
        },
        "limitations": [
            "The focused public harness checks a 32-logit LM-head tile, not the full vocabulary.",
            "Selected indices are argmax results over the 32 checked logits; no full-vocabulary token claim is made.",
            "Independent Reviewer acceptance is still required before bounded publication.",
        ],
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sums_path = write_sums(output_dir)
    summary = {
        "status": result["status"],
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": sha256(result_path),
        "sha256sums": str(sums_path.relative_to(ROOT)),
        "sha256sums_sha256": sha256(sums_path),
        "baseline_cycles": baseline_cycles,
        "successor_cycles": successor_cycles,
        "baseline_primary_reads": baseline_primary_reads,
        "successor_primary_reads": successor_primary_reads,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
