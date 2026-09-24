#!/usr/bin/env python3
"""Replay the first live chat residual boundary through common-domain RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_attention_compose_reference import AttentionComposePhaseReplay
from tools.ace2_chat_demo import (
    DYNAMIC_SCALE32_FLAG,
    PACKAGE_V2_DYNAMIC_RECORD,
    PINNED_EMBEDDING_OFFSET,
    _PositionReferenceMemory,
    _dynamic_rms_output_sidecar,
    _pack_int8_bytes,
    _position_expected_payload,
    quantized_embedding,
    read_ace2rt2_package_metadata,
    reconstruct_ace2rt2_schedule,
)
from tools.ace2_down_projection_residual_fusion_reference import fuse_lane
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    scale32_ratio,
)


DEFAULT_PACKAGE = (
    ROOT
    / "build/ace2_chat_demo/hello-world-empty-system-max8-20260805"
    / "runtime_package.bin"
)
DEFAULT_OUTPUT = (
    ROOT
    / "build/ace2_chat_diagnostics"
    / "hello-world-first-attention-residual-rtl-boundary-20260805.json"
)
FROZEN_SCALES = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1"
    / "paired-smoke-20260801-v1/derived_scales.json"
)
CORE_RTL = ROOT / "rtl/ace2_down_projection_residual_fusion_core.sv"
TESTBENCH = ROOT / "verification/tb/ace2_chat_residual_boundary_tb.sv"
REFERENCE = ROOT / "tools/ace2_down_projection_residual_fusion_reference.py"
MAKEFILE = ROOT / "Makefile"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def first_nonempty_line(payload: str) -> str:
    return next((line.strip() for line in payload.splitlines() if line.strip()), "")


def write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def signed_int8(raw: bytes) -> list[int]:
    return np.frombuffer(raw, dtype=np.int8).astype(int).tolist()


def int8_bytes(values: list[int]) -> bytes:
    return bytes(value & 0xFF for value in values)


def int8_stats(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.int16)
    return {
        "count": len(values),
        "min": int(array.min()),
        "max": int(array.max()),
        "zero_count": int(np.count_nonzero(array == 0)),
        "positive_saturation_count": int(np.count_nonzero(array == 127)),
        "negative_saturation_count": int(np.count_nonzero(array == -128)),
        "sha256": sha256_bytes(int8_bytes(values)),
    }


def residual_scales(
    scales: dict[str, Any], command: dict[str, Any]
) -> tuple[float, float, float]:
    layer = int(command["layer_id"])
    if command["operator"] != "attention_residual_add":
        raise RuntimeError("this replay requires an attention residual command")
    lhs_name = (
        "model.layers.0.input_layernorm.input"
        if layer == 0
        else f"model.layers.{layer - 1}.post_mlp_residual"
    )
    return (
        float(scales["operators"][lhs_name]["scale"]),
        float(
            scales["linears"][f"model.layers.{layer}.self_attn.o_proj"][
                "output_scale"
            ]
        ),
        float(
            scales["operators"][f"model.layers.{layer}.post_attention_residual"][
                "scale"
            ]
        ),
    )


def extract_first_boundary(
    package_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[int], list[int]]:
    package = read_ace2rt2_package_metadata(package_path)
    reconstruction = reconstruct_ace2rt2_schedule(package_path, package)
    commands = reconstruction["commands"]
    target = next(
        command
        for command in commands
        if command["operator"] == "attention_residual_add"
    )
    if int(target["token_step"]) != 0 or int(target["layer_id"]) != 0:
        raise RuntimeError("first attention residual boundary moved from position 0 layer 0")

    package_raw = package_path.read_bytes()
    sidecar_stop = int(package["command_records_offset"])
    sidecar_start = sidecar_stop - (
        int(package["dynamic_scale32_sidecar_count"])
        * PACKAGE_V2_DYNAMIC_RECORD.size
    )
    prompt_sidecars = [
        PACKAGE_V2_DYNAMIC_RECORD.unpack_from(package_raw, offset)
        for offset in range(
            sidecar_start,
            sidecar_stop,
            PACKAGE_V2_DYNAMIC_RECORD.size,
        )
    ]

    compose_states: dict[tuple[int, int, int], AttentionComposePhaseReplay] = {}
    with accepted_runtime.IMAGE.open("rb") as image_handle:
        with mmap.mmap(image_handle.fileno(), 0, access=mmap.ACCESS_READ) as image_map:
            memory = _PositionReferenceMemory(image_map)
            first_command = commands[0]
            scale_record = memory.read(int(first_command["scale_addr"]), 1808)
            embedding_scale = np.float32(struct.unpack_from("<d", scale_record, 1800)[0])
            embedding = quantized_embedding(
                int(package["prompt_tokens"][0]),
                PINNED_EMBEDDING_OFFSET,
                embedding_scale,
            )
            memory.write(int(first_command["src0_addr"]), _pack_int8_bytes(embedding))
            if int(first_command["flags"]) & DYNAMIC_SCALE32_FLAG:
                payload_addr, host_plan_sidecar = prompt_sidecars[0]
                if int(payload_addr) != int(first_command["src0_addr"]):
                    raise RuntimeError("prompt sidecar payload address differs")
                memory.write(int(payload_addr) - 64, host_plan_sidecar)

            for command in commands:
                if int(command["ordinal"]) == int(target["ordinal"]):
                    lhs = signed_int8(
                        memory.read(int(command["src0_addr"]), int(command["n"]))
                    )
                    rhs = signed_int8(
                        memory.read(int(command["src1_addr"]), int(command["n"]))
                    )
                    return package, {
                        "schedule_sha256": reconstruction["schedule_sha256"],
                        "command": command,
                    }, lhs, rhs

                payload, _saturation, expected_base = _position_expected_payload(
                    command, memory, compose_states
                )
                if payload:
                    memory.write(expected_base, payload)
                if (
                    command["operator"] == "input_rmsnorm"
                    and int(command["flags"]) & DYNAMIC_SCALE32_FLAG
                ):
                    memory.write(
                        expected_base - 64,
                        _dynamic_rms_output_sidecar(
                            command,
                            memory.read(int(command["src0_addr"]) - 64, 64),
                        ),
                    )
    raise RuntimeError("first attention residual command was not reached")


def clamp_s8(value: int) -> int:
    return max(-128, min(127, value))


def pack_vector(
    lhs: int,
    rhs: int,
    lhs_scale32: int,
    rhs_scale32: int,
    destination_scale32: int,
    result: Any,
) -> int:
    return (
        (lhs & 0xFF)
        | ((rhs & 0xFF) << 8)
        | (lhs_scale32 << 16)
        | (rhs_scale32 << 48)
        | (destination_scale32 << 80)
        | ((result.output_s8 & 0xFF) << 112)
        | ((result.numerator_s96 & ((1 << 96) - 1)) << 120)
        | (result.denominator_u64 << 216)
        | ((result.common_exponent & 0xFF) << 280)
        | (int(result.positive_saturation) << 288)
        | (int(result.negative_saturation) << 289)
    )


def run_command(argv: list[str], *, cwd: Path) -> tuple[str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={"PATH": str(Path(sys.executable).parent) + ":/usr/local/bin:/usr/bin:/bin"},
    )
    duration = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}: {' '.join(argv)}\n"
            + completed.stdout
        )
    return completed.stdout, duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build/ace2_chat_residual_boundary",
    )
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    args = parser.parse_args()

    started = time.perf_counter()
    package_path = args.package.resolve()
    output_path = args.output.resolve()
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    package, boundary, lhs, rhs = extract_first_boundary(package_path)
    command = boundary["command"]
    if len(lhs) != 896 or len(rhs) != 896:
        raise RuntimeError("chat residual boundary width differs from 896")

    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))
    lhs_scale, rhs_scale, destination_scale = residual_scales(scales, command)
    lhs_scale32 = ceil_scale32_from_float(lhs_scale)
    rhs_scale32 = ceil_scale32_from_float(rhs_scale)
    destination_scale32 = ceil_scale32_from_float(destination_scale)

    exact_results = [
        fuse_lane(
            lhs_value,
            rhs_value,
            lhs_scale32,
            rhs_scale32,
            destination_scale32,
        )
        for lhs_value, rhs_value in zip(lhs, rhs, strict=True)
    ]
    exact_outputs = [result.output_s8 for result in exact_results]
    direct_outputs = [clamp_s8(a + b) for a, b in zip(lhs, rhs, strict=True)]
    per_input_outputs = [
        clamp_s8(
            fuse_lane(
                a,
                0,
                lhs_scale32,
                destination_scale32,
                destination_scale32,
            ).rounded_integer
            + fuse_lane(
                b,
                0,
                rhs_scale32,
                destination_scale32,
                destination_scale32,
            ).rounded_integer
        )
        for a, b in zip(lhs, rhs, strict=True)
    ]
    float_fused_outputs = np.clip(
        np.rint(
            (
                np.asarray(lhs, dtype=np.float64) * lhs_scale
                + np.asarray(rhs, dtype=np.float64) * rhs_scale
            )
            / destination_scale
        ),
        -128,
        127,
    ).astype(np.int16).astype(int).tolist()

    vector_path = build_dir / "first_attention_residual_scale32_vectors.mem"
    vector_lines = [
        f"{pack_vector(a, b, lhs_scale32, rhs_scale32, destination_scale32, result):073x}"
        for a, b, result in zip(lhs, rhs, exact_results, strict=True)
    ]
    write_atomic(vector_path, "\n".join(vector_lines) + "\n")

    executable = build_dir / "ace2_chat_residual_boundary_tb.vvp"
    compile_argv = [
        args.iverilog,
        "-g2012",
        "-s",
        "ace2_chat_residual_boundary_tb",
        "-o",
        str(executable),
        str(CORE_RTL),
        str(TESTBENCH),
    ]
    compile_stdout, compile_seconds = run_command(compile_argv, cwd=ROOT)
    simulate_argv = [args.vvp, str(executable), f"+VECTORS={vector_path}"]
    simulate_stdout, simulate_seconds = run_command(simulate_argv, cwd=ROOT)
    iverilog_version_stdout, _ = run_command([args.iverilog, "-V"], cwd=ROOT)
    vvp_version_stdout, _ = run_command([args.vvp, "-V"], cwd=ROOT)
    pass_line = next(
        (
            line
            for line in simulate_stdout.splitlines()
            if line.startswith("ACE2_CHAT_RESIDUAL_BOUNDARY_RTL_PASS")
        ),
        None,
    )
    if pass_line is None:
        raise RuntimeError("simulator did not emit the residual-boundary PASS marker")

    compile_log = build_dir / "iverilog.log"
    simulate_log = build_dir / "vvp.log"
    write_atomic(compile_log, compile_stdout)
    write_atomic(simulate_log, simulate_stdout)

    record = {
        "schema_version": 1,
        "status": "PASS_RTL_REFERENCE_AGREEMENT",
        "classification": (
            "first_live_chat_command_boundary_only_not_end_to_end_accelerator_completion"
        ),
        "package": artifact(package_path),
        "schedule_sha256": boundary["schedule_sha256"],
        "prompt_token_count": int(package["prompt_token_count"]),
        "max_new_tokens": int(package["max_new_tokens"]),
        "boundary": {
            "command_ordinal": int(command["ordinal"]),
            "position": int(command["token_step"]),
            "layer_id": int(command["layer_id"]),
            "operator": command["operator"],
            "lanes": int(command["n"]),
            "src0_addr": f"0x{int(command['src0_addr']):016x}",
            "src1_addr": f"0x{int(command['src1_addr']):016x}",
            "dst_addr": f"0x{int(command['dst_addr']):016x}",
            "lhs": int8_stats(lhs),
            "rhs": int8_stats(rhs),
            "direct_add": int8_stats(direct_outputs),
            "common_domain_one_round": int8_stats(exact_outputs),
            "per_input_requantize_then_add": int8_stats(per_input_outputs),
            "frozen_float_one_round": int8_stats(float_fused_outputs),
            "direct_vs_common_domain_differing_lanes": sum(
                a != b for a, b in zip(direct_outputs, exact_outputs, strict=True)
            ),
            "per_input_vs_one_round_differing_lanes": sum(
                a != b for a, b in zip(per_input_outputs, exact_outputs, strict=True)
            ),
            "frozen_float_vs_scale32_differing_lanes": sum(
                a != b
                for a, b in zip(float_fused_outputs, exact_outputs, strict=True)
            ),
        },
        "scale_contract": {
            "encoding": "ceil_to_normalized_scale32",
            "lhs": {
                "frozen_float": lhs_scale,
                "packed_u32": lhs_scale32,
                "packed_hex": f"0x{lhs_scale32:08x}",
                "exact_ratio": list(scale32_ratio(lhs_scale32)),
            },
            "rhs": {
                "frozen_float": rhs_scale,
                "packed_u32": rhs_scale32,
                "packed_hex": f"0x{rhs_scale32:08x}",
                "exact_ratio": list(scale32_ratio(rhs_scale32)),
            },
            "destination": {
                "frozen_float": destination_scale,
                "packed_u32": destination_scale32,
                "packed_hex": f"0x{destination_scale32:08x}",
                "exact_ratio": list(scale32_ratio(destination_scale32)),
            },
        },
        "rtl": {
            "status": "PASS_BIT_EXACT_896_OF_896",
            "pass_marker": pass_line,
            "valid_latency_cycles_per_lane": 10,
            "compile_argv": [
                args.iverilog,
                "-g2012",
                "-s",
                "ace2_chat_residual_boundary_tb",
                "-o",
                repo_path(executable),
                repo_path(CORE_RTL),
                repo_path(TESTBENCH),
            ],
            "simulate_argv": [
                args.vvp,
                repo_path(executable),
                f"+VECTORS={repo_path(vector_path)}",
            ],
            "tool_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "iverilog": first_nonempty_line(iverilog_version_stdout),
                "vvp": first_nonempty_line(vvp_version_stdout),
            },
            "compile_wall_seconds": compile_seconds,
            "simulate_wall_seconds": simulate_seconds,
            "vectors": artifact(vector_path),
            "compile_log": artifact(compile_log),
            "simulate_log": artifact(simulate_log),
        },
        "provenance": {
            "core_rtl_changed": False,
            "synthesis_sta_status": "NOT_RUN_BOUNDARY_REPLAY_WITHOUT_CORE_RTL_CHANGE",
            "independent_reviewer_status": "PENDING",
            "sealed_certified_evidence_mutated": False,
            "sources": [
                artifact(Path(__file__)),
                artifact(TESTBENCH),
                artifact(CORE_RTL),
                artifact(REFERENCE),
                artifact(FROZEN_SCALES),
                artifact(MAKEFILE),
            ],
        },
        "total_wall_seconds": time.perf_counter() - started,
    }
    write_atomic(
        output_path,
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )
    print(
        "ACE2_CHAT_RESIDUAL_BOUNDARY_PASS "
        f"command={command['ordinal']} lanes=896 "
        f"direct_diff={record['boundary']['direct_vs_common_domain_differing_lanes']} "
        f"two_round_diff={record['boundary']['per_input_vs_one_round_differing_lanes']} "
        f"wall_seconds={record['total_wall_seconds']:.6f} "
        f"artifact={output_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
