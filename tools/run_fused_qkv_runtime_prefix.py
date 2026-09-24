#!/usr/bin/env python3
"""Run a frozen Qwen layer-0 RMSNorm/QKV prefix in legacy and fused modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime  # noqa: E402
from tools.ace2_chat_demo import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    FUSED_QKV_OPCODE,
    QKV_SCHEDULE_FUSED,
    QKV_SCHEDULE_LEGACY,
    build_ace2rt2_package,
    build_full_prompt_commands,
    canonical_bytes,
    load_tokenizer,
    parse_journal_v2,
    projection_reference,
    read_ace2rt2_package_metadata,
    sha256_bytes,
    tokenize_prompt,
)
from tools.model_hardware_contract import runtime_preflight  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "evidence/verification/fused-qkv-runtime-prefix-v1/attempt-0001"
)
FROZEN_PROMPT = "Bounded fused QKV runtime verification."
FROZEN_MAX_NEW_TOKENS = 1
PREFIX = "layer0_position0_input_rmsnorm_through_qkv"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def command_version(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return f"unavailable(exit={completed.returncode})"
    return (completed.stdout or completed.stderr).splitlines()[0]


def execute_prefix(
    *,
    label: str,
    package: Path,
    stop_after: int,
    output: Path,
) -> dict[str, Any]:
    argv = [
        str(accepted_runtime.DEFAULT_BINARY.resolve()),
        "--package",
        str(package.resolve()),
        "--image",
        str(accepted_runtime.IMAGE.resolve()),
        "--model",
        str(accepted_runtime.MODEL.resolve()),
        "--output",
        str(output.resolve()),
        "--stop-after",
        str(stop_after),
        "--timeout-cycles",
        "100000000",
        "--read-response-latency-cycles",
        "1",
        "--persistence-batch-commands",
        "1",
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    wall_seconds = time.perf_counter() - start
    write_atomic(output / "stdout.log", completed.stdout.encode())
    write_atomic(output / "stderr.log", completed.stderr.encode())
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} RTL prefix failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    records = parse_journal_v2(output / "progress.journal")
    if summary["status"] != "PREFIX_COMPLETE" or len(records) != stop_after:
        raise RuntimeError(f"{label} RTL prefix did not complete exactly")
    return {
        "label": label,
        "stop_after": stop_after,
        "wall_seconds": wall_seconds,
        "segment_simulator_cycles": int(summary["segment_simulator_cycles"]),
        "read_beats": sum(int(record["read_beats"]) for record in records),
        "write_beats": sum(int(record["write_beats"]) for record in records),
        "records": records,
        "summary": file_record(output / "summary.json"),
        "journal": file_record(output / "progress.journal"),
        "stdout": file_record(output / "stdout.log"),
        "stderr": file_record(output / "stderr.log"),
    }


def writes_for_records(
    records: list[dict[str, Any]],
) -> list[tuple[int, int, bytes]]:
    return [
        (int(address), int(strobe), bytes(data))
        for record in records
        for address, strobe, data in record["writes"]
    ]


def expected_writes(
    commands: list[dict[str, Any]],
    outputs: list[list[int]],
) -> list[tuple[int, int, bytes]]:
    writes = []
    for command, values in zip(commands, outputs, strict=True):
        payload = bytes(value & 0xFF for value in values)
        for offset in range(0, len(payload), 16):
            writes.append(
                (
                    int(command["dst_addr"]) + offset,
                    0xFFFF,
                    payload[offset : offset + 16],
                )
            )
    return writes


def redacted_tokenizer_preflight() -> tuple[list[int], dict[str, Any]]:
    tokenizer = load_tokenizer()
    tokenization = tokenize_prompt(
        tokenizer,
        FROZEN_PROMPT,
        DEFAULT_SYSTEM_PROMPT,
    )
    tokens = list(map(int, tokenization["chat_template_token_ids"]))
    if not tokens:
        raise RuntimeError("frozen tokenizer preflight produced no tokens")
    return tokens, {
        "status": "PASS",
        "repository": "Qwen/Qwen2.5-0.5B",
        "revision": accepted_runtime.REVISION,
        "prompt_sha256": sha256_bytes(FROZEN_PROMPT.encode()),
        "chat_template_token_count": len(tokens),
        "chat_template_tokens_sha256": sha256_bytes(
            np.asarray(tokens, dtype="<u4").tobytes()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"official attempt output is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_paths = [
        Path(__file__),
        ROOT / "tools/ace2_chat_demo.py",
        ROOT / "verification/verilator/ace2_shell_runtime_main.cpp",
        ROOT / "rtl/ace2_pkg.sv",
        ROOT / "rtl/ace2_shell.sv",
        ROOT / "constraints/ace2_rmsnorm_core.sdc",
        ROOT / "evidence/verification/fused-qkv-v3/sky130/results.json",
    ]
    contract = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "name": "fused-qkv-runtime-prefix-v1",
        "fixed_external_benchmark": False,
        "official_full_chat_invoked": False,
        "sealed_attempt_replayed": False,
        "prefix": PREFIX,
        "frozen_prompt_sha256": sha256_bytes(FROZEN_PROMPT.encode()),
        "max_new_tokens": FROZEN_MAX_NEW_TOKENS,
        "modes": {
            "legacy": "three opcode-0x01 Q/K/V descriptors",
            "fused": "one opcode-0x0b ordered Q/K/V descriptor",
        },
        "pass_criteria": {
            "oracle": "all 1152 Q/K/V bytes equal one Python fixed-point oracle",
            "equivalence": "legacy and fused write address/strobe/data streams are exact",
            "runtime": "both RTL prefixes complete without genuine command error",
            "preflight": "tokenizer, package, KV, and bounded decode schedule checks pass",
            "rejection": "aligned noncanonical fused weight/metadata and corrupt output bases are rejected before execution",
        },
        "tools": {
            "python": platform.python_version(),
            "verilator": command_version(["verilator", "--version"]),
        },
        "sources_and_constraints": {
            str(path.relative_to(ROOT)): file_record(path) for path in source_paths
        },
    }
    write_atomic(output / "contract.json", canonical_bytes(contract))

    tokens, tokenizer_check = redacted_tokenizer_preflight()
    hardware_check = runtime_preflight(
        model_id="qwen2.5-0.5b",
        embedding_shape=[151936, 896],
        max_sequence_positions=32768,
        kv_bytes_per_token_per_layer=272,
    )
    schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
    legacy_commands = build_full_prompt_commands(
        schedule,
        prompt_token_count=len(tokens),
        max_new_tokens=FROZEN_MAX_NEW_TOKENS,
        qkv_schedule_mode=QKV_SCHEDULE_LEGACY,
    )
    fused_commands = build_full_prompt_commands(
        schedule,
        prompt_token_count=len(tokens),
        max_new_tokens=FROZEN_MAX_NEW_TOKENS,
        qkv_schedule_mode=QKV_SCHEDULE_FUSED,
    )
    expected_command_delta = -2 * 24 * len(tokens)
    if len(fused_commands) - len(legacy_commands) != expected_command_delta:
        raise RuntimeError("fused full-schedule command delta differs")

    schedule_preflight = {
        "status": "PASS",
        "positions": len(tokens),
        "legacy_commands": len(legacy_commands),
        "fused_commands": len(fused_commands),
        "command_delta": len(fused_commands) - len(legacy_commands),
        "kv_write_commands": sum(
            command["operator"] == "kv_write" for command in fused_commands
        ),
        "lm_head_tile_commands": sum(
            command["operator"] == "lm_head_tile" for command in fused_commands
        ),
        "decode_bound_positions": len(tokens),
    }
    if (
        schedule_preflight["kv_write_commands"] != 24 * len(tokens)
        or schedule_preflight["lm_head_tile_commands"] != 4748 * len(tokens)
    ):
        raise RuntimeError("KV/decode schedule preflight count differs")

    embedding_offset, embedding_shape = accepted_runtime.embedding_tensor_offset(
        accepted_runtime.MODEL
    )
    package_dir = output / "packages"
    legacy_package = package_dir / "legacy.bin"
    fused_package = package_dir / "fused.bin"
    legacy_package_record = build_ace2rt2_package(
        legacy_package,
        prompt_token_ids=tokens,
        max_new_tokens=FROZEN_MAX_NEW_TOKENS,
        commands=legacy_commands,
        embedding_offset=embedding_offset,
        embedding_shape=embedding_shape,
    )
    fused_package_record = build_ace2rt2_package(
        fused_package,
        prompt_token_ids=tokens,
        max_new_tokens=FROZEN_MAX_NEW_TOKENS,
        commands=fused_commands,
        embedding_offset=embedding_offset,
        embedding_shape=embedding_shape,
    )
    if (
        read_ace2rt2_package_metadata(legacy_package)["qkv_schedule_mode"]
        != QKV_SCHEDULE_LEGACY
        or read_ace2rt2_package_metadata(fused_package)["qkv_schedule_mode"]
        != QKV_SCHEDULE_FUSED
    ):
        raise RuntimeError("runtime package QKV mode preflight differs")

    first_fused = next(
        index
        for index, command in enumerate(fused_commands)
        if command["operator"] == "fused_qkv"
    )
    rejection_cases = {}
    for field, delta in (
        ("src1_addr", 16),
        ("scale_addr", 16),
        ("dst_addr", 1),
    ):
        corrupted = [dict(command) for command in fused_commands]
        corrupted[first_fused][field] += delta
        case = {"status": "FAIL"}
        try:
            build_ace2rt2_package(
                output / f"corrupt-{field}-should-not-exist.bin",
                prompt_token_ids=tokens,
                max_new_tokens=FROZEN_MAX_NEW_TOKENS,
                commands=corrupted,
                embedding_offset=embedding_offset,
                embedding_shape=embedding_shape,
            )
        except RuntimeError as error:
            if "fused-QKV" not in str(error):
                raise
            case = {"status": "PASS_REJECTED", "error": str(error)}
        if case["status"] != "PASS_REJECTED":
            raise RuntimeError(f"corrupt fused-QKV {field} was accepted")
        rejection_cases[field] = case
    rejection = {"status": "PASS_REJECTED", "cases": rejection_cases}

    legacy_stop = next(
        int(command["ordinal"]) + 1
        for command in legacy_commands
        if command["token_step"] == 0
        and command["layer_id"] == 0
        and command["operator"] == "v_proj"
    )
    fused_stop = next(
        int(command["ordinal"]) + 1
        for command in fused_commands
        if command["token_step"] == 0
        and command["layer_id"] == 0
        and command["operator"] == "fused_qkv"
    )
    legacy_run = execute_prefix(
        label="legacy",
        package=legacy_package,
        stop_after=legacy_stop,
        output=output / "legacy",
    )
    fused_run = execute_prefix(
        label="fused",
        package=fused_package,
        stop_after=fused_stop,
        output=output / "fused",
    )

    rms_writes = writes_for_records([legacy_run["records"][0]])
    normalized_bytes = b"".join(data for _, _, data in rms_writes)
    normalized = np.frombuffer(normalized_bytes, dtype=np.int8).astype(int).tolist()
    legacy_qkv_commands = legacy_commands[1:4]
    oracle_outputs = [
        projection_reference(command, normalized) for command in legacy_qkv_commands
    ]
    oracle_writes = expected_writes(legacy_qkv_commands, oracle_outputs)
    legacy_writes = writes_for_records(legacy_run["records"][1:4])
    fused_writes = writes_for_records([fused_run["records"][1]])
    bit_exact = oracle_writes == legacy_writes == fused_writes
    if not bit_exact:
        raise RuntimeError("legacy/fused/oracle QKV write streams differ")
    if any(
        record["done_error"] and not record["saturation"]
        for record in legacy_run["records"] + fused_run["records"]
    ):
        raise RuntimeError("bounded prefix reported a genuine command error")

    for run in (legacy_run, fused_run):
        run.pop("records")
    deltas = {
        "full_schedule_commands": (
            fused_package_record["commands"] - legacy_package_record["commands"]
        ),
        "bounded_prefix_commands": fused_stop - legacy_stop,
        "bounded_prefix_read_beats": (
            fused_run["read_beats"] - legacy_run["read_beats"]
        ),
        "bounded_prefix_simulator_cycles": (
            fused_run["segment_simulator_cycles"]
            - legacy_run["segment_simulator_cycles"]
        ),
        "bounded_prefix_wall_seconds": (
            fused_run["wall_seconds"] - legacy_run["wall_seconds"]
        ),
    }
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "PASS_BIT_EXACT",
        "classification": "rtl_executed_bounded_qwen_runtime_prefix",
        "official_full_chat_invoked": False,
        "sealed_attempt_replayed": False,
        "prefix": PREFIX,
        "opcode": FUSED_QKV_OPCODE,
        "bytes_compared": 1152,
        "qkv_output_sha256": sha256_bytes(
            b"".join(data for _, _, data in oracle_writes)
        ),
        "tokenizer_preflight": tokenizer_check,
        "hardware_preflight": hardware_check,
        "kv_decode_schedule_preflight": schedule_preflight,
        "descriptor_rejection": rejection,
        "legacy": {
            **legacy_run,
            "package": legacy_package_record,
            "qkv_commands": 3,
        },
        "fused": {
            **fused_run,
            "package": fused_package_record,
            "qkv_commands": 1,
        },
        "deltas_fused_minus_legacy": deltas,
        "equivalence": {
            "oracle_legacy_fused_write_stream_exact": bit_exact,
            "writes_compared": len(oracle_writes),
            "legacy_compatibility": "PASS_BIT_EXACT",
        },
        "accepted_rtl_evidence": {
            "functional_contract": file_record(
                ROOT / "benchmark/fused_qkv_v3/contract.json"
            ),
            "canonical_sky130": file_record(
                ROOT / "evidence/verification/fused-qkv-v3/sky130/results.json"
            ),
            "scope": "pre-existing independently accepted opcode-0x0b RTL; this run changes and measures host/runtime integration only",
        },
    }
    write_atomic(output / "result.json", canonical_bytes(result))
    artifact_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
        for path in artifact_paths
    )
    write_atomic(output / "SHA256SUMS", sums.encode())
    print(
        "ACE2_FUSED_QKV_RUNTIME_PREFIX_PASS "
        f"bytes=1152 commands_delta={deltas['bounded_prefix_commands']} "
        f"reads_delta={deltas['bounded_prefix_read_beats']} "
        f"cycles_delta={deltas['bounded_prefix_simulator_cycles']} "
        f"wall_seconds_delta={deltas['bounded_prefix_wall_seconds']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
