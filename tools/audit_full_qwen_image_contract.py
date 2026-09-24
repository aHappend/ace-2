#!/usr/bin/env python3
"""Audit the accepted full-Qwen image contract without model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = "build-full-qwen-packed-w4-metadata-image-v1"
BOUNDARY = ROOT / "evidence/verification/rtl-full-qwen-autoregressive-integration-v1"
INVENTORY = BOUNDARY / "integration_inventory.json"
SCHEDULE = BOUNDARY / "command_schedule.json"
SHELL = ROOT / "rtl/ace2_shell.sv"
SHELL_TB = ROOT / "verification/tb/ace2_shell_tb.sv"
DEFAULT_OUTPUT = (
    ROOT
    / "evidence/verification"
    / MISSION_ID
    / "preflight_contract_audit.json"
)

EXPECTED_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
EXPECTED_LAYOUT = "16-byte activation-scale record followed by 896 signed-Q7.8 gains"
EXPECTED_IMAGE_BYTES = {
    "packed_w4": 246_980_608,
    "projection_metadata": 7_297_024,
    "rmsnorm_metadata": 88_592,
    "operator_aux_metadata": 55_296,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def line_number(text: str, needle: str) -> int:
    offset = text.find(needle)
    require(offset >= 0, f"required source statement missing: {needle}")
    return text.count("\n", 0, offset) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at-utc", required=True)
    args = parser.parse_args()
    require(
        EXPECTED_TIMESTAMP.fullmatch(args.generated_at_utc) is not None,
        "--generated-at-utc must use YYYY-MM-DDTHH:MM:SSZ",
    )

    for path in (INVENTORY, SCHEDULE, SHELL, SHELL_TB):
        require(path.is_file(), f"missing required input: {path}")

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    shell_text = SHELL.read_text(encoding="utf-8")
    shell_tb_text = SHELL_TB.read_text(encoding="utf-8")

    image_layout = inventory["image_layout"]
    measured_image_bytes = {
        name: int(image_layout[name]["bytes"]) for name in EXPECTED_IMAGE_BYTES
    }
    require(measured_image_bytes == EXPECTED_IMAGE_BYTES, "accepted image byte contract changed")

    norms = [
        record
        for layer in inventory["layers"]
        for record in layer["norms"]
    ] + [inventory["model"]["final_norm"]]
    require(len(norms) == 49, "accepted inventory no longer contains 49 RMSNorm records")
    require(
        {record["image_bytes"] for record in norms} == {1_808},
        "accepted RMSNorm record size changed",
    )
    require(
        {record["layout"] for record in norms} == {EXPECTED_LAYOUT},
        "accepted RMSNorm packing statement changed",
    )
    norm_by_addr = {record["image_addr"]: record for record in norms}
    require(len(norm_by_addr) == len(norms), "RMSNorm record addresses are not unique")
    ordered_norms = sorted(norms, key=lambda record: record["image_addr"])
    for left, right in zip(ordered_norms, ordered_norms[1:]):
        require(
            left["image_addr"] + left["image_bytes"] == right["image_addr"],
            "RMSNorm records are not contiguous at the accepted addresses",
        )
    require(
        ordered_norms[0]["image_addr"] == image_layout["rmsnorm_metadata"]["base"]
        and ordered_norms[-1]["image_addr"] + ordered_norms[-1]["image_bytes"]
        == image_layout["rmsnorm_metadata"]["end"],
        "RMSNorm records do not cover the accepted region exactly",
    )

    commands = schedule["commands"]
    require(schedule["command_count"] == 13_914, "accepted command count changed")
    require(len(commands) == 13_914, "schedule command list length differs")
    rms_commands = [
        item
        for item in commands
        if item["operator"]
        in {"input_rmsnorm", "post_attention_rmsnorm", "final_rmsnorm"}
    ]
    require(len(rms_commands) == 98, "accepted two-token RMSNorm command count changed")
    require(
        {item["scale_addr"] for item in rms_commands} == set(norm_by_addr),
        "accepted RMSNorm commands no longer point at every record base",
    )
    require(
        all(item["scale_addr"] in norm_by_addr for item in rms_commands),
        "RMSNorm command points outside the accepted record map",
    )

    shell_gain_low = (
        "ST_SCALE_ACT_RECV:\n"
        "                    prefix_mem_req_addr_q <= scale_addr_q + (beat_idx_ext_w << 5);"
    )
    shell_gain_high = (
        "ST_GAIN_RECV0:\n"
        "                    prefix_mem_req_addr_q <=\n"
        "                        scale_addr_q + (beat_idx_ext_w << 5) + 64'd16;"
    )
    shell_scale_act = (
        "ST_ACT_RECV: begin\n"
        "                    if (accepted_read_w) begin\n"
        "                        if (beat_idx_q == LAST_BEAT) begin\n"
        "                            prefix_mem_req_addr_q <= src0_addr_q;"
    )
    tb_gain_window = (
        "(addr >= GAIN_BASE) && (addr < (GAIN_BASE + TEST_BEATS*32))"
    )
    tb_gain_index = "beat = (addr - GAIN_BASE) >> 5;"
    for statement in (
        shell_gain_low,
        shell_gain_high,
        shell_scale_act,
        tb_gain_window,
        tb_gain_index,
    ):
        require(
            statement in (shell_text if statement.startswith("ST_") else shell_tb_text),
            "current RTL/testbench RMSNorm fetch contract changed",
        )

    gains_per_record = 896
    gain_bytes = gains_per_record * 2
    declared_header_bytes = 16
    record_bytes = norms[0]["image_bytes"]
    require(record_bytes == declared_header_bytes + gain_bytes, "RMSNorm arithmetic changed")
    require(gain_bytes == 56 * 32, "RMSNorm gain beat arithmetic changed")

    report = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "generated_at_utc": args.generated_at_utc,
        "status": "BLOCKED_GENUINE_RMSNORM_PACKING_INCOMPATIBILITY",
        "classification": "zero_model_execution_static_contract_audit",
        "accepted_contract": {
            "command_count": schedule["command_count"],
            "image_bytes": measured_image_bytes,
            "total_image_bytes": sum(measured_image_bytes.values()),
            "rmsnorm_record_count": len(norms),
            "rmsnorm_record_bytes": record_bytes,
            "rmsnorm_layout": EXPECTED_LAYOUT,
            "rmsnorm_schedule_commands": len(rms_commands),
            "rmsnorm_unique_scale_addresses": len({item["scale_addr"] for item in rms_commands}),
            "schedule_scale_address_binding": "every RMSNorm command points to image_addr, the declared record base",
        },
        "current_executable_contract": {
            "scale_activation_source": "src0_addr, not cmd_scale_addr",
            "first_gain_byte_offset_from_cmd_scale_addr": 0,
            "gain_bytes_consumed_per_command": gain_bytes,
            "gain_layout": "56 contiguous 32-byte beats; each beat is 16 little-endian signed-int16 Q7.8 gains",
            "source_locations": {
                "shell_scale_activation_reread": {
                    "path": SHELL.relative_to(ROOT).as_posix(),
                    "line": line_number(shell_text, shell_scale_act),
                },
                "shell_gain_low_address": {
                    "path": SHELL.relative_to(ROOT).as_posix(),
                    "line": line_number(shell_text, shell_gain_low),
                },
                "shell_gain_high_address": {
                    "path": SHELL.relative_to(ROOT).as_posix(),
                    "line": line_number(shell_text, shell_gain_high),
                },
                "testbench_gain_window": {
                    "path": SHELL_TB.relative_to(ROOT).as_posix(),
                    "line": line_number(shell_tb_text, tb_gain_window),
                },
                "testbench_gain_index": {
                    "path": SHELL_TB.relative_to(ROOT).as_posix(),
                    "line": line_number(shell_tb_text, tb_gain_index),
                },
            },
        },
        "incompatibility": {
            "per_record_declared_leading_metadata_bytes": declared_header_bytes,
            "per_record_rtl_leading_metadata_bytes": 0,
            "per_record_unconsumed_bytes_if_gains_start_at_base": record_bytes - gain_bytes,
            "region_bytes_affected": len(norms) * declared_header_bytes,
            "header_first_effect": (
                "the shell interprets the 16-byte header as gain channels 0..7, shifts the remaining gains by eight channels, "
                "and never reads gain channels 888..895"
            ),
            "gains_first_effect": (
                "placing gains at image_addr makes RTL consumption correct but changes the accepted header-first packing convention; "
                "the final 16 bytes become trailing metadata or padding"
            ),
            "why_generation_stopped": (
                "No image can simultaneously preserve the accepted schedule addresses, the accepted header-first RMSNorm layout, "
                "and the current executable shell fetch behavior"
            ),
        },
        "required_resolution": {
            "operator_or_boundary_decision_required": True,
            "compatible_options": [
                "approve gains-first RMSNorm records with 16 trailing metadata/padding bytes at the existing addresses",
                "revise every accepted RMSNorm schedule scale address to image_addr+16 while preserving header-first records",
            ],
            "not_authorized_here": [
                "silent packing-convention change",
                "accepted schedule-address change",
                "RTL change",
            ],
        },
        "inputs": {
            name: file_record(path)
            for name, path in (
                ("accepted_inventory", INVENTORY),
                ("accepted_schedule", SCHEDULE),
                ("current_shell", SHELL),
                ("current_shell_testbench", SHELL_TB),
                ("audit_source", Path(__file__)),
            )
        },
        "mission_local_guards": {
            "model_constructed": False,
            "model_called": False,
            "transformers_imported": False,
            "safetensors_opened": False,
            "calibration_or_recalibration": False,
            "rtl_changed": False,
            "decoder_commands_run": False,
            "synthesis_opensta_ppa_run": False,
            "publication_state_mutated": False,
        },
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(report))
    print(
        "ACE2_FULL_QWEN_IMAGE_CONTRACT_BLOCKED "
        f"reason=rmsnorm_header_offset records={len(norms)} "
        f"affected_bytes={len(norms) * declared_header_bytes} "
        f"report={output}"
    )


if __name__ == "__main__":
    main()
