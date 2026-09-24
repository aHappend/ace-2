#!/usr/bin/env python3
"""Generate focused RTL vectors for layer-16 down_proj accumulator requantization."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-down-proj-accumulator-requantization-repair-v1/"
    "candidate-0002"
)
GENERATED = ROOT / "verification/generated/ace2_layer16_down_proj_accumulator_requantizer"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_i64(path: Path) -> list[int]:
    data = path.read_bytes()
    require(len(data) % 8 == 0, f"misaligned int64 tensor: {path}")
    return [item[0] for item in struct.iter_unpack("<q", data)]


def read_i8(path: Path) -> list[int]:
    data = path.read_bytes()
    return [value - 256 if value >= 128 else value for value in data]


def write_hex(path: Path, values: Iterable[int], bits: int) -> None:
    width = bits // 4
    mask = (1 << bits) - 1
    path.write_text(
        "".join(f"{int(value) & mask:0{width}x}\n" for value in values),
        encoding="ascii",
    )


def build(destination: Path) -> None:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    group_size = int(result["rtl_validation_down_output_group_size"])
    group = CANDIDATE / f"groups/group-{group_size:03d}"
    require(group.is_dir(), "RTL validation group is absent")

    accumulator = read_i64(group / "down_accumulator_s32.bin")
    multiplier = read_i64(group / "down_multiplier_s32.bin")
    shifts = read_i64(group / "down_shift_u6.bin")
    scales = read_i64(group / "down_output_group_scale32.bin")
    expected = read_i8(group / "down_output_s8.bin")
    saturation = list((group / "down_saturation.bin").read_bytes())
    outputs = len(expected)
    require(outputs == 896, "model down-projection output width differs")
    require(
        len(accumulator) == len(multiplier) == len(shifts) == len(saturation) == outputs,
        "model requantization tensor lengths differ",
    )
    require(len(scales) == outputs // group_size, "group Scale32 count differs")
    require(all(-(1 << 31) <= value < (1 << 31) for value in accumulator), "accumulator escaped s32")
    require(all(0 <= value < (1 << 31) for value in multiplier), "multiplier escaped nonnegative s32")
    require(all(0 <= value <= 63 for value in shifts), "right shift escaped u6")
    require(
        all((value >> 24) == 0 and (value & 0xFFFF) != 0 for value in scales),
        "invalid Scale32 output record",
    )

    destination.mkdir(parents=True)
    write_hex(destination / "down_accumulator_s32.hex", accumulator, 32)
    write_hex(destination / "down_multiplier_s32.hex", multiplier, 32)
    write_hex(destination / "down_shift_u6.hex", shifts, 8)
    write_hex(destination / "down_output_group_scale32.hex", scales, 32)
    write_hex(destination / "down_expected_s8.hex", expected, 8)
    write_hex(destination / "down_expected_saturation.hex", saturation, 8)
    constants = destination / "ace2_layer16_down_proj_accumulator_requantizer_constants.svh"
    constants.write_text(
        "// Generated from checkpoint-176 candidate evidence.\n"
        f"localparam integer MODEL_DOWN_OUTPUTS = {outputs};\n"
        f"localparam integer MODEL_DOWN_OUTPUT_GROUP_SIZE = {group_size};\n"
        f"localparam integer MODEL_DOWN_OUTPUT_GROUP_COUNT = {len(scales)};\n",
        encoding="ascii",
    )
    manifest = {
        "schema_version": 1,
        "candidate_status": result["status"],
        "selected_as_repair": result["selected_down_output_group_size"] is not None,
        "validation_group_size": group_size,
        "shape": {
            "down_outputs": outputs,
            "down_output_group_size": group_size,
            "down_output_groups": len(scales),
        },
        "expected": {
            "down_saturation_count": sum(saturation),
            "reference_rank": (
                result["selected_reference_rank"]
                if result["selected_down_output_group_size"] is not None
                else result["best_observed_reference_rank"]
            ),
            "top_token_id": (
                result["selected_top_token_id"]
                if result["selected_down_output_group_size"] is not None
                else result["best_observed_top_token_id"]
            ),
        },
        "sources": {
            "candidate_result": artifact(CANDIDATE / "result.json"),
            "group_result": artifact(group / "result.json"),
            "accumulator": artifact(group / "down_accumulator_s32.bin"),
            "multiplier": artifact(group / "down_multiplier_s32.bin"),
            "shift": artifact(group / "down_shift_u6.bin"),
            "scale32": artifact(group / "down_output_group_scale32.bin"),
            "expected_s8": artifact(group / "down_output_s8.bin"),
            "saturation": artifact(group / "down_saturation.bin"),
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members = sorted(path for path in destination.iterdir() if path.is_file())
    (destination / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in members
            if path.name != "SHA256SUMS"
        ),
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        require(GENERATED.is_dir(), "generated vectors do not exist")
        with tempfile.TemporaryDirectory(prefix="ace2-down-acc-vectors-") as temporary:
            fresh = Path(temporary) / "generated"
            build(fresh)
            expected = {path.name: path.read_bytes() for path in fresh.iterdir() if path.is_file()}
            actual = {path.name: path.read_bytes() for path in GENERATED.iterdir() if path.is_file()}
            require(actual == expected, "generated vectors differ from candidate evidence")
        print("ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_VECTORS_CHECK_PASS")
        return 0
    if GENERATED.exists():
        shutil.rmtree(GENERATED)
    build(GENERATED)
    print(f"ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_VECTORS_WRITTEN path={GENERATED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
