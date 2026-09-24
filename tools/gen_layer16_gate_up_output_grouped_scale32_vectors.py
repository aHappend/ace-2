#!/usr/bin/env python3
"""Generate focused RTL vectors for the selected layer-16 gate/up quantizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
DEFAULT_OUTPUT = ROOT / "verification/generated/ace2_layer16_gate_up_output_grouped_scale32"
POSITIONS = 30
CHANNELS = 4864
STREAMS = ("gate", "up")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read(path: Path, dtype: str, count: int) -> np.ndarray:
    value = np.fromfile(path, dtype=dtype)
    if value.size != count:
        raise ValueError(f"{path.name} count differs: {value.size} != {count}")
    return value


def _write_hex(path: Path, values: Iterable[int], width: int) -> None:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def _selected() -> tuple[dict[str, object], Path, int]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    selected_id = str(result["selected_contract_id"])
    ordered = result["ordered_gate_up_output_contracts"]
    matches = [
        (index, item)
        for index, item in enumerate(ordered)
        if item["contract_id"] == selected_id
    ]
    if len(matches) != 1:
        raise ValueError("selected gate/up contract is not unique")
    index, item = matches[0]
    group_size = int(item["output_group_size"])
    directory = CANDIDATE / "contracts" / f"{index:02d}-{selected_id}"
    return result, directory, group_size


def write_vectors(output: Path) -> dict[str, object]:
    result, selected, group_size = _selected()
    if group_size <= 0 or CHANNELS % group_size:
        raise ValueError("selected gate/up group does not divide 4,864 channels")
    group_count = CHANNELS // group_size
    output_count = CHANNELS * len(STREAMS)
    field_count = POSITIONS * CHANNELS
    scale_count = POSITIONS * group_count

    combined: dict[str, list[np.ndarray]] = {
        "accumulator": [],
        "multiplier": [],
        "shift": [],
        "expected": [],
        "saturation": [],
        "scale": [],
    }
    for stream in STREAMS:
        accumulator = _read(
            selected / f"{stream}_accumulator_s32.bin", "<i8", field_count
        ).reshape(POSITIONS, CHANNELS)[-1]
        multiplier = _read(
            selected / f"{stream}_multiplier_s32.bin", "<i8", field_count
        ).reshape(POSITIONS, CHANNELS)[-1]
        shift = _read(
            selected / f"{stream}_shift_u6.bin", "<i8", field_count
        ).reshape(POSITIONS, CHANNELS)[-1]
        expected = _read(
            selected / f"{stream}_output_s8.bin", "i1", field_count
        ).reshape(POSITIONS, CHANNELS)[-1]
        saturation = _read(
            selected / f"{stream}_saturation.bin", "u1", field_count
        ).reshape(POSITIONS, CHANNELS)[-1]
        scales = _read(
            selected / f"{stream}_output_group_scale32.bin", "<i8", scale_count
        ).reshape(POSITIONS, group_count)[-1]
        if np.any(accumulator < -(1 << 31)) or np.any(accumulator >= (1 << 31)):
            raise ValueError(f"{stream} accumulator escaped signed 32 bits")
        if np.any(multiplier < 0) or np.any(multiplier > 0x7FFFFFFF):
            raise ValueError(f"{stream} multiplier escaped positive signed 32 bits")
        if np.any(shift < 0) or np.any(shift > 63):
            raise ValueError(f"{stream} shift escaped six bits")
        if np.any(scales < 0) or np.any(scales > 0xFFFFFFFF):
            raise ValueError(f"{stream} Scale32 metadata escaped 32 bits")
        combined["accumulator"].append(accumulator)
        combined["multiplier"].append(multiplier)
        combined["shift"].append(shift)
        combined["expected"].append(expected)
        combined["saturation"].append(saturation)
        combined["scale"].append(scales)

    output.mkdir(parents=True, exist_ok=True)
    flattened = {name: np.concatenate(values) for name, values in combined.items()}
    for name, width in (
        ("accumulator", 32),
        ("multiplier", 32),
        ("shift", 8),
        ("expected", 8),
        ("saturation", 8),
        ("scale", 32),
    ):
        _write_hex(output / f"{name}.hex", flattened[name], width)

    constants = output / "ace2_layer16_gate_up_output_grouped_scale32_constants.svh"
    constants.write_text(
        "\n".join(
            (
                f"localparam integer MODEL_STREAM_CHANNELS = {CHANNELS};",
                f"localparam integer MODEL_STREAM_COUNT = {len(STREAMS)};",
                f"localparam integer MODEL_OUTPUTS = {output_count};",
                f"localparam integer MODEL_OUTPUT_GROUP_SIZE = {group_size};",
                f"localparam integer MODEL_GROUPS_PER_STREAM = {group_count};",
                f"localparam integer MODEL_SCALE_RECORDS = {group_count * len(STREAMS)};",
                f"localparam integer MODEL_SATURATION_COUNT = {int(flattened['saturation'].sum())};",
                f"localparam integer MODEL_SELECTED_REFERENCE_RANK = {int(result['selected_reference_rank'])};",
                f"localparam integer MODEL_BASELINE_REFERENCE_RANK = {int(result['baseline_reference_rank'])};",
                "",
            )
        ),
        encoding="ascii",
    )

    files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "classification": "candidate_only_focused_rtl_vectors_no_official_attempt",
        "boundary": "model.layers.16.mlp.gate_up_proj.output_to_signed_a8",
        "selected_candidate": {
            "contract_id": result["selected_contract_id"],
            "output_group_size": group_size,
            "reference_rank": result["selected_reference_rank"],
            "top_token_id": result["selected_top_token_id"],
            "selection_reason": result["selection_reason"],
            "exact_repeat_passed": result["selected_exact_repeat_passed"],
        },
        "shape": {
            "streams": len(STREAMS),
            "channels_per_stream": CHANNELS,
            "outputs": output_count,
            "groups_per_stream": group_count,
            "scale_records": group_count * len(STREAMS),
        },
        "expected": {
            "saturation_count": int(flattened["saturation"].sum()),
        },
        "bindings": {
            "candidate_result": _artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": _artifact(CANDIDATE / "SHA256SUMS"),
            "selected_result": _artifact(selected / "result.json"),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in members),
        encoding="ascii",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.check:
        if not output.is_dir():
            raise SystemExit("generated gate/up vector directory is absent")
        expected = {
            path.name: path.read_bytes()
            for path in output.iterdir()
            if path.is_file()
        }
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ace2-gate-up-vectors-", dir=ROOT / "build") as temporary:
            regenerated = Path(temporary)
            write_vectors(regenerated)
            observed = {
                path.name: path.read_bytes()
                for path in regenerated.iterdir()
                if path.is_file()
            }
        if observed != expected:
            raise SystemExit("generated gate/up vectors are not byte-exact")
        print("ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_VECTORS_CHECK_PASS")
        return 0
    if output.exists():
        raise SystemExit("output directory already exists")
    manifest = write_vectors(output)
    print(
        "ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_VECTORS_PASS "
        f"outputs={manifest['shape']['outputs']} group={manifest['selected_candidate']['output_group_size']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
