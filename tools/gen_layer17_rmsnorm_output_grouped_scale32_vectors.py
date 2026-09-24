#!/usr/bin/env python3
"""Generate focused vectors for grouped layer-17 RMSNorm output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer17-rmsnorm-output-source-grouped-repair-v1/"
    "candidate-0001"
)
DEFAULT_OUTPUT = ROOT / "verification/generated/ace2_layer17_rmsnorm_output_grouped_scale32"
HIDDEN = 896
LANES = 16


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


def _read(group: Path, name: str, dtype: str, count: int) -> np.ndarray:
    value = np.fromfile(group / f"{name}.bin", dtype=dtype)
    if value.size != count:
        raise ValueError(f"{name} count differs: {value.size} != {count}")
    return value


def _write_hex(path: Path, values: Iterable[int], width: int) -> None:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def write_vectors(output: Path) -> dict[str, object]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    group_size = int(result["rtl_validation_rms_output_group_size"])
    if group_size <= 0 or HIDDEN % group_size or group_size % LANES:
        raise ValueError("RTL validation RMSNorm-output group is invalid")
    group = CANDIDATE / f"groups/group-{group_size:03d}"
    group_result = json.loads((group / "result.json").read_text(encoding="utf-8"))
    group_count = HIDDEN // group_size

    source = _read(group, "sum_s8", "i1", HIDDEN)
    gains = _read(group, "rms_gain_s16_q8", "<i2", HIDDEN)
    scales = _read(group, "rms_output_group_scale32", "<i8", group_count)
    expected = _read(group, "rms_output_s8", "i1", HIDDEN)
    sumsq = int(_read(group, "rms_sumsq", "<i8", 1)[0])
    inv_rms = int(_read(group, "rms_inv_q30", "<i8", 1)[0])
    saturation = int(_read(group, "rms_saturation", "u1", 1)[0])

    if np.any(gains < -(1 << 15)) or np.any(gains >= (1 << 15)):
        raise ValueError("RMSNorm gain metadata escaped signed 16 bits")
    if np.any(scales < 0) or np.any(scales > 0xFFFFFFFF):
        raise ValueError("RMSNorm group scale escaped Scale32 storage")

    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[Iterable[int], int]] = {
        "sum_s8.hex": (source, 8),
        "rms_gain_s16_q8.hex": (gains, 16),
        "rms_output_group_scale32.hex": (scales, 32),
        "expected_rms_s8.hex": (expected, 8),
    }
    for name, (values, width) in files.items():
        _write_hex(output / name, values, width)

    constants = output / "ace2_layer17_rmsnorm_output_grouped_scale32_constants.svh"
    constants.write_text(
        "\n".join(
            (
                f"localparam integer MODEL_HIDDEN = {HIDDEN};",
                f"localparam integer MODEL_LANES = {LANES};",
                f"localparam integer MODEL_BEATS = {HIDDEN // LANES};",
                f"localparam integer MODEL_RMS_GROUP_SIZE = {group_size};",
                f"localparam integer MODEL_RMS_GROUP_COUNT = {group_count};",
                f"localparam integer MODEL_BEATS_PER_GROUP = {group_size // LANES};",
                f"localparam [47:0] MODEL_EXPECTED_SUMSQ = 48'd{sumsq};",
                f"localparam [31:0] MODEL_EXPECTED_INV_RMS_Q30 = 32'd{inv_rms};",
                f"localparam MODEL_EXPECTED_SATURATION = 1'b{saturation};",
                f"localparam integer MODEL_BEST_REFERENCE_RANK = {int(result['best_observed_reference_rank'])};",
                f"localparam integer MODEL_BASELINE_REFERENCE_RANK = {int(result['baseline_reference_rank'])};",
                "",
            )
        ),
        encoding="ascii",
    )

    manifest = {
        "schema_version": 1,
        "classification": "candidate_only_focused_rtl_vectors_no_official_attempt",
        "boundary": "model.layers.17.input_rmsnorm.output_to_self_attn.q_proj.input",
        "validation_candidate": {
            "rms_output_group_size": group_size,
            "rms_output_group_count": group_count,
            "selected_as_repair": result["selected_rms_output_group_size"] == group_size,
            "reference_rank": result["best_observed_reference_rank"],
            "baseline_reference_rank": result["baseline_reference_rank"],
            "top_token_id": result["best_observed_top_token_id"],
            "first_token_restored": result["first_token_restored"],
        },
        "shape": {
            "samples": HIDDEN,
            "lanes": LANES,
            "beats": HIDDEN // LANES,
            "groups": group_count,
        },
        "expected": {
            "sumsq": sumsq,
            "inv_rms_q30": inv_rms,
            "saturation_seen": bool(saturation),
        },
        "scale32": {
            "minimum": f"0x{int(scales.min()):08x}",
            "maximum": f"0x{int(scales.max()):08x}",
        },
        "search_metrics": group_result["metrics"],
        "bindings": {
            "candidate_result": _artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": _artifact(CANDIDATE / "SHA256SUMS"),
            "selected_group_result": _artifact(group / "result.json"),
            "search_runner": _artifact(
                ROOT
                / "tools/ace2_checkpoint176_layer17_rmsnorm_output_grouped_scale32_search.py"
            ),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members = sorted(path for path in output.iterdir() if path.is_file())
    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in members if path != sums),
        encoding="ascii",
    )
    return manifest


def verify_vectors(output: Path) -> dict[str, object]:
    temporary = output.parent / f".{output.name}.check"
    if temporary.exists():
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
    expected = write_vectors(temporary)
    try:
        expected_files = sorted(path.name for path in temporary.iterdir())
        observed_files = sorted(path.name for path in output.iterdir())
        if expected_files != observed_files:
            raise RuntimeError("generated vector member list differs")
        for name in expected_files:
            if (temporary / name).read_bytes() != (output / name).read_bytes():
                raise RuntimeError(f"generated vector differs: {name}")
    finally:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError("vector output must be repository-relative")
    manifest = verify_vectors(output) if args.check else write_vectors(output)
    print(
        "ACE2_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_VECTOR_"
        + ("CHECK" if args.check else "GENERATION")
        + "_PASS "
        + f"groups={manifest['shape']['groups']} "
        + f"rank={manifest['validation_candidate']['reference_rank']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
