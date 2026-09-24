#!/usr/bin/env python3
"""Generate or verify focused RTL vectors for the selected exact group-1 repair."""

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
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0002"
)
GROUP = CANDIDATE / "groups/group-001"
DEFAULT_OUTPUT = ROOT / "verification/generated/ace2_layer16_group1_scale32"
HIDDEN_SIZE = 896
GROUP_SIZE = 1
GROUP_COUNT = HIDDEN_SIZE
RMS_LANES = 16
Q_OUTPUTS = 896


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


def _read(name: str, dtype: str, count: int) -> np.ndarray:
    value = np.fromfile(GROUP / f"{name}.bin", dtype=dtype)
    if value.size != count:
        raise ValueError(f"{name} count differs: {value.size} != {count}")
    return value


def _scalar(name: str) -> int:
    return int(_read(name, "<i8", 1)[0])


def _pack(values: Iterable[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (int(value) & mask) << (lane * width)
    return packed


def _chunks(values: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, values.size, size):
        yield values[start : start + size]


def _write_hex(path: Path, values: Iterable[int], width: int) -> None:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def write_vectors(output: Path) -> dict[str, object]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    group_result = json.loads((GROUP / "result.json").read_text(encoding="utf-8"))
    if result["selected_source_group_size"] != 1:
        raise ValueError("candidate-0002 no longer selects group 1")
    if result["selected_reference_rank"] >= result["baseline_reference_rank"]:
        raise ValueError("selected group 1 no longer improves the baseline")

    attention_source = _read("attention_source_s8", "i1", HIDDEN_SIZE)
    down_source = _read("down_source_s8", "i1", HIDDEN_SIZE)
    attention_scales = _read("attention_group_scale32", "<i8", GROUP_COUNT)
    down_scales = _read("down_group_scale32", "<i8", GROUP_COUNT)
    attention_group = _read("attention_group_s8", "i1", HIDDEN_SIZE)
    down_group = _read("down_group_s8", "i1", HIDDEN_SIZE)
    sum_s8 = _read("sum_s8", "i1", HIDDEN_SIZE)
    rms_gain = _read("rms_gain_s16_q8", "<i2", HIDDEN_SIZE)
    rms_output = _read("rms_output_s8", "i1", HIDDEN_SIZE)
    q_weight = _read("q_weight_s4", "i1", Q_OUTPUTS * HIDDEN_SIZE)
    q_multiplier = _read("q_multiplier_s32", "<i8", Q_OUTPUTS)
    q_shift = _read("q_shift_u6", "<i8", Q_OUTPUTS)
    q_accumulator = _read("q_accumulator_s32", "<i8", Q_OUTPUTS)
    q_output = _read("q_output_s8", "i1", Q_OUTPUTS)
    q_saturation = _read("q_saturation", "u1", Q_OUTPUTS)

    if np.any(attention_scales < 0) or np.any(attention_scales > 0xFFFFFFFF):
        raise ValueError("attention Scale32 artifact escaped 32 bits")
    if np.any(down_scales < 0) or np.any(down_scales > 0xFFFFFFFF):
        raise ValueError("down Scale32 artifact escaped 32 bits")
    if np.any(q_multiplier < 0) or np.any(q_multiplier > 0x7FFFFFFF):
        raise ValueError("Q multiplier artifact escaped signed 32-bit positive range")
    if np.any(q_shift < 0) or np.any(q_shift > 63):
        raise ValueError("Q shift artifact escaped six bits")
    if int(_read("attention_saturation", "u1", HIDDEN_SIZE).sum()) != 0:
        raise ValueError("selected attention source unexpectedly saturates")
    if int(_read("down_saturation", "u1", HIDDEN_SIZE).sum()) != 0:
        raise ValueError("selected down source unexpectedly saturates")
    if int(_read("sum_saturation", "u1", HIDDEN_SIZE).sum()) != 0:
        raise ValueError("selected sum unexpectedly saturates")

    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[Iterable[int], int]] = {
        "attention_source_s8.hex": (attention_source, 8),
        "down_source_s8.hex": (down_source, 8),
        "attention_group_scale32.hex": (attention_scales, 32),
        "down_group_scale32.hex": (down_scales, 32),
        "expected_attention_group_s8.hex": (attention_group, 8),
        "expected_down_group_s8.hex": (down_group, 8),
        "expected_sum_s8.hex": (sum_s8, 8),
        "rms_gain_s16_q8.hex": (
            (_pack(chunk, 16) for chunk in _chunks(rms_gain, RMS_LANES)),
            RMS_LANES * 16,
        ),
        "expected_rms_s8.hex": (
            (_pack(chunk, 8) for chunk in _chunks(rms_output, RMS_LANES)),
            RMS_LANES * 8,
        ),
        "q_weight_s4.hex": (q_weight, 4),
        "q_multiplier_s32.hex": (q_multiplier, 32),
        "q_shift_u6.hex": (q_shift, 8),
        "q_expected_accumulator_s32.hex": (q_accumulator, 32),
        "q_expected_s8.hex": (q_output, 8),
        "q_expected_saturation.hex": (q_saturation, 8),
    }
    for name, (values, width) in files.items():
        _write_hex(output / name, values, width)

    constants = output / "ace2_layer16_group1_scale32_constants.svh"
    constants.write_text(
        "\n".join(
            (
                f"localparam integer MODEL_HIDDEN_SIZE = {HIDDEN_SIZE};",
                f"localparam integer MODEL_GROUP_SIZE = {GROUP_SIZE};",
                f"localparam integer MODEL_GROUP_COUNT = {GROUP_COUNT};",
                f"localparam integer MODEL_Q_OUTPUTS = {Q_OUTPUTS};",
                f"localparam logic [31:0] MODEL_ATTENTION_BASE_SCALE32 = 32'h{_scalar('attention_base_scale32'):08x};",
                f"localparam logic [31:0] MODEL_DOWN_BASE_SCALE32 = 32'h{_scalar('down_base_scale32'):08x};",
                f"localparam logic [31:0] MODEL_SUM_SCALE32 = 32'h{_scalar('sum_scale32'):08x};",
                f"localparam logic [31:0] MODEL_Q_INPUT_SCALE32 = 32'h{_scalar('q_input_scale32'):08x};",
                f"localparam logic [31:0] MODEL_Q_OUTPUT_SCALE32 = 32'h{_scalar('q_output_scale32'):08x};",
                f"localparam logic [47:0] MODEL_RMS_SUMSQ = 48'd{_scalar('rms_sumsq')};",
                f"localparam logic [31:0] MODEL_RMS_INV_Q30 = 32'd{_scalar('rms_inv_q30')};",
                f"localparam integer MODEL_Q_SATURATION_COUNT = {int(q_saturation.sum())};",
                "",
            )
        ),
        encoding="ascii",
    )

    manifest = {
        "schema_version": 1,
        "classification": "candidate_only_focused_rtl_vectors_no_official_attempt",
        "boundary": "layer16_group1_source_quantize_common_scale_sum_layer17_rmsnorm_q_proj",
        "selected_candidate": {
            "group_size": result["selected_source_group_size"],
            "reference_rank": result["selected_reference_rank"],
            "baseline_reference_rank": result["baseline_reference_rank"],
            "rank_improvement": result["baseline_reference_rank"]
            - result["selected_reference_rank"],
            "top_token_id": result["selected_top_token_id"],
            "first_token_restored": result["first_token_restored"],
        },
        "shape": {
            "hidden": HIDDEN_SIZE,
            "source_group": GROUP_SIZE,
            "source_groups": GROUP_COUNT,
            "rms_lanes": RMS_LANES,
            "q_outputs": Q_OUTPUTS,
            "q_mac_lanes": GROUP_SIZE,
        },
        "expected": {
            "source_saturation_count": 0,
            "sum_saturation_count": 0,
            "rms_saturation": bool(_read("rms_saturation", "u1", 1)[0]),
            "rms_sumsq": _scalar("rms_sumsq"),
            "rms_inv_q30": _scalar("rms_inv_q30"),
            "q_saturation_count": int(q_saturation.sum()),
            "q_saturation_indices": np.flatnonzero(q_saturation).astype(int).tolist(),
        },
        "scales": {
            "attention_base_scale32": f"0x{_scalar('attention_base_scale32'):08x}",
            "down_base_scale32": f"0x{_scalar('down_base_scale32'):08x}",
            "sum_scale32": f"0x{_scalar('sum_scale32'):08x}",
            "q_input_scale32": f"0x{_scalar('q_input_scale32'):08x}",
            "q_output_scale32": f"0x{_scalar('q_output_scale32'):08x}",
        },
        "search_metrics": group_result["metrics"],
        "bindings": {
            "candidate_result": _artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": _artifact(CANDIDATE / "SHA256SUMS"),
            "selected_group_result": _artifact(GROUP / "result.json"),
            "search_runner": _artifact(
                ROOT
                / "tools/ace2_checkpoint176_layer16_source_grouped_scale32_search.py"
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
        "ACE2_LAYER16_GROUP1_SCALE32_VECTOR_"
        + ("CHECK" if args.check else "GENERATION")
        + "_PASS "
        + f"groups={manifest['shape']['source_groups']} "
        + f"rank={manifest['selected_candidate']['reference_rank']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
