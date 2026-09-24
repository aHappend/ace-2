#!/usr/bin/env python3
"""Generate focused vectors for the layer-16 post-attention RMSNorm output repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rtl_arbitrary_text_generation_backend as backend
from tools.ace2_layer16_post_attention_rmsnorm_output_reference import (
    MAX_REBASED_MAGNITUDE,
    reference_grouped_rmsnorm_output,
)


SOURCE_CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
SOURCE = SOURCE_CANDIDATE / "contracts/00-control"
DEFAULT_OUTPUT = ROOT / (
    "verification/generated/"
    "ace2_layer16_post_attention_rmsnorm_output_grouped_scale32"
)
POSITIONS = 30
HIDDEN = 896
OUTPUT_GROUP_SIZE = 128
OUTPUT_GROUPS_PER_POSITION = HIDDEN // OUTPUT_GROUP_SIZE
ORDERED_CONTRACTS = ("control", "group-896", "group-128", "group-064")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def write_vectors(output: Path) -> dict[str, object]:
    source_result = json.loads(
        (SOURCE_CANDIDATE / "result.json").read_text(encoding="utf-8")
    )
    if int(source_result["selected_reference_rank"]) != 3252:
        raise ValueError("preserved control rank differs")
    if int(source_result["selected_top_token_id"]) != 12:
        raise ValueError("preserved control top token differs")
    if not source_result["control_exact_repeat"]["passed"]:
        raise ValueError("preserved exact control repeat is absent")

    source_s8 = _read(
        SOURCE / "post_attention_sum_s8.bin", "i1", POSITIONS * HIDDEN
    ).reshape(POSITIONS, HIDDEN)
    source_scale32 = _read(
        SOURCE / "post_attention_output_group_scale32.bin",
        "<i8",
        POSITIONS * HIDDEN,
    ).reshape(POSITIONS, HIDDEN)
    if np.any(source_scale32 < 0) or np.any(source_scale32 > 0xFFFFFFFF):
        raise ValueError("source Scale32 escaped unsigned 32-bit storage")

    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights:
        gain_tensor = weights.get_tensor(
            "model.layers.16.post_attention_layernorm.weight"
        ).contiguous()
    gain_f32 = gain_tensor.float().cpu().numpy().astype(np.float32)
    if gain_f32.size != HIDDEN:
        raise ValueError("layer-16 post-attention gain shape differs")
    gain_values = gain_f32.astype(np.float64).tolist()

    aligned = np.empty((POSITIONS, HIDDEN), dtype=np.int16)
    gains = np.empty((POSITIONS, HIDDEN), dtype=np.int16)
    expected = np.empty((POSITIONS, HIDDEN), dtype=np.int8)
    saturation = np.empty((POSITIONS, HIDDEN), dtype=np.uint8)
    output_scales = np.empty(
        (POSITIONS, OUTPUT_GROUPS_PER_POSITION), dtype=np.uint32
    )
    inv_rms = np.empty(POSITIONS, dtype=np.uint32)
    sumsq = np.empty(POSITIONS, dtype=np.uint64)
    rebase_shift = np.empty(POSITIONS, dtype=np.uint8)
    common_exponent = np.empty(POSITIONS, dtype=np.int8)

    for position in range(POSITIONS):
        result = reference_grouped_rmsnorm_output(
            source_s8[position].astype(int).tolist(),
            source_scale32[position].astype(int).tolist(),
            gain_values,
            OUTPUT_GROUP_SIZE,
        )
        aligned[position] = np.asarray(result.aligned, dtype=np.int16)
        gains[position] = np.asarray(result.gains_q8, dtype=np.int16)
        expected[position] = np.asarray(result.outputs, dtype=np.int8)
        saturation[position] = np.asarray(result.saturation, dtype=np.uint8)
        output_scales[position] = np.asarray(
            result.output_scale32, dtype=np.uint32
        )
        inv_rms[position] = result.inv_rms_q30
        sumsq[position] = result.sumsq
        rebase_shift[position] = result.rebase_shift
        common_exponent[position] = result.common_exponent

    nonzero_outputs = int(np.count_nonzero(expected))
    if nonzero_outputs == 0:
        raise ValueError("repaired RMSNorm output remains all zero")
    if int(np.max(np.abs(aligned.astype(np.int64)))) > MAX_REBASED_MAGNITUDE:
        raise ValueError("rebased RMSNorm activation escaped bounded width")
    if np.any(inv_rms == 0):
        raise ValueError("rebased RMSNorm reciprocal underflowed")

    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[Iterable[int], int]] = {
        "aligned_activation_s12.hex": (aligned.reshape(-1), 12),
        "gain_s16_q8.hex": (gains.reshape(-1), 16),
        "inv_rms_q30.hex": (inv_rms, 32),
        "output_group_scale32.hex": (output_scales.reshape(-1), 32),
        "expected_s8.hex": (expected.reshape(-1), 8),
        "saturation.hex": (saturation.reshape(-1), 8),
        "sumsq_u48.hex": (sumsq, 48),
        "rebase_shift_u6.hex": (rebase_shift, 8),
        "common_exponent_s8.hex": (common_exponent, 8),
    }
    for name, (values, width) in files.items():
        _write_hex(output / name, values, width)

    constants = output / (
        "ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_constants.svh"
    )
    constants.write_text(
        "\n".join(
            (
                f"localparam integer MODEL_POSITIONS = {POSITIONS};",
                f"localparam integer MODEL_HIDDEN = {HIDDEN};",
                f"localparam integer MODEL_SAMPLES = {POSITIONS * HIDDEN};",
                f"localparam integer MODEL_OUTPUT_GROUP_SIZE = {OUTPUT_GROUP_SIZE};",
                f"localparam integer MODEL_GROUPS_PER_POSITION = {OUTPUT_GROUPS_PER_POSITION};",
                f"localparam integer MODEL_SCALE_RECORDS = {POSITIONS * OUTPUT_GROUPS_PER_POSITION};",
                f"localparam integer MODEL_NONZERO_OUTPUTS = {nonzero_outputs};",
                f"localparam integer MODEL_SATURATIONS = {int(saturation.sum())};",
                f"localparam integer MODEL_MAX_REBASED_MAGNITUDE = {int(np.max(np.abs(aligned.astype(np.int64))))};",
                "",
            )
        ),
        encoding="ascii",
    )

    gain_bytes = gain_f32.astype("<f4", copy=False).tobytes()
    manifest = {
        "schema_version": 1,
        "classification": "candidate_arithmetic_increment_no_official_attempt",
        "boundary": "model.layers.16.post_attention_layernorm.output_to_signed_a8",
        "ordered_contracts": list(ORDERED_CONTRACTS),
        "executed_contract": "group-128",
        "control": {
            "reference_rank": 3252,
            "top_token_id": 12,
            "exact_repeat_passed": True,
            "gate_up_input_nonzero_samples": 0,
        },
        "shape": {
            "positions": POSITIONS,
            "channels": HIDDEN,
            "samples": POSITIONS * HIDDEN,
            "output_group_size": OUTPUT_GROUP_SIZE,
            "output_groups_per_position": OUTPUT_GROUPS_PER_POSITION,
        },
        "numeric_contract": {
            "source": "accepted group-1 signed-A8 plus per-channel Scale32",
            "gain": "model.layers.16.post_attention_layernorm.weight encoded as signed Q7.8 after output-scale selection",
            "gain_representability_floor": True,
            "rebased_internal_activation": "signed 12-bit magnitude bounded to 2047",
            "reciprocal": "nonzero unsigned Q30 derived from rebased 896-channel RMS",
            "product": "signed 61-bit activation*gain*reciprocal",
            "rounding": "signed round-to-nearest ties-to-even at 38 fractional bits",
            "saturation": "signed int8 clamp",
            "carried_output": "signed A8 plus explicit Scale32",
            "runtime_bf16_sidecar": False,
        },
        "observed": {
            "nonzero_outputs": nonzero_outputs,
            "saturation_count": int(saturation.sum()),
            "minimum_inv_rms_q30": int(inv_rms.min()),
            "maximum_inv_rms_q30": int(inv_rms.max()),
            "maximum_rebased_magnitude": int(
                np.max(np.abs(aligned.astype(np.int64)))
            ),
            "minimum_output_scale32": f"0x{int(output_scales.min()):08x}",
            "maximum_output_scale32": f"0x{int(output_scales.max()):08x}",
        },
        "bindings": {
            "source_candidate_result": _artifact(SOURCE_CANDIDATE / "result.json"),
            "source_candidate_sha256s": _artifact(SOURCE_CANDIDATE / "SHA256SUMS"),
            "source_post_attention_sum_s8": _artifact(
                SOURCE / "post_attention_sum_s8.bin"
            ),
            "source_post_attention_scale32": _artifact(
                SOURCE / "post_attention_output_group_scale32.bin"
            ),
            "layer16_post_attention_gain_f32_sha256": _bytes_sha256(gain_bytes),
            "generator": _artifact(Path(__file__).resolve()),
            "reference": _artifact(
                ROOT
                / "tools/ace2_layer16_post_attention_rmsnorm_output_reference.py"
            ),
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "gate_up_w4_mutated": False,
            "carried_activation_wider_than_a8": False,
            "u280_or_stage2_entered": False,
            "rank_selection_completed": False,
            "independent_review_still_required": True,
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
        "ACE2_LAYER16_POST_ATTENTION_RMSNORM_OUTPUT_GROUPED_SCALE32_VECTOR_"
        + ("CHECK" if args.check else "GENERATION")
        + "_PASS "
        + f"samples={manifest['shape']['samples']} "
        + f"nonzero={manifest['observed']['nonzero_outputs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
