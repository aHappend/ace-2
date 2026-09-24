#!/usr/bin/env python3
"""Generate raw-input vectors for the layer-16 runtime RMSNorm metadata core."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rtl_arbitrary_text_generation_backend as backend
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    pack_scale32,
    round_divide_even_signed,
    scale32_ratio,
    unpack_scale32,
)


SOURCE_CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
SOURCE = SOURCE_CANDIDATE / "contracts/00-control"
DEFAULT_OUTPUT = ROOT / (
    "verification/generated/"
    "ace2_layer16_post_attention_rmsnorm_runtime_metadata"
)
POSITIONS = 30
HIDDEN = 896
GROUP_SIZE = 128
GROUPS = HIDDEN // GROUP_SIZE
MAX_REBASED_MAGNITUDE = 2047


@dataclass(frozen=True)
class RuntimeResult:
    aligned: list[int]
    qgains: list[int]
    output_scales: list[int]
    outputs: list[int]
    saturation: list[int]
    common_exponent: int
    rebase_shift: int
    sumsq: int
    mean_square: int
    rms_ceil: int
    inv_rms_q30: int


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


def read_exact(path: Path, dtype: str, count: int) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype)
    if values.size != count:
        raise ValueError(f"{path.name} count differs: {values.size} != {count}")
    return values


def write_hex(path: Path, values: Iterable[int], width: int) -> None:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def round_divide_even_unsigned(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    return quotient + int(
        doubled > denominator or (doubled == denominator and (quotient & 1))
    )


def encode_gain_metadata(weights: list[float]) -> tuple[list[int], list[int]]:
    gain_integers: list[int] = []
    gain_scales: list[int] = []
    for weight in weights:
        if not math.isfinite(weight):
            raise ValueError("layernorm gain contains a non-finite value")
        if weight == 0.0:
            gain_integers.append(0)
            gain_scales.append(pack_scale32(0x8000, -24))
            continue
        scale = ceil_scale32_from_float(abs(weight) / 32767.0)
        scale_numerator, scale_denominator = scale32_ratio(scale)
        weight_numerator, weight_denominator = abs(weight).as_integer_ratio()
        magnitude = round_divide_even_unsigned(
            weight_numerator * scale_denominator,
            weight_denominator * scale_numerator,
        )
        magnitude = min(magnitude, 32767)
        gain_integers.append(-magnitude if weight < 0.0 else magnitude)
        gain_scales.append(scale)
    return gain_integers, gain_scales


def align_frame(
    activations: list[int], input_scales: list[int]
) -> tuple[list[int], int, int]:
    unpacked = [unpack_scale32(record) for record in input_scales]
    common_exponent = min(exponent for _significand, exponent in unpacked)
    exact = [
        activation * significand * (1 << (exponent - common_exponent))
        for activation, (significand, exponent) in zip(
            activations, unpacked, strict=True
        )
    ]
    maximum = max(abs(value) for value in exact)
    rebase_shift = 0
    while round_divide_even_unsigned(maximum, 1 << rebase_shift) > MAX_REBASED_MAGNITUDE:
        rebase_shift += 1
    aligned = [
        round_divide_even_signed(value, 1 << rebase_shift) for value in exact
    ]
    return aligned, common_exponent, rebase_shift


def scale_lane_fits(
    aligned: int,
    gain_integer: int,
    gain_scale32: int,
    inv_rms_q30: int,
    output_exponent: int,
) -> bool:
    gain_significand, gain_exponent = unpack_scale32(gain_scale32)
    gain_magnitude = abs(gain_integer) * gain_significand
    output_magnitude = abs(aligned) * gain_magnitude * inv_rms_q30
    dynamic_shift = output_exponent - gain_exponent + 45
    if dynamic_shift < 0:
        return False
    dynamic_fits = output_magnitude <= 127 * (1 << dynamic_shift)
    floor_shift = output_exponent - gain_exponent + 7
    if floor_shift >= 0:
        floor_fits = gain_magnitude <= 32767 * (1 << floor_shift)
    else:
        floor_fits = gain_magnitude * (1 << -floor_shift) <= 32767
    return dynamic_fits and floor_fits


def derive_qgain(
    gain_integer: int, gain_scale32: int, output_exponent: int
) -> int:
    gain_significand, gain_exponent = unpack_scale32(gain_scale32)
    product = gain_integer * gain_significand
    shift = output_exponent + 7 - gain_exponent
    if shift >= 0:
        result = round_divide_even_signed(product, 1 << shift)
    else:
        result = product << -shift
    if not -32768 <= result <= 32767:
        raise OverflowError("derived Q7.8 gain is not representable")
    return result


def runtime_reference(
    activations: list[int],
    input_scales: list[int],
    gain_integers: list[int],
    gain_scales: list[int],
) -> RuntimeResult:
    aligned, common_exponent, rebase_shift = align_frame(
        activations, input_scales
    )
    sumsq = sum(value * value for value in aligned)
    mean_square = round_divide_even_unsigned(sumsq, HIDDEN)
    rms_ceil = max(math.isqrt(mean_square), 1)
    if rms_ceil * rms_ceil < mean_square:
        rms_ceil += 1
    inv_rms_q30 = (1 << 30) // rms_ceil
    if inv_rms_q30 == 0:
        raise OverflowError("runtime reciprocal underflowed")

    output_scales: list[int] = []
    qgains: list[int] = []
    outputs: list[int] = []
    saturation: list[int] = []
    for group_start in range(0, HIDDEN, GROUP_SIZE):
        group_stop = group_start + GROUP_SIZE
        exponent = -24
        for channel in range(group_start, group_stop):
            while not scale_lane_fits(
                aligned[channel],
                gain_integers[channel],
                gain_scales[channel],
                inv_rms_q30,
                exponent,
            ):
                exponent += 1
                if exponent > 4:
                    raise OverflowError("runtime output Scale32 exceeds exponent range")
        output_scales.append(pack_scale32(0x8000, exponent))
        for channel in range(group_start, group_stop):
            qgain = derive_qgain(
                gain_integers[channel], gain_scales[channel], exponent
            )
            qgains.append(qgain)
            rounded = round_divide_even_signed(
                aligned[channel] * qgain * inv_rms_q30, 1 << 38
            )
            clipped = max(-128, min(127, rounded))
            outputs.append(clipped)
            saturation.append(int(clipped != rounded))

    return RuntimeResult(
        aligned=aligned,
        qgains=qgains,
        output_scales=output_scales,
        outputs=outputs,
        saturation=saturation,
        common_exponent=common_exponent,
        rebase_shift=rebase_shift,
        sumsq=sumsq,
        mean_square=mean_square,
        rms_ceil=rms_ceil,
        inv_rms_q30=inv_rms_q30,
    )


def generate(output: Path) -> dict[str, object]:
    source_result = json.loads(
        (SOURCE_CANDIDATE / "result.json").read_text(encoding="utf-8")
    )
    if int(source_result["selected_reference_rank"]) != 3252:
        raise ValueError("preserved control rank differs")
    if int(source_result["selected_top_token_id"]) != 12:
        raise ValueError("preserved control top token differs")
    if not source_result["control_exact_repeat"]["passed"]:
        raise ValueError("preserved exact control repeat is absent")

    activations = read_exact(
        SOURCE / "post_attention_sum_s8.bin", "i1", POSITIONS * HIDDEN
    ).reshape(POSITIONS, HIDDEN)
    input_scales_i64 = read_exact(
        SOURCE / "post_attention_output_group_scale32.bin",
        "<i8",
        POSITIONS * HIDDEN,
    ).reshape(POSITIONS, HIDDEN)
    if np.any(input_scales_i64 < 0) or np.any(input_scales_i64 > 0xFFFFFFFF):
        raise ValueError("source Scale32 escaped unsigned 32-bit storage")
    input_scales = input_scales_i64.astype(np.uint32)

    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights:
        gain_tensor = weights.get_tensor(
            "model.layers.16.post_attention_layernorm.weight"
        ).contiguous()
    gain_f32 = gain_tensor.float().cpu().numpy().astype(np.float32)
    if gain_f32.size != HIDDEN:
        raise ValueError("layer-16 post-attention gain shape differs")
    gain_values = gain_f32.astype(np.float64).tolist()
    gain_integers, gain_scales = encode_gain_metadata(gain_values)

    aligned = np.empty((POSITIONS, HIDDEN), dtype=np.int16)
    qgains = np.empty((POSITIONS, HIDDEN), dtype=np.int16)
    outputs = np.empty((POSITIONS, HIDDEN), dtype=np.int8)
    saturation = np.empty((POSITIONS, HIDDEN), dtype=np.uint8)
    output_scales = np.empty((POSITIONS, GROUPS), dtype=np.uint32)
    common_exponents = np.empty(POSITIONS, dtype=np.int8)
    rebase_shifts = np.empty(POSITIONS, dtype=np.uint8)
    sumsqs = np.empty(POSITIONS, dtype=np.uint64)
    mean_squares = np.empty(POSITIONS, dtype=np.uint64)
    rms_ceils = np.empty(POSITIONS, dtype=np.uint16)
    inv_rms = np.empty(POSITIONS, dtype=np.uint32)

    for position in range(POSITIONS):
        result = runtime_reference(
            activations[position].astype(int).tolist(),
            input_scales[position].astype(int).tolist(),
            gain_integers,
            gain_scales,
        )
        aligned[position] = np.asarray(result.aligned, dtype=np.int16)
        qgains[position] = np.asarray(result.qgains, dtype=np.int16)
        outputs[position] = np.asarray(result.outputs, dtype=np.int8)
        saturation[position] = np.asarray(result.saturation, dtype=np.uint8)
        output_scales[position] = np.asarray(
            result.output_scales, dtype=np.uint32
        )
        common_exponents[position] = result.common_exponent
        rebase_shifts[position] = result.rebase_shift
        sumsqs[position] = result.sumsq
        mean_squares[position] = result.mean_square
        rms_ceils[position] = result.rms_ceil
        inv_rms[position] = result.inv_rms_q30

    nonzero = int(np.count_nonzero(outputs))
    if nonzero == 0:
        raise ValueError("runtime-derived RMSNorm output remains all zero")
    if int(np.max(np.abs(aligned.astype(np.int64)))) > MAX_REBASED_MAGNITUDE:
        raise ValueError("runtime aligned activation escaped signed-12 bound")
    if np.any(inv_rms == 0):
        raise ValueError("runtime reciprocal underflowed")

    output.mkdir(parents=True, exist_ok=True)
    vector_files: dict[str, tuple[Iterable[int], int]] = {
        "input_activation_s8.hex": (activations.reshape(-1), 8),
        "input_scale32.hex": (input_scales.reshape(-1), 32),
        "gain_s16.hex": (gain_integers, 16),
        "gain_scale32.hex": (gain_scales, 32),
        "expected_aligned_s12.hex": (aligned.reshape(-1), 12),
        "expected_qgain_s16_q8.hex": (qgains.reshape(-1), 16),
        "expected_output_scale32.hex": (output_scales.reshape(-1), 32),
        "expected_s8.hex": (outputs.reshape(-1), 8),
        "expected_saturation.hex": (saturation.reshape(-1), 8),
        "expected_common_exponent_s8.hex": (common_exponents, 8),
        "expected_rebase_shift_u6.hex": (rebase_shifts, 8),
        "expected_sumsq_u48.hex": (sumsqs, 48),
        "expected_mean_square_u48.hex": (mean_squares, 48),
        "expected_rms_ceil_u12.hex": (rms_ceils, 12),
        "expected_inv_rms_q30.hex": (inv_rms, 31),
    }
    for name, (values, width) in vector_files.items():
        write_hex(output / name, values, width)

    constants = output / "ace2_layer16_post_attention_rmsnorm_runtime_metadata_constants.svh"
    constants.write_text(
        "\n".join(
            (
                f"localparam integer MODEL_POSITIONS = {POSITIONS};",
                f"localparam integer MODEL_HIDDEN = {HIDDEN};",
                f"localparam integer MODEL_SAMPLES = {POSITIONS * HIDDEN};",
                f"localparam integer MODEL_OUTPUT_GROUP_SIZE = {GROUP_SIZE};",
                f"localparam integer MODEL_GROUPS_PER_POSITION = {GROUPS};",
                f"localparam integer MODEL_SCALE_RECORDS = {POSITIONS * GROUPS};",
                f"localparam integer MODEL_NONZERO_OUTPUTS = {nonzero};",
                f"localparam integer MODEL_SATURATIONS = {int(saturation.sum())};",
                "",
            )
        ),
        encoding="ascii",
    )

    manifest = {
        "schema_version": 1,
        "classification": "bounded_runtime_metadata_core_increment_no_official_attempt",
        "boundary": "model.layers.16.post_attention_layernorm.output_to_signed_a8",
        "executed_contract": "group-128-power2-scale32",
        "runtime_inputs": [
            "signed A8 activation",
            "per-channel input Scale32",
            "immutable signed-16 layernorm gain integer",
            "immutable per-channel gain Scale32",
        ],
        "forbidden_runtime_sidecars_absent": [
            "aligned_activation",
            "rebase_shift",
            "sumsq",
            "inv_rms",
            "q7_8_gain",
            "output_scale32",
            "expected_output",
        ],
        "shape": {
            "positions": POSITIONS,
            "channels": HIDDEN,
            "samples": POSITIONS * HIDDEN,
            "output_group_size": GROUP_SIZE,
            "output_groups_per_position": GROUPS,
        },
        "numeric_contract": {
            "alignment": "minimum live input Scale32 exponent, bounded ties-to-even rebase",
            "mean_square": "ties-to-even sumsq/896",
            "rms": "ceil integer square root",
            "reciprocal": "floor(2^30/rms_ceil), nonzero Q30",
            "output_scale": "smallest power-of-two Scale32 satisfying live group dynamic range and signed-Q7.8 gain floor",
            "gain": "signed Q7.8 derived from immutable signed-16 plus Scale32 metadata",
            "requantization": "single ties-to-even shift by 38 with signed-A8 saturation",
        },
        "control": {
            "reference_rank": 3252,
            "top_token_id": 12,
            "exact_repeat_passed": True,
            "rank_replay_performed": False,
        },
        "observed": {
            "nonzero_outputs": nonzero,
            "saturation_count": int(saturation.sum()),
            "minimum_inv_rms_q30": int(inv_rms.min()),
            "maximum_inv_rms_q30": int(inv_rms.max()),
            "minimum_output_scale32": f"0x{int(output_scales.min()):08x}",
            "maximum_output_scale32": f"0x{int(output_scales.max()):08x}",
            "maximum_rebased_magnitude": int(
                np.max(np.abs(aligned.astype(np.int64)))
            ),
        },
        "bindings": {
            "source_candidate_result": artifact(SOURCE_CANDIDATE / "result.json"),
            "source_candidate_sha256s": artifact(SOURCE_CANDIDATE / "SHA256SUMS"),
            "source_post_attention_sum_s8": artifact(
                SOURCE / "post_attention_sum_s8.bin"
            ),
            "source_post_attention_scale32": artifact(
                SOURCE / "post_attention_output_group_scale32.bin"
            ),
            "layer16_post_attention_gain_f32_sha256": hashlib.sha256(
                gain_f32.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "runtime_bf16_sidecar": False,
            "runtime_precomputed_metadata_sidecar": False,
            "carried_activation_wider_than_a8": False,
            "u280_or_stage2_entered": False,
            "rank_selection_completed": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in members
            if path.name != "SHA256SUMS"
        ),
        encoding="ascii",
    )
    return manifest


def compare_directories(expected: Path, actual: Path) -> None:
    expected_files = sorted(path.name for path in expected.iterdir() if path.is_file())
    actual_files = sorted(path.name for path in actual.iterdir() if path.is_file())
    if expected_files != actual_files:
        raise ValueError("generated runtime vector member list differs")
    for name in expected_files:
        if (expected / name).read_bytes() != (actual / name).read_bytes():
            raise ValueError(f"generated runtime vector differs: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.check:
        if not output.is_dir():
            raise FileNotFoundError(output)
        with tempfile.TemporaryDirectory(prefix="ace2-runtime-rmsnorm-") as temporary:
            regenerated = Path(temporary) / "generated"
            manifest = generate(regenerated)
            compare_directories(output, regenerated)
    else:
        if output.exists():
            raise FileExistsError(output)
        manifest = generate(output)
    print(
        "ACE2_LAYER16_POST_ATTENTION_RMSNORM_RUNTIME_METADATA_VECTORS "
        f"samples={manifest['shape']['samples']} "
        f"nonzero={manifest['observed']['nonzero_outputs']} "
        f"saturations={manifest['observed']['saturation_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
