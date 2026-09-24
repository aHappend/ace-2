#!/usr/bin/env python3
"""Generate the frozen all-layer DPRF Scale32 model-image region.

The source is an already-recorded 128-token calibration artifact.  This tool
does not load a model or dataset and does not execute a quality metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from ace2_quality_contracts import ceil_scale32_from_float, unpack_scale32


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark/raw/quality/operator-paired-smoke-20260731T081721Z/derived_scales.json"
RUN_CONTRACT = ROOT / "benchmark/raw/quality/operator-paired-smoke-20260731T081721Z/run_contract.json"
OUT_JSON = ROOT / "reference/generated/down_projection_residual_fusion_metadata.json"
OUT_BIN = ROOT / "reference/generated/down_projection_residual_fusion_metadata.bin"
LAYERS = 24
LANES = 896
HEADER_BYTES = 384
PAIR_BYTES = 192
ACCUMULATOR_BYTES = 86016
IMMUTABLE_BYTES = 86592


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def generated_payload() -> tuple[dict[str, object], bytes]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    run_contract = json.loads(RUN_CONTRACT.read_text(encoding="utf-8"))
    if run_contract["model"]["revision"] != "060db6499f32faf8b98477b0a26969ef7d8b9987":
        raise RuntimeError("metadata source model revision changed")

    pair_section = bytearray()
    accumulator_section = bytearray()
    layers: list[dict[str, object]] = []
    for layer in range(LAYERS):
        module = source["linears"][f"model.layers.{layer}.mlp.down_proj"]
        residual_source = source["operators"][f"model.layers.{layer}.post_attention_residual"]
        destination_source = source["operators"][f"model.layers.{layer}.post_mlp_residual"]
        input_scale = float(module["hardware_input_scale"])
        weight_scales = [float(value) for value in module["weight_scale"]]
        if len(weight_scales) != LANES:
            raise RuntimeError(f"layer {layer} does not contain 896 weight scales")
        residual_record = ceil_scale32_from_float(float(residual_source["scale"]))
        destination_record = ceil_scale32_from_float(float(destination_source["scale"]))
        accumulator_records = [
            ceil_scale32_from_float(input_scale * weight_scale)
            for weight_scale in weight_scales
        ]
        unpack_scale32(residual_record)
        unpack_scale32(destination_record)
        for record in accumulator_records:
            unpack_scale32(record)
        pair_section.extend(struct.pack("<II", residual_record, destination_record))
        for record in accumulator_records:
            accumulator_section.extend(struct.pack("<I", record))
        layers.append(
            {
                "layer_id": layer,
                "module": f"model.layers.{layer}.mlp.down_proj",
                "hardware_input_scale": input_scale,
                "weight_scale_count": len(weight_scales),
                "qweight_sha256": module["qweight_sha256"],
                "residual_input_scale_source": float(residual_source["scale"]),
                "destination_scale_source": float(destination_source["scale"]),
                "residual_scale32": residual_record,
                "destination_scale32": destination_record,
                "accumulator_scale32": accumulator_records,
            }
        )

    if len(pair_section) != PAIR_BYTES or len(accumulator_section) != ACCUMULATOR_BYTES:
        raise AssertionError("DPRF metadata section size mismatch")
    metadata_digest = hashlib.sha256(pair_section + accumulator_section).digest()
    identity64 = int.from_bytes(metadata_digest[:8], "little")
    header_section = bytearray()
    for layer in range(LAYERS):
        header_section.extend(
            b"DPRF"
            + bytes((1, layer))
            + struct.pack("<H", LANES)
            + metadata_digest[:8]
        )
    image = bytes(header_section + pair_section + accumulator_section)
    if len(header_section) != HEADER_BYTES or len(image) != IMMUTABLE_BYTES:
        raise AssertionError("DPRF immutable image size mismatch")
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_id": "shared_down_projection_residual_fusion_v1",
        "classification": "model_metadata_generation_only_no_quality_metric",
        "model": run_contract["model"],
        "calibration_source": artifact(SOURCE),
        "calibration_run_contract": artifact(RUN_CONTRACT),
        "derivation": {
            "accumulator_lsb_scale32": "ceil_scale32(hardware_input_scale_times_per_output_weight_scale)",
            "residual_input_scale32": "ceil_scale32(post_attention_residual_static_scale)",
            "destination_scale32": "ceil_scale32(post_mlp_residual_static_scale)",
            "quality_metric_used": False,
        },
        "layout": {
            "canonical_base_address": "0x800000000005c300",
            "header_offset_bytes": 0,
            "header_bytes": HEADER_BYTES,
            "pair_offset_bytes": HEADER_BYTES,
            "pair_bytes": PAIR_BYTES,
            "accumulator_offset_bytes": HEADER_BYTES + PAIR_BYTES,
            "accumulator_bytes": ACCUMULATOR_BYTES,
            "immutable_image_bytes": IMMUTABLE_BYTES,
            "runtime_state_offset_bytes": IMMUTABLE_BYTES,
            "runtime_state_bytes": 128,
            "allocation_bytes": IMMUTABLE_BYTES + 128,
        },
        "identity": {
            "metadata_sha256": metadata_digest.hex(),
            "identity64_little_endian": identity64,
            "immutable_image_sha256": sha256_bytes(image),
        },
        "record_counts": {
            "layers": LAYERS,
            "lanes_per_layer": LANES,
            "accumulator_scale32": LAYERS * LANES,
            "residual_destination_pairs": LAYERS,
        },
        "layers": layers,
        "regeneration_command": ".venv/bin/python tools/gen_down_projection_residual_fusion_metadata.py",
    }
    return payload, image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload, image = generated_payload()
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if args.check:
        if not OUT_JSON.is_file() or OUT_JSON.read_bytes() != encoded:
            raise SystemExit("generated DPRF metadata JSON is stale")
        if not OUT_BIN.is_file() or OUT_BIN.read_bytes() != image:
            raise SystemExit("generated DPRF metadata binary is stale")
        print("ACE2_DPRF_METADATA_CHECK_PASS layers=24 lanes=896 immutable_bytes=86592")
        return 0
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_bytes(encoded)
    OUT_BIN.write_bytes(image)
    print("ACE2_DPRF_METADATA_GENERATED layers=24 lanes=896 immutable_bytes=86592")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
