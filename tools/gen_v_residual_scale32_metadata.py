#!/usr/bin/env python3
"""Generate the frozen 48-row V-residual Scale32 model-image table."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from ace2_quality_contracts import (
    ceil_scale32_from_ratio,
    scale32_ratio,
    unpack_scale32,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_v_residual_value_correction_attention_v1"
SOURCE = ROOT / "reference/generated/v_residual_scale32_calibration_source.json"
DEFAULT_OUTPUT = ROOT / "reference/generated/v_residual_scale32_metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pack_record(value: float) -> tuple[int, int, int]:
    numerator, denominator = value.as_integer_ratio()
    record = ceil_scale32_from_ratio(numerator, denominator)
    significand, exponent = unpack_scale32(record)
    return record, significand, exponent


def residual_record(baseline_record: int) -> tuple[int, int, int]:
    numerator, denominator = scale32_ratio(baseline_record)
    record = ceil_scale32_from_ratio(numerator, 14 * denominator)
    significand, exponent = unpack_scale32(record)
    return record, significand, exponent


def build() -> dict[str, Any]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("contract_id") != CONTRACT:
        raise RuntimeError("V calibration source contract mismatch")
    if source.get("quality_metrics_executed") is not False:
        raise RuntimeError("V calibration source is not calibration-only")
    if canonical_sha256(source) != source.get("integrity", {}).get("canonical_sha256"):
        raise RuntimeError("V calibration source canonical hash mismatch")
    source_rows = source.get("records", [])
    if len(source_rows) != 24:
        raise RuntimeError("V calibration source must contain 24 layers")

    records: list[dict[str, Any]] = []
    for layer, source_row in enumerate(source_rows):
        if source_row.get("layer") != layer:
            raise RuntimeError("V calibration rows are not layer ordered")
        baseline, baseline_sig, baseline_exp = pack_record(
            float(source_row["baseline_v_output_scale"])
        )
        residual, residual_sig, residual_exp = residual_record(baseline)
        for kv_head in range(2):
            records.append(
                {
                    "layer": layer,
                    "kv_head": kv_head,
                    "baseline_source_decimal": repr(
                        float(source_row["baseline_v_output_scale"])
                    ),
                    "baseline_v_scale32": {
                        "packed_u32": baseline,
                        "packed_hex": f"0x{baseline:08x}",
                        "significand_u16": baseline_sig,
                        "exponent_s8": baseline_exp,
                    },
                    "residual_v_scale32": {
                        "packed_u32": residual,
                        "packed_hex": f"0x{residual:08x}",
                        "significand_u16": residual_sig,
                        "exponent_s8": residual_exp,
                    },
                }
            )
    if len(records) != 48:
        raise RuntimeError("V residual metadata table does not contain 48 rows")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "model": {
            **source["model"],
            "layers": 24,
            "kv_heads_per_layer": 2,
        },
        "calibration_provenance": {
            "source": artifact(SOURCE),
            "scope": source["calibration_scope"],
            "input_observations": source["input_observations"],
            "source_contracts": source["source_contracts"],
            "quality_metrics_executed": False,
        },
        "derivation": {
            "baseline_v_scale32": "smallest_legal_normalized_scale32_ceiling_of_the_frozen_baseline_v_output_scale",
            "residual_v_scale32": "smallest_legal_normalized_scale32_ceiling_of_exact_baseline_v_scale32_divided_by_14",
            "baseline_v_scale32_preservation": "bit_identical_record_passed_to_baseline_and_candidate_paths",
            "kv_head_rule": "the_frozen_baseline_v_scale_is_per_layer_so_the_same_record_is_repeated_for_kv_heads_0_and_1",
            "row_order": "layer_0_to_23_then_kv_head_0_to_1",
            "quality_tuning_permitted": False,
        },
        "record_format": "little_endian_u16_significand_s8_exponent_u8_reserved_zero",
        "records": records,
        "source_artifacts": [
            artifact(SOURCE),
            artifact(Path(__file__).resolve()),
        ],
        "regeneration_command": (
            ".venv/bin/python tools/gen_v_residual_scale32_metadata.py "
            "--output reference/generated/v_residual_scale32_metadata.json"
        ),
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    payload["integrity"]["canonical_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rendered = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("generated V residual Scale32 metadata is stale")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    table = json.loads(rendered)
    print(
        "ACE2_V_RESIDUAL_SCALE32_METADATA_PASS "
        f"rows={len(table['records'])} hash={table['integrity']['canonical_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
