#!/usr/bin/env python3
"""Read-only c02 audit of sealed c01 packed-W4 and Scale32 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = {
    "base": ROOT / "build/w4a8-full-model-software-contract-v1/base/c01-mse-clip-grid/attempt-0001",
    "checkpoint-176": ROOT
    / "build/w4a8-full-model-software-contract-v1/checkpoint-176/c01-mse-clip-grid/attempt-0001",
}
DEFAULT_OUTPUT = (
    ROOT
    / "evidence/diagnostics/w4a8-c02-sealed-packed-scale32-audit-v1/result.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def unpack_scale32(record: int) -> tuple[int, int]:
    if not isinstance(record, int) or not 0 <= record <= 0xFFFFFFFF:
        raise ValueError(f"invalid Scale32 word: {record!r}")
    if record >> 24:
        raise ValueError(f"Scale32 reserved byte is nonzero: 0x{record:08x}")
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    exponent = exponent_u8 - 256 if exponent_u8 & 0x80 else exponent_u8
    if not 0x8000 <= significand <= 0xFFFF:
        raise ValueError(f"Scale32 significand is not normalized: 0x{record:08x}")
    if not -24 <= exponent <= 4:
        raise ValueError(f"Scale32 exponent is outside -24..4: 0x{record:08x}")
    return significand, exponent


def scale32_value(record: int) -> float:
    significand, exponent = unpack_scale32(record)
    return float(significand * (2.0 ** (exponent - 15)))


def add_scale_record(stats: dict[str, Any], record: int, context: str) -> None:
    try:
        value = scale32_value(record)
    except ValueError as exc:
        stats["errors"].append(f"{context}: {exc}")
        return
    stats["record_count"] += 1
    stats["minimum_value"] = min(stats["minimum_value"], value)
    stats["maximum_value"] = max(stats["maximum_value"], value)


def audit_layout(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    layout = load_json(root / "packed-w4-layout.json")
    packed_path = root / "packed-w4.bin"
    errors: list[str] = []
    expected_offset = 0
    names: set[str] = set()
    for index, entry in enumerate(layout):
        name = entry.get("name")
        if name in names:
            errors.append(f"duplicate layout name: {name}")
        names.add(name)
        expected_bytes = entry["in_features"] * entry["out_features"] // 2
        if entry["in_features"] % 2:
            errors.append(f"odd input width at {name}")
        if entry["bytes"] != expected_bytes:
            errors.append(f"packed byte count differs at {name}")
        if entry["offset"] != expected_offset:
            errors.append(f"noncontiguous offset at layout index {index}")
        expected_offset += entry["bytes"]
    if expected_offset != packed_path.stat().st_size:
        errors.append("layout extent differs from packed-w4.bin size")
    packed_sha256 = sha256_file(packed_path)
    if packed_sha256 != manifest["packed_w4_sha256"]:
        errors.append("packed-w4.bin SHA-256 differs from sealed manifest")

    nibble_counts = [0] * 16
    with packed_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            for byte in chunk:
                nibble_counts[byte & 0xF] += 1
                nibble_counts[byte >> 4] += 1
    signed_counts = {
        str(code if code < 8 else code - 16): count
        for code, count in enumerate(nibble_counts)
    }
    return {
        "errors": errors,
        "layout_entry_count": len(layout),
        "layout_sha256": sha256_file(root / "packed-w4-layout.json"),
        "module_names": sorted(names),
        "packed_bytes": packed_path.stat().st_size,
        "packed_sha256": packed_sha256,
        "signed_int4_code_counts": signed_counts,
        "status": "PASS" if not errors else "FAIL",
    }


def audit_scales(root: Path, layout_names: list[str]) -> dict[str, Any]:
    weight = load_json(root / "weight-scale32.json")
    activation = load_json(root / "activation-scale32.json")
    operator = load_json(root / "operator-scale32.json")
    kv_cache = load_json(root / "kv-cache-scale32.json")
    layout = {entry["name"]: entry for entry in load_json(root / "packed-w4-layout.json")}
    stats: dict[str, Any] = {
        "errors": [],
        "record_count": 0,
        "minimum_value": float("inf"),
        "maximum_value": 0.0,
    }
    if set(weight) != set(layout_names):
        stats["errors"].append("weight Scale32 module set differs from packed layout")
    if set(activation) != set(layout_names):
        stats["errors"].append("activation Scale32 module set differs from packed layout")

    weight_record_count = 0
    for name, entry in weight.items():
        records = entry.get("records", [])
        weight_record_count += len(records)
        if name in layout and len(records) != layout[name]["out_features"]:
            stats["errors"].append(f"weight Scale32 row count differs at {name}")
        for index, record in enumerate(records):
            add_scale_record(stats, record, f"weight.{name}[{index}]")

    for name, entry in activation.items():
        for field in (
            "hardware_input_scale32_record",
            "input_scale32_record",
            "output_scale32_record",
        ):
            record = entry.get(field)
            if record is not None:
                add_scale_record(stats, record, f"activation.{name}.{field}")
        for index, record in enumerate(entry.get("output_head_scale32_records", [])):
            add_scale_record(stats, record, f"activation.{name}.head[{index}]")

    exact_operator_exports = 0
    operator_value_counts: Counter[float] = Counter()
    for name, entry in operator.items():
        record = entry["scale32_record"]
        add_scale_record(stats, record, f"operator.{name}")
        realized = scale32_value(record)
        if realized == entry["scale"]:
            exact_operator_exports += 1
        else:
            stats["errors"].append(f"operator Scale32 float differs at {name}")
        operator_value_counts[entry["absmax"]] += 1

    kv_crosslink_count = 0
    for layer_name, entry in kv_cache.items():
        layer = int(layer_name.split("-")[-1])
        key_name = f"model.layers.{layer}.self_attn.k_proj"
        value_name = f"model.layers.{layer}.self_attn.v_proj"
        key_record = entry["key_producer_scale32"]
        value_record = entry["value_output_scale32"]
        add_scale_record(stats, key_record, f"kv.{layer_name}.key")
        add_scale_record(stats, value_record, f"kv.{layer_name}.value")
        if key_record != activation[key_name]["output_scale32_record"]:
            stats["errors"].append(f"KV key Scale32 cross-link differs at {layer_name}")
        elif value_record != activation[value_name]["output_scale32_record"]:
            stats["errors"].append(f"KV value Scale32 cross-link differs at {layer_name}")
        else:
            kv_crosslink_count += 1

    repeated_absmax = [
        {"absmax": value, "occurrences": count}
        for value, count in sorted(operator_value_counts.items())
        if count >= 3
    ]
    if stats["minimum_value"] == float("inf"):
        stats["minimum_value"] = None
    stats.update(
        {
            "exact_operator_export_count": exact_operator_exports,
            "kv_crosslink_count": kv_crosslink_count,
            "operator_count": len(operator),
            "repeated_operator_absmax_values": repeated_absmax,
            "status": "PASS" if not stats["errors"] else "FAIL",
            "weight_record_count": weight_record_count,
        }
    )
    return stats


def audit_hash_graph(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    bindings = {
        "activation_scale32_sha256": "activation-scale32.json",
        "calibration_observations_sha256": "calibration-observations.json",
        "configuration_sha256": "configuration.json",
        "input_observations_sha256": "input-observations.json",
        "kv_cache_scale32_sha256": "kv-cache-scale32.json",
        "operator_scale32_sha256": "operator-scale32.json",
        "packed_w4_sha256": "packed-w4.bin",
        "raw_output_sha256": "raw-output.json",
        "weight_scale32_sha256": "weight-scale32.json",
    }
    observed = {field: sha256_file(root / filename) for field, filename in bindings.items()}
    mismatches = [field for field, digest in observed.items() if digest != manifest[field]]
    return {
        "checked_artifact_count": len(bindings),
        "mismatches": mismatches,
        "observed_sha256": observed,
        "status": "PASS" if not mismatches else "FAIL",
    }


def audit_model(alias: str, root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    results = load_json(root / "results.json")
    layout_audit = audit_layout(root, manifest)
    scale_audit = audit_scales(root, layout_audit["module_names"])
    hash_audit = audit_hash_graph(root, manifest)
    passed = all(
        audit["status"] == "PASS" for audit in (layout_audit, scale_audit, hash_audit)
    )
    return {
        "alias": alias,
        "classification": results["classification"],
        "contract_sha256": manifest["contract_sha256"],
        "hash_graph": hash_audit,
        "model_identity_sha256": manifest["model_identity_sha256"],
        "packed_w4": layout_audit,
        "quality_metrics": results["quality"]["metrics"],
        "scale32": scale_audit,
        "sealed_attempt_root": root.relative_to(ROOT).as_posix(),
        "status": "PASS_STRUCTURAL_AUDIT" if passed else "FAIL_STRUCTURAL_AUDIT",
    }


def build_result() -> dict[str, Any]:
    models = {alias: audit_model(alias, path) for alias, path in DEFAULT_MODELS.items()}
    structural_pass = all(value["status"] == "PASS_STRUCTURAL_AUDIT" for value in models.values())
    return {
        "boundary_classification": {
            "earliest_exonerated_boundary": (
                "sealed packed-W4 byte layout/hash and exported Scale32 structural realization"
                if structural_pass
                else None
            ),
            "earliest_unresolved_boundary": (
                "activation quantization and first projection requantization"
                if structural_pass
                else "packed-weight/Scale32 realization"
            ),
            "reason": (
                "Sealed c01 contains no reference-versus-W4A8 stage tensors, saturation counts, "
                "or per-boundary logits; later normalization/nonlinear, attention/KV/cache, final "
                "norm/lm_head, and decode ranking therefore remain unresolved."
            ),
        },
        "models": models,
        "prohibitions_observed": {
            "benchmark_scoring_performed": False,
            "candidate_or_official_namespace_created": False,
            "execute_command_invoked": False,
            "frozen_bf16_control_mutated": False,
            "sealed_c01_mutated": False,
        },
        "remediation_classification": {
            "decision": "INCONCLUSIVE_NO_C02_EXECUTION_BUNDLE",
            "reason": (
                "Structural serialization checks alone cannot distinguish a correctable PTQ error "
                "from a failure requiring full QAT or quantization-aware LoRA."
            ),
        },
        "schema_version": 1,
        "status": "PASS_BOUNDED_UNRESOLVED_BOUNDARY" if structural_pass else "FAIL_EARLY_BOUNDARY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if any(root.resolve() == output or root.resolve() in output.parents for root in DEFAULT_MODELS.values()):
        raise SystemExit("refusing to write beneath a sealed c01 attempt")
    result = build_result()
    payload = canonical_bytes(result)
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit("c02 audit result is missing or nondeterministic")
        print(f"PASS {output.relative_to(ROOT)} {hashlib.sha256(payload).hexdigest()}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"WROTE {output.relative_to(ROOT)} {hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
