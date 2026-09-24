#!/usr/bin/env python3
"""Run the exactly-once, prompt-only blinded V4 separation scan."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v4"
MANIFEST = DATASET / "dataset_manifest.json"
TRAIN = DATASET / "train.jsonl"
PROBES = DATASET / "synthetic_probes.jsonl"
DEV = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/dev.jsonl"
HOLDOUT = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/holdout.jsonl"
HELPER = ROOT / "tools/run_qwen25_05b_bf16_lora_product_v3_blinded_cross_split_scan.py"
SIDECAR = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v4-cross-split-blinded-sidecar.json"
ATTESTATION = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v4-cross-split-blinded-attestation.json"
OPERATOR_APPROVAL_REFERENCE = "LIVE OPERATOR V4 FREEZE DIRECTIVE 2026-08-07"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def json_payload(value: dict[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def write_with_companion(path: Path, payload: bytes) -> None:
    write_new(path, payload)
    companion = path.with_suffix(path.suffix + ".sha256")
    write_new(companion, f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n".encode("ascii"))


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("ace2_v4_blinded_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import blinded scanner helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collision_summary(helper: Any, left: list[Any], right: list[Any], threshold: float) -> dict[str, Any]:
    near_count, maximum = helper.near_collision_result(left, right, threshold)
    return {
        "exact_normalized_prompt_collision_count": helper.exact_collision_count(left, right),
        "maximum_five_gram_jaccard": maximum,
        "near_duplicate_collision_count": near_count,
        "template_family_collision_count": helper.template_collision_count(left, right),
    }


def passed(result: dict[str, Any], threshold: float) -> bool:
    return (
        result["exact_normalized_prompt_collision_count"] == 0
        and result["near_duplicate_collision_count"] == 0
        and result["template_family_collision_count"] == 0
        and result["maximum_five_gram_jaccard"] <= threshold
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator-approval-reference",
        required=True,
        choices=[OPERATOR_APPROVAL_REFERENCE],
    )
    args = parser.parse_args()
    outputs = (
        SIDECAR,
        SIDECAR.with_suffix(SIDECAR.suffix + ".sha256"),
        ATTESTATION,
        ATTESTATION.with_suffix(ATTESTATION.suffix + ".sha256"),
    )
    if any(path.exists() for path in outputs):
        raise RuntimeError("V4 blinded scan artifacts already exist; refusing a repeat scan")

    helper = import_helper()
    manifest = load_json(MANIFEST)
    threshold = float(manifest["leakage_policy"]["maximum_train_or_probe_to_evaluation_five_gram_jaccard"])
    bindings = {
        "train": {
            "count": manifest["counts"]["train"],
            "path": TRAIN,
            "sha256": manifest["files"]["train.jsonl"]["sha256"],
        },
        "probes": {
            "count": manifest["counts"]["synthetic_probes"],
            "path": PROBES,
            "sha256": manifest["files"]["synthetic_probes.jsonl"]["sha256"],
        },
        "dev": {
            "count": manifest["evaluation_split_bindings"]["dev"]["count"],
            "path": DEV,
            "sha256": manifest["evaluation_split_bindings"]["dev"]["sha256"],
        },
        "holdout": {
            "count": manifest["evaluation_split_bindings"]["holdout"]["count"],
            "path": HOLDOUT,
            "sha256": manifest["evaluation_split_bindings"]["holdout"]["sha256"],
        },
    }
    source_bindings: dict[str, dict[str, Any]] = {}
    for name, binding in bindings.items():
        observed = sha256_file(binding["path"])
        if observed != binding["sha256"]:
            raise RuntimeError(f"{name} source hash differs")
        source_bindings[name] = {
            "count": binding["count"],
            "path": str(binding["path"].relative_to(ROOT)),
            "sha256": observed,
        }

    train = helper.scan_split(TRAIN, "train", bindings["train"]["count"])
    probes = helper.scan_split(PROBES, "probe", bindings["probes"]["count"])
    dev = helper.scan_split(DEV, "dev", bindings["dev"]["count"])
    holdout = helper.scan_split(HOLDOUT, "holdout", bindings["holdout"]["count"])
    evaluation = dev + holdout
    comparisons = {
        "probe_to_evaluation": collision_summary(helper, probes, evaluation, threshold),
        "train_to_evaluation": collision_summary(helper, train, evaluation, threshold),
        "train_to_probe": collision_summary(helper, train, probes, threshold),
    }
    status = "PASS" if all(passed(item, threshold) for item in comparisons.values()) else "NO-GO"
    method = {
        "answer_fields_accessed": False,
        "helper": {"path": str(HELPER.relative_to(ROOT)), "sha256": sha256_file(HELPER)},
        "human_exposure_to_evaluation_text": False,
        "method_id": "ace2-blinded-cross-split-leakage-v1",
        "near_duplicate_metric": "normalized-token-five-gram-jaccard",
        "normalization": "NFKC-casefold-collapse-whitespace",
        "prompt_only_access": True,
        "raw_prompts_or_fingerprints_emitted": False,
        "scan_ordinal": 1,
        "template_family_method": "normalized-lexical-skeleton-v1",
    }
    sidecar = {
        "aggregate_comparisons": comparisons,
        "fingerprint_sets": {
            "dev": helper.split_commitments(dev),
            "holdout": helper.split_commitments(holdout),
            "probes": helper.split_commitments(probes),
            "train": helper.split_commitments(train),
        },
        "method": method,
        "near_duplicate_threshold_strictly_greater_than": threshold,
        "scanner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256_file(Path(__file__).resolve())},
        "schema_version": 1,
        "source_bindings": source_bindings,
        "status": status,
    }
    sidecar_payload = json_payload(sidecar)
    write_with_companion(SIDECAR, sidecar_payload)

    results = {
        "comparisons": comparisons,
        "raw_prompts_or_fingerprints_included": False,
        "status": status,
    }
    attestation = {
        "input_bindings": {
            name: {"count": item["count"], "sha256": item["sha256"]}
            for name, item in source_bindings.items()
        },
        "method": method,
        "producer": {
            "fresh_non_training_session": True,
            "network_access_used": False,
            "operator_approval_reference": args.operator_approval_reference,
            "training_executed": False,
        },
        "results": results,
        "schema_version": 1,
        "sidecar": {"path": str(SIDECAR.relative_to(ROOT)), "sha256": hashlib.sha256(sidecar_payload).hexdigest()},
    }
    attestation_payload = json_payload(attestation)
    write_with_companion(ATTESTATION, attestation_payload)
    print(json.dumps({"comparisons": comparisons, "status": status}, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
