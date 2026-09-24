#!/usr/bin/env python3
"""Fail-closed audit of the frozen, marker-free ACE-2 LoRA V5 package."""

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
GENERATOR = ROOT / "tools/freeze_qwen25_05b_bf16_lora_product_v5.py"
DATASET = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v5"
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v5"
CALIBRATION = ROOT / "research/diagnostics/qwen25_lora_v5_selector_calibration.json"
SELF_TEST = ROOT / "research/diagnostics/qwen25_lora_v5_package_self_test.json"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5"


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


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not companion.is_file():
        return False
    fields = companion.read_text(encoding="ascii").split()
    return len(fields) == 2 and fields[0] == sha256_file(path) and fields[1] == path.name


def import_generator() -> Any:
    spec = importlib.util.spec_from_file_location("ace2_v5_generator_for_audit", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import V5 generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_payload() -> dict[str, Any]:
    generator = import_generator()
    generator.check_final()
    manifest = load_json(DATASET / "dataset_manifest.json")
    recipe = load_json(DATASET / "training_recipe.json")
    calibration = load_json(CALIBRATION)
    self_test = load_json(SELF_TEST)
    markers = list((ROOT / "build").glob("qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/attempt-*/ATTEMPT_CONSUMPTION_MARKER.json"))
    core = [
        DATASET / name
        for name in (
            "train.jsonl",
            "patch.jsonl",
            "selector.jsonl",
            "freeze_contract.json",
            "training_recipe.json",
            "dataset_manifest.json",
            "execution_package_manifest.json",
            "freeze_manifest.json",
        )
    ] + [PACKAGE / "frozen_config.json", CALIBRATION, SELF_TEST]
    checks = {
        "all_companions_match": all(companion_matches(path) for path in core),
        "attempt_marker_absent": not markers,
        "calibration_direction_gate_pass": calibration.get("status") == "PASS" and min(calibration["match_counts"].values()) >= 6,
        "candidate_policy_no_fallback": recipe["checkpoint_policy"]["candidate_steps"] == [10, 20] and recipe["early_stop"]["fallback_allowed"] is False,
        "counts_exact": manifest["counts"] == {"additive_patch": 40, "selector_cases": 56, "train": 320, "v1_backbone": 280},
        "official_namespace_absent": not RUN_ROOT.exists(),
        "parent_adapter_exact": sha256_file(generator.V1_PARENT_ADAPTER) == generator.V1_PARENT_ADAPTER_SHA256,
        "self_test_pass": self_test.get("status") == "PASS",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "Non-consuming V5 freeze audit; no model namespace, attempt marker, training, official evaluation, merge, RTL, or PPA execution.",
        "failure_taxonomy": None if not failed else "PACKAGE_INTERLOCK_FAILURE",
        "first_failed_gate": failed[0] if failed else None,
        "official_namespace_exists": RUN_ROOT.exists(),
        "schema_version": 1,
        "status": "PASS" if not failed else "NO_GO",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/freeze-audit.json")
    args = parser.parse_args()
    output = ROOT / args.output
    payload = json.dumps(audit_payload(), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise RuntimeError("freeze audit output already exists; refusing overwrite")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    with output.with_suffix(output.suffix + ".sha256").open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{hashlib.sha256(payload).hexdigest()}  {output.name}\n")
    if load_json(output)["status"] != "PASS":
        raise RuntimeError(f"V5 freeze audit failed: {load_json(output)['first_failed_gate']}")
    print("ACE2_LORA_V5_FREEZE_AUDIT_PASS")


if __name__ == "__main__":
    main()
