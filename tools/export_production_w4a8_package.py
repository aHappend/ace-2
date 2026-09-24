#!/usr/bin/env python3
"""Offline exporter for the production Qwen2.5-0.5B ACE2W4M1 package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch import nn

try:
    from . import model_hardware_contract as hardware
    from . import qualify_stage1_w4a8_software as runtime
    from . import run_file_backed_w4a8_chat as chat
    from . import w4a8_full_model_evaluator as evaluator
except ImportError:
    import model_hardware_contract as hardware
    import qualify_stage1_w4a8_software as runtime
    import run_file_backed_w4a8_chat as chat
    import w4a8_full_model_evaluator as evaluator


ModelBuilder = Callable[[dict[str, Any], dict[str, Any]], nn.Module]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_calibration_binding(
    contract: dict[str, Any],
    manifest: dict[str, Any],
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    frozen = contract["evaluation_contract"]["calibration_set"]
    expected = {
        "config": frozen["config"],
        "field": "text",
        "indices": frozen["record_indices"],
        "repository": frozen["repository"],
        "revision": frozen["revision"],
        "split": frozen["split"],
        "token_limit": frozen["token_limit"],
    }
    calibration = manifest["datasets"]["c4_calibration"]
    require(
        calibration == expected,
        "frozen C4 calibration manifest differs from the software contract",
    )
    require(
        manifest["model"]
        == {
            "repository": model_spec["repository"],
            "revision": model_spec["revision"],
        },
        "frozen prompt manifest model identity differs",
    )
    return calibration


def validate_tokenizer(tokenizer: Any, contract: dict[str, Any]) -> None:
    template = tokenizer.chat_template
    require(isinstance(template, str), "pinned tokenizer chat template is missing")
    expected = contract["model_identities"]["qwen2.5-0.5b-instruct"][
        "chat_template_sha256"
    ]
    require(
        hashlib.sha256(template.encode("utf-8")).hexdigest() == expected,
        "pinned tokenizer chat template differs",
    )


def build_grouped_model(
    descriptor: dict[str, Any],
    contract: dict[str, Any],
) -> nn.Module:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_spec = chat.base_model_spec(contract)
    chat.verify_local_model(contract, model_spec)
    chat.set_determinism()
    quality_config = evaluator.load_json(evaluator.QUALITY_CONFIG_PATH)
    evaluator.validate_contract_support(contract, quality_config)
    manifest = evaluator.require_canonical_json(evaluator.PROMPT_MANIFEST_PATH)
    calibration_spec = validate_calibration_binding(
        contract,
        manifest,
        model_spec,
    )
    tokenizer = evaluator.load_tokenizer(model_spec)
    validate_tokenizer(tokenizer, contract)
    calibration_texts = evaluator.selected_local_texts(calibration_spec)
    generation = contract["shared_w4a8_contract"]["generation"]
    calibration_inputs = evaluator._tokenize_calibration(
        tokenizer,
        calibration_texts,
        generation,
        calibration_spec["token_limit"],
    )

    source_model = evaluator.load_model(model_spec)
    ranges, operator_ranges = evaluator.fixed.calibrate(
        source_model,
        calibration_inputs,
    )
    fixed_model = copy.deepcopy(source_model)
    candidate = evaluator.candidate_spec(contract, "c01-mse-clip-grid")
    evaluator.replace_linears_for_candidate(
        fixed_model,
        ranges,
        candidate,
        rope_diagnostic_mechanism=evaluator.fixed.ACTIVE_ROPE_MECHANISM,
    )
    evaluator.replace_fixed_operators_for_candidate(
        fixed_model,
        operator_ranges,
        candidate,
        rope_diagnostic_mechanism=evaluator.fixed.ACTIVE_ROPE_MECHANISM,
    )
    runtime.install_candidate(fixed_model, source_model)
    return fixed_model.eval()


def execute(
    destination: Path,
    *,
    model_builder: ModelBuilder = build_grouped_model,
) -> dict[str, Any]:
    require(isinstance(destination, Path), "package destination must be a Path")
    require(destination.parent.is_dir(), "package destination directory is missing")
    if destination.exists():
        raise FileExistsError(destination)

    contract, contract_sha256 = chat.load_contract()
    descriptor = hardware.load_descriptor("qwen2.5-0.5b")
    model = model_builder(descriptor, contract)
    runtime.export_installed_full_qwen_payload_package(
        model,
        destination,
        descriptor,
    )
    package = chat.validate_package(destination, descriptor)
    return {
        "byte_count": len(package),
        "contract_sha256": contract_sha256,
        "member_count": len(runtime.full_qwen_payload_names()),
        "model_id": "qwen2.5-0.5b-instruct",
        "package_path": str(destination.resolve()),
        "package_sha256": hashlib.sha256(package).hexdigest(),
        "schema_version": 1,
        "status": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one canonical grouped-W4A8 package from pinned local model "
            "and calibration inputs without network access or replacement."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = execute(args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ACE2_W4A8_EXPORT_ERROR: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
