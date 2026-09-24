#!/usr/bin/env python3
"""Run the non-consuming pinned-model W4A8 candidate installation smoke."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

import qualify_stage1_w4a8_software as quality
from qwen_instruct_option_b import SNAPSHOT, verify_source_snapshot
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM, reconstruct_fixed_model


SMOKE_PROMPT = "Exercise one software integration step."


def expect_rejection(
    label: str,
    operation: Callable[[], Any],
    expected_message: str,
) -> dict[str, str]:
    try:
        operation()
    except (RuntimeError, TypeError, ValueError) as error:
        quality.require(
            expected_message in str(error),
            f"{label} rejected for an unexpected reason: {error}",
        )
        return {"case": label, "error": str(error)}
    raise RuntimeError(f"{label} was not rejected")


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().to(torch.float64).numpy().tobytes()
    ).hexdigest()


def malformed_contract_checks(
    q_projection: quality.GroupedW4A8Linear,
    down_projection: quality.GroupedW4A8Linear,
) -> list[dict[str, str]]:
    carrier = torch.zeros((1, 1, q_projection.in_features), dtype=torch.float32)
    checks = [
        expect_rejection(
            "fractional_float_carrier",
            lambda: q_projection.forward_hardware_input(
                carrier.index_fill(-1, torch.tensor([0]), 0.5)
            ),
            "exact integers",
        ),
        expect_rejection(
            "nan_float_carrier",
            lambda: q_projection.forward_hardware_input(
                carrier.index_fill(-1, torch.tensor([0]), torch.nan)
            ),
            "finite",
        ),
        expect_rejection(
            "infinite_float_carrier",
            lambda: q_projection.forward_hardware_input(
                carrier.index_fill(-1, torch.tensor([0]), torch.inf)
            ),
            "finite",
        ),
        expect_rejection(
            "overflow_float_carrier",
            lambda: q_projection.forward_hardware_input(
                carrier.index_fill(-1, torch.tensor([0]), 128.0)
            ),
            "outside signed int8",
        ),
        expect_rejection(
            "wrong_rank_carrier",
            lambda: q_projection.forward_hardware_input(carrier[0]),
            "activation geometry",
        ),
        expect_rejection(
            "wrong_width_carrier",
            lambda: q_projection.forward_hardware_input(carrier[..., :-1]),
            "input width",
        ),
        expect_rejection(
            "unsupported_carrier_dtype",
            lambda: q_projection.forward_hardware_input(carrier.to(torch.int16)),
            "exact float container",
        ),
    ]
    original_records = q_projection.hardware_input_scale32_records
    q_projection.hardware_input_scale32_records = None
    try:
        checks.append(
            expect_rejection(
                "missing_static_scale32",
                lambda: q_projection.forward_hardware_input(carrier),
                "metadata is missing",
            )
        )
    finally:
        q_projection.hardware_input_scale32_records = original_records
    q_projection.hardware_input_scale32_records = original_records.to(torch.int32)
    try:
        checks.append(
            expect_rejection(
                "wrong_dtype_static_scale32",
                lambda: q_projection.forward_hardware_input(carrier),
                "not int64",
            )
        )
    finally:
        q_projection.hardware_input_scale32_records = original_records
    q_projection.hardware_input_scale32_records = original_records[:-1]
    try:
        checks.append(
            expect_rejection(
                "wrong_shape_static_scale32",
                lambda: q_projection.forward_hardware_input(carrier),
                "record geometry",
            )
        )
    finally:
        q_projection.hardware_input_scale32_records = original_records
    dynamic_carrier = torch.zeros(
        (1, 1, down_projection.in_features), dtype=torch.int8
    )
    dynamic_records = torch.full(
        (1, 1, down_projection.groups),
        quality.ceil_scale32_from_float(1.0),
        dtype=torch.int64,
    )
    checks.extend(
        (
            expect_rejection(
                "wrong_shape_dynamic_scale32",
                lambda: down_projection.forward_grouped_input(
                    dynamic_carrier, dynamic_records[..., :-1]
                ),
                "Scale32 geometry",
            ),
            expect_rejection(
                "wrong_dtype_dynamic_scale32",
                lambda: down_projection.forward_grouped_input(
                    dynamic_carrier, dynamic_records.to(torch.int32)
                ),
                "not int64",
            ),
            expect_rejection(
                "invalid_dynamic_scale32",
                lambda: down_projection.forward_grouped_input(
                    dynamic_carrier, torch.full_like(dynamic_records, -1)
                ),
                "Scale32",
            ),
        )
    )
    return checks


def run_smoke(output_path: Path) -> None:
    quality.require(not output_path.exists(), "immutable smoke evidence already exists")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    source_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_model = reconstruct_fixed_model(
        quality.load_json(quality.DERIVED_SCALES_PATH)
    ).eval()
    quality.install_candidate(fixed_model, source_model)
    representative = {
        "q_proj": fixed_model.model.layers[0].self_attn.q_proj,
        "k_proj": fixed_model.model.layers[0].self_attn.k_proj,
        "v_proj": fixed_model.model.layers[0].self_attn.v_proj,
        "o_proj": fixed_model.model.layers[0].self_attn.o_proj,
        "gate_proj": fixed_model.model.layers[0].mlp.gate_proj,
        "up_proj": fixed_model.model.layers[0].mlp.up_proj,
        "down_proj": fixed_model.model.layers[0].mlp.down_proj,
    }
    quality.require(
        all(isinstance(module, quality.GroupedW4A8Linear) for module in representative.values()),
        "candidate wrappers are not installed on every projection family",
    )
    sequence = quality.render(tokenizer, DEFAULT_SYSTEM, SMOKE_PROMPT)
    with torch.inference_mode():
        logits = fixed_model(input_ids=sequence, use_cache=False).logits[:, -1, :]
    quality.require(bool(torch.all(torch.isfinite(logits))), "smoke logits are not finite")
    projection_families: dict[str, dict[str, Any]] = {}
    for name, module in representative.items():
        if name == "down_proj":
            call_count = module.dynamic_input_calls
            association_count = module.dynamic_input_calls
        else:
            call_count = module.hardware_input_calls
            association_count = module.bound_scale32_associations
        quality.require(call_count > 0, f"{name} was not exercised through the model call path")
        quality.require(
            association_count == call_count,
            f"{name} did not preserve Scale32 association",
        )
        projection_families[name] = {
            "bound_scale32_associations": association_count,
            "call_count": call_count,
            "dynamic_scale32": name == "down_proj",
            "exact_float_carrier_conversions": module.exact_float_carrier_conversions,
            "groups": module.groups,
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
    for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
        quality.require(
            projection_families[name]["exact_float_carrier_conversions"] > 0,
            f"{name} did not exercise the exact float carrier",
        )
    malformed = malformed_contract_checks(
        representative["q_proj"], representative["down_proj"]
    )
    evidence = {
        "forward": {
            "finite_logits": True,
            "input_tokens": int(sequence.numel()),
            "logits_sha256": tensor_sha256(logits),
            "selected_token_id": int(torch.argmax(logits)),
        },
        "malformed_contract": {
            "all_rejected": True,
            "cases": malformed,
        },
        "model_source": verify_source_snapshot(),
        "projection_families": projection_families,
        "qualification_source": quality.file_record(Path(quality.__file__).resolve()),
        "schema_version": 1,
        "smoke_source": quality.file_record(Path(__file__).resolve()),
        "status": "PASSED",
    }
    quality.write_exclusive(output_path, evidence)
    print(f"ACE2_W4A8_CANDIDATE_SMOKE status=PASSED output={output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_smoke(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
