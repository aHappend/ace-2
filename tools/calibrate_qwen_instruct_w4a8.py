#!/usr/bin/env python3
"""Derive a fresh deterministic Instruct-specific W4A8 scale contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ace2_full_model_fixed_point as fixed_point
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    CALIBRATION_ID,
    CHAT_TEMPLATE_SHA256,
    REPOSITORY,
    REVISION,
    ROOT,
    SNAPSHOT,
    SOURCE_CONTRACT,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
    write_json,
)


TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's requested response format exactly."
CALIBRATION_USERS = (
    "Reply with exactly two lowercase words: amber compass",
    "Give the result of 17 plus 25 as a short sentence.",
    "Summarize this in one clause: A small satellite crossed the evening sky while the city lights came on.",
    "Answer in English with three words: What color is a clear daytime sky?",
)


def prompt_records(tokenizer: object) -> tuple[list[torch.Tensor], list[dict[str, object]]]:
    prompts: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    for index, user in enumerate(CALIBRATION_USERS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        require(isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[0] == 1, "chat tokenization shape differs")
        raw_ids = [int(value) for value in ids[0].tolist()]
        prompts.append(ids)
        records.append(
            {
                "ordinal": index,
                "messages": messages,
                "token_count": len(raw_ids),
                "token_ids": raw_ids,
                "token_ids_sha256": sha256_bytes(
                    b"".join(value.to_bytes(4, "little") for value in raw_ids)
                ),
            }
        )
    return prompts, records


def write_sums(directory: Path, names: list[str]) -> None:
    rows = [f"{sha256_file(directory / name)}  {name}" for name in sorted(names)]
    (directory / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CALIBRATION_DIR)
    parser.add_argument("--generated-at-utc", required=True)
    args = parser.parse_args()
    require(TIMESTAMP.fullmatch(args.generated_at_utc) is not None, "timestamp must use YYYY-MM-DDTHH:MM:SSZ")
    output = args.output.resolve()
    require(not output.exists(), f"calibration output already exists: {output}")
    source_contract = verify_source_contract()
    source = verify_source_snapshot()
    versions = verify_versions()
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    require(sha256_bytes(tokenizer.chat_template.encode()) == CHAT_TEMPLATE_SHA256, "loaded chat template differs")
    prompts, prompt_manifest = prompt_records(tokenizer)
    contract = {
        "schema_version": 1,
        "contract_id": CALIBRATION_ID,
        "generated_at_utc": args.generated_at_utc,
        "status": "FROZEN_BEFORE_CALIBRATION",
        "source_model": source,
        "source_contract": file_record(SOURCE_CONTRACT),
        "chat": {
            "system_prompt": SYSTEM_PROMPT,
            "add_generation_prompt": True,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "prompts": prompt_manifest,
        },
        "quantization": {
            "weight": "symmetric per-output-channel signed W4, absmax/7, round-to-nearest-even, clamp[-8,7]",
            "activation": "static signed A8, observed BF16 absmax/127, round-to-nearest-even, clamp[-128,127]",
            "rope_mechanism": fixed_point.ACTIVE_ROPE_MECHANISM,
            "accepted_integer_semantics_source": file_record(ROOT / "tools/ace2_full_model_fixed_point.py"),
        },
        "environment": {
            "python": platform.python_version(),
            "packages": versions,
            "device": "cpu",
            "offline_only": True,
            "trust_remote_code": False,
        },
        "calibrator": file_record(Path(__file__)),
    }
    output.mkdir(parents=True)
    write_json(output / "calibration_contract.json", contract)

    fixed_point.seed_everything(load_json(ROOT / "benchmark/quality/QUALITY_CONFIG.json"))
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    ranges, operator_ranges = fixed_point.calibrate(model, prompts)
    fixed_point.replace_linears(
        model,
        ranges,
        rope_diagnostic_mechanism=fixed_point.ACTIVE_ROPE_MECHANISM,
    )
    fixed_point.replace_fixed_operators(
        model,
        operator_ranges,
        rope_diagnostic_mechanism=fixed_point.ACTIVE_ROPE_MECHANISM,
    )
    scales = fixed_point.derived_scale_table(ranges, operator_ranges, model)
    require(len(scales.get("linears", {})) == 169, "linear scale count differs")
    require(len(scales.get("operators", {})) == 122, "operator scale count differs")
    require(len(scales.get("attention", {})) == 24, "attention scale count differs")
    write_json(output / "derived_scales.json", scales)
    report = {
        "schema_version": 1,
        "mission_id": CALIBRATION_ID,
        "generated_at_utc": args.generated_at_utc,
        "status": "PASS_INSTRUCT_SPECIFIC_W4A8_CALIBRATION",
        "source_model": {"repository": REPOSITORY, "revision": REVISION},
        "contract": file_record(output / "calibration_contract.json"),
        "derived_scales": file_record(output / "derived_scales.json"),
        "counts": {"linears": 169, "operators": 122, "attention": 24, "prompts": len(prompts)},
        "base_scale_artifact_reused": False,
        "base_model_substitution_performed": False,
        "network_access_performed": False,
    }
    write_json(output / "calibration_report.json", report)
    write_sums(output, ["calibration_contract.json", "calibration_report.json", "derived_scales.json"])
    print(
        "ACE2_QWEN_INSTRUCT_W4A8_CALIBRATION_PASS "
        f"scales_sha256={sha256_file(output / 'derived_scales.json')} output={output.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
