#!/usr/bin/env python3
"""Authenticated full-model quantized software oracle for Option B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ace2_full_model_fixed_point as fixed_point
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    REPOSITORY,
    REVISION,
    ROOT,
    SNAPSHOT,
    TERMINATION_TOKEN_IDS,
    canonical_bytes,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
    verify_wrapped_contract,
)


DEFAULT_SYSTEM = "You are a helpful assistant. Follow the user's requested response format exactly."


def reconstruct_fixed_model(scales: dict[str, Any]) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    ranges: dict[str, Any] = {}
    for name, metadata in scales["linears"].items():
        head_scales = metadata.get("output_head_scales") or []
        ranges[name] = fixed_point.CalibrationRange(
            input_absmax=float(metadata["input_absmax"]),
            output_absmax=float(metadata["output_absmax"]),
            output_head_absmax=[float(value) * 127.0 for value in head_scales] or None,
        )
    operators = {
        name: fixed_point.ObservedRange(absmax=float(metadata["absmax"]))
        for name, metadata in scales["operators"].items()
    }
    fixed_point.replace_linears(
        model,
        ranges,
        rope_diagnostic_mechanism=fixed_point.ACTIVE_ROPE_MECHANISM,
    )
    fixed_point.replace_fixed_operators(
        model,
        operators,
        rope_diagnostic_mechanism=fixed_point.ACTIVE_ROPE_MECHANISM,
    )
    reconstructed = fixed_point.derived_scale_table(ranges, operators, model)
    require(canonical_bytes(reconstructed) == canonical_bytes(scales), "reconstructed quantized scale table differs")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--contract", type=Path, default=IMAGE_DIR / "oracle_contract.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(1 <= args.max_new_tokens <= 32, "max-new-tokens must be 1..32")
    contract_path = args.contract.resolve()
    contract = verify_wrapped_contract(contract_path)
    require(contract.get("contract_id") == "qwen-instruct-option-b-w4a8-oracle-v1", "oracle contract differs")
    require(contract["source"]["sha256"] == sha256_file(Path(__file__)), "oracle source differs")
    verify_source_contract()
    source = verify_source_snapshot()
    versions = verify_versions()
    for record in contract["authenticated_artifacts"].values():
        path = ROOT / record["path"]
        require(path.is_file(), f"oracle artifact is missing: {record['path']}")
        require(path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"], f"oracle artifact differs: {record['path']}")
    scales_path = CALIBRATION_DIR / "derived_scales.json"
    scales = load_json(scales_path)
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2, "oracle tokenization shape differs")
    model = reconstruct_fixed_model(scales)
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index in range(args.max_new_tokens):
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
            token = int(torch.argmax(logits).item())
            logit = float(logits[token].item())
            output_scale = float(model.lm_head.output_scale)
            logit_s8 = int(round(logit / output_scale))
            require(-128 <= logit_s8 <= 127, "oracle lm-head logit escaped signed A8")
            generated.append(token)
            steps.append(
                {
                    "index": index,
                    "token_id": token,
                    "selected_logit_s8": logit_s8,
                    "selected_logit_scale": output_scale,
                }
            )
            if token in TERMINATION_TOKEN_IDS:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[token]], dtype=input_ids.dtype)], dim=1)
    visible_ids = [token for token in generated if token not in TERMINATION_TOKEN_IDS]
    decoded = tokenizer.decode(visible_ids, skip_special_tokens=True)
    result = {
        "schema_version": 1,
        "classification": "qwen_instruct_option_b_full_model_w4a8_quantized_oracle",
        "status": "PASS",
        "model": {"repository": REPOSITORY, "revision": REVISION, "source": source},
        "oracle_contract": {"path": contract_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(contract_path)},
        "environment": versions,
        "generation_policy": {
            "chat_template": True,
            "add_generation_prompt": True,
            "greedy_argmax": True,
            "sampling": False,
            "use_cache": False,
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
            "max_new_tokens": args.max_new_tokens,
        },
        "prompt": {
            "user_utf8_bytes": len(args.prompt.encode()),
            "user_sha256": sha256_bytes(args.prompt.encode()),
            "system_utf8_bytes": len(args.system_prompt.encode()),
            "system_sha256": sha256_bytes(args.system_prompt.encode()),
            "chat_template_token_count": int(input_ids.shape[1]) - len(visible_ids),
        },
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "steps": steps,
        "scope_guards": {
            "accelerator_executed": False,
            "demo_retargeted": False,
            "network_access_performed": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(result))
    print(f"ACE2_QWEN_INSTRUCT_W4A8_ORACLE_PASS tokens={len(generated)} output={output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
