#!/usr/bin/env python3
"""Read-only CPU attribution for the sealed arbitrary-text RTL attempt."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from peft import PeftModel
import peft
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/attempt-0001"
    / "checkpoints/checkpoint-176"
)
CHECKPOINT_IDENTITY = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/attempt-0001"
    / "probe-selection/epoch-4/checkpoint_identity.json"
)
ATTEMPT = ROOT / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0001"
SEALED_TOKENS = [35167, 98495, 55318, 1451]
TERMINATION_IDS = {151643, 151645}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_fingerprint() -> dict[str, str]:
    return {
        name: sha256_file(ATTEMPT / name)
        for name in ("SHA256SUMS", "manifest.json", "run_summary.json")
    }


def decode_piece(tokenizer: Any, token_id: int) -> str:
    if token_id >= len(tokenizer):
        return "<UNMAPPED>"
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def stable_top_ids(logits: torch.Tensor, count: int) -> list[int]:
    return [int(value) for value in torch.argsort(logits, descending=True, stable=True)[:count]]


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    value = logits[token_id]
    ids = torch.arange(logits.numel(), device=logits.device)
    better = (logits > value) | ((logits == value) & (ids < token_id))
    return int(better.sum().item()) + 1


def manual_trace(
    model: Any,
    tokenizer: Any,
    initial_ids: list[int],
    forced_tokens: list[int] | None = None,
) -> dict[str, Any]:
    input_ids = torch.tensor([initial_ids], dtype=torch.long)
    past = None
    selected_tokens: list[int] = []
    steps: list[dict[str, Any]] = []
    with torch.inference_mode():
        for step in range(4):
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones((1, len(initial_ids) + step), dtype=torch.long),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = output.logits[0, -1].float().cpu()
            greedy = int(torch.argmax(logits).item())
            selected = forced_tokens[step] if forced_tokens is not None else greedy
            top_ids = stable_top_ids(logits, 3)
            cache = output.past_key_values
            steps.append(
                {
                    "step": step,
                    "input_token_ids_this_call": [int(value) for value in input_ids[0]],
                    "cache_sequence_length_after_call": (
                        int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else None
                    ),
                    "bf16_argmax_token_id": greedy,
                    "bf16_argmax_piece": decode_piece(tokenizer, greedy),
                    "selected_token_id": selected,
                    "selected_piece": decode_piece(tokenizer, selected),
                    "selected_rank_in_bf16": rank_of(logits, selected),
                    "selected_bf16_logit": float(logits[selected]),
                    "bf16_top_candidates": [
                        {
                            "token_id": token_id,
                            "piece": decode_piece(tokenizer, token_id),
                            "logit": float(logits[token_id]),
                        }
                        for token_id in top_ids
                    ],
                }
            )
            selected_tokens.append(selected)
            past = cache
            if selected in TERMINATION_IDS:
                break
            input_ids = torch.tensor([[selected]], dtype=torch.long)
    return {
        "generated_token_ids": selected_tokens,
        "decoded_text": tokenizer.decode(
            selected_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "steps": steps,
    }


def main() -> None:
    before = sealed_fingerprint()
    adapter_config = json.loads((CHECKPOINT / "adapter_config.json").read_text())
    checkpoint_identity = json.loads(CHECKPOINT_IDENTITY.read_text())
    base_path = Path(adapter_config["base_model_name_or_path"])

    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        attn_implementation="eager",
        device_map=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        model,
        CHECKPOINT,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    model.eval()

    raw_ids = tokenizer.encode("Hello", add_special_tokens=False)
    if raw_ids != [9707]:
        raise RuntimeError(f"raw prompt tokenization changed: {raw_ids}")
    native = manual_trace(model, tokenizer, raw_ids)
    aligned = manual_trace(model, tokenizer, raw_ids, SEALED_TOKENS)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=torch.tensor([raw_ids], dtype=torch.long),
            attention_mask=torch.ones((1, len(raw_ids)), dtype=torch.long),
            do_sample=False,
            eos_token_id=sorted(TERMINATION_IDS),
            max_new_tokens=4,
            num_beams=1,
            pad_token_id=151643,
            use_cache=True,
        )
    generate_ids = [int(value) for value in generated[0, len(raw_ids) :]]
    if generate_ids != native["generated_token_ids"]:
        raise RuntimeError("manual KV loop differs from model.generate")

    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    chat = manual_trace(model, tokenizer, [int(value) for value in chat_ids])

    w4_steps: list[dict[str, Any]] = []
    for step, selected_token in enumerate(SEALED_TOKENS):
        head = ATTEMPT / f"heads/step-{step:02d}"
        values = torch.frombuffer(
            bytearray((head / "tensors/lm_head_output_s8.bin").read_bytes()),
            dtype=torch.int8,
        ).clone().to(torch.int16)
        scale = struct.unpack(
            "<d", (head / "tensors/lm_head_output_scale_f64le.bin").read_bytes()
        )[0]
        top_ids = sorted(range(values.numel()), key=lambda token_id: (-int(values[token_id]), token_id))
        execution = json.loads((head / "head_execution.json").read_text())
        bf16_argmax = aligned["steps"][step]["bf16_argmax_token_id"]
        if top_ids[0] != selected_token:
            raise RuntimeError(f"fixed-point top token changed at step {step}")
        head_result = execution["lm_head"]
        if not head_result["selected_token_agreement"] or head_result["top_token"] != selected_token:
            raise RuntimeError(f"RTL/reference selector agreement changed at step {step}")
        w4_steps.append(
            {
                "step": step,
                "selected_token_id": selected_token,
                "selected_piece": decode_piece(tokenizer, selected_token),
                "selected_logit_s8": int(values[selected_token]),
                "output_scale": scale,
                "top_minus_runner_up_s8": int(values[top_ids[0]]) - int(values[top_ids[1]]),
                "sealed_token_rank_in_bf16": aligned["steps"][step]["selected_rank_in_bf16"],
                "bf16_argmax_token_id": bf16_argmax,
                "bf16_argmax_piece": decode_piece(tokenizer, bf16_argmax),
                "bf16_argmax_rank_in_w4": top_ids.index(bf16_argmax) + 1,
                "w4_top_candidates": [
                    {
                        "token_id": token_id,
                        "piece": decode_piece(tokenizer, token_id),
                        "logit_s8": int(values[token_id]),
                    }
                    for token_id in top_ids[:3]
                ],
                "rtl_selected_token_agreement": True,
            }
        )

    after = sealed_fingerprint()
    if before != after:
        raise RuntimeError("sealed attempt fingerprint changed during diagnostic")

    result = {
        "status": "PASS_BOUNDED_READ_ONLY_DIAGNOSTIC",
        "software": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "contract": {
            "prompt_utf8_hex": b"Hello".hex(),
            "prompt_sha256": hashlib.sha256(b"Hello").hexdigest(),
            "prompt_mode": "encode(add_special_tokens=False)",
            "prompt_token_ids": raw_ids,
            "checkpoint_tree_sha256": checkpoint_identity["tree_sha256"],
            "adapter_config_sha256": sha256_file(CHECKPOINT / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(CHECKPOINT / "adapter_model.safetensors"),
            "tokenizer_json_sha256": sha256_file(base_path / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(base_path / "tokenizer_config.json"),
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "termination_token_ids": sorted(TERMINATION_IDS),
            "max_new_tokens": 4,
            "use_cache": True,
        },
        "native_raw_prompt": native,
        "native_generate_matches_manual_kv_loop": True,
        "bf16_on_sealed_prefixes": aligned,
        "sealed_w4a8_rtl": {
            "generated_token_ids": SEALED_TOKENS,
            "decoded_text": tokenizer.decode(
                SEALED_TOKENS,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "steps": w4_steps,
        },
        "chat_template_sensitivity": {
            "prompt_token_ids": [int(value) for value in chat_ids],
            "native": chat,
        },
        "sealed_attempt_fingerprint_before_and_after": before,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
