#!/usr/bin/env python3
"""Execute pre-PTQ controls for the frozen full-model W4A8 contract."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "design/QWEN25_05B_W4A8_FULL_MODEL_SOFTWARE_CONTRACT_V1.json"
CONTRACT_SHA_PATH = CONTRACT_PATH.with_suffix(CONTRACT_PATH.suffix + ".sha256")
QUALITY_CONFIG_PATH = ROOT / "benchmark/quality/QUALITY_CONFIG.json"
MODEL_ALIASES = ("base", "checkpoint-176")
SEMANTIC_PASS_STATUS = "PASS_SEMANTIC_EQUIVALENCE"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def load_contract() -> tuple[dict[str, Any], str]:
    require(CONTRACT_PATH.is_file(), "software contract is missing")
    contract = load_json(CONTRACT_PATH)
    require(CONTRACT_PATH.read_bytes() == canonical_bytes(contract), "contract JSON is not canonical")
    fields = CONTRACT_SHA_PATH.read_text(encoding="utf-8").strip().split()
    require(
        len(fields) == 2 and fields[1] == CONTRACT_PATH.name,
        "contract SHA256 companion differs",
    )
    contract_sha256 = sha256_file(CONTRACT_PATH)
    require(fields[0] == contract_sha256, "contract SHA256 differs")
    require(
        contract["status"] == "FROZEN_PRE_PTQ_V2_CONTROLS_PENDING_FRESH_L2",
        "runner requires the corrected v2 pre-PTQ contract",
    )
    for label, record in contract["frozen_references"].items():
        path = ROOT / record["path"]
        require(path.is_file(), f"frozen reference is missing: {label}")
        require(sha256_file(path) == record["sha256"], f"frozen reference differs: {label}")
    return contract, contract_sha256


def snapshot_path(repository: str, revision: str) -> Path:
    repository_key = repository.replace("/", "--")
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / f"models--{repository_key}"
        / "snapshots"
        / revision
    )


def model_spec(contract: dict[str, Any], alias: str) -> dict[str, Any]:
    base = contract["model_identities"]["qwen2.5-0.5b-instruct"]
    policy = contract["artifact_policy"]
    if alias == "base":
        return {
            "alias": alias,
            "artifact_root": policy["base_root"],
            "contract_identity": base,
            "model_id": base["model_id"],
            "repository": base["repository"],
            "revision": base["revision"],
        }
    require(alias == "checkpoint-176", f"unsupported model alias: {alias}")
    checkpoint = contract["model_identities"]["ace2_lora_v4_checkpoint176"]
    return {
        "adapter": checkpoint["adapter"],
        "alias": alias,
        "artifact_root": policy["checkpoint176_root"],
        "contract_identity": checkpoint,
        "model_id": checkpoint["model_id"],
        "repository": base["repository"],
        "revision": base["revision"],
    }


def verify_local_model_inputs(contract: dict[str, Any], spec: dict[str, Any]) -> None:
    base = contract["model_identities"]["qwen2.5-0.5b-instruct"]
    snapshot = snapshot_path(spec["repository"], spec["revision"])
    require(snapshot.is_dir(), "pinned local model snapshot is missing")
    for filename, expected in base["files"].items():
        path = snapshot / filename
        require(path.is_file(), f"pinned model file is missing: {filename}")
        require(sha256_file(path) == expected, f"pinned model file differs: {filename}")
    if spec["alias"] == "checkpoint-176":
        adapter = spec["adapter"]
        for field in ("adapter_config", "adapter_model"):
            record = adapter[field]
            path = ROOT / record["path"]
            require(path.is_file(), f"checkpoint-176 {field} is missing")
            require(sha256_file(path) == record["sha256"], f"checkpoint-176 {field} differs")


def set_determinism() -> dict[str, Any]:
    quality = load_json(QUALITY_CONFIG_PATH)
    seeds = quality["determinism"]
    os.environ["PYTHONHASHSEED"] = str(seeds["python_seed"])
    random.seed(seeds["python_seed"])
    np.random.seed(seeds["numpy_seed"])
    torch.manual_seed(seeds["torch_seed"])
    torch.use_deterministic_algorithms(seeds["torch_deterministic_algorithms"])
    return seeds


def load_tokenizer(contract: dict[str, Any], spec: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        spec["repository"],
        revision=spec["revision"],
        local_files_only=True,
    )
    resolved = tokenizer.init_kwargs.get("_commit_hash")
    require(resolved in (None, spec["revision"]), "tokenizer revision differs")
    template = tokenizer.chat_template
    require(isinstance(template, str), "pinned tokenizer chat template is missing")
    expected = contract["model_identities"]["qwen2.5-0.5b-instruct"][
        "chat_template_sha256"
    ]
    require(sha256_bytes(template.encode("utf-8")) == expected, "chat template hash differs")
    return tokenizer


def load_model(spec: dict[str, Any]) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        spec["repository"],
        revision=spec["revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,
    )
    resolved = getattr(model.config, "_commit_hash", None)
    require(resolved == spec["revision"], "model revision differs")
    if spec["alias"] == "checkpoint-176":
        adapter_path = ROOT / spec["adapter"]["adapter_config"]["path"]
        model = PeftModel.from_pretrained(
            model,
            adapter_path.parent,
            is_trainable=False,
            local_files_only=True,
        ).merge_and_unload()
    model = model.eval()
    bad_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16
    ]
    require(not bad_parameters, f"non-BF16 model parameters remain: {bad_parameters[:4]}")
    return model


def tensor_bf16_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    digest = hashlib.sha256()
    digest.update(b"torch.bfloat16\0")
    digest.update(len(tensor.shape).to_bytes(4, "big"))
    for dimension in tensor.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(tensor.view(torch.uint16).numpy().tobytes(order="C"))
    return digest.hexdigest()


def render_prompt_ids(
    tokenizer: Any,
    system_message: str,
    user_message: str,
) -> torch.Tensor:
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )
    require(input_ids.ndim == 2 and input_ids.shape[0] == 1, "chat template shape differs")
    return input_ids.to(dtype=torch.long, device="cpu")


def decode_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: dict[str, Any],
    generation: dict[str, Any],
    *,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    input_ids = render_prompt_ids(
        tokenizer,
        generation["canonical_system_message"],
        prompt["user"],
    )
    sequence = input_ids.clone()
    generated: list[int] = []
    selected_tokens: list[int] = []
    logit_hashes: list[str] = []
    cache_steps: list[dict[str, Any]] = []
    past_key_values = None
    cache_input = input_ids
    limit = max_new_tokens if max_new_tokens is not None else generation["max_new_tokens"]
    require(1 <= limit <= generation["max_new_tokens"], "diagnostic generation limit differs")
    termination_reason = "max_new_tokens"

    with torch.inference_mode():
        for step_index in range(limit):
            attention_mask = torch.ones_like(sequence, dtype=torch.long)
            cache_output = model(
                input_ids=cache_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = cache_output.past_key_values
            cache_logits = cache_output.logits[:, -1, :].to(torch.bfloat16).contiguous()
            prefix_output = model(
                input_ids=sequence,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            prefix_logits = prefix_output.logits[:, -1, :].to(torch.bfloat16).contiguous()
            cache_hash = tensor_bf16_sha256(cache_logits)
            prefix_hash = tensor_bf16_sha256(prefix_logits)
            logit_exact = torch.equal(cache_logits, prefix_logits)
            mismatch_count = int((cache_logits != prefix_logits).sum().item())
            max_abs_difference = float(
                (cache_logits.float() - prefix_logits.float()).abs().max().item()
            )
            selected = int(torch.argmax(cache_logits, dim=-1).item())
            prefix_selected = int(torch.argmax(prefix_logits, dim=-1).item())
            require(
                selected == prefix_selected,
                f"cache/full-prefix selected tokens differ for {prompt['prompt_id']} step {step_index}",
            )
            selected_tokens.append(selected)
            logit_hashes.append(cache_hash)
            cache_steps.append(
                {
                    "cache_bf16_logit_tensor_sha256": cache_hash,
                    "full_prefix_selected_token": prefix_selected,
                    "full_prefix_bf16_logit_tensor_sha256": prefix_hash,
                    "logit_exact": logit_exact,
                    "logit_max_abs_difference": max_abs_difference,
                    "logit_mismatch_count": mismatch_count,
                    "selected_token_exact": selected == prefix_selected,
                    "step_index": step_index,
                }
            )
            generated.append(selected)
            next_token = torch.tensor([[selected]], dtype=torch.long)
            sequence = torch.cat((sequence, next_token), dim=1)
            cache_input = next_token
            if selected in generation["termination_token_ids"]:
                termination_reason = f"termination_token_id:{selected}"
                break

    identity = {
        "chat_template_input_ids": input_ids.reshape(-1).tolist(),
        "generated_token_ids": generated,
        "per_step_full_bf16_logit_tensor_sha256": logit_hashes,
        "per_step_selected_token": selected_tokens,
        "termination_reason": termination_reason,
    }
    return {
        "cache_equivalence": cache_steps,
        "decoded_text": tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=generation["clean_up_tokenization_spaces"],
        ),
        "identity": identity,
        "prompt_id": prompt["prompt_id"],
    }


def classify_cache_numerics(
    contract: dict[str, Any],
    alias: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    control = contract["evaluation_contract"]["bf16_control"]
    rows = [
        {
            "logit_max_abs_difference": step["logit_max_abs_difference"],
            "logit_mismatch_count": step["logit_mismatch_count"],
            "prompt_id": result["prompt_id"],
            "step_index": step["step_index"],
        }
        for result in results
        for step in result["cache_equivalence"]
    ]
    require(rows, f"cache diagnostic is empty for {alias}")
    worst = max(
        rows,
        key=lambda row: (
            row["logit_max_abs_difference"],
            row["logit_mismatch_count"],
            row["prompt_id"],
            row["step_index"],
        ),
    )
    bound = float(control["bf16_logit_max_abs_difference_bound"])
    above_bound_count = sum(row["logit_max_abs_difference"] > bound for row in rows)
    if above_bound_count == 0:
        classification = "BF16_NUMERICAL_DIAGNOSTIC_WITHIN_BOUND"
    else:
        require(
            alias in control["fp32_exonerated_models"],
            f"BF16 cache diagnostic exceeds bound without FP32 evidence for {alias}",
        )
        record = contract["frozen_references"]["fp32_cache_diagnostic"]
        evidence = load_json(ROOT / record["path"])
        require(evidence["status"] == "PASS", "FP32 cache diagnostic did not pass")
        require(
            evidence["classification"] == control["fp32_required_classification"],
            "FP32 cache diagnostic classification differs",
        )
        source_worst = evidence["source_bf16_worst_step"]
        require(
            worst == source_worst,
            f"BF16 worst cache diagnostic differs from frozen FP32 evidence for {alias}",
        )
        require(
            all(
                step["selected_token_exact"]
                for result in results
                for step in result["cache_equivalence"]
            ),
            f"BF16 semantic equivalence failed while applying FP32 evidence for {alias}",
        )
        classification = "BF16_NUMERICAL_DIAGNOSTIC_BOUND_EXCEEDED_FP32_EXONERATED"
    require(
        classification in control["allowed_numerical_diagnostic_classifications"],
        f"BF16 numerical diagnostic classification is not allowed for {alias}",
    )
    return {
        "above_bound_step_count": above_bound_count,
        "bound": bound,
        "classification": classification,
        "worst_step": worst,
    }


def run_replay(
    contract: dict[str, Any],
    contract_sha256: str,
    spec: dict[str, Any],
    tokenizer: Any,
    replay_index: int,
    *,
    prompts: list[dict[str, Any]] | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    seeds = set_determinism()
    model = load_model(spec)
    evaluation = contract["evaluation_contract"]
    generation = contract["shared_w4a8_contract"]["generation"]
    selected_prompts = prompts or evaluation["canonical_generation_prompts"]
    results = [
        decode_prompt(
            model,
            tokenizer,
            prompt,
            generation,
            max_new_tokens=max_new_tokens,
        )
        for prompt in selected_prompts
    ]
    identities = [result["identity"] for result in results]
    semantic_equivalence_passed = all(
        step["selected_token_exact"]
        for result in results
        for step in result["cache_equivalence"]
    )
    require(semantic_equivalence_passed, f"BF16 semantic equivalence failed for {spec['alias']}")
    numerical_diagnostic = classify_cache_numerics(contract, spec["alias"], results)
    replay = {
        "cache_equivalence_passed": semantic_equivalence_passed,
        "contract_sha256": contract_sha256,
        "created_at_utc": utc_now(),
        "identity_fields": evaluation["bf16_control"]["identity_fields"],
        "identity_sha256": canonical_sha256(identities),
        "model": {
            "alias": spec["alias"],
            "model_id": spec["model_id"],
            "model_identity_sha256": canonical_sha256(spec["contract_identity"]),
        },
        "prompts": results,
        "replay_index": replay_index,
        "schema_version": 1,
        "seeds": seeds,
        "semantic_equivalence_passed": semantic_equivalence_passed,
        "numerical_diagnostic": numerical_diagnostic,
        "status": SEMANTIC_PASS_STATUS,
    }
    del model
    gc.collect()
    return replay


def source_hashes(contract_sha256: str) -> dict[str, str]:
    return {
        "contract_sha256": contract_sha256,
        "quality_config_sha256": sha256_file(QUALITY_CONFIG_PATH),
        "quality_prompt_manifest_sha256": sha256_file(
            ROOT / "benchmark/quality/PROMPT_MANIFEST.json"
        ),
        "runner_sha256": sha256_file(Path(__file__)),
    }


def execute_bf16_control(contract: dict[str, Any], contract_sha256: str, alias: str) -> Path:
    control = contract["evaluation_contract"]["bf16_control"]
    require(control["required_before_ptq"], "contract does not require pre-PTQ BF16 control")
    require(control["cache_equivalence_required"], "contract cache-equivalence requirement differs")
    require(
        control["cache_equivalence_fields"]
        == ["generated_token_ids", "per_step_selected_token", "termination_reason"],
        "contract cache-equivalence fields differ",
    )
    require(
        control["cache_logit_comparison"]
        == "enforce_0p25_bound_as_a_separate_numerical_diagnostic_with_only_the_frozen_FP32_exoneration_while_semantic_equivalence_requires_exact_selected_tokens_and_termination",
        "contract cache-logit diagnostic policy differs",
    )
    require(
        control["semantic_equivalence_rule"]
        == "every_cache_step_selected_token_must_equal_full_prefix_and_duplicate_replays_must_match_generated_tokens_selected_tokens_and_termination",
        "contract BF16 semantic-equivalence rule differs",
    )
    require(control["exact_replay_count"] == 2, "runner implements exactly two BF16 replays")
    spec = model_spec(contract, alias)
    verify_local_model_inputs(contract, spec)
    target = ROOT / spec["artifact_root"] / contract["artifact_policy"]["bf16_control_namespace"]
    require(not target.exists(), f"immutable BF16 control already exists: {target.relative_to(ROOT)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bf16-control-", dir=target.parent))
    try:
        tokenizer = load_tokenizer(contract, spec)
        replays = [
            run_replay(contract, contract_sha256, spec, tokenizer, replay_index)
            for replay_index in range(1, control["exact_replay_count"] + 1)
        ]
        identity_hashes = [replay["identity_sha256"] for replay in replays]
        require(len(set(identity_hashes)) == 1, f"duplicate BF16 replay differs for {alias}")
        numerical_classifications = [
            replay["numerical_diagnostic"]["classification"] for replay in replays
        ]
        require(
            len(set(numerical_classifications)) == 1,
            f"duplicate BF16 numerical classification differs for {alias}",
        )
        replay_records = []
        for replay in replays:
            path = temporary / f"replay-{replay['replay_index']:04d}.json"
            write_json(path, replay)
            replay_records.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
            )
        summary = {
            "cache_equivalence_passed": all(
                replay["cache_equivalence_passed"] for replay in replays
            ),
            "contract_control": control,
            "created_at_utc": utc_now(),
            "exact_replay_count": len(replays),
            "identity_sha256": identity_hashes[0],
            "model": replays[0]["model"],
            "numerical_diagnostic_classification": numerical_classifications[0],
            "replays": replay_records,
            "schema_version": 1,
            "source_hashes": source_hashes(contract_sha256),
            "status": SEMANTIC_PASS_STATUS,
        }
        write_json(temporary / "summary.json", summary)
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def verify_bf16_control(contract: dict[str, Any], contract_sha256: str, alias: str) -> dict[str, Any]:
    spec = model_spec(contract, alias)
    target = ROOT / spec["artifact_root"] / contract["artifact_policy"]["bf16_control_namespace"]
    summary_path = target / "summary.json"
    require(summary_path.is_file(), f"BF16 control is missing for {alias}")
    summary = load_json(summary_path)
    require(summary["status"] == SEMANTIC_PASS_STATUS, f"BF16 control did not pass for {alias}")
    require(summary["source_hashes"] == source_hashes(contract_sha256), f"BF16 source hashes differ for {alias}")
    require(
        summary["contract_control"] == contract["evaluation_contract"]["bf16_control"],
        f"BF16 contract binding differs for {alias}",
    )
    require(summary["model"]["alias"] == alias, f"BF16 model alias differs for {alias}")
    require(
        summary["model"]["model_id"] == spec["model_id"],
        f"BF16 model identity differs for {alias}",
    )
    identity_hashes = []
    for record in summary["replays"]:
        path = target / record["path"]
        require(path.is_file(), f"BF16 replay is missing for {alias}: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"BF16 replay hash differs for {alias}")
        replay = load_json(path)
        require(replay["status"] == SEMANTIC_PASS_STATUS, f"BF16 replay did not pass for {alias}")
        require(replay["cache_equivalence_passed"], f"cache equivalence failed for {alias}")
        require(replay["semantic_equivalence_passed"], f"semantic equivalence failed for {alias}")
        observed_numerics = classify_cache_numerics(contract, alias, replay["prompts"])
        require(
            replay["numerical_diagnostic"] == observed_numerics,
            f"BF16 numerical diagnostic differs for {alias}",
        )
        require(
            summary["numerical_diagnostic_classification"]
            == observed_numerics["classification"],
            f"BF16 summary numerical classification differs for {alias}",
        )
        identity_hashes.append(replay["identity_sha256"])
    require(
        len(identity_hashes) == summary["exact_replay_count"] == 2,
        f"BF16 replay count differs for {alias}",
    )
    require(
        len(set(identity_hashes)) == 1 and identity_hashes[0] == summary["identity_sha256"],
        f"duplicate BF16 identities differ for {alias}",
    )
    return {
        "alias": alias,
        "identity_sha256": summary["identity_sha256"],
        "numerical_diagnostic_classification": summary[
            "numerical_diagnostic_classification"
        ],
        "path": target.relative_to(ROOT).as_posix(),
        "status": SEMANTIC_PASS_STATUS,
    }


def diagnose_cache(contract: dict[str, Any], contract_sha256: str, alias: str) -> dict[str, Any]:
    spec = model_spec(contract, alias)
    verify_local_model_inputs(contract, spec)
    tokenizer = load_tokenizer(contract, spec)
    prompt = contract["evaluation_contract"]["canonical_generation_prompts"][0]
    replay = run_replay(
        contract,
        contract_sha256,
        spec,
        tokenizer,
        0,
        prompts=[prompt],
        max_new_tokens=2,
    )
    return {
        "alias": alias,
        "cache_equivalence_passed": replay["cache_equivalence_passed"],
        "identity_sha256": replay["identity_sha256"],
        "status": replay["status"],
    }


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("accelerate", "datasets", "lm-eval", "peft", "safetensors", "torch", "transformers"):
        packages[name] = importlib.metadata.version(name)
    return {
        "packages": packages,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_threads": torch.get_num_threads(),
    }


def selected_aliases(value: str) -> list[str]:
    return list(MODEL_ALIASES) if value == "all" else [value]


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("diagnose-cache", "bf16-control", "verify-controls", "ptq-preflight"),
        required=True,
    )
    parser.add_argument("--model", choices=(*MODEL_ALIASES, "all"), required=True)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    args = parser.parse_args()
    if args.action == "diagnose-cache" and args.model == "all":
        raise SystemExit("diagnose-cache requires one model")
    require(args.threads >= 1, "thread count must be positive")
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    contract, contract_sha256 = load_contract()
    aliases = selected_aliases(args.model)

    if args.action == "diagnose-cache":
        result: Any = diagnose_cache(contract, contract_sha256, aliases[0])
    elif args.action == "bf16-control":
        result = [
            {
                "alias": alias,
                "path": execute_bf16_control(contract, contract_sha256, alias)
                .relative_to(ROOT)
                .as_posix(),
                "status": "PASS",
            }
            for alias in aliases
        ]
    else:
        controls = [verify_bf16_control(contract, contract_sha256, alias) for alias in aliases]
        result = {
            "action": args.action,
            "controls": controls,
            "environment": runtime_environment(),
            "ptq_authorized": False,
            "status": "PASS" if args.action == "verify-controls" else "BLOCKED_PENDING_FRESH_L2_ACCEPTANCE",
        }
        if args.action == "ptq-preflight":
            require(args.model == "all", "PTQ preflight requires both models")
            require(
                contract["scope_guards"]["ptq_candidate_execution_authorized"] is False,
                "contract unexpectedly authorizes PTQ candidate execution",
            )
            result["blocker"] = "fresh_l2_acceptance_of_corrected_v2_bf16_controls_required"
            result["candidate_order"] = [
                candidate["candidate_id"] for candidate in contract["candidate_matrix"]
            ]
            result["maximum_genuine_candidate_model_attempts"] = contract[
                "stopping_rules"
            ]["maximum_genuine_candidate_model_attempts"]

    print(
        "ACE2_W4A8_CONTRACT_RUNNER "
        + json.dumps(
            {
                "action": args.action,
                "contract_sha256": contract_sha256,
                "model": args.model,
                "result": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
