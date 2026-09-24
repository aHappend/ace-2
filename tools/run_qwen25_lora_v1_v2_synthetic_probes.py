#!/usr/bin/env python3
"""Run hash-frozen, aggregate-only synthetic probes on immutable V1/V2 weights.

This diagnostic never opens official dev/holdout records and never emits raw
generated text or per-case results. It is not a checkpoint selector and cannot
create a product candidate.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.util
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_fixtures.json"
DISJOINT = ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_disjoint_attestation.json"
V1_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V1_CONTRACT.json"
V2_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"
V1_RUNTIME = ROOT / "pilot/qwen25_05b_bf16_lora_product_v1/common.py"
V2_RUNTIME = ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/runtime.py"
V1_RUN = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v1/attempt-0001"
V2_RUN = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v2/cpu-attempt-0001"
VARIANTS = (
    ("source", None, None),
    ("v1_epoch_1", V1_RUN / "checkpoints/checkpoint-18", 1),
    ("v1_epoch_2", V1_RUN / "checkpoints/checkpoint-36", 2),
    ("v1_epoch_3", V1_RUN / "checkpoints/checkpoint-54", 3),
    ("v2_epoch_4", V2_RUN / "checkpoints/checkpoint-108", 4),
    ("v2_epoch_5", V2_RUN / "checkpoints/checkpoint-135", 5),
    ("v2_epoch_6", V2_RUN / "checkpoints/checkpoint-162", 6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {relative(path)}")
    return value


def verify_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    require(sidecar.read_text(encoding="ascii").split() == [digest, path.name], f"sidecar differs: {relative(path)}")
    return digest


def load_runtime() -> Any:
    spec = importlib.util.spec_from_file_location("ace2_v2_probe_runtime", V2_RUNTIME)
    require(spec is not None and spec.loader is not None, "could not load frozen runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_snapshot(contract: dict[str, Any]) -> Path:
    audit_path = ROOT / contract["source_model"]["local_audit_path"]
    require(sha256_file(audit_path) == contract["source_model"]["local_audit_sha256"], "source audit differs")
    audit = load_json(audit_path)
    snapshot = Path(audit["snapshot"]["path"])
    require(snapshot.is_dir(), "source snapshot is unavailable")
    require(snapshot.name == contract["source_model"]["revision"], "source revision differs")
    for name, expected in contract["source_model"]["files"].items():
        require(sha256_file(snapshot / name) == expected, f"source file differs: {name}")
    return snapshot


def apostrophe_folded(value: str) -> str:
    return value.translate({0x2018: "'", 0x2019: "'", 0x02BC: "'"})


def static_scorer_probe(runtime: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    category_passes: collections.Counter[str] = collections.Counter()
    target_passes = 0
    target_compatibility_passes = 0
    legacy_count = 0
    legacy_strict_passes = 0
    legacy_compatibility_passes = 0
    legacy_required_any_only_failures = 0
    for case in cases:
        target = runtime.hard_check(case, case["target_response"])
        compatible = runtime.hard_check(case, apostrophe_folded(case["target_response"]))
        target_passes += int(target["passed"])
        target_compatibility_passes += int(compatible["passed"])
        category_passes[case["category"]] += int(target["passed"])
        if "legacy_v2_style_response" in case:
            legacy_count += 1
            legacy = runtime.hard_check(case, case["legacy_v2_style_response"])
            legacy_compatible = runtime.hard_check(
                case, apostrophe_folded(case["legacy_v2_style_response"])
            )
            legacy_strict_passes += int(legacy["passed"])
            legacy_compatibility_passes += int(legacy_compatible["passed"])
            failed = [name for name, passed in legacy["checks"].items() if not passed]
            legacy_required_any_only_failures += int(failed == ["required_any"])
    require(target_passes == len(cases), "canonical synthetic targets do not all pass the frozen scorer")
    require(legacy_count == 2, "legacy V2 punctuation fixture count differs")
    require(legacy_strict_passes == 0, "legacy V2 punctuation unexpectedly passes the frozen scorer")
    require(legacy_compatibility_passes == legacy_count, "apostrophe folding does not repair legacy V2 punctuation")
    return {
        "canonical_target_category_passes": dict(sorted(category_passes.items())),
        "canonical_target_count": len(cases),
        "canonical_target_frozen_passes": target_passes,
        "canonical_target_punctuation_equivalent_passes": target_compatibility_passes,
        "legacy_v2_curly_apostrophe_count": legacy_count,
        "legacy_v2_curly_apostrophe_frozen_passes": legacy_strict_passes,
        "legacy_v2_curly_apostrophe_punctuation_equivalent_passes": legacy_compatibility_passes,
        "legacy_v2_failures_only_at_required_any": legacy_required_any_only_failures,
    }


def ordered_response_commitment(rows: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256(b"ace2/synthetic-response-commitment/v1\x00")
    for identifier, response in rows:
        encoded_id = identifier.encode("utf-8")
        encoded_response = response.encode("utf-8")
        digest.update(len(encoded_id).to_bytes(4, "big"))
        digest.update(encoded_id)
        digest.update(len(encoded_response).to_bytes(8, "big"))
        digest.update(encoded_response)
    return digest.hexdigest()


def aggregate_variant(
    runtime: Any,
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    strict_by_category: collections.Counter[str] = collections.Counter()
    compatible_by_category: collections.Counter[str] = collections.Counter()
    totals: collections.Counter[str] = collections.Counter()
    generated_counts: list[int] = []
    response_commitment_rows: list[tuple[str, str]] = []
    strict_passes = 0
    compatibility_passes = 0
    strict_fail_compatibility_pass = 0
    eos_count = 0
    limit_count = 0
    ascii_contraction_responses = 0
    curly_apostrophe_responses = 0
    explicit_cannot_responses = 0
    safe_only_required_any_failures = 0
    safe_redirect_check_passes = 0
    safe_sentence_check_passes = 0
    for case in cases:
        generated = runtime.generate_response(
            model,
            tokenizer,
            case["input_messages"],
            max_new_tokens=max_new_tokens,
        )
        response = generated["decoded_text"]
        strict = runtime.hard_check(case, response)
        compatible = runtime.hard_check(case, apostrophe_folded(response))
        category = case["category"]
        totals[category] += 1
        strict_by_category[category] += int(strict["passed"])
        compatible_by_category[category] += int(compatible["passed"])
        strict_passes += int(strict["passed"])
        compatibility_passes += int(compatible["passed"])
        strict_fail_compatibility_pass += int(not strict["passed"] and compatible["passed"])
        generated_counts.append(int(generated["generated_token_count"]))
        eos_count += int(generated["terminated_by_eos"])
        limit_count += int(generated["reached_limit_without_eos"])
        folded = response.casefold()
        ascii_contraction_responses += int(bool(re.search(r"\b(?:can|won)'t\b", folded)))
        curly_apostrophe_responses += int("\u2019" in response or "\u2018" in response or "\u02bc" in response)
        explicit_cannot_responses += int(bool(re.search(r"\bcannot\b", folded)))
        response_commitment_rows.append((case["id"], response))
        if category == "safe_refusal":
            failed = [name for name, passed in strict["checks"].items() if not passed]
            safe_only_required_any_failures += int(failed == ["required_any"])
            safe_redirect_check_passes += int(
                all(
                    passed
                    for name, passed in strict["checks"].items()
                    if name.startswith("required_substring_")
                )
            )
            safe_sentence_check_passes += int(strict["checks"].get("max_sentences", False))
    return {
        "category_frozen_passes": dict(sorted(strict_by_category.items())),
        "category_punctuation_equivalent_passes": dict(sorted(compatible_by_category.items())),
        "category_totals": dict(sorted(totals.items())),
        "frozen_hard_pass_count": strict_passes,
        "generated_token_count": {
            "maximum": max(generated_counts),
            "mean": statistics.fmean(generated_counts),
            "minimum": min(generated_counts),
            "sum": sum(generated_counts),
        },
        "generation_style": {
            "ascii_contraction_response_count": ascii_contraction_responses,
            "curly_apostrophe_response_count": curly_apostrophe_responses,
            "explicit_cannot_response_count": explicit_cannot_responses,
        },
        "punctuation_equivalent_hard_pass_count": compatibility_passes,
        "response_count": len(cases),
        "response_ordered_commitment_sha256": ordered_response_commitment(response_commitment_rows),
        "safe_refusal_diagnostics": {
            "only_required_any_failed_count": safe_only_required_any_failures,
            "redirect_checks_passed_count": safe_redirect_check_passes,
            "sentence_limit_check_passed_count": safe_sentence_check_passes,
        },
        "strict_fail_punctuation_equivalent_pass_count": strict_fail_compatibility_pass,
        "terminated_by_eos_count": eos_count,
        "token_limit_without_eos_count": limit_count,
    }


def configure_determinism() -> None:
    import torch
    from transformers import set_seed

    os.environ["ACE2_LORA_DEVICE"] = "cpu"
    os.environ["PYTHONHASHSEED"] = "260827"
    set_seed(260827, deterministic=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def release() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_report() -> dict[str, Any]:
    fixture_hash = verify_sidecar(FIXTURE)
    disjoint_hash = verify_sidecar(DISJOINT)
    fixture = load_json(FIXTURE)
    disjoint = load_json(DISJOINT)
    require(disjoint["results"]["status"] == "PASS", "synthetic fixtures are not disjoint")
    require(disjoint["fixture"]["sha256"] == fixture_hash, "disjointness fixture binding differs")
    cases = fixture["cases"]
    require(len(cases) == fixture["policy"]["expected_case_count"], "fixture count differs")

    v1_contract = load_json(V1_CONTRACT)
    v2_contract = load_json(V2_CONTRACT)
    require(v1_contract["source_model"]["revision"] == v2_contract["source_model"]["revision"], "source revisions differ")
    snapshot = source_snapshot(v2_contract)
    runtime = load_runtime()
    configure_determinism()
    tokenizer = runtime.load_tokenizer(snapshot, {"contract": v2_contract})
    max_new_tokens = 128
    static = static_scorer_probe(runtime, cases)

    variants: dict[str, Any] = {}
    for identifier, checkpoint, epoch in VARIANTS:
        base = runtime.load_causal_model(snapshot)
        if checkpoint is None:
            model = base
            identity = {
                "checkpoint_epoch": None,
                "kind": "source",
                "model_sha256": v2_contract["source_model"]["files"]["model.safetensors"],
            }
        else:
            from peft import PeftModel

            require(checkpoint.is_dir(), f"checkpoint missing: {relative(checkpoint)}")
            model = PeftModel.from_pretrained(
                base,
                checkpoint,
                is_trainable=False,
                autocast_adapter_dtype=False,
                low_cpu_mem_usage=True,
            )
            identity = {
                "adapter_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
                "checkpoint_epoch": epoch,
                "checkpoint_path": relative(checkpoint),
                "kind": identifier.split("_", 1)[0],
            }
        model.eval()
        variants[identifier] = {
            "aggregate": aggregate_variant(runtime, model, tokenizer, cases, max_new_tokens),
            "identity": identity,
        }
        del model, base
        release()

    return {
        "evidence_boundary": {
            "checkpoint_selection_or_tuning": False,
            "official_dev_or_holdout_opened": False,
            "official_evaluation_replayed": False,
            "raw_generated_responses_emitted": False,
            "synthetic_inference_only": True,
            "training_executed": False,
        },
        "fixture_binding": {
            "case_count": len(cases),
            "disjoint_attestation_path": relative(DISJOINT),
            "disjoint_attestation_sha256": disjoint_hash,
            "fixture_path": relative(FIXTURE),
            "fixture_sha256": fixture_hash,
        },
        "generation_contract": {
            "add_generation_prompt": True,
            "decode_clean_up_tokenization_spaces": False,
            "decode_skip_special_tokens": True,
            "do_sample": False,
            "eos_token_ids": list(runtime.TERMINATION_TOKEN_IDS),
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "pad_token_id": 151643,
        },
        "probe_id": "qwen25-lora-v1-v2-fresh-synthetic-probes-v1",
        "schema_version": 1,
        "static_scorer_conformance": static,
        "tool_bindings": {
            relative(V1_RUNTIME): sha256_file(V1_RUNTIME),
            relative(V2_RUNTIME): sha256_file(V2_RUNTIME),
            relative(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
        "variants": variants,
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_report(path: Path) -> None:
    payload = canonical_bytes(build_report())
    digest = sha256_bytes(payload)
    atomic_write(path, payload)
    atomic_write(sidecar_path(path), f"{digest}  {path.name}\n".encode("ascii"))
    print(f"WROTE {relative(path)} sha256={digest}")


def verify_report(path: Path) -> None:
    expected = canonical_bytes(build_report())
    observed = path.read_bytes()
    require(observed == expected, "synthetic probe results differ from deterministic replay")
    digest = sha256_bytes(observed)
    require(sidecar_path(path).read_text(encoding="ascii").split() == [digest, path.name], "probe result sidecar differs")
    print(f"QWEN25_LORA_SYNTHETIC_PROBES_VERIFIED sha256={digest}")


def main() -> None:
    args = parse_args()
    if args.output is not None:
        write_report(args.output.resolve())
    else:
        verify_report(args.verify.resolve())


if __name__ == "__main__":
    main()
