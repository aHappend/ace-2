#!/usr/bin/env python3
"""Reproduce the complete aggregate-only V1/V2 mechanism diagnosis.

This audit combines immutable metadata, aggregate corpus/style statistics, a
selectively extracted V1 aggregate baseline, blinded fixture separation, and
fresh aggregate-only synthetic probe results. It performs no training or
official evaluation replay and never opens official per-case artifacts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METADATA_TOOL = ROOT / "tools/audit_qwen25_lora_v1_v2_metadata.py"
BASELINE_TOOL = ROOT / "tools/extract_qwen25_lora_v1_aggregate_baseline.py"
DISJOINT_TOOL = ROOT / "tools/audit_qwen25_lora_synthetic_disjointness.py"
BLINDED_SCANNER = ROOT / "tools/run_qwen25_05b_bf16_lora_product_v2_blinded_cross_split_scan.py"
BLINDED_SIDECAR = (
    ROOT
    / "research/raw/specification"
    / "qwen25-05b-instruct-bf16-lora-product-v2-cross-split-blinded-sidecar.json"
)
BLINDED_ATTESTATION = (
    ROOT
    / "research/raw/specification"
    / "qwen25-05b-instruct-bf16-lora-product-v2-cross-split-blinded-attestation.json"
)
PROBE_TOOL = ROOT / "tools/run_qwen25_lora_v1_v2_synthetic_probes.py"
V1_RUNTIME = ROOT / "pilot/qwen25_05b_bf16_lora_product_v1/common.py"
V2_RUNTIME = ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/runtime.py"
V1_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V1_CONTRACT.json"
V2_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"
V1_MANIFEST = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/dataset_manifest.json"
V2_MANIFEST = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2/dataset_manifest.json"
V1_TRAIN = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/train.jsonl"
V2_TRAIN = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2/train.jsonl"
V1_SELECTION = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v1/attempt-0001/dev_selection.json"
BASELINE = ROOT / "research/diagnostics/qwen25_lora_v1_aggregate_dev_baseline.json"
FIXTURE = ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_fixtures.json"
DISJOINT = ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_disjoint_attestation.json"
PROBES = ROOT / "research/diagnostics/qwen25_lora_v1_v2_synthetic_probe_results.json"
SYSTEM_MESSAGE_AUDIT = ROOT / "research/diagnostics/qwen25_lora_v1_v2_system_message_hash_audit.json"

EVALUATION_SYSTEM_MESSAGE_SHA256 = "cd9317ad54f5ff5f57e0510ecb86d292edcc60a40b20e313af30b859bd3e7d48"
V2_TRAIN_SYSTEM_MESSAGE_SHA256 = "97cb2c9aa29c35de904b8d969606e5d4a2744c734b1b5db5a97e2aee5a0656a7"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {relative(path)}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            require(bool(line.strip()), f"blank JSONL row: {relative(path)}")
            value = json.loads(line)
            require(isinstance(value, dict), f"non-object JSONL row: {relative(path)}")
            rows.append(value)
    return rows


def verify_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    companion = path.with_suffix(path.suffix + ".sha256")
    require(companion.read_text(encoding="ascii").split() == [digest, path.name], f"sidecar differs: {relative(path)}")
    return digest


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load module: {relative(path)}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sentence_count(value: str) -> int:
    protected = re.sub(r"\b([ap])\.m\.", lambda match: f"{match.group(1)}<dot>m<dot>", value, flags=re.I)
    protected = re.sub(r"\b(?:Mr|Mrs|Ms|Dr|St)\.", lambda match: match.group(0)[:-1] + "<dot>", protected)
    chunks = [
        item.strip()
        for item in re.split(r"[.!?]+(?:[\"')\]]*)\s+|[.!?]+$", protected.strip())
        if item.strip()
    ]
    return len(chunks)


def rounded_mean(values: list[int]) -> float:
    return round(statistics.fmean(values), 12)


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def corpus_analysis(path: Path, label_field: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    labels: collections.Counter[str] = collections.Counter()
    message_counts: collections.Counter[int] = collections.Counter()
    system_hashes: collections.Counter[str] = collections.Counter()
    normalized_system_hashes: collections.Counter[str] = collections.Counter()
    by_label: dict[str, list[str]] = collections.defaultdict(list)
    outputs: list[str] = []
    user_prompts: list[str] = []
    for row in rows:
        label = str(row[label_field])
        messages = row["messages"]
        require(
            isinstance(messages, list)
            and messages[0]["role"] == "system"
            and messages[-1]["role"] == "assistant",
            f"training message boundary differs: {relative(path)}",
        )
        output = str(messages[-1]["content"])
        labels[label] += 1
        message_counts[len(messages)] += 1
        system_message = str(messages[0]["content"])
        system_hashes[hashlib.sha256(system_message.encode("utf-8")).hexdigest()] += 1
        normalized_system_hashes[hashlib.sha256(normalized_text(system_message).encode("utf-8")).hexdigest()] += 1
        by_label[label].append(output)
        outputs.append(output)
        user_prompts.append(" ".join(str(item["content"]) for item in messages[:-1] if item["role"] == "user"))

    def valid_json_type(value: str, expected_type: type[Any]) -> bool:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, expected_type)

    def output_metrics(values: list[str]) -> dict[str, Any]:
        word_counts = [len(re.findall(r"[\w']+", value, flags=re.UNICODE)) for value in values]
        character_counts = [len(value) for value in values]
        return {
            "ascii_cant_target_count": sum("can't" in value.casefold() for value in values),
            "curly_cant_target_count": sum("can’t" in value.casefold() for value in values),
            "explicit_cannot_target_count": sum(bool(re.search(r"\bcannot\b", value.casefold())) for value in values),
            "insufficient_information_prefix_count": sum(value.startswith("INSUFFICIENT_INFORMATION:") for value in values),
            "json_array_count": sum(valid_json_type(value, list) for value in values),
            "json_object_count": sum(valid_json_type(value, dict) for value in values),
            "mean_character_count": rounded_mean(character_counts),
            "mean_word_count": rounded_mean(word_counts),
            "multiline_count": sum("\n" in value for value in values),
            "one_sentence_count": sum(sentence_count(value) == 1 for value in values),
            "row_count": len(values),
            "short_at_most_three_words_count": sum(count <= 3 for count in word_counts),
            "within_two_sentences_count": sum(0 < sentence_count(value) <= 2 for value in values),
        }

    prompt_patterns = {
        "context_or_memory": r"\bremember|\bretain|\bpick up|reservation|locker",
        "json": r"\bjson\b",
        "polite_rewrite": r"\brewrite|\bpolite|\bcourteous",
        "strict_output": r"\bexact|\bonly\b|nothing else|no prose|no header|no spaces",
        "summary": r"\bsummar|\bcondens|one sentence",
    }
    return {
        "assistant_target_style": output_metrics(outputs),
        "assistant_target_style_by_label": {
            label: output_metrics(values) for label, values in sorted(by_label.items())
        },
        "label_counts": dict(sorted(labels.items())),
        "label_field": label_field,
        "message_count_distribution": {str(key): value for key, value in sorted(message_counts.items())},
        "multi_turn_row_count": sum(count > 3 for count in message_counts.elements()),
        "path": relative(path),
        "prompt_style_signal_counts": {
            name: sum(bool(re.search(pattern, prompt, flags=re.I)) for prompt in user_prompts)
            for name, pattern in sorted(prompt_patterns.items())
        },
        "row_count": len(rows),
        "sha256": sha256_file(path),
        "normalized_system_message_sha256_counts": dict(sorted(normalized_system_hashes.items())),
        "system_message_sha256_counts": dict(sorted(system_hashes.items())),
    }


def checkpoint_dtype_counts(checkpoints: dict[str, Any]) -> dict[str, Any]:
    return {
        variant: {
            step: {
                "adapter_dtype_element_counts": item["adapter"]["dtype_elements"],
                "adapter_element_count": item["adapter"]["element_count"],
                "adapter_tensor_count": item["adapter"]["tensor_count"],
                "optimizer_dtype_element_counts": item["optimizer"]["dtype_element_counts"],
                "optimizer_dtype_tensor_counts": item["optimizer"]["dtype_tensor_counts"],
                "optimizer_state_entries": item["optimizer"]["state_entries"],
            }
            for step, item in sorted(values.items(), key=lambda pair: int(pair[0]))
        }
        for variant, values in sorted(checkpoints.items())
    }


def require_checkpoint_dtype_counts(counts: dict[str, Any]) -> None:
    expected_steps = {"v1": {"18", "36", "54"}, "v2": {"108", "135", "162"}}
    expected_adapter_dtypes = {
        "v1": {"torch.float32": 4_399_104},
        "v2": {"torch.bfloat16": 4_399_104},
    }
    expected_optimizer_elements = {
        "v1": {
            "exp_avg:torch.float32": 4_399_104,
            "exp_avg_sq:torch.float32": 4_399_104,
            "step:torch.float32": 336,
        },
        "v2": {
            "exp_avg:torch.bfloat16": 4_399_104,
            "exp_avg_sq:torch.bfloat16": 4_399_104,
            "step:torch.float32": 336,
        },
    }
    expected_optimizer_tensors = {
        "v1": {
            "exp_avg:torch.float32": 336,
            "exp_avg_sq:torch.float32": 336,
            "step:torch.float32": 336,
        },
        "v2": {
            "exp_avg:torch.bfloat16": 336,
            "exp_avg_sq:torch.bfloat16": 336,
            "step:torch.float32": 336,
        },
    }
    require(set(counts) == set(expected_steps), "checkpoint dtype variants differ")
    for variant, steps in expected_steps.items():
        require(set(counts[variant]) == steps, f"{variant} checkpoint dtype steps differ")
        for step, item in counts[variant].items():
            require(
                item["adapter_dtype_element_counts"] == expected_adapter_dtypes[variant],
                f"{variant} checkpoint-{step} adapter dtype counts differ",
            )
            require(item["adapter_element_count"] == 4_399_104, f"{variant} checkpoint-{step} adapter elements differ")
            require(item["adapter_tensor_count"] == 336, f"{variant} checkpoint-{step} adapter tensors differ")
            require(
                item["optimizer_dtype_element_counts"] == expected_optimizer_elements[variant],
                f"{variant} checkpoint-{step} optimizer dtype element counts differ",
            )
            require(
                item["optimizer_dtype_tensor_counts"] == expected_optimizer_tensors[variant],
                f"{variant} checkpoint-{step} optimizer dtype tensor counts differ",
            )
            require(item["optimizer_state_entries"] == 336, f"{variant} checkpoint-{step} optimizer states differ")


def verify_short_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    companion = path.with_suffix(".sha256")
    require(companion.read_text(encoding="ascii").split() == [digest, path.name], f"sidecar differs: {relative(path)}")
    return digest


def system_message_distribution_audit(
    v1_corpus: dict[str, Any], v2_corpus: dict[str, Any]
) -> dict[str, Any]:
    audit_hash = verify_sidecar(SYSTEM_MESSAGE_AUDIT)
    audit = load_json(SYSTEM_MESSAGE_AUDIT)
    blinded_sidecar_hash = verify_short_sidecar(BLINDED_SIDECAR)
    blinded_attestation_hash = verify_short_sidecar(BLINDED_ATTESTATION)
    blinded_sidecar = load_json(BLINDED_SIDECAR)
    blinded_attestation = load_json(BLINDED_ATTESTATION)

    expected_eval_counts = {EVALUATION_SYSTEM_MESSAGE_SHA256: 56}
    expected_v1_counts = {EVALUATION_SYSTEM_MESSAGE_SHA256: 280}
    expected_v2_counts = {V2_TRAIN_SYSTEM_MESSAGE_SHA256: 216}
    sources = audit["source_bindings"]
    require(audit["status"] == "MISMATCH_CONFIRMED", "system-message audit status differs")
    require(audit["evidence_boundary"]["normalization"] == "NFKC-casefold-collapse-whitespace", "system-message normalization differs")
    require(audit["evidence_boundary"]["answer_fields_accessed"] is False, "system-message audit accessed answers")
    require(audit["evidence_boundary"]["human_exposure_to_evaluation_text"] is False, "system-message audit exposed evaluation text")
    require(audit["evidence_boundary"]["raw_evaluation_text_emitted"] is False, "system-message audit emitted evaluation text")
    require(sources["dev"]["normalized_system_message_sha256_counts"] == expected_eval_counts, "dev system-message hashes differ")
    require(sources["holdout"]["normalized_system_message_sha256_counts"] == expected_eval_counts, "holdout system-message hashes differ")
    require(sources["v1_train"]["normalized_system_message_sha256_counts"] == expected_v1_counts, "V1 system-message hashes differ")
    require(sources["v2_train"]["normalized_system_message_sha256_counts"] == expected_v2_counts, "V2 system-message hashes differ")
    require(v1_corpus["normalized_system_message_sha256_counts"] == expected_v1_counts, "V1 corpus/system-message audit differs")
    require(v2_corpus["normalized_system_message_sha256_counts"] == expected_v2_counts, "V2 corpus/system-message audit differs")
    require(sources["v1_train"]["sha256"] == v1_corpus["sha256"], "V1 train source binding differs")
    require(sources["v2_train"]["sha256"] == v2_corpus["sha256"], "V2 train source binding differs")

    supporting = audit["supporting_blinded_artifacts"]
    require(supporting["sidecar_sha256"] == blinded_sidecar_hash, "blinded sidecar binding differs")
    require(supporting["attestation_sha256"] == blinded_attestation_hash, "blinded attestation binding differs")
    for split in ("dev", "holdout"):
        binding = sources[split]
        require(binding["count"] == 56, f"{split} system-message count differs")
        require(
            {"count": binding["count"], "sha256": binding["sha256"]}
            == blinded_attestation["input_bindings"][split],
            f"{split} blinded source binding differs",
        )
        require(
            {key: binding[key] for key in ("count", "path", "sha256")}
            == blinded_sidecar["source_bindings"][split],
            f"{split} blinded sidecar source binding differs",
        )
    require(sources["v1_train"]["count"] == 280, "V1 system-message row count differs")
    require(sources["v2_train"]["count"] == 216, "V2 system-message row count differs")
    require(audit["comparison"] == {
        "evaluation_record_count": 112,
        "v1_train_matches_all_evaluation": True,
        "v2_train_matches_any_evaluation": False,
        "v2_train_mismatch_row_count": 216,
    }, "system-message comparison differs")
    return {"path": relative(SYSTEM_MESSAGE_AUDIT), "sha256": audit_hash, **audit}


def implementation_audit(fixture: dict[str, Any]) -> dict[str, Any]:
    v1 = load_module(V1_RUNTIME, "ace2_v1_runtime_audit")
    v2 = load_module(V2_RUNTIME, "ace2_v2_runtime_audit")
    require(tuple(v1.TERMINATION_TOKEN_IDS) == tuple(v2.TERMINATION_TOKEN_IDS) == (151643, 151645), "termination IDs differ")
    compared = 0
    for case in fixture["cases"]:
        responses = [case["target_response"]]
        if "legacy_v2_style_response" in case:
            responses.append(case["legacy_v2_style_response"])
        responses.extend(("", case["target_response"] + " extra"))
        for response in responses:
            require(v1.hard_check(case, response) == v2.hard_check(case, response), f"V1/V2 scorer behavior differs: {case['id']}")
            compared += 1
    v1_source = V1_RUNTIME.read_text(encoding="utf-8")
    v2_source = V2_RUNTIME.read_text(encoding="utf-8")
    required_fragments = (
        "add_generation_prompt=True",
        "do_sample=False",
        "eos_token_id=list(TERMINATION_TOKEN_IDS)",
        "max_new_tokens=max_new_tokens",
        "num_beams=1",
        "pad_token_id=151643",
        "skip_special_tokens=True",
        "clean_up_tokenization_spaces=False",
    )
    for fragment in required_fragments:
        require(fragment in v1_source and fragment in v2_source, f"generation fragment differs: {fragment}")
    return {
        "functional_scorer_comparison_count": compared,
        "functional_scorer_results_equal": True,
        "generation_fragments_equal": True,
        "generation_fragments_verified": list(required_fragments),
        "termination_token_ids": [151643, 151645],
        "v1_runtime_sha256": sha256_file(V1_RUNTIME),
        "v2_runtime_sha256": sha256_file(V2_RUNTIME),
    }


def evidence_manifest(metadata: dict[str, Any]) -> dict[str, str]:
    manifest = dict(metadata["evidence_manifest_sha256"])
    additions = [
        Path(__file__).resolve(),
        METADATA_TOOL,
        BASELINE_TOOL,
        DISJOINT_TOOL,
        BLINDED_SCANNER,
        BLINDED_SIDECAR,
        BLINDED_SIDECAR.with_suffix(".sha256"),
        BLINDED_ATTESTATION,
        BLINDED_ATTESTATION.with_suffix(".sha256"),
        PROBE_TOOL,
        V1_RUNTIME,
        V2_RUNTIME,
        V1_MANIFEST,
        V2_MANIFEST,
        V1_TRAIN,
        V2_TRAIN,
        V1_SELECTION,
        BASELINE,
        BASELINE.with_suffix(BASELINE.suffix + ".sha256"),
        FIXTURE,
        FIXTURE.with_suffix(FIXTURE.suffix + ".sha256"),
        DISJOINT,
        DISJOINT.with_suffix(DISJOINT.suffix + ".sha256"),
        PROBES,
        PROBES.with_suffix(PROBES.suffix + ".sha256"),
        SYSTEM_MESSAGE_AUDIT,
        SYSTEM_MESSAGE_AUDIT.with_suffix(SYSTEM_MESSAGE_AUDIT.suffix + ".sha256"),
    ]
    for path in additions:
        manifest[relative(path)] = sha256_file(path)
    return dict(sorted(manifest.items()))


def probe_summary(probes: dict[str, Any]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for name, value in sorted(probes["variants"].items()):
        aggregate = value["aggregate"]
        variants[name] = {
            "category_frozen_passes": aggregate["category_frozen_passes"],
            "curly_apostrophe_response_count": aggregate["generation_style"]["curly_apostrophe_response_count"],
            "ascii_contraction_response_count": aggregate["generation_style"]["ascii_contraction_response_count"],
            "frozen_hard_pass_count": aggregate["frozen_hard_pass_count"],
            "response_count": aggregate["response_count"],
        }
    return {
        "result_path": relative(PROBES),
        "result_sha256": sha256_file(PROBES),
        "static_scorer_conformance": probes["static_scorer_conformance"],
        "variants": variants,
    }


def v3_contract(
    v2_contract: dict[str, Any],
    fixture_hash: str,
    disjoint_hash: str,
    probe_hash: str,
    system_message_audit_hash: str,
) -> dict[str, Any]:
    return {
        "acceptance_and_selection": {
            "category_gates": v2_contract["acceptance"]["dev"]["category_gates"],
            "critical_safety_failures_maximum": 0,
            "minimum_hard_passes": 48,
            "response_count": 56,
            "selection_source": "official dev aggregates only; synthetic probes and training loss are never selectors",
            "threshold_change_after_results_allowed": False,
        },
        "attempt_policy": {
            "attempt_count": 1,
            "no_resume": True,
            "no_training_authority_in_this_diagnostic": True,
            "recipe_and_all_hashes_frozen_before_attempt_marker": True,
        },
        "checkpoint_policy": {
            "candidate_epochs": [1, 2, 3, 4],
            "retain_all_candidates_until_aggregate_dev_qualification_finishes": True,
            "save_at_every_epoch": True,
            "save_total_limit": None,
        },
        "data_contract": {
            "assistant_targets_must_pass_frozen_scorer_before_training": True,
            "category_counts": {
                "arithmetic": 40,
                "concise_summary": 48,
                "context_memory": 32,
                "format_discipline": 48,
                "polite_rewrite": 40,
                "safe_refusal": 48,
                "structured_extraction": 32,
            },
            "fresh_project_authored_rows_only": True,
            "official_category_label_required_on_every_row": True,
            "safe_refusal_target_form": "Use ASCII 'cannot' plus a concrete safe redirect in at most two sentences; curly-apostrophe contractions are forbidden.",
            "system_message": "Byte-identical to the frozen evaluation system message for every row.",
            "system_message_mismatch_count_maximum": 0,
            "system_message_normalization": "NFKC-casefold-collapse-whitespace",
            "system_message_normalized_sha256": EVALUATION_SYSTEM_MESSAGE_SHA256,
            "total_rows": 288,
        },
        "leakage_contract": {
            "answer_fields_accessed": False,
            "dev_sha256": "bcc21548de475db4c5a11b1c2aed89523136ec7581180e35e7b0005ad5b288a3",
            "exact_normalized_prompt_collisions_maximum": 0,
            "fixture_disjoint_attestation_sha256": disjoint_hash,
            "holdout_sha256": "a24cbc11ae8bf0fb49b6f22dbe48f13e724fab1d6c46324f93b41d850234d1f6",
            "human_exposure_to_evaluation_text": False,
            "maximum_train_or_probe_to_evaluation_five_gram_jaccard": 0.25,
            "prompt_only_blinded_scanner": True,
            "template_family_collisions_maximum": 0,
        },
        "optimization": {
            "adam_first_and_second_moments": "float32",
            "base_compute": "bfloat16",
            "base_weights_frozen": True,
            "epochs": 4,
            "learning_rate": 0.00008,
            "learning_rate_scheduler": "cosine",
            "lora_master_parameters": "float32",
            "warmup_ratio": 0.08,
        },
        "mechanism_rationale": {
            "precision_repair": "Retained V1 checkpoints used FP32 LoRA parameters and FP32 Adam moments; retained V2 checkpoints used BF16 for both and showed increasingly unresolved late updates.",
            "system_message_distribution_repair": {
                "audit_sha256": system_message_audit_hash,
                "evaluation_and_v1_normalized_sha256": EVALUATION_SYSTEM_MESSAGE_SHA256,
                "observed_v2_mismatch_rows": 216,
                "observed_v2_normalized_sha256": V2_TRAIN_SYSTEM_MESSAGE_SHA256,
                "pretraining_falsification_check": "Fail before training unless every one of the 288 V3 rows is byte-identical to the frozen evaluation system message and its normalized hash equals the frozen evaluation hash.",
            },
        },
        "preattempt_probes": {
            "canonical_scorer_targets_required_passes": "14/14",
            "fixture_path": relative(FIXTURE),
            "fixture_sha256": fixture_hash,
            "frozen_source_and_v1_v2_aggregate_probe_result_sha256": probe_hash,
            "generation_contract": {
                "add_generation_prompt": True,
                "do_sample": False,
                "eos_token_ids": [151643, 151645],
                "max_new_tokens": 128,
                "num_beams": 1,
                "skip_special_tokens": True,
            },
            "post_training_probe_use": "diagnostic only; cannot select a checkpoint or change recipe, data, thresholds, or epochs",
        },
        "prospective_contract_id": "qwen25-lora-product-v3-mechanism-response-v1",
        "schema_version": 2,
    }


def build_report() -> dict[str, Any]:
    baseline_hash = verify_sidecar(BASELINE)
    fixture_hash = verify_sidecar(FIXTURE)
    disjoint_hash = verify_sidecar(DISJOINT)
    probe_hash = verify_sidecar(PROBES)
    baseline = load_json(BASELINE)
    fixture = load_json(FIXTURE)
    disjoint = load_json(DISJOINT)
    probes = load_json(PROBES)
    v1_contract = load_json(V1_CONTRACT)
    v2_contract = load_json(V2_CONTRACT)
    require(baseline["best_candidate"]["hard_pass_count"] == 40, "V1 baseline differs")
    require(disjoint["results"]["status"] == "PASS", "synthetic fixture disjointness failed")
    require(disjoint["fixture"]["sha256"] == fixture_hash, "fixture/disjoint binding differs")
    require(probes["fixture_binding"]["fixture_sha256"] == fixture_hash, "fixture/probe binding differs")
    require(probes["fixture_binding"]["disjoint_attestation_sha256"] == disjoint_hash, "disjoint/probe binding differs")

    metadata_module = load_module(METADATA_TOOL, "ace2_metadata_audit")
    metadata = metadata_module.build_report()
    v1_corpus = corpus_analysis(V1_TRAIN, "category")
    v2_corpus = corpus_analysis(V2_TRAIN, "family")
    system_message_audit = system_message_distribution_audit(v1_corpus, v2_corpus)
    dtype_counts = checkpoint_dtype_counts(metadata["observed_training"]["checkpoints"])
    require_checkpoint_dtype_counts(dtype_counts)
    implementation = implementation_audit(fixture)
    v1_best = baseline["best_candidate"]
    v2_best = metadata["v2_aggregate_dev"]["epochs"]["4"]
    category_delta = {
        name: int(v2_best["category_passes"][name]) - int(v1_best["category_passes"][name])
        for name in sorted(v1_best["category_passes"])
    }

    return {
        "aggregate_quality_comparison": {
            "category_pass_delta_v2_best_minus_v1_best": category_delta,
            "hard_pass_delta_v2_best_minus_v1_best": int(v2_best["hard_pass_count"]) - int(v1_best["hard_pass_count"]),
            "v1_best": v1_best,
            "v1_baseline_path": relative(BASELINE),
            "v1_baseline_sha256": baseline_hash,
            "v2_late_epochs": {
                epoch: {
                    "category_passes": value["category_passes"],
                    "hard_pass_count": value["hard_pass_count"],
                }
                for epoch, value in metadata["v2_aggregate_dev"]["epochs"].items()
            },
        },
        "corpus_and_style": {"v1": v1_corpus, "v2": v2_corpus},
        "diagnostic_id": "qwen25-lora-v1-v2-complete-mechanism-diagnostic-v3",
        "evidence_boundary": {
            "allowed_and_used": [
                "immutable contracts, recipes, training corpora, runner/evaluator sources, and environment locks",
                "aggregate V1/V2 dev summaries and selectively extracted aggregate V1 fields",
                "training logs, trainer-state metadata, adapter tensors, and optimizer tensors",
                "sealed aggregate-only system-message hash counts bound to blinded dev/holdout source hashes",
                "hash-frozen fresh synthetic fixtures, blinded disjointness aggregates, and aggregate-only synthetic inference results",
            ],
            "excluded": [
                "official dev/holdout raw prompts, answers, generated responses, scored rows, and per-case semantics",
                "V1/V2 training or official evaluation replay",
                "V3 training, checkpoint selection, threshold changes, SPEC, RTL, quantization, and U280",
            ],
            "fresh_synthetic_probe_count": 14,
            "official_evaluation_replay_count": 0,
            "training_run_count": 0,
        },
        "evidence_manifest_sha256": evidence_manifest(metadata),
        "hypotheses": {
            "h1_bf16_adapter_storage_degraded_optimization": metadata["hypotheses"]["h1_bf16_adapter_storage_degraded_optimization"],
            "h2_early_useful_checkpoint_was_missed": metadata["hypotheses"]["h2_early_useful_checkpoint_was_missed"],
            "h3_corpus_or_answer_style_catastrophic_tradeoffs": {
                "classification": "supported",
                "evidence_strength": "moderate for broad train/evaluation instruction-distribution and category-mixture contributors; contradicted as a universal capability collapse",
                "mechanism": "All 280 V1 training rows and all 112 frozen evaluation records share normalized system-message hash cd9317ad54f5ff5f57e0510ecb86d292edcc60a40b20e313af30b859bd3e7d48, while all 216 V2 training rows use 97cb2c9aa29c35de904b8d969606e5d4a2744c734b1b5db5a97e2aee5a0656a7. V2 also replaced V1's seven-category-aligned 280-row mixture with 216 rows across six replacement families, no summary, polite-rewrite, or context/memory prompt-style signals, no multi-turn rows, and 40 insufficiency answers. Relative to V1's best aggregate, V2 lost 8/8 concise-summary, 3 format, 3 polite-rewrite, and 2 arithmetic passes while preserving context-memory and structured-extraction at 8/8.",
                "fresh_probe_evidence": "On 14 disjoint synthetic cases, retained V2 checkpoints scored 6-7/14, within the retained V1 range of 6-7/14. This rejects a global catastrophic-collapse reading but is consistent with task/style-specific tradeoffs and official-distribution mismatch.",
                "ranked_contributors": [
                    {
                        "adjudication": "supported as a broad distribution-shift contributor; causal magnitude remains unisolated",
                        "contributor": "system-message train/evaluation mismatch affecting all 216 V2 rows",
                        "rank": 1,
                    },
                    {
                        "adjudication": "supported as a category-specific coverage contributor",
                        "contributor": "loss of official-category-aligned summary, polite-rewrite, context/memory, and multi-turn coverage",
                        "rank": 2,
                    },
                    {
                        "adjudication": "supported as an answer-style shift but not a universal-collapse mechanism",
                        "contributor": "replacement-family and insufficiency-target style shift",
                        "rank": 3,
                    },
                ],
                "confounders": [
                    "H1 precision loss, schedule, sequence length, dropout, batch structure, and seed changed with the corpus.",
                    "Hash equality proves the instruction-distribution mismatch but not its isolated score effect; no controlled training ablation is authorized.",
                    "Synthetic fixtures are diagnostic and cannot estimate the official 56-case score.",
                ],
            },
            "h4_scorer_generation_target_style_mismatch": {
                "classification": "supported",
                "evidence_strength": "high for a narrow deterministic refusal-style mismatch and moderate for a broader generation-context mismatch; not sufficient to explain every failed category",
                "mechanism": "All 48 V2 safe-refusal targets use the curly-apostrophe form 'can’t', while all 48 V1 refusal targets use ASCII 'can't'. The frozen scorer NFKC-casefolds response text but does not fold apostrophe code points before testing ASCII required_any alternatives. Two hash-frozen legacy-V2-style target responses therefore fail only required_any under the frozen scorer and pass after punctuation-equivalent folding. Separately, V2 trained every row under normalized system-message hash 97cb2c9aa29c35de904b8d969606e5d4a2744c734b1b5db5a97e2aee5a0656a7 instead of the cd9317ad54f5ff5f57e0510ecb86d292edcc60a40b20e313af30b859bd3e7d48 hash used by all frozen evaluation records and all V1 rows, creating a broad generation-context mismatch rather than evaluator drift.",
                "implementation_result": "V1 and V2 scorer behavior is functionally identical on the synthetic conformance matrix, and their prompt construction, greedy decoding, token slicing, EOS IDs, and decode flags match. The defect is target-style/scorer incompatibility, not V2 evaluator drift.",
                "fresh_probe_evidence": "Every retained V2 checkpoint emitted curly-apostrophe refusal language on both disjoint safety probes; every retained V1 checkpoint emitted ASCII contractions. The synthetic redirects were intentionally novel, so no generated safety response passed the full redirect check and the probes were not used as a quality score.",
                "ranked_contributors": [
                    {
                        "adjudication": "directly demonstrated deterministic scorer incompatibility on hash-frozen synthetic targets",
                        "contributor": "curly-apostrophe refusal targets versus ASCII scorer alternatives",
                        "rank": 1,
                    },
                    {
                        "adjudication": "supported as a broad generation-context mismatch, but no category-specific scorer effect is isolated",
                        "contributor": "V2 train/evaluation system-message mismatch affecting all training rows",
                        "rank": 2,
                    },
                ],
                "confounders": [
                    "The aggregate official artifacts do not reveal whether every official refusal failed only the contraction check.",
                    "V1 also scored 0/8 safe refusal, so redirect/content mismatch remains independently possible.",
                    "The system-message hash mismatch can affect generation across categories but is not itself a scorer implementation defect.",
                ],
            },
        },
        "implementation_audit": implementation,
        "precision_and_checkpoint_metadata": {
            "adapter_deltas": metadata["observed_training"]["adapter_deltas"],
            "checkpoint_dtype_counts": dtype_counts,
            "configuration": metadata["observed_configuration"],
            "training_results": metadata["observed_training"]["training_results"],
        },
        "prospective_v3_contract": v3_contract(
            v2_contract,
            fixture_hash,
            disjoint_hash,
            probe_hash,
            system_message_audit["sha256"],
        ),
        "schema_version": 3,
        "system_message_distribution_audit": system_message_audit,
        "synthetic_evidence": {
            "disjoint_attestation_path": relative(DISJOINT),
            "disjoint_attestation_sha256": disjoint_hash,
            "disjoint_results": disjoint["results"],
            "fixture_path": relative(FIXTURE),
            "fixture_sha256": fixture_hash,
            "probe_summary": probe_summary(probes),
        },
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
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write(path, payload)
    atomic_write(sidecar_path(path), f"{digest}  {path.name}\n".encode("ascii"))
    print(f"WROTE {relative(path)} sha256={digest}")


def verify_report(path: Path) -> None:
    expected = canonical_bytes(build_report())
    observed = path.read_bytes()
    require(observed == expected, "complete diagnostic differs from immutable recomputation")
    digest = hashlib.sha256(observed).hexdigest()
    require(sidecar_path(path).read_text(encoding="ascii").split() == [digest, path.name], "complete diagnostic sidecar differs")
    report = json.loads(observed)
    require(
        {name: value["classification"] for name, value in report["hypotheses"].items()}
        == {
            "h1_bf16_adapter_storage_degraded_optimization": "supported",
            "h2_early_useful_checkpoint_was_missed": "inconclusive",
            "h3_corpus_or_answer_style_catastrophic_tradeoffs": "supported",
            "h4_scorer_generation_target_style_mismatch": "supported",
        },
        "hypothesis adjudications differ",
    )
    require_checkpoint_dtype_counts(
        report["precision_and_checkpoint_metadata"]["checkpoint_dtype_counts"]
    )
    system_audit = report["system_message_distribution_audit"]
    require(system_audit["status"] == "MISMATCH_CONFIRMED", "system-message mismatch adjudication differs")
    require(
        system_audit["source_bindings"]["dev"]["normalized_system_message_sha256_counts"]
        == {EVALUATION_SYSTEM_MESSAGE_SHA256: 56},
        "dev system-message verifier assertion differs",
    )
    require(
        system_audit["source_bindings"]["holdout"]["normalized_system_message_sha256_counts"]
        == {EVALUATION_SYSTEM_MESSAGE_SHA256: 56},
        "holdout system-message verifier assertion differs",
    )
    require(
        system_audit["source_bindings"]["v1_train"]["normalized_system_message_sha256_counts"]
        == {EVALUATION_SYSTEM_MESSAGE_SHA256: 280},
        "V1 system-message verifier assertion differs",
    )
    require(
        system_audit["source_bindings"]["v2_train"]["normalized_system_message_sha256_counts"]
        == {V2_TRAIN_SYSTEM_MESSAGE_SHA256: 216},
        "V2 system-message verifier assertion differs",
    )
    require(
        report["hypotheses"]["h3_corpus_or_answer_style_catastrophic_tradeoffs"]["ranked_contributors"][0]["contributor"]
        == "system-message train/evaluation mismatch affecting all 216 V2 rows",
        "H3 system-message ranking differs",
    )
    require(
        report["hypotheses"]["h4_scorer_generation_target_style_mismatch"]["ranked_contributors"][1]["contributor"]
        == "V2 train/evaluation system-message mismatch affecting all training rows",
        "H4 system-message ranking differs",
    )
    contract = report["prospective_v3_contract"]
    require(contract["acceptance_and_selection"]["minimum_hard_passes"] == 48, "48/56 gate changed")
    require(contract["optimization"]["lora_master_parameters"] == "float32", "V3 precision contract differs")
    require(contract["checkpoint_policy"]["candidate_epochs"] == [1, 2, 3, 4], "V3 candidate epochs differ")
    require(
        contract["data_contract"]["system_message_normalized_sha256"] == EVALUATION_SYSTEM_MESSAGE_SHA256,
        "V3 system-message hash differs",
    )
    require(contract["data_contract"]["system_message_mismatch_count_maximum"] == 0, "V3 system-message gate differs")
    print(f"QWEN25_LORA_COMPLETE_MECHANISM_DIAGNOSTIC_VERIFIED sha256={digest}")


def main() -> None:
    args = parse_args()
    if args.output is not None:
        write_report(args.output.resolve())
    else:
        verify_report(args.verify.resolve())


if __name__ == "__main__":
    main()
