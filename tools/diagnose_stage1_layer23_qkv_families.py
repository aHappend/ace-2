#!/usr/bin/env python3
"""Isolate layer-23 Q/K/V and projection quantizer-family causality."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import peft
import tokenizers
import torch
import transformers
from peft import PeftModel
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import diagnose_stage1_same_prompt_causal as prior


MISSION_ID = "stage1layer23qkvfamilies01"
LAYER = 23
COMPONENTS = ("q", "k", "v")
OUTPUT = ROOT / "diagnosis/stage1layer23qkvfamilies01"
RESULT = OUTPUT / "result.json"
SUMS = OUTPUT / "SHA256SUMS"
REVIEW_REQUEST = OUTPUT / "fresh-l2-review-request.json"
PRIOR_RESULT = ROOT / "diagnosis/stage1samepromptcausal01/result.json"
PRIOR_RESULT_SHA256 = (
    "0e35e7ccafa5dc90324af4dee91e4a9c7d91075055965b6894c0adb750c1bff4"
)
FAMILIES = (
    "bf16_weight_with_a8_input_output",
    "w4_weight_with_bf16_input_a8_output",
)
QUANTIZER_PATH_FLAGS = {
    "bf16_projection_output_to_a8": {
        "w4_weight": False,
        "a8_input": False,
        "a8_output": True,
    },
    "bf16_weight_with_a8_input_output": {
        "w4_weight": False,
        "a8_input": True,
        "a8_output": True,
    },
    "w4_weight_with_bf16_input_a8_output": {
        "w4_weight": True,
        "a8_input": False,
        "a8_output": True,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return prior.localizer.file_record(path)


def capture_layer_context(
    snapshot: Path, token_ids: list[int]
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        attn_implementation="eager",
        device_map=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        model,
        prior.backend.canonical.ADAPTER_DIR,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    layer = model.model.model.layers[LAYER]
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def capture_layer_input(
        _module: nn.Module, inputs: tuple[Any, ...]
    ) -> None:
        require(
            bool(inputs) and isinstance(inputs[0], torch.Tensor),
            "BF16 layer input capture differs",
        )
        captures["layer_input"] = inputs[0][0].float().cpu().contiguous()

    def capture_output(name: str):
        def hook(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            value = output[0] if isinstance(output, tuple) else output
            require(
                isinstance(value, torch.Tensor),
                f"BF16 capture differs: {name}",
            )
            captures[name] = value[0].float().cpu().contiguous()

        return hook

    handles.append(layer.register_forward_pre_hook(capture_layer_input))
    handles.append(
        layer.input_layernorm.register_forward_hook(capture_output("input_norm"))
    )
    handles.append(layer.self_attn.q_proj.register_forward_hook(capture_output("q")))
    handles.append(layer.self_attn.k_proj.register_forward_hook(capture_output("k")))
    handles.append(layer.self_attn.v_proj.register_forward_hook(capture_output("v")))

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()
    logits = output.logits[0, -1].float().cpu().contiguous()
    require(
        set(captures) == {"layer_input", "input_norm", "q", "k", "v"},
        "BF16 component capture set differs",
    )
    require(
        all(value.shape[0] == len(token_ids) for value in captures.values()),
        "BF16 capture token count differs",
    )
    require(
        int(torch.argmax(logits)) == prior.REFERENCE_TOKEN,
        "captured BF16 top token changed",
    )
    require(
        prior.localizer.tensor_sha256(logits) == prior.BF16_LOGITS_SHA256,
        "captured BF16 logits differ from the frozen reference",
    )
    layer_input = captures.pop("layer_input")
    del model, output, input_ids
    gc.collect()
    return layer_input, captures, logits


class ComponentFamilyCache(prior.BoundaryProjectionCache):
    """Apply one Q/K/V BF16 substitution while retaining the A8 output."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        self.component: str | None = None
        self.family: str | None = None

    def configure_component(
        self,
        captures: dict[str, torch.Tensor],
        component: str,
        family: str,
    ) -> None:
        require(component in COMPONENTS, "component outside Q/K/V")
        require(
            family in {"bf16_projection_output_to_a8", *FAMILIES},
            "unsupported quantizer-family ablation",
        )
        self.layer_index = LAYER
        self.pattern = re.compile(rf"layer{LAYER}_position(\d+)_(q|k|v)")
        self.captures = captures
        self.input_boundary = None
        self.output_boundaries.clear()
        self.component = component
        self.family = family

    def clear_substitution(self) -> None:
        super().clear_substitution()
        self.component = None
        self.family = None

    def _merged_for(self, component: str) -> torch.Tensor:
        return self.merged[
            f"model.layers.{LAYER}.self_attn.{component}_proj"
        ]

    def _apply_component_ablation(
        self,
        result: dict[str, Any],
        match: re.Match[str] | None,
        merged: torch.Tensor,
    ) -> dict[str, Any]:
        if (
            match is None
            or self.component is None
            or self.family is None
            or match.group(2) != self.component
        ):
            return result
        position = int(match.group(1))
        metadata = self.metadata.get(merged.data_ptr())
        require(metadata is not None, "projection metadata absent for ablation")

        if self.family == "bf16_projection_output_to_a8":
            source = self.captures[self.component][position]
            source_description = "checkpoint176_exact_bf16_projection_output"
        elif self.family == "bf16_weight_with_a8_input_output":
            dequantized_input = (
                result["input_q"].to(torch.float32) * float(result["input_scale"])
            )
            source = torch.mv(
                merged.to(torch.bfloat16),
                dequantized_input.to(torch.bfloat16),
            ).float()
            source_description = (
                "checkpoint176_bf16_weight_times_baseline_dequantized_a8_input"
            )
        else:
            dequantized_w4 = (
                metadata["qweight"].to(torch.float32)
                * metadata["weight_scale"].to(torch.float32)[:, None]
            )
            source = torch.mv(
                dequantized_w4.to(torch.bfloat16),
                self.captures["input_norm"][position].to(torch.bfloat16),
            ).float()
            source_description = (
                "baseline_dequantized_w4_weight_times_checkpoint176_bf16_input"
            )

        result["output_q"] = prior.backend.canonical.quantize_int8(
            source, float(result["output_scale"])
        )
        result["component_family_ablation"] = {
            "layer": LAYER,
            "component": self.component,
            "family": self.family,
            "position": position,
            "source": source_description,
            "source_tensor_sha256": prior.localizer.tensor_sha256(
                source.to(torch.float32).contiguous()
            ),
            "a8_output_quantizer_retained": True,
        }
        return result

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        result = super().derive_projection(
            name,
            merged,
            input_q,
            input_scale,
            float_input,
            source_hashes,
        )
        return self._apply_component_ablation(
            result, self.pattern.fullmatch(name), merged
        )

    def from_fixed_metadata(
        self,
        name: str,
        input_q: torch.Tensor,
        input_scale: float,
        qweight: torch.Tensor,
        multiplier: torch.Tensor,
        right_shift: torch.Tensor,
        output_scale: float,
    ) -> dict[str, Any]:
        result = super().from_fixed_metadata(
            name,
            input_q,
            input_scale,
            qweight,
            multiplier,
            right_shift,
            output_scale,
        )
        match = self.pattern.fullmatch(name)
        merged = (
            self._merged_for(match.group(2))
            if match is not None
            else torch.empty(0)
        )
        return self._apply_component_ablation(result, match, merged)


def configuration(component: str, family: str) -> dict[str, Any]:
    quantizer_path_flags = QUANTIZER_PATH_FLAGS[family]
    value = {
        "configuration_id": f"layer23-{component}-{family}",
        "layer": LAYER,
        "component": component,
        "family": family,
        "initial_state": (
            "checkpoint-176 BF16 layer-23 input for all 34 frozen prefix positions"
        ),
        **quantizer_path_flags,
        "downstream": "unchanged accepted W4A8 layer-23 tail and tied W4 head",
    }
    return {
        "value": value,
        "sha256": prior.localizer.sha256_bytes(canonical_bytes(value)),
    }


def decorate_record(
    record: dict[str, Any], component: str, family: str
) -> dict[str, Any]:
    record = copy.deepcopy(record)
    record["configuration"] = configuration(component, family)
    record["component"] = component
    record["quantizer_family_ablation"] = family
    return record


def metrics(record: dict[str, Any]) -> dict[str, Any]:
    comparison = record["comparison_vs_checkpoint176_bf16"]
    return {
        "top_token_id": int(record["top_token_id"]),
        "token_39814_rank": int(record["reference_rank"]),
        "top8_overlap_count": int(comparison["top8_overlap_count"]),
        "jensen_shannon_divergence_nats": float(
            comparison["jensen_shannon_divergence_nats"]
        ),
        "relative_l2": float(comparison["raw_score_error"]["relative_l2"]),
        "logits_sha256": record["accepted_w4a8_logits"]["sha256"],
        "configuration_sha256": record["configuration"]["sha256"],
    }


def recovered(
    global_baseline: dict[str, Any], record: dict[str, Any]
) -> tuple[bool, str]:
    if record["top_token_id"] == prior.REFERENCE_TOKEN:
        return True, "exact_token_and_rank_recovery"
    if prior.materially_improves(global_baseline, record):
        return True, "preregistered_material_rank_top8_jsd_recovery"
    return False, "no_preregistered_causal_recovery"


def add_failure_analysis(
    layer23_baseline: dict[str, Any],
    global_baseline: dict[str, Any],
    record: dict[str, Any],
) -> None:
    did_recover, recovery_kind = recovered(global_baseline, record)
    record["causal_recovery"] = {
        "recovered": did_recover,
        "kind": recovery_kind,
    }
    if did_recover:
        return
    comparison = record["comparison_vs_checkpoint176_bf16"]
    baseline_comparison = layer23_baseline[
        "comparison_vs_checkpoint176_bf16"
    ]
    record["failure_analysis"] = {
        "failure_taxonomy_class": "QUALITY_CAUSAL_RECOVERY_MISS",
        "root_cause_hypothesis": (
            f"Replacing only {record['component']} family "
            f"{record['quantizer_family_ablation']} is insufficient; the causal "
            "error lies in another family or a component interaction."
        ),
        "regression": {
            "token_rank_delta_vs_layer23_baseline": int(
                record["reference_rank"] - layer23_baseline["reference_rank"]
            ),
            "top8_overlap_delta_vs_layer23_baseline": int(
                comparison["top8_overlap_count"]
                - baseline_comparison["top8_overlap_count"]
            ),
            "jsd_delta_nats_vs_layer23_baseline": float(
                comparison["jensen_shannon_divergence_nats"]
                - baseline_comparison["jensen_shannon_divergence_nats"]
            ),
        },
    }


def run_record(
    cache: ComponentFamilyCache,
    captures: dict[str, torch.Tensor],
    component: str,
    family: str,
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
    bf16_logits: torch.Tensor,
) -> dict[str, Any]:
    cache.configure_component(captures, component, family)
    label = f"layer23-{component}-{family}"
    record, _ = prior.run_cut_record(
        LAYER,
        label,
        hidden_states,
        weights,
        adapter,
        norm_gain,
        embedding,
        head,
        tokenizer,
        bf16_logits,
    )
    record = decorate_record(record, component, family)
    summary = metrics(record)
    print(
        "ACE2_LAYER23_QKV_FAMILY "
        f"component={component} family={family} "
        f"top={summary['top_token_id']} rank={summary['token_39814_rank']} "
        f"overlap={summary['top8_overlap_count']} "
        f"js={summary['jensen_shannon_divergence_nats']:.9f}",
        flush=True,
    )
    return record


def recommendation(component: str, causal_family: str) -> str:
    path = f"model.layers.{LAYER}.self_attn.{component}_proj"
    if causal_family == "w4_weight_quantization":
        return (
            f"Run one software-only {path} candidate with an alternative W4 row-scale "
            "quantizer while retaining the frozen A8 input/output quantizers; require "
            "token 39814 rank 1 with no top-8-overlap or JSD regression before any RTL work."
        )
    if causal_family == "a8_input_activation_quantization":
        return (
            f"Run one software-only {path} candidate that changes only its A8 input "
            "quantizer while retaining frozen W4 weights and A8 output quantization; "
            "require token 39814 rank 1 with no top-8-overlap or JSD regression."
        )
    return (
        f"Run one software-only {path} candidate that jointly changes only its W4 "
        "weight and A8 input quantizers while retaining A8 output quantization; require "
        "token 39814 rank 1 with no top-8-overlap or JSD regression."
    )


def generate() -> None:
    require(not OUTPUT.exists(), "layer-23 QKV family diagnosis already exists")
    require(
        prior.localizer.sha256_file(PRIOR_RESULT) == PRIOR_RESULT_SHA256,
        "prior incomplete diagnosis changed",
    )
    prior.verify()
    OUTPUT.mkdir(parents=True)
    started = time.monotonic()
    sealed_before = prior.sealed_records()
    snapshot = prior.generation_runner.resolve_snapshot()
    prior.backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    token_ids = prior.generation_runner.canonical_chat_token_ids(
        tokenizer, prior.PROMPT
    )
    require(len(token_ids) == 34, "frozen prompt token count changed")
    require(
        prior.localizer.sha256_bytes(
            canonical_bytes(token_ids)
        )
        == prior.TOKEN_IDS_SHA256,
        "frozen prompt token IDs changed",
    )
    require(
        prior.localizer.sha256_bytes(
            prior.PROMPT.encode("utf-8")
        )
        == prior.PROMPT_SHA256,
        "frozen prompt changed",
    )
    prior.localizer.PROMPT = prior.PROMPT
    prior.localizer.TOKEN_IDS = token_ids
    prior.localizer.REFERENCE_TOKEN = prior.REFERENCE_TOKEN

    print("ACE2_LAYER23_QKV_FAMILY_BF16_CAPTURE", flush=True)
    layer_input, captures, bf16_logits = capture_layer_context(
        snapshot, token_ids
    )
    hidden_states = [torch.empty(0)] * LAYER + [layer_input]
    capture_bindings = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": prior.localizer.tensor_sha256(value),
        }
        for name, value in sorted(
            {"layer_input": layer_input, **captures}.items()
        )
    }

    old_result = read_json(PRIOR_RESULT)
    global_baseline = old_result["layer_cut_scan"]["baseline"]
    layer23_baseline = next(
        record
        for record in [
            *old_result["layer_cut_scan"]["coarse_trace"],
            *old_result["layer_cut_scan"]["refinement_trace"],
        ]
        if record["cut"] == LAYER
    )
    component_records: list[dict[str, Any]] = []
    family_records: list[dict[str, Any]] = []
    suppressed_messages: list[str] = []

    prior_output = prior.OUTPUT
    prior.OUTPUT = OUTPUT
    try:
        with safe_open(
            prior.backend.canonical.MODEL, framework="pt", device="cpu"
        ) as weights, safe_open(
            prior.backend.canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            embedding = weights.get_tensor(
                "model.embed_tokens.weight"
            ).contiguous()
            norm_gain = weights.get_tensor("model.norm.weight").contiguous()
            head = prior.localizer.derive_lm_head_weights(embedding)
            cache = ComponentFamilyCache(weights, adapter)
            guard = prior.localizer.SubstitutionRequireGuard()
            cache.install()
            guard.install()
            try:
                for component in COMPONENTS:
                    record = run_record(
                        cache,
                        captures,
                        component,
                        "bf16_projection_output_to_a8",
                        hidden_states,
                        weights,
                        adapter,
                        norm_gain,
                        embedding,
                        head,
                        tokenizer,
                        bf16_logits,
                    )
                    add_failure_analysis(
                        layer23_baseline, global_baseline, record
                    )
                    component_records.append(record)

                recovered_components = [
                    record["component"]
                    for record in component_records
                    if record["causal_recovery"]["recovered"]
                ]
                family_targets = (
                    recovered_components
                    if recovered_components
                    else list(COMPONENTS)
                )
                for component in family_targets:
                    for family in FAMILIES:
                        record = run_record(
                            cache,
                            captures,
                            component,
                            family,
                            hidden_states,
                            weights,
                            adapter,
                            norm_gain,
                            embedding,
                            head,
                            tokenizer,
                            bf16_logits,
                        )
                        add_failure_analysis(
                            layer23_baseline, global_baseline, record
                        )
                        family_records.append(record)
            finally:
                cache.clear_substitution()
                suppressed_messages.extend(guard.suppressed)
                guard.restore()
                cache.restore()
    finally:
        prior.OUTPUT = prior_output

    exact_components = [
        record
        for record in component_records
        if record["top_token_id"] == prior.REFERENCE_TOKEN
    ]
    material_components = [
        record
        for record in component_records
        if record["causal_recovery"]["recovered"]
    ]
    selected = (
        exact_components[0]
        if exact_components
        else (material_components[0] if material_components else None)
    )
    require(
        selected is not None,
        "no individual Q/K/V component restored the preregistered distribution",
    )
    selected_component = selected["component"]
    selected_family_records = [
        record
        for record in family_records
        if record["component"] == selected_component
    ]
    weight_record = next(
        record
        for record in selected_family_records
        if record["quantizer_family_ablation"]
        == "bf16_weight_with_a8_input_output"
    )
    input_record = next(
        record
        for record in selected_family_records
        if record["quantizer_family_ablation"]
        == "w4_weight_with_bf16_input_a8_output"
    )
    weight_recovers = weight_record["causal_recovery"]["recovered"]
    input_recovers = input_record["causal_recovery"]["recovered"]
    if weight_recovers and not input_recovers:
        causal_family = "w4_weight_quantization"
        causal_record = weight_record
    elif input_recovers and not weight_recovers:
        causal_family = "a8_input_activation_quantization"
        causal_record = input_record
    else:
        causal_family = "joint_w4_weight_and_a8_input_quantization"
        causal_record = selected

    sealed_after = prior.sealed_records()
    require(
        sealed_after == sealed_before,
        "sealed attempt-0002 changed during diagnosis",
    )
    conclusion = {
        "component": selected_component,
        "quantizer_family": causal_family,
        "causal_configuration_id": causal_record["configuration"]["value"][
            "configuration_id"
        ],
        "causal_configuration_sha256": causal_record["configuration"][
            "sha256"
        ],
        "causal_metrics": metrics(causal_record),
        "a8_output_quantization": (
            "not_blocking_recovery because every component-output substitution "
            "retained the frozen A8 output quantizer"
        ),
        "l2_maximum_used_as_causal_evidence": False,
        "selection_policy": (
            "Q, K, V order; exact token recovery first, otherwise only the frozen "
            "material rank/top-8/JSD criterion; never select by L2 magnitude"
        ),
        "minimal_repair_recommendation": recommendation(
            selected_component, causal_family
        ),
    }
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": "COMPONENT_AND_QUANTIZER_FAMILY_CAUSALLY_LOCALIZED",
        "classification": (
            "bounded_nonofficial_software_only_layer23_qkv_diagnosis"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "frozen_contract": {
            "prompt_sha256": prior.PROMPT_SHA256,
            "token_ids_sha256": prior.TOKEN_IDS_SHA256,
            "token_count": len(token_ids),
            "position": 0,
            "checkpoint176_bf16_token_id": prior.REFERENCE_TOKEN,
            "checkpoint176_bf16_logits_sha256": prior.BF16_LOGITS_SHA256,
            "model_sha256": prior.MODEL_SHA256,
            "adapter_model_sha256": prior.ADAPTER_SHA256,
            "adapter_config_sha256": prior.ADAPTER_CONFIG_SHA256,
            "checkpoint_tree_sha256": prior.CHECKPOINT_TREE_SHA256,
            "prior_incomplete_diagnosis": prior.localizer.file_record(
                PRIOR_RESULT
            ),
            "sealed_attempt_before": sealed_before,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
        },
        "runner": prior.localizer.file_record(Path(__file__).resolve()),
        "bf16_capture_bindings": capture_bindings,
        "layer23_baseline_metrics": {
            "global_w4a8": metrics(
                decorate_record(
                    global_baseline, "qkv", "unchanged_global_w4a8"
                )
            ),
            "bf16_layer23_input_then_w4a8": metrics(
                decorate_record(
                    layer23_baseline,
                    "qkv",
                    "bf16_layer23_input_then_unchanged_w4a8",
                )
            ),
        },
        "component_isolation": {
            "method": (
                "Replace exactly one of Q, K, or V with its checkpoint-176 BF16 "
                "projection output, requantize it with the unchanged tensor-wide A8 "
                "output scale, and leave the other components and downstream path unchanged."
            ),
            "records": component_records,
        },
        "quantizer_family_ablation": {
            "method": (
                "For each causally recovering component, independently bypass W4 "
                "weights while retaining A8 input/output, then bypass A8 input while "
                "retaining dequantized W4 weights and A8 output."
            ),
            "records": family_records,
        },
        "causal_conclusion": conclusion,
        "suppressed_nonfunctional_cache_effect_assertions": {
            "count": len(suppressed_messages),
            "unique_messages": sorted(set(suppressed_messages)),
        },
        "sealed_attempt_after": sealed_after,
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "attempt_0003_created": False,
            "full_rtl_generation_run": False,
            "stage2_entered": False,
        },
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    write_json(RESULT, result)
    review_request = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "review_type": "Fresh-L2",
        "status": "PENDING_INDEPENDENT_REVIEW",
        "result": prior.localizer.file_record(RESULT),
        "source": prior.localizer.file_record(Path(__file__).resolve()),
        "requested_checks": [
            "Recompute the frozen prompt, token, model, adapter, and checkpoint bindings.",
            "Confirm Q, K, and V were isolated individually by bypassing W4/A8 input with A8 output retained.",
            "Recompute token-39814 rank, top-8 overlap, and JSD from every logits file.",
            "Confirm W4-weight and A8-input families were separated without L2 selection.",
            "Confirm exactly one repair recommendation follows the causal metrics.",
            "Confirm attempt-0002 and official namespaces remain unchanged.",
        ],
    }
    write_json(REVIEW_REQUEST, review_request)
    prior.localizer.write_sums(OUTPUT)
    print(
        "ACE2_LAYER23_QKV_FAMILY_RESULT "
        f"component={selected_component} family={causal_family} "
        f"output={prior.localizer.public_path(OUTPUT)}",
        flush=True,
    )


def all_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *result["component_isolation"]["records"],
        *result["quantizer_family_ablation"]["records"],
    ]


def verify() -> None:
    require(RESULT.is_file(), "layer-23 QKV family result is missing")
    require(REVIEW_REQUEST.is_file(), "Fresh-L2 review request is missing")
    require(SUMS.is_file(), "layer-23 QKV family SHA256SUMS is missing")
    for line in SUMS.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = OUTPUT / relative
        require(path.is_file(), f"summed artifact is missing: {relative}")
        require(
            prior.localizer.sha256_file(path) == expected,
            f"summed artifact hash differs: {relative}",
        )

    result = read_json(RESULT)
    contract = result["frozen_contract"]
    require(
        contract["prompt_sha256"] == prior.PROMPT_SHA256
        and contract["token_ids_sha256"] == prior.TOKEN_IDS_SHA256
        and contract["token_count"] == 34,
        "frozen prompt binding changed",
    )
    require(
        contract["model_sha256"] == prior.MODEL_SHA256
        and contract["adapter_model_sha256"] == prior.ADAPTER_SHA256
        and contract["checkpoint_tree_sha256"]
        == prior.CHECKPOINT_TREE_SHA256,
        "frozen model binding changed",
    )
    require(
        prior.localizer.sha256_file(PRIOR_RESULT) == PRIOR_RESULT_SHA256,
        "prior incomplete diagnosis changed",
    )
    reference_path = (
        ROOT
        / "diagnosis/stage1qualitydiag01/logits/"
        "checkpoint176-bf16-greedy-step-00-f32le.bin"
    )
    require(
        prior.localizer.sha256_file(reference_path)
        == prior.BF16_LOGITS_SHA256,
        "BF16 logits reference changed",
    )
    reference = np.fromfile(reference_path, dtype="<f4")
    require(
        reference.shape == (prior.backend.MODEL_OUTPUT_DOMAIN,),
        "BF16 logits shape changed",
    )
    records = all_records(result)
    require(
        [record["component"] for record in result["component_isolation"]["records"]]
        == list(COMPONENTS),
        "Q/K/V isolation order or membership changed",
    )
    for record in records:
        config = record["configuration"]
        family = record["quantizer_family_ablation"]
        require(
            prior.localizer.sha256_bytes(
                canonical_bytes(config["value"])
            )
            == config["sha256"],
            "configuration hash changed",
        )
        require(
            {
                key: config["value"][key]
                for key in ("w4_weight", "a8_input", "a8_output")
            }
            == QUANTIZER_PATH_FLAGS[family],
            f"quantizer-path semantics changed for {family}",
        )
        logits_record = record["accepted_w4a8_logits"]
        path = ROOT / logits_record["path"]
        require(
            prior.localizer.sha256_file(path) == logits_record["sha256"],
            "logits artifact hash changed",
        )
        scores = np.fromfile(path, dtype="<f8")
        require(
            scores.shape == (prior.backend.MODEL_OUTPUT_DOMAIN,),
            "candidate logits shape changed",
        )
        recomputed = prior.quality.score_comparison(reference, scores)
        require(
            recomputed == record["comparison_vs_checkpoint176_bf16"],
            f"metrics changed for {config['value']['configuration_id']}",
        )
        require(
            record["causal_recovery"]["recovered"]
            == recovered(
                read_json(PRIOR_RESULT)["layer_cut_scan"]["baseline"],
                record,
            )[0],
            "causal recovery classification changed",
        )

    conclusion = result["causal_conclusion"]
    causal_record = next(
        record
        for record in records
        if record["configuration"]["value"]["configuration_id"]
        == conclusion["causal_configuration_id"]
    )
    require(
        metrics(causal_record) == conclusion["causal_metrics"],
        "causal metrics changed",
    )
    require(
        conclusion["causal_configuration_sha256"]
        == causal_record["configuration"]["sha256"],
        "causal configuration binding changed",
    )
    recommendation_text = conclusion["minimal_repair_recommendation"]
    require(
        isinstance(recommendation_text, str) and recommendation_text.count(".") >= 1,
        "exactly one repair recommendation is required",
    )
    lowered = recommendation_text.lower()
    require(
        "per-head" not in lowered
        and "group-64" not in lowered
        and "grouped" not in lowered,
        "rejected scale candidate reappeared in the recommendation",
    )
    require(
        prior.sealed_records()
        == contract["sealed_attempt_before"]
        == result["sealed_attempt_after"],
        "sealed attempt-0002 binding changed",
    )
    require(
        not (
            ROOT
            / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0003"
        ).exists(),
        "attempt-0003 exists",
    )
    request = read_json(REVIEW_REQUEST)
    require(
        request["status"] == "PENDING_INDEPENDENT_REVIEW"
        and request["result"] == prior.localizer.file_record(RESULT)
        and request["source"]
        == prior.localizer.file_record(Path(__file__).resolve()),
        "Fresh-L2 review request binding changed",
    )
    print(
        "PASS stage1layer23qkvfamilies01 "
        f"component={conclusion['component']} "
        f"family={conclusion['quantizer_family']} "
        f"rank={conclusion['causal_metrics']['token_39814_rank']} "
        f"overlap={conclusion['causal_metrics']['top8_overlap_count']} "
        f"js={conclusion['causal_metrics']['jensen_shannon_divergence_nats']:.9f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        generate()
        verify()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ACE2_LAYER23_QKV_FAMILY_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
