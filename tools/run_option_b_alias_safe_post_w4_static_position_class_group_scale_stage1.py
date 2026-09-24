#!/usr/bin/env python3
"""Prepare and execute the exactly-once alias-safe post-W4 V2 candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_grouped_scale32_stage1 as grouped
import run_option_b_static_position_class_group_scale_stage1 as base
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    TERMINATION_TOKEN_IDS,
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
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM
from run_all_transformer_per32_stage1 import (
    aggregate_prompts,
    generate_reference,
    log_message,
    prompt_binding,
    sequence_disagreements,
    utc_now,
)


CANDIDATE_ID = "option_b_alias_safe_post_w4_static_position_class_group_scale_w4a8_v2"
MISSION_ID = "daa1cf3e365f"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_POST_W4_STATIC_POSITION_CLASS_GROUP_SCALE_W4A8_V2_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_POST_W4_STATIC_POSITION_CLASS_GROUP_SCALE_W4A8_V2_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
SOURCE_CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
CALIBRATION_CONTRACT_PATH = CALIBRATION_DIR / "calibration_contract.json"
CALIBRATION_REPORT_PATH = CALIBRATION_DIR / "calibration_report.json"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
SOURCE_EMBEDDING_SHA256 = "4e96b0df6d274768cbb7e72404011853d23349999b658dc2f4dfb3c431ea223f"
OUTPUT_DIR = ROOT / "build/stage1-option-b-alias-safe-post-w4-static-position-class-group-scale-w4a8-v2"
PREATTEMPT_DIR = OUTPUT_DIR / "preattempt"
TABLE_PATH = PREATTEMPT_DIR / "static_position_class_scale_table.json"
TABLE_COMPANION = PREATTEMPT_DIR / "static_position_class_scale_table.sha256"
INPUT_MANIFEST_PATH = PREATTEMPT_DIR / "non_evaluator_inputs.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
COVERAGE_PATH = PREATTEMPT_DIR / "static_table_coverage.json"
PREFLIGHT_PATH = PREATTEMPT_DIR / "fresh_candidate_preflight.json"
SELF_TEST_PATH = PREATTEMPT_DIR / "runner_self_test.json"
CHRONOLOGY_PATH = PREATTEMPT_DIR / "preparation_chronology.json"
REPAIR_PATH = PREATTEMPT_DIR / "preparation_repair.json"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
READY_PATH = OUTPUT_DIR / "PREATTEMPT_READY.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
MARKER_PATH = ATTEMPT_DIR / "execution_started.json"
RUNNER_PATH = Path(__file__).resolve()
BASE_RUNNER_PATH = ROOT / "tools/run_option_b_static_position_class_group_scale_stage1.py"
GROUPED_PATH = ROOT / "tools/run_option_b_grouped_scale32_stage1.py"
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16
ADDITIONAL_PROMPTS = (
    "Name one primary color in one lowercase word.",
    "Return only the integer result of 9 multiplied by 7.",
    "In five words, describe gentle rain.",
    "Answer yes or no: Is water wet?",
)
STATE_MATCH_KEYS = (
    "embedding_bf16_sha256",
    "lm_head_packed_w4_sha256",
    "lm_head_scale32_sha256",
    "transformer_packed_w4_scope_sha256",
    "transformer_weight_scale32_scope_sha256",
    "weight_manifest_sha256",
    "candidate_state_manifest_sha256",
)

base.TABLE_PATH = TABLE_PATH


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def require_project_python() -> dict[str, Any]:
    grouped.require_project_python()
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), "project Python is missing")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")
    return {
        "entrypoint": "./.venv/bin/python",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {**verify_versions(), "numpy": importlib.metadata.version("numpy")},
        "torch_num_threads": TORCH_THREADS,
    }


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")


def write_companion(path: Path, companion: Path) -> None:
    companion.write_text(
        f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )


def atomic_write_json(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def raw_tensor_bytes(value: Tensor) -> bytes:
    flat = value.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes(order="C")


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(raw_tensor_bytes(value)).hexdigest()


def state_manifest(model: nn.Module) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name, value in sorted(model.state_dict().items()):
        raw = raw_tensor_bytes(value)
        records.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    body = {
        "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "tensor_count": len(records),
        "tensors": records,
    }
    return {**body, "candidate_state_manifest_sha256": canonical_sha256(body)}


def audit_attempt_namespace(expect_root_absent: bool) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for marker in sorted((ROOT / "build").glob("**/execution_started.json")):
        try:
            record = load_json(marker)
        except Exception:
            continue
        if record.get("candidate_id") == CANDIDATE_ID:
            matches.append(file_record(marker))
    attempts = sorted(path.name for path in OUTPUT_DIR.glob("attempt-*") if path.is_dir()) if OUTPUT_DIR.exists() else []
    if expect_root_absent:
        require(not OUTPUT_DIR.exists(), "V2 namespace existed before preparation")
    require(not matches, "a prior V2 execution marker exists")
    require(not attempts, f"V2 attempt namespace is already consumed: {attempts}")
    return {
        "candidate_id": CANDIDATE_ID,
        "root_existed_at_audit": OUTPUT_DIR.exists(),
        "matching_execution_markers": matches,
        "attempt_directories": attempts,
        "authorization_consumed": False,
    }


def recover_nonqualifying_preparation() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {
            "fresh_namespace": True,
            "recovered_nonqualifying_preparation": False,
            "task_start_namespace_absence_enforced": True,
        }
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a consumed attempt namespace")
    require(not PREATTEMPT_DIR.exists(), "cannot replace frozen pre-attempt artifacts")
    require(not CONTRACT_PATH.exists() and not READY_PATH.exists(), "cannot replace frozen pre-attempt closure")
    stale = sorted(OUTPUT_DIR.glob("preparation_failure_*.json"))
    failure_taxonomies: list[str] = []
    for path in stale:
        record = load_json(path)
        error = str(record.get("error", ""))
        if record.get("error_type") == "StaticScaleOverflow":
            failure_taxonomies.append("PREATTEMPT_STATIC_SCALE_CLOSURE_REQUIRED")
        elif "did not converge" in error:
            failure_taxonomies.append("PREATTEMPT_STATIC_SCALE_CLOSURE_NONCONVERGENCE")
        else:
            failure_taxonomies.append("PREATTEMPT_EXECUTION_FAILURE")
    temp_table = OUTPUT_DIR / ".static_position_class_scale_table.preflight.json"
    if temp_table.exists():
        stale.append(temp_table)
    require(stale, "existing V2 namespace has ambiguous provenance")
    archive_root = OUTPUT_DIR / "preflight-failures"
    archive_root.mkdir(exist_ok=True)
    ordinal = 1
    while (archive_root / f"failure-{ordinal:04d}").exists():
        ordinal += 1
    archive = archive_root / f"failure-{ordinal:04d}"
    archive.mkdir()
    archived: list[dict[str, Any]] = []
    for path in stale:
        destination = archive / path.name
        os.replace(path, destination)
        destination.chmod(0o444)
        archived.append(file_record(destination))
    return {
        "fresh_namespace": False,
        "recovered_nonqualifying_preparation": True,
        "task_start_namespace_absence_enforced": True,
        "known_prior_runner_sha256": [
            "922deac9de089c8f8005e5a155adb3ccae35fa9aa535607d4deb809284b93869",
            "19760f55cf1b4aa636e47060627707a4e382ddcb22160f08e93698ac3513eff5",
        ],
        "failure_taxonomies": failure_taxonomies,
        "root_cause_hypotheses": [
            "raw post-W4 boundary maxima did not include activation growth caused by upstream frozen-A8 quantization",
            "whole-matrix sweeps propagated late prompt scale changes too slowly to reach a global zero-update pass within 16 sweeps",
        ],
        "regression": "per-prompt closure must converge and a subsequent complete global sweep plus fresh candidate B replay must have zero updates and zero overflows",
        "authorization_consumed": False,
        "official_execution_marker_exists": False,
        "archived_artifacts": archived,
    }


def alias_split_self_test() -> dict[str, Any]:
    embedding = nn.Embedding(4, 4, dtype=torch.bfloat16)
    head = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    head.weight = embedding.weight
    require(head.weight is embedding.weight, "synthetic alias object is missing")
    before = tensor_sha256(embedding.weight)
    head.weight = nn.Parameter(head.weight.detach().clone(), requires_grad=False)
    require(head.weight is not embedding.weight, "synthetic alias object was not separated")
    require(head.weight.data_ptr() != embedding.weight.data_ptr(), "synthetic alias storage was not separated")
    require(tensor_sha256(head.weight) == before == tensor_sha256(embedding.weight), "synthetic split changed bytes")
    with torch.no_grad():
        head.weight.add_(1)
    require(tensor_sha256(embedding.weight) == before, "synthetic head write changed embedding")
    return {
        "same_object_before": True,
        "same_storage_before": True,
        "different_object_after": True,
        "different_storage_after": True,
        "embedding_unchanged_after_head_write": True,
    }


def runner_self_test() -> dict[str, Any]:
    require_project_python()
    static = base.static_quantizer_self_test()
    alias = alias_split_self_test()
    probe = {"runner": sha256_file(RUNNER_PATH), "static": static, "alias": alias}
    require(canonical_sha256(probe) != canonical_sha256({**probe, "runner": "0" * 64}), "hash guard self-test failed")
    return {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "static_quantizer": static,
        "alias_split": alias,
        "hash_mismatch_rejected": True,
        "runner_sha256": sha256_file(RUNNER_PATH),
    }


def load_candidate(label: str) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    embedding = model.model.embed_tokens.weight
    lm_head = model.lm_head.weight
    require(model.config.tie_word_embeddings is True, "source tie_word_embeddings differs")
    require(lm_head is embedding, "source lm_head and embedding Parameter objects differ")
    require(lm_head.data_ptr() == embedding.data_ptr(), "source lm_head and embedding storage differs")
    source_shape = list(embedding.shape)
    source_dtype = str(embedding.dtype)
    source_object_id = id(embedding)
    source_pointer = embedding.data_ptr()
    source_hash = tensor_sha256(embedding)
    require(source_hash == SOURCE_EMBEDDING_SHA256, "source embedding BF16 hash differs")

    cloned = lm_head.detach().clone()
    require(tensor_sha256(cloned) == source_hash, "detached lm_head clone differs before split")
    model.lm_head.weight = nn.Parameter(cloned, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding, "lm_head Parameter object remained tied")
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head storage remained tied")
    require(tensor_sha256(model.lm_head.weight) == source_hash, "lm_head clone bytes differ after split")
    require(id(model.model.embed_tokens.weight) == source_object_id, "embedding Parameter object changed during split")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed during split")

    weight_manifest, grouped_hashes = grouped.quantize_model_weights(model)
    require(len(weight_manifest) == 169, "W4 linear count differs")
    require(sum(record["module"] != "lm_head" for record in weight_manifest) == 168, "transformer W4 count differs")
    require(model.config.tie_word_embeddings is False, "candidate config was retied")
    require(model.model.embed_tokens.weight is embedding, "embedding Parameter object changed after W4")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed after W4")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after W4")
    lm_record = next(record for record in weight_manifest if record["module"] == "lm_head")
    transformer_records = [record for record in weight_manifest if record["module"] != "lm_head"]
    state = state_manifest(model)
    identity = {
        "embedding_bf16_sha256": embedding_hash,
        "lm_head_packed_w4_sha256": lm_record["packed_w4_sha256"],
        "lm_head_scale32_sha256": lm_record["weight_scale32_sha256"],
        "transformer_packed_w4_scope_sha256": grouped_hashes["transformer_packed_w4_scope_sha256"],
        "transformer_weight_scale32_scope_sha256": grouped_hashes["transformer_weight_scale32_scope_sha256"],
        "weight_manifest_sha256": grouped_hashes["weight_manifest_sha256"],
        "transformer_weight_manifest_sha256": canonical_sha256(transformer_records),
        "candidate_state_manifest_sha256": state["candidate_state_manifest_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "construction_label": label,
        "source_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "config_tie_word_embeddings_after_split": False,
        "module_counts": {"transformer_linears": 168, "lm_head": 1, "total_w4_linears": 169},
        "identity": identity,
        "weight_manifest": weight_manifest,
        "state_manifest": state,
    }
    witness = {
        "construction_label": label,
        "source": {
            "same_parameter_object": True,
            "same_storage_pointer": True,
            "embedding_object_id": source_object_id,
            "embedding_storage_pointer": source_pointer,
            "embedding_shape": source_shape,
            "embedding_dtype": source_dtype,
            "embedding_bf16_sha256": source_hash,
        },
        "post_split": {
            "different_parameter_objects": model.lm_head.weight is not embedding,
            "different_storage_pointers": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "embedding_object_preserved": id(embedding) == source_object_id,
            "embedding_storage_preserved": embedding.data_ptr() == source_pointer,
            "pre_w4_byte_identical": True,
            "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        },
        "post_w4": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_packed_w4_sha256": lm_record["packed_w4_sha256"],
            "lm_head_scale32_sha256": lm_record["weight_scale32_sha256"],
            "transformer_w4_count": len(transformer_records),
        },
    }
    return model, manifest, witness


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> Tensor:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(ids, Tensor) and ids.ndim == 2 and ids.shape[0] == 1, "chat tokenization shape differs")
    return ids


def non_evaluator_inputs(tokenizer: Any) -> tuple[list[Tensor], dict[str, Any]]:
    calibration = load_json(CALIBRATION_CONTRACT_PATH)
    frozen = calibration["chat"]["prompts"]
    require(len(frozen) == 4, "frozen calibration prompt count differs")
    tensors: list[Tensor] = []
    records: list[dict[str, Any]] = []
    for item in frozen:
        ids = tokenize_messages(tokenizer, item["messages"])
        token_ids = [int(value) for value in ids[0].tolist()]
        require(token_ids == item["token_ids"], "frozen calibration token IDs differ")
        tensors.append(ids)
        records.append(
            {
                "case_id": f"frozen_calibration_{int(item['ordinal']):02d}",
                "source": "frozen_calibration_contract",
                "messages": item["messages"],
                "token_ids": token_ids,
                "token_ids_sha256": item["token_ids_sha256"],
            }
        )
    for index, prompt in enumerate(ADDITIONAL_PROMPTS):
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        ids = tokenize_messages(tokenizer, messages)
        token_ids = [int(value) for value in ids[0].tolist()]
        tensors.append(ids)
        records.append(
            {
                "case_id": f"additional_preflight_{index:02d}",
                "source": "engineer_task_additional_prompt",
                "messages": messages,
                "prompt_sha256": sha256_bytes(prompt.encode()),
                "token_ids": token_ids,
                "token_ids_sha256": sha256_bytes(b"".join(value.to_bytes(4, "little") for value in token_ids)),
            }
        )
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "input_count": len(records),
        "official_nine_prompt_inputs_used": False,
        "calibration_contract": file_record(CALIBRATION_CONTRACT_PATH),
        "records": records,
    }
    require(len(tensors) == 8, "non-evaluator input count differs")
    return tensors, manifest


def run_generation(model: nn.Module, input_ids: Tensor, boundary: Any) -> tuple[list[int], int]:
    prompt_length = int(input_ids.shape[1])
    prefix = input_ids
    generated: list[int] = []
    forwards = 0
    with torch.inference_mode():
        for _index in range(MAX_NEW_TOKENS):
            boundary.begin_forward(prompt_length)
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1]
            forwards += 1
            token = int(logits.argmax())
            generated.append(token)
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
        boundary.begin_forward(prompt_length)
        model(input_ids=prefix, use_cache=False)
        forwards += 1
    return generated, forwards


class OfflineTableClosurePolicy(base.BoundaryHooks):
    """Monotonically close a not-yet-frozen table over its frozen inputs."""

    def __init__(self, table: dict[str, Any]) -> None:
        super().__init__()
        self.table = table
        self.prompt_length: int | None = None
        self.pass_index = 0
        self.pass_updates: list[dict[str, Any]] = []
        self.total_record_updates = 0

    def begin_pass(self, pass_index: int) -> None:
        self.pass_index = pass_index
        self.pass_updates = []

    def begin_forward(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length

    def apply(self, name: str, value: Tensor, layout: str) -> Tensor:
        require(self.prompt_length is not None, "offline closure prompt length is unset")
        event = self.table["events"][name]
        require(event["layout"] == layout, f"offline closure layout differs: {name}")
        lanes = int(event["group_lanes"])
        groups = int(event["group_count"])
        canonical, restore = base.canonicalize(value, layout)
        require(canonical.shape[-1] == lanes * groups, f"offline closure width differs: {name}")
        output = torch.empty_like(canonical, dtype=torch.float64)
        for position_class, (start, stop) in base.class_slices(int(canonical.shape[1]), self.prompt_length).items():
            if start == stop:
                continue
            class_record = event["classes"][position_class]
            records = [int(item) for item in class_record["scale32_records"]]
            collected = [float(item) for item in class_record["calibration_absmax"]]
            selected = canonical[:, start:stop].detach().to(torch.float64)
            flat = selected.reshape(-1, groups, lanes)
            maxima = [float(item) for item in flat.abs().amax(dim=(0, 2)).cpu().tolist()]
            for group_index, maximum in enumerate(maxima):
                collected[group_index] = max(collected[group_index], maximum)
                old_record = records[group_index]
                old_limit = 127.0 * grouped.scale32_value(old_record)
                if maximum > old_limit:
                    new_record = grouped.scale32_record(collected[group_index] / 127.0)
                    require(127.0 * grouped.scale32_value(new_record) >= maximum, "offline closure Scale32 record does not cover maximum")
                    records[group_index] = new_record
                    self.pass_updates.append(
                        {
                            "event": name,
                            "position_class": position_class,
                            "group": group_index,
                            "observed_maximum": maximum,
                            "old_record": old_record,
                            "new_record": new_record,
                        }
                    )
                    self.total_record_updates += 1
            class_record["calibration_absmax"] = collected
            class_record["scale32_records"] = records
            class_record["scale_values"] = [grouped.scale32_value(item) for item in records]
            scales = torch.tensor(class_record["scale_values"], dtype=torch.float64, device=value.device)
            quantized = torch.round(flat / scales[None, :, None])
            require(bool(torch.all(quantized >= -127) and torch.all(quantized <= 127)), f"offline closure escaped symmetric A8: {name}:{position_class}")
            q8 = quantized.to(torch.int8)
            require(not bool(torch.any(q8 == -128)), "offline closure produced reserved signed-A8 value")
            output[:, start:stop] = (q8.to(torch.float64) * scales[None, :, None]).reshape_as(selected)
        return restore(output).to(value.dtype)

    def install(self, model: nn.Module) -> None:
        self._install_common(model, self.apply, self.apply, self.apply)


def derive_table(model: nn.Module, inputs: list[Tensor], input_manifest: dict[str, Any], candidate_manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    observer = base.CalibrationObserver()
    observer.install(model)
    generated: list[dict[str, Any]] = []
    try:
        for record, input_ids in zip(input_manifest["records"], inputs, strict=True):
            tokens, forwards = run_generation(model, input_ids, observer)
            generated.append(
                {
                    "case_id": record["case_id"],
                    "generated_token_ids": tokens,
                    "generated_token_ids_sha256": sha256_bytes(b"".join(value.to_bytes(4, "little") for value in tokens)),
                    "model_forward_count": forwards,
                }
            )
        events = observer.table_events()
    finally:
        observer.uninstall()
    table = {
        "schema_version": 2,
        "candidate_id": CANDIDATE_ID,
        "classification": "offline_fixed_post_w4_layer_event_position_class_group_a8_scale_table",
        "derivation": {
            "candidate_state_manifest_sha256": candidate_manifest["identity"]["candidate_state_manifest_sha256"],
            "weight_manifest_sha256": candidate_manifest["identity"]["weight_manifest_sha256"],
            "source": "exact alias-separated post-W4 candidate A activations",
            "input_manifest_sha256": canonical_sha256(input_manifest),
            "official_nine_prompt_inputs_used": False,
            "position_classes": {"sink": "absolute position zero", "prompt": "positions one through prompt length minus one", "decode": "positions at or beyond prompt length"},
            "group_scale_rule": "smallest normalized Scale32 record covering absolute maximum divided by 127",
            "rounding": "round_to_nearest_ties_to_even",
            "saturation_or_silent_clipping_permitted": False,
            "runtime_scale_selection_permitted": False,
            "dynamic_calibration_permitted": False,
        },
        "geometry": {
            "decoder_layers": base.LAYERS,
            "event_families": list(base.EVENT_SPECS),
            "event_count": base.LAYERS * len(base.EVENT_SPECS),
            "groups_per_layer": base.GROUPS_PER_LAYER,
            "position_classes": list(base.POSITION_CLASSES),
            "records_per_layer": base.GROUPS_PER_LAYER * len(base.POSITION_CLASSES),
            "bytes_per_layer": base.GROUPS_PER_LAYER * len(base.POSITION_CLASSES) * 4,
            "bytes_all_layers": base.GROUPS_PER_LAYER * len(base.POSITION_CLASSES) * 4 * base.LAYERS,
        },
        "generated_sequences": generated,
        "events": events,
    }
    for event in table["events"].values():
        for class_record in event["classes"].values():
            class_record["raw_post_w4_absmax"] = list(class_record["calibration_absmax"])
    closure = OfflineTableClosurePolicy(table)
    closure.install(model)
    sweep_summaries: list[dict[str, Any]] = []
    closure_call_index = 0
    final_generated: list[dict[str, Any]] = []
    try:
        for sweep_index in range(1, 33):
            sweep_update_count = 0
            prompt_summaries: list[dict[str, Any]] = []
            for record, input_ids in zip(input_manifest["records"], inputs, strict=True):
                local_passes: list[dict[str, Any]] = []
                for local_index in range(1, 65):
                    closure_call_index += 1
                    closure.begin_pass(closure_call_index)
                    tokens, forwards = run_generation(model, input_ids, closure)
                    updates = list(closure.pass_updates)
                    update_count = len(updates)
                    sweep_update_count += update_count
                    local_passes.append(
                        {
                            "local_pass_index": local_index,
                            "scale32_record_update_count": update_count,
                            "updates_sha256": canonical_sha256(updates),
                            "first_updates": updates[:8],
                            "generated_token_ids": tokens,
                            "model_forward_count": forwards,
                        }
                    )
                    if update_count == 0:
                        break
                else:
                    counts = [item["scale32_record_update_count"] for item in local_passes]
                    raise RuntimeError(
                        f"offline per-prompt closure did not converge case={record['case_id']} "
                        f"sweep={sweep_index} update_counts={counts}"
                    )
                prompt_summaries.append(
                    {
                        "case_id": record["case_id"],
                        "local_pass_count": len(local_passes),
                        "local_passes": local_passes,
                    }
                )
            sweep_summaries.append(
                {
                    "sweep_index": sweep_index,
                    "scale32_record_update_count": sweep_update_count,
                    "prompts": prompt_summaries,
                }
            )
            if sweep_update_count == 0:
                final_generated = [
                    {
                        "case_id": item["case_id"],
                        "generated_token_ids": item["local_passes"][-1]["generated_token_ids"],
                        "model_forward_count": item["local_passes"][-1]["model_forward_count"],
                    }
                    for item in prompt_summaries
                ]
                break
        else:
            counts = [item["scale32_record_update_count"] for item in sweep_summaries]
            raise RuntimeError(f"offline static-table closure did not converge in 32 global sweeps update_counts={counts}")
    finally:
        closure.uninstall()
    require(sweep_summaries[-1]["scale32_record_update_count"] == 0, "offline closure lacks a zero-update final sweep")
    table["derivation"]["offline_static_table_closure"] = {
        "algorithm": "monotonic per-prompt Scale32 closure over frozen non-evaluator inputs followed by a complete global zero-update sweep",
        "global_sweep_count": len(sweep_summaries),
        "total_scale32_record_updates": closure.total_record_updates,
        "runtime_or_official_dynamic_scale_selection": False,
        "final_generated_sequences": final_generated,
        "sweeps": sweep_summaries,
    }
    coverage = table_coverage(table)
    return table, coverage


def table_coverage(table: dict[str, Any]) -> dict[str, Any]:
    expected_events = {base.event_name(layer, family) for layer in range(base.LAYERS) for family in base.EVENT_SPECS}
    require(set(table["events"]) == expected_events, "static table event closure differs")
    record_count = 0
    class_count = 0
    for name in sorted(expected_events):
        event = table["events"][name]
        require(set(event["classes"]) == set(base.POSITION_CLASSES), f"position classes differ: {name}")
        for position_class in base.POSITION_CLASSES:
            records = event["classes"][position_class]["scale32_records"]
            require(len(records) == event["group_count"], f"Scale32 group count differs: {name}:{position_class}")
            record_count += len(records)
            class_count += 1
    require(len(expected_events) == 312, "event count differs")
    require(class_count == 936, "event-position-class count differs")
    require(record_count == 13_176, "Scale32 record count differs")
    require(record_count * 4 == 52_704, "static table byte count differs")
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "all_312_layer_event_names_present": True,
        "all_three_position_classes_present_at_every_event": True,
        "event_count": len(expected_events),
        "event_position_class_count": class_count,
        "scale32_record_count": record_count,
        "static_table_bytes_all_24_layers": record_count * 4,
    }


def compare_candidates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        key: candidate_a["identity"][key] == candidate_b["identity"][key]
        for key in STATE_MATCH_KEYS
    }
    require(all(comparisons.values()), f"candidate A/B state differs: {comparisons}")
    return {"state_match_keys": list(STATE_MATCH_KEYS), "comparisons": comparisons, "all_match": True}


def preflight_table(model: nn.Module, table: dict[str, Any], inputs: list[Tensor], input_manifest: dict[str, Any]) -> dict[str, Any]:
    policy = base.StaticActivationPolicy(table)
    policy.install(model)
    generated: list[dict[str, Any]] = []
    forwards = 0
    try:
        for record, input_ids in zip(input_manifest["records"], inputs, strict=True):
            tokens, count = run_generation(model, input_ids, policy)
            forwards += count
            generated.append({"case_id": record["case_id"], "generated_token_ids": tokens, "model_forward_count": count})
        summary = policy.summary(forwards)
    finally:
        policy.uninstall()
    require(summary["runtime_scale_selection_count"] == 0, "runtime scale selection occurred")
    require(summary["dynamic_calibration_count"] == 0, "dynamic calibration occurred")
    require(summary["per_token_scale_sidecar_bytes"] == 0, "per-token scale sidecar was emitted")
    require(set(summary["class_position_vectors_across_all_boundaries"]) == set(base.POSITION_CLASSES), "preflight position-class coverage differs")
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_ZERO_OVERFLOW_FRESH_CANDIDATE",
        "fresh_model_reconstruction": True,
        "fixed_table_overflow_count": 0,
        "runtime_scale_selection_count": 0,
        "dynamic_calibration_count": 0,
        "per_token_scale_sidecar_bytes": 0,
        "official_nine_prompt_inputs_used": False,
        "official_execution_marker_exists": MARKER_PATH.exists(),
        "activation_execution": summary,
        "generated_sequences": generated,
    }


def artifact_records() -> dict[str, Any]:
    paths = {
        "engineer_task": TASK_PATH,
        "engineer_task_companion": TASK_COMPANION,
        "planning_requirements": PLAN_PATH,
        "planning_requirements_companion": PLAN_COMPANION,
        "source_contract": SOURCE_CONTRACT_PATH,
        "calibration_contract": CALIBRATION_CONTRACT_PATH,
        "calibration_report": CALIBRATION_REPORT_PATH,
        "image_manifest": IMAGE_DIR / "manifest.json",
        "official_matrix": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "base_static_runner_source": BASE_RUNNER_PATH,
        "grouped_weight_utility": GROUPED_PATH,
        "shared_evaluator_source": ROOT / "tools/run_all_transformer_per32_stage1.py",
        "prompt_source": ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py",
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "oracle_source": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
        "identity_helper_source": ROOT / "tools/qwen_instruct_option_b.py",
        "fixed_point_reference": ROOT / "tools/ace2_full_model_fixed_point.py",
        "quality_contracts": ROOT / "tools/ace2_quality_contracts.py",
        "non_evaluator_inputs": INPUT_MANIFEST_PATH,
        "candidate_a_manifest": CANDIDATE_A_PATH,
        "candidate_b_manifest": CANDIDATE_B_PATH,
        "alias_separation_witness": ALIAS_WITNESS_PATH,
        "static_scale_table": TABLE_PATH,
        "static_scale_table_companion": TABLE_COMPANION,
        "static_table_coverage": COVERAGE_PATH,
        "fresh_candidate_preflight": PREFLIGHT_PATH,
        "runner_self_test": SELF_TEST_PATH,
        "preparation_chronology": CHRONOLOGY_PATH,
    }
    if REPAIR_PATH.is_file():
        paths["preparation_repair"] = REPAIR_PATH
    failure_root = OUTPUT_DIR / "preflight-failures"
    if failure_root.is_dir():
        for index, path in enumerate(sorted(item for item in failure_root.rglob("*") if item.is_file())):
            paths[f"nonqualifying_preflight_failure_{index:02d}"] = path
    return {name: file_record(path) for name, path in paths.items()}


def build_contract(environment: dict[str, Any], source_identity: dict[str, Any], namespace_audit: dict[str, Any]) -> dict[str, Any]:
    task = load_json(TASK_PATH)
    require(task["candidate_id"] == CANDIDATE_ID, "Engineer task candidate differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1, "Engineer task attempt budget differs")
    require(task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "Engineer task was already consumed")
    matrix = load_json(MATRIX_PATH)
    prompts = prompt_binding(matrix)
    require(len(prompts) == 9, "official prompt count differs")
    artifacts = artifact_records()
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "authority": {
            "engineer_task": artifacts["engineer_task"],
            "planning_requirements": artifacts["planning_requirements"],
            "attempt_id": "attempt-0001",
            "exact_authorized_candidate_attempts": 1,
            "attempts_consumed_before_execution": 0,
            "selected_policy_id_before_execution": None,
        },
        "source_model": source_identity,
        "separation_claim": {
            "policy_id": "source_tied_checkpoint_runtime_split_clone_lm_head_v1",
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_preserved": True,
            "lm_head_storage_separate_before_w4": True,
            "transformer_w4_linears": 168,
            "separated_lm_head_w4_linears": 1,
            "candidate_a_and_b_state_manifests_match": True,
            "witness": artifacts["alias_separation_witness"],
        },
        "static_table": {
            "table": artifacts["static_scale_table"],
            "coverage": artifacts["static_table_coverage"],
            "derivation_inputs": artifacts["non_evaluator_inputs"],
            "official_matrix_inputs_used_for_derivation_or_preflight": False,
            "records_per_decoder_layer": 549,
            "bytes_per_decoder_layer": 2_196,
            "bytes_all_24_layers": 52_704,
            "fixed_table_overflow_count": 0,
            "runtime_scale_selection_count": 0,
            "dynamic_calibration_count": 0,
            "per_token_scale_sidecar_bytes": 0,
        },
        "attempt": {
            "directory": ATTEMPT_DIR.relative_to(ROOT).as_posix(),
            "execution_marker": MARKER_PATH.relative_to(ROOT).as_posix(),
            "matrix": artifacts["official_matrix"],
            "prompt_count": 9,
            "generation": "deterministic greedy argmax; no sampling; use_cache=false; maximum six generated tokens; termination IDs 151643 and 151645",
            "score_policy": "binary only: 9/9 complete token arrays exactly equal BF16 and 9/9 response-gate outcomes equal BF16",
            "exact_command": ["./.venv/bin/python", RUNNER_PATH.relative_to(ROOT).as_posix(), "execute"],
            "replay_resume_replacement_tuning_or_gate_relaxation_permitted": False,
            "network_access_permitted": False,
        },
        "environment": environment,
        "namespace_audit": namespace_audit,
        "artifacts": artifacts,
        "expected_evidence": [
            "execution_started.json",
            "run.log",
            "results.json or failure.json",
            "terminal_status.json",
            "fresh_reviewer_input.json when numerical evaluation completes",
            "SHA256SUMS",
            "SHA256SUMS.sha256",
        ],
        "claim_boundary": {
            "selected_policy_id": None,
            "product_policy_accepted": False,
            "rtl_capability_accepted": False,
            "ppa_or_timing_claimed": False,
            "demo_u280_or_stage2_authorized": False,
            "current_stage_remains": "specification",
        },
    }
    return {"contract": contract, "contract_sha256": canonical_sha256(contract)}


def verify_bound_record(record: dict[str, Any]) -> None:
    path = ROOT / record["path"]
    require(path.is_file(), f"bound artifact is missing: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"bound artifact size differs: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"bound artifact hash differs: {record['path']}")


def load_verified_contract(require_unconsumed: bool) -> dict[str, Any]:
    require_project_python()
    require(CONTRACT_PATH.is_file() and READY_PATH.is_file(), "pre-attempt closure is missing")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID, "predeclared contract candidate differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == sha256_file(RUNNER_PATH), "runner changed after freeze")
    for record in contract["artifacts"].values():
        verify_bound_record(record)
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_companion(TABLE_PATH, TABLE_COMPANION)
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    require({**verify_versions(), "numpy": importlib.metadata.version("numpy")} == contract["environment"]["packages"], "package versions differ")
    ready = load_json(READY_PATH)
    require(ready["status"] == "PREATTEMPT_READY", "pre-attempt ready status differs")
    require(ready["candidate_id"] == CANDIDATE_ID, "pre-attempt ready candidate differs")
    require(ready["contract"]["sha256"] == sha256_file(CONTRACT_PATH), "ready contract file hash differs")
    require(ready["contract_sha256"] == wrapper["contract_sha256"], "ready contract canonical hash differs")
    require(ready["bound_artifacts"] == contract["artifacts"], "ready artifact snapshot differs")
    require(ready["official_execution_marker_exists"] is False, "ready record claims an execution marker")
    if require_unconsumed:
        audit_attempt_namespace(expect_root_absent=False)
        require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "exactly-once attempt is consumed")
    return wrapper


def prepare() -> dict[str, Any]:
    environment = require_project_python()
    recovery = recover_nonqualifying_preparation()
    namespace_audit = audit_attempt_namespace(expect_root_absent=recovery["fresh_namespace"])
    namespace_audit["task_start_namespace_absence_enforced"] = recovery["task_start_namespace_absence_enforced"]
    namespace_audit["nonqualifying_preparation_recovery"] = recovery
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = OUTPUT_DIR / ".preflight-write-probe"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, b"preflight\n")
    os.fsync(fd)
    os.close(fd)
    require(probe.read_bytes() == b"preflight\n", "output-path readback differs")
    probe.unlink()
    chronology: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "preparation_started_at_utc": utc_now(),
        "namespace_audit": namespace_audit,
        "preparation_recovery": recovery,
        "output_path_preflight": {"exclusive_create": True, "readback": True, "collision_behavior": "O_EXCL", "execution_marker_created": False},
    }
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    inputs, input_manifest = non_evaluator_inputs(tokenizer)
    self_test = runner_self_test()

    chronology["candidate_a_started_at_utc"] = utc_now()
    candidate_a, manifest_a, witness_a = load_candidate("candidate_a_table_derivation")
    table, coverage = derive_table(candidate_a, inputs, input_manifest, manifest_a)
    chronology["candidate_a_and_table_complete_at_utc"] = utc_now()
    del candidate_a
    gc.collect()

    temp_table = OUTPUT_DIR / ".static_position_class_scale_table.preflight.json"
    write_json(temp_table, table)
    base.TABLE_PATH = temp_table
    coverage["static_scale_table_sha256"] = sha256_file(temp_table)

    chronology["candidate_b_started_at_utc"] = utc_now()
    candidate_b, manifest_b, witness_b = load_candidate("candidate_b_fresh_preflight")
    state_comparison = compare_candidates(manifest_a, manifest_b)
    preflight = preflight_table(candidate_b, table, inputs, input_manifest)
    require(preflight["official_execution_marker_exists"] is False, "execution marker appeared during preflight")
    preflight["candidate_state_comparison"] = state_comparison
    preflight["embedding_hash_matches_source"] = manifest_b["identity"]["embedding_bf16_sha256"] == SOURCE_EMBEDDING_SHA256
    preflight["lm_head_storage_is_separate"] = witness_b["post_split"]["different_storage_pointers"]
    preflight["all_312_layer_event_names_present"] = coverage["all_312_layer_event_names_present"]
    preflight["all_three_position_classes_present_at_every_event"] = coverage["all_three_position_classes_present_at_every_event"]
    chronology["candidate_b_preflight_complete_at_utc"] = utc_now()
    del candidate_b
    gc.collect()

    PREATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    os.replace(temp_table, TABLE_PATH)
    base.TABLE_PATH = TABLE_PATH
    preflight["activation_execution"]["fixed_table"] = file_record(TABLE_PATH)
    write_companion(TABLE_PATH, TABLE_COMPANION)
    write_json(INPUT_MANIFEST_PATH, input_manifest)
    write_json(CANDIDATE_A_PATH, manifest_a)
    write_json(CANDIDATE_B_PATH, manifest_b)
    write_json(ALIAS_WITNESS_PATH, {"schema_version": 1, "candidate_id": CANDIDATE_ID, "candidate_a": witness_a, "candidate_b": witness_b, "state_comparison": state_comparison})
    write_json(COVERAGE_PATH, coverage)
    write_json(PREFLIGHT_PATH, preflight)
    write_json(SELF_TEST_PATH, self_test)
    if recovery["recovered_nonqualifying_preparation"]:
        write_json(REPAIR_PATH, {"schema_version": 1, "candidate_id": CANDIDATE_ID, **recovery})
    chronology["pre_attempt_artifacts_frozen_at_utc"] = utc_now()
    write_json(CHRONOLOGY_PATH, chronology)

    wrapper = build_contract(environment, source_identity, namespace_audit)
    write_json(CONTRACT_PATH, wrapper)
    ready = {
        "schema_version": 1,
        "status": "PREATTEMPT_READY",
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "contract": file_record(CONTRACT_PATH),
        "contract_sha256": wrapper["contract_sha256"],
        "bound_artifacts": wrapper["contract"]["artifacts"],
        "preflight": file_record(PREFLIGHT_PATH),
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "verification": "all bound source, configuration, input, table, candidate-state, runner, and tool hashes verified",
    }
    write_json(READY_PATH, ready)
    load_verified_contract(require_unconsumed=True)
    for path in list(PREATTEMPT_DIR.iterdir()) + [CONTRACT_PATH, READY_PATH]:
        if path.is_file():
            path.chmod(0o444)
    return {
        "status": "PREATTEMPT_READY",
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "table_sha256": sha256_file(TABLE_PATH),
        "candidate_state_manifest_sha256": manifest_b["identity"]["candidate_state_manifest_sha256"],
        "fixed_table_overflow_count": 0,
        "attempts_consumed": 0,
    }


def write_checksum_bundle(paths: list[Path]) -> None:
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    companion.write_text(
        f"{sha256_file(sums_path)}  {sums_path.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )


def seal_attempt(paths: list[Path]) -> None:
    for path in paths + [ATTEMPT_DIR / "SHA256SUMS", ATTEMPT_DIR / "SHA256SUMS.sha256"]:
        if path.is_file():
            path.chmod(0o444)
    ATTEMPT_DIR.chmod(0o555)


def execute_once() -> int:
    wrapper = load_verified_contract(require_unconsumed=True)
    ATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    marker = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "attempt_id": "attempt-0001",
        "started_at_utc": utc_now(),
        "contract": file_record(CONTRACT_PATH),
        "contract_sha256": wrapper["contract_sha256"],
        "preattempt_ready": file_record(READY_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "matrix_sha256": MATRIX_SHA256,
        "static_scale_table_sha256": sha256_file(TABLE_PATH),
        "selected_policy_id_before_execution": None,
        "authorization_consumed": True,
    }
    atomic_write_json(MARKER_PATH, marker)
    started = time.monotonic()
    run_log_path = ATTEMPT_DIR / "run.log"
    candidate_prompt_count = 0
    result_path: Path | None = None
    reviewer_path: Path | None = None
    failure_path: Path | None = None
    return_code = 5
    terminal_class = "EVALUATOR_NO_EXECUTION"
    policy: base.StaticActivationPolicy | None = None
    run_log = run_log_path.open("w", encoding="utf-8")
    try:
        log_message(run_log, f"official_attempt_started candidate={CANDIDATE_ID} attempt=attempt-0001")
        torch.set_num_threads(TORCH_THREADS)
        torch.set_num_interop_threads(1)
        source_contract = verify_source_contract()
        source_identity = verify_source_snapshot()
        versions = verify_versions()
        matrix = load_json(MATRIX_PATH)
        prompt_specs = {record["case_id"]: record for record in prompt_binding(matrix)}
        from discriminate_qwen_instruct_w4_weight_policies import PROMPTS, chat_input

        require(len(PROMPTS) == 9, "official prompt count differs")
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
        inputs = {prompt.case_id: chat_input(tokenizer, prompt.prompt, DEFAULT_SYSTEM) for prompt in PROMPTS}
        reference_model = AutoModelForCausalLM.from_pretrained(
            SNAPSHOT,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()
        references: dict[str, dict[str, Any]] = {}
        log_message(run_log, "bf16_reference_recomputation_started prompts=9")
        for prompt in PROMPTS:
            reference = generate_reference(reference_model, tokenizer, inputs[prompt.case_id], prompt.expected)
            frozen = prompt_specs[prompt.case_id]
            require(reference["generated_token_ids"] == frozen["bf16_generated_token_ids"], f"BF16 tokens differ: {prompt.case_id}")
            require(reference["response_gate"]["status"] == frozen["bf16_response_gate_status"], f"BF16 response gate differs: {prompt.case_id}")
            references[prompt.case_id] = reference
            log_message(run_log, f"bf16_prompt_complete case={prompt.case_id}")
        del reference_model
        gc.collect()

        candidate_model, official_manifest, official_witness = load_candidate("official_attempt_candidate")
        preflight_manifest = load_json(CANDIDATE_B_PATH)
        comparison = compare_candidates(preflight_manifest, official_manifest)
        require(comparison["all_match"], "official candidate state differs from preflight")
        table = load_json(TABLE_PATH)
        policy = base.StaticActivationPolicy(table)
        policy.install(candidate_model)
        prompt_results: list[dict[str, Any]] = []
        model_forwards = 0
        for prompt in PROMPTS:
            reference = references[prompt.case_id]
            candidate, forwards = base.generate_candidate(
                candidate_model,
                tokenizer,
                inputs[prompt.case_id],
                prompt.expected,
                reference,
                policy,
            )
            model_forwards += forwards
            candidate_prompt_count += 1
            token_disagreements = sequence_disagreements(reference["generated_token_ids"], candidate["generated_token_ids"])
            gate_match = candidate["response_gate"]["status"] == reference["response_gate"]["status"]
            public_reference = dict(reference)
            public_reference.pop("_logits")
            prompt_results.append(
                {
                    "case_id": prompt.case_id,
                    "prompt_sha256": prompt_specs[prompt.case_id]["prompt_sha256"],
                    "expected_sha256": prompt_specs[prompt.case_id]["expected_sha256"],
                    "bf16": public_reference,
                    "candidate": candidate,
                    "token_disagreements": token_disagreements,
                    "response_gate_outcome_match": gate_match,
                }
            )
            log_message(run_log, f"candidate_prompt_complete case={prompt.case_id} exact={candidate['exact_bf16_sequence_match']} gate_match={gate_match}")
        activation_summary = policy.summary(model_forwards)
        policy.uninstall()
        policy = None
        aggregate = aggregate_prompts(prompt_results)
        disagreement_set = [
            {
                "case_id": item["case_id"],
                "token_disagreements": item["token_disagreements"],
                "bf16_response_gate_status": item["bf16"]["response_gate"]["status"],
                "candidate_response_gate_status": item["candidate"]["response_gate"]["status"],
                "response_gate_outcome_match": item["response_gate_outcome_match"],
            }
            for item in prompt_results
            if item["token_disagreements"] or not item["response_gate_outcome_match"]
        ]
        eligible = aggregate["exact_bf16_sequence_match_count"] == 9 and aggregate["response_gate_outcome_match_count"] == 9
        numerical_status = "PASS_ELIGIBLE_FOR_FRESH_REVIEW" if eligible else "BLOCKED_EXACT_BF16_DISAGREEMENT"
        result = {
            "schema_version": 1,
            "classification": "qwen_instruct_stage1_option_b_alias_safe_post_w4_static_position_class_single_candidate",
            "status": numerical_status,
            "mission_id": MISSION_ID,
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "attempt-0001",
            "contract": file_record(CONTRACT_PATH),
            "preattempt_ready": file_record(READY_PATH),
            "matrix": file_record(MATRIX_PATH),
            "static_scale_table": file_record(TABLE_PATH),
            "source_model": source_identity,
            "source_contract_status": source_contract["status"],
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": {**versions, "numpy": importlib.metadata.version("numpy")}, "torch_num_threads": torch.get_num_threads()},
            "official_candidate_manifest": official_manifest,
            "official_alias_witness": official_witness,
            "preflight_state_match": comparison,
            "activation_execution": activation_summary,
            "prompts": prompt_results,
            "aggregate": aggregate,
            "disagreement_set": disagreement_set,
            "eligibility": {"rule": wrapper["contract"]["attempt"]["score_policy"], "eligible": eligible, "eligible_candidate_id": CANDIDATE_ID if eligible else None, "selected_policy_id": None, "fresh_reviewer_required": True},
            "scope_guards": {"candidate_attempt_count": 1, "alternate_policy_executed": False, "official_inputs_used_before_marker": False, "runtime_dynamic_calibration_executed": False, "per_token_scale_sidecars_emitted": False, "runner_changed_after_freeze": False, "rtl_mutated": False, "ppa_executed": False, "network_access_performed": False},
            "claim_boundary": "Numerical-policy attempt only; no RTL, latency, timing, PPA, demo, U280, product-selection, or stage-advance claim.",
            "timing": {"completed_at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started},
        }
        result_path = ATTEMPT_DIR / "results.json"
        write_json(result_path, result)
        reviewer = {
            "schema_version": 1,
            "status": "READY_FOR_FRESH_REVIEW",
            "mission_id": MISSION_ID,
            "candidate_id": CANDIDATE_ID,
            "numerical_result": numerical_status,
            "contract": file_record(CONTRACT_PATH),
            "preattempt_ready": file_record(READY_PATH),
            "results": file_record(result_path),
            "selected_policy_id_before_adjudication": None,
            "independent_acceptance_required": True,
            "requested_review": ["pre-attempt chronology and hash closure", "source-tied then storage-separated lm_head policy", "post-W4 candidate A/B state equality", "zero-overflow non-evaluator preflight before marker", "exactly one consumed attempt", "nine exact BF16 token arrays and nine BF16 response-gate outcome comparisons"],
            "verification_command": ["./.venv/bin/python", RUNNER_PATH.relative_to(ROOT).as_posix(), "verify-result"],
            "official_attempt_regeneration_forbidden": True,
        }
        reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
        write_json(reviewer_path, reviewer)
        terminal_class = "NUMERICAL_SUCCESS" if eligible else "NUMERICAL_FAILURE"
        return_code = 0 if eligible else 4
        log_message(run_log, f"official_attempt_complete status={numerical_status} exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 gate_matches={aggregate['response_gate_outcome_match_count']}/9 selected_policy_id=null")
    except Exception as exc:
        if policy is not None:
            policy.uninstall()
        terminal_class = "EVALUATOR_NO_EXECUTION" if candidate_prompt_count == 0 else "EVALUATOR_PARTIAL_EXECUTION_FAILURE"
        failure = {
            "schema_version": 1,
            "status": "BLOCKED_" + terminal_class,
            "classification": terminal_class,
            "mission_id": MISSION_ID,
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "attempt-0001",
            "candidate_prompt_results_completed": candidate_prompt_count,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "numerical_correctness_conclusion": None,
            "selected_policy_id": None,
            "authorization_consumed": True,
            "failed_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        }
        failure_path = ATTEMPT_DIR / "failure.json"
        write_json(failure_path, failure)
        log_message(run_log, f"official_attempt_failed classification={terminal_class} error_type={type(exc).__name__} error={exc}")
        return_code = 5
    finally:
        run_log.flush()
        os.fsync(run_log.fileno())
        run_log.close()

    terminal_path = ATTEMPT_DIR / "terminal_status.json"
    write_json(
        terminal_path,
        {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "attempt-0001",
            "classification": terminal_class,
            "return_code": return_code,
            "candidate_prompt_results_completed": candidate_prompt_count,
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    evidence = [CONTRACT_PATH, READY_PATH, MARKER_PATH, run_log_path, terminal_path]
    if result_path is not None:
        evidence.append(result_path)
    if reviewer_path is not None:
        evidence.append(reviewer_path)
    if failure_path is not None:
        evidence.append(failure_path)
    write_checksum_bundle(evidence)
    seal_attempt([MARKER_PATH, run_log_path, terminal_path] + [path for path in (result_path, reviewer_path, failure_path) if path is not None])
    return return_code


def parse_sums(path: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        records.append((digest, ROOT / relative))
    return records


def verify_result() -> dict[str, Any]:
    wrapper = load_verified_contract(require_unconsumed=False)
    attempts = sorted(path.name for path in OUTPUT_DIR.glob("attempt-*") if path.is_dir())
    require(attempts == ["attempt-0001"], "official attempt set differs")
    marker = load_json(MARKER_PATH)
    require(marker["authorization_consumed"] is True, "attempt authorization was not consumed")
    require(marker["runner_sha256"] == sha256_file(RUNNER_PATH), "executed runner hash differs")
    require(marker["contract_sha256"] == wrapper["contract_sha256"], "executed contract hash differs")
    sums = ATTEMPT_DIR / "SHA256SUMS"
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    verify_companion(sums, companion)
    for digest, path in parse_sums(sums):
        require(path.is_file() and sha256_file(path) == digest, f"official evidence hash differs: {path}")
    terminal = load_json(ATTEMPT_DIR / "terminal_status.json")
    failure_path = ATTEMPT_DIR / "failure.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        require(failure["numerical_correctness_conclusion"] is None, "execution failure drew a numerical conclusion")
        require(failure["selected_policy_id"] is None, "execution failure selected a policy")
        return {
            "status": failure["status"],
            "classification": failure["classification"],
            "candidate_id": CANDIDATE_ID,
            "attempt_count": 1,
            "candidate_prompt_results_completed": failure["candidate_prompt_results_completed"],
            "eligible": None,
            "selected_policy_id": None,
            "failure_sha256": sha256_file(failure_path),
            "terminal_return_code": terminal["return_code"],
        }
    result = load_json(ATTEMPT_DIR / "results.json")
    require(result["candidate_id"] == CANDIDATE_ID, "result candidate differs")
    require(len(result["prompts"]) == 9, "result prompt count differs")
    require(result["official_candidate_manifest"]["identity"]["candidate_state_manifest_sha256"] == load_json(CANDIDATE_B_PATH)["identity"]["candidate_state_manifest_sha256"], "official candidate state differs from preflight")
    disagreements: list[dict[str, Any]] = []
    exact_count = 0
    gate_count = 0
    for item in result["prompts"]:
        token_disagreements = sequence_disagreements(item["bf16"]["generated_token_ids"], item["candidate"]["generated_token_ids"])
        require(token_disagreements == item["token_disagreements"], f"token disagreement record differs: {item['case_id']}")
        gate_match = item["bf16"]["response_gate"]["status"] == item["candidate"]["response_gate"]["status"]
        require(gate_match == item["response_gate_outcome_match"], f"response-gate match differs: {item['case_id']}")
        exact_count += int(not token_disagreements)
        gate_count += int(gate_match)
        if token_disagreements or not gate_match:
            disagreements.append(
                {
                    "case_id": item["case_id"],
                    "token_disagreements": token_disagreements,
                    "bf16_response_gate_status": item["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": item["candidate"]["response_gate"]["status"],
                    "response_gate_outcome_match": gate_match,
                }
            )
    require(disagreements == result["disagreement_set"], "result disagreement set differs")
    require(exact_count == result["aggregate"]["exact_bf16_sequence_match_count"], "exact sequence aggregate differs")
    require(gate_count == result["aggregate"]["response_gate_outcome_match_count"], "response-gate aggregate differs")
    eligible = exact_count == 9 and gate_count == 9
    require(result["eligibility"]["eligible"] == eligible, "eligibility differs")
    require(result["eligibility"]["selected_policy_id"] is None, "result selected a policy before review")
    return {
        "status": result["status"],
        "candidate_id": CANDIDATE_ID,
        "attempt_count": 1,
        "eligible": eligible,
        "exact_sequence_matches": exact_count,
        "gate_outcome_matches": gate_count,
        "disagreement_case_ids": [item["case_id"] for item in disagreements],
        "selected_policy_id": None,
        "results_sha256": sha256_file(ATTEMPT_DIR / "results.json"),
        "reviewer_input_sha256": sha256_file(ATTEMPT_DIR / "fresh_reviewer_input.json"),
        "terminal_return_code": terminal["return_code"],
    }


def record_preparation_failure(exc: Exception) -> None:
    if MARKER_PATH.exists():
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failure = OUTPUT_DIR / f"preparation_failure_{time.time_ns()}.json"
    write_json(
        failure,
        {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "classification": "PREATTEMPT_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "traceback": traceback.format_exc(),
            "attempt_authorization_consumed": False,
            "official_execution_marker_exists": False,
            "failed_at_utc": utc_now(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "verify-ready", "execute", "verify-result"))
    args = parser.parse_args()
    if args.command == "self-test":
        print("ACE2_OPTION_B_V2_RUNNER_SELF_TEST_PASS " + json.dumps(runner_self_test(), sort_keys=True))
        return 0
    if args.command == "prepare":
        try:
            result = prepare()
        except Exception as exc:
            record_preparation_failure(exc)
            raise
        print("ACE2_OPTION_B_V2_PREATTEMPT_READY " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "verify-ready":
        wrapper = load_verified_contract(require_unconsumed=True)
        print("ACE2_OPTION_B_V2_READY_VERIFIED " + json.dumps({"candidate_id": CANDIDATE_ID, "attempts_consumed": 0, "contract_sha256": wrapper["contract_sha256"]}, sort_keys=True))
        return 0
    if args.command == "execute":
        return execute_once()
    summary = verify_result()
    print("ACE2_OPTION_B_V2_RESULT_VERIFIED " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
