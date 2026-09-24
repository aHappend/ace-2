#!/usr/bin/env python3
"""Capture only token-0/head-0 V and replay from layer-0 attention value."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_attention_score_token0_reference_batch_replay as prior
from ace2_attention_value_reference import AttentionValueCase, reference_attention_value


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-attention-value-token0-v-reference-batch-replay-v1/mission.json"
)
BASE = prior.base
ROPE_Q = prior.ROPE_Q
HEAD0 = prior.HEAD0
HEAD1 = prior.HEAD1
KV_MERGE = prior.KV_MERGE
KV_REPLAY = prior.KV_REPLAY
TOKEN0_K = prior.CAPTURE
ATTN_MERGE = prior.MERGE
ATTN_REPLAY = prior.REPLAY
PPA_SUMS = prior.PPA_SUMS

CAPTURE = ROOT / "evidence/diagnostics/c4-layer0-attention-value-token0-v-minimal-reference-capture-20260803-v1"
MERGE = ROOT / "evidence/diagnostics/c4-layer0-attention-value-inputs-merged-20260803-v1"
REPLAY = ROOT / "evidence/verification/rtl-attention-value-token0-reference-batch-replay-v1"

EXPECTED_RTL_TREE = "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be"
EXPECTED_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_MISSION = "bc59e6bf42f05a566a562a6bde125f9ea0e8db6098fadf11152c0fea2b661f7a"
EXPECTED_PRIOR_RUNNER = "76ab7ca2270ffee5049253dbe89a31a80c07c173bc77bfdd10d1a08984fc296e"
EXPECTED_FIXED_MODEL = "b5f1c1800560894a728f199d36b31ef9d624b5afa9d56ee544b0d467270c1c74"
EXPECTED_ATTN_REFERENCE = "c6260ab46e54fc0045253fb564566c9e1852a8806e477f7e9b9aaf6278c74b2e"
EXPECTED_SHELL_TB = "cefd1e4533f9e6de9f85583c0095f3eb2e81329c190f0e0044d045e31da76d18"
EXPECTED_SHELL = "3bb8caab4f06e6be9b170b5b3d91cb89b237715132e52507cd60f0514c61ab30"
EXPECTED_TOKEN0_K_SUMS = "801d777b09a0d14500fca14bf8e03ae4e9bb5eede9b2a034021b91c671f2bdb3"
EXPECTED_ATTN_MERGE_SUMS = "088e478cb83994d48fe8a365649c6e0502c658da56d5d8090dd6972bdf226a73"
EXPECTED_ATTN_REPLAY_SUMS = "22a3addf99fd8afe757e62c303add0dbc5a97726a675550d8870aafcaf298604"
EXPECTED_ATTN_REPORT = "1c7fda411e29449b1596e38e0d1cf44f94b38c220237e1e16db48cb26f38d115"

DATASET = "c4_en_512"
RECORD = 64
LAYER = 0
QUERY_TOKEN = 1
CAPTURE_TOKEN = 0
HEAD = 0
HEAD_DIM = 64
HIDDEN = 896
CONTEXT = 2
ORDER = BASE.ORDER


class Token0VCaptureComplete(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    prior.require(condition, message)


def artifact(path: Path, public_path: str | None = None) -> dict[str, Any]:
    return prior.artifact(path, public_path)


def accepted_packages() -> dict[str, Any]:
    packages = prior.accepted_packages()
    require(prior.sha256_file(TOKEN0_K / "SHA256SUMS") == EXPECTED_TOKEN0_K_SUMS, "token-0 K package changed")
    require(prior.sha256_file(ATTN_MERGE / "SHA256SUMS") == EXPECTED_ATTN_MERGE_SUMS, "attention input merge changed")
    require(prior.sha256_file(ATTN_REPLAY / "SHA256SUMS") == EXPECTED_ATTN_REPLAY_SUMS, "attention replay package changed")
    require(prior.sha256_file(ATTN_REPLAY / "ordered_prefix_report.json") == EXPECTED_ATTN_REPORT, "accepted attention replay report changed")
    member_counts = {
        "token0_k": prior.validate_manifest(TOKEN0_K / "SHA256SUMS", TOKEN0_K),
        "attention_merge": prior.validate_manifest(ATTN_MERGE / "SHA256SUMS", ATTN_MERGE),
        "attention_replay": prior.validate_manifest(ATTN_REPLAY / "SHA256SUMS", ATTN_REPLAY),
    }
    report = json.loads((ATTN_REPLAY / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    merge = json.loads((ATTN_MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    require(report["rtl_tree"]["sha256"] == EXPECTED_RTL_TREE, "accepted attention replay RTL changed")
    require(report["maximal_passing_prefix"] == ORDER[:9], "accepted attention prefix changed")
    require(report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.attention_value", "accepted attention boundary changed")
    require(merge["source_packages_modified"] is False, "accepted attention merge modified source packages")
    packages.update({"attention_merge": merge, "attention_report": report, "new_member_counts": member_counts})
    return packages


def qualifying_capture_matches() -> list[str]:
    matches: list[str] = []
    for path in sorted((ROOT / "evidence/diagnostics").glob("*/witness.json")):
        if CAPTURE in path.parents:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "v_cache_head0_token0_s8" in text:
            matches.append(prior.relative(path))
    return matches


def preflight_payload(source_archive: Path) -> dict[str, Any]:
    require(BASE.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed")
    require(prior.sha256_file(MISSION) == EXPECTED_MISSION, "mission contract changed")
    require(prior.sha256_file(Path(prior.__file__)) == EXPECTED_PRIOR_RUNNER, "accepted attention runner changed")
    require(prior.sha256_file(ROOT / "tools/ace2_full_model_fixed_point.py") == EXPECTED_FIXED_MODEL, "fixed model source changed")
    require(prior.sha256_file(ROOT / "tools/ace2_attention_value_reference.py") == EXPECTED_ATTN_REFERENCE, "attention-value reference changed")
    require(prior.sha256_file(ROOT / "verification/tb/ace2_shell_tb.sv") == EXPECTED_SHELL_TB, "shell harness changed")
    require(prior.sha256_file(ROOT / "rtl/ace2_shell.sv") == EXPECTED_SHELL, "shell RTL changed")
    require(prior.sha256_file(PPA_SUMS) == prior.EXPECTED_PPA_SUMS, "canonical PPA aggregate changed")
    ppa_members = prior.validate_manifest(PPA_SUMS, ROOT)
    packages = accepted_packages()
    matches = qualifying_capture_matches()
    require(not matches, f"a qualifying token-0 V capture already exists: {matches}")
    live_binding, _calibration, _evaluation, manifest, versions = BASE.live_input_binding()
    require(live_binding == packages["head0"]["input_binding"], "live C4 input binding changed")
    model_spec = manifest["model"]
    require(model_spec["repository"] == "Qwen/Qwen2.5-0.5B", "model repository changed")
    require(model_spec["revision"] == EXPECTED_REVISION, "model revision changed")
    cached_files = BASE.cached_model_bindings(model_spec["repository"], model_spec["revision"])
    accepted_cached = json.loads((HEAD0 / "preflight.json").read_text(encoding="utf-8"))["model"]["cached_files"]
    require(cached_files == accepted_cached, "cached model/config/tokenizer hashes changed")
    tools = {name: shutil.which(name) for name in ("iverilog", "vvp")}
    require(all(tools.values()), f"required RTL tools unavailable: {tools}")
    return {
        "schema_version": 1,
        "mission_id": "rtl-attention-value-token0-v-reference-batch-replay-v1",
        "status": "PASS_MODEL_EXECUTION_NOT_CONSUMED",
        "completed_at_utc": prior.utc_now(),
        "model_execution_count": 0,
        "coordinate": {
            "capture_token": CAPTURE_TOKEN,
            "dataset": DATASET,
            "dataset_record_index": RECORD,
            "head": HEAD,
            "layer": LAYER,
            "query_token": QUERY_TOKEN,
        },
        "input_binding": live_binding,
        "model": {
            "repository": model_spec["repository"],
            "revision": model_spec["revision"],
            "cached_files": cached_files,
        },
        "retained_packages": {
            "accepted_rope_q": artifact(ROPE_Q / "SHA256SUMS"),
            "immutable_head0": artifact(HEAD0 / "SHA256SUMS"),
            "immutable_head1": artifact(HEAD1 / "SHA256SUMS"),
            "merged_two_head_kv_write": artifact(KV_MERGE / "SHA256SUMS"),
            "accepted_kv_write_replay": artifact(KV_REPLAY / "SHA256SUMS"),
            "minimal_token0_k": artifact(TOKEN0_K / "SHA256SUMS"),
            "merged_attention_score_inputs": artifact(ATTN_MERGE / "SHA256SUMS"),
            "accepted_attention_score_softmax_replay": artifact(ATTN_REPLAY / "SHA256SUMS"),
            "member_counts": {**packages["member_counts"], **packages["new_member_counts"]},
        },
        "absence_of_existing_qualifying_capture": {
            "match_count": 0,
            "matches": [],
            "search": "evidence/diagnostics/*/witness.json v_cache_head0_token0_s8",
        },
        "output_path_semantics": {
            "capture": prior.relative(CAPTURE),
            "merge": prior.relative(MERGE),
            "replay": prior.relative(REPLAY),
            "raw_stdout": prior.relative(CAPTURE / "capture.stdout"),
            "raw_stderr": prior.relative(CAPTURE / "capture.stderr"),
            "all_absent_before_preflight": True,
            "create_modes": "exclusive mkdir/open; no overwrite",
        },
        "capture_scope": {
            "persisted_tensor_names": ["v_cache_head0_token0_s8"],
            "persisted_tensor_count": 1,
            "metadata_only": [
                "v_projection_output_scale",
                "value_cache_format",
                "attention_value_probability_format",
                "attention_value_rounding",
            ],
            "rope_q_recapture": False,
            "k_recapture": False,
            "token1_recapture": False,
            "head1_recapture": False,
        },
        "mission": prior.external_artifact(MISSION, "handoff:rtl-attention-value-token0-v-reference-batch-replay-v1/mission.json"),
        "protected_state": prior.protected_state(),
        "runtime": {
            "packages": versions,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "rtl_tree": BASE.rtl_tree(),
        "canonical_ppa": {
            "reused_not_rerun": True,
            "member_count": ppa_members,
            "sha256_manifest": artifact(PPA_SUMS),
        },
        "source": artifact(SELF),
        "source_archive": artifact(source_archive),
        "tool_lookup": {name: Path(value).name for name, value in tools.items()},
        "forbidden_flows_run": [],
    }


def run_preflight() -> None:
    require(not CAPTURE.exists(), f"capture path already exists: {CAPTURE}")
    require(not MERGE.exists(), f"merge path already exists: {MERGE}")
    require(not REPLAY.exists(), f"replay path already exists: {REPLAY}")
    source_hash = prior.sha256_file(SELF)
    payload = preflight_payload(SELF)
    require(payload["source"]["sha256"] == source_hash, "source changed during preflight")
    CAPTURE.mkdir(parents=True, exist_ok=False)
    source_archive = CAPTURE / "preflight_source_at_execution.py"
    shutil.copy2(SELF, source_archive)
    payload["source_archive"] = artifact(source_archive)
    require(payload["source_archive"]["sha256"] == payload["source"]["sha256"], "preflight source archive differs")
    prior.write_json(CAPTURE / "preflight.json", payload)
    (CAPTURE / "preflight.log").write_text(
        "ACE2_ATTN_VALUE_TOKEN0_V_PREFLIGHT_PASS\n"
        "model_execution_count=0\n"
        "coordinate=c4_en_512:record64:layer0:query1:value0:head0\n"
        "persisted_tensor_count=1\n"
        "existing_qualifying_capture_count=0\n",
        encoding="utf-8",
    )
    print("ACE2_ATTN_VALUE_TOKEN0_V_PREFLIGHT_PASS model_execution_count=0")


def validate_preflight() -> dict[str, Any]:
    path = CAPTURE / "preflight.json"
    require(path.is_file(), "preflight artifact is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["status"] == "PASS_MODEL_EXECUTION_NOT_CONSUMED", "preflight status changed")
    require(payload["model_execution_count"] == 0, "preflight consumed model execution")
    require(prior.sha256_file(SELF) == payload["source"]["sha256"], "runner source changed after preflight")
    require(artifact(CAPTURE / "preflight_source_at_execution.py") == payload["source_archive"], "preflight source archive changed")
    require(BASE.rtl_tree()["sha256"] == payload["rtl_tree"]["sha256"], "RTL changed after preflight")
    require(prior.protected_state() == payload["protected_state"], "protected state changed after preflight")
    accepted_packages()
    return payload


class MinimalVTrace:
    def __init__(self, layer: Any, token1_v: Tensor) -> None:
        self.layer = layer
        self.token1_v = token1_v
        self.v_token0: Tensor | None = None
        self.v_token1_match = False
        self.call_count = 0
        self._original: Callable[..., Any] | None = None

    def install(self) -> None:
        module = self.layer.self_attn.v_proj
        original = module.forward_hardware_input
        self._original = original

        def wrapper(_self: nn.Module, *args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.call_count += 1
            require(isinstance(result, Tensor), "V projection result is not a tensor")
            self.v_token0 = result[0, CAPTURE_TOKEN, :HEAD_DIM].detach().cpu().to(torch.int8).clone()
            live_token1 = result[0, QUERY_TOKEN, :HEAD_DIM].detach().cpu().to(torch.int8)
            self.v_token1_match = torch.equal(live_token1, self.token1_v)
            raise Token0VCaptureComplete

        module.__dict__["forward_hardware_input"] = types.MethodType(wrapper, module)

    def restore(self) -> None:
        if self._original is not None:
            self.layer.self_attn.v_proj.__dict__.pop("forward_hardware_input", None)


def execute_capture_worker() -> None:
    preflight = validate_preflight()
    marker = CAPTURE / "capture_execution_consumed.json"
    source_archive = CAPTURE / "capture_source_at_execution.py"
    require(marker.is_file(), "capture consumption marker is missing")
    require(source_archive.is_file(), "capture source archive is missing")
    require(
        prior.sha256_file(SELF) == preflight["source"]["sha256"],
        "live capture source differs from preflight; rerun prohibited",
    )
    require(prior.sha256_file(source_archive) == preflight["source"]["sha256"], "capture source differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}, "capture requires CPU-only visibility")
    packages = accepted_packages()
    live_binding, calibration_prompt, evaluation_prompt, manifest, versions = BASE.live_input_binding()
    require(live_binding == preflight["input_binding"], "live input changed after preflight")
    config = BASE.load_contracts(require_rtl_binding=False)[1]
    BASE.seed_everything(config)
    model_spec = manifest["model"]
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "model is not CPU-only")
    require(getattr(model.config, "_commit_hash", None) == EXPECTED_REVISION, "resolved model revision changed")
    ranges, operator_ranges = BASE.calibrate(model, [calibration_prompt])
    BASE.replace_linears(model, ranges, rope_diagnostic_mechanism=None)
    BASE.replace_fixed_operators(model, operator_ranges, rope_diagnostic_mechanism=None)
    layer = model.model.layers[LAYER]
    require(isinstance(layer, BASE.FixedDecoderLayer), "layer 0 is not FixedDecoderLayer")
    require(isinstance(layer.self_attn, BASE.FixedAttention), "layer 0 attention is not fixed")
    token1_v = BASE.read_tensor(packages["head0"]["tensor_manifest"]["kv_write_v_head0_token1_s8"]).to(torch.int8)
    trace = MinimalVTrace(layer, token1_v)
    trace.install()
    evaluation_forward_count = 0
    BASE.seed_everything(config)
    try:
        with torch.inference_mode():
            evaluation_forward_count += 1
            model(input_ids=evaluation_prompt, use_cache=False)
    except Token0VCaptureComplete:
        pass
    finally:
        trace.restore()
    require(evaluation_forward_count == 1, "evaluation capture forward count changed")
    require(trace.call_count == 1, "V projection call count changed")
    require(trace.v_token0 is not None, "token-0 V cache payload was not captured")
    require(trace.v_token1_match, "live token-1 V differs from immutable head 0")
    v_scale = float(layer.self_attn.v_proj.output_scale)
    require(v_scale == float(packages["head0"]["operator_metadata"]["v_projection_output_scale"]), "V projection output scale changed")
    tensors = {
        "v_cache_head0_token0_s8": BASE.write_tensor(CAPTURE, "v_cache_head0_token0_s8", trace.v_token0),
    }
    require(set(tensors) == {"v_cache_head0_token0_s8"}, "capture tensor scope expanded")
    witness = {
        "schema_version": 1,
        "classification": "single_forward_extraction_only_token0_head0_v_for_attention_value",
        "capture": {
            "calibration_dependency_forward_count": 1,
            "completed_at_utc": prior.utc_now(),
            "consumption_marker": artifact(marker),
            "evaluation_forward_count": evaluation_forward_count,
            "persisted_tensor_count": 1,
            "record_count_retained": 1,
            "layer_count_retained": 1,
            "head_count_retained": 1,
            "token_count_retained": 1,
            "rope_q_recaptured": False,
            "k_recaptured": False,
            "token1_recaptured": False,
            "head1_recaptured": False,
        },
        "coordinate": {
            "dataset": DATASET,
            "dataset_record_index": RECORD,
            "head": HEAD,
            "layer": LAYER,
            "token": CAPTURE_TOKEN,
            "query_token": QUERY_TOKEN,
            "token_id": int(evaluation_prompt[0, CAPTURE_TOKEN]),
        },
        "input_binding": live_binding,
        "model": {
            "repository": model_spec["repository"],
            "revision": model_spec["revision"],
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "dtype": "bfloat16_source_with_fixed_point_extraction",
        },
        "attention_value_metadata": {
            "query_head": HEAD,
            "mapped_kv_head": 0,
            "active_tokens": [0, 1],
            "v_projection_output_scale": v_scale,
            "value_cache_format": "signed_int8_head_dim_64",
            "probability_format": "unsigned_int16_Q0.15",
            "accumulator_format": "signed_int32",
            "rounding": "round_to_nearest_ties_to_even_after_15_fractional_bits",
            "output_format": "signed_int8",
        },
        "compatibility": {
            "immutable_token1_v_exact_match": True,
            "live_input_binding_exact_match": True,
            "retained_attention_score_softmax_reused_not_recaptured": True,
        },
        "tensor_manifest": tensors,
        "source_bindings": {
            "capture_tool": artifact(source_archive),
            "fixed_model": artifact(ROOT / "tools/ace2_full_model_fixed_point.py"),
            "attention_value_reference": artifact(ROOT / "tools/ace2_attention_value_reference.py"),
            "accepted_attention_score_softmax": artifact(ATTN_REPLAY / "SHA256SUMS"),
            "immutable_head0": artifact(HEAD0 / "SHA256SUMS"),
            "rtl_tree": BASE.rtl_tree(),
        },
        "runtime": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": versions,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "scope_exclusions": [
            "rope_q_payload",
            "k_projection_or_k_cache_payload",
            "token1_payload",
            "head1_payload",
            "attention_score_or_softmax_payload",
            "o_proj_or_later_reference_capture",
            "other_records_heads_layers_tokens",
            "rtl_mutation",
            "synthesis_opensta_ppa",
            "baseline_candidate_benchmark_scale32",
            "frontier_authority_seal_mutation",
        ],
    }
    prior.write_json(CAPTURE / "witness.json", witness)
    prior.write_json(
        CAPTURE / "capture_contract.json",
        {
            "schema_version": 1,
            "status": "CAPTURE_COMPLETE_MERGE_REPLAY_PENDING",
            "completed_at_utc": prior.utc_now(),
            "coordinate": witness["coordinate"],
            "evaluation_forward_count": 1,
            "persisted_tensor_count": 1,
            "source": artifact(source_archive),
            "witness": artifact(CAPTURE / "witness.json"),
        },
    )
    (CAPTURE / "capture.log").write_text(
        "ACE2_ATTN_VALUE_TOKEN0_V_CAPTURE_COMPLETE\n"
        "coordinate=c4_en_512:record64:layer0:query1:value0:head0\n"
        "evaluation_forward_count=1\n"
        "persisted_tensor_count=1\n"
        "rope_q_recaptured=false\n"
        "k_recaptured=false\n"
        "token1_recaptured=false\n"
        "head1_recaptured=false\n"
        "rtl_mutated=false\n",
        encoding="utf-8",
    )
    print(
        "ACE2_ATTN_VALUE_TOKEN0_V_CAPTURE_COMPLETE "
        f"tensors=1 evaluation_forwards=1 witness_sha256={prior.sha256_file(CAPTURE / 'witness.json')}"
    )


def merge_packages() -> None:
    require(not MERGE.exists(), "attention-value merge output already exists")
    packages = accepted_packages()
    prior.validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE)
    captured = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    head0 = packages["head0"]
    attention_merge = packages["attention_merge"]
    prior_logical = attention_merge["logical_interfaces"]
    logical = {
        "softmax_head0_query_token1_active_q15": prior_logical["softmax_reference_head0_query_token1_active_q15"],
        "v_cache_head0_token0_s8": captured["tensor_manifest"]["v_cache_head0_token0_s8"],
        "v_cache_head0_token1_s8": prior_logical["v_cache_head0_token1_s8"],
        "attention_value_head0_token1_reference_s8": head0["tensor_manifest"]["attention_value_head0_token1_s8"],
        "o_proj_input_token1_s8": head0["tensor_manifest"]["o_proj_input_token1_s8"],
        "o_proj_output_token1_s8": head0["tensor_manifest"]["o_proj_output_token1_s8"],
    }
    MERGE.mkdir(parents=True, exist_ok=False)
    prior.write_json(
        MERGE / "merge_manifest.json",
        {
            "schema_version": 1,
            "mission_id": "rtl-attention-value-token0-v-reference-batch-replay-v1",
            "classification": "derived_hash_bound_attention_value_merge_no_source_package_modification",
            "coordinate": {
                "dataset": DATASET,
                "dataset_record_index": RECORD,
                "layer": LAYER,
                "query_token": QUERY_TOKEN,
                "active_tokens": [0, 1],
                "query_head": HEAD,
                "mapped_kv_head": 0,
            },
            "source_packages": {
                "accepted_rope_q": artifact(ROPE_Q / "SHA256SUMS"),
                "immutable_head0": artifact(HEAD0 / "SHA256SUMS"),
                "immutable_head1": artifact(HEAD1 / "SHA256SUMS"),
                "merged_two_head_kv_write": artifact(KV_MERGE / "SHA256SUMS"),
                "accepted_kv_write_replay": artifact(KV_REPLAY / "SHA256SUMS"),
                "minimal_token0_k": artifact(TOKEN0_K / "SHA256SUMS"),
                "merged_attention_score_inputs": artifact(ATTN_MERGE / "SHA256SUMS"),
                "accepted_attention_score_softmax_replay": artifact(ATTN_REPLAY / "SHA256SUMS"),
                "new_minimal_token0_v": artifact(CAPTURE / "SHA256SUMS"),
            },
            "logical_interfaces": logical,
            "attention_value_metadata": captured["attention_value_metadata"],
            "compatibility": {
                "dataset_record_layer_exact_match": True,
                "model_revision_exact_match": True,
                "live_token1_v_exact_match": True,
                "softmax_source_hash_exact_match": True,
                "retained_packages_unmodified": True,
                "no_recapture_of_rope_q_k_token1_or_head1": True,
            },
            "counts": {
                "active_probability_bytes": 4,
                "active_v_bytes": 128,
                "attention_value_reference_bytes": 64,
                "o_proj_input_bytes": HIDDEN,
                "o_proj_output_reference_bytes": HIDDEN,
            },
            "source_packages_modified": False,
        },
    )
    prior.write_sha256s(MERGE)


def render_attention_value_vectors(
    probabilities: list[int], values: list[list[int]], expected: list[int], saturation: bool
) -> str:
    padded_probabilities = probabilities + [0] * (8 - len(probabilities))
    padded_values = values + [[0] * HEAD_DIM for _ in range(8 - len(values))]
    lines = [
        "// Generated for the immutable C4 record64 layer0 attention-value replay.",
        "localparam integer ATTN_VALUE_CASE_COUNT = 1;",
        "localparam integer ATTN_VALUE_CONTEXT_MAX = 8;",
        "localparam integer ATTN_VALUE_HEAD_DIM = 64;",
        "localparam integer ATTN_VALUE_BEATS_PER_VECTOR = 4;",
        "reg [15:0] attn_value_context_count [0:ATTN_VALUE_CASE_COUNT-1];",
        "reg [16*8-1:0] attn_value_probability_word [0:ATTN_VALUE_CASE_COUNT-1];",
        "reg [127:0] attn_value_v_beats [0:ATTN_VALUE_CASE_COUNT*ATTN_VALUE_CONTEXT_MAX*ATTN_VALUE_BEATS_PER_VECTOR-1];",
        "reg [127:0] attn_value_expected_beats [0:ATTN_VALUE_CASE_COUNT*ATTN_VALUE_BEATS_PER_VECTOR-1];",
        "reg attn_value_expected_saturation [0:ATTN_VALUE_CASE_COUNT-1];",
        "initial begin",
        f"  attn_value_context_count[0] = 16'd{len(probabilities)};",
        f"  attn_value_probability_word[0] = {prior.sv_hex(prior.pack(padded_probabilities, 16), 128)};",
    ]
    for token, row in enumerate(padded_values):
        for beat in range(4):
            lines.append(
                f"  attn_value_v_beats[{token * 4 + beat}] = "
                f"{prior.sv_hex(prior.pack(row[beat * 16:(beat + 1) * 16], 8), 128)};"
            )
    for beat in range(4):
        lines.append(
            f"  attn_value_expected_beats[{beat}] = "
            f"{prior.sv_hex(prior.pack(expected[beat * 16:(beat + 1) * 16], 8), 128)};"
        )
    lines.append(f"  attn_value_expected_saturation[0] = 1'b{int(saturation)};")
    lines.append("end")
    return "\n".join(lines) + "\n"


def execution_testbench() -> str:
    text = (ROOT / "verification/tb/ace2_shell_tb.sv").read_text(encoding="utf-8")
    text = text.replace(
        '    `include "../generated/attention_value_vectors.svh"',
        '    `include "c4_attention_value_vectors.svh"',
        1,
    )
    require('`include "c4_attention_value_vectors.svh"' in text, "attention-value include replacement failed")
    text = text.replace(
        "    integer attn_score_only_mode;",
        "    integer attn_score_only_mode;\n    integer c4_attention_value_replay_mode;",
        1,
    )
    text = text.replace(
        '        attn_score_only_mode = $test$plusargs("ATTN_SCORE_ONLY");',
        '        attn_score_only_mode = $test$plusargs("ATTN_SCORE_ONLY");\n'
        '        c4_attention_value_replay_mode = $test$plusargs("C4_ATTENTION_VALUE_REPLAY");',
        1,
    )
    branch = (
        "        if (c4_attention_value_replay_mode) begin\n"
        "            send_attn_value_cmd(0, 0, 16'h3e02);\n"
        "            wait_attn_value_done_and_compare(0, 0, 16'h3e02);\n"
        "            if (failures != 0) begin\n"
        "                $display(\"ACE2_C4_ATTENTION_VALUE_FAIL failures=%0d\", failures);\n"
        "                $fatal(1, \"ACE2_C4_ATTENTION_VALUE_FAIL\");\n"
        "            end\n"
        "            $display(\"ACE2_C4_ATTENTION_VALUE_PASS writes=%0d cycles=%0d saturation=%0d out0=%032x out1=%032x out2=%032x out3=%032x\", observed_count, last_command_cycles, attn_value_expected_saturation[0], observed_output[0], observed_output[1], observed_output[2], observed_output[3]);\n"
        "            $display(\"ACE2_C4_ATTENTION_VALUE_ORDERED_REPLAY_PASS operators=1\");\n"
        "            $finish;\n"
        "        end\n\n"
    )
    needle = "        if (qproj_stride_only_mode) begin"
    require(needle in text, "focused replay insertion point changed")
    text = text.replace(needle, branch + needle, 1)
    return text


def _raw_i8(values: list[int]) -> bytes:
    return bytes(value & 0xFF for value in values)


def _raw_i32(values: list[int]) -> bytes:
    return b"".join((value & 0xFFFFFFFF).to_bytes(4, "little") for value in values)


def run_replay() -> None:
    if REPLAY.exists():
        require(not (REPLAY / "SHA256SUMS").exists(), "sealed replay output already exists")
        require(not (REPLAY / "ordered_prefix_report.json").exists(), "completed replay report already exists")
        retained = [path for path in REPLAY.iterdir() if path.name != "failed_attempts"]
        if retained:
            attempts = REPLAY / "failed_attempts"
            attempts.mkdir(exist_ok=True)
            attempt_index = 1
            while (attempts / f"attempt-{attempt_index:04d}").exists():
                attempt_index += 1
            attempt = attempts / f"attempt-{attempt_index:04d}"
            attempt.mkdir()
            for path in retained:
                shutil.move(str(path), str(attempt / path.name))
    else:
        REPLAY.mkdir(parents=True, exist_ok=False)
    packages = accepted_packages()
    prior.validate_manifest(MERGE / "SHA256SUMS", MERGE)
    merge = json.loads((MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    logical = merge["logical_interfaces"]
    probabilities = prior.tensor_i16(logical["softmax_head0_query_token1_active_q15"])
    value0 = prior.tensor_i8(logical["v_cache_head0_token0_s8"])
    value1 = prior.tensor_i8(logical["v_cache_head0_token1_s8"])
    expected = prior.tensor_i8(logical["attention_value_head0_token1_reference_s8"])
    o_proj_input = prior.tensor_i8(logical["o_proj_input_token1_s8"])
    require(len(probabilities) == CONTEXT, "active probability count changed")
    require(len(value0) == HEAD_DIM and len(value1) == HEAD_DIM, "active V shape changed")
    require(len(expected) == HEAD_DIM, "attention-value reference shape changed")
    require(len(o_proj_input) == HIDDEN, "o_proj input shape changed")
    reference = reference_attention_value(
        AttentionValueCase("c4_record64_layer0_head0_query1", probabilities, [value0, value1])
    )
    require(reference.outputs == expected, "attention-value independent reference differs from retained reference")
    require(reference.outputs == o_proj_input[:HEAD_DIM], "attention-value output is not the head-0 o_proj input slice")
    generated = REPLAY / "generated"
    generated.mkdir()
    (generated / "c4_attention_value_vectors.svh").write_text(
        render_attention_value_vectors(
            probabilities, [value0, value1], reference.outputs, reference.saturation_seen
        ),
        encoding="utf-8",
    )
    shutil.copy2(SELF, REPLAY / "replay_source_at_execution.py")
    tb = REPLAY / "replay_tb_source_at_execution.sv"
    tb.write_text(execution_testbench(), encoding="utf-8")
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    image = ROOT / "build/rtl-attention-value-token0-reference-batch-replay-v1/ordered.vvp"
    image.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [
        "iverilog",
        "-g2012",
        "-Wall",
        "-Irtl",
        "-Irtl/generated",
        "-Iverification/generated",
        "-Iverification/tb",
        f"-I{generated}",
        "-s",
        "ace2_shell_tb",
        "-o",
        str(image),
        *[str(path) for path in rtl_sources],
        str(tb),
    ]
    compiled = subprocess.run(compile_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "iverilog.log").write_text(
        "command=" + " ".join(compile_command) + "\n" + compiled.stdout + compiled.stderr,
        encoding="utf-8",
    )
    require(compiled.returncode == 0, "ordered replay compilation failed")
    replay_command = ["vvp", str(image), "+C4_ATTENTION_VALUE_REPLAY"]
    replayed = subprocess.run(replay_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "replay.stdout").write_text(replayed.stdout, encoding="utf-8")
    (REPLAY / "replay.stderr").write_text(replayed.stderr, encoding="utf-8")
    (REPLAY / "replay.log").write_text(
        "command=" + " ".join(replay_command) + "\n" + replayed.stdout + replayed.stderr,
        encoding="utf-8",
    )
    require(replayed.returncode == 0, "ordered replay execution failed")
    match = re.search(
        r"ACE2_C4_ATTENTION_VALUE_PASS writes=(\d+) cycles=(\d+) saturation=(\d+) "
        r"out0=([0-9a-fA-F]+) out1=([0-9a-fA-F]+) out2=([0-9a-fA-F]+) out3=([0-9a-fA-F]+)",
        replayed.stdout,
    )
    require(match is not None, "attention-value replay pass marker missing")
    require("ACE2_C4_ATTENTION_VALUE_ORDERED_REPLAY_PASS operators=1" in replayed.stdout, "ordered replay completion marker missing")
    require("MISMATCH" not in replayed.stdout and "_FAIL" not in replayed.stdout, "ordered replay mismatch reported")
    require(int(match.group(1)) == 4, "attention-value replay write count changed")
    require(bool(int(match.group(3))) == reference.saturation_seen, "attention-value saturation result changed")
    rtl_raw = b"".join(int(match.group(index), 16).to_bytes(16, "little") for index in range(4, 8))
    require(rtl_raw == _raw_i8(expected), "RTL attention-value bytes differ from retained reference")
    boundary = {
        "schema_version": 1,
        "mission_id": "rtl-attention-value-token0-v-reference-batch-replay-v1",
        "operator": "layer_0.o_proj",
        "classification": "UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_RETAINED_O_PROJ_WEIGHTS_METADATA_INCOMPLETE",
        "available_inputs": {
            "o_proj_input_token1_s8": logical["o_proj_input_token1_s8"],
            "o_proj_output_token1_reference_s8": logical["o_proj_output_token1_s8"],
            "head0_input_slice_from_attention_value": logical["attention_value_head0_token1_reference_s8"],
        },
        "continuity": {
            "attention_value_output_equals_o_proj_input_head0_slice": True,
            "compared_bytes": HEAD_DIM,
            "full_o_proj_input_bytes": HIDDEN,
        },
        "missing_fields": [
            "layer_0.o_proj.input.weights_s4[896][896]",
            "layer_0.o_proj.input.bias_accumulator_s32[896]",
            "layer_0.o_proj.input.multiplier_s32[896]",
            "layer_0.o_proj.input.right_shift_u6[896]",
        ],
        "scope_reason": "The immutable retained package contains the exact full o_proj input and output reference but not the exact projection weights or per-channel requantization metadata. The V-only capture authorization forbids acquiring them.",
        "simulation_executed": False,
        "later_operators_attempted": False,
    }
    prior.write_json(REPLAY / "boundary_probe.json", boundary)
    report = {
        "schema_version": 1,
        "mission_id": "rtl-attention-value-token0-v-reference-batch-replay-v1",
        "producer_role": "engineer",
        "review_status": "PENDING_FRESH_REVIEWER",
        "coordinate": {
            "dataset": DATASET,
            "dataset_record_index": RECORD,
            "layer": LAYER,
            "query_token": QUERY_TOKEN,
            "active_tokens": [0, 1],
            "query_head": HEAD,
            "mapped_kv_head": 0,
        },
        "capture": {
            "consumption_marker": artifact(CAPTURE / "capture_execution_consumed.json"),
            "evaluation_forward_count": 1,
            "persisted_tensor_count": 1,
            "rope_q_k_token1_head1_reused_not_recaptured": True,
            "witness": artifact(CAPTURE / "witness.json"),
            "merge_manifest": artifact(MERGE / "merge_manifest.json"),
        },
        "frozen_order": ORDER,
        "rtl_tree": BASE.rtl_tree(),
        "prior_maximal_prefix": ORDER[:9],
        "attempted_operators": [
            {
                "operator": "layer_0.attention_value",
                "status": "PASS_EXACT",
                "input_hashes": {
                    "probabilities_active_q15": logical["softmax_head0_query_token1_active_q15"]["sha256"],
                    "v_token0_head0": logical["v_cache_head0_token0_s8"]["sha256"],
                    "v_token1_head0": logical["v_cache_head0_token1_s8"]["sha256"],
                },
                "reference_hashes": {
                    "attention_value_head0_s8": logical["attention_value_head0_token1_reference_s8"]["sha256"],
                    "accumulator_s32": prior.sha256_bytes(_raw_i32(reference.accumulators)),
                },
                "output_hashes": {
                    "rtl_attention_value_head0_s8": prior.sha256_bytes(rtl_raw),
                },
                "counts": {
                    "active_tokens": CONTEXT,
                    "output_elements": HEAD_DIM,
                    "compared_outputs": HEAD_DIM,
                    "mismatches": 0,
                },
                "saturation": {
                    "reference_saturation_seen": reference.saturation_seen,
                    "rtl_result": "MATCH",
                },
                "cycle_handshake_latency": {
                    "simulation_executed": True,
                    "total_command_cycles": int(match.group(2)),
                    "handshake_result": "PASS_NO_TIMEOUT_OR_DROPPED_WRITE",
                    "latency_result": "PASS_BOUNDED_BY_MAINTAINED_SHELL_WATCHDOG",
                },
                "continuity": {
                    "output_equals_o_proj_input_head0_slice": True,
                    "compared_bytes": HEAD_DIM,
                },
                "raw_log": artifact(REPLAY / "replay.log"),
            },
            {
                "operator": "layer_0.o_proj",
                "status": boundary["classification"],
                "counts": {
                    "retained_input_bytes": HIDDEN,
                    "retained_output_reference_bytes": HIDDEN,
                    "missing_weight_elements": HIDDEN * HIDDEN,
                    "missing_metadata_channels": HIDDEN,
                    "mismatch_count": None,
                },
                "cycle_handshake_latency": {
                    "simulation_executed": False,
                    "handshake_result": "NOT_EVALUATED_INTERFACE_UNAVAILABLE",
                    "latency_result": "NOT_EVALUATED_INTERFACE_UNAVAILABLE",
                },
                "boundary_probe": artifact(REPLAY / "boundary_probe.json"),
            },
        ],
        "newly_passing_operators": ["layer_0.attention_value"],
        "maximal_passing_prefix": ORDER[:10],
        "first_failing_or_unsupported_operator": {
            "operator": "layer_0.o_proj",
            "classification": boundary["classification"],
            "boundary_probe": artifact(REPLAY / "boundary_probe.json"),
        },
        "batch_result": "STOPPED_AT_FIRST_PRECISELY_UNAVAILABLE_EXECUTABLE_INTERFACE",
        "later_operators_attempted": False,
        "existing_current_tree_ppa": {
            "reused_not_rerun": True,
            "sha256_manifest": artifact(PPA_SUMS),
        },
        "protected_state_pre_post_exact_match": prior.protected_state()
        == json.loads((CAPTURE / "preflight.json").read_text(encoding="utf-8"))["protected_state"],
        "forbidden_flows_run": [],
        "state_mutations": {
            "rtl": False,
            "frontier_ledger_manifest_traceability_pipeline_seals": False,
            "synthesis_opensta_ppa": False,
            "baseline_candidate_benchmark_scale32": False,
        },
        "review_requirement": "Fresh Reviewer must independently accept or reject this maximal-prefix result.",
    }
    require(report["protected_state_pre_post_exact_match"], "protected state changed during capture/replay")
    prior.write_json(REPLAY / "ordered_prefix_report.json", report)
    (REPLAY / "batch_replay.log").write_text(
        "ACE2_ATTENTION_VALUE_TOKEN0_ORDERED_BATCH_REPLAY\n"
        f"rtl_tree_sha256={EXPECTED_RTL_TREE}\n"
        "coordinate=c4_en_512:record64:layer0:query1:active_tokens0-1:head0\n"
        "layer_0.attention_value=PASS_EXACT outputs=64 mismatches=0\n"
        "layer_0.o_proj=UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_RETAINED_O_PROJ_WEIGHTS_METADATA_INCOMPLETE\n"
        "maximal_passing_operator=layer_0.attention_value\n"
        "later_operators_attempted=false\n"
        "synthesis_opensta_ppa_executed=false\n",
        encoding="utf-8",
    )
    prior.write_sha256s(REPLAY)
    print(
        "ACE2_ATTENTION_VALUE_TOKEN0_BATCH_REPLAY_BOUNDARY "
        "maximal_prefix=layer_0.attention_value first_boundary=layer_0.o_proj mismatches=0"
    )


def launch_capture_and_replay() -> None:
    preflight = validate_preflight()
    require(not (CAPTURE / "capture_execution_consumed.json").exists(), "capture already consumed")
    require(not (CAPTURE / "witness.json").exists(), "capture witness already exists")
    require(not MERGE.exists(), "merge output already exists")
    require(not REPLAY.exists(), "replay output already exists")
    source_archive = CAPTURE / "capture_source_at_execution.py"
    shutil.copy2(SELF, source_archive)
    require(prior.sha256_file(source_archive) == preflight["source"]["sha256"], "source changed after preflight")
    stdout_handle = (CAPTURE / "capture.stdout").open("x", encoding="utf-8")
    stderr_handle = (CAPTURE / "capture.stderr").open("x", encoding="utf-8")
    marker = CAPTURE / "capture_execution_consumed.json"
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "status": "CONSUMED_BEFORE_SINGLE_WORKER_LAUNCH",
                "consumed_at_utc": prior.utc_now(),
                "command": [sys.executable, prior.relative(SELF), "_capture_worker"],
                "coordinate": {
                    "dataset": DATASET,
                    "dataset_record_index": RECORD,
                    "head": HEAD,
                    "layer": LAYER,
                    "token": CAPTURE_TOKEN,
                    "query_token": QUERY_TOKEN,
                },
                "authorized_persisted_tensors": ["v_cache_head0_token0_s8"],
                "source_archive": artifact(source_archive),
                "raw_stdout": prior.relative(CAPTURE / "capture.stdout"),
                "raw_stderr": prior.relative(CAPTURE / "capture.stderr"),
                "rerun_permitted": False,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(marker, 0o444)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    worker = subprocess.run(
        [sys.executable, str(SELF), "_capture_worker"],
        cwd=ROOT,
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        check=False,
    )
    stdout_handle.close()
    stderr_handle.close()
    prior.write_json(
        CAPTURE / "capture_worker_status.json",
        {
            "schema_version": 1,
            "completed_at_utc": prior.utc_now(),
            "returncode": worker.returncode,
            "rerun_permitted": False,
            "stdout": artifact(CAPTURE / "capture.stdout"),
            "stderr": artifact(CAPTURE / "capture.stderr"),
        },
    )
    require(worker.returncode == 0, "single capture worker failed; authorization remains consumed")
    prior.write_sha256s(CAPTURE)
    merge_packages()
    run_replay()


def require_total_post_marker_forward_budget(witness: dict[str, Any]) -> None:
    capture = witness["capture"]
    calibration_count = capture.get("calibration_dependency_forward_count")
    evaluation_count = capture.get("evaluation_forward_count")
    require(type(calibration_count) is int, "calibration forward count is missing")
    require(type(evaluation_count) is int, "evaluation forward count is missing")
    total_count = calibration_count + evaluation_count
    require(
        total_count <= 1,
        f"capture exceeded one total post-marker model forward: observed {total_count}",
    )


def validate_completed_capture() -> dict[str, Any]:
    preflight = json.loads((CAPTURE / "preflight.json").read_text(encoding="utf-8"))
    require(preflight["status"] == "PASS_MODEL_EXECUTION_NOT_CONSUMED", "preflight status changed")
    require(preflight["model_execution_count"] == 0, "preflight execution count changed")
    require(artifact(CAPTURE / "preflight_source_at_execution.py") == preflight["source_archive"], "preflight source archive changed")
    prior.validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE)
    marker = json.loads((CAPTURE / "capture_execution_consumed.json").read_text(encoding="utf-8"))
    witness = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    require(marker["rerun_permitted"] is False, "consumption marker permits rerun")
    require(witness["capture"]["evaluation_forward_count"] == 1, "capture was not exactly once")
    require_total_post_marker_forward_budget(witness)
    require(witness["capture"]["persisted_tensor_count"] == 1, "capture tensor scope changed")
    require(set(witness["tensor_manifest"]) == {"v_cache_head0_token0_s8"}, "capture contains unauthorized tensors")
    for field in ("rope_q_recaptured", "k_recaptured", "token1_recaptured", "head1_recaptured"):
        require(witness["capture"][field] is False, f"capture scope changed: {field}")
    require(prior.protected_state() == preflight["protected_state"], "protected state changed after capture")
    require(BASE.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed after capture")
    accepted_packages()
    return preflight


def resume_replay() -> None:
    validate_completed_capture()
    prior.validate_manifest(MERGE / "SHA256SUMS", MERGE)
    run_replay()


def run_check() -> None:
    preflight = validate_completed_capture()
    prior.validate_manifest(MERGE / "SHA256SUMS", MERGE)
    prior.validate_manifest(REPLAY / "SHA256SUMS", REPLAY)
    marker = json.loads((CAPTURE / "capture_execution_consumed.json").read_text(encoding="utf-8"))
    witness = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    merge = json.loads((MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((REPLAY / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    require(marker["rerun_permitted"] is False, "consumption marker permits rerun")
    require((CAPTURE / "capture_execution_consumed.json").stat().st_mode & 0o222 == 0, "consumption marker is writable")
    require(witness["capture"]["evaluation_forward_count"] == 1, "capture was not exactly once")
    require(witness["capture"]["persisted_tensor_count"] == 1, "capture tensor scope changed")
    require(set(witness["tensor_manifest"]) == {"v_cache_head0_token0_s8"}, "capture contains unauthorized tensors")
    require(merge["source_packages_modified"] is False, "merge modified a source package")
    require(report["newly_passing_operators"] == ["layer_0.attention_value"], "passing operators changed")
    require(report["maximal_passing_prefix"] == ORDER[:10], "maximal prefix changed")
    require(report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.o_proj", "first boundary changed")
    require(report["later_operators_attempted"] is False, "replay continued beyond first boundary")
    require(
        report["state_mutations"]
        == {
            "rtl": False,
            "frontier_ledger_manifest_traceability_pipeline_seals": False,
            "synthesis_opensta_ppa": False,
            "baseline_candidate_benchmark_scale32": False,
        },
        "forbidden mutation record changed",
    )
    require(prior.protected_state() == preflight["protected_state"], "protected state changed")
    require(BASE.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed after replay")
    print(
        "ACE2_ATTENTION_VALUE_TOKEN0_CAPTURE_BATCH_REPLAY_CHECK_PASS "
        f"capture_sha256={prior.sha256_file(CAPTURE / 'witness.json')} "
        f"merge_sha256={prior.sha256_file(MERGE / 'merge_manifest.json')} "
        f"report_sha256={prior.sha256_file(REPLAY / 'ordered_prefix_report.json')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("preflight", "capture-and-replay", "resume-replay", "check", "_capture_worker"),
    )
    args = parser.parse_args()
    if args.mode == "preflight":
        run_preflight()
    elif args.mode == "capture-and-replay":
        launch_capture_and_replay()
    elif args.mode == "resume-replay":
        resume_replay()
    elif args.mode == "check":
        run_check()
    else:
        execute_capture_worker()


if __name__ == "__main__":
    main()
