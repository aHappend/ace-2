#!/usr/bin/env python3
"""Capture only token-0/head-0 K and replay from layer-0 attention score."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_layer0_minimal_reference_capture_batch_replay as base
from ace2_softmax_reference import SoftmaxCase, reference_softmax


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-attention-score-token0-reference-batch-replay-v1/mission.json"
)
ROPE_Q = ROOT / "evidence/diagnostics/c4-rope-q-token1-capture-replay-20260803-v2"
HEAD0 = ROOT / "evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2"
HEAD1 = ROOT / "evidence/diagnostics/c4-layer0-kv-head1-minimal-reference-capture-20260803-v1"
KV_MERGE = ROOT / "evidence/diagnostics/c4-layer0-kv-write-two-head-merged-20260803-v1"
KV_REPLAY = ROOT / "evidence/verification/rtl-kv-write-two-head-reference-batch-replay-v1"
PPA_SUMS = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/SHA256SUMS"

CAPTURE = ROOT / "evidence/diagnostics/c4-layer0-attention-score-token0-k-minimal-reference-capture-20260803-v1"
MERGE = ROOT / "evidence/diagnostics/c4-layer0-attention-score-inputs-merged-20260803-v1"
REPLAY = ROOT / "evidence/verification/rtl-attention-score-token0-reference-batch-replay-v1"

EXPECTED_RTL_TREE = "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be"
EXPECTED_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_ROPE_Q_SUMS = "9d1640be7cc126c689b53f2a0009069af0fdd02050cf4114659d0e8a000893ac"
EXPECTED_ROPE_Q_WITNESS = "2ad63d87d0fd53664e606faed32e60d5f5e69c94903f07f93de6ae1822657404"
EXPECTED_HEAD0_SUMS = "7e3c768d9eff34b327859621e0e093243cf7562837611ad0bd37480a86703e97"
EXPECTED_HEAD1_SUMS = "6c7d6bcbf8e5034ffd88df0eb074da83959029de4b32ba0681fb310f9fe49228"
EXPECTED_KV_MERGE_SUMS = "4b6926066e9a8ca97e44d759653ff12a3c2ae642ca5ed4ecad11c8b0fee7ddd0"
EXPECTED_KV_REPLAY_SUMS = "ee23b5846fc17f479add74fa6a05646846421e0f05b3608455bf109648b00985"
EXPECTED_KV_REPLAY_REPORT = "5be60b3d287be14e4c5c6a5d7cb1f52454be32799c6616b8c8f903e21a73c2de"
EXPECTED_PPA_SUMS = "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4"
EXPECTED_BASE_SOURCE = "c23fb184455f45e5f60250e7cab80f5cb4f152f86c5e4270bd75082e0252b1ef"
EXPECTED_FIXED_MODEL = "b5f1c1800560894a728f199d36b31ef9d624b5afa9d56ee544b0d467270c1c74"
EXPECTED_SHELL_TB = "cefd1e4533f9e6de9f85583c0095f3eb2e81329c190f0e0044d045e31da76d18"
EXPECTED_SHELL = "3bb8caab4f06e6be9b170b5b3d91cb89b237715132e52507cd60f0514c61ab30"

DATASET = "c4_en_512"
RECORD = 64
LAYER = 0
QUERY_TOKEN = 1
CAPTURE_TOKEN = 0
HEAD = 0
HEAD_DIM = 64
CONTEXT = 2
ORDER = base.ORDER

PROTECTED_PATHS = [
    "design/RTL_MANIFEST.json",
    "design/RTL_MANIFEST.sha256",
    "design/PPA_FRONTIER_LEDGER.json",
    "design/RTL_TRACEABILITY.md",
    "design/RTL_TRACEABILITY.sha256",
    "research/PIPELINE_STATE.json",
    "research/PUBLIC_STATUS.json",
    "research/ENVIRONMENT_AUDIT.json",
]


class Token0KCaptureComplete(RuntimeError):
    pass


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


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path, public_path: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": public_path or relative(path),
        "sha256": sha256_file(path),
    }


def external_artifact(path: Path, public_path: str) -> dict[str, Any]:
    require(path.is_file(), f"missing external artifact: {path}")
    return {"bytes": path.stat().st_size, "path": public_path, "sha256": sha256_file(path)}


def validate_manifest(path: Path, base_dir: Path) -> int:
    count = 0
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row:
            continue
        expected, name = row.split("  ", 1)
        member = Path(name)
        member = member if member.is_absolute() else base_dir / member
        require(member.is_file(), f"missing manifest member: {name}")
        require(sha256_file(member) == expected, f"changed manifest member: {name}")
        count += 1
    return count


def write_sha256s(base_dir: Path) -> Path:
    output = base_dir / "SHA256SUMS"
    rows = []
    for member in sorted(base_dir.rglob("*")):
        if member.is_file() and member != output:
            rows.append(f"{sha256_file(member)}  {member.relative_to(base_dir).as_posix()}\n")
    output.write_text("".join(rows), encoding="utf-8")
    return output


def protected_state() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in PROTECTED_PATHS}


def tensor_raw(record: dict[str, Any]) -> bytes:
    path = ROOT / record["path"]
    raw = path.read_bytes()
    require(len(raw) == record["bytes"], f"tensor byte count changed: {path}")
    require(sha256_bytes(raw) == record["sha256"], f"tensor hash changed: {path}")
    return raw


def tensor_i8(record: dict[str, Any]) -> list[int]:
    raw = tensor_raw(record)
    return [value - 256 if value >= 128 else value for value in raw]


def tensor_i16(record: dict[str, Any]) -> list[int]:
    raw = tensor_raw(record)
    require(len(raw) % 2 == 0, "int16 tensor byte count is odd")
    values = []
    for index in range(0, len(raw), 2):
        value = int.from_bytes(raw[index : index + 2], "little")
        values.append(value - 65536 if value >= 32768 else value)
    return values


def query_record(rope_q: dict[str, Any]) -> dict[str, Any]:
    values = rope_q["rope"]["expected_output_s8"]
    raw = bytes(value & 0xFF for value in values)
    require(len(raw) == HEAD_DIM, "accepted RoPE-Q query length changed")
    return {
        "bytes": len(raw),
        "dtype": "int8",
        "elements": len(values),
        "json_pointer": "/rope/expected_output_s8",
        "path": relative(ROPE_Q / "witness.json"),
        "sha256": sha256_bytes(raw),
        "shape": [HEAD_DIM],
    }


def accepted_packages() -> dict[str, Any]:
    require(sha256_file(ROPE_Q / "SHA256SUMS") == EXPECTED_ROPE_Q_SUMS, "RoPE-Q aggregate changed")
    require(sha256_file(ROPE_Q / "witness.json") == EXPECTED_ROPE_Q_WITNESS, "RoPE-Q witness changed")
    require(sha256_file(HEAD0 / "SHA256SUMS") == EXPECTED_HEAD0_SUMS, "head-0 package changed")
    require(sha256_file(HEAD1 / "SHA256SUMS") == EXPECTED_HEAD1_SUMS, "head-1 package changed")
    require(sha256_file(KV_MERGE / "SHA256SUMS") == EXPECTED_KV_MERGE_SUMS, "KV merge changed")
    require(sha256_file(KV_REPLAY / "SHA256SUMS") == EXPECTED_KV_REPLAY_SUMS, "KV replay changed")
    require(
        sha256_file(KV_REPLAY / "ordered_prefix_report.json") == EXPECTED_KV_REPLAY_REPORT,
        "accepted KV replay report changed",
    )
    member_counts = {
        "rope_q": validate_manifest(ROPE_Q / "SHA256SUMS", ROPE_Q),
        "head0": validate_manifest(HEAD0 / "SHA256SUMS", HEAD0),
        "head1": validate_manifest(HEAD1 / "SHA256SUMS", HEAD1),
        "kv_merge": validate_manifest(KV_MERGE / "SHA256SUMS", KV_MERGE),
        "kv_replay": validate_manifest(KV_REPLAY / "SHA256SUMS", KV_REPLAY),
    }
    rope_q = json.loads((ROPE_Q / "witness.json").read_text(encoding="utf-8"))
    head0 = json.loads((HEAD0 / "witness.json").read_text(encoding="utf-8"))
    head1 = json.loads((HEAD1 / "witness.json").read_text(encoding="utf-8"))
    kv_merge = json.loads((KV_MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    kv_report = json.loads((KV_REPLAY / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    require(rope_q["coordinate"] == {
        "batch": 0,
        "dataset": DATASET,
        "dataset_record_index": RECORD,
        "head": HEAD,
        "layer": LAYER,
        "operator": "rope_q",
        "token": QUERY_TOKEN,
        "token_id": rope_q["coordinate"]["token_id"],
    }, "RoPE-Q coordinate changed")
    for witness, expected_head in ((head0, 0), (head1, 1)):
        coordinate = witness["coordinate"]
        require(coordinate["dataset"] == DATASET and coordinate["dataset_record_index"] == RECORD, "KV package record changed")
        require(coordinate["layer"] == LAYER and coordinate["token"] == QUERY_TOKEN, "KV package layer/token changed")
        require(coordinate["head"] == expected_head, "KV package head changed")
    require(kv_merge["source_packages_modified"] is False, "accepted KV merge is not immutable")
    require(kv_report["rtl_tree"]["sha256"] == EXPECTED_RTL_TREE, "accepted replay RTL tree changed")
    require(kv_report["maximal_passing_prefix"] == ORDER[:7], "accepted replay prefix changed")
    require(kv_report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.attention_score", "accepted boundary changed")
    require(head0["input_binding"]["evaluation_witness"]["record_sha256"] == rope_q["dataset_provenance"]["evaluation_witness"]["record_sha256"], "record hash mismatch between packages")
    require(head0["input_binding"]["evaluation_witness"]["token_sequence_sha256"] == rope_q["dataset_provenance"]["evaluation_witness"]["token_sequence_sha256"], "token hash mismatch between packages")
    return {
        "rope_q": rope_q,
        "head0": head0,
        "head1": head1,
        "kv_merge": kv_merge,
        "kv_report": kv_report,
        "member_counts": member_counts,
    }


def qualifying_capture_matches() -> list[str]:
    matches: list[str] = []
    for path in sorted((ROOT / "evidence/diagnostics").glob("*/witness.json")):
        if CAPTURE in path.parents:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "k_cache_head0_token0_s8" in text or "rope_k_head0_token0_s8" in text:
            matches.append(relative(path))
    return matches


def preflight_payload(source_archive: Path) -> dict[str, Any]:
    require(base.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed")
    require(sha256_file(Path(base.__file__)) == EXPECTED_BASE_SOURCE, "accepted capture helper changed")
    require(sha256_file(ROOT / "tools/ace2_full_model_fixed_point.py") == EXPECTED_FIXED_MODEL, "fixed model source changed")
    require(sha256_file(ROOT / "verification/tb/ace2_shell_tb.sv") == EXPECTED_SHELL_TB, "shell harness changed")
    require(sha256_file(ROOT / "rtl/ace2_shell.sv") == EXPECTED_SHELL, "shell RTL changed")
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SUMS, "canonical PPA aggregate changed")
    ppa_members = validate_manifest(PPA_SUMS, ROOT)
    packages = accepted_packages()
    require(not qualifying_capture_matches(), "a qualifying token-0 K capture already exists")
    live_binding, _calibration, _evaluation, manifest, versions = base.live_input_binding()
    require(live_binding == packages["head0"]["input_binding"], "live C4 input binding changed")
    model_spec = manifest["model"]
    require(model_spec["repository"] == "Qwen/Qwen2.5-0.5B", "model repository changed")
    require(model_spec["revision"] == EXPECTED_REVISION, "model revision changed")
    cached_files = base.cached_model_bindings(model_spec["repository"], model_spec["revision"])
    accepted_cached = json.loads((HEAD0 / "preflight.json").read_text(encoding="utf-8"))["model"]["cached_files"]
    require(cached_files == accepted_cached, "cached model/config/tokenizer hashes changed")
    tools = {name: shutil.which(name) for name in ("iverilog", "vvp")}
    require(all(tools.values()), f"required RTL tools unavailable: {tools}")
    return {
        "schema_version": 1,
        "mission_id": "rtl-attention-score-token0-reference-batch-replay-v1",
        "status": "PASS_MODEL_EXECUTION_NOT_CONSUMED",
        "completed_at_utc": utc_now(),
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
            "member_counts": packages["member_counts"],
        },
        "absence_of_existing_qualifying_capture": {
            "match_count": 0,
            "matches": [],
            "search": "evidence/diagnostics/*/witness.json token-0/head-0 K payload names",
        },
        "output_path_semantics": {
            "capture": relative(CAPTURE),
            "merge": relative(MERGE),
            "replay": relative(REPLAY),
            "all_absent_before_preflight": True,
            "create_modes": "exclusive mkdir/open; no overwrite",
        },
        "capture_scope": {
            "persisted_tensor_names": [
                "k_projection_head0_token0_s8",
                "k_cache_head0_token0_s8",
            ],
            "persisted_tensor_count": 2,
            "metadata_only": [
                "query_rope_output_scale",
                "key_rope_output_scale",
                "attention_score_multiplier_s32",
                "attention_score_right_shift_u6",
            ],
            "rope_q_recapture": False,
            "head1_recapture": False,
        },
        "mission": external_artifact(MISSION, "handoff:rtl-attention-score-token0-reference-batch-replay-v1/mission.json"),
        "protected_state": protected_state(),
        "runtime": {
            "packages": versions,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "rtl_tree": base.rtl_tree(),
        "canonical_ppa": {"reused_not_rerun": True, "member_count": ppa_members, "sha256_manifest": artifact(PPA_SUMS)},
        "source": artifact(SELF),
        "source_archive": artifact(source_archive),
        "tool_lookup": {name: Path(value).name for name, value in tools.items()},
        "forbidden_flows_run": [],
    }


def run_preflight() -> None:
    require(not CAPTURE.exists(), f"capture path already exists: {CAPTURE}")
    require(not MERGE.exists(), f"merge path already exists: {MERGE}")
    require(not REPLAY.exists(), f"replay path already exists: {REPLAY}")
    source_hash = sha256_file(SELF)
    payload = preflight_payload(SELF)
    require(payload["source"]["sha256"] == source_hash, "source changed during preflight")
    CAPTURE.mkdir(parents=True, exist_ok=False)
    source_archive = CAPTURE / "preflight_source_at_execution.py"
    shutil.copy2(SELF, source_archive)
    payload["source_archive"] = artifact(source_archive)
    require(payload["source_archive"]["sha256"] == payload["source"]["sha256"], "preflight source archive differs")
    write_json(CAPTURE / "preflight.json", payload)
    (CAPTURE / "preflight.log").write_text(
        "ACE2_ATTN_SCORE_TOKEN0_K_PREFLIGHT_PASS\n"
        "model_execution_count=0\n"
        "coordinate=c4_en_512:record64:layer0:query1:key0:head0\n"
        "persisted_tensor_count=2\n"
        "rope_q_recapture=false\n"
        "existing_qualifying_capture_count=0\n",
        encoding="utf-8",
    )
    print("ACE2_ATTN_SCORE_TOKEN0_K_PREFLIGHT_PASS model_execution_count=0")


def validate_preflight() -> dict[str, Any]:
    path = CAPTURE / "preflight.json"
    require(path.is_file(), "preflight artifact is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["status"] == "PASS_MODEL_EXECUTION_NOT_CONSUMED", "preflight status changed")
    require(payload["model_execution_count"] == 0, "preflight consumed model execution")
    require(sha256_file(SELF) == payload["source"]["sha256"], "runner source changed after preflight")
    require(artifact(CAPTURE / "preflight_source_at_execution.py") == payload["source_archive"], "preflight source archive changed")
    require(base.rtl_tree()["sha256"] == payload["rtl_tree"]["sha256"], "RTL changed after preflight")
    require(protected_state() == payload["protected_state"], "protected state changed after preflight")
    accepted_packages()
    return payload


class MinimalKTrace:
    def __init__(self, layer: base.FixedDecoderLayer, token1_projection: Tensor, token1_rope: Tensor) -> None:
        self.layer = layer
        self.token1_projection = token1_projection
        self.token1_rope = token1_rope
        self.k_projection_token0: Tensor | None = None
        self.k_rope_token0: Tensor | None = None
        self.k_projection_token1_match = False
        self.k_rope_token1_match = False
        self.rope_call_count = 0
        self._method_original: Callable[..., Any] | None = None
        self._rope_original: Callable[..., Any] | None = None

    def install(self) -> None:
        module = self.layer.self_attn.k_proj
        original_method = module.forward_hardware_input
        self._method_original = original_method

        def method_wrapper(_self: nn.Module, *args: Any, **kwargs: Any) -> Any:
            result = original_method(*args, **kwargs)
            require(isinstance(result, Tensor), "K projection result is not a tensor")
            self.k_projection_token0 = result[0, CAPTURE_TOKEN, :HEAD_DIM].detach().cpu().to(torch.int8).clone()
            live_token1 = result[0, QUERY_TOKEN, :HEAD_DIM].detach().cpu().to(torch.int8)
            self.k_projection_token1_match = torch.equal(live_token1, self.token1_projection)
            return result

        module.__dict__["forward_hardware_input"] = types.MethodType(method_wrapper, module)
        original_rope = base.fixed.fixed_rope_raw_with_saturation
        self._rope_original = original_rope

        def rope_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original_rope(*args, **kwargs)
            self.rope_call_count += 1
            if self.rope_call_count == 2:
                require(isinstance(result, tuple) and isinstance(result[0], Tensor), "K RoPE result changed")
                self.k_rope_token0 = result[0][0, HEAD, CAPTURE_TOKEN].detach().cpu().to(torch.int8).clone()
                live_token1 = result[0][0, HEAD, QUERY_TOKEN].detach().cpu().to(torch.int8)
                self.k_rope_token1_match = torch.equal(live_token1, self.token1_rope)
                raise Token0KCaptureComplete
            return result

        base.fixed.fixed_rope_raw_with_saturation = rope_wrapper

    def restore(self) -> None:
        if self._method_original is not None:
            self.layer.self_attn.k_proj.__dict__.pop("forward_hardware_input", None)
        if self._rope_original is not None:
            base.fixed.fixed_rope_raw_with_saturation = self._rope_original


def execute_capture_worker() -> None:
    preflight = validate_preflight()
    marker = CAPTURE / "capture_execution_consumed.json"
    source_archive = CAPTURE / "capture_source_at_execution.py"
    require(marker.is_file(), "capture consumption marker is missing")
    require(source_archive.is_file(), "capture source archive is missing")
    require(sha256_file(source_archive) == preflight["source"]["sha256"], "capture source differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}, "capture requires CPU-only visibility")
    packages = accepted_packages()
    live_binding, calibration_prompt, evaluation_prompt, manifest, versions = base.live_input_binding()
    require(live_binding == preflight["input_binding"], "live input changed after preflight")
    config = base.load_contracts(require_rtl_binding=False)[1]
    base.seed_everything(config)
    model_spec = manifest["model"]
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "model is not CPU-only")
    require(getattr(model.config, "_commit_hash", None) == EXPECTED_REVISION, "resolved model revision changed")
    ranges, operator_ranges = base.calibrate(model, [calibration_prompt])
    base.replace_linears(model, ranges, rope_diagnostic_mechanism=None)
    base.replace_fixed_operators(model, operator_ranges, rope_diagnostic_mechanism=None)
    layer = model.model.layers[LAYER]
    require(isinstance(layer, base.FixedDecoderLayer), "layer 0 is not FixedDecoderLayer")
    require(isinstance(layer.self_attn, base.FixedAttention), "layer 0 attention is not fixed")
    head0 = packages["head0"]
    token1_projection = base.read_tensor(head0["tensor_manifest"]["k_projection_output_s8"]).to(torch.int8)
    token1_rope = base.read_tensor(head0["tensor_manifest"]["kv_write_k_head0_token1_s8"]).to(torch.int8)
    trace = MinimalKTrace(layer, token1_projection, token1_rope)
    trace.install()
    evaluation_forward_count = 0
    base.seed_everything(config)
    try:
        with torch.inference_mode():
            evaluation_forward_count += 1
            model(input_ids=evaluation_prompt, use_cache=False)
    except Token0KCaptureComplete:
        pass
    finally:
        trace.restore()
    require(evaluation_forward_count == 1, "evaluation capture forward count changed")
    require(trace.rope_call_count == 2, "capture did not stop at the K RoPE return")
    require(trace.k_projection_token0 is not None, "token-0 K projection was not captured")
    require(trace.k_rope_token0 is not None, "token-0 K cache payload was not captured")
    require(trace.k_projection_token1_match, "live token-1 K projection differs from immutable head 0")
    require(trace.k_rope_token1_match, "live token-1 K RoPE differs from immutable head 0")
    query_scale = float(layer.self_attn.query_rope_output_scales[HEAD])
    key_scale = float(layer.self_attn.key_rope_output_scales[HEAD])
    multiplier = int(layer.self_attn.score_multiplier[HEAD])
    right_shift = int(layer.self_attn.score_right_shift[HEAD])
    require(query_scale == float(packages["rope_q"]["rope"]["output_scale"]), "query RoPE scale changed")
    require(key_scale == float(head0["rope_k"]["output_scale"]), "key RoPE scale changed")
    require(multiplier == int(head0["operator_metadata"]["attention_score_multiplier_s32"]), "attention multiplier changed")
    require(right_shift == int(head0["operator_metadata"]["attention_score_right_shift_u6"]), "attention shift changed")
    tensors = {
        "k_projection_head0_token0_s8": base.write_tensor(CAPTURE, "k_projection_head0_token0_s8", trace.k_projection_token0),
        "k_cache_head0_token0_s8": base.write_tensor(CAPTURE, "k_cache_head0_token0_s8", trace.k_rope_token0),
    }
    require(set(tensors) == {"k_projection_head0_token0_s8", "k_cache_head0_token0_s8"}, "capture tensor scope expanded")
    witness = {
        "schema_version": 1,
        "classification": "single_forward_extraction_only_token0_head0_k_for_attention_score",
        "capture": {
            "calibration_dependency_forward_count": 1,
            "completed_at_utc": utc_now(),
            "consumption_marker": artifact(marker),
            "evaluation_forward_count": evaluation_forward_count,
            "persisted_tensor_count": len(tensors),
            "record_count_retained": 1,
            "layer_count_retained": 1,
            "head_count_retained": 1,
            "token_count_retained": 1,
            "rope_q_recaptured": False,
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
        "attention_score_scale_metadata": {
            "query_head": HEAD,
            "mapped_kv_head": 0,
            "query_rope_output_scale": query_scale,
            "key_rope_output_scale": key_scale,
            "multiplier_s32": multiplier,
            "right_shift_u6": right_shift,
            "score_format": "signed_int16_Q6.9_centered_over_two_active_tokens",
        },
        "compatibility": {
            "immutable_token1_k_projection_exact_match": True,
            "immutable_token1_k_rope_exact_match": True,
            "live_input_binding_exact_match": True,
            "accepted_rope_q_reused_not_recaptured": True,
        },
        "tensor_manifest": tensors,
        "source_bindings": {
            "capture_tool": artifact(source_archive),
            "fixed_model": artifact(ROOT / "tools/ace2_full_model_fixed_point.py"),
            "accepted_rope_q": artifact(ROPE_Q / "SHA256SUMS"),
            "immutable_head0": artifact(HEAD0 / "SHA256SUMS"),
            "immutable_head1": artifact(HEAD1 / "SHA256SUMS"),
            "merged_two_head_kv": artifact(KV_MERGE / "SHA256SUMS"),
            "rtl_tree": base.rtl_tree(),
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
            "token1_recapture",
            "head1_recapture",
            "v_projection_or_v_cache_payload",
            "softmax_or_later_reference_capture",
            "other_records_heads_layers_tokens",
            "rtl_mutation",
            "synthesis_opensta_ppa",
            "baseline_candidate_benchmark_scale32",
            "frontier_authority_seal_mutation",
        ],
    }
    write_json(CAPTURE / "witness.json", witness)
    write_json(
        CAPTURE / "capture_contract.json",
        {
            "schema_version": 1,
            "status": "CAPTURE_COMPLETE_MERGE_REPLAY_PENDING",
            "completed_at_utc": utc_now(),
            "coordinate": witness["coordinate"],
            "evaluation_forward_count": 1,
            "persisted_tensor_count": 2,
            "rope_q_recaptured": False,
            "source": artifact(source_archive),
            "witness": artifact(CAPTURE / "witness.json"),
        },
    )
    (CAPTURE / "capture.log").write_text(
        "ACE2_ATTN_SCORE_TOKEN0_K_CAPTURE_COMPLETE\n"
        "coordinate=c4_en_512:record64:layer0:query1:key0:head0\n"
        "evaluation_forward_count=1\n"
        "persisted_tensor_count=2\n"
        "rope_q_recaptured=false\n"
        "head1_recaptured=false\n"
        "rtl_mutated=false\n",
        encoding="utf-8",
    )
    print(
        "ACE2_ATTN_SCORE_TOKEN0_K_CAPTURE_COMPLETE "
        f"tensors=2 evaluation_forwards=1 witness_sha256={sha256_file(CAPTURE / 'witness.json')}"
    )


def merge_packages() -> None:
    require(not MERGE.exists(), "attention-score merge output already exists")
    packages = accepted_packages()
    validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE)
    captured = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    head0 = packages["head0"]
    query = query_record(packages["rope_q"])
    expected_scores = head0["tensor_manifest"]["attention_score_head0_query_token1_active_s16"]
    expected_probabilities = head0["tensor_manifest"]["softmax_head0_query_token1_active_q15"]
    logical = {
        "query_head0_token1_s8": query,
        "k_cache_head0_token0_s8": captured["tensor_manifest"]["k_cache_head0_token0_s8"],
        "k_cache_head0_token1_s8": head0["tensor_manifest"]["kv_write_k_head0_token1_s8"],
        "score_reference_head0_query_token1_active_s16": expected_scores,
        "softmax_reference_head0_query_token1_active_q15": expected_probabilities,
        "v_cache_head0_token1_s8": head0["tensor_manifest"]["kv_write_v_head0_token1_s8"],
    }
    MERGE.mkdir(parents=True, exist_ok=False)
    write_json(
        MERGE / "merge_manifest.json",
        {
            "schema_version": 1,
            "mission_id": "rtl-attention-score-token0-reference-batch-replay-v1",
            "classification": "derived_hash_bound_attention_score_merge_no_source_package_modification",
            "coordinate": {"dataset": DATASET, "dataset_record_index": RECORD, "layer": LAYER, "query_token": QUERY_TOKEN, "active_tokens": [0, 1], "query_head": HEAD},
            "source_packages": {
                "accepted_rope_q": artifact(ROPE_Q / "SHA256SUMS"),
                "immutable_head0": artifact(HEAD0 / "SHA256SUMS"),
                "immutable_head1": artifact(HEAD1 / "SHA256SUMS"),
                "merged_two_head_kv_write": artifact(KV_MERGE / "SHA256SUMS"),
                "new_minimal_token0_k": artifact(CAPTURE / "SHA256SUMS"),
            },
            "logical_interfaces": logical,
            "attention_score_scale_metadata": captured["attention_score_scale_metadata"],
            "compatibility": {
                "dataset_record_layer_exact_match": True,
                "model_revision_exact_match": True,
                "live_token1_k_projection_exact_match": True,
                "live_token1_k_rope_exact_match": True,
                "rope_q_reused_not_recaptured": True,
                "head0_head1_kv_packages_unmodified": True,
            },
            "counts": {"query_bytes": 64, "active_k_bytes": 128, "score_reference_bytes": 4, "softmax_reference_bytes": 4},
            "source_packages_modified": False,
        },
    )
    write_sha256s(MERGE)


def clamp_s16(value: int) -> tuple[int, bool]:
    if value > 32767:
        return 32767, True
    if value < -32768:
        return -32768, True
    return value, False


def score_reference(query: list[int], keys: list[list[int]], multiplier: int, right_shift: int) -> dict[str, Any]:
    accumulators = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    scaled = [base.round_shift_even(acc * multiplier, right_shift) for acc in accumulators]
    core = [clamp_s16(value) for value in scaled]
    row_max = max(scaled)
    centered_raw = [value - row_max for value in scaled]
    centered = [clamp_s16(value) for value in centered_raw]
    return {
        "accumulators": accumulators,
        "scaled_precenter": scaled,
        "core_scores": [value for value, _ in core],
        "core_saturation": [saturated for _, saturated in core],
        "scores": [value for value, _ in centered],
        "saturation": [saturated for _, saturated in centered],
    }


def pack(values: list[int], width: int) -> int:
    result = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        result |= (value & mask) << (lane * width)
    return result


def sv_hex(value: int, width: int) -> str:
    return f"{width}'h{value & ((1 << width) - 1):0{width // 4}x}"


def render_attention_score_vectors(query: list[int], keys: list[list[int]], reference: dict[str, Any], multiplier: int, right_shift: int) -> str:
    lines = [
        "// Generated for the immutable C4 record64 layer0 attention-score replay.",
        "localparam integer ATTN_SCORE_CASE_COUNT = 1;",
        "localparam integer ATTN_SCORE_HEAD_DIM = 64;",
        "localparam integer ATTN_SCORE_MAC_LANES = 1;",
        "localparam integer ATTN_SCORE_CONTEXT_MAX = 8;",
        "localparam integer ATTN_SCORE_BEATS_PER_VECTOR = 4;",
        "localparam [127:0] ATTN_SCORE_INVALID_NEGATIVE_MULTIPLIER = 128'h00000000000000000000000080000001;",
        "localparam [127:0] ATTN_SCORE_INVALID_RESERVED_BITS = 128'h80000000000000000000000000000001;",
        "reg [15:0] attn_score_context_count [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg signed [31:0] attn_score_multiplier [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [5:0] attn_score_right_shift [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg attn_score_expected_saturation [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg attn_score_expected_saturation_token [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg attn_score_expected_core_saturation_token [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg [31:0] attn_score_expected_acc [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg [16*8-1:0] attn_score_expected_word [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [16*8-1:0] attn_score_expected_core_word [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [8*16-1:0] attn_score_q_beats [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_BEATS_PER_VECTOR-1];",
        "reg [8*16-1:0] attn_score_k_beats [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX*ATTN_SCORE_BEATS_PER_VECTOR-1];",
        "initial begin",
        f"  attn_score_context_count[0] = 16'd{len(keys)};",
        f"  attn_score_multiplier[0] = 32'sd{multiplier};",
        f"  attn_score_right_shift[0] = 6'd{right_shift};",
        f"  attn_score_expected_saturation[0] = 1'b{int(any(reference['saturation']))};",
        f"  attn_score_expected_word[0] = {sv_hex(pack(reference['scores'] + [0] * 6, 16), 128)};",
        f"  attn_score_expected_core_word[0] = {sv_hex(pack(reference['core_scores'] + [0] * 6, 16), 128)};",
    ]
    for beat in range(4):
        lines.append(f"  attn_score_q_beats[{beat}] = {sv_hex(pack(query[beat*16:(beat+1)*16], 8), 128)};")
    padded_keys = keys + [[0] * HEAD_DIM for _ in range(8 - len(keys))]
    for token, key in enumerate(padded_keys):
        acc = reference["accumulators"][token] if token < len(keys) else 0
        saturated = reference["saturation"][token] if token < len(keys) else False
        core_saturated = reference["core_saturation"][token] if token < len(keys) else False
        lines.append(f"  attn_score_expected_acc[{token}] = 32'h{acc & 0xFFFFFFFF:08x};")
        lines.append(f"  attn_score_expected_saturation_token[{token}] = 1'b{int(saturated)};")
        lines.append(f"  attn_score_expected_core_saturation_token[{token}] = 1'b{int(core_saturated)};")
        for beat in range(4):
            flat = token * 4 + beat
            lines.append(f"  attn_score_k_beats[{flat}] = {sv_hex(pack(key[beat*16:(beat+1)*16], 8), 128)};")
    lines.append("end")
    return "\n".join(lines) + "\n"


def render_softmax_vectors(scores: list[int], probabilities: list[int]) -> tuple[str, dict[str, Any]]:
    result = reference_softmax(SoftmaxCase("c4_record64_layer0_head0_query1", scores))
    require(
        result.probabilities_q0_15[: len(probabilities)] == probabilities,
        "immutable softmax reference differs from independent reference",
    )
    require(
        not any(result.probabilities_q0_15[len(probabilities) :]),
        "independent softmax padded lanes are nonzero",
    )
    padded_scores = scores + [0] * (8 - len(scores))
    lines = [
        "// Generated for the immutable C4 record64 layer0 softmax replay.",
        "localparam integer SOFTMAX_CASE_COUNT = 1;",
        "localparam integer SOFTMAX_CONTEXT_MAX = 8;",
        "reg [15:0] softmax_context_count [0:SOFTMAX_CASE_COUNT-1];",
        "reg [16*8-1:0] softmax_score_word [0:SOFTMAX_CASE_COUNT-1];",
        "reg [16*8-1:0] softmax_expected_exp_word [0:SOFTMAX_CASE_COUNT-1];",
        "reg [18:0] softmax_expected_exp_sum [0:SOFTMAX_CASE_COUNT-1];",
        "reg [16*8-1:0] softmax_expected_word [0:SOFTMAX_CASE_COUNT-1];",
        "initial begin",
        f"  softmax_context_count[0] = 16'd{len(scores)};",
        f"  softmax_score_word[0] = {sv_hex(pack(padded_scores, 16), 128)};",
        f"  softmax_expected_exp_word[0] = {sv_hex(pack(result.exp_weights_q15, 16), 128)};",
        f"  softmax_expected_exp_sum[0] = 19'd{result.exp_sum_q15};",
        f"  softmax_expected_word[0] = {sv_hex(pack(probabilities + [0] * (8-len(probabilities)), 16), 128)};",
        "end",
    ]
    return "\n".join(lines) + "\n", {
        "exp_sum_q15": result.exp_sum_q15,
        "exp_weights_q15": result.exp_weights_q15,
        "max_score_q6_9": result.max_score_q6_9,
    }


def execution_testbench() -> str:
    text = (ROOT / "verification/tb/ace2_shell_tb.sv").read_text(encoding="utf-8")
    text = text.replace('    `include "../generated/attention_score_vectors.svh"', '    `include "c4_attention_score_vectors.svh"', 1)
    text = text.replace('    `include "../generated/softmax_vectors.svh"', '    `include "c4_softmax_vectors.svh"', 1)
    require('`include "c4_attention_score_vectors.svh"' in text, "attention-score include replacement failed")
    require('`include "c4_softmax_vectors.svh"' in text, "softmax include replacement failed")
    text = text.replace("    integer attn_score_only_mode;", "    integer attn_score_only_mode;\n    integer c4_ordered_replay_mode;", 1)
    text = text.replace(
        '        attn_score_only_mode = $test$plusargs("ATTN_SCORE_ONLY");',
        '        attn_score_only_mode = $test$plusargs("ATTN_SCORE_ONLY");\n'
        '        c4_ordered_replay_mode = $test$plusargs("C4_ORDERED_REPLAY");',
        1,
    )
    branch = (
        "        if (c4_ordered_replay_mode) begin\n"
        "            current_attn_metadata_mode = 0;\n"
        "            send_attn_score_cmd(0, 0, 16'h3e00);\n"
        "            wait_attn_score_done_and_compare(0, 0, 16'h3e00);\n"
        "            if (failures != 0) begin\n"
        "                $display(\"ACE2_C4_ATTENTION_SCORE_FAIL failures=%0d\", failures);\n"
        "                $fatal(1, \"ACE2_C4_ATTENTION_SCORE_FAIL\");\n"
        "            end\n"
        "            $display(\"ACE2_C4_ATTENTION_SCORE_PASS writes=%0d cycles=%0d score_word=%032x\", observed_count, last_command_cycles, observed_output[0]);\n"
        "            send_softmax_cmd(0, 0, 16'h3e01);\n"
        "            wait_softmax_done_and_compare(0, 0, 16'h3e01);\n"
        "            if (failures != 0) begin\n"
        "                $display(\"ACE2_C4_SOFTMAX_FAIL failures=%0d\", failures);\n"
        "                $fatal(1, \"ACE2_C4_SOFTMAX_FAIL\");\n"
        "            end\n"
        "            $display(\"ACE2_C4_SOFTMAX_PASS writes=%0d cycles=%0d probability_word=%032x\", observed_count, last_command_cycles, observed_output[0]);\n"
        "            $display(\"ACE2_C4_ORDERED_REPLAY_PASS operators=2\");\n"
        "            $finish;\n"
        "        end\n\n"
    )
    needle = "        if (qproj_stride_only_mode) begin"
    require(needle in text, "focused replay insertion point changed")
    text = text.replace(needle, branch + needle, 1)
    return text


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
    validate_manifest(MERGE / "SHA256SUMS", MERGE)
    merge = json.loads((MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    logical = merge["logical_interfaces"]
    query = packages["rope_q"]["rope"]["expected_output_s8"]
    key0 = tensor_i8(logical["k_cache_head0_token0_s8"])
    key1 = tensor_i8(logical["k_cache_head0_token1_s8"])
    expected_scores = tensor_i16(logical["score_reference_head0_query_token1_active_s16"])
    expected_probabilities = tensor_i16(logical["softmax_reference_head0_query_token1_active_q15"])
    metadata = merge["attention_score_scale_metadata"]
    reference = score_reference(query, [key0, key1], int(metadata["multiplier_s32"]), int(metadata["right_shift_u6"]))
    require(reference["scores"] == expected_scores, f"attention-score independent reference mismatch: {reference['scores']} != {expected_scores}")
    generated = REPLAY / "generated"
    generated.mkdir()
    (generated / "c4_attention_score_vectors.svh").write_text(
        render_attention_score_vectors(query, [key0, key1], reference, int(metadata["multiplier_s32"]), int(metadata["right_shift_u6"])),
        encoding="utf-8",
    )
    softmax_svh, softmax_reference = render_softmax_vectors(expected_scores, expected_probabilities)
    (generated / "c4_softmax_vectors.svh").write_text(softmax_svh, encoding="utf-8")
    shutil.copy2(SELF, REPLAY / "replay_source_at_execution.py")
    tb = REPLAY / "replay_tb_source_at_execution.sv"
    tb.write_text(execution_testbench(), encoding="utf-8")
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv")
    image = ROOT / "build/rtl-attention-score-token0-reference-batch-replay-v1/ordered.vvp"
    image.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [
        "iverilog", "-g2012", "-Wall", "-Irtl", "-Irtl/generated", "-Iverification/generated", "-Iverification/tb", f"-I{generated}",
        "-s", "ace2_shell_tb", "-o", str(image), *[str(path) for path in rtl_sources], str(tb),
    ]
    compiled = subprocess.run(compile_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "iverilog.log").write_text("command=" + " ".join(compile_command) + "\n" + compiled.stdout + compiled.stderr, encoding="utf-8")
    require(compiled.returncode == 0, "ordered replay compilation failed")
    replay_command = ["vvp", str(image), "+C4_ORDERED_REPLAY"]
    replayed = subprocess.run(replay_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "replay.stdout").write_text(replayed.stdout, encoding="utf-8")
    (REPLAY / "replay.stderr").write_text(replayed.stderr, encoding="utf-8")
    (REPLAY / "replay.log").write_text("command=" + " ".join(replay_command) + "\n" + replayed.stdout + replayed.stderr, encoding="utf-8")
    require(replayed.returncode == 0, "ordered replay execution failed")
    score_match = re.search(r"ACE2_C4_ATTENTION_SCORE_PASS writes=(\d+) cycles=(\d+) score_word=([0-9a-fA-F]+)", replayed.stdout)
    softmax_match = re.search(r"ACE2_C4_SOFTMAX_PASS writes=(\d+) cycles=(\d+) probability_word=([0-9a-fA-F]+)", replayed.stdout)
    require(score_match is not None, "attention-score replay pass marker missing")
    require(softmax_match is not None, "softmax replay pass marker missing")
    require("ACE2_C4_ORDERED_REPLAY_PASS operators=2" in replayed.stdout, "ordered replay completion marker missing")
    require("MISMATCH" not in replayed.stdout and "_FAIL" not in replayed.stdout, "ordered replay mismatch reported")
    require(int(score_match.group(1)) == 1 and int(softmax_match.group(1)) == 1, "ordered replay write count changed")
    available_v = logical["v_cache_head0_token1_s8"]
    boundary = {
        "schema_version": 1,
        "mission_id": "rtl-attention-score-token0-reference-batch-replay-v1",
        "operator": "layer_0.attention_value",
        "classification": "UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_RETAINED_ACTIVE_TOKEN_V_INCOMPLETE",
        "available_inputs": {
            "softmax_head0_query_token1_active_q15": logical["softmax_reference_head0_query_token1_active_q15"],
            "v_cache_head0_token1_s8": available_v,
        },
        "missing_fields": ["layer_0.attention_value.input.v_cache_head0_token0_s8[64]"],
        "scope_reason": "The authorized capture added only token-0 head-0 K. No retained immutable package contains token-0 head-0 V, and this mission forbids a broader or repeated capture.",
        "simulation_executed": False,
        "later_operators_attempted": False,
    }
    write_json(REPLAY / "boundary_probe.json", boundary)
    report = {
        "schema_version": 1,
        "mission_id": "rtl-attention-score-token0-reference-batch-replay-v1",
        "producer_role": "engineer",
        "review_status": "PENDING_FRESH_REVIEWER",
        "coordinate": {"dataset": DATASET, "dataset_record_index": RECORD, "layer": LAYER, "query_token": QUERY_TOKEN, "active_tokens": [0, 1], "query_head": HEAD},
        "capture": {
            "consumption_marker": artifact(CAPTURE / "capture_execution_consumed.json"),
            "evaluation_forward_count": 1,
            "persisted_tensor_count": 2,
            "rope_q_reused_not_recaptured": True,
            "head0_head1_kv_reused_not_recaptured": True,
            "witness": artifact(CAPTURE / "witness.json"),
            "merge_manifest": artifact(MERGE / "merge_manifest.json"),
        },
        "frozen_order": ORDER,
        "rtl_tree": base.rtl_tree(),
        "prior_maximal_prefix": ORDER[:7],
        "attempted_operators": [
            {
                "operator": "layer_0.attention_score",
                "status": "PASS_EXACT",
                "input_hashes": {"query_token1_head0": logical["query_head0_token1_s8"]["sha256"], "k_token0_head0": logical["k_cache_head0_token0_s8"]["sha256"], "k_token1_head0": logical["k_cache_head0_token1_s8"]["sha256"]},
                "reference_hashes": {"scores_active_s16": logical["score_reference_head0_query_token1_active_s16"]["sha256"]},
                "output_hashes": {"rtl_score_word_active_s16": sha256_bytes(b"".join((value & 0xFFFF).to_bytes(2, "little") for value in reference["scores"]))},
                "scale_metadata": metadata,
                "counts": {"active_tokens": CONTEXT, "dot_products": CONTEXT, "compared_scores": CONTEXT, "mismatches": 0},
                "saturation": {"reference_saturated_tokens": sum(reference["saturation"]), "rtl_result": "MATCH"},
                "cycle_handshake_latency": {"simulation_executed": True, "total_command_cycles": int(score_match.group(2)), "handshake_result": "PASS_NO_TIMEOUT_OR_DROPPED_WRITE", "latency_result": "PASS_BOUNDED_BY_MAINTAINED_SHELL_WATCHDOG"},
                "raw_log": artifact(REPLAY / "replay.log"),
            },
            {
                "operator": "layer_0.softmax",
                "status": "PASS_EXACT",
                "input_hashes": {"scores_active_s16": logical["score_reference_head0_query_token1_active_s16"]["sha256"]},
                "reference_hashes": {"probabilities_active_q15": logical["softmax_reference_head0_query_token1_active_q15"]["sha256"]},
                "counts": {"active_tokens": CONTEXT, "compared_probabilities": CONTEXT, "probability_sum_q15": sum(expected_probabilities), "mismatches": 0},
                "saturation": {"result": "NOT_APPLICABLE_SOFTMAX_OUTPUT_Q0_15"},
                "cycle_handshake_latency": {"simulation_executed": True, "total_command_cycles": int(softmax_match.group(2)), "handshake_result": "PASS_NO_TIMEOUT_OR_DROPPED_WRITE", "latency_result": "PASS_BOUNDED_BY_MAINTAINED_SHELL_WATCHDOG"},
                "independent_reference": softmax_reference,
                "raw_log": artifact(REPLAY / "replay.log"),
            },
            {
                "operator": "layer_0.attention_value",
                "status": boundary["classification"],
                "counts": {"active_tokens": CONTEXT, "retained_v_tokens": 1, "missing_input_count": 1, "mismatch_count": None},
                "cycle_handshake_latency": {"simulation_executed": False, "handshake_result": "NOT_EVALUATED_INTERFACE_UNAVAILABLE", "latency_result": "NOT_EVALUATED_INTERFACE_UNAVAILABLE"},
                "boundary_probe": artifact(REPLAY / "boundary_probe.json"),
            },
        ],
        "newly_passing_operators": ["layer_0.attention_score", "layer_0.softmax"],
        "maximal_passing_prefix": ORDER[:9],
        "first_failing_or_unsupported_operator": {"operator": "layer_0.attention_value", "classification": boundary["classification"], "boundary_probe": artifact(REPLAY / "boundary_probe.json")},
        "batch_result": "STOPPED_AT_FIRST_PRECISELY_UNAVAILABLE_EXECUTABLE_INTERFACE",
        "later_operators_attempted": False,
        "existing_current_tree_ppa": {"reused_not_rerun": True, "sha256_manifest": artifact(PPA_SUMS)},
        "protected_state_pre_post_exact_match": protected_state() == json.loads((CAPTURE / "preflight.json").read_text(encoding="utf-8"))["protected_state"],
        "forbidden_flows_run": [],
        "state_mutations": {"rtl": False, "frontier_ledger_manifest_traceability_pipeline_seals": False, "synthesis_opensta_ppa": False, "baseline_candidate_benchmark_scale32": False},
        "review_requirement": "Fresh Reviewer must independently accept or reject this maximal-prefix result.",
    }
    require(report["protected_state_pre_post_exact_match"], "protected state changed during capture/replay")
    write_json(REPLAY / "ordered_prefix_report.json", report)
    (REPLAY / "batch_replay.log").write_text(
        "ACE2_ATTENTION_SCORE_TOKEN0_ORDERED_BATCH_REPLAY\n"
        f"rtl_tree_sha256={EXPECTED_RTL_TREE}\n"
        "coordinate=c4_en_512:record64:layer0:query1:active_tokens0-1:head0\n"
        "layer_0.attention_score=PASS_EXACT active_scores=2 mismatches=0\n"
        "layer_0.softmax=PASS_EXACT active_probabilities=2 mismatches=0\n"
        "layer_0.attention_value=UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_RETAINED_ACTIVE_TOKEN_V_INCOMPLETE\n"
        "maximal_passing_operator=layer_0.softmax\n"
        "later_operators_attempted=false\n"
        "synthesis_opensta_ppa_executed=false\n",
        encoding="utf-8",
    )
    write_sha256s(REPLAY)
    print("ACE2_ATTENTION_SCORE_TOKEN0_BATCH_REPLAY_BOUNDARY maximal_prefix=layer_0.softmax first_boundary=layer_0.attention_value mismatches=0")


def launch_capture_and_replay() -> None:
    preflight = validate_preflight()
    require(not (CAPTURE / "capture_execution_consumed.json").exists(), "capture already consumed")
    require(not (CAPTURE / "witness.json").exists(), "capture witness already exists")
    require(not MERGE.exists(), "merge output already exists")
    require(not REPLAY.exists(), "replay output already exists")
    source_archive = CAPTURE / "capture_source_at_execution.py"
    shutil.copy2(SELF, source_archive)
    require(sha256_file(source_archive) == preflight["source"]["sha256"], "source changed after preflight")
    stdout_handle = (CAPTURE / "capture.stdout").open("x", encoding="utf-8")
    stderr_handle = (CAPTURE / "capture.stderr").open("x", encoding="utf-8")
    marker = CAPTURE / "capture_execution_consumed.json"
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "status": "CONSUMED_BEFORE_SINGLE_WORKER_LAUNCH",
                "consumed_at_utc": utc_now(),
                "command": [sys.executable, relative(SELF), "_capture_worker"],
                "coordinate": {"dataset": DATASET, "dataset_record_index": RECORD, "head": HEAD, "layer": LAYER, "token": CAPTURE_TOKEN, "query_token": QUERY_TOKEN},
                "authorized_persisted_tensors": ["k_projection_head0_token0_s8", "k_cache_head0_token0_s8"],
                "source_archive": artifact(source_archive),
                "raw_stdout": relative(CAPTURE / "capture.stdout"),
                "raw_stderr": relative(CAPTURE / "capture.stderr"),
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
    worker = subprocess.run([sys.executable, str(SELF), "_capture_worker"], cwd=ROOT, env=env, stdout=stdout_handle, stderr=stderr_handle, check=False)
    stdout_handle.close()
    stderr_handle.close()
    write_json(
        CAPTURE / "capture_worker_status.json",
        {"schema_version": 1, "completed_at_utc": utc_now(), "returncode": worker.returncode, "rerun_permitted": False, "stdout": artifact(CAPTURE / "capture.stdout"), "stderr": artifact(CAPTURE / "capture.stderr")},
    )
    require(worker.returncode == 0, "single capture worker failed; authorization remains consumed")
    write_sha256s(CAPTURE)
    merge_packages()
    run_replay()


def validate_completed_capture() -> dict[str, Any]:
    preflight = json.loads((CAPTURE / "preflight.json").read_text(encoding="utf-8"))
    require(preflight["status"] == "PASS_MODEL_EXECUTION_NOT_CONSUMED", "preflight status changed")
    require(preflight["model_execution_count"] == 0, "preflight consumed model execution")
    require(
        artifact(CAPTURE / "preflight_source_at_execution.py") == preflight["source_archive"],
        "preflight source archive changed",
    )
    validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE)
    marker = json.loads((CAPTURE / "capture_execution_consumed.json").read_text(encoding="utf-8"))
    witness = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    require(marker["rerun_permitted"] is False, "consumption marker permits rerun")
    require(witness["capture"]["evaluation_forward_count"] == 1, "capture was not exactly once")
    require(witness["capture"]["persisted_tensor_count"] == 2, "capture tensor scope changed")
    require(set(witness["tensor_manifest"]) == {"k_projection_head0_token0_s8", "k_cache_head0_token0_s8"}, "capture contains unauthorized tensors")
    require(witness["capture"]["rope_q_recaptured"] is False, "RoPE-Q was recaptured")
    require(protected_state() == preflight["protected_state"], "protected state changed after capture")
    require(base.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed after capture")
    accepted_packages()
    return preflight


def resume_replay() -> None:
    validate_completed_capture()
    validate_manifest(MERGE / "SHA256SUMS", MERGE)
    run_replay()


def run_check() -> None:
    preflight = validate_completed_capture()
    require(preflight["model_execution_count"] == 0, "preflight execution count changed")
    validate_manifest(MERGE / "SHA256SUMS", MERGE)
    validate_manifest(REPLAY / "SHA256SUMS", REPLAY)
    marker = json.loads((CAPTURE / "capture_execution_consumed.json").read_text(encoding="utf-8"))
    witness = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    merge = json.loads((MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((REPLAY / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    require(marker["rerun_permitted"] is False, "consumption marker permits rerun")
    require((CAPTURE / "capture_execution_consumed.json").stat().st_mode & 0o222 == 0, "consumption marker is writable")
    require(witness["capture"]["evaluation_forward_count"] == 1, "capture was not exactly once")
    require(witness["capture"]["persisted_tensor_count"] == 2, "capture tensor scope changed")
    require(set(witness["tensor_manifest"]) == {"k_projection_head0_token0_s8", "k_cache_head0_token0_s8"}, "capture contains unauthorized tensors")
    require(witness["capture"]["rope_q_recaptured"] is False, "RoPE-Q was recaptured")
    require(merge["source_packages_modified"] is False, "merge modified a source package")
    require(report["newly_passing_operators"] == ["layer_0.attention_score", "layer_0.softmax"], "passing operators changed")
    require(report["maximal_passing_prefix"] == ORDER[:9], "maximal prefix changed")
    require(report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.attention_value", "first boundary changed")
    require(report["later_operators_attempted"] is False, "replay continued beyond first boundary")
    require(report["state_mutations"] == {"rtl": False, "frontier_ledger_manifest_traceability_pipeline_seals": False, "synthesis_opensta_ppa": False, "baseline_candidate_benchmark_scale32": False}, "forbidden mutation record changed")
    require(protected_state() == preflight["protected_state"], "protected state changed")
    require(base.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed after replay")
    print(
        "ACE2_ATTENTION_SCORE_TOKEN0_CAPTURE_BATCH_REPLAY_CHECK_PASS "
        f"capture_sha256={sha256_file(CAPTURE / 'witness.json')} "
        f"merge_sha256={sha256_file(MERGE / 'merge_manifest.json')} "
        f"report_sha256={sha256_file(REPLAY / 'ordered_prefix_report.json')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "capture-and-replay", "resume-replay", "check", "_capture_worker"))
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
