#!/usr/bin/env python3
"""Capture one retained layer-0 coordinate and replay to the first RTL boundary."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ace2_full_model_fixed_point as fixed
import capture_replay_c4_rope_q as rope_q_core
from ace2_full_model_fixed_point import (
    FixedAttention,
    FixedDecoderLayer,
    FixedRMSNorm,
    W4A8Linear,
    calibrate,
    hash_records,
    hash_token_sequences,
    load_contracts,
    quantize_int8,
    replace_fixed_operators,
    replace_linears,
    seed_everything,
    selected_texts,
    tokenize_prompts,
    validate_runtime,
)
from ace2_projection_reference import (
    ProjectionCase,
    pack_meta,
    reference_projection,
    round_shift_even,
    saturate_int8,
    to_sint,
)
from ace2_rope_reference import RopeCase, reference_rope


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-layer0-minimal-reference-capture-batch-replay-v1/mission.json"
)
CAPTURE = ROOT / "evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2"
REPLAY = ROOT / "evidence/verification/rtl-layer0-minimal-reference-capture-batch-replay-v1"
PRIOR = ROOT / "evidence/diagnostics/c4-rope-q-token1-capture-replay-20260803-v2"
PRIOR_SUMS = PRIOR / "SHA256SUMS"
PRIOR_WITNESS = PRIOR / "witness.json"
PRIOR_PREFLIGHT = PRIOR / "preflight.json"
PRIOR_BATCH = ROOT / "evidence/verification/rtl-current-tree-ordered-operator-batch-replay-v1"
PPA_SUMS = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/SHA256SUMS"

EXPECTED_RTL_TREE = "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be"
EXPECTED_PRIOR_SUMS = "9d1640be7cc126c689b53f2a0009069af0fdd02050cf4114659d0e8a000893ac"
EXPECTED_PRIOR_WITNESS = "2ad63d87d0fd53664e606faed32e60d5f5e69c94903f07f93de6ae1822657404"
EXPECTED_PRIOR_PREFLIGHT = "d598802bef8a60b3f375ddf2d1f6bf72c8b876b01a4959f4c413a332b120cf07"
EXPECTED_PPA_SUMS = "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4"
EXPECTED_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_RECORD_SHA256 = "a5c1e98fba3be31fc775446d53772b726abd2b6ef1f684171be1262db7df2acc"
EXPECTED_TOKEN_SEQUENCE_SHA256 = "5614de646c94a5b4b457142971d02c74ccc741b304c95b5c7c44a5c36fa6c436"
DATASET = "c4_en_512"
CALIBRATION = "c4_calibration"
TOKEN = 1
HEAD = 0
HEAD_DIM = 64
HIDDEN = 896
ORDER = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
    "layer_0.rope_q",
    "layer_0.rope_k",
    "layer_0.kv_write",
    "layer_0.attention_score",
    "layer_0.softmax",
    "layer_0.attention_value",
    "layer_0.o_proj",
    "layer_0.attention_residual_add",
    "layer_0.post_attention_rmsnorm",
    "layer_0.mlp_gate_proj",
    "layer_0.mlp_up_proj",
    "layer_0.silu_gate",
    "layer_0.mlp_down_proj",
    "layer_0.mlp_residual_add",
]


class Layer0CaptureComplete(RuntimeError):
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


def validate_manifest(path: Path, base: Path) -> int:
    count = 0
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row:
            continue
        expected, name = row.split("  ", 1)
        member = Path(name)
        member = member if member.is_absolute() else base / member
        require(member.is_file(), f"missing manifest member: {name}")
        require(sha256_file(member) == expected, f"changed manifest member: {name}")
        count += 1
    return count


def rtl_tree() -> dict[str, Any]:
    rows = []
    for path in sorted(ROOT.glob("rtl/**/*.sv")):
        rows.append((sha256_file(path), relative(path)))
    payload = "".join(f"{digest}  {name}\n" for digest, name in rows).encode()
    digest = sha256_bytes(payload)
    require(len(rows) == 23, f"RTL file count changed: {len(rows)}")
    require(digest == EXPECTED_RTL_TREE, f"RTL tree changed: {digest}")
    return {
        "file_count": len(rows),
        "sha256": digest,
        "hash_method": "path-sorted sha256 manifest over rtl/**/*.sv",
        "source_hashes": [{"path": name, "sha256": value} for value, name in rows],
    }


def tensor_bytes(tensor: Tensor) -> tuple[bytes, str]:
    value = tensor.detach().cpu().contiguous()
    logical_dtype = str(value.dtype).removeprefix("torch.")
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes(order="C")
    elif value.dtype == torch.bool:
        raw = value.to(torch.uint8).numpy().tobytes(order="C")
    else:
        raw = value.numpy().tobytes(order="C")
    return raw, logical_dtype


def write_tensor(base: Path, name: str, tensor: Tensor) -> dict[str, Any]:
    raw, dtype = tensor_bytes(tensor)
    path = base / "tensors" / f"{name}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    value = tensor.detach().cpu()
    numeric = value.to(torch.float64) if value.numel() else value
    return {
        "bytes": len(raw),
        "dtype": dtype,
        "elements": value.numel(),
        "max": None if not value.numel() else float(numeric.max()),
        "min": None if not value.numel() else float(numeric.min()),
        "path": relative(path),
        "sha256": sha256_bytes(raw),
        "shape": list(value.shape),
    }


TORCH_DTYPES = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
}


def read_tensor(record: dict[str, Any]) -> Tensor:
    dtype = record["dtype"]
    require(dtype in TORCH_DTYPES, f"unsupported replay tensor dtype: {dtype}")
    raw = (ROOT / record["path"]).read_bytes()
    require(sha256_bytes(raw) == record["sha256"], f"tensor changed: {record['path']}")
    value = torch.frombuffer(bytearray(raw), dtype=TORCH_DTYPES[dtype]).clone()
    return value.reshape(record["shape"])


def passthrough_wrapper(
    original: Callable[..., Any],
    recorder: Callable[[tuple[Any, ...], dict[str, Any], Any], None],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        recorder(args, kwargs, result)
        return result

    return wrapped


def extraction_only_selftest() -> dict[str, Any]:
    calls = 0
    captured: list[Tensor] = []
    source = torch.tensor([3, -4], dtype=torch.int8)
    output = (torch.tensor([8, 9], dtype=torch.int16), 7)

    def original(value: Tensor) -> tuple[Tensor, int]:
        nonlocal calls
        calls += 1
        return output

    def record(args: tuple[Any, ...], _kwargs: dict[str, Any], result: Any) -> None:
        captured.append(args[0].detach().cpu().clone())
        captured.append(result[0].detach().cpu().clone())

    wrapped = passthrough_wrapper(original, record)
    returned = wrapped(source)
    require(returned is output, "extraction wrapper replaced the original output object")
    require(calls == 1, "extraction wrapper changed call count")
    require(torch.equal(source, torch.tensor([3, -4], dtype=torch.int8)), "wrapper changed input")
    require(torch.equal(captured[0], source), "wrapper input capture changed values")
    require(torch.equal(captured[1], output[0]), "wrapper output capture changed values")
    return {
        "call_count": calls,
        "input_unchanged": True,
        "output_object_identity_preserved": True,
        "output_values_unchanged": True,
        "recording_operation": "detach_cpu_clone_after_original_return",
    }


def cached_model_bindings(repository: str, revision: str) -> list[dict[str, Any]]:
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    ]
    resolved: dict[str, Path] = {}
    for name in names:
        value = cached_file(
            repository,
            name,
            revision=revision,
            _raise_exceptions_for_gated_repo=True,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=True,
        )
        if value:
            resolved[name] = Path(value)
    if "model.safetensors.index.json" in resolved:
        index = json.loads(resolved["model.safetensors.index.json"].read_text(encoding="utf-8"))
        for shard in sorted(set(index["weight_map"].values())):
            value = cached_file(repository, shard, revision=revision)
            resolved[shard] = Path(value)
    require("config.json" in resolved, "pinned model config is unavailable")
    require("tokenizer.json" in resolved, "pinned tokenizer is unavailable")
    require(
        "model.safetensors" in resolved or "model.safetensors.index.json" in resolved,
        "pinned model weights are unavailable",
    )
    return [
        {"bytes": path.stat().st_size, "filename": name, "sha256": sha256_file(path)}
        for name, path in sorted(resolved.items())
    ]


def live_input_binding() -> tuple[dict[str, Any], Tensor, Tensor, dict[str, Any], dict[str, Any]]:
    manifest, config, _ = load_contracts(require_rtl_binding=False)
    versions = validate_runtime(config)
    seed_everything(config)
    model_spec = manifest["model"]
    require(model_spec["repository"] == "Qwen/Qwen2.5-0.5B", "model repository changed")
    require(model_spec["revision"] == EXPECTED_REVISION, "model revision changed")
    calibration_spec = manifest["datasets"][CALIBRATION]
    evaluation_spec = manifest["datasets"][DATASET]
    calibration_text = selected_texts(calibration_spec, limit=1)
    evaluation_text = selected_texts(evaluation_spec, limit=1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"], revision=model_spec["revision"]
    )
    calibration_prompt = tokenize_prompts(
        tokenizer, calibration_text, calibration_spec["token_limit"]
    )[0]
    evaluation_prompt = tokenize_prompts(
        tokenizer, evaluation_text, evaluation_spec["token_limit"]
    )[0]
    evaluation_record_hash, evaluation_record_count = hash_records(evaluation_text)
    evaluation_token_hash, evaluation_sequence_count, evaluation_token_count = hash_token_sequences(
        [evaluation_prompt]
    )
    calibration_record_hash, calibration_record_count = hash_records(calibration_text)
    calibration_token_hash, calibration_sequence_count, calibration_token_count = hash_token_sequences(
        [calibration_prompt]
    )
    require(evaluation_record_count == 1, "evaluation record count changed")
    require(evaluation_record_hash == EXPECTED_RECORD_SHA256, "evaluation record hash changed")
    require(
        evaluation_token_hash == EXPECTED_TOKEN_SEQUENCE_SHA256,
        "evaluation token sequence hash changed",
    )
    require(evaluation_prompt.shape[1] > TOKEN, "evaluation prompt is too short")
    binding = {
        "calibration_dependency_not_captured": {
            "record_count": calibration_record_count,
            "record_sha256": calibration_record_hash,
            "token_count": calibration_token_count,
            "token_sequence_count": calibration_sequence_count,
            "token_sequence_sha256": calibration_token_hash,
        },
        "evaluation_witness": {
            "dataset": DATASET,
            "dataset_record_index": evaluation_spec["indices"]["start"],
            "record_count": evaluation_record_count,
            "record_sha256": evaluation_record_hash,
            "token": TOKEN,
            "token_count": evaluation_token_count,
            "token_id": int(evaluation_prompt[0, TOKEN]),
            "token_sequence_count": evaluation_sequence_count,
            "token_sequence_sha256": evaluation_token_hash,
        },
    }
    return binding, calibration_prompt, evaluation_prompt, manifest, versions


def preflight_payload(source_archive: Path) -> dict[str, Any]:
    require(sha256_file(PRIOR_SUMS) == EXPECTED_PRIOR_SUMS, "accepted RoPE-Q aggregate changed")
    prior_members = validate_manifest(PRIOR_SUMS, PRIOR)
    require(sha256_file(PRIOR_WITNESS) == EXPECTED_PRIOR_WITNESS, "accepted witness changed")
    require(sha256_file(PRIOR_PREFLIGHT) == EXPECTED_PRIOR_PREFLIGHT, "accepted trig preflight changed")
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SUMS, "accepted canonical PPA aggregate changed")
    prior = json.loads(PRIOR_WITNESS.read_text(encoding="utf-8"))
    trig = json.loads(PRIOR_PREFLIGHT.read_text(encoding="utf-8"))
    gate = trig["two_source_bf16_preflight"]
    require(gate["lane_count"] == 64, "accepted trig lane count changed")
    require(gate["exact_lane_equality"], "accepted trig preflight is not exact")
    require(not gate["cosine_mismatch_lane_indices"], "accepted cosine mismatches appeared")
    require(not gate["sine_mismatch_lane_indices"], "accepted sine mismatches appeared")
    coordinate = prior["coordinate"]
    require(coordinate["dataset_record_index"] == 64, "retained record changed")
    require(coordinate["layer"] == 0 and coordinate["token"] == TOKEN, "retained layer/token changed")
    require(coordinate["head"] == HEAD, "retained head changed")
    input_binding, _calibration, _evaluation, manifest, versions = live_input_binding()
    source_bindings = prior["source_bindings"]
    for name in (
        "benchmark/quality/PROMPT_MANIFEST.json",
        "benchmark/quality/QUALITY_CONFIG.json",
        "tools/ace2_full_model_fixed_point.py",
        "tools/ace2_projection_reference.py",
        "tools/ace2_rope_reference.py",
    ):
        expected = source_bindings[name]["sha256"]
        require(sha256_file(ROOT / name) == expected, f"pinned source changed: {name}")
    tools = {name: shutil.which(name) for name in ("iverilog", "vvp")}
    require(all(tools.values()), f"required RTL tools unavailable: {tools}")
    model_spec = manifest["model"]
    return {
        "schema_version": 1,
        "mission_id": "rtl-layer0-minimal-reference-capture-batch-replay-v1",
        "status": "PASS_MODEL_EXECUTION_NOT_CONSUMED",
        "completed_at_utc": utc_now(),
        "model_execution_count": 0,
        "coordinate": {
            "dataset": DATASET,
            "dataset_record_index": 64,
            "layer": 0,
            "token": TOKEN,
            "head": HEAD,
        },
        "input_binding": input_binding,
        "model": {
            "repository": model_spec["repository"],
            "revision": model_spec["revision"],
            "cached_files": cached_model_bindings(model_spec["repository"], model_spec["revision"]),
        },
        "accepted_two_source_bf16_trig_preflight": {
            "artifact": artifact(PRIOR_PREFLIGHT),
            "cosine_mismatches": 0,
            "exact_lane_equality": True,
            "lane_count": 64,
            "sine_mismatches": 0,
        },
        "accepted_rope_q_capture": {
            "member_count": prior_members,
            "sha256_manifest": artifact(PRIOR_SUMS),
            "witness": artifact(PRIOR_WITNESS),
        },
        "extraction_only_proof": extraction_only_selftest(),
        "hook_contract": {
            "forward_hooks_return": "None",
            "method_and_function_wrappers": "call original exactly once, clone only after return, return identical output object",
            "scope_stop": "raise Layer0CaptureComplete only after layer-0 output is captured",
        },
        "mission": artifact(MISSION, "handoff:rtl-layer0-minimal-reference-capture-batch-replay-v1/mission.json"),
        "runtime": {
            "packages": versions,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "rtl_tree": rtl_tree(),
        "source": artifact(SELF),
        "source_archive": artifact(source_archive),
        "tool_lookup": {name: Path(value).name for name, value in tools.items()},
        "forbidden_flows_run": [],
    }


def run_preflight() -> None:
    require(not CAPTURE.exists(), f"capture directory already exists: {CAPTURE}")
    require(not REPLAY.exists(), f"replay directory already exists: {REPLAY}")
    CAPTURE.mkdir(parents=True)
    source_archive = CAPTURE / "preflight_source_at_execution.py"
    shutil.copy2(SELF, source_archive)
    payload = preflight_payload(source_archive)
    write_json(CAPTURE / "preflight.json", payload)
    (CAPTURE / "preflight.log").write_text(
        "ACE2_LAYER0_CAPTURE_PREFLIGHT_PASS\n"
        "model_execution_count=0\n"
        "coordinate=c4_en_512:record64:layer0:token1:head0\n"
        "trig_lanes=64\n"
        "trig_mismatches=0\n"
        "extraction_only=true\n",
        encoding="utf-8",
    )
    print("ACE2_LAYER0_CAPTURE_PREFLIGHT_PASS model_execution_count=0")


def validate_preflight() -> dict[str, Any]:
    path = CAPTURE / "preflight.json"
    require(path.is_file(), "preflight artifact is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["status"] == "PASS_MODEL_EXECUTION_NOT_CONSUMED", "preflight status changed")
    require(payload["model_execution_count"] == 0, "preflight consumed model execution")
    require(artifact(CAPTURE / "preflight_source_at_execution.py") == payload["source_archive"], "preflight source archive changed")
    require(sha256_file(SELF) == payload["source"]["sha256"], "capture tool changed after preflight")
    require(rtl_tree()["sha256"] == payload["rtl_tree"]["sha256"], "RTL changed after preflight")
    require(sha256_file(PRIOR_SUMS) == EXPECTED_PRIOR_SUMS, "accepted preflight dependency changed")
    return payload


class ExtractionTrace:
    def __init__(self, layer: FixedDecoderLayer) -> None:
        self.layer = layer
        self.methods: dict[str, dict[str, Tensor]] = {}
        self.calls: dict[str, list[dict[str, Any]]] = {}
        self.norms: dict[str, Tensor] = {}
        self.layer_input: Tensor | None = None
        self.position_embeddings: tuple[Tensor, Tensor] | None = None
        self.layer_output: Tensor | None = None
        self._method_overrides: list[tuple[nn.Module, str]] = []
        self._global_originals: list[tuple[str, Callable[..., Any]]] = []
        self._hooks: list[Any] = []

    @staticmethod
    def _clone(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, tuple):
            return tuple(ExtractionTrace._clone(item) for item in value)
        if isinstance(value, list):
            return [ExtractionTrace._clone(item) for item in value]
        return value

    def _override_method(self, module: nn.Module, method: str, name: str) -> None:
        original = getattr(module, method)

        def record(args: tuple[Any, ...], _kwargs: dict[str, Any], result: Any) -> None:
            self.methods[name] = {
                "input": self._clone(args[0]),
                "output": self._clone(result),
            }

        wrapped = passthrough_wrapper(original, record)
        module.__dict__[method] = types.MethodType(lambda _self, *a, **k: wrapped(*a, **k), module)
        self._method_overrides.append((module, method))

    def _override_global(self, name: str) -> None:
        original = getattr(fixed, name)

        def record(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            self.calls.setdefault(name, []).append(
                {
                    "args": self._clone(args),
                    "kwargs": self._clone(kwargs),
                    "result": self._clone(result),
                }
            )

        setattr(fixed, name, passthrough_wrapper(original, record))
        self._global_originals.append((name, original))

    def install(self) -> None:
        layer = self.layer

        def layer_pre(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            positions = kwargs.get("position_embeddings")
            require(isinstance(hidden, Tensor), "layer-0 hidden input was not captured")
            require(isinstance(positions, tuple) and len(positions) == 2, "position embeddings missing")
            self.layer_input = hidden.detach().cpu().clone()
            self.position_embeddings = tuple(value.detach().cpu().clone() for value in positions)

        def layer_post(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> None:
            self.layer_output = output.detach().cpu().clone()
            raise Layer0CaptureComplete

        def norm_hook(name: str) -> Callable[..., None]:
            def hook(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> None:
                self.norms[name] = output.detach().cpu().clone()

            return hook

        self._hooks.extend(
            [
                layer.register_forward_pre_hook(layer_pre, with_kwargs=True),
                layer.register_forward_hook(layer_post),
                layer.input_layernorm.register_forward_hook(norm_hook("input_rmsnorm")),
                layer.post_attention_layernorm.register_forward_hook(norm_hook("post_attention_rmsnorm")),
            ]
        )
        for module, method, name in (
            (layer.self_attn.q_proj, "forward_hardware_input", "q_proj"),
            (layer.self_attn.k_proj, "forward_hardware_input", "k_proj"),
            (layer.self_attn.v_proj, "forward_hardware_input", "v_proj"),
            (layer.self_attn.o_proj, "forward_hardware_input", "o_proj"),
            (layer.mlp.gate_proj, "forward_hardware_input", "mlp_gate_proj"),
            (layer.mlp.up_proj, "forward_hardware_input", "mlp_up_proj"),
            (layer.mlp.down_proj, "accumulator_quantized", "mlp_down_accumulator"),
            (layer.mlp.down_proj, "requantize_accumulator", "mlp_down_requantize"),
        ):
            self._override_method(module, method, name)
        for name in (
            "fixed_rope_raw_with_saturation",
            "repeat_kv",
            "fixed_attention_scores_raw",
            "fixed_softmax_raw",
            "fixed_attention_value_raw",
            "fixed_residual_add",
            "fixed_silu_gate_raw",
        ):
            self._override_global(name)

    def restore(self) -> None:
        for hook in self._hooks:
            hook.remove()
        for module, method in self._method_overrides:
            module.__dict__.pop(method, None)
        for name, original in self._global_originals:
            setattr(fixed, name, original)


def projection_metadata(module: W4A8Linear, channel_start: int, channel_stop: int, qinput: Tensor) -> dict[str, Any]:
    weights = module.qweight[channel_start:channel_stop].detach().cpu().to(torch.int64)
    multipliers = module.multiplier[channel_start:channel_stop].detach().cpu().to(torch.int64)
    shifts = module.right_shift[channel_start:channel_stop].detach().cpu().to(torch.int64)
    biases = (
        torch.zeros(channel_stop - channel_start, dtype=torch.int64)
        if module.bias_accumulator is None
        else module.bias_accumulator[channel_start:channel_stop].detach().cpu().to(torch.int64)
    )
    dots: list[int] = []
    biased: list[int] = []
    saturations: list[int] = []
    metadata: list[dict[str, Any]] = []
    activations = qinput.to(torch.int64).tolist()
    for index in range(channel_stop - channel_start):
        dot = sum(a * int(w) for a, w in zip(activations, weights[index].tolist(), strict=True))
        with_bias = dot + int(biases[index])
        rounded = round_shift_even(with_bias * int(multipliers[index]), int(shifts[index]))
        _clipped, saturated = saturate_int8(rounded)
        packed = pack_meta(int(multipliers[index]), int(shifts[index]), 0, int(biases[index]))
        decoded = {
            "bias_accumulator_s32": to_sint((packed >> 48) & 0xFFFFFFFF, 32),
            "multiplier_s32": to_sint(packed & 0xFFFFFFFF, 32),
            "output_zero_point_s8": to_sint((packed >> 40) & 0xFF, 8),
            "right_shift_u6": (packed >> 32) & 0x3F,
        }
        dots.append(dot)
        biased.append(with_bias)
        saturations.append(int(saturated))
        metadata.append({"packed_u128": packed, **decoded})
    return {
        "weights": weights,
        "multipliers": multipliers,
        "shifts": shifts,
        "biases": biases,
        "dots": torch.tensor(dots, dtype=torch.int64),
        "biased": torch.tensor(biased, dtype=torch.int64),
        "saturations": torch.tensor(saturations, dtype=torch.uint8),
        "metadata": metadata,
    }


def execute_capture_worker() -> None:
    preflight = validate_preflight()
    marker = CAPTURE / "capture_execution_consumed.json"
    source_archive = CAPTURE / "capture_source_at_execution.py"
    require(marker.is_file(), "capture consumption marker is missing")
    require(source_archive.is_file(), "capture source archive is missing")
    require(sha256_file(source_archive) == preflight["source"]["sha256"], "capture source differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("capture requires CUDA_VISIBLE_DEVICES empty or -1")
    input_binding, calibration_prompt, evaluation_prompt, manifest, versions = live_input_binding()
    require(input_binding == preflight["input_binding"], "live input changed after preflight")
    config = load_contracts(require_rtl_binding=False)[1]
    seed_everything(config)
    model_spec = manifest["model"]
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "model is not CPU-only")
    require(getattr(model.config, "_commit_hash", None) == EXPECTED_REVISION, "resolved revision changed")
    ranges, operator_ranges = calibrate(model, [calibration_prompt])
    replace_linears(model, ranges, rope_diagnostic_mechanism=None)
    replace_fixed_operators(model, operator_ranges, rope_diagnostic_mechanism=None)
    layer = model.model.layers[0]
    require(isinstance(layer, FixedDecoderLayer), "layer 0 is not FixedDecoderLayer")
    require(isinstance(layer.self_attn, FixedAttention), "layer 0 attention is not fixed")
    require(isinstance(layer.input_layernorm, FixedRMSNorm), "layer 0 RMSNorm is not fixed")
    trace = ExtractionTrace(layer)
    trace.install()
    seed_everything(config)
    evaluation_forward_count = 0
    try:
        with torch.inference_mode():
            evaluation_forward_count += 1
            model(input_ids=evaluation_prompt, use_cache=False)
    except Layer0CaptureComplete:
        pass
    finally:
        trace.restore()
    require(evaluation_forward_count == 1, "retained-coordinate forward count changed")
    require(trace.layer_output is not None, "layer-0 output was not captured")
    require(trace.position_embeddings is not None, "position embeddings were not captured")
    require(len(trace.calls.get("fixed_rope_raw_with_saturation", [])) == 2, "expected Q and K RoPE calls")
    require(len(trace.calls.get("fixed_attention_scores_raw", [])) == 1, "attention score call missing")
    require(len(trace.calls.get("fixed_softmax_raw", [])) == 1, "softmax call missing")
    require(len(trace.calls.get("fixed_attention_value_raw", [])) == 1, "attention value call missing")
    require(len(trace.calls.get("fixed_residual_add", [])) == 2, "residual calls changed")
    require(len(trace.calls.get("fixed_silu_gate_raw", [])) == 1, "SiLU gate call missing")

    tensors: dict[str, dict[str, Any]] = {}

    def retain(name: str, value: Tensor) -> None:
        tensors[name] = write_tensor(CAPTURE, name, value)

    input_norm = trace.norms["input_rmsnorm"]
    require(torch.equal(input_norm, torch.round(input_norm)), "input RMSNorm is not integer-valued")
    k_input = input_norm[0, TOKEN].to(torch.int8)
    k_all = trace.methods["k_proj"]["output"]
    k_head = k_all[0, TOKEN, :HEAD_DIM].to(torch.int8)
    k_meta = projection_metadata(layer.self_attn.k_proj, 0, HEAD_DIM, k_input)
    projection_case = ProjectionCase(
        name="frozen_c4_layer0_k_head0_token1",
        rows=1,
        reduction_size=HIDDEN,
        activations=[k_input.to(torch.int64).tolist()],
        weights=k_meta["weights"].tolist(),
        multipliers=k_meta["multipliers"].tolist(),
        right_shifts=k_meta["shifts"].tolist(),
        output_zero_points=[0] * HEAD_DIM,
        bias_accumulators=k_meta["biases"].tolist(),
    )
    projection_reference = reference_projection(projection_case)
    require(projection_reference.outputs[0] == k_head.to(torch.int64).tolist(), "K projection reference mismatch")
    cos, sin = trace.position_embeddings
    cos_q15 = torch.round(cos[0, TOKEN].to(torch.float64) * 32767.0).clamp(-32768, 32767).to(torch.int16)
    sin_q15 = torch.round(sin[0, TOKEN].to(torch.float64) * 32767.0).clamp(-32768, 32767).to(torch.int16)
    prior_preflight = json.loads(PRIOR_PREFLIGHT.read_text(encoding="utf-8"))["metadata"]
    require(cos_q15.to(torch.int64).tolist() == prior_preflight["cos_q15_s16"], "cos metadata changed")
    require(sin_q15.to(torch.int64).tolist() == prior_preflight["sin_q15_s16"], "sin metadata changed")
    conversion_q9 = int(layer.self_attn.rope_conversion_q9)
    rope_case = RopeCase(
        name="frozen_c4_layer0_k_head0_token1",
        sequence_position=TOKEN,
        activations=k_head.to(torch.int64).tolist(),
        scales_q9=[conversion_q9] * HEAD_DIM,
        cos_q15=cos_q15.to(torch.int64).tolist(),
        sin_q15=sin_q15.to(torch.int64).tolist(),
    )
    rope_reference = reference_rope(rope_case)
    k_rope_call = trace.calls["fixed_rope_raw_with_saturation"][1]
    k_rope_all, k_rope_saturations = k_rope_call["result"]
    k_rope = k_rope_all[0, HEAD, TOKEN].to(torch.int8)
    require(k_rope.to(torch.int64).tolist() == rope_reference.outputs, "K RoPE reference mismatch")
    preclamp = []
    for index in range(HEAD_DIM):
        pair = index + 32 if index < 32 else index - 32
        current = int(k_head[index]) * conversion_q9
        paired = int(k_head[pair]) * conversion_q9
        rotated = current * int(cos_q15[index]) + (paired * int(sin_q15[index]) if index >= 32 else -paired * int(sin_q15[index]))
        preclamp.append(round_shift_even(rotated, 24))

    for name, value in (
        ("k_projection_input_s8", k_input),
        ("k_projection_weights_s4", k_meta["weights"].to(torch.int8)),
        ("k_projection_bias_s32", k_meta["biases"].to(torch.int64)),
        ("k_projection_multiplier_s32", k_meta["multipliers"].to(torch.int64)),
        ("k_projection_right_shift_u6", k_meta["shifts"].to(torch.int64)),
        ("k_projection_dot_s32", k_meta["dots"].to(torch.int64)),
        ("k_projection_biased_s32", k_meta["biased"].to(torch.int64)),
        ("k_projection_saturation", k_meta["saturations"]),
        ("k_projection_output_s8", k_head),
        ("rope_k_cos_q15_s16", cos_q15),
        ("rope_k_sin_q15_s16", sin_q15),
        ("rope_k_expected_preclamp", torch.tensor(preclamp, dtype=torch.int64)),
        ("rope_k_output_s8", k_rope),
    ):
        retain(name, value)

    value_all = trace.methods["v_proj"]["output"]
    retain("kv_write_k_head0_token1_s8", k_rope)
    retain("kv_write_v_head0_token1_s8", value_all[0, TOKEN, :HEAD_DIM].to(torch.int8))
    score = trace.calls["fixed_attention_scores_raw"][0]["result"]
    probabilities = trace.calls["fixed_softmax_raw"][0]["result"]
    attention_value = trace.calls["fixed_attention_value_raw"][0]["result"]
    retain("attention_score_head0_query_token1_active_s16", score[0, HEAD, TOKEN, : TOKEN + 1].to(torch.int16))
    retain("softmax_head0_query_token1_active_q15", probabilities[0, HEAD, TOKEN, : TOKEN + 1].to(torch.int16))
    retain("attention_value_head0_token1_s8", attention_value[0, HEAD, TOKEN].to(torch.int8))
    retain("o_proj_input_token1_s8", trace.methods["o_proj"]["input"][0, TOKEN].to(torch.int8))
    retain("o_proj_output_token1_s8", trace.methods["o_proj"]["output"][0, TOKEN].to(torch.int8))
    attention_residual = trace.calls["fixed_residual_add"][0]["result"]
    retain("attention_residual_token1_bf16", attention_residual[0, TOKEN].to(torch.bfloat16))
    retain("attention_residual_token1_s8", quantize_int8(attention_residual[0, TOKEN], layer.post_attention_scale))
    post_norm = trace.norms["post_attention_rmsnorm"]
    retain("post_attention_rmsnorm_token1_s8", post_norm[0, TOKEN].to(torch.int8))
    retain("mlp_gate_proj_token1_s8", trace.methods["mlp_gate_proj"]["output"][0, TOKEN].to(torch.int8))
    retain("mlp_up_proj_token1_s8", trace.methods["mlp_up_proj"]["output"][0, TOKEN].to(torch.int8))
    silu = trace.calls["fixed_silu_gate_raw"][0]["result"]
    retain("silu_gate_token1_s8", silu[0, TOKEN].to(torch.int8))
    down_acc = trace.methods["mlp_down_accumulator"]["output"]
    down_output = trace.methods["mlp_down_requantize"]["output"]
    retain("mlp_down_proj_accumulator_token1_s32", down_acc[TOKEN].to(torch.int32))
    retain("mlp_down_proj_output_token1_s8", down_output[0, TOKEN].to(torch.int8))
    final_residual = trace.calls["fixed_residual_add"][1]["result"]
    retain("mlp_residual_token1_bf16", final_residual[0, TOKEN].to(torch.bfloat16))
    retain("mlp_residual_token1_s8", quantize_int8(final_residual[0, TOKEN], layer.post_mlp_scale))

    witness = {
        "schema_version": 1,
        "classification": "single_retained_coordinate_extraction_only_layer0_reference_capture",
        "capture": {
            "calibration_dependency_forward_count": 1,
            "completed_at_utc": utc_now(),
            "consumption_marker": artifact(marker),
            "evaluation_forward_count": evaluation_forward_count,
            "head_count_retained": 1,
            "layer_count_retained": 1,
            "record_count_retained": 1,
            "token_count_retained": 1,
        },
        "coordinate": {
            "dataset": DATASET,
            "dataset_record_index": 64,
            "head": HEAD,
            "layer": 0,
            "token": TOKEN,
            "token_id": input_binding["evaluation_witness"]["token_id"],
        },
        "input_binding": input_binding,
        "model": {
            "dtype": "bfloat16_source_with_fixed_point_layer0_capture",
            "repository": model_spec["repository"],
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "revision": model_spec["revision"],
        },
        "k_projection": {
            "input_scale": float(layer.self_attn.k_proj.hardware_input_scale),
            "output_head_scale": float(layer.self_attn.k_proj.output_head_scales[HEAD]),
            "metadata": k_meta["metadata"],
            "tensor_names": [name for name in tensors if name.startswith("k_projection_")],
        },
        "rope_k": {
            "conversion_q9_s16": conversion_q9,
            "output_scale": float(layer.self_attn.key_rope_output_scales[HEAD]),
            "saturated_elements": int(k_rope_saturations),
            "tensor_names": [name for name in tensors if name.startswith("rope_k_")],
        },
        "later_layer0_references": {
            "retained_scope": "token1/head0 where head-addressable; token1 layer vectors after head concatenation",
            "tensor_names": [
                name for name in tensors if not name.startswith("k_projection_") and not name.startswith("rope_k_")
            ],
        },
        "operator_metadata": {
            "attention_score_multiplier_s32": int(layer.self_attn.score_multiplier[HEAD]),
            "attention_score_right_shift_u6": int(layer.self_attn.score_right_shift[HEAD]),
            "attention_residual_scale": float(layer.post_attention_scale),
            "post_attention_rmsnorm_output_scale": float(layer.post_attention_layernorm.output_scale),
            "mlp_gate_output_scale": float(layer.mlp.gate_proj.output_scale),
            "mlp_up_output_scale": float(layer.mlp.up_proj.output_scale),
            "mlp_down_output_scale": float(layer.mlp.down_proj.output_scale),
            "mlp_residual_scale": float(layer.post_mlp_scale),
            "silu_multiplier_s32": int(layer.mlp.silu_multiplier),
            "silu_right_shift_u6": int(layer.mlp.silu_right_shift),
            "v_projection_output_scale": float(layer.self_attn.v_proj.output_scale),
        },
        "tensor_manifest": tensors,
        "source_bindings": {
            "capture_tool": artifact(source_archive),
            "fixed_model": artifact(ROOT / "tools/ace2_full_model_fixed_point.py"),
            "projection_reference": artifact(ROOT / "tools/ace2_projection_reference.py"),
            "rope_reference": artifact(ROOT / "tools/ace2_rope_reference.py"),
            "prior_capture_sha256s": artifact(PRIOR_SUMS),
            "prior_preflight": artifact(PRIOR_PREFLIGHT),
            "rtl_tree": rtl_tree(),
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
            "other_records",
            "other_retained_tokens",
            "other_retained_heads",
            "other_layers",
            "quality_baseline",
            "candidate_or_benchmark",
            "synthesis_opensta_ppa",
            "rtl_mutation",
            "protected_state_mutation",
        ],
    }
    write_json(CAPTURE / "witness.json", witness)
    write_json(
        CAPTURE / "capture_contract.json",
        {
            "schema_version": 1,
            "status": "CAPTURE_COMPLETE_REPLAY_PENDING",
            "completed_at_utc": utc_now(),
            "coordinate": witness["coordinate"],
            "consumption_marker": artifact(marker),
            "evaluation_forward_count": 1,
            "source": artifact(source_archive),
            "tensor_count": len(tensors),
            "witness": artifact(CAPTURE / "witness.json"),
        },
    )
    (CAPTURE / "capture.log").write_text(
        "ACE2_LAYER0_REFERENCE_CAPTURE_COMPLETE\n"
        "coordinate=c4_en_512:record64:layer0:token1:head0\n"
        "evaluation_forward_count=1\n"
        "retained_head_count=1\n"
        f"tensor_count={len(tensors)}\n"
        f"rope_k_saturated_elements={int(k_rope_saturations)}\n"
        "model_baseline_invoked=false\n"
        "rtl_mutated=false\n",
        encoding="utf-8",
    )
    print(
        "ACE2_LAYER0_REFERENCE_CAPTURE_COMPLETE "
        f"tensors={len(tensors)} evaluation_forwards=1 witness_sha256={sha256_file(CAPTURE / 'witness.json')}"
    )


def build_rope_k_replay_witness(witness: dict[str, Any]) -> dict[str, Any]:
    tensors = witness["tensor_manifest"]
    projection = witness["k_projection"]
    rope = witness["rope_k"]
    return {
        "projection": {
            "input_s8": read_tensor(tensors["k_projection_input_s8"]).to(torch.int64).tolist(),
            "weights_s4": read_tensor(tensors["k_projection_weights_s4"]).to(torch.int64).tolist(),
            "metadata": projection["metadata"],
            "output_s8": read_tensor(tensors["k_projection_output_s8"]).to(torch.int64).tolist(),
            "dot_accumulator_s32": read_tensor(tensors["k_projection_dot_s32"]).to(torch.int64).tolist(),
            "biased_accumulator_s32": read_tensor(tensors["k_projection_biased_s32"]).to(torch.int64).tolist(),
            "saturation_per_output": read_tensor(tensors["k_projection_saturation"]).to(torch.int64).tolist(),
        },
        "rope": {
            "conversion_q9_s16": rope["conversion_q9_s16"],
            "cos_q15_s16": read_tensor(tensors["rope_k_cos_q15_s16"]).to(torch.int64).tolist(),
            "sin_q15_s16": read_tensor(tensors["rope_k_sin_q15_s16"]).to(torch.int64).tolist(),
            "expected_output_preclamp": read_tensor(tensors["rope_k_expected_preclamp"]).to(torch.int64).tolist(),
            "expected_output_s8": read_tensor(tensors["rope_k_output_s8"]).to(torch.int64).tolist(),
        },
    }


def run_replay() -> None:
    witness_path = CAPTURE / "witness.json"
    require(witness_path.is_file(), "capture witness is missing")
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    require(witness["capture"]["evaluation_forward_count"] == 1, "capture forward count changed")
    require(witness["coordinate"] == {
        "dataset": DATASET,
        "dataset_record_index": 64,
        "head": HEAD,
        "layer": 0,
        "token": TOKEN,
        "token_id": witness["coordinate"]["token_id"],
    }, "capture coordinate changed")
    REPLAY.mkdir(parents=True, exist_ok=False)
    generated = REPLAY / "generated"
    generated.mkdir()
    replay_witness = build_rope_k_replay_witness(witness)
    include = generated / "c4_rope_q_witness.svh"
    include.write_text(rope_q_core.render_svh(replay_witness), encoding="utf-8")
    maintained_tb = ROOT / "verification/tb/ace2_c4_rope_q_replay_tb.sv"
    tb_text = maintained_tb.read_text(encoding="utf-8")
    needle = '$display("ACE2_C4_ROPE_Q_REPLAY_PASS projection_outputs=64 rope_outputs=64 mismatches=0");'
    require(needle in tb_text, "maintained RoPE replay completion marker changed")
    tb_text = tb_text.replace(
        needle,
        '$display("ACE2_C4_ROPE_Q_REPLAY_CYCLES total_sim_cycles=%0d", $time/10);\n        ' + needle,
    )
    replay_tb = REPLAY / "replay_tb_source_at_execution.sv"
    replay_tb.write_text(tb_text, encoding="utf-8")
    image = ROOT / "build/rtl-layer0-minimal-reference-capture-batch-replay-v1/rope_k.vvp"
    image.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [
        "iverilog",
        "-g2012",
        "-Wall",
        f"-I{generated}",
        "-o",
        str(image),
        str(ROOT / "rtl/ace2_w4a8_proj_core.sv"),
        str(ROOT / "rtl/ace2_rope_core.sv"),
        str(replay_tb),
    ]
    compile_run = subprocess.run(compile_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "iverilog.log").write_text(
        "command=" + " ".join(compile_command) + "\n" + compile_run.stdout + compile_run.stderr,
        encoding="utf-8",
    )
    require(compile_run.returncode == 0, "RoPE-K replay compilation failed")
    replay_command = ["vvp", str(image)]
    replay_run = subprocess.run(replay_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "replay.stdout").write_text(replay_run.stdout, encoding="utf-8")
    (REPLAY / "replay.stderr").write_text(replay_run.stderr, encoding="utf-8")
    (REPLAY / "replay.log").write_text(
        "command=" + " ".join(replay_command) + "\n" + replay_run.stdout + replay_run.stderr,
        encoding="utf-8",
    )
    require(replay_run.returncode == 0, "RoPE-K replay failed")
    require("ACE2_C4_ROPE_Q_REPLAY_PASS" in replay_run.stdout, "RoPE-K pass marker missing")
    lane_rows = re.findall(
        r"ROPE_Q_WITNESS lane=(\d+) q_expected=(-?\d+) q_actual=(-?\d+) rope_expected=(-?\d+) rope_actual=(-?\d+)",
        replay_run.stdout,
    )
    require(len(lane_rows) == 64, f"RoPE-K lane count changed: {len(lane_rows)}")
    def signed8(text: str) -> int:
        value = int(text)
        return value - 256 if value >= 128 else value

    mismatches = [
        row
        for row in lane_rows
        if int(row[1]) != signed8(row[2]) or int(row[3]) != signed8(row[4])
    ]
    cycle_match = re.search(r"total_sim_cycles=(\d+)", replay_run.stdout)
    require(cycle_match is not None, "RoPE-K cycle observation missing")
    require(not mismatches, f"RoPE-K exact mismatches: {mismatches[:1]}")
    rope_tensor = witness["tensor_manifest"]["rope_k_output_s8"]
    kv_k = witness["tensor_manifest"]["kv_write_k_head0_token1_s8"]
    kv_v = witness["tensor_manifest"]["kv_write_v_head0_token1_s8"]
    boundary = {
        "schema_version": 1,
        "mission_id": "rtl-layer0-minimal-reference-capture-batch-replay-v1",
        "operator": "layer_0.kv_write",
        "classification": "UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_FROZEN_HEAD_SCOPE",
        "available_retained_interface": {
            "k_head0_bytes": kv_k["bytes"],
            "k_head0_sha256": kv_k["sha256"],
            "v_head0_bytes": kv_v["bytes"],
            "v_head0_sha256": kv_v["sha256"],
        },
        "required_shell_interface": {
            "k_bytes": 128,
            "v_bytes": 128,
            "metadata_bytes": 16,
            "kv_heads": 2,
            "source": artifact(ROOT / "verification/tb/ace2_shell_tb.sv"),
        },
        "missing_fields": [
            "layer_0.kv_write.input.k_head1_s8[64]",
            "layer_0.kv_write.input.v_head1_s8[64]",
            "layer_0.kv_write.input.joint_two_head_scale_metadata[16B]",
        ],
        "scope_reason": "The operator contract permits retaining only head 0; capturing head 1 would violate the frozen no-other-head boundary.",
        "narrow_root_cone": "The first unavailable cone is the GQA head-local RoPE-K output to the two-KV-head shell KV_WRITE descriptor packing boundary; RTL arithmetic is not implicated.",
        "simulation_executed": False,
        "later_operators_attempted": False,
    }
    write_json(REPLAY / "boundary_probe.json", boundary)
    report = {
        "schema_version": 1,
        "mission_id": "rtl-layer0-minimal-reference-capture-batch-replay-v1",
        "producer_role": "engineer",
        "review_status": "PENDING_FRESH_REVIEWER",
        "coordinate": witness["coordinate"],
        "capture": {
            "consumption_marker": witness["capture"]["consumption_marker"],
            "evaluation_forward_count": 1,
            "tensor_count": len(witness["tensor_manifest"]),
            "witness": artifact(witness_path),
        },
        "frozen_order": ORDER,
        "rtl_tree": rtl_tree(),
        "prior_maximal_prefix": ORDER[:5],
        "attempted_operators": [
            {
                "operator": "layer_0.rope_k",
                "status": "PASS_EXACT",
                "input_hashes": {
                    "projection_input_s8": witness["tensor_manifest"]["k_projection_input_s8"]["sha256"],
                    "projection_weights_s4": witness["tensor_manifest"]["k_projection_weights_s4"]["sha256"],
                    "cos_q15_s16": witness["tensor_manifest"]["rope_k_cos_q15_s16"]["sha256"],
                    "sin_q15_s16": witness["tensor_manifest"]["rope_k_sin_q15_s16"]["sha256"],
                },
                "reference_hashes": {
                    "projection_output_s8": witness["tensor_manifest"]["k_projection_output_s8"]["sha256"],
                    "rope_output_s8": rope_tensor["sha256"],
                },
                "rtl_source_hashes": {
                    "projection_core": artifact(ROOT / "rtl/ace2_w4a8_proj_core.sv"),
                    "rope_core": artifact(ROOT / "rtl/ace2_rope_core.sv"),
                    "maintained_harness": artifact(maintained_tb),
                    "execution_copy": artifact(replay_tb),
                },
                "counts": {
                    "projection_outputs": 64,
                    "rope_outputs": 64,
                    "compared_outputs": 64,
                    "mismatches": 0,
                },
                "saturation": {
                    "reference_saturated_elements": witness["rope_k"]["saturated_elements"],
                    "rtl_result": "MATCH",
                },
                "cycle_handshake_latency": {
                    "simulation_executed": True,
                    "total_sim_cycles": int(cycle_match.group(1)),
                    "handshake_result": "PASS_NO_TIMEOUT",
                    "latency_result": "PASS_BOUNDED_BY_MAINTAINED_HARNESS",
                },
                "raw_log": artifact(REPLAY / "replay.log"),
            },
            {
                "operator": "layer_0.kv_write",
                "status": boundary["classification"],
                "counts": {
                    "required_k_bytes": 128,
                    "retained_k_bytes": 64,
                    "required_v_bytes": 128,
                    "retained_v_bytes": 64,
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
        "newly_passing_operators": ["layer_0.rope_k"],
        "maximal_passing_prefix": ORDER[:6],
        "first_failing_or_unsupported_operator": {
            "operator": "layer_0.kv_write",
            "classification": boundary["classification"],
            "boundary_probe": artifact(REPLAY / "boundary_probe.json"),
        },
        "batch_result": "STOPPED_AT_FIRST_PRECISELY_UNAVAILABLE_EXECUTABLE_INTERFACE",
        "later_operators_attempted": False,
        "existing_current_tree_ppa": {
            "reused_not_rerun": True,
            "sha256_manifest": artifact(PPA_SUMS),
        },
        "forbidden_flows_run": [],
        "state_mutations": {
            "rtl": False,
            "frontier_ledger_manifest_traceability_pipeline_seals": False,
            "synthesis_opensta_ppa": False,
            "baseline_candidate_benchmark_scale32": False,
        },
        "review_requirement": "Fresh Reviewer must independently accept or reject this maximal-prefix result.",
    }
    write_json(REPLAY / "ordered_prefix_report.json", report)
    (REPLAY / "batch_replay.log").write_text(
        "ACE2_LAYER0_ORDERED_BATCH_REPLAY\n"
        f"rtl_tree_sha256={EXPECTED_RTL_TREE}\n"
        "coordinate=c4_en_512:record64:layer0:token1:head0\n"
        "layer_0.rope_k=PASS_EXACT mismatches=0 outputs=64\n"
        "layer_0.kv_write=UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_FROZEN_HEAD_SCOPE\n"
        "maximal_passing_operator=layer_0.rope_k\n"
        "later_operators_attempted=false\n"
        "synthesis_opensta_ppa_executed=false\n",
        encoding="utf-8",
    )
    write_sha256s(REPLAY)
    print(
        "ACE2_LAYER0_ORDERED_BATCH_REPLAY_BOUNDARY "
        "maximal_prefix=layer_0.rope_k first_boundary=layer_0.kv_write mismatches=0"
    )


def write_sha256s(base: Path) -> Path:
    path = base / "SHA256SUMS"
    rows = []
    for member in sorted(base.rglob("*")):
        if member.is_file() and member != path:
            rows.append(f"{sha256_file(member)}  {member.relative_to(base).as_posix()}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def launch_capture_and_replay() -> None:
    preflight = validate_preflight()
    require(not (CAPTURE / "capture_execution_consumed.json").exists(), "capture already consumed")
    require(not (CAPTURE / "witness.json").exists(), "capture witness already exists")
    require(not REPLAY.exists(), "replay output already exists")
    source_archive = CAPTURE / "capture_source_at_execution.py"
    shutil.copy2(SELF, source_archive)
    require(sha256_file(source_archive) == preflight["source"]["sha256"], "source changed after preflight")
    for name in ("capture.stdout", "capture.stderr"):
        require(not (CAPTURE / name).exists(), f"raw capture log already exists: {name}")
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
                "coordinate": {
                    "dataset": DATASET,
                    "dataset_record_index": 64,
                    "head": HEAD,
                    "layer": 0,
                    "token": TOKEN,
                },
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
    write_json(
        CAPTURE / "capture_worker_status.json",
        {
            "schema_version": 1,
            "completed_at_utc": utc_now(),
            "returncode": worker.returncode,
            "rerun_permitted": False,
            "stdout": artifact(CAPTURE / "capture.stdout"),
            "stderr": artifact(CAPTURE / "capture.stderr"),
        },
    )
    require(worker.returncode == 0, "single capture worker failed; authorization remains consumed")
    write_sha256s(CAPTURE)
    run_replay()


def run_check() -> None:
    preflight = validate_preflight()
    require(preflight["model_execution_count"] == 0, "preflight execution count changed")
    validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE)
    validate_manifest(REPLAY / "SHA256SUMS", REPLAY)
    witness = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    report = json.loads((REPLAY / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    marker = json.loads((CAPTURE / "capture_execution_consumed.json").read_text(encoding="utf-8"))
    require(marker["rerun_permitted"] is False, "consumption marker permits rerun")
    require(witness["capture"]["evaluation_forward_count"] == 1, "capture was not exactly once")
    require(witness["capture"]["head_count_retained"] == 1, "capture retained another head")
    require(report["newly_passing_operators"] == ["layer_0.rope_k"], "maximal prefix changed")
    require(report["maximal_passing_prefix"] == ORDER[:6], "reported prefix changed")
    require(report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.kv_write", "boundary changed")
    require(report["state_mutations"] == {
        "rtl": False,
        "frontier_ledger_manifest_traceability_pipeline_seals": False,
        "synthesis_opensta_ppa": False,
        "baseline_candidate_benchmark_scale32": False,
    }, "forbidden mutation record changed")
    require(rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed after replay")
    print(
        "ACE2_LAYER0_MINIMAL_CAPTURE_BATCH_REPLAY_CHECK_PASS "
        f"capture_sha256={sha256_file(CAPTURE / 'witness.json')} "
        f"report_sha256={sha256_file(REPLAY / 'ordered_prefix_report.json')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "capture-and-replay", "check", "_capture_worker"))
    args = parser.parse_args()
    if args.mode == "preflight":
        run_preflight()
    elif args.mode == "capture-and-replay":
        launch_capture_and_replay()
    elif args.mode == "check":
        run_check()
    else:
        execute_capture_worker()


if __name__ == "__main__":
    main()
