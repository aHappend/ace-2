#!/usr/bin/env python3
"""Capture and replay one frozen C4 layer-0 Q-head/token RoPE witness."""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from ace2_full_model_fixed_point import (
    FixedAttention,
    FixedRMSNorm,
    PROMPT_MANIFEST,
    QUALITY_CONFIG,
    ROOT,
    W4A8Linear,
    calibrate,
    fixed_rope_raw_with_saturation,
    hash_records,
    hash_token_sequences,
    load_contracts,
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


EXPECTED_RTL_TREE_SHA256 = (
    "d73285cff6587a5075bdae233bf77c0dbce98f37b7872d9093e44c4b617fb196"
)
SEALED_BASELINE_SHA256 = (
    "49918c2fa27b87a2fc944b44450122c0ca0fce0c164d0018bc61bd8897801d4e"
)
PIPELINE_STATE_SHA256 = (
    "9fb4c5ec67e328c1892d9658c34c3b829574f86fe5f4cd324160ee2bfa3f6b26"
)
SEALED_BASELINE = (
    ROOT
    / "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/recovery"
    / "baseline_terminal_no_go_v2/BASELINE_TERMINAL_NO_GO.json"
)
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
RTL_MANIFEST = ROOT / "design/RTL_MANIFEST.json"
PRIOR_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-rmsnorm-numerical-bisect-fastpath-v1/round-0003.json"
)
MISSION_CONTEXT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-rope-q-nonzero-position-c4-replay-v1/mission.json"
)
PRIOR_TOKEN0_OUTPUT = ROOT / "evidence/diagnostics/c4-rope-q-capture-replay-20260802-v1"
PRIOR_TOKEN0_SHA256SUMS_SHA256 = (
    "78204f8e788bf1797715c81a91fb2ec7959985c0ac17bdaa0cdc3cc69c024674"
)
PRIOR_TOKEN0_VERIFICATION_SHA256 = (
    "4b4b1884bb40b1f807b81611aa4849e1994965908d583650e1a80936797e304f"
)
PRIOR_TOKEN0_WITNESS_SHA256 = (
    "7032b5b93af3cb7c50cc7c48967a343a184a95155caae4496057ac368992f913"
)
CAPTURE_SOURCE = Path(__file__).resolve()
TESTBENCH_SOURCE = ROOT / "verification/tb/ace2_c4_rope_q_replay_tb.sv"
PROJECTION_RTL = ROOT / "rtl/ace2_w4a8_proj_core.sv"
ROPE_RTL = ROOT / "rtl/ace2_rope_core.sv"
PROJECTION_REFERENCE = ROOT / "tools/ace2_projection_reference.py"
ROPE_REFERENCE = ROOT / "tools/ace2_rope_reference.py"
FIXED_MODEL_SOURCE = ROOT / "tools/ace2_full_model_fixed_point.py"
DATASET_NAME = "c4_en_512"
CALIBRATION_NAME = "c4_calibration"
HEAD_INDEX = 0
TOKEN_INDEX = 1
HEAD_DIM = 64
HIDDEN_SIZE = 896
INPUT_BEAT_LANES = 16
WEIGHT_BEAT_LANES = 16
ROPE_BEAT_LANES = 16


class _Layer0CaptureComplete(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def external_artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate_hash(records: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(records):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(records[relative].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_prior_token0_witness() -> dict[str, Any]:
    sha256s = PRIOR_TOKEN0_OUTPUT / "SHA256SUMS"
    verification = PRIOR_TOKEN0_OUTPUT / "verification.json"
    witness = PRIOR_TOKEN0_OUTPUT / "witness.json"
    if sha256_file(sha256s) != PRIOR_TOKEN0_SHA256SUMS_SHA256:
        raise RuntimeError("prior token-0 witness SHA256SUMS changed")
    if sha256_file(verification) != PRIOR_TOKEN0_VERIFICATION_SHA256:
        raise RuntimeError("prior token-0 verification changed")
    if sha256_file(witness) != PRIOR_TOKEN0_WITNESS_SHA256:
        raise RuntimeError("prior token-0 witness changed")
    for row in sha256s.read_text(encoding="utf-8").splitlines():
        expected, relative = row.split("  ", 1)
        path = PRIOR_TOKEN0_OUTPUT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"prior token-0 artifact changed: {relative}")
    prior = json.loads(witness.read_text(encoding="utf-8"))
    if prior.get("coordinate", {}).get("token") != 0:
        raise RuntimeError("prior witness is no longer the token-0 limited provenance")
    return {
        "classification": "immutable_identity_rotation_limited_provenance_not_counted",
        "sha256s": artifact(sha256s),
        "verification": artifact(verification),
        "witness": artifact(witness),
    }


def verify_frozen_state() -> dict[str, Any]:
    if sha256_file(SEALED_BASELINE) != SEALED_BASELINE_SHA256:
        raise RuntimeError("sealed Dynamic Scale32 terminal no-go changed")
    if sha256_file(PIPELINE_STATE) != PIPELINE_STATE_SHA256:
        raise RuntimeError("PIPELINE_STATE changed from the reviewed prestate")
    manifest = json.loads(RTL_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("rtl_hash") != EXPECTED_RTL_TREE_SHA256:
        raise RuntimeError("RTL manifest tree hash differs from the accepted tree")
    source_records = {
        row["path"]: row["sha256"] for row in manifest.get("source_hashes", [])
    }
    if not source_records:
        raise RuntimeError("RTL manifest source records are missing")
    for relative, expected in source_records.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"accepted RTL source differs: {relative}")
    if aggregate_hash(source_records) != EXPECTED_RTL_TREE_SHA256:
        raise RuntimeError("accepted RTL aggregate hash does not reproduce")
    return {
        "accepted_rtl_tree_sha256": EXPECTED_RTL_TREE_SHA256,
        "prior_token0_witness": verify_prior_token0_witness(),
        "pipeline_state": {
            **artifact(PIPELINE_STATE),
            "mtime_ns": PIPELINE_STATE.stat().st_mtime_ns,
        },
        "sealed_dynamic_scale32_terminal_no_go": {
            **artifact(SEALED_BASELINE),
            "mtime_ns": SEALED_BASELINE.stat().st_mtime_ns,
        },
    }


def capture_layer0_inputs(
    model: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    captured: dict[str, Any] = {}
    layer = model.model.layers[0]

    def capture_attention(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        position_embeddings = kwargs.get("position_embeddings")
        if not isinstance(hidden_states, Tensor):
            raise RuntimeError("layer-0 attention hidden states were not captured")
        if (
            not isinstance(position_embeddings, tuple)
            or len(position_embeddings) != 2
            or not all(isinstance(value, Tensor) for value in position_embeddings)
        ):
            raise RuntimeError("layer-0 RoPE position metadata was not captured")
        captured["hidden_states"] = hidden_states.detach().cpu()
        captured["position_embeddings"] = tuple(
            value.detach().cpu() for value in position_embeddings
        )
        raise _Layer0CaptureComplete

    hook = layer.self_attn.register_forward_pre_hook(capture_attention, with_kwargs=True)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    except _Layer0CaptureComplete:
        pass
    finally:
        hook.remove()
    if set(captured) != {"hidden_states", "position_embeddings"}:
        raise RuntimeError(f"incomplete layer-0 capture: {sorted(captured)}")
    return captured["hidden_states"], captured["position_embeddings"]


def pack_signed(values: list[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (value & mask) << (lane * width)
    return packed


def chunks(values: list[int], size: int) -> list[list[int]]:
    if len(values) % size:
        raise ValueError("witness vector does not divide into complete beats")
    return [values[index : index + size] for index in range(0, len(values), size)]


def sv_hex(value: int, width: int) -> str:
    digits = (width + 3) // 4
    return f"{width}'h{value & ((1 << width) - 1):0{digits}x}"


def sv_signed_decimal(value: int, width: int) -> str:
    return f"-{width}'sd{-value}" if value < 0 else f"{width}'sd{value}"


def render_svh(witness: dict[str, Any]) -> str:
    projection = witness["projection"]
    rope = witness["rope"]
    inputs = projection["input_s8"]
    weights = projection["weights_s4"]
    metadata = projection["metadata"]
    input_beats = [pack_signed(beat, 8) for beat in chunks(inputs, INPUT_BEAT_LANES)]
    weight_beats = [
        pack_signed(beat, 4)
        for row in weights
        for beat in chunks(row, WEIGHT_BEAT_LANES)
    ]
    cos_beats = [pack_signed(beat, 16) for beat in chunks(rope["cos_q15_s16"], 16)]
    sin_beats = [pack_signed(beat, 16) for beat in chunks(rope["sin_q15_s16"], 16)]
    scale_beats = [
        pack_signed([rope["conversion_q9_s16"]] * 16, 16) for _ in range(4)
    ]
    expected_rope_beats = [
        pack_signed(beat, 8) for beat in chunks(rope["expected_output_s8"], 16)
    ]
    rope_sat_beats = []
    for beat in chunks(rope["expected_output_preclamp"], 16):
        rope_sat_beats.append(int(any(value < -128 or value > 127 for value in beat)))

    lines = [
        "localparam integer WITNESS_K_SIZE = 896;",
        "localparam integer WITNESS_OUTPUTS = 64;",
        "localparam integer WITNESS_INPUT_BEATS = 56;",
        "localparam integer WITNESS_WEIGHT_BEATS_PER_OUTPUT = 56;",
        "localparam integer WITNESS_GROUPS = 224;",
        "localparam integer WITNESS_ROPE_BEATS = 4;",
        "reg [127:0] witness_input_beats [0:WITNESS_INPUT_BEATS-1];",
        "reg [63:0] witness_weight_beats [0:WITNESS_OUTPUTS*WITNESS_WEIGHT_BEATS_PER_OUTPUT-1];",
        "reg [127:0] witness_meta [0:WITNESS_OUTPUTS-1];",
        "reg [7:0] witness_q_expected [0:WITNESS_OUTPUTS-1];",
        "reg signed [31:0] witness_dot_expected [0:WITNESS_OUTPUTS-1];",
        "reg signed [31:0] witness_biased_expected [0:WITNESS_OUTPUTS-1];",
        "reg witness_projection_sat_expected [0:WITNESS_OUTPUTS-1];",
        "reg [255:0] witness_rope_scale_beats [0:WITNESS_ROPE_BEATS-1];",
        "reg [255:0] witness_rope_cos_beats [0:WITNESS_ROPE_BEATS-1];",
        "reg [255:0] witness_rope_sin_beats [0:WITNESS_ROPE_BEATS-1];",
        "reg [127:0] witness_rope_expected_beats [0:WITNESS_ROPE_BEATS-1];",
        "reg witness_rope_sat_expected [0:WITNESS_ROPE_BEATS-1];",
        "task initialize_witness;",
        "begin",
    ]
    for index, value in enumerate(input_beats):
        lines.append(f"witness_input_beats[{index}] = {sv_hex(value, 128)};")
    for index, value in enumerate(weight_beats):
        lines.append(f"witness_weight_beats[{index}] = {sv_hex(value, 64)};")
    for index, row in enumerate(metadata):
        lines.append(f"witness_meta[{index}] = {sv_hex(row['packed_u128'], 128)};")
        lines.append(f"witness_q_expected[{index}] = {sv_hex(projection['output_s8'][index], 8)};")
        lines.append(
            f"witness_dot_expected[{index}] = "
            f"{sv_signed_decimal(projection['dot_accumulator_s32'][index], 32)};"
        )
        lines.append(
            f"witness_biased_expected[{index}] = "
            f"{sv_signed_decimal(projection['biased_accumulator_s32'][index], 32)};"
        )
        lines.append(
            f"witness_projection_sat_expected[{index}] = 1'b{projection['saturation_per_output'][index]};"
        )
    for index in range(4):
        lines.append(f"witness_rope_scale_beats[{index}] = {sv_hex(scale_beats[index], 256)};")
        lines.append(f"witness_rope_cos_beats[{index}] = {sv_hex(cos_beats[index], 256)};")
        lines.append(f"witness_rope_sin_beats[{index}] = {sv_hex(sin_beats[index], 256)};")
        lines.append(f"witness_rope_expected_beats[{index}] = {sv_hex(expected_rope_beats[index], 128)};")
        lines.append(f"witness_rope_sat_expected[{index}] = 1'b{rope_sat_beats[index]};")
    lines.extend(["end", "endtask", ""])
    return "\n".join(lines)


def model_free_rope_metadata() -> dict[str, Any]:
    manifest, config, _rtl_binding = load_contracts(require_rtl_binding=False)
    versions = validate_runtime(config)
    model_spec = manifest["model"]
    model_config = AutoConfig.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        local_files_only=True,
    )
    resolved_revision = getattr(model_config, "_commit_hash", None)
    if resolved_revision != model_spec["revision"]:
        raise RuntimeError(
            f"model config resolved to {resolved_revision}, expected {model_spec['revision']}"
        )
    head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    if head_dim != HEAD_DIM:
        raise RuntimeError(f"model-free RoPE head dimension is {head_dim}, expected {HEAD_DIM}")
    if getattr(model_config, "rope_scaling", None) not in (None, {}):
        raise RuntimeError("model-free preflight does not implement configured RoPE scaling")
    if TOKEN_INDEX >= model_config.max_position_embeddings:
        raise RuntimeError("token coordinate exceeds the frozen model position limit")
    rope_theta = float(model_config.rope_theta)
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    position = torch.tensor([[[TOKEN_INDEX]]], dtype=torch.float32)
    frequencies = (inv_freq.reshape(1, -1, 1) @ position).transpose(1, 2)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos_source = embedding.cos().to(torch.bfloat16)
    sin_source = embedding.sin().to(torch.bfloat16)
    cos_q15 = (
        torch.round(cos_source[0, 0].to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
    )
    sin_q15 = (
        torch.round(sin_source[0, 0].to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
    )
    nonzero_sine_lanes = [
        index for index, value in enumerate(sin_q15.tolist()) if value != 0
    ]
    nonidentity_lanes = [
        index
        for index, (cosine, sine) in enumerate(
            zip(cos_q15.tolist(), sin_q15.tolist(), strict=True)
        )
        if cosine != 32767 or sine != 0
    ]
    if not nonzero_sine_lanes:
        raise RuntimeError("preflight failed: token 1 has no representable nonzero sine lane")
    if not nonidentity_lanes:
        raise RuntimeError("preflight failed: token 1 Q15 RoPE transform is identity")
    return {
        "coordinate": {
            "head": HEAD_INDEX,
            "head_dim": HEAD_DIM,
            "layer": 0,
            "operator": "rope_q",
            "token": TOKEN_INDEX,
        },
        "cos_q15_s16": cos_q15.tolist(),
        "formula": (
            "inv_freq[i]=1/(rope_theta**((2*i)/head_dim)); "
            "angle=token*inv_freq; trig computed in fp32 then cast to frozen-model "
            "bfloat16 before q15=clamp(round_to_nearest_even(trig*32767))"
        ),
        "model_config": {
            "head_dim": head_dim,
            "max_position_embeddings": model_config.max_position_embeddings,
            "model_type": model_config.model_type,
            "repository": model_spec["repository"],
            "requested_revision": model_spec["revision"],
            "resolved_revision": resolved_revision,
            "rope_scaling": model_config.rope_scaling,
            "rope_theta": rope_theta,
        },
        "nonidentity_lane_indices": nonidentity_lanes,
        "nonzero_sine_lane_indices": nonzero_sine_lanes,
        "packages": versions,
        "proof": {
            "expected_rope_transform_non_identity": True,
            "representable_nonzero_sine": True,
        },
        "sin_q15_s16": sin_q15.tolist(),
    }


def source_bindings() -> dict[str, dict[str, Any]]:
    paths = [
        CAPTURE_SOURCE,
        TESTBENCH_SOURCE,
        PROJECTION_RTL,
        ROPE_RTL,
        PROJECTION_REFERENCE,
        ROPE_REFERENCE,
        FIXED_MODEL_SOURCE,
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        RTL_MANIFEST,
        PRIOR_REVIEW,
        MISSION_CONTEXT,
    ]
    bindings = {}
    for path in paths:
        if path == PRIOR_REVIEW:
            bindings["accepted_projection_review"] = external_artifact(path)
        elif path == MISSION_CONTEXT:
            bindings["operator_mission_context"] = external_artifact(path)
        else:
            bindings[path.relative_to(ROOT).as_posix()] = artifact(path)
    return bindings


def ensure_no_prior_token1_output(output_dir: Path) -> None:
    prior = [
        path
        for path in (ROOT / "evidence/diagnostics").glob(
            "c4-rope-q-token1-capture-replay-*"
        )
        if path != output_dir
    ]
    if prior:
        raise RuntimeError(f"prior token-1 capture directory exists: {prior[0]}")


def run_preflight(output_dir: Path) -> None:
    if output_dir.exists():
        raise RuntimeError("token-1 preflight output already exists")
    ensure_no_prior_token1_output(output_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("preflight requires CUDA_VISIBLE_DEVICES to be empty or -1")
    prestate = verify_frozen_state()
    metadata = model_free_rope_metadata()
    output_dir.mkdir(parents=True)
    shutil.copy2(CAPTURE_SOURCE, output_dir / "preflight_source_at_execution.py")
    preflight = {
        "schema_version": 1,
        "status": "pass_model_execution_not_consumed",
        "completed_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "model_execution_count": 0,
        "metadata": metadata,
        "frozen_prestate": prestate,
        "source": artifact(CAPTURE_SOURCE),
        "source_archive": artifact(output_dir / "preflight_source_at_execution.py"),
    }
    write_json(output_dir / "preflight.json", preflight)
    (output_dir / "preflight.log").write_text(
        "\n".join(
            [
                "model_execution_count=0",
                f"token={TOKEN_INDEX}",
                f"head={HEAD_INDEX}",
                f"nonzero_sine_lane_count={len(metadata['nonzero_sine_lane_indices'])}",
                f"first_nonzero_sine_lane={metadata['nonzero_sine_lane_indices'][0]}",
                f"first_nonzero_sine_q15={metadata['sin_q15_s16'][metadata['nonzero_sine_lane_indices'][0]]}",
                f"nonidentity_lane_count={len(metadata['nonidentity_lane_indices'])}",
                "expected_rope_transform_non_identity=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "ACE2_C4_ROPE_Q_PREFLIGHT_PASS model_executions=0 token=1 head=0 "
        f"nonzero_sine_lanes={len(metadata['nonzero_sine_lane_indices'])} "
        f"nonidentity_lanes={len(metadata['nonidentity_lane_indices'])}"
    )


def load_preflight(output_dir: Path) -> dict[str, Any]:
    ensure_no_prior_token1_output(output_dir)
    path = output_dir / "preflight.json"
    if not path.is_file():
        raise RuntimeError("model-free token-1 preflight is missing")
    preflight = json.loads(path.read_text(encoding="utf-8"))
    metadata = preflight.get("metadata", {})
    proof = metadata.get("proof", {})
    coordinate = metadata.get("coordinate", {})
    if preflight.get("model_execution_count") != 0:
        raise RuntimeError("preflight incorrectly records model execution")
    if coordinate.get("token") != TOKEN_INDEX or coordinate.get("head") != HEAD_INDEX:
        raise RuntimeError("preflight coordinate differs from token 1/head 0")
    if not proof.get("representable_nonzero_sine"):
        raise RuntimeError("preflight lacks a representable nonzero sine proof")
    if not proof.get("expected_rope_transform_non_identity"):
        raise RuntimeError("preflight lacks a non-identity transform proof")
    if not metadata.get("nonzero_sine_lane_indices"):
        raise RuntimeError("preflight nonzero sine lane list is empty")
    return preflight


def run_capture(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise RuntimeError("run the model-free preflight before capture")
    preflight = load_preflight(output_dir)
    capture_consumption = output_dir / "capture_execution_consumed.json"
    if capture_consumption.exists() or (output_dir / "witness.json").exists():
        raise RuntimeError("token-1 capture authorization was already consumed")
    prestate = verify_frozen_state()
    if preflight.get("frozen_prestate") != prestate:
        raise RuntimeError("sealed state changed after the model-free preflight")
    manifest, config, _rtl_binding = load_contracts(require_rtl_binding=False)
    versions = validate_runtime(config)
    seed_everything(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("capture requires CUDA_VISIBLE_DEVICES to be empty or -1")

    calibration_spec = manifest["datasets"][CALIBRATION_NAME]
    evaluation_spec = manifest["datasets"][DATASET_NAME]
    calibration_text = selected_texts(calibration_spec, limit=1)
    evaluation_text = selected_texts(evaluation_spec, limit=1)
    model_spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"], revision=model_spec["revision"]
    )
    calibration_prompt = tokenize_prompts(
        tokenizer, calibration_text, calibration_spec["token_limit"]
    )[0]
    evaluation_prompt = tokenize_prompts(
        tokenizer, evaluation_text, evaluation_spec["token_limit"]
    )[0]
    if evaluation_prompt.shape[1] <= TOKEN_INDEX:
        raise RuntimeError("frozen evaluation prompt is shorter than the token coordinate")

    started = utc_now()
    shutil.copy2(CAPTURE_SOURCE, output_dir / "capture_source_at_execution.py")
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("frozen model was not loaded entirely on CPU")
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != model_spec["revision"]:
        raise RuntimeError(
            f"model resolved to {resolved_revision}, expected {model_spec['revision']}"
        )
    write_json(
        capture_consumption,
        {
            "schema_version": 1,
            "status": "authorization_consumed_before_first_frozen_model_execution",
            "consumed_at_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
            "coordinate": {"dataset_record_count": 1, "head": HEAD_INDEX, "token": TOKEN_INDEX},
            "cpu_only": True,
            "source_archive": artifact(output_dir / "capture_source_at_execution.py"),
        },
    )
    ranges, operator_ranges = calibrate(model, [calibration_prompt])
    replace_linears(model, ranges, rope_diagnostic_mechanism=None)
    replace_fixed_operators(model, operator_ranges, rope_diagnostic_mechanism=None)
    layer = model.model.layers[0]
    if not isinstance(layer.input_layernorm, FixedRMSNorm):
        raise TypeError("layer-0 fixed RMSNorm was not installed")
    if not isinstance(layer.self_attn, FixedAttention):
        raise TypeError("layer-0 fixed attention was not installed")
    if not isinstance(layer.self_attn.q_proj, W4A8Linear):
        raise TypeError("layer-0 Q projection is not W4A8Linear")

    seed_everything(config)
    hidden_states, position_embeddings = capture_layer0_inputs(model, evaluation_prompt)
    if not torch.equal(hidden_states, torch.round(hidden_states)):
        raise RuntimeError("captured fixed RMSNorm output is not an integer container")
    q_input = hidden_states[0, TOKEN_INDEX].to(torch.int8)
    q_proj = layer.self_attn.q_proj
    with torch.inference_mode():
        q_all = q_proj.forward_quantized(q_input.reshape(1, HIDDEN_SIZE))[0]
    q_head = q_all[HEAD_INDEX * HEAD_DIM : (HEAD_INDEX + 1) * HEAD_DIM]
    channel_start = HEAD_INDEX * HEAD_DIM
    channel_stop = channel_start + HEAD_DIM
    weights = q_proj.qweight[channel_start:channel_stop].detach().cpu().to(torch.int64)
    multipliers = q_proj.multiplier[channel_start:channel_stop].detach().cpu().to(torch.int64)
    shifts = q_proj.right_shift[channel_start:channel_stop].detach().cpu().to(torch.int64)
    if q_proj.bias_accumulator is None:
        biases = torch.zeros(HEAD_DIM, dtype=torch.int64)
    else:
        biases = q_proj.bias_accumulator[channel_start:channel_stop].detach().cpu().to(torch.int64)
    zero_points = torch.zeros(HEAD_DIM, dtype=torch.int64)

    projection_case = ProjectionCase(
        name="frozen_c4_layer0_q_head0_token1",
        rows=1,
        reduction_size=HIDDEN_SIZE,
        activations=[q_input.to(torch.int64).tolist()],
        weights=weights.tolist(),
        multipliers=multipliers.tolist(),
        right_shifts=shifts.tolist(),
        output_zero_points=zero_points.tolist(),
        bias_accumulators=biases.tolist(),
    )
    projection_reference = reference_projection(projection_case)
    if projection_reference.outputs[0] != q_head.to(torch.int64).tolist():
        raise RuntimeError("captured Q output differs from the repaired projection contract")

    dot_accumulators: list[int] = []
    biased_accumulators: list[int] = []
    saturation_per_output: list[int] = []
    metadata: list[dict[str, Any]] = []
    for output_index in range(HEAD_DIM):
        dot = sum(
            int(activation) * int(weight)
            for activation, weight in zip(q_input.tolist(), weights[output_index].tolist(), strict=True)
        )
        biased = dot + int(biases[output_index])
        rounded = round_shift_even(biased * int(multipliers[output_index]), int(shifts[output_index]))
        _clipped, saturated = saturate_int8(rounded + int(zero_points[output_index]))
        packed = pack_meta(
            int(multipliers[output_index]),
            int(shifts[output_index]),
            int(zero_points[output_index]),
            int(biases[output_index]),
        )
        decoded = {
            "bias_accumulator_s32": to_sint((packed >> 48) & 0xFFFFFFFF, 32),
            "multiplier_s32": to_sint(packed & 0xFFFFFFFF, 32),
            "output_zero_point_s8": to_sint((packed >> 40) & 0xFF, 8),
            "right_shift_u6": (packed >> 32) & 0x3F,
        }
        if decoded != {
            "bias_accumulator_s32": int(biases[output_index]),
            "multiplier_s32": int(multipliers[output_index]),
            "output_zero_point_s8": int(zero_points[output_index]),
            "right_shift_u6": int(shifts[output_index]),
        }:
            raise RuntimeError("projection metadata decode differs from accepted layout")
        if ((packed >> 38) & 0x3) or (packed >> 80):
            raise RuntimeError("projection metadata reserved bits are nonzero")
        dot_accumulators.append(dot)
        biased_accumulators.append(biased)
        saturation_per_output.append(int(saturated))
        metadata.append({"packed_u128": packed, **decoded})

    cos, sin = position_embeddings
    cos_q15 = (
        torch.round(cos[0, TOKEN_INDEX].to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
    )
    sin_q15 = (
        torch.round(sin[0, TOKEN_INDEX].to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
    )
    preflight_metadata = preflight["metadata"]
    if cos_q15.tolist() != preflight_metadata["cos_q15_s16"]:
        raise RuntimeError("captured token-1 cosine metadata differs from model-free preflight")
    if sin_q15.tolist() != preflight_metadata["sin_q15_s16"]:
        raise RuntimeError("captured token-1 sine metadata differs from model-free preflight")
    conversion_q9 = int(layer.self_attn.rope_conversion_q9)
    rope_case = RopeCase(
        name="frozen_c4_layer0_q_head0_token1",
        sequence_position=TOKEN_INDEX,
        activations=q_head.to(torch.int64).tolist(),
        scales_q9=[conversion_q9] * HEAD_DIM,
        cos_q15=cos_q15.tolist(),
        sin_q15=sin_q15.tolist(),
    )
    rope_reference = reference_rope(rope_case)
    fixed_rope, fixed_saturations = fixed_rope_raw_with_saturation(
        q_head.reshape(1, 1, 1, HEAD_DIM), conversion_q9, cos[:, TOKEN_INDEX:TOKEN_INDEX + 1], sin[:, TOKEN_INDEX:TOKEN_INDEX + 1]
    )
    fixed_rope_list = fixed_rope.reshape(-1).to(torch.int64).tolist()
    if fixed_rope_list != rope_reference.outputs:
        raise RuntimeError("independent RoPE reference differs from fixed software path")

    preclamp: list[int] = []
    for index in range(HEAD_DIM):
        pair_index = index + 32 if index < 32 else index - 32
        current = int(q_head[index]) * conversion_q9
        paired = int(q_head[pair_index]) * conversion_q9
        if index < 32:
            rotated = current * int(cos_q15[index]) - paired * int(sin_q15[index])
        else:
            rotated = current * int(cos_q15[index]) + paired * int(sin_q15[index])
        preclamp.append(round_shift_even(rotated, 24))

    calibration_record_hash, calibration_record_count = hash_records(calibration_text)
    evaluation_record_hash, evaluation_record_count = hash_records(evaluation_text)
    calibration_token_hash, calibration_sequences, calibration_tokens = hash_token_sequences([calibration_prompt])
    evaluation_token_hash, evaluation_sequences, evaluation_tokens = hash_token_sequences([evaluation_prompt])
    completed = utc_now()
    witness = {
        "schema_version": 1,
        "classification": "single_cpu_frozen_c4_rope_q_capture_replay_witness_not_model_baseline",
        "capture": {
            "completed_at_utc": completed,
            "started_at_utc": started,
            "authorization_consumption": artifact(capture_consumption),
            "exactly_once_cpu_execution": True,
            "evaluation_witness_count": 1,
            "head_count": 1,
            "token_count": 1,
        },
        "coordinate": {
            "batch": 0,
            "dataset": DATASET_NAME,
            "dataset_record_index": evaluation_spec["indices"]["start"],
            "head": HEAD_INDEX,
            "layer": 0,
            "operator": "rope_q",
            "token": TOKEN_INDEX,
            "token_id": int(evaluation_prompt[0, TOKEN_INDEX]),
        },
        "model": {
            **model_spec,
            "resolved_revision": resolved_revision,
            "dtype": "bfloat16_source_with_contract_fixed_point_capture",
        },
        "dataset_provenance": {
            "calibration_dependency_not_replay_witness": {
                "config": calibration_spec["config"],
                "record_count": calibration_record_count,
                "record_sha256": calibration_record_hash,
                "repository": calibration_spec["repository"],
                "revision": calibration_spec["revision"],
                "split": calibration_spec["split"],
                "token_sequence_count": calibration_sequences,
                "token_sequence_sha256": calibration_token_hash,
                "token_count": calibration_tokens,
            },
            "evaluation_witness": {
                "config": evaluation_spec["config"],
                "record_count": evaluation_record_count,
                "record_sha256": evaluation_record_hash,
                "repository": evaluation_spec["repository"],
                "revision": evaluation_spec["revision"],
                "split": evaluation_spec["split"],
                "token_sequence_count": evaluation_sequences,
                "token_sequence_sha256": evaluation_token_hash,
                "token_count": evaluation_tokens,
            },
        },
        "projection": {
            "bias_layout": {
                "bias_accumulator_s32": [79, 48],
                "multiplier_s32": [31, 0],
                "output_zero_point_s8": [47, 40],
                "reserved_high_zero": [127, 80],
                "reserved_low_zero": [39, 38],
                "right_shift_u6": [37, 32],
            },
            "biased_accumulator_s32": biased_accumulators,
            "dot_accumulator_s32": dot_accumulators,
            "input_s8": q_input.to(torch.int64).tolist(),
            "input_scale": float(q_proj.hardware_input_scale),
            "metadata": metadata,
            "output_head_scale": float(q_proj.output_head_scales[HEAD_INDEX]),
            "output_s8": q_head.to(torch.int64).tolist(),
            "rounding": "round_to_nearest_ties_to_even",
            "saturation": "signed_int8_after_bias_then_multiplier_shift_and_zero_point",
            "saturation_per_output": saturation_per_output,
            "weights_s4": weights.tolist(),
        },
        "rope": {
            "conversion_q9_s16": conversion_q9,
            "conversion_realized": conversion_q9 / 512.0,
            "cos_q15_s16": cos_q15.tolist(),
            "expected_output_preclamp": preclamp,
            "expected_output_s8": rope_reference.outputs,
            "fixed_software_output_s8": fixed_rope_list,
            "fixed_software_saturated_elements": fixed_saturations,
            "independent_reference": artifact(ROPE_REFERENCE),
            "output_scale": float(layer.self_attn.query_rope_output_scales[HEAD_INDEX]),
            "pairing": "split_half_lane_i_pairs_with_i_plus_or_minus_32",
            "rounding": "round_to_nearest_ties_to_even_after_signed_q6_9_times_q1_15_shift_24",
            "saturation": "signed_int8_after_rounding",
            "sin_q15_s16": sin_q15.tolist(),
        },
        "runtime": {
            "cpu_only": True,
            "cuda_available_after_visibility_mask": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": versions,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "model_free_preflight": {
            "artifact": artifact(output_dir / "preflight.json"),
            "expected_rope_transform_non_identity": preflight_metadata["proof"][
                "expected_rope_transform_non_identity"
            ],
            "nonidentity_lane_indices": preflight_metadata["nonidentity_lane_indices"],
            "nonzero_sine_lane_indices": preflight_metadata["nonzero_sine_lane_indices"],
            "representable_nonzero_sine": preflight_metadata["proof"][
                "representable_nonzero_sine"
            ],
        },
        "scope_exclusions": [
            "model_baseline",
            "paired_smoke",
            "candidate",
            "PPA",
            "benchmark",
            "execution_authority_or_reservation_or_ledger",
            "Dynamic_Scale32_generation_or_execution",
            "PIPELINE_STATE_change",
            "additional_heads_tokens_or_datasets",
            "downstream_attention",
        ],
        "source_bindings": source_bindings(),
        "frozen_prestate": prestate,
    }

    incomplete = output_dir / "capture_artifacts.incomplete"
    incomplete.mkdir()
    generated = incomplete / "generated"
    generated.mkdir()
    write_json(incomplete / "witness.json", witness)
    (generated / "c4_rope_q_witness.svh").write_text(render_svh(witness), encoding="utf-8")
    capture_contract = {
        "schema_version": 1,
        "status": "capture_complete_replay_pending",
        "command": [sys.executable, *sys.argv],
        "completed_at_utc": completed,
        "coordinate": witness["coordinate"],
        "exactly_once_cpu_execution_consumed": True,
        "authorization_consumption": artifact(capture_consumption),
        "model_free_preflight": artifact(output_dir / "preflight.json"),
        "prestate": prestate,
        "witness_sha256": sha256_file(incomplete / "witness.json"),
        "generated_svh_sha256": sha256_file(generated / "c4_rope_q_witness.svh"),
    }
    write_json(incomplete / "capture_contract.json", capture_contract)
    (incomplete / "capture.log").write_text(
        "\n".join(
            [
                f"capture_started_at_utc={started}",
                f"capture_completed_at_utc={completed}",
                f"model_revision={resolved_revision}",
                f"dataset={DATASET_NAME}",
                f"record_index={evaluation_spec['indices']['start']}",
                f"token={TOKEN_INDEX}",
                f"head={HEAD_INDEX}",
                f"projection_outputs={len(q_head)}",
                f"projection_bias_nonzero={sum(int(value) != 0 for value in biases.tolist())}",
                f"rope_outputs={len(rope_reference.outputs)}",
                f"rope_saturated_elements={fixed_saturations}",
                "dynamic_scale32_invoked=false",
                "model_baseline_invoked=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(incomplete / "generated", output_dir / "generated")
    for name in ("witness.json", "capture_contract.json", "capture.log"):
        os.replace(incomplete / name, output_dir / name)
    incomplete.rmdir()
    print(
        "ACE2_C4_ROPE_Q_CAPTURE_COMPLETE "
        f"coordinate=[0,0,{TOKEN_INDEX},{HEAD_INDEX}] q_lanes=64 "
        f"rope_lanes=64 witness_sha256={sha256_file(output_dir / 'witness.json')}"
    )


WITNESS_RE = re.compile(
    r"^ROPE_Q_WITNESS lane=(?P<lane>\d+) q_expected=(?P<q_expected>-?\d+) "
    r"q_actual=(?P<q_actual>-?\d+) rope_expected=(?P<rope_expected>-?\d+) "
    r"rope_actual=(?P<rope_actual>-?\d+) dot=(?P<dot>-?\d+) "
    r"bias=(?P<bias>-?\d+) biased=(?P<biased>-?\d+) "
    r"multiplier=(?P<multiplier>-?\d+) shift=(?P<shift>\d+) zp=(?P<zp>-?\d+)$"
)


def run_replay(output_dir: Path) -> None:
    preflight = load_preflight(output_dir)
    witness_path = output_dir / "witness.json"
    if not witness_path.is_file():
        raise RuntimeError("capture witness is missing")
    if (output_dir / "replay_results.json").exists():
        raise RuntimeError("replay already exists; use the retained witness only after a proven fix")
    verify_frozen_state()
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    if witness["capture"]["head_count"] != 1 or witness["capture"]["token_count"] != 1:
        raise RuntimeError("capture is not exactly one head/token witness")
    if witness.get("coordinate", {}).get("token") != TOKEN_INDEX:
        raise RuntimeError("retained witness is not token 1")
    if witness["rope"]["cos_q15_s16"] != preflight["metadata"]["cos_q15_s16"]:
        raise RuntimeError("retained cosine metadata differs from preflight")
    if witness["rope"]["sin_q15_s16"] != preflight["metadata"]["sin_q15_s16"]:
        raise RuntimeError("retained sine metadata differs from preflight")
    captured_generated = output_dir / "generated/c4_rope_q_witness.svh"
    if not captured_generated.is_file():
        raise RuntimeError("generated witness include is missing")
    replay_generated_dir = output_dir / "replay_generated"
    replay_generated_dir.mkdir(exist_ok=True)
    generated = replay_generated_dir / "c4_rope_q_witness.svh"
    generated.write_text(render_svh(witness), encoding="utf-8")
    shutil.copy2(CAPTURE_SOURCE, output_dir / "replay_source_at_execution.py")
    build_dir = ROOT / "build" / output_dir.name
    build_dir.mkdir(parents=True, exist_ok=True)
    image = build_dir / "ace2_c4_rope_q_replay_tb.vvp"
    compile_command = [
        "iverilog",
        "-g2012",
        "-Wall",
        f"-I{generated.parent}",
        "-o",
        str(image),
        str(PROJECTION_RTL),
        str(ROPE_RTL),
        str(TESTBENCH_SOURCE),
    ]
    compile_run = subprocess.run(
        compile_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    (output_dir / "iverilog.log").write_text(
        "command=" + " ".join(compile_command) + "\n" + compile_run.stdout + compile_run.stderr,
        encoding="utf-8",
    )
    if compile_run.returncode:
        raise RuntimeError("focused Q-to-RoPE replay did not compile")
    replay_command = ["vvp", str(image)]
    replay_run = subprocess.run(
        replay_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    replay_log = "command=" + " ".join(replay_command) + "\n" + replay_run.stdout + replay_run.stderr
    (output_dir / "replay.log").write_text(replay_log, encoding="utf-8")
    rows = []
    for line in replay_run.stdout.splitlines():
        match = WITNESS_RE.match(line)
        if match:
            rows.append({key: int(value) for key, value in match.groupdict().items()})
    rows.sort(key=lambda row: row["lane"])
    if len(rows) != HEAD_DIM:
        raise RuntimeError(f"focused replay emitted {len(rows)} lane records, expected 64")
    mismatches = [
        row
        for row in rows
        if row["q_expected"] != row["q_actual"]
        or row["rope_expected"] != row["rope_actual"]
    ]
    result = {
        "schema_version": 1,
        "status": "match" if replay_run.returncode == 0 and not mismatches else "mismatch",
        "generated_at_utc": utc_now(),
        "capture_witness": artifact(witness_path),
        "model_free_preflight": artifact(output_dir / "preflight.json"),
        "commands": {
            "compile": compile_command,
            "replay": replay_command,
        },
        "expected_actual": rows,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "projection_outputs_checked": HEAD_DIM,
        "rope_outputs_checked": HEAD_DIM,
        "rtl_sources": {
            "projection": artifact(PROJECTION_RTL),
            "rope": artifact(ROPE_RTL),
            "testbench": artifact(TESTBENCH_SOURCE),
        },
        "logs": {
            "iverilog": artifact(output_dir / "iverilog.log"),
            "replay": artifact(output_dir / "replay.log"),
        },
        "replay_generated_include": artifact(generated),
        "replay_source_archive": artifact(output_dir / "replay_source_at_execution.py"),
    }
    write_json(output_dir / "replay_results.json", result)
    if result["status"] != "match":
        raise RuntimeError("concrete focused Q-to-RoPE RTL mismatch proven")
    print(
        "ACE2_C4_ROPE_Q_REPLAY_PASS projection_outputs=64 rope_outputs=64 "
        f"replay_sha256={sha256_file(output_dir / 'replay_results.json')}"
    )


def write_sha256s(output_dir: Path) -> Path:
    excluded = {"SHA256SUMS", "verification.json"}
    paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name not in excluded
    )
    target = output_dir / "SHA256SUMS"
    target.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return target


def run_verify(output_dir: Path) -> None:
    prestate = verify_frozen_state()
    preflight = load_preflight(output_dir)
    witness = json.loads((output_dir / "witness.json").read_text(encoding="utf-8"))
    replay = json.loads((output_dir / "replay_results.json").read_text(encoding="utf-8"))
    preflight_source_archive = output_dir / "preflight_source_at_execution.py"
    capture_source_archive = output_dir / "capture_source_at_execution.py"
    replay_source_archive = output_dir / "replay_source_at_execution.py"
    if (
        not preflight_source_archive.is_file()
        or sha256_file(preflight_source_archive) != preflight["source"]["sha256"]
    ):
        raise RuntimeError("preflight-time source archive differs")
    expected_capture_source_sha256 = witness["source_bindings"][
        "tools/capture_replay_c4_rope_q.py"
    ]["sha256"]
    if (
        not capture_source_archive.is_file()
        or sha256_file(capture_source_archive) != expected_capture_source_sha256
    ):
        raise RuntimeError("capture-time source archive does not match the witness binding")
    if (
        not replay_source_archive.is_file()
        or sha256_file(replay_source_archive)
        != replay["replay_source_archive"]["sha256"]
    ):
        raise RuntimeError("replay-time source archive differs")
    if replay.get("status") != "match" or replay.get("mismatch_count") != 0:
        raise RuntimeError("replay is not a bit-exact match")
    if witness["capture"] != {
        **witness["capture"],
        "evaluation_witness_count": 1,
        "head_count": 1,
        "token_count": 1,
    }:
        raise RuntimeError("witness scope is not exactly one head/token")
    if witness["coordinate"] != {
        "batch": 0,
        "dataset": DATASET_NAME,
        "dataset_record_index": 64,
        "head": HEAD_INDEX,
        "layer": 0,
        "operator": "rope_q",
        "token": TOKEN_INDEX,
        "token_id": witness["coordinate"]["token_id"],
    }:
        raise RuntimeError("witness coordinate differs from the frozen coordinate")
    proof = preflight["metadata"]["proof"]
    if not proof.get("representable_nonzero_sine"):
        raise RuntimeError("verified preflight has no representable nonzero sine")
    if not proof.get("expected_rope_transform_non_identity"):
        raise RuntimeError("verified preflight does not prove a non-identity transform")
    if witness["rope"]["cos_q15_s16"] != preflight["metadata"]["cos_q15_s16"]:
        raise RuntimeError("witness cosine metadata differs from preflight")
    if witness["rope"]["sin_q15_s16"] != preflight["metadata"]["sin_q15_s16"]:
        raise RuntimeError("witness sine metadata differs from preflight")
    capture_consumption = output_dir / "capture_execution_consumed.json"
    if (
        not capture_consumption.is_file()
        or artifact(capture_consumption) != witness["capture"]["authorization_consumption"]
    ):
        raise RuntimeError("exactly-once capture consumption marker differs")
    if len(replay["expected_actual"]) != HEAD_DIM:
        raise RuntimeError("replay does not contain 64 bit-exact lane records")
    if witness["frozen_prestate"] != prestate:
        raise RuntimeError("sealed-path or pipeline prestate changed")
    sha256s = write_sha256s(output_dir)
    verification = {
        "schema_version": 1,
        "status": "engineer_complete_pending_fresh_reviewer",
        "verified_at_utc": utc_now(),
        "exactly_once_scope": {
            "cpu_capture_executions": 1,
            "evaluation_witnesses": 1,
            "heads": 1,
            "tokens": 1,
        },
        "bit_exact_replay": {
            "projection_outputs": 64,
            "rope_outputs": 64,
            "mismatches": 0,
        },
        "model_free_preflight": {
            "artifact": artifact(output_dir / "preflight.json"),
            "expected_rope_transform_non_identity": True,
            "nonidentity_lane_count": len(
                preflight["metadata"]["nonidentity_lane_indices"]
            ),
            "nonzero_sine_lane_count": len(
                preflight["metadata"]["nonzero_sine_lane_indices"]
            ),
            "representable_nonzero_sine": True,
        },
        "execution_source_archives": {
            "preflight": artifact(preflight_source_archive),
            "capture": artifact(capture_source_archive),
            "replay": artifact(replay_source_archive),
        },
        "rtl_change_required": False,
        "sealed_path_noninterference": prestate,
        "frontier_disposition": (
            "rope_q hardware semantics are bound only for this frozen witness; "
            "the accepted formal frontier remains through layer_0.v_proj pending Fresh Reviewer"
        ),
        "sha256s": artifact(sha256s),
    }
    write_json(output_dir / "verification.json", verification)
    print(
        "ACE2_C4_ROPE_Q_VERIFY_PASS exactly_once=1 mismatches=0 "
        f"sha256s_sha256={sha256_file(sha256s)}"
    )


def resolve_output(value: Path) -> Path:
    resolved = value if value.is_absolute() else ROOT / value
    resolved = resolved.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise SystemExit("--output-dir must be below the repository root")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "capture", "replay", "verify"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = resolve_output(args.output_dir)
    if args.mode == "preflight":
        run_preflight(output_dir)
    elif args.mode == "capture":
        run_capture(output_dir)
    elif args.mode == "replay":
        run_replay(output_dir)
    else:
        run_verify(output_dir)


if __name__ == "__main__":
    main()
