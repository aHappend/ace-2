#!/usr/bin/env python3
"""Derive layer-0 o_proj metadata from raw safetensors and replay unchanged RTL."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors import safe_open

from ace2_projection_reference import ProjectionCase, pack_int8, pack_meta, pack_w4, reference_projection


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MISSION_ID = "rtl-o-proj-offline-metadata-batch-replay-v1"
OUT = ROOT / f"evidence/verification/{MISSION_ID}"
BUILD = ROOT / f"build/{MISSION_ID}"
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    f"{MISSION_ID}/mission.json"
)
PREFIX_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "manager-record-attention-value-certified-frontier-v1/round-0003.json"
)
PRIOR = ROOT / "evidence/verification/rtl-attention-value-offline-v-authority-replay-v1"
CAPTURE = ROOT / "evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2"
SCALES = ROOT / "evidence/layer0_tile_bfp_score_attention_v1/paired-smoke-20260801-v1/derived_scales.json"
PPA_SUMS = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/SHA256SUMS"
SHELL_TB = ROOT / "verification/tb/ace2_shell_tb.sv"
SHELL = ROOT / "rtl/ace2_shell.sv"
PROJECTION_REFERENCE = ROOT / "tools/ace2_projection_reference.py"

MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_NAME = "model.layers.0.self_attn.o_proj.weight"
BIAS_NAME = "model.layers.0.self_attn.o_proj.bias"
HIDDEN = 896
INPUT_SCALE = 0.0021376722440944883
OUTPUT_SCALE = 0.0043983759842519685
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

EXPECTED = {
    "mission": "3e3d175da84b428bd9ff54339287b4cea8f76c1cc849f0ebb09d1f4e2e39e595",
    "prefix_review": "7920dca943558d5a1358e2ee198a788f7d5518bd31743240825d3b9c2d79caa6",
    "prior_sums": "ea258971aebee28c7ea680e9f4272c37b08c55afcdb77494011827107af8f199",
    "prior_report": "d03d8171dee9e2a7af6be06f70b7c97915a022c8aeb2ba115f17bfc363e57717",
    "capture_sums": "7e3c768d9eff34b327859621e0e093243cf7562837611ad0bd37480a86703e97",
    "capture_witness": "835c9d8f4432b2005998bac74c5536a9acd171f8042c33ac537af37d18869c3d",
    "scales": "78280eb606e0c14ea74163f72f45cbb045670fb26f400c054013b9726266ebaa",
    "ppa_sums": "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4",
    "shell_tb": "cefd1e4533f9e6de9f85583c0095f3eb2e81329c190f0e0044d045e31da76d18",
    "shell": "3bb8caab4f06e6be9b170b5b3d91cb89b237715132e52507cd60f0514c61ab30",
    "projection_reference": "648f969ce3e4dca6b4067da2ad41b5b5bf4cfe0a9d5420852827d085dbf57d11",
    "model": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    "rtl_tree": "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be",
    "qweight_s4_in_s8": "17cd84c9ac31fe6ff478f8114ab873a62446c6fb53a7bc33e6bb9af8d772f6c2",
    "input": "67be92a3d931be7421c14c861605794888a3e489c85e9147135411935644b88a",
    "output": "5217926db46bd1ffdbc9d9ad51ec11e1fc8295064041fd12fe011049c62bafb6",
}

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

FORBIDDEN_IMPORT_ROOTS = {"transformers", "datasets", "tokenizers"}
FORBIDDEN_CALL_NAMES = {
    "__call__",
    "forward",
    "register_forward_hook",
    "register_forward_pre_hook",
    "calibrate",
    "capture_layer0_inputs",
    "execute_capture_worker",
    "extract",
}


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


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path, public_path: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path.name}")
    return {
        "bytes": path.stat().st_size,
        "path": public_path or relative(path),
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def write_sha256s(base: Path) -> None:
    rows = []
    output = base / "SHA256SUMS"
    for member in sorted(base.rglob("*")):
        if member.is_file() and member != output:
            rows.append(f"{sha256_file(member)}  {member.relative_to(base).as_posix()}\n")
    output.write_text("".join(rows), encoding="utf-8")


def validate_manifest(path: Path, base: Path) -> int:
    count = 0
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row:
            continue
        expected, name = row.split("  ", 1)
        member = base / name
        require(member.is_file(), f"manifest member missing: {name}")
        require(sha256_file(member) == expected, f"manifest member changed: {name}")
        count += 1
    return count


def protected_state() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in PROTECTED_PATHS}


def tree_hash(pattern: str) -> dict[str, Any]:
    paths = sorted(ROOT.glob(pattern))
    rows = [(sha256_file(path), relative(path)) for path in paths]
    manifest = "".join(f"{digest}  {path}\n" for digest, path in rows).encode()
    return {
        "file_count": len(rows),
        "sha256": sha256_bytes(manifest),
        "hash_method": f"path-sorted sha256 manifest over {pattern}",
        "source_hashes": [{"path": path, "sha256": digest} for digest, path in rows],
    }


def direct_children() -> list[int]:
    path = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return [] if not text else [int(value) for value in text.split()]


def static_zero_model_guard(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad_imports: list[str] = []
    bad_calls: list[str] = []
    subprocess_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS:
                    bad_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                bad_imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                    subprocess_calls.append(name)
            if name in FORBIDDEN_CALL_NAMES:
                bad_calls.append(name)
    require(not bad_imports, f"forbidden model-stack imports: {bad_imports}")
    require(not bad_calls, f"forbidden model/capture calls: {bad_calls}")
    require(subprocess_calls == ["run"], f"subprocess API scope changed: {subprocess_calls}")
    return {
        "ast_parsed": True,
        "forbidden_import_count": 0,
        "forbidden_call_count": 0,
        "subprocess_apis": subprocess_calls,
        "allowed_subprocess_classes": ["iverilog_compile", "vvp_simulation"],
    }


class ModuleExecutionGuard:
    def __init__(self) -> None:
        self.module_call_count = 0
        self.direct_forward_count = 0
        self._call = torch.nn.Module.__call__
        self._call_impl = torch.nn.Module._call_impl
        self._profile = sys.getprofile()

    def __enter__(self) -> "ModuleExecutionGuard":
        guard = self

        def blocked(_module: torch.nn.Module, *_args: Any, **_kwargs: Any) -> Any:
            guard.module_call_count += 1
            raise RuntimeError("torch.nn.Module execution is forbidden")

        def profile(frame: Any, event: str, _arg: Any) -> Any:
            if event == "call" and frame.f_code.co_name == "forward":
                owner = frame.f_locals.get("self")
                if isinstance(owner, torch.nn.Module):
                    guard.direct_forward_count += 1
                    raise RuntimeError("direct torch.nn.Module.forward execution is forbidden")
            return profile

        torch.nn.Module.__call__ = blocked
        torch.nn.Module._call_impl = blocked
        sys.setprofile(profile)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        sys.setprofile(self._profile)
        torch.nn.Module.__call__ = self._call
        torch.nn.Module._call_impl = self._call_impl


def model_file() -> Path:
    path = (
        Path(HF_HUB_CACHE)
        / "models--Qwen--Qwen2.5-0.5B"
        / "snapshots"
        / MODEL_REVISION
        / "model.safetensors"
    )
    require(path.is_file(), "pinned model safetensors is unavailable")
    require(sha256_file(path) == EXPECTED["model"], "pinned model safetensors changed")
    return path


def round_shift_even(value: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    require(not bool(torch.any(shift < 0)) and not bool(torch.any(shift > 63)), "right shift outside 0..63")
    safe = shift.clamp(min=1)
    magnitude = value.abs()
    base = torch.bitwise_right_shift(magnitude, safe)
    shifted_one = torch.bitwise_left_shift(torch.ones_like(safe), safe)
    mask = torch.where(safe == 63, torch.full_like(safe, (1 << 63) - 1), shifted_one - 1)
    remainder = torch.bitwise_and(magnitude, mask)
    half = torch.bitwise_left_shift(torch.ones_like(safe), safe - 1)
    increment = (remainder > half) | ((remainder == half) & ((base & 1) == 1))
    rounded = base + increment.to(base.dtype)
    signed = torch.where(value < 0, -rounded, rounded)
    return torch.where(shift == 0, value, signed)


def derive_multiplier(real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    real = real.detach().to(torch.float64)
    require(not bool(torch.any(real < 0)) and bool(torch.all(torch.isfinite(real))), "invalid multiplier")
    multiplier = torch.zeros_like(real, dtype=torch.int64)
    right_shift = torch.full_like(real, -1, dtype=torch.int64)
    for shift in range(63, -1, -1):
        candidate = torch.round(real * math.ldexp(1.0, shift))
        select = (right_shift < 0) & (candidate <= (1 << 31) - 1)
        multiplier = torch.where(select, candidate.to(torch.int64), multiplier)
        right_shift = torch.where(select, torch.full_like(right_shift, shift), right_shift)
    require(not bool(torch.any(right_shift < 0)), "unrepresentable multiplier")
    return multiplier, right_shift


def tensor_bytes(record: dict[str, Any]) -> bytes:
    path = ROOT / record["path"]
    raw = path.read_bytes()
    require(len(raw) == record["bytes"], f"tensor byte count changed: {path.name}")
    require(sha256_bytes(raw) == record["sha256"], f"tensor hash changed: {path.name}")
    return raw


def i8_values(raw: bytes) -> list[int]:
    return [value - 256 if value >= 128 else value for value in raw]


def raw_i32(values: list[int]) -> bytes:
    return b"".join(struct.pack("<i", value) for value in values)


def raw_i64(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", value) for value in values)


def raw_f64(values: list[float]) -> bytes:
    return b"".join(struct.pack("<d", value) for value in values)


def pack_s4(values: list[int]) -> bytes:
    require(len(values) % 2 == 0, "signed-int4 image requires an even element count")
    return bytes((values[index] & 0xF) | ((values[index + 1] & 0xF) << 4) for index in range(0, len(values), 2))


def validate_sources() -> dict[str, Any]:
    checks = {
        MISSION: EXPECTED["mission"],
        PREFIX_REVIEW: EXPECTED["prefix_review"],
        PRIOR / "SHA256SUMS": EXPECTED["prior_sums"],
        PRIOR / "ordered_prefix_report.json": EXPECTED["prior_report"],
        CAPTURE / "SHA256SUMS": EXPECTED["capture_sums"],
        CAPTURE / "witness.json": EXPECTED["capture_witness"],
        SCALES: EXPECTED["scales"],
        PPA_SUMS: EXPECTED["ppa_sums"],
        SHELL_TB: EXPECTED["shell_tb"],
        SHELL: EXPECTED["shell"],
        PROJECTION_REFERENCE: EXPECTED["projection_reference"],
    }
    for path, expected in checks.items():
        require(sha256_file(path) == expected, f"bound source changed: {path.name}")
    member_counts = {
        "prior_replay": validate_manifest(PRIOR / "SHA256SUMS", PRIOR),
        "immutable_capture": validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE),
        "canonical_ppa": validate_manifest(PPA_SUMS, ROOT),
    }
    prior = json.loads((PRIOR / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    witness = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    scales = json.loads(SCALES.read_text(encoding="utf-8"))["linears"][WEIGHT_NAME.removesuffix(".weight")]
    require(prior["maximal_passing_prefix"] == ORDER[:10], "accepted attention-value prefix changed")
    require(prior["first_failing_or_unsupported_operator"]["operator"] == "layer_0.o_proj", "accepted boundary changed")
    require(witness["tensor_manifest"]["o_proj_input_token1_s8"]["sha256"] == EXPECTED["input"], "o_proj input changed")
    require(witness["tensor_manifest"]["o_proj_output_token1_s8"]["sha256"] == EXPECTED["output"], "o_proj output changed")
    require(scales["input_scale"] == INPUT_SCALE, "o_proj calibrated input scale changed")
    require(scales["hardware_input_scale"] == INPUT_SCALE, "o_proj hardware input scale changed")
    require(scales["output_scale"] == OUTPUT_SCALE, "o_proj output scale changed")
    require(scales["bias_accumulator"] is None, "o_proj unexpectedly gained bias metadata")
    require(scales["qweight_sha256"] == EXPECTED["qweight_s4_in_s8"], "accepted o_proj qweight hash changed")
    rtl = tree_hash("rtl/**/*.sv")
    require(rtl["file_count"] == 23, "RTL file count changed")
    require(rtl["sha256"] == EXPECTED["rtl_tree"], "RTL tree changed")
    return {
        "member_counts": member_counts,
        "prior": prior,
        "witness": witness,
        "scales": scales,
        "rtl": rtl,
        "constraints": tree_hash("constraints/**/*"),
        "source_guard": static_zero_model_guard(SELF),
    }


def derive(packages: dict[str, Any]) -> dict[str, Any]:
    witness = packages["witness"]
    input_record = witness["tensor_manifest"]["o_proj_input_token1_s8"]
    output_record = witness["tensor_manifest"]["o_proj_output_token1_s8"]
    input_values = i8_values(tensor_bytes(input_record))
    expected_output = i8_values(tensor_bytes(output_record))
    require(len(input_values) == HIDDEN and len(expected_output) == HIDDEN, "o_proj retained geometry changed")
    children_before = direct_children()
    require(not children_before, f"unexpected child processes before derivation: {children_before}")
    with ModuleExecutionGuard() as guard:
        with safe_open(model_file(), framework="pt", device="cpu") as weights:
            names = set(weights.keys())
            require(WEIGHT_NAME in names, "raw o_proj weight is missing")
            require(BIAS_NAME not in names, "raw o_proj bias unexpectedly exists")
            raw_weight = weights.get_tensor(WEIGHT_NAME)
            require(tuple(raw_weight.shape) == (HIDDEN, HIDDEN), "raw o_proj weight geometry changed")
            require(raw_weight.dtype == torch.bfloat16, "raw o_proj weight dtype changed")
            source_raw = raw_weight.contiguous().view(torch.uint16).numpy().tobytes(order="C")
            weight = raw_weight.to(torch.float64)
        weight_scale = weight.abs().amax(dim=1) / 7.0
        weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
        qweight = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
        multiplier, right_shift = derive_multiplier(INPUT_SCALE * weight_scale / OUTPUT_SCALE)
        bias = torch.zeros(HIDDEN, dtype=torch.int64)
        activations = torch.tensor(input_values, dtype=torch.int64)
        dot = (qweight.to(torch.int64) * activations[None, :]).sum(dim=1)
        accumulator = dot + bias
        require(not bool(torch.any(accumulator < -(1 << 31))) and not bool(torch.any(accumulator >= (1 << 31))), "o_proj accumulator overflow")
        rounded = round_shift_even(accumulator * multiplier, right_shift)
        output = rounded.clamp(-128, 127).to(torch.int8)
    require(guard.module_call_count == 0, "derivation attempted a torch module call")
    require(guard.direct_forward_count == 0, "derivation attempted a direct module forward")
    children_after = direct_children()
    require(not children_after, f"unexpected child processes after derivation: {children_after}")

    qweight_values = qweight.reshape(-1).to(torch.int64).tolist()
    multiplier_values = multiplier.tolist()
    shift_values = right_shift.tolist()
    bias_values = bias.tolist()
    dot_values = dot.tolist()
    accumulator_values = accumulator.tolist()
    output_values = output.to(torch.int64).tolist()
    require(sha256_bytes(bytes(value & 0xFF for value in qweight_values)) == EXPECTED["qweight_s4_in_s8"], "derived qweight hash differs")
    require(multiplier_values == packages["scales"]["multiplier"], "derived multiplier metadata differs")
    require(shift_values == packages["scales"]["right_shift"], "derived right-shift metadata differs")
    require(output_values == expected_output, "derived o_proj output differs from immutable reference")
    require(not any(bias_values), "bias accumulators are not all zero")

    independent = reference_projection(
        ProjectionCase(
            "c4_record64_layer0_token1_o_proj",
            1,
            HIDDEN,
            [input_values],
            qweight.to(torch.int64).tolist(),
            multiplier_values,
            shift_values,
            [0] * HIDDEN,
            bias_values,
        )
    )
    require(independent.outputs == [expected_output], "independent projection reference differs")
    return {
        "input_record": input_record,
        "output_record": output_record,
        "input": input_values,
        "expected_output": expected_output,
        "qweight": qweight.to(torch.int64).tolist(),
        "qweight_flat": qweight_values,
        "weight_scale": weight_scale.tolist(),
        "multiplier": multiplier_values,
        "right_shift": shift_values,
        "bias": bias_values,
        "dot": dot_values,
        "accumulator": accumulator_values,
        "output": output_values,
        "rounded": rounded.tolist(),
        "saturation_count": int(((rounded < -128) | (rounded > 127)).sum()),
        "source_weight_sha256": sha256_bytes(source_raw),
        "source_weight_bytes": len(source_raw),
        "guard": {
            "module_call_count": guard.module_call_count,
            "direct_forward_count": guard.direct_forward_count,
        },
        "process_children": {"before": children_before, "after": children_after},
        "independent_saturation_seen": independent.saturation_seen,
    }


def sv_hex(value: int, width: int = 128) -> str:
    return f"{width}'h{value & ((1 << width) - 1):0{width // 4}x}"


def chunks(values: list[int], count: int) -> list[list[int]]:
    return [values[index : index + count] for index in range(0, len(values), count)]


def render_vectors(derived: dict[str, Any]) -> str:
    lines = [
        "// Generated from pinned raw layer-0 o_proj weight and immutable scales.",
        "localparam integer PROJ_CASE_COUNT = 6;",
        "localparam integer PROJ_CASE_BALANCED = 0;",
        "localparam integer PROJ_CASE_SATURATION = 0;",
        "localparam integer PROJ_CASE_MLP_GATE = 1;",
        "localparam integer PROJ_CASE_MLP_DOWN = 2;",
        "localparam integer PROJ_CASE_RMSNORM_CONSUMER = 3;",
        "localparam integer PROJ_CASE_C4_V_BIAS = 4;",
        "localparam integer PROJ_C4_V_OUTPUT_CHANNEL = 0;",
        "localparam signed [31:0] PROJ_C4_V_DOT_ACC = 32'sd0;",
        "localparam signed [31:0] PROJ_C4_V_BIAS_ACC = 32'sd0;",
        "localparam signed [31:0] PROJ_C4_V_BIASED_ACC = 32'sd0;",
        "localparam signed [31:0] PROJ_C4_V_MULTIPLIER = 32'sd0;",
        "localparam [5:0] PROJ_C4_V_RIGHT_SHIFT = 6'd0;",
        "localparam [7:0] PROJ_C4_V_DOT_ONLY_OUTPUT = 8'd0;",
        "localparam [7:0] PROJ_C4_V_BIASED_OUTPUT = 8'd0;",
        "localparam integer PROJ_MAX_ROWS = 1;",
        "localparam integer PROJ_MAX_K = 4864;",
        "localparam integer PROJ_MAX_GROUPS = 1216;",
        "localparam integer PROJ_GROUP_INDEX_WIDTH = 11;",
        "localparam integer PROJ_BEATS = 56;",
        "localparam integer PROJ_INPUT_BEATS = 56;",
        "localparam integer PROJ_MAX_INPUT_BEATS = 56;",
        "localparam integer PROJ_MAX_OUTPUTS = 896;",
        "localparam integer PROJ_MAX_OUTPUT_BEATS = 56;",
        "localparam integer PROJ_MAC_LANES = 4;",
        "localparam integer PROJ_GROUPS_PER_WEIGHT_BEAT = 4;",
        "localparam integer PROJ_WEIGHT_BEATS_PER_OUTPUT = 56;",
        "localparam integer PROJ_MAX_WEIGHT_BEATS_PER_OUTPUT = 56;",
        "localparam integer PROJ_TOTAL_WEIGHT_BEATS = 50176;",
        "localparam integer PROJ_TOTAL_META_BEATS = 896;",
        "reg [15:0] proj_case_rows [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_k [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_input_beats [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_groups [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_outputs [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_output_beats [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_weight_beats_per_output [0:PROJ_CASE_COUNT-1];",
        "reg [31:0] proj_case_weight_offset [0:PROJ_CASE_COUNT-1];",
        "reg [31:0] proj_case_meta_offset [0:PROJ_CASE_COUNT-1];",
        "reg proj_expected_saturation [0:PROJ_CASE_COUNT-1];",
        "reg [8*16-1:0] proj_input_beats [0:PROJ_CASE_COUNT*PROJ_MAX_ROWS*PROJ_MAX_INPUT_BEATS-1];",
        "reg [4*32-1:0] proj_weight_beats [0:PROJ_TOTAL_WEIGHT_BEATS-1];",
        "reg [8*16-1:0] proj_meta_beats [0:PROJ_TOTAL_META_BEATS-1];",
        "reg [8*16-1:0] proj_expected_beats [0:PROJ_CASE_COUNT*PROJ_MAX_ROWS*PROJ_MAX_OUTPUT_BEATS-1];",
        "integer oproj_init_case;",
        "initial begin",
        "  for (oproj_init_case = 0; oproj_init_case < PROJ_CASE_COUNT; oproj_init_case = oproj_init_case + 1) begin",
        "    proj_case_rows[oproj_init_case] = 16'd1;",
        "    proj_case_k[oproj_init_case] = 16'd896;",
        "    proj_case_input_beats[oproj_init_case] = 16'd56;",
        "    proj_case_groups[oproj_init_case] = 16'd224;",
        "    proj_case_outputs[oproj_init_case] = 16'd896;",
        "    proj_case_output_beats[oproj_init_case] = 16'd56;",
        "    proj_case_weight_beats_per_output[oproj_init_case] = 16'd56;",
        "    proj_case_weight_offset[oproj_init_case] = 32'd0;",
        "    proj_case_meta_offset[oproj_init_case] = 32'd0;",
        "    proj_expected_saturation[oproj_init_case] = 1'b0;",
        "  end",
    ]
    for beat, values in enumerate(chunks(derived["input"], 16)):
        lines.append(f"  proj_input_beats[{beat}] = {sv_hex(pack_int8(values))};")
    for output, row in enumerate(derived["qweight"]):
        for beat, values in enumerate(chunks(row, 16)):
            lines.append(f"  proj_weight_beats[{output * 56 + beat}] = {sv_hex(pack_w4(values))};")
        lines.append(
            f"  proj_meta_beats[{output}] = {sv_hex(pack_meta(derived['multiplier'][output], derived['right_shift'][output], 0, derived['bias'][output]))};"
        )
    for beat, values in enumerate(chunks(derived["expected_output"], 16)):
        lines.append(f"  proj_expected_beats[{beat}] = {sv_hex(pack_int8(values))};")
    lines.append("end")
    return "\n".join(lines) + "\n"


def execution_testbench() -> str:
    text = SHELL_TB.read_text(encoding="utf-8")
    text = text.replace(
        '    `include "../generated/projection_vectors.svh"',
        '    `include "c4_o_proj_offline_vectors.svh"',
        1,
    )
    require('`include "c4_o_proj_offline_vectors.svh"' in text, "projection include replacement failed")
    text = text.replace(
        "    integer qproj_stride_only_mode;",
        "    integer qproj_stride_only_mode;\n    integer c4_o_proj_offline_replay_mode;",
        1,
    )
    text = text.replace(
        '        qproj_stride_only_mode = $test$plusargs("QPROJ_STRIDE_ONLY");',
        '        qproj_stride_only_mode = $test$plusargs("QPROJ_STRIDE_ONLY");\n'
        '        c4_o_proj_offline_replay_mode = $test$plusargs("C4_O_PROJ_OFFLINE_REPLAY");',
        1,
    )
    branch = (
        "        if (c4_o_proj_offline_replay_mode) begin\n"
        "            send_oproj_cmd(0, 0, 16'h3e04);\n"
        "            wait_qproj_done_and_compare(0, 0, 16'h3e04);\n"
        "            if (failures != 0) begin\n"
        "                $display(\"ACE2_C4_OPROJ_OFFLINE_FAIL failures=%0d\", failures);\n"
        "                $fatal(1, \"ACE2_C4_OPROJ_OFFLINE_FAIL\");\n"
        "            end\n"
        "            for (case_index = 0; case_index < observed_count; case_index = case_index + 1) begin\n"
        "                $display(\"ACE2_C4_OPROJ_BEAT beat=%0d data=%032x\", case_index, observed_output[case_index]);\n"
        "            end\n"
        "            $display(\"ACE2_C4_OPROJ_OFFLINE_PASS writes=%0d cycles=%0d saturation=%0d\", observed_count, last_command_cycles, cmd_done_saturation);\n"
        "            $display(\"ACE2_C4_OPROJ_ORDERED_REPLAY_PASS operators=1\");\n"
        "            $finish;\n"
        "        end\n\n"
    )
    needle = "        if (qproj_stride_only_mode) begin"
    require(needle in text, "focused replay insertion point changed")
    return text.replace(needle, branch + needle, 1)


def run_command(command: list[str], classification: str) -> subprocess.CompletedProcess[str]:
    allowed = {"iverilog": "iverilog_compile", "vvp": "vvp_simulation"}
    require(command[0] in allowed, f"forbidden subprocess executable: {command[0]}")
    require(allowed[command[0]] == classification, "subprocess classification differs")
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def persist_derived(derived: dict[str, Any]) -> dict[str, Any]:
    directory = OUT / "derived"
    directory.mkdir()
    payloads = {
        "o_proj_weights_s4_packed": pack_s4(derived["qweight_flat"]),
        "o_proj_weights_s4_in_s8": bytes(value & 0xFF for value in derived["qweight_flat"]),
        "o_proj_weight_scale_f64": raw_f64(derived["weight_scale"]),
        "o_proj_bias_accumulator_s32": raw_i32(derived["bias"]),
        "o_proj_multiplier_s32": raw_i32(derived["multiplier"]),
        "o_proj_right_shift_u6": bytes(derived["right_shift"]),
        "o_proj_dot_accumulator_s32": raw_i32(derived["dot"]),
        "o_proj_biased_accumulator_s32": raw_i32(derived["accumulator"]),
        "o_proj_output_s8": bytes(value & 0xFF for value in derived["output"]),
    }
    manifest: dict[str, Any] = {}
    for name, raw in payloads.items():
        path = directory / f"{name}.bin"
        path.write_bytes(raw)
        manifest[name] = artifact(path)
    return manifest


def run() -> None:
    require(not OUT.exists(), "o_proj evidence output already exists")
    require(shutil.which("iverilog") is not None, "iverilog is unavailable")
    require(shutil.which("vvp") is not None, "vvp is unavailable")
    packages = validate_sources()
    protected_before = protected_state()
    ppa_before = sha256_file(PPA_SUMS)
    rtl_before = packages["rtl"]
    constraints_before = packages["constraints"]
    derived = derive(packages)
    OUT.mkdir(parents=True)
    generated = OUT / "generated"
    generated.mkdir()
    vectors = generated / "c4_o_proj_offline_vectors.svh"
    vectors.write_text(render_vectors(derived), encoding="utf-8")
    shutil.copy2(SELF, OUT / "replay_source_at_execution.py")
    tb = OUT / "replay_tb_source_at_execution.sv"
    tb.write_text(execution_testbench(), encoding="utf-8")
    tensor_manifest = persist_derived(derived)
    metadata = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "completed_at_utc": utc_now(),
        "model": {
            "repository": "Qwen/Qwen2.5-0.5B",
            "revision": MODEL_REVISION,
            "safetensors_sha256": EXPECTED["model"],
            "tensor_name": WEIGHT_NAME,
            "tensor_dtype": "bfloat16",
            "tensor_shape": [HIDDEN, HIDDEN],
            "tensor_raw_sha256": derived["source_weight_sha256"],
            "tensor_raw_bytes": derived["source_weight_bytes"],
            "bias_tensor_present": False,
        },
        "fixed_point_contract": {
            "weight_quantization": "symmetric_per_output_channel_signed_int4_complete_896_reduction",
            "input_scale": INPUT_SCALE,
            "output_scale": OUTPUT_SCALE,
            "accumulator": "signed_int32_complete_reduction",
            "rounding": "round_to_nearest_ties_to_even",
            "output": "signed_int8_saturating",
            "bias_policy": "absent_source_bias_maps_to_896_zero_signed_int32_accumulators",
        },
        "derived_artifacts": tensor_manifest,
        "derived_hashes": {
            "qweight_s4_in_s8": EXPECTED["qweight_s4_in_s8"],
            "qweight_s4_packed": tensor_manifest["o_proj_weights_s4_packed"]["sha256"],
            "weight_scale_f64": tensor_manifest["o_proj_weight_scale_f64"]["sha256"],
            "bias_accumulator_s32": tensor_manifest["o_proj_bias_accumulator_s32"]["sha256"],
            "multiplier_s32": tensor_manifest["o_proj_multiplier_s32"]["sha256"],
            "right_shift_u6": tensor_manifest["o_proj_right_shift_u6"]["sha256"],
        },
        "accepted_scale_metadata_exact_match": True,
        "immutable_input": derived["input_record"],
        "immutable_output": derived["output_record"],
        "independent_integer_reference_exact_match": True,
        "zero_model_execution_guard": derived["guard"],
        "process_children": derived["process_children"],
        "source_bindings": {
            "mission": artifact(MISSION, f"handoff:{MISSION_ID}/mission.json"),
            "accepted_prefix_review": artifact(PREFIX_REVIEW, "handoff:manager-record-attention-value-certified-frontier-v1/round-0003.json"),
            "accepted_attention_value_replay": artifact(PRIOR / "SHA256SUMS"),
            "immutable_capture": artifact(CAPTURE / "SHA256SUMS"),
            "immutable_scales": artifact(SCALES),
            "projection_reference": artifact(PROJECTION_REFERENCE),
            "rtl_tree_sha256": rtl_before["sha256"],
            "constraint_tree_sha256": constraints_before["sha256"],
        },
    }
    write_json(OUT / "derived_metadata.json", metadata)

    BUILD.mkdir(parents=True, exist_ok=True)
    image = BUILD / "ordered.vvp"
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    compile_command = [
        "iverilog",
        "-g2012",
        "-Wall",
        "-Irtl",
        "-Irtl/generated",
        "-Iverification/generated",
        "-Iverification/tb",
        f"-I{relative(generated)}",
        "-s",
        "ace2_shell_tb",
        "-o",
        relative(image),
        *[relative(path) for path in rtl_sources],
        relative(tb),
    ]
    compiled = run_command(compile_command, "iverilog_compile")
    (OUT / "iverilog.log").write_text(
        "command=" + " ".join(compile_command) + "\n" + compiled.stdout + compiled.stderr,
        encoding="utf-8",
    )
    require(compiled.returncode == 0, "unchanged-tree o_proj compilation failed")
    replay_command = ["vvp", relative(image), "+C4_O_PROJ_OFFLINE_REPLAY"]
    replayed = run_command(replay_command, "vvp_simulation")
    (OUT / "replay.stdout").write_text(replayed.stdout, encoding="utf-8")
    (OUT / "replay.stderr").write_text(replayed.stderr, encoding="utf-8")
    (OUT / "replay.log").write_text(
        "command=" + " ".join(replay_command) + "\n" + replayed.stdout + replayed.stderr,
        encoding="utf-8",
    )
    require(replayed.returncode == 0, "unchanged-tree o_proj execution failed")
    require("ACE2_C4_OPROJ_ORDERED_REPLAY_PASS operators=1" in replayed.stdout, "ordered replay pass marker missing")
    require("MISMATCH" not in replayed.stdout and "_FAIL" not in replayed.stdout, "o_proj replay reported failure")
    summary = re.search(r"ACE2_C4_OPROJ_OFFLINE_PASS writes=(\d+) cycles=(\d+) saturation=(\d+)", replayed.stdout)
    require(summary is not None, "o_proj replay summary missing")
    beat_rows = re.findall(r"ACE2_C4_OPROJ_BEAT beat=(\d+) data=([0-9a-fA-F]+)", replayed.stdout)
    require(len(beat_rows) == 56, f"o_proj output beat count changed: {len(beat_rows)}")
    require([int(row[0]) for row in beat_rows] == list(range(56)), "o_proj beat order changed")
    rtl_raw = b"".join(int(row[1], 16).to_bytes(16, "little") for row in beat_rows)
    require(rtl_raw == bytes(value & 0xFF for value in derived["expected_output"]), "RTL o_proj bytes differ")
    require(int(summary.group(1)) == 56, "o_proj write count changed")
    require(int(summary.group(3)) == int(derived["independent_saturation_seen"]), "o_proj saturation flag changed")

    residual_output = packages["witness"]["tensor_manifest"]["attention_residual_token1_s8"]
    boundary = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "operator": "layer_0.attention_residual_add",
        "classification": "UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_IMMUTABLE_RESIDUAL_SOURCE_INPUT_ABSENT",
        "available_inputs": {
            "o_proj_output_token1_s8": derived["output_record"],
            "o_proj_output_scale": OUTPUT_SCALE,
            "attention_residual_output_token1_s8": residual_output,
            "attention_residual_destination_scale": packages["witness"]["operator_metadata"]["attention_residual_scale"],
        },
        "missing_fields": [
            "layer_0.attention_residual_add.input.residual_source_token1_s8_at_destination_scale[896]",
            "layer_0.attention_residual_add.input.hash_bound_two_operand_descriptor_payload",
        ],
        "scope_reason": "The immutable package retains o_proj input/output and the post-add result, but not the original layer-0 residual operand quantized at the frozen destination scale. This mission authorizes raw o_proj weight derivation only; no model/capture or unrelated raw-tensor reconstruction is permitted.",
        "simulation_executed": False,
        "later_operators_attempted": False,
    }
    write_json(OUT / "boundary_probe.json", boundary)
    zero = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "completed_at_utc": utc_now(),
        "status": "PASS_ZERO_MODEL_EXECUTION_RTL_ONLY_SUBPROCESSES",
        "model_execution_count": 0,
        "model_forward_count": 0,
        "model_call_count": 0,
        "calibration_forward_count": 0,
        "capture_count": 0,
        "hook_count": 0,
        "extraction_worker_count": 0,
        "subprocess_count": 2,
        "subprocess_classes": ["iverilog_compile", "vvp_simulation"],
        "torch_module_guard": derived["guard"],
        "process_children_before_after_derivation": derived["process_children"],
        "static_source_guard": packages["source_guard"],
        "source": artifact(OUT / "replay_source_at_execution.py"),
    }
    write_json(OUT / "zero_model_execution.json", zero)

    protected_after = protected_state()
    ppa_after = sha256_file(PPA_SUMS)
    rtl_after = tree_hash("rtl/**/*.sv")
    constraints_after = tree_hash("constraints/**/*")
    require(protected_after == protected_before, "protected state changed")
    require(ppa_after == ppa_before == EXPECTED["ppa_sums"], "canonical PPA aggregate changed")
    require(rtl_after == rtl_before, "RTL tree changed during replay")
    require(constraints_after == constraints_before, "constraint tree changed during replay")

    oproj_record = {
        "operator": "layer_0.o_proj",
        "status": "PASS_EXACT",
        "input_hashes": {
            "immutable_input_s8": derived["input_record"]["sha256"],
            "raw_source_weight_bf16": derived["source_weight_sha256"],
            "qweight_s4_in_s8": EXPECTED["qweight_s4_in_s8"],
            "qweight_s4_packed": tensor_manifest["o_proj_weights_s4_packed"]["sha256"],
            "bias_accumulator_s32": tensor_manifest["o_proj_bias_accumulator_s32"]["sha256"],
            "multiplier_s32": tensor_manifest["o_proj_multiplier_s32"]["sha256"],
            "right_shift_u6": tensor_manifest["o_proj_right_shift_u6"]["sha256"],
        },
        "reference_hashes": {
            "dot_accumulator_s32": tensor_manifest["o_proj_dot_accumulator_s32"]["sha256"],
            "biased_accumulator_s32": tensor_manifest["o_proj_biased_accumulator_s32"]["sha256"],
            "immutable_output_s8": derived["output_record"]["sha256"],
        },
        "output_hashes": {"rtl_o_proj_output_s8": sha256_bytes(rtl_raw)},
        "counts": {
            "input_elements": HIDDEN,
            "weight_elements": HIDDEN * HIDDEN,
            "output_channels": HIDDEN,
            "metadata_channels": HIDDEN,
            "bias_nonzero_channels": 0,
            "compared_outputs": HIDDEN,
            "mismatches": 0,
            "rtl_output_writes": int(summary.group(1)),
            "reference_saturations": derived["saturation_count"],
        },
        "cycles": {"total_command_cycles": int(summary.group(2))},
        "handshake_latency": {
            "simulation_executed": True,
            "handshake_result": "PASS_NO_TIMEOUT_OR_DROPPED_WRITE",
            "latency_result": "PASS_BOUNDED_BY_MAINTAINED_SHELL_WATCHDOG",
        },
        "invariants": {
            "raw_weight_only_no_model_construction": True,
            "accepted_scale_metadata_exact_match": True,
            "independent_integer_reference_exact_match": True,
            "rtl_output_byte_exact_reference_match": True,
            "rtl_saturation_matches_reference": True,
            "rtl_tree_unchanged": True,
            "constraints_unchanged": True,
        },
        "rtl_source_hashes": {
            "rtl_tree": rtl_after["sha256"],
            "constraint_tree": constraints_after["sha256"],
            "shell": artifact(SHELL),
            "maintained_harness": artifact(SHELL_TB),
            "execution_harness": artifact(tb),
            "generated_vectors": artifact(vectors),
        },
        "raw_log": artifact(OUT / "replay.log"),
    }
    boundary_record = {
        "operator": boundary["operator"],
        "status": boundary["classification"],
        "counts": {"missing_input_tensors": 1, "later_operators_attempted": 0},
        "handshake_latency": {
            "simulation_executed": False,
            "handshake_result": "NOT_EVALUATED_INTERFACE_INPUT_ABSENT",
            "latency_result": "NOT_EVALUATED_INTERFACE_INPUT_ABSENT",
        },
        "boundary_probe": artifact(OUT / "boundary_probe.json"),
    }
    report = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "producer_role": "engineer",
        "review_status": "PENDING_FRESH_REVIEWER",
        "completed_at_utc": utc_now(),
        "frozen_order": ORDER,
        "prior_maximal_prefix": ORDER[:10],
        "attempted_operators": [oproj_record, boundary_record],
        "newly_passing_operators": ["layer_0.o_proj"],
        "maximal_passing_prefix": ORDER[:11],
        "first_failing_or_unsupported_operator": {
            "operator": boundary["operator"],
            "classification": boundary["classification"],
            "boundary_probe": artifact(OUT / "boundary_probe.json"),
        },
        "continuous_replay": {
            "start_operator": "layer_0.o_proj",
            "last_passing_operator": "layer_0.o_proj",
            "first_real_boundary": boundary["operator"],
            "later_operators_attempted": False,
        },
        "batch_result": "STOPPED_AT_FIRST_PRECISELY_UNAVAILABLE_EXECUTABLE_INTERFACE",
        "later_operators_attempted": False,
        "zero_model_execution": artifact(OUT / "zero_model_execution.json"),
        "derived_metadata": artifact(OUT / "derived_metadata.json"),
        "rtl_tree": rtl_after,
        "constraint_tree": constraints_after,
        "canonical_ppa": {
            "reused_not_rerun": True,
            "pre_post_exact_match": True,
            "sha256_manifest": artifact(PPA_SUMS),
        },
        "protected_state_pre_post_exact_match": True,
        "forbidden_flows_run": [],
        "state_mutations": {
            "rtl": False,
            "frontier_authority_seal": False,
            "synthesis_opensta_ppa": False,
            "baseline_candidate_benchmark_scale32": False,
        },
        "source_bindings": metadata["source_bindings"],
        "member_counts": packages["member_counts"],
        "review_requirement": "Fresh Reviewer must independently accept or reject this o_proj offline-metadata replay boundary.",
    }
    write_json(OUT / "ordered_prefix_report.json", report)
    (OUT / "batch_replay.log").write_text(
        "ACE2_O_PROJ_OFFLINE_METADATA_BATCH_REPLAY\n"
        f"rtl_tree_sha256={rtl_after['sha256']}\n"
        "model_execution_count=0\n"
        "capture_count=0\n"
        f"layer_0.o_proj=PASS_EXACT outputs=896 mismatches=0 cycles={int(summary.group(2))}\n"
        f"layer_0.attention_residual_add={boundary['classification']}\n"
        "maximal_passing_operator=layer_0.o_proj\n"
        "later_operators_attempted=false\n"
        "synthesis_opensta_ppa_executed=false\n",
        encoding="utf-8",
    )
    write_sha256s(OUT)
    print(
        "ACE2_O_PROJ_OFFLINE_METADATA_REPLAY_BOUNDARY "
        f"cycles={int(summary.group(2))} maximal_prefix=layer_0.o_proj "
        "first_boundary=layer_0.attention_residual_add mismatches=0 model_execution_count=0"
    )


def check() -> None:
    require(OUT.is_dir(), "o_proj evidence output is missing")
    packages = validate_sources()
    members = validate_manifest(OUT / "SHA256SUMS", OUT)
    require(sha256_file(OUT / "replay_source_at_execution.py") == sha256_file(SELF), "archived replay source changed")
    derived = derive(packages)
    metadata = json.loads((OUT / "derived_metadata.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    zero = json.loads((OUT / "zero_model_execution.json").read_text(encoding="utf-8"))
    require(metadata["model"]["tensor_raw_sha256"] == derived["source_weight_sha256"], "raw source tensor binding changed")
    require(metadata["immutable_input"]["sha256"] == EXPECTED["input"], "immutable input binding changed")
    require(metadata["immutable_output"]["sha256"] == EXPECTED["output"], "immutable output binding changed")
    require(report["newly_passing_operators"] == ["layer_0.o_proj"], "passing operator set changed")
    require(report["maximal_passing_prefix"] == ORDER[:11], "maximal prefix changed")
    require(report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.attention_residual_add", "first boundary changed")
    require(report["later_operators_attempted"] is False, "replay continued beyond first boundary")
    oproj = report["attempted_operators"][0]
    require(oproj["status"] == "PASS_EXACT", "o_proj is not exact")
    require(oproj["counts"]["mismatches"] == 0 and oproj["counts"]["compared_outputs"] == HIDDEN, "o_proj counts changed")
    require(oproj["output_hashes"]["rtl_o_proj_output_s8"] == EXPECTED["output"], "RTL o_proj output hash changed")
    for name in (
        "model_execution_count",
        "model_forward_count",
        "model_call_count",
        "calibration_forward_count",
        "capture_count",
        "hook_count",
        "extraction_worker_count",
    ):
        require(zero[name] == 0, f"zero-execution counter changed: {name}")
    require(zero["subprocess_classes"] == ["iverilog_compile", "vvp_simulation"], "subprocess scope changed")
    require(protected_state() == protected_state(), "protected state read is unstable")
    require(sha256_file(PPA_SUMS) == EXPECTED["ppa_sums"], "canonical PPA aggregate changed")
    require(tree_hash("rtl/**/*.sv")["sha256"] == EXPECTED["rtl_tree"], "RTL tree changed")
    print(
        "ACE2_O_PROJ_OFFLINE_METADATA_REPLAY_CHECK_PASS "
        f"members={members} report_sha256={sha256_file(OUT / 'ordered_prefix_report.json')} "
        "mismatches=0 model_execution_count=0 first_boundary=layer_0.attention_residual_add"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "check"))
    args = parser.parse_args()
    if args.mode == "run":
        run()
    else:
        check()


if __name__ == "__main__":
    main()
