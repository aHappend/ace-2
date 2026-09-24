#!/usr/bin/env python3
"""Build one forward-free authority bundle and replay the remaining layer-0 tail."""

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
from typing import Any, Iterable, Iterator

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors import safe_open

from ace2_mlp_residual_reference import MlpResidualCase, reference_mlp_residual_add
from ace2_projection_reference import ProjectionCase, pack_int8, pack_meta, pack_w4, reference_projection
from ace2_residual_reference import reference_residual_add
from ace2_rmsnorm_reference import derive_scaled_gains_q8, reference_rmsnorm
from ace2_silu_gate_reference import SILU_LUT, SiluGateCase, reference_silu_gate


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MISSION_ID = "rtl-layer0-remaining-offline-authority-batch-replay-v1"
OUT = ROOT / f"evidence/verification/{MISSION_ID}"
BUILD = ROOT / f"build/{MISSION_ID}"
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    f"{MISSION_ID}/mission.json"
)
PREFIX_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "manager-record-o-proj-certified-frontier-v1/round-0002.json"
)
PRIOR = ROOT / "evidence/verification/rtl-o-proj-offline-metadata-batch-replay-v1"
CAPTURE = ROOT / "evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2"
RECONSTRUCTION = (
    ROOT
    / "evidence/diagnostics/c4-layer0-attention-value-token0-v-offline-reconstruction-20260803-v1"
)
TOKEN_IDS = RECONSTRUCTION / "tensors/token_ids_u32le.bin"
SCALES = ROOT / "evidence/layer0_tile_bfp_score_attention_v1/paired-smoke-20260801-v1/derived_scales.json"
PPA_SUMS = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/SHA256SUMS"
SHELL_TB = ROOT / "verification/tb/ace2_shell_tb.sv"
SHELL = ROOT / "rtl/ace2_shell.sv"
CONFIG = (
    Path(HF_HUB_CACHE)
    / "models--Qwen--Qwen2.5-0.5B"
    / "snapshots"
    / "060db6499f32faf8b98477b0a26969ef7d8b9987"
    / "config.json"
)
MODEL = CONFIG.parent / "model.safetensors"

MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
HIDDEN = 896
INTERMEDIATE = 4864
TOKEN_INDEX = 1
TOKEN_ID = 33597
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
TAIL = ORDER[11:]

EXPECTED = {
    "mission": "19ee10d014a01e6e5e8bf15729b6b53f7b4db6a5a788e136fabce4b7bac4fdd2",
    "prefix_review": "54ec4b9c974c2ea9dd6060a0534dffcd79871bd3f8903da59b97f48f4f09c3e7",
    "prior_sums": "0f3096299647b33e7dcb841a0429d4002811f6da2d7d4e06c5887db1fb248a10",
    "prior_report": "69fc76ae1c6663c5ca8e0ab963e1d354e30fcc031fa791a01aad3d4fa089cef6",
    "capture_sums": "7e3c768d9eff34b327859621e0e093243cf7562837611ad0bd37480a86703e97",
    "capture_witness": "835c9d8f4432b2005998bac74c5536a9acd171f8042c33ac537af37d18869c3d",
    "reconstruction": "8e0a3d43579d636bb607e94f252b05eefa1cb8286b0b0119567d53244c84f3bf",
    "token_ids": "786d5f77ef052be6c25c07df04fdb7efeb3284adfbaf7571215178dfb488232e",
    "scales": "78280eb606e0c14ea74163f72f45cbb045670fb26f400c054013b9726266ebaa",
    "fixed_rules": "b5f1c1800560894a728f199d36b31ef9d624b5afa9d56ee544b0d467270c1c74",
    "residual_reference": "060b591327d7de0066efe39a9a9bcea1ba3ed49c9528383679f0a62c4367fa40",
    "rmsnorm_reference": "400cd5c4858f08b78283bd1bc18fe8bfc6bec719819e9296fc2ccc5d62a79c8a",
    "projection_reference": "648f969ce3e4dca6b4067da2ad41b5b5bf4cfe0a9d5420852827d085dbf57d11",
    "silu_reference": "c8d7ef5e872570f3e7f1f4d04f4978d8022f1bd2285638b6b48100bb115cf046",
    "mlp_residual_reference": "d5fd78ea9826d48043fe912e8d0211c58b40c6a0467c32d1e6c375d09df01b5e",
    "shell_tb": "cefd1e4533f9e6de9f85583c0095f3eb2e81329c190f0e0044d045e31da76d18",
    "shell": "3bb8caab4f06e6be9b170b5b3d91cb89b237715132e52507cd60f0514c61ab30",
    "ppa_sums": "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4",
    "config": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
    "model": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    "rtl_tree": "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be",
}

REFERENCE_PATHS = {
    "fixed_rules": ROOT / "tools/ace2_full_model_fixed_point.py",
    "residual_reference": ROOT / "tools/ace2_residual_reference.py",
    "rmsnorm_reference": ROOT / "tools/ace2_rmsnorm_reference.py",
    "projection_reference": ROOT / "tools/ace2_projection_reference.py",
    "silu_reference": ROOT / "tools/ace2_silu_gate_reference.py",
    "mlp_residual_reference": ROOT / "tools/ace2_mlp_residual_reference.py",
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


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path, public_path: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": public_path or relative(path),
        "sha256": sha256_file(path),
    }


def tree_hash(pattern: str) -> dict[str, Any]:
    paths = sorted(ROOT.glob(pattern))
    rows = [(sha256_file(path), relative(path)) for path in paths]
    payload = "".join(f"{digest}  {path}\n" for digest, path in rows).encode()
    return {
        "file_count": len(rows),
        "sha256": sha256_bytes(payload),
        "hash_method": f"path-sorted sha256 manifest over {pattern}",
        "source_hashes": [{"path": path, "sha256": digest} for digest, path in rows],
    }


def protected_state() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in PROTECTED_PATHS}


def validate_manifest(path: Path, base: Path) -> int:
    count = 0
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row:
            continue
        expected, name = row.split("  ", 1)
        candidate = Path(name)
        if candidate.is_absolute():
            member = candidate
        elif (base / candidate).is_file():
            member = base / candidate
        else:
            member = ROOT / candidate
        require(member.is_file(), f"manifest member missing: {name}")
        require(sha256_file(member) == expected, f"manifest member changed: {name}")
        count += 1
    return count


def write_sha256s(base: Path, output_name: str = "SHA256SUMS") -> None:
    output = base / output_name
    rows = []
    for member in sorted(base.rglob("*")):
        if member.is_file() and member != output:
            rows.append(f"{sha256_file(member)}  {member.relative_to(base).as_posix()}\n")
    output.write_text("".join(rows), encoding="utf-8")


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


def raw_tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes(order="C")
    return tensor.numpy().tobytes(order="C")


def tensor_record_bytes(record: dict[str, Any]) -> bytes:
    path = ROOT / record["path"]
    raw = path.read_bytes()
    require(len(raw) == record["bytes"], f"tensor byte count changed: {path.name}")
    require(sha256_bytes(raw) == record["sha256"], f"tensor hash changed: {path.name}")
    return raw


def i8_values(raw: bytes) -> list[int]:
    return [value - 256 if value >= 128 else value for value in raw]


def quantize_int8(values: torch.Tensor, scale: float) -> list[int]:
    require(math.isfinite(scale) and scale > 0, "invalid activation scale")
    return torch.round(values.to(torch.float64) / scale).clamp(-128, 127).to(torch.int8).to(torch.int64).tolist()


def round_shift_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (base & 1)):
        base += 1
    return sign * base


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


def infer_bf16_input_scale_from_silu_metadata(multiplier: int, right_shift: int) -> tuple[float, float]:
    """Recover the unique BF16 calibration absmax implied by frozen SiLU metadata."""
    bit_patterns = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.uint16)
    values = bit_patterns.view(torch.bfloat16).to(torch.float64).tolist()
    matches: list[tuple[float, float]] = []
    for absmax in values:
        scale = absmax / 127.0
        real_multiplier = 1.0 / ((1 << 21) * scale)
        candidate: tuple[int, int] | None = None
        for shift in range(63, -1, -1):
            value = round(real_multiplier * math.ldexp(1.0, shift))
            if value <= (1 << 31) - 1:
                candidate = (value, shift)
                break
        if candidate == (multiplier, right_shift):
            matches.append((absmax, scale))
    require(len(matches) == 1, f"SiLU metadata maps to {len(matches)} BF16 input scales")
    return matches[0]


def raw_i16(values: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<h", int(value)) for value in values)


def raw_i32(values: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<i", int(value)) for value in values)


def raw_i64(values: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<q", int(value)) for value in values)


def raw_u32(values: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<I", int(value)) for value in values)


def raw_f64(values: Iterable[float]) -> bytes:
    return b"".join(struct.pack("<d", float(value)) for value in values)


def packed_s4(tensor: torch.Tensor) -> bytes:
    flat = tensor.reshape(-1).to(torch.int16)
    require(flat.numel() % 2 == 0, "signed-int4 image must have an even element count")
    packed = torch.bitwise_or(flat[0::2] & 0xF, torch.bitwise_left_shift(flat[1::2] & 0xF, 4))
    return packed.to(torch.uint8).numpy().tobytes(order="C")


def write_payload(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return artifact(path)


def validate_sources() -> dict[str, Any]:
    checks = {
        MISSION: EXPECTED["mission"],
        PREFIX_REVIEW: EXPECTED["prefix_review"],
        PRIOR / "SHA256SUMS": EXPECTED["prior_sums"],
        PRIOR / "ordered_prefix_report.json": EXPECTED["prior_report"],
        CAPTURE / "SHA256SUMS": EXPECTED["capture_sums"],
        CAPTURE / "witness.json": EXPECTED["capture_witness"],
        RECONSTRUCTION / "reconstruction.json": EXPECTED["reconstruction"],
        TOKEN_IDS: EXPECTED["token_ids"],
        SCALES: EXPECTED["scales"],
        PPA_SUMS: EXPECTED["ppa_sums"],
        SHELL_TB: EXPECTED["shell_tb"],
        SHELL: EXPECTED["shell"],
        CONFIG: EXPECTED["config"],
        MODEL: EXPECTED["model"],
    }
    for key, path in REFERENCE_PATHS.items():
        checks[path] = EXPECTED[key]
    for path, expected in checks.items():
        require(path.is_file(), f"bound source missing: {path}")
        require(sha256_file(path) == expected, f"bound source changed: {path.name}")

    prior = json.loads((PRIOR / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    witness = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    reconstruction = json.loads((RECONSTRUCTION / "reconstruction.json").read_text(encoding="utf-8"))
    scales = json.loads(SCALES.read_text(encoding="utf-8"))["linears"]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(prior["maximal_passing_prefix"] == ORDER[:11], "accepted o_proj prefix changed")
    require(
        prior["first_failing_or_unsupported_operator"]["operator"] == "layer_0.attention_residual_add",
        "accepted boundary changed",
    )
    require(witness["coordinate"]["dataset"] == "c4_en_512", "dataset coordinate changed")
    require(witness["coordinate"]["dataset_record_index"] == 64, "record coordinate changed")
    require(witness["coordinate"]["layer"] == 0, "layer coordinate changed")
    require(witness["coordinate"]["token"] == TOKEN_INDEX, "token coordinate changed")
    require(witness["coordinate"]["token_id"] == TOKEN_ID, "token id coordinate changed")
    require(reconstruction["calibration_input_binding"]["record_sha256"] == "2a3a0ca3a728b47ed8a825c136c34d56e04b82ecb1117465199113985ed9087f", "record hash changed")
    require(config["hidden_size"] == HIDDEN, "config hidden size changed")
    require(config["intermediate_size"] == INTERMEDIATE, "config intermediate size changed")
    require(config["rms_norm_eps"] == 1e-6, "config RMSNorm epsilon changed")
    rtl = tree_hash("rtl/**/*.sv")
    require(rtl["file_count"] == 23, "RTL file count changed")
    require(rtl["sha256"] == EXPECTED["rtl_tree"], "RTL tree changed")
    return {
        "prior": prior,
        "witness": witness,
        "reconstruction": reconstruction,
        "scales": scales,
        "config": config,
        "rtl": rtl,
        "constraints": tree_hash("constraints/**/*"),
        "source_guard": static_zero_model_guard(SELF),
        "member_counts": {
            "prior_replay": validate_manifest(PRIOR / "SHA256SUMS", PRIOR),
            "immutable_capture": validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE),
            "canonical_ppa": validate_manifest(PPA_SUMS, ROOT),
        },
    }


def derive_projection(
    weights: Any,
    tensor_name: str,
    input_values: list[int],
    expected_values: list[int],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    weight_name = tensor_name + ".weight"
    bias_name = tensor_name + ".bias"
    names = set(weights.keys())
    require(weight_name in names, f"raw weight missing: {weight_name}")
    require(bias_name not in names, f"raw bias unexpectedly exists: {bias_name}")
    raw_weight = weights.get_tensor(weight_name)
    expected_shape = (len(expected_values), len(input_values))
    require(tuple(raw_weight.shape) == expected_shape, f"raw weight geometry changed: {weight_name}")
    require(raw_weight.dtype == torch.bfloat16, f"raw weight dtype changed: {weight_name}")
    source_raw = raw_tensor_bytes(raw_weight)
    weight = raw_weight.to(torch.float64)
    weight_scale = weight.abs().amax(dim=1) / 7.0
    weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
    qweight = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
    multiplier, right_shift = derive_multiplier(
        float(metadata["hardware_input_scale"]) * weight_scale / float(metadata["output_scale"])
    )
    if metadata.get("multiplier") is not None:
        require(multiplier.tolist() == metadata["multiplier"], f"multiplier metadata differs: {tensor_name}")
    if metadata.get("right_shift") is not None:
        require(right_shift.tolist() == metadata["right_shift"], f"shift metadata differs: {tensor_name}")
    qweight_hash = sha256_bytes(raw_tensor_bytes(qweight))
    require(qweight_hash == metadata["qweight_sha256"], f"qweight hash differs: {tensor_name}")
    activations = torch.tensor(input_values, dtype=torch.int64)
    dot = (qweight.to(torch.int64) * activations[None, :]).sum(dim=1)
    require(not bool(torch.any(dot < -(1 << 31))) and not bool(torch.any(dot >= (1 << 31))), f"accumulator overflow: {tensor_name}")
    bias = torch.zeros(len(expected_values), dtype=torch.int64)
    rounded = torch.tensor(
        [
            round_shift_even(int(acc) * int(mult), int(shift))
            for acc, mult, shift in zip(dot.tolist(), multiplier.tolist(), right_shift.tolist(), strict=True)
        ],
        dtype=torch.int64,
    )
    output = rounded.clamp(-128, 127).to(torch.int8).to(torch.int64).tolist()
    require(output == expected_values, f"derived output differs: {tensor_name}")
    independent = reference_projection(
        ProjectionCase(
            tensor_name,
            1,
            len(input_values),
            [input_values],
            qweight.to(torch.int64).tolist(),
            multiplier.tolist(),
            right_shift.tolist(),
            [0] * len(expected_values),
            [0] * len(expected_values),
        )
    )
    require(independent.outputs == [expected_values], f"independent projection differs: {tensor_name}")
    return {
        "name": tensor_name,
        "input": input_values,
        "output": output,
        "qweight": qweight,
        "weight_scale": weight_scale,
        "multiplier": multiplier,
        "right_shift": right_shift,
        "bias": bias,
        "dot": dot,
        "accumulator": dot + bias,
        "rounded": rounded,
        "saturation_count": int(((rounded < -128) | (rounded > 127)).sum()),
        "source_weight_sha256": sha256_bytes(source_raw),
        "source_weight_bytes": len(source_raw),
        "qweight_sha256": qweight_hash,
        "input_scale": float(metadata["hardware_input_scale"]),
        "output_scale": float(metadata["output_scale"]),
    }


def derive_all(packages: dict[str, Any]) -> dict[str, Any]:
    witness = packages["witness"]
    tensor_manifest = witness["tensor_manifest"]
    operator_metadata = witness["operator_metadata"]
    token_raw = TOKEN_IDS.read_bytes()
    require(len(token_raw) % 4 == 0, "token ID image is malformed")
    token_ids = list(struct.unpack(f"<{len(token_raw) // 4}I", token_raw))
    require(token_ids[TOKEN_INDEX] == TOKEN_ID, "raw token ID differs")
    children_before = direct_children()
    require(not children_before, f"unexpected child processes before derivation: {children_before}")

    def captured_i8(name: str) -> list[int]:
        return i8_values(tensor_record_bytes(tensor_manifest[name]))

    with ModuleExecutionGuard() as guard:
        with safe_open(MODEL, framework="pt", device="cpu") as weights:
            embedding_table = weights.get_tensor("model.embed_tokens.weight")
            require(tuple(embedding_table.shape) == (151936, HIDDEN), "embedding geometry changed")
            embedding = embedding_table[TOKEN_ID].contiguous()
            norm_weight = weights.get_tensor("model.layers.0.post_attention_layernorm.weight").contiguous()
            require(tuple(norm_weight.shape) == (HIDDEN,), "post-attention RMSNorm gain geometry changed")
            require(embedding.dtype == torch.bfloat16 and norm_weight.dtype == torch.bfloat16, "raw BF16 source dtype changed")

            attention_scale = float(operator_metadata["attention_residual_scale"])
            oproj_scale = float(packages["scales"]["model.layers.0.self_attn.o_proj"]["output_scale"])
            oproj_output = captured_i8("o_proj_output_token1_s8")
            residual_source = quantize_int8(embedding, attention_scale)
            oproj_at_destination = quantize_int8(torch.tensor(oproj_output, dtype=torch.float64) * oproj_scale, attention_scale)
            attention_output, attention_saturation = reference_residual_add(residual_source, oproj_at_destination)
            require(attention_output == captured_i8("attention_residual_token1_s8"), "attention residual derivation differs")

            norm_output_scale = float(operator_metadata["post_attention_rmsnorm_output_scale"])
            scaled_gains = derive_scaled_gains_q8(norm_weight.to(torch.float64).tolist(), norm_output_scale)
            norm_result = reference_rmsnorm(attention_output, scaled_gains)
            require(norm_result.outputs == captured_i8("post_attention_rmsnorm_token1_s8"), "post-attention RMSNorm derivation differs")

            gate = derive_projection(
                weights,
                "model.layers.0.mlp.gate_proj",
                norm_result.outputs,
                captured_i8("mlp_gate_proj_token1_s8"),
                packages["scales"]["model.layers.0.mlp.gate_proj"],
            )
            up = derive_projection(
                weights,
                "model.layers.0.mlp.up_proj",
                norm_result.outputs,
                captured_i8("mlp_up_proj_token1_s8"),
                packages["scales"]["model.layers.0.mlp.up_proj"],
            )

            gate_scale = gate["output_scale"]
            up_scale = up["output_scale"]
            gate_q6_9 = torch.round(torch.tensor(gate["output"], dtype=torch.float64) * gate_scale * (1 << 9)).clamp(-32768, 32767).to(torch.int64)
            up_q6_9 = torch.round(torch.tensor(up["output"], dtype=torch.float64) * up_scale * (1 << 9)).clamp(-32768, 32767).to(torch.int64)
            silu_multiplier = int(operator_metadata["silu_multiplier_s32"])
            silu_right_shift = int(operator_metadata["silu_right_shift_u6"])
            down_input_absmax, down_input_scale = infer_bf16_input_scale_from_silu_metadata(
                silu_multiplier,
                silu_right_shift,
            )
            silu_result = reference_silu_gate(
                SiluGateCase(
                    "layer_0.silu_gate",
                    gate_q6_9.tolist(),
                    up_q6_9.tolist(),
                    silu_multiplier,
                    silu_right_shift,
                    0,
                )
            )
            require(silu_result.outputs == captured_i8("silu_gate_token1_s8"), "SiLU derivation differs")
            table_index = torch.bitwise_right_shift(gate_q6_9, 6).clamp(-64, 64)
            lut_output = torch.tensor([SILU_LUT[int(index)] for index in table_index.tolist()], dtype=torch.int64)
            product_q9_21 = lut_output * up_q6_9
            requant_product = product_q9_21 * silu_multiplier
            silu_rounded = torch.tensor(
                [round_shift_even(int(value), silu_right_shift) for value in requant_product.tolist()],
                dtype=torch.int64,
            )

            down_metadata = dict(packages["scales"]["model.layers.0.mlp.down_proj"])
            down_metadata["input_scale"] = down_input_scale
            down_metadata["hardware_input_scale"] = down_input_scale
            down_metadata["multiplier"] = None
            down_metadata["right_shift"] = None
            down = derive_projection(
                weights,
                "model.layers.0.mlp.down_proj",
                silu_result.outputs,
                captured_i8("mlp_down_proj_output_token1_s8"),
                down_metadata,
            )
            captured_down_acc = list(
                struct.unpack(
                    f"<{HIDDEN}i",
                    tensor_record_bytes(tensor_manifest["mlp_down_proj_accumulator_token1_s32"]),
                )
            )
            require(down["accumulator"].tolist() == captured_down_acc, "down-projection accumulator differs")

            mlp_scale = float(operator_metadata["mlp_residual_scale"])
            attention_at_mlp_scale = quantize_int8(
                torch.tensor(attention_output, dtype=torch.float64) * attention_scale,
                mlp_scale,
            )
            mlp_result = reference_mlp_residual_add(
                MlpResidualCase(
                    "layer_0.mlp_residual_add",
                    down["output"],
                    attention_at_mlp_scale,
                )
            )
            require(mlp_result.outputs == captured_i8("mlp_residual_token1_s8"), "MLP residual derivation differs")

    require(guard.module_call_count == 0, "derivation attempted a torch module call")
    require(guard.direct_forward_count == 0, "derivation attempted a direct module forward")
    children_after = direct_children()
    require(not children_after, f"unexpected child processes after derivation: {children_after}")
    return {
        "token_ids": token_ids,
        "embedding": embedding,
        "norm_weight": norm_weight,
        "attention": {
            "residual_source": residual_source,
            "oproj_at_destination": oproj_at_destination,
            "output": attention_output,
            "scale": attention_scale,
            "saturation_seen": attention_saturation,
        },
        "rmsnorm": {
            "input": attention_output,
            "scaled_gains": scaled_gains,
            "output": norm_result.outputs,
            "sumsq": norm_result.sumsq,
            "inv_rms_q30": norm_result.inv_rms_q30,
            "saturation_seen": norm_result.saturation_seen,
            "output_scale": norm_output_scale,
        },
        "projections": {"gate": gate, "up": up, "down": down},
        "silu": {
            "gate_q6_9": gate_q6_9.tolist(),
            "up_q6_9": up_q6_9.tolist(),
            "table_index": table_index.tolist(),
            "lut_output_q3_12": lut_output.tolist(),
            "product_q9_21": product_q9_21.tolist(),
            "requant_product": requant_product.tolist(),
            "rounded": silu_rounded.tolist(),
            "output": silu_result.outputs,
            "multiplier": silu_multiplier,
            "right_shift": silu_right_shift,
            "derived_down_input_absmax_bf16": down_input_absmax,
            "derived_down_input_scale": down_input_scale,
            "saturation_seen": silu_result.saturation_seen,
            "positive_saturation_count": silu_result.positive_saturation_count,
            "negative_saturation_count": silu_result.negative_saturation_count,
            "lut_clip_low_count": silu_result.lut_clip_low_count,
            "lut_clip_high_count": silu_result.lut_clip_high_count,
        },
        "mlp_residual": {
            "down": down["output"],
            "residual_at_destination": attention_at_mlp_scale,
            "output": mlp_result.outputs,
            "scale": mlp_scale,
            "saturation_seen": mlp_result.saturation_seen,
            "positive_saturation_count": mlp_result.positive_saturation_count,
            "negative_saturation_count": mlp_result.negative_saturation_count,
        },
        "guard": {
            "module_call_count": guard.module_call_count,
            "direct_forward_count": guard.direct_forward_count,
        },
        "process_children": {"before": children_before, "after": children_after},
    }


def persist_derived(bundle: Path, derived: dict[str, Any]) -> dict[str, Any]:
    directory = bundle / "derived"
    payloads: dict[str, bytes] = {
        "token_ids_u32le": raw_u32(derived["token_ids"]),
        "token1_embedding_bf16": raw_tensor_bytes(derived["embedding"]),
        "post_attention_rmsnorm_weight_bf16": raw_tensor_bytes(derived["norm_weight"]),
        "attention_residual_source_s8": bytes(value & 0xFF for value in derived["attention"]["residual_source"]),
        "attention_oproj_at_destination_s8": bytes(value & 0xFF for value in derived["attention"]["oproj_at_destination"]),
        "attention_residual_output_s8": bytes(value & 0xFF for value in derived["attention"]["output"]),
        "post_attention_rmsnorm_input_s8": bytes(value & 0xFF for value in derived["rmsnorm"]["input"]),
        "post_attention_rmsnorm_scaled_gains_q8_s16": raw_i16(derived["rmsnorm"]["scaled_gains"]),
        "post_attention_rmsnorm_output_s8": bytes(value & 0xFF for value in derived["rmsnorm"]["output"]),
        "silu_gate_q6_9_s16": raw_i16(derived["silu"]["gate_q6_9"]),
        "silu_up_q6_9_s16": raw_i16(derived["silu"]["up_q6_9"]),
        "silu_table_index_s16": raw_i16(derived["silu"]["table_index"]),
        "silu_lut_output_q3_12_s16": raw_i16(derived["silu"]["lut_output_q3_12"]),
        "silu_product_q9_21_s32": raw_i32(derived["silu"]["product_q9_21"]),
        "silu_requant_product_s64": raw_i64(derived["silu"]["requant_product"]),
        "silu_rounded_s64": raw_i64(derived["silu"]["rounded"]),
        "silu_output_s8": bytes(value & 0xFF for value in derived["silu"]["output"]),
        "mlp_residual_down_s8": bytes(value & 0xFF for value in derived["mlp_residual"]["down"]),
        "mlp_residual_stream_at_destination_s8": bytes(value & 0xFF for value in derived["mlp_residual"]["residual_at_destination"]),
        "mlp_residual_output_s8": bytes(value & 0xFF for value in derived["mlp_residual"]["output"]),
    }
    for short_name, projection in derived["projections"].items():
        prefix = f"mlp_{short_name}_proj"
        payloads.update(
            {
                f"{prefix}_weights_s4_packed": packed_s4(projection["qweight"]),
                f"{prefix}_weights_s4_in_s8": raw_tensor_bytes(projection["qweight"]),
                f"{prefix}_weight_scale_f64": raw_f64(projection["weight_scale"].tolist()),
                f"{prefix}_bias_accumulator_s32": raw_i32(projection["bias"].tolist()),
                f"{prefix}_multiplier_s32": raw_i32(projection["multiplier"].tolist()),
                f"{prefix}_right_shift_u6": bytes(projection["right_shift"].tolist()),
                f"{prefix}_dot_accumulator_s32": raw_i32(projection["dot"].tolist()),
                f"{prefix}_biased_accumulator_s32": raw_i32(projection["accumulator"].tolist()),
                f"{prefix}_rounded_s64": raw_i64(projection["rounded"].tolist()),
                f"{prefix}_output_s8": bytes(value & 0xFF for value in projection["output"]),
            }
        )
    return {name: write_payload(directory / f"{name}.bin", raw) for name, raw in payloads.items()}


def chunks(values: list[int], count: int) -> Iterator[list[int]]:
    for index in range(0, len(values), count):
        yield values[index : index + count]


def hex_word(value: int, width: int) -> str:
    return f"{value & ((1 << width) - 1):0{width // 4}x}"


def write_hex_words(path: Path, values: Iterable[int], width: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(hex_word(int(value), width) + "\n")
    return artifact(path)


def vector_rel(path: Path) -> str:
    return relative(path)


def render_projection_vectors(vector_dir: Path, derived: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    cases = [derived["projections"]["gate"], derived["projections"]["up"], derived["projections"]["down"]]
    max_input_beats = INTERMEDIATE // 16
    max_output_beats = INTERMEDIATE // 16
    input_words: list[int] = []
    expected_words: list[int] = []
    for case in cases:
        packed_input = [pack_int8(values) for values in chunks(case["input"], 16)]
        input_words.extend(packed_input + [0] * (max_input_beats - len(packed_input)))
        packed_output = [pack_int8(values) for values in chunks(case["output"], 16)]
        expected_words.extend(packed_output + [0] * (max_output_beats - len(packed_output)))

    def weight_words() -> Iterator[int]:
        for case in cases:
            for row in case["qweight"]:
                values = row.to(torch.int64).tolist()
                for group in chunks(values, 16):
                    yield pack_w4(group)

    def meta_words() -> Iterator[int]:
        for case in cases:
            for multiplier, shift, bias in zip(
                case["multiplier"].tolist(),
                case["right_shift"].tolist(),
                case["bias"].tolist(),
                strict=True,
            ):
                yield pack_meta(multiplier, shift, 0, bias)

    files = {
        "input_hex": write_hex_words(vector_dir / "projection_input.hex", input_words, 128),
        "weight_hex": write_hex_words(vector_dir / "projection_weight.hex", weight_words(), 128),
        "meta_hex": write_hex_words(vector_dir / "projection_meta.hex", meta_words(), 128),
        "expected_hex": write_hex_words(vector_dir / "projection_expected.hex", expected_words, 128),
    }
    weight_beats = [len(case["output"]) * (len(case["input"]) // 16) for case in cases]
    meta_beats = [len(case["output"]) for case in cases]
    weight_offsets = [0, weight_beats[0], weight_beats[0] + weight_beats[1]]
    meta_offsets = [0, meta_beats[0], meta_beats[0] + meta_beats[1]]
    lines = [
        "// Forward-free layer-0 remaining projection bundle.",
        "localparam integer PROJ_CASE_COUNT = 3;",
        "localparam integer PROJ_CASE_BALANCED = 0;",
        "localparam integer PROJ_CASE_SATURATION = 0;",
        "localparam integer PROJ_CASE_MLP_GATE = 0;",
        "localparam integer PROJ_CASE_MLP_DOWN = 2;",
        "localparam integer PROJ_CASE_RMSNORM_CONSUMER = 0;",
        "localparam integer PROJ_CASE_C4_V_BIAS = 0;",
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
        "localparam integer PROJ_MAX_INPUT_BEATS = 304;",
        "localparam integer PROJ_MAX_OUTPUTS = 4864;",
        "localparam integer PROJ_MAX_OUTPUT_BEATS = 304;",
        "localparam integer PROJ_MAC_LANES = 4;",
        "localparam integer PROJ_GROUPS_PER_WEIGHT_BEAT = 4;",
        "localparam integer PROJ_WEIGHT_BEATS_PER_OUTPUT = 56;",
        "localparam integer PROJ_MAX_WEIGHT_BEATS_PER_OUTPUT = 304;",
        f"localparam integer PROJ_TOTAL_WEIGHT_BEATS = {sum(weight_beats)};",
        f"localparam integer PROJ_TOTAL_META_BEATS = {sum(meta_beats)};",
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
        "reg [127:0] proj_input_beats [0:PROJ_CASE_COUNT*PROJ_MAX_ROWS*PROJ_MAX_INPUT_BEATS-1];",
        "reg [127:0] proj_weight_beats [0:PROJ_TOTAL_WEIGHT_BEATS-1];",
        "reg [127:0] proj_meta_beats [0:PROJ_TOTAL_META_BEATS-1];",
        "reg [127:0] proj_expected_beats [0:PROJ_CASE_COUNT*PROJ_MAX_ROWS*PROJ_MAX_OUTPUT_BEATS-1];",
        "initial begin",
    ]
    for index, case in enumerate(cases):
        lines.extend(
            [
                f"  proj_case_rows[{index}] = 16'd1;",
                f"  proj_case_k[{index}] = 16'd{len(case['input'])};",
                f"  proj_case_input_beats[{index}] = 16'd{len(case['input']) // 16};",
                f"  proj_case_groups[{index}] = 16'd{len(case['input']) // 4};",
                f"  proj_case_outputs[{index}] = 16'd{len(case['output'])};",
                f"  proj_case_output_beats[{index}] = 16'd{len(case['output']) // 16};",
                f"  proj_case_weight_beats_per_output[{index}] = 16'd{len(case['input']) // 16};",
                f"  proj_case_weight_offset[{index}] = 32'd{weight_offsets[index]};",
                f"  proj_case_meta_offset[{index}] = 32'd{meta_offsets[index]};",
                f"  proj_expected_saturation[{index}] = 1'b{int(case['saturation_count'] != 0)};",
            ]
        )
    lines.extend(
        [
            f'  $readmemh("{vector_rel(vector_dir / "projection_input.hex")}", proj_input_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "projection_weight.hex")}", proj_weight_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "projection_meta.hex")}", proj_meta_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "projection_expected.hex")}", proj_expected_beats);',
            "end",
        ]
    )
    return "\n".join(lines) + "\n", files


def render_simple_vectors(vector_dir: Path, derived: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    files: dict[str, Any] = {}
    includes: dict[str, str] = {}

    def int8_words(values: list[int]) -> list[int]:
        return [pack_int8(group) for group in chunks(values, 16)]

    attention = derived["attention"]
    files["residual_lhs_hex"] = write_hex_words(vector_dir / "residual_lhs.hex", int8_words(attention["residual_source"]), 128)
    files["residual_rhs_hex"] = write_hex_words(vector_dir / "residual_rhs.hex", int8_words(attention["oproj_at_destination"]), 128)
    files["residual_expected_hex"] = write_hex_words(vector_dir / "residual_expected.hex", int8_words(attention["output"]), 128)
    includes["residual"] = "\n".join(
        [
            "// Forward-free layer-0 attention residual bundle.",
            "localparam integer RESIDUAL_CASE_COUNT = 1;",
            "localparam integer RESIDUAL_BEATS = 56;",
            "reg [127:0] residual_lhs_beats [0:RESIDUAL_CASE_COUNT*RESIDUAL_BEATS-1];",
            "reg [127:0] residual_rhs_beats [0:RESIDUAL_CASE_COUNT*RESIDUAL_BEATS-1];",
            "reg [127:0] residual_expected_beats [0:RESIDUAL_CASE_COUNT*RESIDUAL_BEATS-1];",
            "reg residual_expected_saturation [0:RESIDUAL_CASE_COUNT-1];",
            "initial begin",
            f"  residual_expected_saturation[0] = 1'b{int(attention['saturation_seen'])};",
            f'  $readmemh("{vector_rel(vector_dir / "residual_lhs.hex")}", residual_lhs_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "residual_rhs.hex")}", residual_rhs_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "residual_expected.hex")}", residual_expected_beats);',
            "end",
            "",
        ]
    )

    rms = derived["rmsnorm"]
    files["rms_input_hex"] = write_hex_words(vector_dir / "rms_input.hex", int8_words(rms["input"]), 128)
    files["rms_gain_hex"] = write_hex_words(
        vector_dir / "rms_gain.hex",
        [sum((value & 0xFFFF) << (lane * 16) for lane, value in enumerate(group)) for group in chunks(rms["scaled_gains"], 16)],
        256,
    )
    files["rms_expected_hex"] = write_hex_words(vector_dir / "rms_expected.hex", int8_words(rms["output"]), 128)
    includes["rmsnorm"] = "\n".join(
        [
            "// Forward-free layer-0 post-attention RMSNorm bundle.",
            "localparam integer POST_RMS_CASE_COUNT = 1;",
            "localparam integer POST_RMS_BEATS = 56;",
            "reg [127:0] post_rms_input_beats [0:POST_RMS_CASE_COUNT*POST_RMS_BEATS-1];",
            "reg [255:0] post_rms_gain_beats [0:POST_RMS_CASE_COUNT*POST_RMS_BEATS-1];",
            "reg [127:0] post_rms_expected_beats [0:POST_RMS_CASE_COUNT*POST_RMS_BEATS-1];",
            "reg [47:0] post_rms_expected_sumsq [0:POST_RMS_CASE_COUNT-1];",
            "reg [31:0] post_rms_expected_inv [0:POST_RMS_CASE_COUNT-1];",
            "reg post_rms_expected_saturation [0:POST_RMS_CASE_COUNT-1];",
            "initial begin",
            f"  post_rms_expected_sumsq[0] = 48'h{rms['sumsq']:012x};",
            f"  post_rms_expected_inv[0] = 32'h{rms['inv_rms_q30']:08x};",
            f"  post_rms_expected_saturation[0] = 1'b{int(rms['saturation_seen'])};",
            f'  $readmemh("{vector_rel(vector_dir / "rms_input.hex")}", post_rms_input_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "rms_gain.hex")}", post_rms_gain_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "rms_expected.hex")}", post_rms_expected_beats);',
            "end",
            "",
        ]
    )

    silu = derived["silu"]
    silu_gate_words = [sum((value & 0xFFFF) << (lane * 16) for lane, value in enumerate(group)) for group in chunks(silu["gate_q6_9"], 8)]
    silu_up_words = [sum((value & 0xFFFF) << (lane * 16) for lane, value in enumerate(group)) for group in chunks(silu["up_q6_9"], 8)]
    files["silu_gate_hex"] = write_hex_words(vector_dir / "silu_gate.hex", silu_gate_words, 128)
    files["silu_up_hex"] = write_hex_words(vector_dir / "silu_up.hex", silu_up_words, 128)
    files["silu_expected_hex"] = write_hex_words(vector_dir / "silu_expected.hex", int8_words(silu["output"]), 128)
    includes["silu"] = "\n".join(
        [
            "// Forward-free layer-0 SiLU-gate bundle.",
            "localparam integer SILU_CASE_COUNT = 1;",
            "localparam integer SILU_MAX_LENGTH = 4864;",
            "localparam integer SILU_MAX_INPUT_BEATS = 608;",
            "localparam integer SILU_MAX_OUTPUT_BEATS = 304;",
            "localparam [5:0] SILU_REQUIRED_BOUNDARY_COVERAGE = 6'h00;",
            "reg [15:0] silu_case_length [0:SILU_CASE_COUNT-1];",
            "reg signed [31:0] silu_case_multiplier [0:SILU_CASE_COUNT-1];",
            "reg [5:0] silu_case_right_shift [0:SILU_CASE_COUNT-1];",
            "reg signed [7:0] silu_case_zero_point [0:SILU_CASE_COUNT-1];",
            "reg silu_expected_saturation [0:SILU_CASE_COUNT-1];",
            "reg [5:0] silu_case_boundary_coverage [0:SILU_CASE_COUNT-1];",
            "reg [127:0] silu_gate_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
            "reg [127:0] silu_up_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
            "reg [127:0] silu_expected_beats [0:SILU_CASE_COUNT*SILU_MAX_OUTPUT_BEATS-1];",
            "initial begin",
            "  silu_case_length[0] = 16'd4864;",
            f"  silu_case_multiplier[0] = 32'sh{silu['multiplier'] & 0xFFFFFFFF:08x};",
            f"  silu_case_right_shift[0] = 6'd{silu['right_shift']};",
            "  silu_case_zero_point[0] = 8'sh00;",
            f"  silu_expected_saturation[0] = 1'b{int(silu['saturation_seen'])};",
            "  silu_case_boundary_coverage[0] = 6'h00;",
            f'  $readmemh("{vector_rel(vector_dir / "silu_gate.hex")}", silu_gate_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "silu_up.hex")}", silu_up_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "silu_expected.hex")}", silu_expected_beats);',
            "end",
            "",
        ]
    )

    mlp = derived["mlp_residual"]
    files["mlp_residual_down_hex"] = write_hex_words(vector_dir / "mlp_residual_down.hex", int8_words(mlp["down"]), 128)
    files["mlp_residual_stream_hex"] = write_hex_words(vector_dir / "mlp_residual_stream.hex", int8_words(mlp["residual_at_destination"]), 128)
    files["mlp_residual_expected_hex"] = write_hex_words(vector_dir / "mlp_residual_expected.hex", int8_words(mlp["output"]), 128)
    includes["mlp_residual"] = "\n".join(
        [
            "// Forward-free layer-0 MLP residual bundle.",
            "localparam integer MLP_RESIDUAL_CASE_COUNT = 1;",
            "localparam integer MLP_RESIDUAL_BEATS = 56;",
            "reg [127:0] mlp_residual_down_beats [0:MLP_RESIDUAL_CASE_COUNT*MLP_RESIDUAL_BEATS-1];",
            "reg [127:0] mlp_residual_stream_beats [0:MLP_RESIDUAL_CASE_COUNT*MLP_RESIDUAL_BEATS-1];",
            "reg [127:0] mlp_residual_expected_beats [0:MLP_RESIDUAL_CASE_COUNT*MLP_RESIDUAL_BEATS-1];",
            "reg mlp_residual_expected_saturation [0:MLP_RESIDUAL_CASE_COUNT-1];",
            "initial begin",
            f"  mlp_residual_expected_saturation[0] = 1'b{int(mlp['saturation_seen'])};",
            f'  $readmemh("{vector_rel(vector_dir / "mlp_residual_down.hex")}", mlp_residual_down_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "mlp_residual_stream.hex")}", mlp_residual_stream_beats);',
            f'  $readmemh("{vector_rel(vector_dir / "mlp_residual_expected.hex")}", mlp_residual_expected_beats);',
            "end",
            "",
        ]
    )
    return includes, files


def generate_vectors(bundle: Path, derived: dict[str, Any]) -> dict[str, Any]:
    vector_dir = bundle / "vectors"
    vector_dir.mkdir(parents=True)
    projection_text, files = render_projection_vectors(vector_dir, derived)
    projection_include = vector_dir / "layer0_remaining_projection_vectors.svh"
    projection_include.write_text(projection_text, encoding="utf-8")
    files["projection_include"] = artifact(projection_include)
    simple, simple_files = render_simple_vectors(vector_dir, derived)
    files.update(simple_files)
    names = {
        "residual": "layer0_remaining_residual_vectors.svh",
        "rmsnorm": "layer0_remaining_rmsnorm_vectors.svh",
        "silu": "layer0_remaining_silu_vectors.svh",
        "mlp_residual": "layer0_remaining_mlp_residual_vectors.svh",
    }
    for key, filename in names.items():
        path = vector_dir / filename
        path.write_text(simple[key], encoding="utf-8")
        files[f"{key}_include"] = artifact(path)
    return files


def execution_testbench() -> str:
    text = SHELL_TB.read_text(encoding="utf-8")
    replacements = {
        '    `include "../generated/projection_vectors.svh"': '    `include "layer0_remaining_projection_vectors.svh"',
        '    `include "../generated/residual_vectors.svh"': '    `include "layer0_remaining_residual_vectors.svh"',
        '    `include "../generated/mlp_residual_vectors.svh"': '    `include "layer0_remaining_mlp_residual_vectors.svh"',
        '    `include "../generated/post_attention_rmsnorm_vectors.svh"': '    `include "layer0_remaining_rmsnorm_vectors.svh"',
        '    `include "../generated/silu_gate_vectors.svh"': '    `include "layer0_remaining_silu_vectors.svh"',
    }
    for old, new in replacements.items():
        require(old in text, f"maintained include changed: {old}")
        text = text.replace(old, new, 1)
    text = text.replace(
        "    integer qproj_stride_only_mode;",
        "    integer qproj_stride_only_mode;\n    integer layer0_remaining_replay_mode;\n    integer layer0_failure_beat;",
        1,
    )
    text = text.replace(
        '        qproj_stride_only_mode = $test$plusargs("QPROJ_STRIDE_ONLY");',
        '        qproj_stride_only_mode = $test$plusargs("QPROJ_STRIDE_ONLY");\n'
        '        layer0_remaining_replay_mode = $test$plusargs("LAYER0_REMAINING_REPLAY");',
        1,
    )
    branch = r'''        if (layer0_remaining_replay_mode) begin
            // Keep inactive projection lookup windows below the residual address map.
            current_proj_case = 2;
            send_residual_cmd_full(0, 16'd896, RESIDUAL_LHS_BASE, RESIDUAL_RHS_BASE,
                                   RESIDUAL_OUT_BASE, 16'h7000);
            wait_residual_done_and_compare(0, residual_expected_saturation[0], 16'h7000);
            last_command_cycles = cycle_count - command_start_cycle;
            if (failures != 0) begin
                for (layer0_failure_beat = 0; layer0_failure_beat < observed_count; layer0_failure_beat = layer0_failure_beat + 1)
                    $display("ACE2_LAYER0_FAILURE_VECTOR operator=attention_residual_add beat=%0d observed=%032x expected=%032x", layer0_failure_beat, observed_output[layer0_failure_beat], residual_expected_beats[layer0_failure_beat]);
                $display("ACE2_LAYER0_REMAINING_STOP operator=attention_residual_add failures=%0d", failures);
                $fatal(1, "ACE2_LAYER0_REMAINING_RTL_FAILURE");
            end
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("ACE2_LAYER0_REMAINING_BEAT operator=attention_residual_add beat=%0d data=%032x", case_index, observed_output[case_index]);
            $display("ACE2_LAYER0_REMAINING_PASS operator=attention_residual_add elements=896 writes=%0d cycles=%0d saturation=%0d", observed_count, last_command_cycles, cmd_done_saturation);

            send_post_rms_cmd_full(0, 16'd896, POST_RMS_GAIN_BASE, 16'h7001);
            wait_post_rms_done_and_compare(0, post_rms_expected_saturation[0], 16'h7001);
            last_command_cycles = cycle_count - command_start_cycle;
            if (failures != 0) begin
                for (layer0_failure_beat = 0; layer0_failure_beat < observed_count; layer0_failure_beat = layer0_failure_beat + 1)
                    $display("ACE2_LAYER0_FAILURE_VECTOR operator=post_attention_rmsnorm beat=%0d observed=%032x expected=%032x", layer0_failure_beat, observed_output[layer0_failure_beat], post_rms_expected_beats[layer0_failure_beat]);
                $display("ACE2_LAYER0_REMAINING_STOP operator=post_attention_rmsnorm failures=%0d", failures);
                $fatal(1, "ACE2_LAYER0_REMAINING_RTL_FAILURE");
            end
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("ACE2_LAYER0_REMAINING_BEAT operator=post_attention_rmsnorm beat=%0d data=%032x", case_index, observed_output[case_index]);
            $display("ACE2_LAYER0_REMAINING_PASS operator=post_attention_rmsnorm elements=896 writes=%0d cycles=%0d saturation=%0d sumsq=%0d inv=%0d", observed_count, last_command_cycles, cmd_done_saturation, cmd_done_sumsq, cmd_done_inv);

            send_mlp_proj_cmd(0, 16'h7002);
            wait_qproj_done_and_compare(0, proj_expected_saturation[0], 16'h7002);
            if (failures != 0) begin
                for (layer0_failure_beat = 0; layer0_failure_beat < observed_count; layer0_failure_beat = layer0_failure_beat + 1)
                    $display("ACE2_LAYER0_FAILURE_VECTOR operator=mlp_gate_proj beat=%0d observed=%032x expected=%032x", layer0_failure_beat, observed_output[layer0_failure_beat], proj_expected_beats[layer0_failure_beat]);
                $display("ACE2_LAYER0_REMAINING_STOP operator=mlp_gate_proj failures=%0d", failures);
                $fatal(1, "ACE2_LAYER0_REMAINING_RTL_FAILURE");
            end
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("ACE2_LAYER0_REMAINING_BEAT operator=mlp_gate_proj beat=%0d data=%032x", case_index, observed_output[case_index]);
            $display("ACE2_LAYER0_REMAINING_PASS operator=mlp_gate_proj elements=4864 writes=%0d cycles=%0d saturation=%0d", observed_count, last_command_cycles, cmd_done_saturation);

            send_mlp_up_proj_cmd(1, 16'h7003);
            wait_qproj_done_and_compare(1, proj_expected_saturation[1], 16'h7003);
            if (failures != 0) begin
                for (layer0_failure_beat = 0; layer0_failure_beat < observed_count; layer0_failure_beat = layer0_failure_beat + 1)
                    $display("ACE2_LAYER0_FAILURE_VECTOR operator=mlp_up_proj beat=%0d observed=%032x expected=%032x", layer0_failure_beat, observed_output[layer0_failure_beat], proj_expected_beats[304 + layer0_failure_beat]);
                $display("ACE2_LAYER0_REMAINING_STOP operator=mlp_up_proj failures=%0d", failures);
                $fatal(1, "ACE2_LAYER0_REMAINING_RTL_FAILURE");
            end
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("ACE2_LAYER0_REMAINING_BEAT operator=mlp_up_proj beat=%0d data=%032x", case_index, observed_output[case_index]);
            $display("ACE2_LAYER0_REMAINING_PASS operator=mlp_up_proj elements=4864 writes=%0d cycles=%0d saturation=%0d", observed_count, last_command_cycles, cmd_done_saturation);

            send_silu_gate_cmd(0, 16'd4864, 16'h7004);
            wait_silu_gate_done_and_compare(0, 0, 16'h7004);
            if (failures != 0) begin
                for (layer0_failure_beat = 0; layer0_failure_beat < observed_count; layer0_failure_beat = layer0_failure_beat + 1)
                    $display("ACE2_LAYER0_FAILURE_VECTOR operator=silu_gate beat=%0d observed=%032x expected=%032x", layer0_failure_beat, observed_output[layer0_failure_beat], silu_expected_beats[layer0_failure_beat]);
                $display("ACE2_LAYER0_REMAINING_STOP operator=silu_gate failures=%0d", failures);
                $fatal(1, "ACE2_LAYER0_REMAINING_RTL_FAILURE");
            end
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("ACE2_LAYER0_REMAINING_BEAT operator=silu_gate beat=%0d data=%032x", case_index, observed_output[case_index]);
            $display("ACE2_LAYER0_REMAINING_PASS operator=silu_gate elements=4864 writes=%0d cycles=%0d saturation=%0d", observed_count, last_command_cycles, cmd_done_saturation);

            send_mlp_down_proj_cmd(2, 16'h7005);
            wait_qproj_done_and_compare(2, proj_expected_saturation[2], 16'h7005);
            if (failures != 0) begin
                for (layer0_failure_beat = 0; layer0_failure_beat < observed_count; layer0_failure_beat = layer0_failure_beat + 1)
                    $display("ACE2_LAYER0_FAILURE_VECTOR operator=mlp_down_proj beat=%0d observed=%032x expected=%032x", layer0_failure_beat, observed_output[layer0_failure_beat], proj_expected_beats[608 + layer0_failure_beat]);
                $display("ACE2_LAYER0_REMAINING_STOP operator=mlp_down_proj failures=%0d", failures);
                $fatal(1, "ACE2_LAYER0_REMAINING_RTL_FAILURE");
            end
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("ACE2_LAYER0_REMAINING_BEAT operator=mlp_down_proj beat=%0d data=%032x", case_index, observed_output[case_index]);
            $display("ACE2_LAYER0_REMAINING_PASS operator=mlp_down_proj elements=896 writes=%0d cycles=%0d saturation=%0d", observed_count, last_command_cycles, cmd_done_saturation);

            send_mlp_residual_cmd_full(0, 16'd896, MLP_RESIDUAL_DOWN_BASE,
                                       MLP_RESIDUAL_STREAM_BASE, MLP_RESIDUAL_OUT_BASE, 16'h7006);
            wait_mlp_residual_done_and_compare(0, mlp_residual_expected_saturation[0], 16'h7006);
            if (failures != 0) begin
                for (layer0_failure_beat = 0; layer0_failure_beat < observed_count; layer0_failure_beat = layer0_failure_beat + 1)
                    $display("ACE2_LAYER0_FAILURE_VECTOR operator=mlp_residual_add beat=%0d observed=%032x expected=%032x", layer0_failure_beat, observed_output[layer0_failure_beat], mlp_residual_expected_beats[layer0_failure_beat]);
                $display("ACE2_LAYER0_REMAINING_STOP operator=mlp_residual_add failures=%0d", failures);
                $fatal(1, "ACE2_LAYER0_REMAINING_RTL_FAILURE");
            end
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("ACE2_LAYER0_REMAINING_BEAT operator=mlp_residual_add beat=%0d data=%032x", case_index, observed_output[case_index]);
            $display("ACE2_LAYER0_REMAINING_PASS operator=mlp_residual_add elements=896 writes=%0d cycles=%0d saturation=%0d", observed_count, last_command_cycles, cmd_done_saturation);
            $display("ACE2_LAYER0_REMAINING_ORDERED_REPLAY_PASS operators=7");
            $finish;
        end

'''
    needle = "        if (qproj_stride_only_mode) begin"
    require(needle in text, "focused replay insertion point changed")
    return text.replace(needle, branch + needle, 1)


def run_command(command: list[str], classification: str) -> subprocess.CompletedProcess[str]:
    allowed = {"iverilog": "iverilog_compile", "vvp": "vvp_simulation"}
    require(command[0] in allowed, f"forbidden subprocess executable: {command[0]}")
    require(allowed[command[0]] == classification, "subprocess classification differs")
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def expected_output_bytes(derived: dict[str, Any], operator: str) -> bytes:
    mapping = {
        "attention_residual_add": derived["attention"]["output"],
        "post_attention_rmsnorm": derived["rmsnorm"]["output"],
        "mlp_gate_proj": derived["projections"]["gate"]["output"],
        "mlp_up_proj": derived["projections"]["up"]["output"],
        "silu_gate": derived["silu"]["output"],
        "mlp_down_proj": derived["projections"]["down"]["output"],
        "mlp_residual_add": derived["mlp_residual"]["output"],
    }
    return bytes(value & 0xFF for value in mapping[operator])


def parse_replay(stdout: str, derived: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    summary_pattern = re.compile(
        r"ACE2_LAYER0_REMAINING_PASS operator=(\w+) elements=(\d+) writes=(\d+) cycles=(\d+) saturation=(\d+)"
    )
    beat_pattern = re.compile(r"ACE2_LAYER0_REMAINING_BEAT operator=(\w+) beat=(\d+) data=([0-9a-fA-F]+)")
    summaries = {match.group(1): match.groups()[1:] for match in summary_pattern.finditer(stdout)}
    beats: dict[str, list[tuple[int, str]]] = {}
    for match in beat_pattern.finditer(stdout):
        beats.setdefault(match.group(1), []).append((int(match.group(2)), match.group(3)))
    records: list[dict[str, Any]] = []
    for full_operator in TAIL:
        operator = full_operator.removeprefix("layer_0.")
        if operator not in summaries:
            break
        elements, writes, cycles, saturation = map(int, summaries[operator])
        rows = beats.get(operator, [])
        require([index for index, _data in rows] == list(range(len(rows))), f"RTL beat order changed: {operator}")
        rtl_raw = b"".join(int(data, 16).to_bytes(16, "little") for _index, data in rows)[:elements]
        expected_raw = expected_output_bytes(derived, operator)
        require(rtl_raw == expected_raw, f"RTL byte stream differs despite pass marker: {operator}")
        records.append(
            {
                "operator": full_operator,
                "status": "PASS_EXACT",
                "counts": {
                    "elements": elements,
                    "rtl_output_writes": writes,
                    "compared_outputs": elements,
                    "mismatches": 0,
                },
                "cycles": {"total_command_cycles": cycles},
                "saturation": {"rtl_observed": bool(saturation)},
                "output_hashes": {"rtl_output_s8": sha256_bytes(rtl_raw)},
                "handshake_latency": {
                    "simulation_executed": True,
                    "handshake_result": "PASS_NO_TIMEOUT_OR_DROPPED_WRITE",
                    "latency_result": "PASS_BOUNDED_BY_MAINTAINED_SHELL_WATCHDOG",
                },
            }
        )
    stop = re.search(r"ACE2_LAYER0_REMAINING_STOP operator=(\w+) failures=(\d+)", stdout)
    first_failure = None
    if stop is not None:
        operator = stop.group(1)
        vector = re.search(
            rf"ACE2_LAYER0_FAILURE_VECTOR operator={re.escape(operator)} beat=(\d+) observed=([0-9a-fA-F]+) expected=([0-9a-fA-F]+)",
            stdout,
        )
        first_failure = {
            "operator": f"layer_0.{operator}",
            "classification": "GENUINE_RTL_REPLAY_FAILURE",
            "failure_count": int(stop.group(2)),
            "first_failure_vector": None
            if vector is None
            else {
                "beat": int(vector.group(1)),
                "observed": vector.group(2).lower(),
                "expected": vector.group(3).lower(),
            },
        }
    return records, first_failure


def reference_metadata(derived: dict[str, Any], operator: str) -> dict[str, Any]:
    if operator == "attention_residual_add":
        return {
            "input_hashes": {
                "residual_source_s8": sha256_bytes(bytes(value & 0xFF for value in derived["attention"]["residual_source"])),
                "o_proj_at_destination_s8": sha256_bytes(bytes(value & 0xFF for value in derived["attention"]["oproj_at_destination"])),
            },
            "reference_hashes": {"output_s8": sha256_bytes(expected_output_bytes(derived, operator))},
            "reference_saturation_count": int(derived["attention"]["saturation_seen"]),
        }
    if operator == "post_attention_rmsnorm":
        return {
            "input_hashes": {
                "input_s8": sha256_bytes(bytes(value & 0xFF for value in derived["rmsnorm"]["input"])),
                "scaled_gains_q8_s16": sha256_bytes(raw_i16(derived["rmsnorm"]["scaled_gains"])),
            },
            "reference_hashes": {"output_s8": sha256_bytes(expected_output_bytes(derived, operator))},
            "reference_saturation_count": int(derived["rmsnorm"]["saturation_seen"]),
            "sumsq": derived["rmsnorm"]["sumsq"],
            "inv_rms_q30": derived["rmsnorm"]["inv_rms_q30"],
        }
    if operator in {"mlp_gate_proj", "mlp_up_proj", "mlp_down_proj"}:
        key = operator.removeprefix("mlp_").removesuffix("_proj")
        projection = derived["projections"][key]
        return {
            "input_hashes": {
                "input_s8": sha256_bytes(bytes(value & 0xFF for value in projection["input"])),
                "raw_source_weight_bf16": projection["source_weight_sha256"],
                "qweight_s4_in_s8": projection["qweight_sha256"],
                "multiplier_s32": sha256_bytes(raw_i32(projection["multiplier"].tolist())),
                "right_shift_u6": sha256_bytes(bytes(projection["right_shift"].tolist())),
                "bias_accumulator_s32": sha256_bytes(raw_i32(projection["bias"].tolist())),
            },
            "reference_hashes": {
                "dot_accumulator_s32": sha256_bytes(raw_i32(projection["dot"].tolist())),
                "biased_accumulator_s32": sha256_bytes(raw_i32(projection["accumulator"].tolist())),
                "output_s8": sha256_bytes(expected_output_bytes(derived, operator)),
            },
            "weight_elements": int(projection["qweight"].numel()),
            "metadata_channels": len(projection["output"]),
            "reference_saturation_count": projection["saturation_count"],
        }
    if operator == "silu_gate":
        return {
            "input_hashes": {
                "gate_q6_9_s16": sha256_bytes(raw_i16(derived["silu"]["gate_q6_9"])),
                "up_q6_9_s16": sha256_bytes(raw_i16(derived["silu"]["up_q6_9"])),
                "metadata": sha256_bytes(raw_i32([derived["silu"]["multiplier"]]) + bytes([derived["silu"]["right_shift"]])),
            },
            "reference_hashes": {
                "lut_output_q3_12_s16": sha256_bytes(raw_i16(derived["silu"]["lut_output_q3_12"])),
                "product_q9_21_s32": sha256_bytes(raw_i32(derived["silu"]["product_q9_21"])),
                "output_s8": sha256_bytes(expected_output_bytes(derived, operator)),
            },
            "reference_saturation_count": derived["silu"]["positive_saturation_count"] + derived["silu"]["negative_saturation_count"],
        }
    return {
        "input_hashes": {
            "down_s8": sha256_bytes(bytes(value & 0xFF for value in derived["mlp_residual"]["down"])),
            "residual_at_destination_s8": sha256_bytes(bytes(value & 0xFF for value in derived["mlp_residual"]["residual_at_destination"])),
        },
        "reference_hashes": {"output_s8": sha256_bytes(expected_output_bytes(derived, operator))},
        "reference_saturation_count": derived["mlp_residual"]["positive_saturation_count"] + derived["mlp_residual"]["negative_saturation_count"],
    }


def authority_manifest(packages: dict[str, Any], derived: dict[str, Any], artifacts: dict[str, Any], vectors: dict[str, Any]) -> dict[str, Any]:
    projection_sources = {}
    for key, projection in derived["projections"].items():
        projection_sources[key] = {
            "tensor_name": projection["name"] + ".weight",
            "tensor_shape": list(projection["qweight"].shape),
            "tensor_dtype": "bfloat16",
            "tensor_raw_sha256": projection["source_weight_sha256"],
            "tensor_raw_bytes": projection["source_weight_bytes"],
            "bias_tensor_present": False,
        }
    return {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "created_at_utc": utc_now(),
        "coordinate": {
            "dataset": "c4_en_512",
            "dataset_record_index": 64,
            "layer": 0,
            "token": TOKEN_INDEX,
            "token_id": TOKEN_ID,
        },
        "model": {
            "repository": "Qwen/Qwen2.5-0.5B",
            "revision": MODEL_REVISION,
            "config_sha256": EXPECTED["config"],
            "safetensors_sha256": EXPECTED["model"],
            "raw_projection_sources": projection_sources,
            "raw_embedding_row_sha256": sha256_bytes(raw_tensor_bytes(derived["embedding"])),
            "raw_post_attention_rmsnorm_weight_sha256": sha256_bytes(raw_tensor_bytes(derived["norm_weight"])),
        },
        "frozen_order": TAIL,
        "fixed_point_contract": {
            "weight_quantization": "symmetric_per_output_channel_signed_int4_absmax_div_7",
            "activation": "signed_int8",
            "accumulator": "signed_int32_complete_reduction",
            "rounding": "round_to_nearest_ties_to_even",
            "output": "signed_int8_saturating",
            "rmsnorm_gain": "signed_int16_Q7.8_weight_divided_by_output_scale",
            "rmsnorm_reciprocal": "ceil_integer_root_then_floor_2^30_div_root",
            "silu_metadata_source": "hash-bound immutable zero-model capture metadata",
            "silu_multiplier_s32": derived["silu"]["multiplier"],
            "silu_right_shift_u6": derived["silu"]["right_shift"],
            "down_input_absmax_bf16_unique_from_silu_metadata": derived["silu"]["derived_down_input_absmax_bf16"],
            "down_input_scale_unique_from_silu_metadata": derived["silu"]["derived_down_input_scale"],
            "attention_residual_scale": derived["attention"]["scale"],
            "post_attention_rmsnorm_output_scale": derived["rmsnorm"]["output_scale"],
            "mlp_residual_scale": derived["mlp_residual"]["scale"],
        },
        "derived_artifacts": artifacts,
        "rtl_vector_artifacts": vectors,
        "zero_model_execution_guard": derived["guard"],
        "process_children": derived["process_children"],
        "source_bindings": {
            "mission": artifact(MISSION, f"handoff:{MISSION_ID}/mission.json"),
            "accepted_o_proj_review": artifact(PREFIX_REVIEW, "handoff:manager-record-o-proj-certified-frontier-v1/round-0002.json"),
            "accepted_o_proj_replay": artifact(PRIOR / "SHA256SUMS"),
            "immutable_capture": artifact(CAPTURE / "SHA256SUMS"),
            "raw_token_ids": artifact(TOKEN_IDS),
            "raw_token_reconstruction": artifact(RECONSTRUCTION / "reconstruction.json"),
            "immutable_scales": artifact(SCALES),
            "fixed_point_rules": artifact(REFERENCE_PATHS["fixed_rules"]),
            "independent_references": {key: artifact(path) for key, path in REFERENCE_PATHS.items() if key != "fixed_rules"},
            "rtl_tree_sha256": packages["rtl"]["sha256"],
            "constraint_tree_sha256": packages["constraints"]["sha256"],
        },
        "immutable_capture_cross_checks": {
            name: packages["witness"]["tensor_manifest"][name]["sha256"]
            for name in (
                "attention_residual_token1_s8",
                "post_attention_rmsnorm_token1_s8",
                "mlp_gate_proj_token1_s8",
                "mlp_up_proj_token1_s8",
                "silu_gate_token1_s8",
                "mlp_down_proj_accumulator_token1_s32",
                "mlp_down_proj_output_token1_s8",
                "mlp_residual_token1_s8",
            )
        },
    }


def write_zero_model(packages: dict[str, Any], derived: dict[str, Any]) -> None:
    write_json(
        OUT / "zero_model_execution.json",
        {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "completed_at_utc": utc_now(),
            "status": "PASS_ZERO_MODEL_EXECUTION_RTL_ONLY_SUBPROCESSES",
            "model_construction_count": 0,
            "model_execution_count": 0,
            "model_forward_count": 0,
            "model_call_count": 0,
            "calibration_forward_count": 0,
            "capture_count": 0,
            "hook_count": 0,
            "extraction_worker_count": 0,
            "model_subprocess_count": 0,
            "subprocess_count": 2,
            "subprocess_classes": ["iverilog_compile", "vvp_simulation"],
            "torch_module_guard": derived["guard"],
            "process_children_before_after_derivation": derived["process_children"],
            "static_source_guard": packages["source_guard"],
            "source": artifact(OUT / "replay_source_at_execution.py"),
        },
    )


def finish_report(
    packages: dict[str, Any],
    derived: dict[str, Any],
    protected_before: dict[str, str],
    rtl_before: dict[str, Any],
    constraints_before: dict[str, Any],
    ppa_before: str,
    compile_ok: bool,
    replayed: subprocess.CompletedProcess[str] | None,
) -> tuple[dict[str, Any], bool]:
    protected_after = protected_state()
    rtl_after = tree_hash("rtl/**/*.sv")
    constraints_after = tree_hash("constraints/**/*")
    ppa_after = sha256_file(PPA_SUMS)
    require(protected_after == protected_before, "protected state changed")
    require(rtl_after == rtl_before, "RTL tree changed during replay")
    require(constraints_after == constraints_before, "constraint tree changed during replay")
    require(ppa_after == ppa_before == EXPECTED["ppa_sums"], "canonical PPA aggregate changed")
    validate_manifest(OUT / "bundle/SHA256SUMS", OUT / "bundle")

    records: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    if compile_ok and replayed is not None:
        records, first_failure = parse_replay(replayed.stdout, derived)
    else:
        first_failure = {
            "operator": TAIL[0],
            "classification": "NON_DERIVABLE_MAINTAINED_RTL_INTERFACE_FAILURE",
            "first_failure_vector": None,
        }
    for record in records:
        operator = record["operator"].removeprefix("layer_0.")
        metadata = reference_metadata(derived, operator)
        record.update(metadata)
        record["saturation"]["reference_count"] = metadata["reference_saturation_count"]
        record["saturation"]["result"] = "PASS_EXACT"
        record["rtl_source_hashes"] = {
            "rtl_tree": rtl_after["sha256"],
            "constraint_tree": constraints_after["sha256"],
            "shell": artifact(SHELL),
            "maintained_harness": artifact(SHELL_TB),
            "execution_harness": artifact(OUT / "replay_tb_source_at_execution.sv"),
        }

    all_pass = len(records) == len(TAIL) and first_failure is None and replayed is not None and replayed.returncode == 0
    if replayed is not None and "ACE2_LAYER0_REMAINING_ORDERED_REPLAY_PASS operators=7" in replayed.stdout:
        require(all_pass, "ordered pass marker conflicts with parsed replay")
    passing = [record["operator"] for record in records]
    maximal_prefix = ORDER[:11] + passing
    report = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "producer_role": "engineer",
        "review_status": "PENDING_FRESH_REVIEWER",
        "completed_at_utc": utc_now(),
        "frozen_order": ORDER,
        "prior_maximal_prefix": ORDER[:11],
        "attempted_operators": records,
        "newly_passing_operators": passing,
        "maximal_passing_prefix": maximal_prefix,
        "first_failing_or_unsupported_operator": first_failure,
        "continuous_replay": {
            "start_operator": TAIL[0],
            "last_passing_operator": passing[-1] if passing else ORDER[10],
            "later_operators_attempted_only_after_exact_pass": True,
            "ordered_single_vvp_execution": replayed is not None,
        },
        "batch_result": "LAYER0_COMPLETE_EXACT" if all_pass else "STOPPED_AT_FIRST_GENUINE_RTL_OR_INTERFACE_FAILURE",
        "zero_model_execution": artifact(OUT / "zero_model_execution.json"),
        "authority_bundle": {
            "hash_bound_before_replay": True,
            "pre_post_exact_match": True,
            "binding": artifact(OUT / "bundle_binding.json"),
            "sha256_manifest": artifact(OUT / "bundle/SHA256SUMS"),
            "authority_manifest": artifact(OUT / "bundle/authority_manifest.json"),
        },
        "rtl_tree": rtl_after,
        "constraint_tree": constraints_after,
        "canonical_ppa": {
            "reused_not_rerun": True,
            "pre_post_exact_match": True,
            "sha256_manifest": artifact(PPA_SUMS),
        },
        "protected_state": {
            "pre_post_exact_match": True,
            "hashes": protected_after,
        },
        "forbidden_flows_run": [],
        "state_mutations": {
            "rtl": False,
            "frontier_authority_seal": False,
            "synthesis_opensta_ppa": False,
            "baseline_candidate_benchmark": False,
            "sealed_dynamic_scaling_path": False,
        },
        "member_counts": packages["member_counts"],
        "review_requirement": "Fresh Reviewer must independently accept or reject the complete authority bundle and ordered replay.",
    }
    return report, all_pass


def run() -> None:
    require(not OUT.exists(), "remaining layer-0 evidence output already exists")
    require(shutil.which("iverilog") is not None, "iverilog is unavailable")
    require(shutil.which("vvp") is not None, "vvp is unavailable")
    packages = validate_sources()
    protected_before = protected_state()
    ppa_before = sha256_file(PPA_SUMS)
    rtl_before = packages["rtl"]
    constraints_before = packages["constraints"]
    derived = derive_all(packages)

    OUT.mkdir(parents=True)
    bundle = OUT / "bundle"
    bundle.mkdir()
    shutil.copy2(SELF, OUT / "replay_source_at_execution.py")
    artifacts = persist_derived(bundle, derived)
    vectors = generate_vectors(bundle, derived)
    write_json(bundle / "authority_manifest.json", authority_manifest(packages, derived, artifacts, vectors))
    write_sha256s(bundle)
    bundle_sum_hash = sha256_file(bundle / "SHA256SUMS")
    write_json(
        OUT / "bundle_binding.json",
        {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "bound_at_utc": utc_now(),
            "status": "HASH_BOUND_BEFORE_RTL_REPLAY",
            "sha256_manifest": artifact(bundle / "SHA256SUMS"),
            "member_count": validate_manifest(bundle / "SHA256SUMS", bundle),
        },
    )

    testbench = OUT / "replay_tb_source_at_execution.sv"
    testbench.write_text(execution_testbench(), encoding="utf-8")
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
        f"-I{relative(bundle / 'vectors')}",
        "-s",
        "ace2_shell_tb",
        "-o",
        relative(image),
        *[relative(path) for path in rtl_sources],
        relative(testbench),
    ]
    compiled = run_command(compile_command, "iverilog_compile")
    (OUT / "iverilog.log").write_text(
        "command=" + " ".join(compile_command) + "\n" + compiled.stdout + compiled.stderr,
        encoding="utf-8",
    )
    replayed: subprocess.CompletedProcess[str] | None = None
    if compiled.returncode == 0:
        replay_command = ["vvp", relative(image), "+LAYER0_REMAINING_REPLAY"]
        replayed = run_command(replay_command, "vvp_simulation")
        (OUT / "replay.stdout").write_text(replayed.stdout, encoding="utf-8")
        (OUT / "replay.stderr").write_text(replayed.stderr, encoding="utf-8")
        (OUT / "replay.log").write_text(
            "command=" + " ".join(replay_command) + "\n" + replayed.stdout + replayed.stderr,
            encoding="utf-8",
        )
    else:
        (OUT / "replay.stdout").write_text("", encoding="utf-8")
        (OUT / "replay.stderr").write_text("", encoding="utf-8")
        (OUT / "replay.log").write_text("simulation_not_started_compile_failure\n", encoding="utf-8")

    require(sha256_file(bundle / "SHA256SUMS") == bundle_sum_hash, "authority bundle changed during replay")
    validate_manifest(bundle / "SHA256SUMS", bundle)
    write_zero_model(packages, derived)
    report, all_pass = finish_report(
        packages,
        derived,
        protected_before,
        rtl_before,
        constraints_before,
        ppa_before,
        compiled.returncode == 0,
        replayed,
    )
    write_json(OUT / "ordered_prefix_report.json", report)
    lines = [
        "ACE2_LAYER0_REMAINING_OFFLINE_AUTHORITY_BATCH_REPLAY",
        f"rtl_tree_sha256={rtl_before['sha256']}",
        f"authority_bundle_sha256={bundle_sum_hash}",
        "model_execution_count=0",
        "model_subprocess_count=0",
    ]
    for record in report["attempted_operators"]:
        lines.append(
            f"{record['operator']}={record['status']} outputs={record['counts']['compared_outputs']} "
            f"mismatches={record['counts']['mismatches']} cycles={record['cycles']['total_command_cycles']}"
        )
    lines.extend(
        [
            f"batch_result={report['batch_result']}",
            "synthesis_opensta_ppa_executed=false",
            "",
        ]
    )
    (OUT / "batch_replay.log").write_text("\n".join(lines), encoding="utf-8")
    write_sha256s(OUT)
    if all_pass:
        print(
            "ACE2_LAYER0_REMAINING_OFFLINE_AUTHORITY_REPLAY_PASS "
            f"operators=7 maximal_prefix={ORDER[-1]} authority_sha256={bundle_sum_hash} model_execution_count=0"
        )
    else:
        failure = report["first_failing_or_unsupported_operator"]
        raise RuntimeError(
            f"ordered replay stopped at {failure['operator']}: {failure['classification']}"
        )


def check() -> None:
    require(OUT.is_dir(), "remaining layer-0 evidence output is missing")
    packages = validate_sources()
    members = validate_manifest(OUT / "SHA256SUMS", OUT)
    bundle_members = validate_manifest(OUT / "bundle/SHA256SUMS", OUT / "bundle")
    require(sha256_file(OUT / "replay_source_at_execution.py") == sha256_file(SELF), "archived replay source changed")
    derived = derive_all(packages)
    report = json.loads((OUT / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    zero = json.loads((OUT / "zero_model_execution.json").read_text(encoding="utf-8"))
    authority = json.loads((OUT / "bundle/authority_manifest.json").read_text(encoding="utf-8"))
    binding = json.loads((OUT / "bundle_binding.json").read_text(encoding="utf-8"))
    require(binding["status"] == "HASH_BOUND_BEFORE_RTL_REPLAY", "bundle was not pre-bound")
    require(binding["sha256_manifest"]["sha256"] == sha256_file(OUT / "bundle/SHA256SUMS"), "bundle binding changed")
    require(authority["frozen_order"] == TAIL, "authority operator order changed")
    require(report["newly_passing_operators"] == TAIL, "remaining passing set changed")
    require(report["maximal_passing_prefix"] == ORDER, "layer-0 maximal prefix changed")
    require(report["first_failing_or_unsupported_operator"] is None, "ordered replay now reports a failure")
    require(report["batch_result"] == "LAYER0_COMPLETE_EXACT", "batch result changed")
    require(len(report["attempted_operators"]) == 7, "attempted operator count changed")
    for record in report["attempted_operators"]:
        operator = record["operator"].removeprefix("layer_0.")
        require(record["status"] == "PASS_EXACT", f"operator is not exact: {operator}")
        require(record["counts"]["mismatches"] == 0, f"operator mismatch count changed: {operator}")
        require(record["output_hashes"]["rtl_output_s8"] == sha256_bytes(expected_output_bytes(derived, operator)), f"operator output hash changed: {operator}")
    for name in (
        "model_construction_count",
        "model_execution_count",
        "model_forward_count",
        "model_call_count",
        "calibration_forward_count",
        "capture_count",
        "hook_count",
        "extraction_worker_count",
        "model_subprocess_count",
    ):
        require(zero[name] == 0, f"zero-execution counter changed: {name}")
    require(zero["subprocess_classes"] == ["iverilog_compile", "vvp_simulation"], "subprocess scope changed")
    require(tree_hash("rtl/**/*.sv")["sha256"] == EXPECTED["rtl_tree"], "RTL tree changed")
    require(sha256_file(PPA_SUMS) == EXPECTED["ppa_sums"], "canonical PPA aggregate changed")
    require(protected_state() == report["protected_state"]["hashes"], "protected state differs from replay")
    print(
        "ACE2_LAYER0_REMAINING_OFFLINE_AUTHORITY_REPLAY_CHECK_PASS "
        f"members={members} bundle_members={bundle_members} operators=7 maximal_prefix={ORDER[-1]} model_execution_count=0"
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
