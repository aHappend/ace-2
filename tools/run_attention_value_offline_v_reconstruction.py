#!/usr/bin/env python3
"""Forward-free raw-weight reconstruction of C4 record64 layer-0 token-0 V."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors import safe_open
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
OUT = ROOT / "evidence/diagnostics/c4-layer0-attention-value-token0-v-offline-reconstruction-20260803-v1"
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-attention-value-offline-v-authority-replay-v1/mission.json"
)
CERTIFIED_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "manager-record-attention-score-softmax-certified-frontier-v1/round-0001.json"
)
PROMPT_MANIFEST = ROOT / "benchmark/quality/PROMPT_MANIFEST.json"
QUALITY_CONFIG = ROOT / "benchmark/quality/QUALITY_CONFIG.json"
FIXED_RULES = ROOT / "tools/ace2_full_model_fixed_point.py"
HEAD0 = ROOT / "evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2"
SOFTMAX_REPLAY = ROOT / "evidence/verification/rtl-attention-score-token0-reference-batch-replay-v1"
CANDIDATE = ROOT / "evidence/diagnostics/c4-layer0-attention-value-token0-v-minimal-reference-capture-20260803-v1"
PPA_SUMS = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/SHA256SUMS"

MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DATASET = "c4_en_512"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
RECORD_INDEX = 64
TOKEN_INDEX = 0
CHECK_TOKEN_INDEX = 1
HEAD = 0
HEAD_DIM = 64
HIDDEN = 896

EXPECTED = {
    "mission": "2c54f0f4291a894fa98a33ec0e7e82d50bf469cba160e9f80f2f9c288622ef43",
    "certified_review": "9439e134167cc643715fe63661eb99687cd75010343eeac5af5140f12d12a35d",
    "prompt_manifest": "9ee394d7d344d6f14e6829cbf8bea27b5508c7fe5344d941495f5668ca32335c",
    "quality_config": "50fe8c986c556bd1247e156c9f65de1e1137ed87542aca9112d089b15d226588",
    "fixed_rules": "b5f1c1800560894a728f199d36b31ef9d624b5afa9d56ee544b0d467270c1c74",
    "head0_sums": "7e3c768d9eff34b327859621e0e093243cf7562837611ad0bd37480a86703e97",
    "softmax_sums": "22a3addf99fd8afe757e62c303add0dbc5a97726a675550d8870aafcaf298604",
    "candidate_sums": "48c884980727aad2fc3be8d87bf3fa5c11ac2c5d5bdd0aefd74663cd13ac8232",
    "ppa_sums": "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4",
    "rtl_tree": "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be",
    "record_sha256": "a5c1e98fba3be31fc775446d53772b726abd2b6ef1f684171be1262db7df2acc",
    "token_sequence_sha256": "5614de646c94a5b4b457142971d02c74ccc741b304c95b5c7c44a5c36fa6c436",
    "calibration_record_sha256": "2a3a0ca3a728b47ed8a825c136c34d56e04b82ecb1117465199113985ed9087f",
    "calibration_token_sequence_sha256": "5acd0242a298c444b14916a523ffea96fb4072ff4bcada8a846d2b0eccfb16e7",
    "config_sha256": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
    "tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "model_safetensors_sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
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


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path, public_path: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": public_path or relative(path),
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def protected_state() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in PROTECTED_PATHS}


def rtl_tree() -> dict[str, Any]:
    rows = [(sha256_file(path), relative(path)) for path in sorted(ROOT.glob("rtl/**/*.sv"))]
    payload = "".join(f"{digest}  {name}\n" for digest, name in rows).encode()
    digest = sha256_bytes(payload)
    require(len(rows) == 23, f"RTL file count changed: {len(rows)}")
    require(digest == EXPECTED["rtl_tree"], f"RTL tree changed: {digest}")
    return {"file_count": len(rows), "sha256": digest}


def direct_children() -> list[int]:
    path = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return [] if not text else [int(value) for value in text.split()]


def static_no_execution_check(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "forward",
                "__call__",
                "register_forward_hook",
                "register_forward_pre_hook",
            }:
                forbidden_calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name) and node.func.id in {
                "calibrate",
                "capture_layer0_inputs",
            }:
                forbidden_calls.append(node.func.id)
    forbidden_imports = [
        name
        for name in imports
        if name.startswith("transformers")
        or "capture" in name
        or name.startswith("run_attention")
        or name.startswith("run_layer0")
    ]
    require(not forbidden_imports, f"forbidden model/capture imports: {forbidden_imports}")
    require(not forbidden_calls, f"forbidden execution calls: {forbidden_calls}")
    return {
        "ast_parsed": True,
        "forbidden_call_count": 0,
        "forbidden_import_count": 0,
        "imports": sorted(imports),
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

        def blocked_call(_module: torch.nn.Module, *_args: Any, **_kwargs: Any) -> Any:
            guard.module_call_count += 1
            raise RuntimeError("torch.nn.Module execution is forbidden in offline reconstruction")

        def profile(frame: Any, event: str, _arg: Any) -> Any:
            if event == "call" and frame.f_code.co_name == "forward":
                owner = frame.f_locals.get("self")
                if isinstance(owner, torch.nn.Module):
                    guard.direct_forward_count += 1
                    raise RuntimeError("direct torch.nn.Module.forward execution is forbidden")
            return profile

        torch.nn.Module.__call__ = blocked_call
        torch.nn.Module._call_impl = blocked_call
        sys.setprofile(profile)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        sys.setprofile(self._profile)
        torch.nn.Module.__call__ = self._call
        torch.nn.Module._call_impl = self._call_impl


def tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes(order="C")
    return tensor.numpy().tobytes(order="C")


def tensor_record(path: Path, value: torch.Tensor) -> dict[str, Any]:
    raw = tensor_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    numeric = value.detach().cpu().to(torch.float64)
    return {
        "bytes": len(raw),
        "dtype": str(value.dtype).removeprefix("torch."),
        "elements": value.numel(),
        "max": float(numeric.max()),
        "min": float(numeric.min()),
        "path": relative(path),
        "sha256": sha256_bytes(raw),
        "shape": list(value.shape),
    }


def raw_record(path: Path, raw: bytes, dtype: str, elements: int, shape: list[int]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "bytes": len(raw),
        "dtype": dtype,
        "elements": elements,
        "path": relative(path),
        "sha256": sha256_bytes(raw),
        "shape": shape,
    }


def token_hash(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(len(token_ids).to_bytes(8, "big"))
    for token in token_ids:
        digest.update(token.to_bytes(4, "big", signed=False))
    return digest.hexdigest()


def record_hash(text: str) -> str:
    raw = text.encode("utf-8")
    return sha256_bytes(len(raw).to_bytes(8, "big") + raw)


def cached_model_files() -> dict[str, Path]:
    snapshot = (
        Path(HF_HUB_CACHE)
        / "models--Qwen--Qwen2.5-0.5B"
        / "snapshots"
        / MODEL_REVISION
    )
    files = {
        "config.json": snapshot / "config.json",
        "tokenizer.json": snapshot / "tokenizer.json",
        "model.safetensors": snapshot / "model.safetensors",
    }
    for name, path in files.items():
        require(path.is_file(), f"pinned cached file is unavailable: {name}")
    require(sha256_file(files["config.json"]) == EXPECTED["config_sha256"], "config hash changed")
    require(sha256_file(files["tokenizer.json"]) == EXPECTED["tokenizer_sha256"], "tokenizer hash changed")
    require(
        sha256_file(files["model.safetensors"]) == EXPECTED["model_safetensors_sha256"],
        "model safetensors hash changed",
    )
    return files


def load_token_ids(
    tokenizer_path: Path,
    dataset_spec: dict[str, Any],
    expected_record_sha256: str,
    expected_token_sha256: str,
    expected_token_count: int,
) -> tuple[str, list[int]]:
    dataset = load_dataset(
        dataset_spec["repository"],
        dataset_spec["config"],
        split=dataset_spec["split"],
        revision=dataset_spec["revision"],
        streaming=True,
    )
    record: dict[str, Any] | None = None
    for index, row in enumerate(dataset):
        if index == dataset_spec["indices"]["start"]:
            record = row
            break
    require(record is not None, "frozen C4 record64 is unavailable")
    text = str(record[dataset_spec["field"]])
    require(record_hash(text) == expected_record_sha256, "dataset record hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_ids = tokenizer.encode(text, add_special_tokens=False).ids[: dataset_spec["token_limit"]]
    require(len(token_ids) == expected_token_count, f"dataset token count changed: {len(token_ids)}")
    require(token_hash(token_ids) == expected_token_sha256, "dataset token IDs changed")
    return text, token_ids


def round_shift_even(value: torch.Tensor, shift: torch.Tensor | int) -> torch.Tensor:
    shift_tensor = torch.as_tensor(shift, dtype=torch.int64, device=value.device)
    shift_tensor = torch.broadcast_to(shift_tensor, value.shape)
    require(bool(torch.all((shift_tensor >= 0) & (shift_tensor <= 63))), "shift outside 0..63")
    safe_shift = shift_tensor.clamp(min=1)
    magnitude = value.abs()
    base = torch.bitwise_right_shift(magnitude, safe_shift)
    mask = torch.bitwise_left_shift(torch.ones_like(safe_shift), safe_shift) - 1
    remainder = torch.bitwise_and(magnitude, mask)
    half = torch.bitwise_left_shift(torch.ones_like(safe_shift), safe_shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((base & 1) == 1))
    rounded = base + increment.to(base.dtype)
    signed = torch.where(value < 0, -rounded, rounded)
    return torch.where(shift_tensor == 0, value, signed)


def derive_multiplier(real_multiplier: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    real = real_multiplier.detach().to(torch.float64)
    multiplier = torch.zeros_like(real, dtype=torch.int64)
    right_shift = torch.full_like(real, -1, dtype=torch.int64)
    for shift in range(63, -1, -1):
        candidate = torch.round(real * math.ldexp(1.0, shift))
        select = (right_shift < 0) & (candidate <= (1 << 31) - 1)
        multiplier = torch.where(select, candidate.to(torch.int64), multiplier)
        right_shift = torch.where(select, torch.full_like(right_shift, shift), right_shift)
    require(bool(torch.all(right_shift >= 0)), "projection multiplier is not representable")
    return multiplier, right_shift


def quantize_int8(value: torch.Tensor, scale: float) -> torch.Tensor:
    require(math.isfinite(scale) and scale > 0, "invalid activation scale")
    return torch.round(value.to(torch.float64) / scale).clamp(-128, 127).to(torch.int8)


def rmsnorm_raw(embedding: torch.Tensor, input_scale: float, output_scale: float, gain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qinput = quantize_int8(embedding, input_scale)
    scaled_gains = torch.round(gain.to(torch.float64) / output_scale * 256.0)
    require(bool(torch.all((scaled_gains >= -32768) & (scaled_gains <= 32767))), "RMS gain overflow")
    scaled_gains_q8 = scaled_gains.to(torch.int16)
    values = qinput.to(torch.int64)
    sumsq = (values * values).sum()
    mean_square = (sumsq + HIDDEN // 2) // HIDDEN
    root = torch.floor(torch.sqrt(mean_square.to(torch.float64))).to(torch.int64)
    root = root + (root * root < mean_square).to(torch.int64)
    root = root.clamp(min=1)
    inv_rms_q30 = (1 << 30) // root
    product = values * scaled_gains_q8.to(torch.int64) * inv_rms_q30
    output = round_shift_even(product, 38).clamp(-128, 127).to(torch.int8)
    return qinput, scaled_gains_q8, output


def v_projection_head0(
    rms_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    input_scale: float,
    output_scale: float,
) -> dict[str, torch.Tensor]:
    head_weight = weight[:HEAD_DIM].to(torch.float64)
    weight_scale = head_weight.abs().amax(dim=1) / 7.0
    weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
    qweight = torch.round(head_weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
    multiplier, right_shift = derive_multiplier(input_scale * weight_scale / output_scale)
    bias_accumulator = (
        torch.zeros(HEAD_DIM, dtype=torch.int64)
        if bias is None
        else torch.round(bias[:HEAD_DIM].to(torch.float64) / (input_scale * weight_scale)).to(torch.int64)
    )
    dot = (qweight.to(torch.int64) * rms_output.to(torch.int64).unsqueeze(0)).sum(dim=1)
    accumulator = dot + bias_accumulator
    require(bool(torch.all((accumulator >= -(1 << 31)) & (accumulator < (1 << 31)))), "V accumulator overflow")
    output = round_shift_even(accumulator * multiplier, right_shift).clamp(-128, 127).to(torch.int8)
    return {
        "qweight": qweight,
        "weight_scale": weight_scale,
        "multiplier": multiplier,
        "right_shift": right_shift,
        "bias_accumulator": bias_accumulator,
        "dot": dot,
        "accumulator": accumulator,
        "output": output,
    }


def tensor_from_record(record: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    raw = (ROOT / record["path"]).read_bytes()
    require(len(raw) == record["bytes"], f"tensor byte count changed: {record['path']}")
    require(sha256_bytes(raw) == record["sha256"], f"tensor hash changed: {record['path']}")
    return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(record["shape"])


def verify_bound_inputs() -> dict[str, Any]:
    bindings = [
        (MISSION, EXPECTED["mission"]),
        (CERTIFIED_REVIEW, EXPECTED["certified_review"]),
        (PROMPT_MANIFEST, EXPECTED["prompt_manifest"]),
        (QUALITY_CONFIG, EXPECTED["quality_config"]),
        (FIXED_RULES, EXPECTED["fixed_rules"]),
        (HEAD0 / "SHA256SUMS", EXPECTED["head0_sums"]),
        (SOFTMAX_REPLAY / "SHA256SUMS", EXPECTED["softmax_sums"]),
        (CANDIDATE / "SHA256SUMS", EXPECTED["candidate_sums"]),
        (PPA_SUMS, EXPECTED["ppa_sums"]),
    ]
    for path, expected in bindings:
        require(sha256_file(path) == expected, f"bound input changed: {path.name}")
    return {
        "head0_member_count": validate_manifest(HEAD0 / "SHA256SUMS", HEAD0),
        "softmax_member_count": validate_manifest(SOFTMAX_REPLAY / "SHA256SUMS", SOFTMAX_REPLAY),
        "candidate_member_count": validate_manifest(CANDIDATE / "SHA256SUMS", CANDIDATE),
    }


def run() -> None:
    require(not OUT.exists(), f"output path already exists: {OUT}")
    static_check = static_no_execution_check(SELF)
    member_counts = verify_bound_inputs()
    initial_protected = protected_state()
    initial_ppa_hash = sha256_file(PPA_SUMS)
    initial_rtl = rtl_tree()
    children_before = direct_children()
    require(not children_before, f"unexpected child processes before reconstruction: {children_before}")

    manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY_CONFIG.read_text(encoding="utf-8"))
    head0_witness = json.loads((HEAD0 / "witness.json").read_text(encoding="utf-8"))
    files = cached_model_files()

    dataset_spec = manifest["datasets"][DATASET]
    require(dataset_spec["indices"]["start"] == RECORD_INDEX, "dataset record index changed")
    require(dataset_spec["revision"] == DATASET_REVISION, "dataset revision changed")
    require(manifest["model"] == {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION}, "model manifest changed")
    require(quality["baseline"]["model_revision"] == MODEL_REVISION, "quality model revision changed")

    rms_output_scale = float(head0_witness["k_projection"]["input_scale"])
    v_input_scale = rms_output_scale
    v_output_scale = float(head0_witness["operator_metadata"]["v_projection_output_scale"])
    require(v_input_scale == rms_output_scale, "V input scale is not bound to RMSNorm output")

    with ModuleExecutionGuard() as guard:
        _record_text, token_ids = load_token_ids(
            files["tokenizer.json"],
            dataset_spec,
            EXPECTED["record_sha256"],
            EXPECTED["token_sequence_sha256"],
            93,
        )
        _calibration_text, calibration_token_ids = load_token_ids(
            files["tokenizer.json"],
            manifest["datasets"]["c4_calibration"],
            EXPECTED["calibration_record_sha256"],
            EXPECTED["calibration_token_sequence_sha256"],
            385,
        )
        require(token_ids[CHECK_TOKEN_INDEX] == 33597, "accepted token1 ID changed")
        config = json.loads(files["config.json"].read_text(encoding="utf-8"))
        require(config["hidden_size"] == HIDDEN, "model hidden size changed")
        require(config["num_attention_heads"] == 14, "query head count changed")
        require(config["num_key_value_heads"] == 2, "KV head count changed")
        require(HIDDEN // config["num_attention_heads"] == HEAD_DIM, "head dimension changed")

        with safe_open(files["model.safetensors"], framework="pt", device="cpu") as weights:
            required_names = {
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.self_attn.v_proj.weight",
            }
            require(required_names.issubset(set(weights.keys())), "required raw tensors are missing")
            embedding_table = weights.get_tensor("model.embed_tokens.weight")
            gain = weights.get_tensor("model.layers.0.input_layernorm.weight")
            v_weight = weights.get_tensor("model.layers.0.self_attn.v_proj.weight")
            bias_name = "model.layers.0.self_attn.v_proj.bias"
            v_bias = weights.get_tensor(bias_name) if bias_name in weights.keys() else None
            require(tuple(embedding_table.shape[1:]) == (HIDDEN,), "embedding geometry changed")
            require(tuple(gain.shape) == (HIDDEN,), "RMS gain geometry changed")
            require(tuple(v_weight.shape) == (2 * HEAD_DIM, HIDDEN), "V weight geometry changed")

            token0_embedding = embedding_table[token_ids[TOKEN_INDEX]].clone()
            token1_embedding = embedding_table[token_ids[CHECK_TOKEN_INDEX]].clone()
            calibration_embeddings = embedding_table[
                torch.tensor(calibration_token_ids, dtype=torch.int64)
            ].to(torch.float64)
            rms_input_absmax = float(calibration_embeddings.abs().amax())
            rms_input_scale = rms_input_absmax / 127.0
            token0_rms_input, scaled_gains, token0_rms_output = rmsnorm_raw(
                token0_embedding, rms_input_scale, rms_output_scale, gain
            )
            token1_rms_input, token1_scaled_gains, token1_rms_output = rmsnorm_raw(
                token1_embedding, rms_input_scale, rms_output_scale, gain
            )
            require(torch.equal(scaled_gains, token1_scaled_gains), "RMS metadata is token-dependent")
            token0_v = v_projection_head0(
                token0_rms_output, v_weight, v_bias, v_input_scale, v_output_scale
            )
            token1_v = v_projection_head0(
                token1_rms_output, v_weight, v_bias, v_input_scale, v_output_scale
            )

    require(guard.module_call_count == 0, "a torch module call was attempted")
    require(guard.direct_forward_count == 0, "a direct module forward was attempted")
    children_after = direct_children()
    require(not children_after, f"unexpected child processes after reconstruction: {children_after}")

    immutable_rms = tensor_from_record(head0_witness["tensor_manifest"]["k_projection_input_s8"], torch.int8)
    immutable_token1_v = tensor_from_record(head0_witness["tensor_manifest"]["kv_write_v_head0_token1_s8"], torch.int8)
    require(torch.equal(token1_rms_output, immutable_rms), "raw token1 RMSNorm does not match immutable accepted payload")
    require(torch.equal(token1_v["output"], immutable_token1_v), "raw token1 V does not match immutable accepted payload")

    derived_v_raw = tensor_bytes(token0_v["output"])
    candidate_witness = json.loads((CANDIDATE / "witness.json").read_text(encoding="utf-8"))
    candidate_record = candidate_witness["tensor_manifest"]["v_cache_head0_token0_s8"]
    candidate_raw = (ROOT / candidate_record["path"]).read_bytes()
    require(sha256_bytes(candidate_raw) == candidate_record["sha256"], "candidate payload hash changed")
    require(derived_v_raw == candidate_raw, "offline-derived token0 V differs from non-authoritative candidate")
    capture_counts = candidate_witness["capture"]
    candidate_forward_count = int(capture_counts["calibration_dependency_forward_count"]) + int(capture_counts["evaluation_forward_count"])
    require(candidate_forward_count == 2, "candidate is not the known two-forward packet")

    require(protected_state() == initial_protected, "protected state changed during reconstruction")
    require(sha256_file(PPA_SUMS) == initial_ppa_hash, "canonical PPA aggregate changed")
    require(rtl_tree() == initial_rtl, "RTL tree changed during reconstruction")

    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SELF, OUT / "reconstruction_source_at_execution.py")
    tensors = {
        "token_ids_u32le": raw_record(
            OUT / "tensors/token_ids_u32le.bin",
            b"".join(token.to_bytes(4, "little", signed=False) for token in token_ids),
            "uint32",
            len(token_ids),
            [len(token_ids)],
        ),
        "token0_embedding_bf16": tensor_record(OUT / "tensors/token0_embedding_bf16.bin", token0_embedding),
        "token0_rmsnorm_input_s8": tensor_record(OUT / "tensors/token0_rmsnorm_input_s8.bin", token0_rms_input),
        "token0_rmsnorm_output_s8": tensor_record(OUT / "tensors/token0_rmsnorm_output_s8.bin", token0_rms_output),
        "v_cache_head0_token0_s8": tensor_record(OUT / "tensors/v_cache_head0_token0_s8.bin", token0_v["output"]),
    }
    zero_execution = {
        "schema_version": 1,
        "status": "PASS_ZERO_MODEL_EXECUTION",
        "completed_at_utc": utc_now(),
        "model_execution_count": 0,
        "model_forward_count": 0,
        "model_call_count": 0,
        "calibration_forward_count": 0,
        "hook_count": 0,
        "capture_count": 0,
        "extraction_worker_count": 0,
        "subprocess_count": 0,
        "torch_module_guard": {
            "module_call_count": guard.module_call_count,
            "direct_forward_count": guard.direct_forward_count,
            "guarded_scope": "dataset/tokenizer/raw-safetensors/reconstruction",
        },
        "process_children": {"before": children_before, "after": children_after},
        "static_source_check": static_check,
        "source": artifact(OUT / "reconstruction_source_at_execution.py"),
    }
    write_json(OUT / "zero_model_execution.json", zero_execution)

    reconstruction = {
        "schema_version": 1,
        "mission_id": "rtl-attention-value-offline-v-authority-replay-v1",
        "classification": "FORWARD_FREE_RAW_WEIGHT_TOKEN0_V_RECONSTRUCTION",
        "status": "PASS_OFFLINE_DERIVATION_PENDING_FRESH_REVIEWER",
        "completed_at_utc": utc_now(),
        "coordinate": {
            "dataset": DATASET,
            "dataset_record_index": RECORD_INDEX,
            "layer": 0,
            "token": TOKEN_INDEX,
            "token_id": token_ids[TOKEN_INDEX],
            "head": HEAD,
            "head_dim": HEAD_DIM,
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "config": {"filename": "config.json", "sha256": EXPECTED["config_sha256"]},
            "tokenizer": {"filename": "tokenizer.json", "sha256": EXPECTED["tokenizer_sha256"]},
            "weights": {"filename": "model.safetensors", "sha256": EXPECTED["model_safetensors_sha256"]},
            "raw_tensor_names": [
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.v_proj.bias",
            ],
        },
        "token_binding": {
            "record_sha256": EXPECTED["record_sha256"],
            "sequence_sha256": EXPECTED["token_sequence_sha256"],
            "token_count": len(token_ids),
            "token0_id": token_ids[0],
            "token1_id": token_ids[1],
            "tensor": tensors["token_ids_u32le"],
        },
        "fixed_point_contract": {
            "rmsnorm_input_scale": rms_input_scale,
            "rmsnorm_input_absmax": rms_input_absmax,
            "rmsnorm_input_scale_provenance": "raw_pinned_calibration_token_embedding_absmax_div_127",
            "rmsnorm_output_scale": rms_output_scale,
            "rmsnorm_gain_format": "signed_int16_Q7.8_weight_divided_by_output_scale",
            "rmsnorm_reciprocal": "ceil_integer_root_then_floor_2^30_div_root",
            "v_projection_input_scale": v_input_scale,
            "v_projection_output_scale": v_output_scale,
            "weight_quantization": "signed_int4_symmetric_per_output_channel_absmax_div_7",
            "accumulator": "signed_int32_dot_plus_quantized_bias",
            "requantization": "signed_int32_multiplier_largest_representable_shift_round_ties_even_then_s8_clamp",
            "rules_source": artifact(FIXED_RULES),
            "quality_contract": artifact(QUALITY_CONFIG),
            "immutable_quantization_metadata": artifact(HEAD0 / "witness.json"),
        },
        "calibration_input_binding": {
            "record_sha256": EXPECTED["calibration_record_sha256"],
            "sequence_sha256": EXPECTED["calibration_token_sequence_sha256"],
            "token_count": len(calibration_token_ids),
            "operation": "raw_embedding_rows_only_no_calibration_forward",
        },
        "derived_tensor_manifest": tensors,
        "intermediate_hashes": {
            "rmsnorm_scaled_gains_q8": sha256_bytes(tensor_bytes(scaled_gains)),
            "v_qweight_head0_s4_in_s8": sha256_bytes(tensor_bytes(token0_v["qweight"])),
            "v_weight_scale_head0_f64": sha256_bytes(tensor_bytes(token0_v["weight_scale"])),
            "v_multiplier_head0_s64": sha256_bytes(tensor_bytes(token0_v["multiplier"])),
            "v_right_shift_head0_s64": sha256_bytes(tensor_bytes(token0_v["right_shift"])),
            "v_bias_accumulator_head0_s64": sha256_bytes(tensor_bytes(token0_v["bias_accumulator"])),
            "v_dot_head0_s64": sha256_bytes(tensor_bytes(token0_v["dot"])),
            "v_accumulator_head0_s64": sha256_bytes(tensor_bytes(token0_v["accumulator"])),
        },
        "immutable_cross_checks": {
            "token1_rmsnorm_exact_match": True,
            "token1_rmsnorm_sha256": sha256_bytes(tensor_bytes(token1_rms_output)),
            "token1_v_exact_match": True,
            "token1_v_sha256": sha256_bytes(tensor_bytes(token1_v["output"])),
            "immutable_head0_package": artifact(HEAD0 / "SHA256SUMS"),
        },
        "prior_candidate": {
            "authority": False,
            "classification": "NON_AUTHORITATIVE_TWO_FORWARD_CANDIDATE_BYTE_EXACT_MATCH_ONLY",
            "forward_count": candidate_forward_count,
            "payload_sha256": sha256_bytes(candidate_raw),
            "byte_exact_match": True,
            "package": artifact(CANDIDATE / "SHA256SUMS"),
            "referenced_only_after_offline_derivation": True,
        },
        "certified_prefix_binding": {
            "review": artifact(CERTIFIED_REVIEW, "handoff:manager-record-attention-score-softmax-certified-frontier-v1/round-0001.json"),
            "accepted_softmax_replay": artifact(SOFTMAX_REPLAY / "SHA256SUMS"),
            "rtl_tree": initial_rtl,
        },
        "zero_model_execution": artifact(OUT / "zero_model_execution.json"),
        "protected_state_pre_post_exact_match": True,
        "canonical_ppa": {"reused_not_rerun": True, "sha256_manifest": artifact(PPA_SUMS)},
        "member_counts": member_counts,
        "state_mutations": {
            "rtl": False,
            "frontier_authority_seal": False,
            "synthesis_opensta_ppa": False,
            "baseline_candidate_benchmark_scale32": False,
        },
        "next_boundary": "hash_bound_merge_and_continuous_rtl_replay_pending",
        "review_status": "PENDING_FRESH_REVIEWER",
    }
    write_json(OUT / "reconstruction.json", reconstruction)
    (OUT / "reconstruction.log").write_text(
        "ACE2_OFFLINE_TOKEN0_V_RECONSTRUCTION_PASS\n"
        "model_execution_count=0\n"
        f"token0_id={token_ids[0]}\n"
        f"v_cache_head0_token0_s8_sha256={tensors['v_cache_head0_token0_s8']['sha256']}\n"
        "candidate_authority=false\n"
        "candidate_byte_exact_match=true\n"
        "rtl_replay_pending=true\n",
        encoding="utf-8",
    )
    print(
        "ACE2_OFFLINE_TOKEN0_V_RECONSTRUCTION_PASS "
        f"model_execution_count=0 token0_id={token_ids[0]} "
        f"v_sha256={tensors['v_cache_head0_token0_s8']['sha256']} candidate_match=1"
    )


if __name__ == "__main__":
    run()
