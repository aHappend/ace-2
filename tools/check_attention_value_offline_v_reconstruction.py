#!/usr/bin/env python3
"""Independent pure-integer checker for the offline token-0 V reconstruction."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import struct
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
RUNNER = ROOT / "tools/run_attention_value_offline_v_reconstruction.py"
OUT = ROOT / "evidence/diagnostics/c4-layer0-attention-value-token0-v-offline-reconstruction-20260803-v1"
PROMPT_MANIFEST = ROOT / "benchmark/quality/PROMPT_MANIFEST.json"
HEAD0 = ROOT / "evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2"
CANDIDATE = ROOT / "evidence/diagnostics/c4-layer0-attention-value-token0-v-minimal-reference-capture-20260803-v1"
PPA_SUMS = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/SHA256SUMS"

MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
HIDDEN = 896
HEAD_DIM = 64
EXPECTED_RECORD_SHA256 = "a5c1e98fba3be31fc775446d53772b726abd2b6ef1f684171be1262db7df2acc"
EXPECTED_TOKEN_SHA256 = "5614de646c94a5b4b457142971d02c74ccc741b304c95b5c7c44a5c36fa6c436"
EXPECTED_CALIBRATION_RECORD_SHA256 = "2a3a0ca3a728b47ed8a825c136c34d56e04b82ecb1117465199113985ed9087f"
EXPECTED_CALIBRATION_TOKEN_SHA256 = "5acd0242a298c444b14916a523ffea96fb4072ff4bcada8a846d2b0eccfb16e7"
EXPECTED_CONFIG_SHA256 = "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"
EXPECTED_TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
EXPECTED_MODEL_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECTED_PPA_SHA256 = "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4"
EXPECTED_RTL_TREE = "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be"

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


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {"bytes": path.stat().st_size, "path": relative(path), "sha256": sha256_file(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def protected_state() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in PROTECTED_PATHS}


def rtl_tree_hash() -> str:
    rows = []
    for path in sorted(ROOT.glob("rtl/**/*.sv")):
        rows.append(f"{sha256_file(path)}  {relative(path)}\n")
    require(len(rows) == 23, f"RTL file count changed: {len(rows)}")
    return sha256_bytes("".join(rows).encode())


def direct_children() -> list[int]:
    path = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return [] if not text else [int(value) for value in text.split()]


def static_no_execution_check(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    calls: list[str] = []
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
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name) and node.func.id in {"calibrate", "capture_layer0_inputs"}:
                calls.append(node.func.id)
    bad_imports = [name for name in imports if name.startswith("transformers") or "capture" in name]
    require(not bad_imports, f"forbidden checker imports: {bad_imports}")
    require(not calls, f"forbidden checker calls: {calls}")
    return {"ast_parsed": True, "forbidden_import_count": 0, "forbidden_call_count": 0}


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
            raise RuntimeError("torch.nn.Module execution is forbidden in independent checking")

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


def cached_model_files() -> dict[str, Path]:
    snapshot = Path(HF_HUB_CACHE) / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / MODEL_REVISION
    files = {
        "config.json": snapshot / "config.json",
        "tokenizer.json": snapshot / "tokenizer.json",
        "model.safetensors": snapshot / "model.safetensors",
    }
    require(sha256_file(files["config.json"]) == EXPECTED_CONFIG_SHA256, "config hash changed")
    require(sha256_file(files["tokenizer.json"]) == EXPECTED_TOKENIZER_SHA256, "tokenizer hash changed")
    require(sha256_file(files["model.safetensors"]) == EXPECTED_MODEL_SHA256, "model hash changed")
    return files


def hash_record(text: str) -> str:
    raw = text.encode("utf-8")
    return sha256_bytes(len(raw).to_bytes(8, "big") + raw)


def hash_tokens(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(len(tokens).to_bytes(8, "big"))
    for token in tokens:
        digest.update(token.to_bytes(4, "big", signed=False))
    return digest.hexdigest()


def load_token_ids(
    tokenizer_path: Path,
    dataset_name: str,
    expected_record_sha256: str,
    expected_token_sha256: str,
    expected_token_count: int,
) -> list[int]:
    manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    spec = manifest["datasets"][dataset_name]
    dataset = load_dataset(
        spec["repository"],
        spec["config"],
        split=spec["split"],
        revision=spec["revision"],
        streaming=True,
    )
    selected: dict[str, Any] | None = None
    for index, row in enumerate(dataset):
        if index == spec["indices"]["start"]:
            selected = row
            break
    require(selected is not None, "C4 record64 is unavailable")
    text = str(selected[spec["field"]])
    require(hash_record(text) == expected_record_sha256, "dataset record hash changed")
    tokens = Tokenizer.from_file(str(tokenizer_path)).encode(text, add_special_tokens=False).ids[:512]
    require(len(tokens) == expected_token_count, "dataset token count changed")
    require(hash_tokens(tokens) == expected_token_sha256, "dataset token sequence changed")
    return tokens


def round_shift_even(value: int, shift: int) -> int:
    require(0 <= shift <= 63, "right shift outside 0..63")
    if shift == 0:
        return value
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (base & 1)):
        base += 1
    return sign * base


def quantize_int8(values: list[float], scale: float) -> list[int]:
    return [max(-128, min(127, int(round(value / scale)))) for value in values]


def rmsnorm_raw(
    embedding: list[float],
    gain: list[float],
    input_scale: float,
    output_scale: float,
) -> tuple[list[int], list[int], list[int]]:
    qinput = quantize_int8(embedding, input_scale)
    gains = [int(round(value / output_scale * 256.0)) for value in gain]
    require(all(-32768 <= value <= 32767 for value in gains), "RMS gain overflow")
    sumsq = sum(value * value for value in qinput)
    mean_square = (sumsq + HIDDEN // 2) // HIDDEN
    root = math.isqrt(mean_square)
    if root * root < mean_square:
        root += 1
    root = max(root, 1)
    inv_rms_q30 = (1 << 30) // root
    outputs = [
        max(-128, min(127, round_shift_even(value * scaled_gain * inv_rms_q30, 38)))
        for value, scaled_gain in zip(qinput, gains, strict=True)
    ]
    return qinput, gains, outputs


def derive_multiplier(real: float) -> tuple[int, int]:
    require(math.isfinite(real) and real >= 0, "invalid real multiplier")
    for shift in range(63, -1, -1):
        candidate = int(round(math.ldexp(real, shift)))
        if candidate <= (1 << 31) - 1:
            return candidate, shift
    raise RuntimeError("unrepresentable multiplier")


def v_projection_head0(
    rms_output: list[int],
    weights: list[list[float]],
    bias: list[float],
    input_scale: float,
    output_scale: float,
) -> dict[str, list[int] | list[float]]:
    qweights: list[list[int]] = []
    weight_scales: list[float] = []
    multipliers: list[int] = []
    right_shifts: list[int] = []
    bias_accumulators: list[int] = []
    dots: list[int] = []
    accumulators: list[int] = []
    outputs: list[int] = []
    for channel in range(HEAD_DIM):
        row = weights[channel]
        scale = max(abs(value) for value in row) / 7.0
        if scale <= 0:
            scale = 1.0
        qrow = [max(-8, min(7, int(round(value / scale)))) for value in row]
        multiplier, shift = derive_multiplier(input_scale * scale / output_scale)
        bias_acc = int(round(bias[channel] / (input_scale * scale)))
        dot = sum(a * w for a, w in zip(rms_output, qrow, strict=True))
        accumulator = dot + bias_acc
        require(-(1 << 31) <= accumulator < (1 << 31), "V accumulator overflow")
        output = max(-128, min(127, round_shift_even(accumulator * multiplier, shift)))
        qweights.append(qrow)
        weight_scales.append(scale)
        multipliers.append(multiplier)
        right_shifts.append(shift)
        bias_accumulators.append(bias_acc)
        dots.append(dot)
        accumulators.append(accumulator)
        outputs.append(output)
    return {
        "qweight": [value for row in qweights for value in row],
        "weight_scale": weight_scales,
        "multiplier": multipliers,
        "right_shift": right_shifts,
        "bias_accumulator": bias_accumulators,
        "dot": dots,
        "accumulator": accumulators,
        "output": outputs,
    }


def pack_i8(values: list[int]) -> bytes:
    return bytes(value & 0xFF for value in values)


def pack_i16(values: list[int]) -> bytes:
    return b"".join(struct.pack("<h", value) for value in values)


def pack_i64(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", value) for value in values)


def pack_f64(values: list[float]) -> bytes:
    return b"".join(struct.pack("<d", value) for value in values)


def read_tensor(record: dict[str, Any]) -> bytes:
    raw = (ROOT / record["path"]).read_bytes()
    require(len(raw) == record["bytes"], f"tensor byte count changed: {record['path']}")
    require(sha256_bytes(raw) == record["sha256"], f"tensor hash changed: {record['path']}")
    return raw


def write_sha256s(base: Path) -> Path:
    output = base / "SHA256SUMS"
    rows = []
    for member in sorted(base.rglob("*")):
        if member.is_file() and member != output:
            rows.append(f"{sha256_file(member)}  {member.relative_to(base).as_posix()}\n")
    output.write_text("".join(rows), encoding="utf-8")
    return output


def validate_manifest(path: Path, base: Path) -> int:
    count = 0
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row:
            continue
        expected, name = row.split("  ", 1)
        member = base / name
        require(member.is_file(), f"missing manifest member: {name}")
        require(sha256_file(member) == expected, f"changed manifest member: {name}")
        count += 1
    return count


def check() -> None:
    require(OUT.is_dir(), "offline reconstruction output is missing")
    require(not (OUT / "independent_check.json").exists(), "independent check already exists")
    require(not (OUT / "SHA256SUMS").exists(), "reconstruction package is already sealed")
    source_check = static_no_execution_check(SELF)
    runner_check = static_no_execution_check(RUNNER)
    initial_protected = protected_state()
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SHA256, "canonical PPA aggregate changed")
    require(rtl_tree_hash() == EXPECTED_RTL_TREE, "RTL tree changed")
    children_before = direct_children()
    require(not children_before, f"unexpected child processes before checking: {children_before}")

    reconstruction = json.loads((OUT / "reconstruction.json").read_text(encoding="utf-8"))
    zero = json.loads((OUT / "zero_model_execution.json").read_text(encoding="utf-8"))
    require(zero["status"] == "PASS_ZERO_MODEL_EXECUTION", "runner zero-execution status changed")
    for field in (
        "model_execution_count",
        "model_forward_count",
        "model_call_count",
        "calibration_forward_count",
        "hook_count",
        "capture_count",
        "extraction_worker_count",
        "subprocess_count",
    ):
        require(zero[field] == 0, f"runner zero-execution counter changed: {field}")
    require(
        sha256_file(RUNNER) == reconstruction["zero_model_execution"]["sha256"]
        or sha256_file(RUNNER) == zero["source"]["sha256"],
        "runner source binding changed",
    )

    head0 = json.loads((HEAD0 / "witness.json").read_text(encoding="utf-8"))
    rms_output_scale = float(head0["k_projection"]["input_scale"])
    v_input_scale = rms_output_scale
    v_output_scale = float(head0["operator_metadata"]["v_projection_output_scale"])

    files = cached_model_files()
    with ModuleExecutionGuard() as guard:
        token_ids = load_token_ids(
            files["tokenizer.json"],
            "c4_en_512",
            EXPECTED_RECORD_SHA256,
            EXPECTED_TOKEN_SHA256,
            93,
        )
        calibration_token_ids = load_token_ids(
            files["tokenizer.json"],
            "c4_calibration",
            EXPECTED_CALIBRATION_RECORD_SHA256,
            EXPECTED_CALIBRATION_TOKEN_SHA256,
            385,
        )
        with safe_open(files["model.safetensors"], framework="pt", device="cpu") as weights:
            embedding_table = weights.get_tensor("model.embed_tokens.weight")
            gain_tensor = weights.get_tensor("model.layers.0.input_layernorm.weight")
            v_weight_tensor = weights.get_tensor("model.layers.0.self_attn.v_proj.weight")
            v_bias_tensor = weights.get_tensor("model.layers.0.self_attn.v_proj.bias")
            token0_embedding = embedding_table[token_ids[0]].to(torch.float64).tolist()
            token1_embedding = embedding_table[token_ids[1]].to(torch.float64).tolist()
            gain = gain_tensor.to(torch.float64).tolist()
            v_weight = v_weight_tensor[:HEAD_DIM].to(torch.float64).tolist()
            v_bias = v_bias_tensor[:HEAD_DIM].to(torch.float64).tolist()
            rms_input_absmax = float(
                embedding_table[torch.tensor(calibration_token_ids, dtype=torch.int64)]
                .to(torch.float64)
                .abs()
                .amax()
            )
            rms_input_scale = rms_input_absmax / 127.0
            token0_embedding_raw = (
                embedding_table[token_ids[0]].contiguous().view(torch.uint16).numpy().tobytes(order="C")
            )

        token0_rms_input, scaled_gains, token0_rms_output = rmsnorm_raw(
            token0_embedding, gain, rms_input_scale, rms_output_scale
        )
        _token1_rms_input, token1_gains, token1_rms_output = rmsnorm_raw(
            token1_embedding, gain, rms_input_scale, rms_output_scale
        )
        require(scaled_gains == token1_gains, "independent RMS gains became token-dependent")
        token0_v = v_projection_head0(
            token0_rms_output, v_weight, v_bias, v_input_scale, v_output_scale
        )
        token1_v = v_projection_head0(
            token1_rms_output, v_weight, v_bias, v_input_scale, v_output_scale
        )

    require(guard.module_call_count == 0, "checker attempted a torch module call")
    require(guard.direct_forward_count == 0, "checker attempted a direct module forward")
    children_after = direct_children()
    require(not children_after, f"unexpected child processes after checking: {children_after}")

    tensors = reconstruction["derived_tensor_manifest"]
    expected_token_bytes = b"".join(token.to_bytes(4, "little", signed=False) for token in token_ids)
    require(read_tensor(tensors["token_ids_u32le"]) == expected_token_bytes, "persisted token IDs differ")
    require(read_tensor(tensors["token0_embedding_bf16"]) == token0_embedding_raw, "persisted embedding row differs")
    require(read_tensor(tensors["token0_rmsnorm_input_s8"]) == pack_i8(token0_rms_input), "persisted RMS input differs")
    require(read_tensor(tensors["token0_rmsnorm_output_s8"]) == pack_i8(token0_rms_output), "persisted RMS output differs")
    derived_v_raw = pack_i8(token0_v["output"])
    require(read_tensor(tensors["v_cache_head0_token0_s8"]) == derived_v_raw, "persisted token0 V differs")

    intermediates = reconstruction["intermediate_hashes"]
    independent_hashes = {
        "rmsnorm_scaled_gains_q8": sha256_bytes(pack_i16(scaled_gains)),
        "v_qweight_head0_s4_in_s8": sha256_bytes(pack_i8(token0_v["qweight"])),
        "v_weight_scale_head0_f64": sha256_bytes(pack_f64(token0_v["weight_scale"])),
        "v_multiplier_head0_s64": sha256_bytes(pack_i64(token0_v["multiplier"])),
        "v_right_shift_head0_s64": sha256_bytes(pack_i64(token0_v["right_shift"])),
        "v_bias_accumulator_head0_s64": sha256_bytes(pack_i64(token0_v["bias_accumulator"])),
        "v_dot_head0_s64": sha256_bytes(pack_i64(token0_v["dot"])),
        "v_accumulator_head0_s64": sha256_bytes(pack_i64(token0_v["accumulator"])),
    }
    require(independent_hashes == intermediates, "independent intermediate hashes differ")

    immutable_rms = read_tensor(head0["tensor_manifest"]["k_projection_input_s8"])
    immutable_token1_v = read_tensor(head0["tensor_manifest"]["kv_write_v_head0_token1_s8"])
    require(pack_i8(token1_rms_output) == immutable_rms, "independent token1 RMS check differs")
    require(pack_i8(token1_v["output"]) == immutable_token1_v, "independent token1 V check differs")

    candidate_witness = json.loads((CANDIDATE / "witness.json").read_text(encoding="utf-8"))
    candidate_raw = read_tensor(candidate_witness["tensor_manifest"]["v_cache_head0_token0_s8"])
    require(derived_v_raw == candidate_raw, "independent token0 V differs from candidate")
    require(reconstruction["prior_candidate"]["authority"] is False, "candidate was promoted to authority")
    require(reconstruction["prior_candidate"]["byte_exact_match"] is True, "candidate match record changed")
    require(reconstruction["prior_candidate"]["referenced_only_after_offline_derivation"] is True, "candidate ordering proof changed")
    require(
        reconstruction["fixed_point_contract"]["rmsnorm_input_absmax"] == rms_input_absmax,
        "runner RMSNorm input absmax differs from independent raw calibration lookup",
    )
    require(
        reconstruction["fixed_point_contract"]["rmsnorm_input_scale"] == rms_input_scale,
        "runner RMSNorm input scale differs from independent raw calibration lookup",
    )

    require(protected_state() == initial_protected, "protected state changed during independent check")
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SHA256, "canonical PPA aggregate changed during check")
    require(rtl_tree_hash() == EXPECTED_RTL_TREE, "RTL tree changed during check")

    shutil.copy2(SELF, OUT / "checker_source_at_execution.py")
    check_report = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_FORWARD_FREE_RECONSTRUCTION",
        "completed_at_utc": utc_now(),
        "model_execution_count": 0,
        "model_forward_count": 0,
        "model_call_count": 0,
        "calibration_forward_count": 0,
        "hook_count": 0,
        "capture_count": 0,
        "extraction_worker_count": 0,
        "subprocess_count": 0,
        "coordinate": reconstruction["coordinate"],
        "derived_v_sha256": sha256_bytes(derived_v_raw),
        "derived_v_bytes": len(derived_v_raw),
        "runner_payload_exact_match": True,
        "immutable_token1_rmsnorm_exact_match": True,
        "immutable_token1_v_exact_match": True,
        "non_authoritative_candidate_exact_match": True,
        "independent_intermediate_hashes": independent_hashes,
        "torch_module_guard": {
            "module_call_count": guard.module_call_count,
            "direct_forward_count": guard.direct_forward_count,
        },
        "process_children": {"before": children_before, "after": children_after},
        "static_source_checks": {"checker": source_check, "runner": runner_check},
        "protected_state_pre_post_exact_match": True,
        "rtl_tree_sha256": EXPECTED_RTL_TREE,
        "canonical_ppa_sha256_manifest": EXPECTED_PPA_SHA256,
        "source": artifact(OUT / "checker_source_at_execution.py"),
        "review_status": "PENDING_FRESH_REVIEWER",
    }
    write_json(OUT / "independent_check.json", check_report)
    write_sha256s(OUT)
    member_count = validate_manifest(OUT / "SHA256SUMS", OUT)
    print(
        "ACE2_OFFLINE_TOKEN0_V_INDEPENDENT_CHECK_PASS "
        f"model_execution_count=0 v_sha256={sha256_bytes(derived_v_raw)} "
        f"candidate_match=1 members={member_count}"
    )


if __name__ == "__main__":
    check()
