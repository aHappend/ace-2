#!/usr/bin/env python3
"""Preflight the canonical two-token, 24-layer LoRA V4 ACE-2 RTL mission."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import safetensors
import torch
import transformers
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_layer0_full_rtl as canonical
import run_lora_v4_layer0_two_token_kv_full_rtl as layer0


CONTRACT = ROOT / "design/LORA_V4_TWO_TOKEN_24LAYER_RTL_CONTRACT.json"
MISSION_SPEC = ROOT / "design/SPEC.md"
BENCHMARK_INTERFACE = ROOT / "design/BENCHMARK_INTERFACE.json"
PREVIEW_ROOT = ROOT / "build/lora-v4-two-token-24layer-rtl-preview"
OFFICIAL_ROOT = ROOT / "evidence/verification/lora-v4-two-token-24layer-rtl-v1"
OFFICIAL_ATTEMPT = OFFICIAL_ROOT / "attempt-0001"
ACCEPTED_ROOT = ROOT / "evidence/verification/lora-v4-layer0-two-token-kv-full-v1"
ACCEPTED_ATTEMPT_1 = ACCEPTED_ROOT / "attempt-0001"
ACCEPTED_ATTEMPT_2 = ACCEPTED_ROOT / "attempt-0002"
ACCEPTED_REPAIR_CONTRACT = ACCEPTED_ROOT / "repair-0002-contract.json"
ACCEPTED_V2_SPEC = ROOT / "design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_SPEC_V2.md"
ACCEPTED_V2_INTERFACE = ROOT / "design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_INTERFACE_V2.json"
ACCEPTED_V2_RUNNER = ROOT / "tools/run_lora_v4_layer0_two_token_kv_full_rtl_v2.py"

LAYERS = 24
HIDDEN = 896
INTERMEDIATE = 4864
LORA_RANK = 16
LORA_ALPHA = 32
TOKEN_IDS = [5576, 12, 17, 467, 19, 32, 23, 6193, 36929, 29295, 33085, 13]
PROJECTIONS = {
    "q": ("self_attn.q_proj", HIDDEN, HIDDEN),
    "k": ("self_attn.k_proj", 128, HIDDEN),
    "v": ("self_attn.v_proj", 128, HIDDEN),
    "o": ("self_attn.o_proj", HIDDEN, HIDDEN),
    "gate": ("mlp.gate_proj", INTERMEDIATE, HIDDEN),
    "up": ("mlp.up_proj", INTERMEDIATE, HIDDEN),
    "down": ("mlp.down_proj", HIDDEN, INTERMEDIATE),
}
EXPECTED_PROJECTION_COUNT = LAYERS * len(PROJECTIONS)
EXPECTED_LOGICAL_WEIGHT_COUNT = LAYERS * sum(out_features * in_features for _, out_features, in_features in PROJECTIONS.values())
EXPECTED_PROJECTION_CHANNELS = LAYERS * 2 * sum(out_features for _, out_features, _ in PROJECTIONS.values())
EXPECTED_ACCEPTED_HASHES = {
    "attempt_0001_sums": "8a9eca9fc23a6ed40c9faa18e76dd01ef6e0587472e0488de92aa625db7a09aa",
    "attempt_0002_sums": "7b27e7f9d93777b5015094579495342f0d5b19e013b7e74255c4a953a7dd5d05",
    "repair_contract": "c112e751419e7e5c224c3f027c39441f8ae2a924b089ad6e211b79119f7cbc35",
    "v2_spec": "e13a6cd8656691ae8476e5ea06b7e12b620d04f6b7de5f8340da2bbfc75973d2",
    "v2_interface": "104afaf2af51945bdb469da841713a8bc2df82f578bfe0bf1ca5037e38512c33",
    "v2_runner": "58e5537af58e0820b75bfbb91889b8b98025ea5295780c21b10f8e684f981d4e",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(canonical.raw_tensor_bytes(value)).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def public_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path, *, public_alias: str | None = None) -> dict[str, Any]:
    return {
        "path": public_alias if public_alias is not None else public_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def next_preview() -> Path:
    index = 1
    while (PREVIEW_ROOT / f"preflight-{index:04d}").exists():
        index += 1
    return PREVIEW_ROOT / f"preflight-{index:04d}"


def latest_preview() -> Path:
    candidates = sorted(path for path in PREVIEW_ROOT.glob("preflight-*") if path.is_dir())
    require(bool(candidates), "no 24-layer preview preflight exists")
    return candidates[-1]


def tool_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "transformers": transformers.__version__,
    }


def expected_adapter_keys() -> set[str]:
    keys: set[str] = set()
    for layer_id in range(LAYERS):
        for suffix, _, _ in PROJECTIONS.values():
            prefix = f"base_model.model.model.layers.{layer_id}.{suffix}"
            keys.add(prefix + ".lora_A.weight")
            keys.add(prefix + ".lora_B.weight")
    return keys


def validate_contract() -> dict[str, Any]:
    require(CONTRACT.is_file(), "24-layer mission contract is absent")
    contract = read_json(CONTRACT)
    require(contract.get("schema_version") == 2 and contract.get("version") == "2.0", "mission contract is not frozen V2")
    require(contract.get("mission_id") == "extend-two-token-24layer-rtl-stack", "mission ID changed")
    require(contract["architecture"]["layers"] == LAYERS, "layer count changed")
    require(contract["fixed_input"]["token_ids"] == TOKEN_IDS, "frozen tokenizer sequence changed")
    require(contract["evaluator_policy"]["projection_channels_total"] == EXPECTED_PROJECTION_CHANNELS, "projection-channel total changed")
    require(contract["quantization"]["weight_projection_count"] == EXPECTED_PROJECTION_COUNT, "projection count changed")
    require(contract["quantization"]["logical_w4_weight_count"] == EXPECTED_LOGICAL_WEIGHT_COUNT, "logical W4 count changed")
    require(contract["official_attempt_gate"]["status"] == "LOCKED_PENDING_PREVIEW_PREPARE_RUN_CHECK_CLOSURE", "official gate changed")
    require("prompt" not in contract["fixed_input"], "private prompt text reappeared in mission contract")
    require("base_model_path" not in contract["canonical_checkpoint"], "private model-cache path reappeared in mission contract")
    normative = contract["normative_contracts"]
    require(sha256_file(MISSION_SPEC) == normative["specification"]["sha256"], "mission SPEC.md hash changed")
    require(sha256_file(BENCHMARK_INTERFACE) == normative["benchmark_interface"]["sha256"], "mission benchmark interface hash changed")
    require(sha256_file(Path(__file__)) == normative["runner"]["sha256"], "24-layer preview runner hash changed")
    return contract


def validate_accepted_parent() -> dict[str, Any]:
    paths = {
        "attempt_0001_sums": ACCEPTED_ATTEMPT_1 / "SHA256SUMS",
        "attempt_0002_sums": ACCEPTED_ATTEMPT_2 / "SHA256SUMS",
        "repair_contract": ACCEPTED_REPAIR_CONTRACT,
        "v2_spec": ACCEPTED_V2_SPEC,
        "v2_interface": ACCEPTED_V2_INTERFACE,
        "v2_runner": ACCEPTED_V2_RUNNER,
    }
    for name, path in paths.items():
        require(path.is_file(), f"accepted parent artifact is absent: {path}")
        require(sha256_file(path) == EXPECTED_ACCEPTED_HASHES[name], f"accepted parent artifact changed: {name}")
    attempt_1_members = layer0.verify_sha256s(ACCEPTED_ATTEMPT_1)
    attempt_2_members = layer0.verify_sha256s(ACCEPTED_ATTEMPT_2)
    require(attempt_1_members == 143, "accepted attempt-0001 member count changed")
    require(attempt_2_members == 143, "accepted attempt-0002 member count changed")
    return {
        "attempt_0001_members": attempt_1_members,
        "attempt_0002_members": attempt_2_members,
        "bindings": {name: file_record(path) for name, path in paths.items()},
    }


def validate_provenance(contract: dict[str, Any]) -> dict[str, Any]:
    required = [canonical.MODEL, canonical.CONFIG, canonical.ADAPTER, canonical.ADAPTER_CONFIG, canonical.CHECKPOINT_IDENTITY]
    for path in required:
        require(path.is_file(), f"canonical artifact is absent: {path}")
    checkpoint = contract["canonical_checkpoint"]
    require(sha256_file(canonical.MODEL) == checkpoint["base_model_sha256"], "base model hash changed")
    require(sha256_file(canonical.ADAPTER) == checkpoint["adapter_sha256"], "adapter hash changed")
    identity = read_json(canonical.CHECKPOINT_IDENTITY)
    require(identity["tree_sha256"] == checkpoint["checkpoint_tree_sha256"], "checkpoint tree hash changed")
    config = read_json(canonical.CONFIG)
    adapter_config = read_json(canonical.ADAPTER_CONFIG)
    require(config["num_hidden_layers"] == LAYERS, "model layer count changed")
    require(config["hidden_size"] == HIDDEN and config["intermediate_size"] == INTERMEDIATE, "model geometry changed")
    require(config["num_attention_heads"] == 14 and config["num_key_value_heads"] == 2, "attention geometry changed")
    require(adapter_config["r"] == LORA_RANK and adapter_config["lora_alpha"] == LORA_ALPHA, "LoRA scaling changed")
    require(set(adapter_config["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}, "LoRA target set changed")
    token_ids = canonical.tokenizer_ids()
    require(token_ids == TOKEN_IDS, "local tokenizer output changed")
    return {
        "model": file_record(canonical.MODEL, public_alias="canonical-source/model.safetensors"),
        "adapter": file_record(canonical.ADAPTER),
        "checkpoint_identity": file_record(canonical.CHECKPOINT_IDENTITY),
        "config": file_record(canonical.CONFIG, public_alias="canonical-source/config.json"),
        "adapter_config": file_record(canonical.ADAPTER_CONFIG),
        "token_ids": token_ids,
    }


def validate_rtl_sources(contract: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for family, frozen in contract["rtl_sources"].items():
        path = ROOT / frozen["path"]
        require(path.is_file(), f"RTL source is absent: {path}")
        require(sha256_file(path) == frozen["sha256"], f"RTL source changed: {family}")
        records[family] = file_record(path)
    return records


def quantize_projection(merged: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = merged.to(torch.float64)
    weight_scale = weight.abs().amax(dim=1) / 7.0
    weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
    qweight = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
    require(int(qweight.min().item()) >= -8 and int(qweight.max().item()) <= 7, "W4 range violation")
    return qweight.contiguous(), weight_scale.contiguous()


def audit_all_layer_weights(log_lines: list[str]) -> dict[str, Any]:
    accepted_manifest = read_json(ACCEPTED_ATTEMPT_2 / "manifest.json")
    accepted_weight_hashes = {
        name: accepted_manifest["tensors"][f"shared_{name}_qweight_s4_in_s8.bin"]["sha256"]
        for name in PROJECTIONS
    }
    layers: list[dict[str, Any]] = []
    projection_count = 0
    logical_weight_count = 0
    packed_w4_payload_bytes = 0
    negative_eight_count = 0
    zero_quantized_weight_count = 0
    expected_keys = expected_adapter_keys()

    with torch.no_grad(), safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        actual_adapter_keys = set(adapter.keys())
        require(actual_adapter_keys == expected_keys, "adapter tensor key set is not exactly 24 layers x 7 projections x A/B")
        model_keys = set(weights.keys())
        require("model.embed_tokens.weight" in model_keys, "embedding tensor is absent")
        for layer_id in range(LAYERS):
            require(f"model.layers.{layer_id}.input_layernorm.weight" in model_keys, f"layer {layer_id} input RMSNorm gain is absent")
            require(f"model.layers.{layer_id}.post_attention_layernorm.weight" in model_keys, f"layer {layer_id} post-attention RMSNorm gain is absent")
            layer_projections: list[dict[str, Any]] = []
            for name, (suffix, out_features, in_features) in PROJECTIONS.items():
                tensor_name = f"model.layers.{layer_id}.{suffix}"
                base_key = tensor_name + ".weight"
                adapter_prefix = "base_model.model." + tensor_name
                lora_a_key = adapter_prefix + ".lora_A.weight"
                lora_b_key = adapter_prefix + ".lora_B.weight"
                require(base_key in model_keys, f"base tensor is absent: {base_key}")
                base_tensor = weights.get_tensor(base_key).contiguous()
                lora_a = adapter.get_tensor(lora_a_key).contiguous()
                lora_b = adapter.get_tensor(lora_b_key).contiguous()
                require(list(base_tensor.shape) == [out_features, in_features], f"base shape changed: {tensor_name}")
                require(list(lora_a.shape) == [LORA_RANK, in_features], f"LoRA A shape changed: {tensor_name}")
                require(list(lora_b.shape) == [out_features, LORA_RANK], f"LoRA B shape changed: {tensor_name}")
                merged, source_hashes = canonical.merge_projection(weights, adapter, tensor_name)
                qweight, weight_scale = quantize_projection(merged)
                qweight_sha256 = sha256_tensor(qweight)
                if layer_id == 0:
                    require(qweight_sha256 == accepted_weight_hashes[name], f"layer-0 accepted W4 tensor changed: {name}")
                logical_weights = out_features * in_features
                projection_count += 1
                logical_weight_count += logical_weights
                packed_w4_payload_bytes += (logical_weights + 1) // 2
                negative_eight_count += int((qweight == -8).sum().item())
                zero_quantized_weight_count += int((qweight == 0).sum().item())
                record = {
                    "name": name,
                    "tensor_name": tensor_name,
                    "shape": [out_features, in_features],
                    "logical_weights": logical_weights,
                    "packed_w4_payload_bytes": (logical_weights + 1) // 2,
                    "source_hashes": source_hashes,
                    "qweight_s4_in_s8_sha256": qweight_sha256,
                    "weight_scale_f64le_sha256": sha256_tensor(weight_scale),
                    "qweight_min": int(qweight.min().item()),
                    "qweight_max": int(qweight.max().item()),
                    "negative_eight_count": int((qweight == -8).sum().item()),
                }
                layer_projections.append(record)
                log_lines.append(
                    f"layer={layer_id:02d} projection={name} shape={out_features}x{in_features} qweight_sha256={qweight_sha256}"
                )
                del base_tensor, lora_a, lora_b, merged, qweight, weight_scale
            layer_record = {
                "layer_id": layer_id,
                "projection_count": len(layer_projections),
                "logical_weights": sum(item["logical_weights"] for item in layer_projections),
                "projection_manifest_sha256": canonical_json_sha256(layer_projections),
                "projections": layer_projections,
            }
            layers.append(layer_record)
            log_lines.append(
                f"layer={layer_id:02d} complete manifest_sha256={layer_record['projection_manifest_sha256']}"
            )

    require(projection_count == EXPECTED_PROJECTION_COUNT, "24-layer projection count mismatch")
    require(logical_weight_count == EXPECTED_LOGICAL_WEIGHT_COUNT, "24-layer logical weight count mismatch")
    require(packed_w4_payload_bytes == (EXPECTED_LOGICAL_WEIGHT_COUNT + 1) // 2, "packed W4 byte count mismatch")
    require([item["layer_id"] for item in layers] == list(range(LAYERS)), "layer order changed")
    return {
        "status": "PASS_REAL_LORA_W4_ALL_24_LAYERS",
        "projection_count": projection_count,
        "logical_weight_count": logical_weight_count,
        "packed_w4_payload_bytes": packed_w4_payload_bytes,
        "negative_eight_count": negative_eight_count,
        "zero_quantized_weight_count": zero_quantized_weight_count,
        "accepted_layer0_qweight_hashes_reproduced": accepted_weight_hashes,
        "ordered_layer_manifest_sha256": canonical_json_sha256(layers),
        "layers": layers,
    }


def write_sha256s(directory: Path) -> None:
    members = sorted(path for path in directory.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256_file(path)}  {path.name}" for path in members]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_preview(directory: Path) -> dict[str, Any]:
    sums = directory / "SHA256SUMS"
    manifest_path = directory / "manifest.json"
    log_path = directory / "preflight.log"
    require(sums.is_file() and manifest_path.is_file() and log_path.is_file(), "preview closure is incomplete")
    count = 0
    for row in sums.read_text(encoding="utf-8").splitlines():
        expected, name = row.split("  ", 1)
        member = directory / name
        require(member.is_file(), f"preview member is absent: {name}")
        require(sha256_file(member) == expected, f"preview member hash changed: {name}")
        count += 1
    manifest = read_json(manifest_path)
    require(manifest["status"] == "PASS_24LAYER_REAL_LORA_W4_ARTIFACT_PREFLIGHT", "preview status is not PASS")
    require(manifest["official_attempt_consumed"] is False, "preview consumed official attempt")
    audit = manifest["weight_audit"]
    require(audit["projection_count"] == EXPECTED_PROJECTION_COUNT, "preview projection count changed")
    require(audit["logical_weight_count"] == EXPECTED_LOGICAL_WEIGHT_COUNT, "preview logical weight count changed")
    require(len(audit["layers"]) == LAYERS, "preview layer count changed")
    require(manifest["accepted_parent"]["attempt_0002_members"] == 143, "accepted parent member count changed")
    return {
        "status": "PASS_24LAYER_PREFLIGHT_READBACK",
        "preview": relative(directory),
        "members": count,
        "manifest_sha256": sha256_file(manifest_path),
        "projection_count": audit["projection_count"],
        "logical_weight_count": audit["logical_weight_count"],
        "packed_w4_payload_bytes": audit["packed_w4_payload_bytes"],
        "ordered_layer_manifest_sha256": audit["ordered_layer_manifest_sha256"],
    }


def preflight() -> None:
    start = time.perf_counter()
    require(not OFFICIAL_ATTEMPT.exists(), "official attempt already exists; preview preflight refuses to run")
    preview = next_preview()
    preview.mkdir(parents=True)
    log_lines = ["preview_only=true", "official_attempt_consumed=false"]
    try:
        contract = validate_contract()
        accepted_parent = validate_accepted_parent()
        provenance = validate_provenance(contract)
        rtl_sources = validate_rtl_sources(contract)
        versions = tool_versions()
        require(versions == {key: contract["toolchain"][key] for key in versions}, "Python dependency versions changed")
        weight_audit = audit_all_layer_weights(log_lines)
        elapsed = time.perf_counter() - start
        peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        manifest = {
            "schema_version": 1,
            "status": "PASS_24LAYER_REAL_LORA_W4_ARTIFACT_PREFLIGHT",
            "claim_boundary": "real checkpoint-176 LoRA merge and established W4 quantization for all seven projections in layers 0 through 23; no A8 activation carry or RTL execution claim",
            "contract": file_record(CONTRACT),
            "normative_specification": file_record(MISSION_SPEC),
            "benchmark_interface": file_record(BENCHMARK_INTERFACE),
            "runner": file_record(Path(__file__)),
            "accepted_parent": accepted_parent,
            "provenance": provenance,
            "rtl_sources": rtl_sources,
            "tool_versions": versions,
            "weight_audit": weight_audit,
            "expected_projection_channels_for_future_rtl": EXPECTED_PROJECTION_CHANNELS,
            "elapsed_wall_seconds": elapsed,
            "peak_rss_kib": peak_rss_kib,
            "official_attempt_consumed": False,
            "official_attempt_path_absent": not OFFICIAL_ATTEMPT.exists(),
            "next_required_stage": "carry both A8 hidden states through layers 0-23, emit per-layer K/V and independent fixed-point traces, then execute preview RTL prepare/run/check",
        }
        write_json(preview / "manifest.json", manifest)
        (preview / "preflight.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        write_sha256s(preview)
        result = verify_preview(preview)
        result.update({"elapsed_wall_seconds": elapsed, "peak_rss_kib": peak_rss_kib})
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        elapsed = time.perf_counter() - start
        failure = {
            "schema_version": 1,
            "status": "FAIL_24LAYER_PREFLIGHT",
            "failure_taxonomy": "ARTIFACT_OR_PROVENANCE_PREFLIGHT",
            "root_cause_hypothesis": str(exc),
            "regression_required": "rerun the same preview preflight after the bounded source or binding repair and require complete 24-layer readback",
            "elapsed_wall_seconds": elapsed,
            "official_attempt_consumed": False,
        }
        log_lines.append(f"failure={exc}")
        write_json(preview / "failure.json", failure)
        (preview / "preflight.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        write_sha256s(preview)
        raise


def official_locked(action: str) -> None:
    validate_contract()
    require(not OFFICIAL_ATTEMPT.exists(), "official attempt exists while the frozen gate is locked")
    raise RuntimeError(
        f"{action} is locked by the frozen mission contract until preview A8 carry, independent golden, and RTL prepare/run/check closure are implemented"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--check-preflight", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.check_preflight:
        print(json.dumps(verify_preview(latest_preview()), sort_keys=True))
    elif args.prepare:
        official_locked("prepare")
    elif args.run:
        official_locked("run")
    else:
        official_locked("check")


if __name__ == "__main__":
    main()
