#!/usr/bin/env python3
"""Production per-layer adapter for the repair-0006 Stage-1 continuation."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
IDENTITY = (
    "stage1-official-rtl-backed-chat-demo-20260831T134355Z-"
    "attempt-0014-preparation-r5-continuation-repair-0006-production-executor"
)
PERMITTED_UNITS = tuple(
    f"position-00/layer-{index:02d}" for index in range(11, 24)
)
HIDDEN = 896


class ProductionExecutorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionExecutorError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def parse_source_bindings(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        require(
            separator == "  "
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and relative
            and relative not in records,
            f"invalid source binding: {line!r}",
        )
        records[relative] = digest
    return records


def validate_external_bindings() -> dict[str, str]:
    manifest = load(PACKAGE / "successor-manifest.json")
    binding = manifest["production_source_closure"]
    path = PACKAGE / binding["path"]
    require(
        path.is_file()
        and not path.is_symlink()
        and sha256_file(path) == binding["sha256"],
        "production source-binding index changed",
    )
    records = parse_source_bindings(path)
    required = set(binding["required_members"])
    require(required <= set(records), "production source-binding members are absent")
    for relative in sorted(required):
        source = ROOT / relative
        require(
            source.is_file()
            and not source.is_symlink()
            and sha256_file(source) == records[relative],
            f"bound production source changed: {relative}",
        )
    return records


def _signed_bytes(raw: bytes) -> list[int]:
    return [value - 256 if value & 0x80 else value for value in raw]


def _decode_state(
    key: str,
    generated_input: Path,
    torch: Any,
) -> dict[str, Any]:
    ordinal = int(key.rsplit("-", 1)[1])
    raw = generated_input.read_bytes()
    checkpoint = load(PACKAGE / "checkpoint-manifest.json")
    if ordinal == 11:
        unit = checkpoint["units"][ordinal]
        require(
            len(raw) == HIDDEN
            and sha256_bytes(raw) == unit["input_sha256"]
            and unit["dependency"]["key"] == "position-00/layer-10",
            "layer-11 input is not the authenticated layer-10 hidden state",
        )
        scale_binding = checkpoint["resume_state"]["hidden_scale"]
        scale_path = PACKAGE / scale_binding["path"]
        require(
            scale_path.is_file()
            and not scale_path.is_symlink()
            and scale_path.stat().st_size == 8
            and sha256_file(scale_path) == scale_binding["sha256"],
            "authenticated layer-10 output scale changed",
        )
        scale = struct.unpack("<d", scale_path.read_bytes())[0]
        fixed_q = torch.tensor(_signed_bytes(raw), dtype=torch.int8)
        return {
            "position": 0,
            "token_id": checkpoint["resume_state"]["decode_state"]["token_id"],
            "fixed_q": fixed_q,
            "fixed_scale": scale,
            "float_hidden": fixed_q.to(torch.float32) * scale,
        }

    try:
        descriptor = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionExecutorError(
            f"layer-{ordinal:02d} input descriptor is invalid"
        ) from error
    require(
        isinstance(descriptor, dict)
        and descriptor.get("schema") == "ace2-stage1-unit-state-v1"
        and descriptor.get("continuation_identity") == IDENTITY
        and descriptor.get("unit_key") == f"position-00/layer-{ordinal - 1:02d}"
        and descriptor.get("next_unit") == key,
        f"layer-{ordinal:02d} descriptor boundary changed",
    )
    state = descriptor.get("state")
    require(isinstance(state, dict), f"layer-{ordinal:02d} state is absent")
    hidden = bytes.fromhex(str(state.get("hidden_s8_hex", "")))
    float_raw = bytes.fromhex(str(state.get("float_hidden_f32le_hex", "")))
    require(
        len(hidden) == HIDDEN
        and sha256_bytes(hidden) == state.get("hidden_sha256")
        and len(float_raw) == HIDDEN * 4
        and sha256_bytes(float_raw) == state.get("float_hidden_sha256")
        and state.get("position") == 0
        and isinstance(state.get("fixed_scale"), (int, float))
        and math.isfinite(float(state["fixed_scale"]))
        and float(state["fixed_scale"]) > 0.0,
        f"layer-{ordinal:02d} state payload is invalid",
    )
    return {
        "position": 0,
        "token_id": int(state["token_id"]),
        "fixed_q": torch.tensor(_signed_bytes(hidden), dtype=torch.int8),
        "fixed_scale": float(state["fixed_scale"]),
        "float_hidden": torch.tensor(
            struct.unpack(f"<{HIDDEN}f", float_raw), dtype=torch.float32
        ),
    }


def _encode_state(
    key: str,
    next_state: dict[str, Any],
    backend: Any,
) -> bytes:
    ordinal = int(key.rsplit("-", 1)[1])
    hidden = backend.raw_int8(next_state["fixed_q"])
    float_raw = backend.tensor_bytes(
        next_state["float_hidden"].to(backend.torch.float32)
    )
    require(
        len(hidden) == HIDDEN and len(float_raw) == HIDDEN * 4,
        f"layer-{ordinal:02d} output state has the wrong width",
    )
    return canonical_bytes(
        {
            "schema": "ace2-stage1-unit-state-v1",
            "continuation_identity": IDENTITY,
            "unit_key": key,
            "next_unit": (
                f"position-00/layer-{ordinal + 1:02d}" if ordinal < 23 else None
            ),
            "state": {
                "position": 0,
                "token_id": int(next_state["token_id"]),
                "hidden_s8_hex": hidden.hex(),
                "hidden_sha256": sha256_bytes(hidden),
                "fixed_scale": float(next_state["fixed_scale"]),
                "float_hidden_f32le_hex": float_raw.hex(),
                "float_hidden_sha256": sha256_bytes(float_raw),
            },
        }
    )


class ProductionUnitExecutor:
    """Execute exactly one bound position-00 layer through reference plus RTL."""

    def __init__(self, runtime_root: Path) -> None:
        validate_external_bindings()
        tools = ROOT / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        import rtl_arbitrary_text_generation_backend as backend
        from safetensors import safe_open

        checkpoint = load(PACKAGE / "checkpoint-manifest.json")
        snapshot = ROOT / checkpoint["model_checkpoint_identity"]["files"][
            "model.safetensors"
        ]["path"]
        self.backend = backend
        self.safe_open = safe_open
        self.runtime_root = runtime_root
        self.snapshot = snapshot.parent
        self.backend.configure_snapshot(self.snapshot)
        self.rank1_sidecar: Any = None

    def __call__(self, key: str, generated_input: Path) -> dict[str, Any]:
        require(key in PERMITTED_UNITS, f"unit is outside the production grant: {key}")
        ordinal = int(key.rsplit("-", 1)[1])
        state = _decode_state(key, generated_input, self.backend.torch)
        cache = self.backend.empty_layer_cache()
        unit_root = generated_input.parent / "production-unit"
        with self.backend.torch.no_grad(), self.safe_open(
            self.backend.canonical.MODEL, framework="pt", device="cpu"
        ) as weights, self.safe_open(
            self.backend.canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            if ordinal == 23 and self.rank1_sidecar is None:
                self.rank1_sidecar = self.backend.Rank1SidecarSession(
                    unit_root / "rank1-sidecar"
                )
            derived, next_state, _template = self.backend.derive_layer_token(
                ordinal,
                state,
                cache,
                None,
                weights,
                adapter,
                rank1_sidecar=self.rank1_sidecar,
                sidecar_working=(
                    unit_root / "rank1-sidecar-step" if ordinal == 23 else None
                ),
            )
            rtl_root = unit_root / "rtl"
            rtl = self.backend.run_layer_rtl(rtl_root, derived)
            cache_record = self.backend.consume_rtl_cache_append(
                cache,
                rtl,
                layer_id=ordinal,
                absolute_position=0,
            )
        mismatches = rtl["integer_boundary_mismatches"]
        require(
            rtl["status"] == "PASS_SINGLE_TOKEN_ALL_BOUNDARIES_RTL"
            and all(int(value) == 0 for value in mismatches.values())
            and cache_record["host_cache_append_replaced"] is True,
            f"layer-{ordinal:02d} RTL/reference result did not authenticate",
        )
        result_path = rtl_root / "rtl_execution.json"
        return {
            "output": _encode_state(key, next_state, self.backend),
            "validation_status": "PASS_RTL_REFERENCE_AUTHENTICATED",
            "evidence": {
                "rtl_reference_agreement": True,
                "status": rtl["status"],
                "layer_id": ordinal,
                "absolute_position": 0,
                "integer_boundary_mismatches": mismatches,
                "rtl_execution_sha256": sha256_file(result_path),
                "rtl_execution": load(result_path),
                "cache_source": cache_record,
                "production_backend": "tools/rtl_arbitrary_text_generation_backend.py",
            },
        }


def make_production_unit_executor(runtime_root: Path) -> ProductionUnitExecutor:
    return ProductionUnitExecutor(runtime_root)
