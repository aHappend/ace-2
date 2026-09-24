#!/usr/bin/env python3
"""Run the authorized BF16-faithful replacement C4 rope_q diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.utils.hub import cached_file

import capture_replay_c4_rope_q as core


ROOT = core.ROOT
SELF = Path(__file__).resolve()
EXPECTED_OUTPUT = (
    ROOT / "evidence/diagnostics/c4-rope-q-token1-capture-replay-20260803-v2"
).resolve()
FAILED_OUTPUT = (
    ROOT / "evidence/diagnostics/c4-rope-q-token1-capture-replay-20260803-v1"
).resolve()
CURRENT_MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-rope-q-bf16-preflight-c4-replay-v1/mission.json"
)

EXPECTED_CAPTURE_SOURCE_SHA256 = (
    "ace278bee5701c77ad43bdf454a4443b85eaff46ea2921a266018c7560e965c8"
)
EXPECTED_QWEN_SOURCE_SHA256 = (
    "a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb"
)
EXPECTED_CONFIG_SHA256 = (
    "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"
)
FAILED_FILES_SHA256 = {
    "capture_execution_consumed.json": (
        "bc23e47f3bc3f85ac04141097797815e055b7234ef1e2ae9b67adcbf0fc1cb6b"
    ),
    "capture_failure.json": (
        "36544e4db810a72c80a54c991e2a40b1de1f75e0a02500584939a57e2aa8c62d"
    ),
    "capture_failure.log": (
        "3160fc0725a32f3eedf747de5e06371e088072da06e01650208a5980f1dfba93"
    ),
    "capture_source_at_execution.py": (
        "1c57e24f11a6338fe994d1ed63e4dd81aa19697b4ff2dd52bc304c7f44f8bfa5"
    ),
    "preflight.json": (
        "191632cfa6128f60fc2fef83188d5098888a9e6cfe0177668ff07a15d836f18e"
    ),
    "preflight.log": (
        "9594000448cdbcb895f471a0cc79cf2f4915fc9a89c66292df7e69e11eec1279"
    ),
    "preflight_source_at_execution.py": (
        "1c57e24f11a6338fe994d1ed63e4dd81aa19697b4ff2dd52bc304c7f44f8bfa5"
    ),
}

_ACTIVE_OUTPUT: Path | None = None
_ORIGINAL_SOURCE_BINDINGS = core.source_bindings


def artifact_any(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return (
        core.artifact(resolved)
        if ROOT.resolve() in resolved.parents
        else core.external_artifact(resolved)
    )


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or core.sha256_file(path) != expected:
        raise RuntimeError(f"{label} differs from the authorized SHA256")


def verify_failed_attempt() -> dict[str, Any]:
    if not FAILED_OUTPUT.is_dir():
        raise RuntimeError("sealed failed token-1 attempt is missing")
    actual = {
        path.relative_to(FAILED_OUTPUT).as_posix()
        for path in FAILED_OUTPUT.rglob("*")
        if path.is_file()
    }
    if actual != set(FAILED_FILES_SHA256):
        raise RuntimeError(
            "sealed failed token-1 attempt file set changed: "
            f"actual={sorted(actual)}"
        )
    files = {}
    for relative, expected in sorted(FAILED_FILES_SHA256.items()):
        path = FAILED_OUTPUT / relative
        require_hash(path, expected, f"failed attempt {relative}")
        files[relative] = core.artifact(path)
    consumed = json.loads(
        (FAILED_OUTPUT / "capture_execution_consumed.json").read_text(
            encoding="utf-8"
        )
    )
    if consumed.get("coordinate") != {
        "dataset_record_count": 1,
        "head": 0,
        "token": 1,
    }:
        raise RuntimeError("sealed failed attempt coordinate changed")
    prohibited = {
        "witness.json",
        "capture_contract.json",
        "replay_results.json",
        "verification.json",
    }
    if prohibited & actual:
        raise RuntimeError("sealed failed attempt unexpectedly contains a witness/replay")
    return {
        "classification": "immutable_consumed_failed_attempt_no_witness_no_replay",
        "path": FAILED_OUTPUT.relative_to(ROOT).as_posix(),
        "files": files,
    }


def resolve_output(value: Path) -> Path:
    output = (value if value.is_absolute() else ROOT / value).resolve()
    if output != EXPECTED_OUTPUT:
        raise SystemExit(
            "replacement output must be "
            f"{EXPECTED_OUTPUT.relative_to(ROOT).as_posix()}"
        )
    return output


def ensure_replacement_scope(output_dir: Path) -> None:
    if output_dir.resolve() != EXPECTED_OUTPUT:
        raise RuntimeError("unexpected replacement output directory")
    verify_failed_attempt()
    actual = {
        path.resolve()
        for path in (ROOT / "evidence/diagnostics").glob(
            "c4-rope-q-token1-capture-replay-*"
        )
        if path.is_dir()
    }
    allowed = {FAILED_OUTPUT, EXPECTED_OUTPUT}
    if not actual <= allowed or FAILED_OUTPUT not in actual:
        raise RuntimeError(
            "unexpected token-1 diagnostic directory set: "
            f"{sorted(str(path) for path in actual)}"
        )


def q15(values: torch.Tensor) -> list[int]:
    return (
        torch.round(values.to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
        .reshape(-1)
        .tolist()
    )


def transform_nonidentity_proof(
    cos_q15: list[int], sin_q15: list[int]
) -> dict[str, Any]:
    probe = [index + 1 for index in range(core.HEAD_DIM)]
    transformed = []
    identity = []
    for index, value in enumerate(probe):
        pair = index + 32 if index < 32 else index - 32
        rotated = value * cos_q15[index]
        rotated += (
            -probe[pair] * sin_q15[index]
            if index < 32
            else probe[pair] * sin_q15[index]
        )
        transformed.append(rotated)
        identity.append(value * 32767)
    changed = [
        index
        for index, (actual, baseline) in enumerate(
            zip(transformed, identity, strict=True)
        )
        if actual != baseline
    ]
    return {
        "probe_s16": probe,
        "identity_numerator": identity,
        "rotated_q15_numerator": transformed,
        "changed_lane_indices": changed,
        "non_identity": bool(changed),
    }


def qwen_metadata() -> tuple[dict[str, Any], Path, Path]:
    manifest, config, _rtl_binding = core.load_contracts(require_rtl_binding=False)
    versions = core.validate_runtime(config)
    model_spec = manifest["model"]
    model_config = core.AutoConfig.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        local_files_only=True,
    )
    resolved_revision = getattr(model_config, "_commit_hash", None)
    if resolved_revision != model_spec["revision"]:
        raise RuntimeError("dummy Qwen rotary config resolved to the wrong revision")
    head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    if head_dim != core.HEAD_DIM:
        raise RuntimeError("dummy Qwen rotary head dimension changed")

    qwen_source = Path(modeling_qwen2.__file__).resolve()
    config_source = Path(
        cached_file(
            model_spec["repository"],
            "config.json",
            revision=model_spec["revision"],
            local_files_only=True,
        )
    ).resolve()
    require_hash(qwen_source, EXPECTED_QWEN_SOURCE_SHA256, "installed Qwen2 source")
    require_hash(config_source, EXPECTED_CONFIG_SHA256, "pinned local model config")

    rotary = Qwen2RotaryEmbedding(model_config, device="cpu")
    dummy = torch.zeros(
        (1, 1, core.HEAD_DIM), dtype=torch.bfloat16, device="cpu"
    )
    position_ids = torch.tensor([[core.TOKEN_INDEX]], dtype=torch.long, device="cpu")
    with torch.inference_mode():
        cos, sin = rotary(dummy, position_ids)
    if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
        raise RuntimeError("dummy Qwen rotary did not return BF16 metadata")
    if cos.device.type != "cpu" or sin.device.type != "cpu":
        raise RuntimeError("dummy Qwen rotary did not remain on CPU")
    if tuple(cos.shape) != (1, 1, core.HEAD_DIM) or tuple(sin.shape) != (
        1,
        1,
        core.HEAD_DIM,
    ):
        raise RuntimeError("dummy Qwen rotary metadata shape changed")
    return (
        {
            "coordinate": {
                "head": core.HEAD_INDEX,
                "head_dim": core.HEAD_DIM,
                "layer": 0,
                "operator": "rope_q",
                "token": core.TOKEN_INDEX,
            },
            "cos_q15_s16": q15(cos[0, 0]),
            "sin_q15_s16": q15(sin[0, 0]),
            "dummy_input": {
                "device": "cpu",
                "dtype": "torch.bfloat16",
                "shape": [1, 1, core.HEAD_DIM],
            },
            "position_ids": [core.TOKEN_INDEX],
            "model_config": {
                "head_dim": head_dim,
                "repository": model_spec["repository"],
                "requested_revision": model_spec["revision"],
                "resolved_revision": resolved_revision,
                "rope_scaling": model_config.rope_scaling,
                "rope_theta": float(model_config.rope_theta),
            },
            "packages": versions,
        },
        qwen_source,
        config_source,
    )


def prepare_preflight_output(output_dir: Path) -> None:
    ensure_replacement_scope(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"preflight.stdout", "preflight.stderr"}
    present = {path.name for path in output_dir.iterdir()}
    if present - allowed:
        raise RuntimeError(
            "replacement preflight output already contains artifacts: "
            f"{sorted(present)}"
        )
    for name in allowed:
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"raw {name} was not redirected before process launch")


def run_preflight(output_dir: Path) -> None:
    prepare_preflight_output(output_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("replacement preflight requires CPU-only visibility")
    require_hash(
        core.CAPTURE_SOURCE,
        EXPECTED_CAPTURE_SOURCE_SHA256,
        "repaired model-free capture source",
    )
    prestate = core.verify_frozen_state()
    failed_attempt = verify_failed_attempt()

    model_free = core.model_free_rope_metadata()
    qwen, qwen_source, config_source = qwen_metadata()

    shutil.copy2(core.CAPTURE_SOURCE, output_dir / "preflight_source_at_execution.py")
    shutil.copy2(SELF, output_dir / "preflight_driver_source_at_execution.py")
    shutil.copy2(
        qwen_source, output_dir / "preflight_qwen2_source_at_execution.py"
    )
    shutil.copy2(config_source, output_dir / "preflight_pinned_config.json")
    core.write_json(output_dir / "preflight_model_free_values.json", model_free)
    core.write_json(output_dir / "preflight_qwen_values.json", qwen)

    cos_mismatches = [
        index
        for index, (lhs, rhs) in enumerate(
            zip(
                model_free["cos_q15_s16"],
                qwen["cos_q15_s16"],
                strict=True,
            )
        )
        if lhs != rhs
    ]
    sin_mismatches = [
        index
        for index, (lhs, rhs) in enumerate(
            zip(
                model_free["sin_q15_s16"],
                qwen["sin_q15_s16"],
                strict=True,
            )
        )
        if lhs != rhs
    ]
    nonzero_sine = [
        index for index, value in enumerate(qwen["sin_q15_s16"]) if value != 0
    ]
    transform = transform_nonidentity_proof(
        qwen["cos_q15_s16"], qwen["sin_q15_s16"]
    )
    passed = (
        len(model_free["cos_q15_s16"]) == core.HEAD_DIM
        and len(model_free["sin_q15_s16"]) == core.HEAD_DIM
        and len(qwen["cos_q15_s16"]) == core.HEAD_DIM
        and len(qwen["sin_q15_s16"]) == core.HEAD_DIM
        and not cos_mismatches
        and not sin_mismatches
        and bool(nonzero_sine)
        and transform["non_identity"]
    )
    preflight = {
        "schema_version": 2,
        "status": (
            "pass_two_source_exact_model_execution_not_consumed"
            if passed
            else "fail_two_source_model_execution_not_consumed"
        ),
        "completed_at_utc": core.utc_now(),
        "command": [sys.executable, *sys.argv],
        "model_execution_count": 0,
        "weighted_model_load_count": 0,
        "metadata": model_free,
        "frozen_prestate": prestate,
        "source": core.artifact(core.CAPTURE_SOURCE),
        "source_archive": core.artifact(
            output_dir / "preflight_source_at_execution.py"
        ),
        "two_source_bf16_preflight": {
            "lane_count": core.HEAD_DIM,
            "exact_lane_equality": not cos_mismatches and not sin_mismatches,
            "cosine_mismatch_lane_indices": cos_mismatches,
            "sine_mismatch_lane_indices": sin_mismatches,
            "representable_nonzero_sine": bool(nonzero_sine),
            "nonzero_sine_lane_indices": nonzero_sine,
            "nonidentity_transform": transform,
            "model_free_values": core.artifact(
                output_dir / "preflight_model_free_values.json"
            ),
            "qwen_values": core.artifact(
                output_dir / "preflight_qwen_values.json"
            ),
            "sources": {
                "model_free_repaired": {
                    "live": core.artifact(core.CAPTURE_SOURCE),
                    "archive": core.artifact(
                        output_dir / "preflight_source_at_execution.py"
                    ),
                },
                "installed_qwen2": {
                    "live": core.external_artifact(qwen_source),
                    "archive": core.artifact(
                        output_dir / "preflight_qwen2_source_at_execution.py"
                    ),
                    "class": "Qwen2RotaryEmbedding",
                },
                "pinned_local_config": {
                    "live": core.external_artifact(config_source),
                    "archive": core.artifact(
                        output_dir / "preflight_pinned_config.json"
                    ),
                },
                "driver": core.artifact(
                    output_dir / "preflight_driver_source_at_execution.py"
                ),
            },
        },
        "failed_attempt_preserved": failed_attempt,
        "operator_mission_context": core.external_artifact(CURRENT_MISSION),
    }
    core.write_json(output_dir / "preflight.json", preflight)
    (output_dir / "preflight.log").write_text(
        "\n".join(
            [
                "weighted_model_load_count=0",
                "model_execution_count=0",
                f"lane_count={core.HEAD_DIM}",
                f"cosine_mismatch_count={len(cos_mismatches)}",
                f"sine_mismatch_count={len(sin_mismatches)}",
                f"nonzero_sine_lane_count={len(nonzero_sine)}",
                f"nonidentity_changed_lane_count={len(transform['changed_lane_indices'])}",
                f"status={'pass' if passed else 'fail'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("two-source BF16 metadata preflight failed closed")
    print(
        "ACE2_C4_ROPE_Q_BF16_PREFLIGHT_PASS "
        "weighted_model_loads=0 lanes=64 cos_mismatches=0 sin_mismatches=0 "
        f"nonzero_sine_lanes={len(nonzero_sine)} "
        f"nonidentity_lanes={len(transform['changed_lane_indices'])}"
    )


def validate_preflight(output_dir: Path) -> dict[str, Any]:
    ensure_replacement_scope(output_dir)
    require_hash(
        core.CAPTURE_SOURCE,
        EXPECTED_CAPTURE_SOURCE_SHA256,
        "repaired model-free capture source",
    )
    preflight_path = output_dir / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("two-source replacement preflight is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    gate = preflight.get("two_source_bf16_preflight", {})
    if preflight.get("status") != "pass_two_source_exact_model_execution_not_consumed":
        raise RuntimeError("two-source replacement preflight did not pass")
    if preflight.get("model_execution_count") != 0:
        raise RuntimeError("preflight records a model execution")
    if preflight.get("weighted_model_load_count") != 0:
        raise RuntimeError("preflight records a weighted model load")
    if gate.get("lane_count") != core.HEAD_DIM:
        raise RuntimeError("preflight lane count changed")
    if not gate.get("exact_lane_equality"):
        raise RuntimeError("preflight does not record exact all-lane equality")
    if gate.get("cosine_mismatch_lane_indices") or gate.get(
        "sine_mismatch_lane_indices"
    ):
        raise RuntimeError("preflight contains a metadata mismatch")
    if not gate.get("representable_nonzero_sine"):
        raise RuntimeError("preflight lacks a representable nonzero sine")
    if not gate.get("nonidentity_transform", {}).get("non_identity"):
        raise RuntimeError("preflight lacks a demonstrated non-identity transform")
    model_free = json.loads(
        (output_dir / "preflight_model_free_values.json").read_text(
            encoding="utf-8"
        )
    )
    qwen = json.loads(
        (output_dir / "preflight_qwen_values.json").read_text(encoding="utf-8")
    )
    for key in ("cos_q15_s16", "sin_q15_s16"):
        if len(model_free[key]) != core.HEAD_DIM or model_free[key] != qwen[key]:
            raise RuntimeError(f"archived two-source {key} values differ")
        if model_free[key] != preflight["metadata"][key]:
            raise RuntimeError(f"preflight metadata differs from archived {key}")
    source_records = gate["sources"]
    require_hash(
        output_dir / "preflight_source_at_execution.py",
        EXPECTED_CAPTURE_SOURCE_SHA256,
        "archived repaired model-free source",
    )
    require_hash(
        Path(source_records["installed_qwen2"]["live"]["path"]),
        EXPECTED_QWEN_SOURCE_SHA256,
        "live installed Qwen2 source",
    )
    require_hash(
        output_dir / "preflight_qwen2_source_at_execution.py",
        EXPECTED_QWEN_SOURCE_SHA256,
        "archived installed Qwen2 source",
    )
    require_hash(
        Path(source_records["pinned_local_config"]["live"]["path"]),
        EXPECTED_CONFIG_SHA256,
        "live pinned config",
    )
    require_hash(
        output_dir / "preflight_pinned_config.json",
        EXPECTED_CONFIG_SHA256,
        "archived pinned config",
    )
    for name in ("preflight.stdout", "preflight.stderr"):
        if not (output_dir / name).is_file():
            raise RuntimeError(f"raw {name} is missing")
    if preflight.get("frozen_prestate") != core.verify_frozen_state():
        raise RuntimeError("sealed state changed after the two-source preflight")
    verify_failed_attempt()
    return preflight


def prepare_replacement_harness(output_dir: Path) -> None:
    archive = output_dir / "replacement_harness_source_at_execution.py"
    if archive.exists():
        if core.sha256_file(archive) != core.sha256_file(SELF):
            raise RuntimeError("replacement harness source archive differs")
    else:
        shutil.copy2(SELF, archive)


def replacement_source_bindings() -> dict[str, dict[str, Any]]:
    if _ACTIVE_OUTPUT is None:
        raise RuntimeError("replacement source binding lacks an active output")
    bindings = _ORIGINAL_SOURCE_BINDINGS()
    bindings["replacement_harness_source_archive"] = core.artifact(
        _ACTIVE_OUTPUT / "replacement_harness_source_at_execution.py"
    )
    bindings["two_source_preflight_driver"] = core.artifact(
        _ACTIVE_OUTPUT / "preflight_driver_source_at_execution.py"
    )
    bindings["two_source_model_free_values"] = core.artifact(
        _ACTIVE_OUTPUT / "preflight_model_free_values.json"
    )
    bindings["two_source_qwen_values"] = core.artifact(
        _ACTIVE_OUTPUT / "preflight_qwen_values.json"
    )
    return bindings


def install_replacement_bindings(output_dir: Path) -> None:
    global _ACTIVE_OUTPUT
    _ACTIVE_OUTPUT = output_dir
    core.MISSION_CONTEXT = CURRENT_MISSION
    core.ensure_no_prior_token1_output = ensure_replacement_scope
    core.source_bindings = replacement_source_bindings


def run_capture(output_dir: Path) -> None:
    validate_preflight(output_dir)
    for name in ("capture.stdout", "capture.stderr"):
        if not (output_dir / name).is_file():
            raise RuntimeError(f"raw {name} was not redirected before process launch")
    prepare_replacement_harness(output_dir)
    install_replacement_bindings(output_dir)
    core.run_capture(output_dir)


def run_replay(output_dir: Path) -> None:
    validate_preflight(output_dir)
    for name in ("replay.stdout", "replay.stderr"):
        if not (output_dir / name).is_file():
            raise RuntimeError(f"raw {name} was not redirected before process launch")
    prepare_replacement_harness(output_dir)
    install_replacement_bindings(output_dir)
    core.run_replay(output_dir)


def run_verify(output_dir: Path) -> None:
    preflight = validate_preflight(output_dir)
    prepare_replacement_harness(output_dir)
    install_replacement_bindings(output_dir)
    core.run_verify(output_dir)
    verification_path = output_dir / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    gate = preflight["two_source_bf16_preflight"]
    verification["two_source_bf16_preflight"] = {
        "lane_count": gate["lane_count"],
        "exact_lane_equality": gate["exact_lane_equality"],
        "cosine_mismatches": len(gate["cosine_mismatch_lane_indices"]),
        "sine_mismatches": len(gate["sine_mismatch_lane_indices"]),
        "representable_nonzero_sine": gate["representable_nonzero_sine"],
        "nonzero_sine_lane_count": len(gate["nonzero_sine_lane_indices"]),
        "nonidentity_changed_lane_count": len(
            gate["nonidentity_transform"]["changed_lane_indices"]
        ),
        "model_free_values": core.artifact(
            output_dir / "preflight_model_free_values.json"
        ),
        "qwen_values": core.artifact(output_dir / "preflight_qwen_values.json"),
        "qwen_source_archive": core.artifact(
            output_dir / "preflight_qwen2_source_at_execution.py"
        ),
        "pinned_config_archive": core.artifact(
            output_dir / "preflight_pinned_config.json"
        ),
    }
    verification["replacement_authorization_provenance"] = {
        "mission": core.external_artifact(CURRENT_MISSION),
        "failed_attempt_preserved": verify_failed_attempt(),
        "replacement_harness": core.artifact(
            output_dir / "replacement_harness_source_at_execution.py"
        ),
    }
    verification["raw_process_logs"] = {
        name: core.artifact(output_dir / name)
        for name in (
            "preflight.stdout",
            "preflight.stderr",
            "capture.stdout",
            "capture.stderr",
            "replay.stdout",
            "replay.stderr",
        )
    }
    verification["frontier_disposition"] = (
        "engineer evidence is bit-exact for the single authorized token-1/head-0 "
        "witness; the accepted frontier remains through layer_0.v_proj pending "
        "Fresh Reviewer disposition"
    )
    core.write_json(verification_path, verification)
    sha256s = core.write_sha256s(output_dir)
    verification["sha256s"] = core.artifact(sha256s)
    core.write_json(verification_path, verification)
    print(
        "ACE2_C4_ROPE_Q_BF16_REPLACEMENT_VERIFY_PASS "
        "lanes=64 exactly_once=1 mismatches=0 "
        f"sha256s_sha256={core.sha256_file(sha256s)}"
    )


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
