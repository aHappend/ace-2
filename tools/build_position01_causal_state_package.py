#!/usr/bin/env python3
"""Create the sealed construction-only position-01 causal-state package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_position01_causal_state_package.py"
RESULT = "reports/ace2-post-transformer-token-selection-consumption-state-0005/runtime/selected-token-result.json"
TERMINAL = "reports/ace2-post-transformer-token-selection-consumption-state-0005/terminal.json"
CONSUMED = "reports/ace2-post-transformer-token-selection-consumption-state-0005/consumed.json"
GRANT = "reports/ace2-post-transformer-token-selection-manager-grant-0005/grant.json"
GRANT_PACKAGE = "reports/ace2-post-transformer-token-selection-authority-0005"
PREREQUISITE = f"{GRANT_PACKAGE}/prerequisite-manifest.json"
LAYER23_PACKAGE = "reports/ace2-layer23-runtime-pass-authority-0003"
LAYER23_STATE = "reports/ace2-layer23-continuation-0017-consumption-state-0003"
CHECKPOINT = f"{LAYER23_STATE}/runtime/checkpoint-manifest.json"
LAYER23_OUTPUT = f"{LAYER23_STATE}/runtime/published/position-00/layer-23/output.bin"
LAYER23_RESULT = f"{LAYER23_STATE}/runtime/published/position-00/layer-23/result.json"
LAYER23_TERMINAL = f"{LAYER23_STATE}/terminal.json"
LAYER23_CONSUMED = f"{LAYER23_STATE}/consumed.json"
ATTEMPT_INVENTORY = "reports/ace2-consumed-stage1-terminal-recovery-0001/attempt-0012-inventory.json"
ATTEMPT_ROOT = "build/ace2_chat_demo/stage1-official-rtl-backed-chat-demo-20260829T175516Z-runtime-output-0012-6f2c8d4e"
EXPECTED_RESULT_SHA256 = "73dd8c2e54cd5275ab0a28e11279fb1b03986e5f82ad7ad6c54e42a575ff1deb"
EXPECTED_CHECKPOINT_SHA256 = "2ce126478f2fbe7af50a316947df78f01cfecc625232a1e3cf94c86e3556c011"
EXPECTED_LAYER23_OUTPUT_SHA256 = "6646ca8322ec0c4271505683eeff751f1a48a6be9d85d19735e877e722fd3187"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required regular file absent: {relative}")
    return {"bytes": path.stat().st_size, "path": relative, "sha256": sha256_file(path)}


def tree_record(relative: str) -> dict[str, str]:
    sidecar = (ROOT / relative / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
    if len(sidecar) != 2 or sidecar[1] != "SHA256SUMS":
        raise RuntimeError(f"invalid tree-root sidecar: {relative}")
    if sha256_file(ROOT / relative / "SHA256SUMS") != sidecar[0]:
        raise RuntimeError(f"tree-root mismatch: {relative}")
    return {"path": relative, "tree_root_sha256": sidecar[0]}


def load(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {relative}")
    return value


def build(output: Path) -> dict[str, Any]:
    if os.path.lexists(output):
        raise RuntimeError(f"create-exclusive package already exists: {output}")
    if sha256_file(ROOT / RESULT) != EXPECTED_RESULT_SHA256:
        raise RuntimeError("selected-token result does not match the frozen digest")
    if sha256_file(ROOT / CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Layer-23 checkpoint does not match the frozen digest")
    if sha256_file(ROOT / LAYER23_OUTPUT) != EXPECTED_LAYER23_OUTPUT_SHA256:
        raise RuntimeError("Layer-23 output does not match the frozen digest")

    checkpoint = load(CHECKPOINT)
    kv_layers = checkpoint.get("resume_state", {}).get("kv_layers")
    if not isinstance(kv_layers, list) or len(kv_layers) != 24:
        raise RuntimeError("the frozen checkpoint does not contain 24 K/V layers")
    kv_publications = []
    for index, kv in enumerate(kv_layers):
        if index < 11:
            prefix = f"{ATTEMPT_ROOT}/tokens/position-00/layer-{index:02d}/rtl_observed"
            source = "ATTEMPT_0012_AUTHENTICATED_RUNTIME"
            paths = {
                "k_s8": f"{prefix}/k_cache_append_s8.bin",
                "v_s8": f"{prefix}/v_cache_append_s8.bin",
            }
        else:
            source = "APPEND_ONLY_AUTHENTICATED_SUCCESSOR_PUBLICATION"
            paths = {
                name: f"{LAYER23_STATE}/runtime/{kv[name]['path']}"
                for name in ("k_s8", "v_s8")
            }
        records = {name: file_record(path) for name, path in paths.items()}
        for name, record in records.items():
            if record["sha256"] != kv[name]["sha256"] or record["bytes"] != kv[name]["bytes"]:
                raise RuntimeError(f"checkpoint K/V mismatch at layer {index:02d}/{name}")
        kv_publications.append(
            {
                "key": f"position-00/layer-{index:02d}",
                "ordinal": index,
                "source": source,
                **records,
            }
        )

    sources = {
        "attempt0012_inventory": file_record(ATTEMPT_INVENTORY),
        "grant0005_consumed": file_record(CONSUMED),
        "grant0005_grant": file_record(GRANT),
        "grant0005_package": tree_record(GRANT_PACKAGE),
        "grant0005_prerequisite": file_record(PREREQUISITE),
        "layer23_checkpoint": file_record(CHECKPOINT),
        "layer23_consumed": file_record(LAYER23_CONSUMED),
        "layer23_output": file_record(LAYER23_OUTPUT),
        "layer23_package": tree_record(LAYER23_PACKAGE),
        "layer23_result": file_record(LAYER23_RESULT),
        "layer23_terminal": file_record(LAYER23_TERMINAL),
        "selected_token_result": file_record(RESULT),
        "selected_token_terminal": file_record(TERMINAL),
    }
    capacity = load(
        "build/ace2_chat_demo/stage1-official-rtl-backed-chat-demo-20260829T175516Z-attempt-0012-preparation/capacity.json"
    )
    zero_activity = {
        "authority_consumed": 0,
        "authority_created": 0,
        "checkpoint_mutation": 0,
        "executor_invocation": 0,
        "generated_tokens": 0,
        "grant_consumed": 0,
        "manager_grant_created": 0,
        "model_execution": 0,
        "ppa_execution": 0,
        "protected_state_delta": 0,
        "reference_execution": 0,
        "rtl_execution": 0,
        "stage1_closure": 0,
        "stage2_execution": 0,
        "u280_execution": 0,
        "unit_execution": 0,
        "workload_execution": 0,
    }
    manifest = {
        "activity": zero_activity,
        "causal_kv_publications": kv_publications,
        "claims_not_made": [
            "arbitrary fresh natural-language input acceptance",
            "readable multi-token output",
            "complete generation/KV/decode loop",
            "position-01 execution",
            "Stage-1 completion",
            "Stage-2 or U280 deployment",
        ],
        "position01_contract": {
            "absolute_position": 1,
            "causal_kv_state_sha256": checkpoint["resume_state"]["kv_state_sha256"],
            "execution_state": "UNSUPPORTED_AWAITING_SEPARATE_RUNTIME_AUTHORITY",
            "input_token_id": 106339,
            "ordered_command_schedule": [
                *[f"position-01/layer-{index:02d}" for index in range(24)],
                "position-01/post-transformer-token-selection",
            ],
            "output_namespace": "reports/ace2-position01-runtime-pass-0001",
            "resource_guard": {
                "construction_live_capacity_claim": "NOT_MADE",
                "enforcement": "RECHECK_IMMEDIATELY_BEFORE_ANY_OUTPUT_OR_WORKLOAD",
                **{
                    key: capacity[key]
                    for key in (
                        "package_max_bytes",
                        "required_available_bytes",
                        "runtime_output_max_bytes",
                        "safety_margin_bytes",
                        "temporary_write_max_bytes",
                    )
                },
            },
            "runtime_pass": {
                "authority_cardinality": 0,
                "consume_before_workload": True,
                "fail_closed": True,
                "permitted_effect": "NONE_IN_THIS_CONSTRUCTION_PACKAGE",
                "requires_separate_external_review_host_binding_and_manager_grant": True,
                "resource_guard_required_before_output_or_workload": True,
            },
        },
        "schema": "ace2-position01-causal-state-successor-package-v1",
        "sources": sources,
        "stage_status": {
            "rtl": "in_progress",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
            "stage1_completion_claimed": False,
            "stage2_completion_claimed": False,
        },
        "status": "CONSTRUCTION_ONLY_READY_NO_EXECUTION_AUTHORITY",
    }

    output.mkdir(parents=True, mode=0o755)
    (output / "manifest.json").write_bytes(canonical_bytes(manifest))
    shutil.copyfile(VALIDATOR, output / "validate_package.py")
    members = {}
    for name in ("manifest.json", "validate_package.py"):
        members[name] = sha256_file(output / name)
    sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(members.items())).encode()
    (output / "SHA256SUMS").write_bytes(sums)
    tree_root = hashlib.sha256(sums).hexdigest()
    (output / "TREE_ROOT.sha256").write_text(f"{tree_root}  SHA256SUMS\n", encoding="ascii")
    return {
        "package": output.as_posix(),
        "package_tree_root_sha256": tree_root,
        "schema": "ace2-position01-causal-state-package-construction-v1",
        "status": "PASS_CREATE_EXCLUSIVE_CONSTRUCTION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
