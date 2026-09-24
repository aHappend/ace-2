#!/usr/bin/env python3
"""Seal the bounded r5 causal-resume authority and its no-execution gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/ace2_chat_demo"
PARENT_RUNTIME = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T134355Z-runtime-output-"
    "0014-r5-continuation-repair-0006-production-executor-reviewer-repair-0002"
)
PARENT_AUTHORITY = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T134355Z-attempt-0014-r5-"
    "continuation-repair-0006-production-executor-reviewer-repair-0002-"
    "authority-state/authority.json"
)
PARENT_CONSUMED = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T134355Z-attempt-0014-r5-"
    "continuation-repair-0006-production-executor-reviewer-repair-0002-"
    "consumption-state/consumed.json"
)
PARENT_TERMINAL = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T134355Z-attempt-0014-r5-"
    "continuation-repair-0006-production-executor-reviewer-repair-0002-"
    "consumption-state/terminal.json"
)
DERIVED_ATTESTATION = BUILD / (
    "stage1-attempt-0014-repair0002-derived-status-attestation-0001/"
    "derived-status-attestation.json"
)
BACKEND = ROOT / "tools/rtl_arbitrary_text_generation_backend.py"

IDENTITY = "ace2:stage1:attempt-0014:r5:causal-resume-0001"
AUTHORITY_ROOT = BUILD / "stage1-attempt-0014-r5-causal-resume-0001-authority-state"
CONSUMPTION_ROOT = BUILD / (
    "stage1-attempt-0014-r5-causal-resume-0001-consumption-state"
)
RUNTIME_ROOT = BUILD / (
    "stage1-attempt-0014-r5-causal-resume-0001-runtime-output"
)
TERMINAL_ROOT = BUILD / (
    "stage1-attempt-0014-r5-causal-resume-0001-terminal-evidence"
)
PERMITTED_UNITS = [
    f"position-00/layer-{layer:02d}/kv-reconstruction"
    for layer in range(11, 24)
]
AUTHENTICATED_UNITS = [
    f"position-00/layer-{layer:02d}" for layer in range(24)
]


class BoundaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryError(message)


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


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"required file absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def file_record(path: Path, expected: str | None = None) -> dict[str, Any]:
    observed = sha256_file(path)
    require(expected is None or observed == expected, f"bound file changed: {path}")
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def seal(root: Path) -> dict[str, Any]:
    members = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in members
    )
    sums_path = root / "SHA256SUMS"
    sums_path.write_text(sums, encoding="ascii")
    tree_root = sha256_file(sums_path)
    (root / "TREE_ROOT.sha256").write_text(
        f"{tree_root}  SHA256SUMS\n", encoding="ascii"
    )
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    return {
        "member_count": len(members),
        "sha256sums_sha256": tree_root,
    }


def write_fresh_tree(destination: Path, files: dict[str, bytes]) -> dict[str, Any]:
    require(not os.path.lexists(destination), f"refusing to overwrite: {destination}")
    temporary = destination.with_name(destination.name + ".preparing")
    require(not os.path.lexists(temporary), f"stale preparation path: {temporary}")
    try:
        temporary.mkdir(parents=True)
        for name, value in files.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return seal(destination)


def authenticate_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    attestation = load(DERIVED_ATTESTATION)
    require(
        attestation.get("status") == "BLOCKED_CAUSAL_STATE_UNAVAILABLE"
        and attestation.get("stage1") == "OPEN"
        and attestation.get("stage2") == "FORBIDDEN",
        "derived causal-state attestation status changed",
    )
    bindings = attestation.get("authenticated_bindings")
    require(isinstance(bindings, dict), "authenticated binding set is absent")
    expected_paths = {
        "authority_sha256": PARENT_AUTHORITY,
        "checkpoint_manifest_sha256": PARENT_RUNTIME / "checkpoint-manifest.json",
        "consumed_sha256": PARENT_CONSUMED,
        "terminal_sha256": PARENT_TERMINAL,
    }
    authenticated: dict[str, Any] = {
        "derived_status_attestation": file_record(DERIVED_ATTESTATION)
    }
    for key, path in expected_paths.items():
        binding = bindings.get(key)
        require(
            isinstance(binding, dict) and binding.get("path") == relative(path),
            f"derived binding path changed: {key}",
        )
        authenticated[key] = file_record(path, str(binding.get("sha256")))

    checkpoint = load(PARENT_RUNTIME / "checkpoint-manifest.json")
    require(
        checkpoint.get("completed_unit_keys") == AUTHENTICATED_UNITS
        and checkpoint.get("next_incomplete_unit") is None
        and checkpoint.get("validation", {}).get("status") == "PASS",
        "position-00 authenticated completion boundary changed",
    )
    require(
        load(PARENT_TERMINAL).get("status") == "PASS_CONTINUATION_COMPLETE",
        "position-00 terminal receipt changed",
    )
    return checkpoint, authenticated


def audit_causal_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    salvage = PARENT_RUNTIME / "salvage-state"
    kv_layers = checkpoint["resume_state"]["kv_layers"]
    retained: list[dict[str, Any]] = []
    for layer in range(11):
        key = f"layer-{layer:02d}"
        for kind in ("k", "v"):
            binding = kv_layers[key][kind]
            path = PARENT_RUNTIME / binding["path"]
            retained.append(file_record(path, binding["sha256"]))

    missing: list[dict[str, Any]] = []
    for layer in range(11, 24):
        key = f"position-00/layer-{layer:02d}"
        unit = checkpoint["units"][layer]
        evidence = unit["evidence"]
        cache_source = evidence["cache_source"]
        result = PARENT_RUNTIME / unit["result"]["path"]
        require(
            unit["key"] == key
            and unit["validation_status"] == "PASS_AUTHENTICATED"
            and evidence["rtl_reference_agreement"] is True
            and result.is_file()
            and sha256_file(result) == unit["result"]["sha256"],
            f"authenticated unit binding changed: {key}",
        )
        for kind in ("k", "v"):
            digest = cache_source[f"rtl_observed_{kind}_sha256"]
            recorded = evidence["rtl_execution"]["kv_cache"][
                f"rtl_observed_{kind}"
            ]
            recorded_path = ROOT / recorded["path"]
            require(
                recorded["sha256"] == digest and recorded["bytes"] == 128,
                f"recorded K/V provenance changed: {key}/{kind}",
            )
            require(
                not recorded_path.exists(),
                f"previously missing K/V byte tensor reappeared: {recorded_path}",
            )
            missing.append(
                {
                    "unit_key": key,
                    "kind": kind,
                    "bytes": 128,
                    "expected_sha256": digest,
                    "recorded_path": recorded["path"],
                    "status": "MISSING_RECOMPUTATION_AUTHORIZED",
                }
            )

    salvage_members = sorted(
        path.relative_to(salvage).as_posix()
        for path in salvage.rglob("*")
        if path.is_file()
    )
    require(
        len(salvage_members) == 24,
        "salvage-state member cardinality changed",
    )
    serialized_keys: set[str] = set()

    def collect_keys(value: Any) -> None:
        if isinstance(value, dict):
            serialized_keys.update(str(key) for key in value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(checkpoint)
    absent_reference_state = {
        key: key not in serialized_keys
        for key in ("float_k", "float_v", "template", "input_norm_scale")
    }
    require(
        all(absent_reference_state.values()),
        "checkpoint unexpectedly gained serialized reference continuation state",
    )
    return {
        "authenticated_retained_int8_kv": retained,
        "authenticated_retained_int8_tensor_count": len(retained),
        "authorized_missing_int8_kv": missing,
        "authorized_missing_int8_tensor_count": len(missing),
        "salvage_state_members": salvage_members,
        "serialized_reference_state_absent": absent_reference_state,
    }


def main() -> int:
    for path in (AUTHORITY_ROOT, CONSUMPTION_ROOT, RUNTIME_ROOT, TERMINAL_ROOT):
        require(not os.path.lexists(path), f"fresh namespace already exists: {path}")

    checkpoint, parent = authenticate_parent()
    state_audit = audit_causal_state(checkpoint)
    backend = file_record(BACKEND)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    command = "python3 -B tools/create_stage1_r5_causal_resume_authority.py\n"
    authority = {
        "schema": "ace2-stage1-r5-causal-resume-authority-v1",
        "authority_identity": IDENTITY,
        "status": "SEALED_GRANTED_UNCONSUMED",
        "consumable": True,
        "created_at_utc": now,
        "authority_cardinality": 1,
        "grant": {
            "purpose": "RECOMPUTE_MISSING_POSITION_00_INT8_KV_ONLY",
            "permitted_unit_keys": PERMITTED_UNITS,
            "permitted_unit_count": len(PERMITTED_UNITS),
            "prohibited_authenticated_unit_keys": AUTHENTICATED_UNITS,
            "position_01_plus_execution": "REQUIRES_COMPLETE_AUTHENTICATED_CAUSAL_STATE",
        },
        "namespaces": {
            "authority": relative(AUTHORITY_ROOT),
            "consumption": relative(CONSUMPTION_ROOT),
            "runtime": relative(RUNTIME_ROOT),
            "terminal_evidence": relative(TERMINAL_ROOT),
        },
        "parent_authentication": parent,
        "backend": backend,
        "activity": {
            "authority_creation_cardinality": 1,
            "authority_consumption_cardinality": 0,
            "model_execution_cardinality": 0,
            "rtl_execution_cardinality": 0,
            "output_cardinality": 0,
        },
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    authority_seal = write_fresh_tree(
        AUTHORITY_ROOT,
        {
            "authority.json": canonical_bytes(authority),
            "command.txt": command.encode("ascii"),
        },
    )
    authority_record = file_record(AUTHORITY_ROOT / "authority.json")

    terminal = {
        "schema": "ace2-stage1-r5-causal-resume-terminal-v1",
        "status": "TERMINAL_NO_EXECUTION_CAUSAL_REFERENCE_STATE_INCOMPLETE",
        "failure_taxonomy": "AUTHORITY_SCOPE_INCOMPLETE_FOR_CAUSAL_RESUME",
        "created_at_utc": now,
        "authority": {
            **authority_record,
            "tree": authority_seal,
            "consumed": False,
        },
        "parent_authentication": parent,
        "causal_state_audit": state_audit,
        "first_causal_failure": {
            "unit_key": "position-00/layer-00/reference-continuation-state",
            "missing": [
                "floating K cache row",
                "floating V cache row",
                "position-00-derived fixed quantization template",
            ],
            "reason": (
                "derive_layer_token requires cache.float_k/cache.float_v for the "
                "floating reference and reuses the position-00 template at later "
                "positions; the authenticated checkpoint retained only int8 K/V "
                "for layers 00-10, and their recomputation is outside this grant"
            ),
            "backend_binding": backend,
        },
        "execution": {
            "authority_consumed": False,
            "continuation_launched": False,
            "model_executed": False,
            "rtl_executed": False,
            "position_01_plus_executed": False,
            "generated_token_count": 0,
            "completion_claimed": False,
        },
        "required_operator_decision": {
            "question": (
                "May a fresh additive authority reconstruct the missing layer-00 "
                "through layer-10 floating reference K/V and quantization templates "
                "with bound comparisons to the authenticated int8 receipts?"
            ),
            "options": [
                "AUTHORIZE_REFERENCE_STATE_RECONSTRUCTION_LAYERS_00_THROUGH_10",
                "AUTHORIZE_NEW_POSITION_01_QUANTIZATION_TRAJECTORY_NOT_CONTINUATION",
                "STOP_AND_KEEP_STAGE1_OPEN",
            ],
        },
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    terminal_seal = write_fresh_tree(
        TERMINAL_ROOT,
        {
            "terminal.json": canonical_bytes(terminal),
            "command.txt": command.encode("ascii"),
        },
    )
    print(
        "ACE2_R5_CAUSAL_RESUME_NO_EXECUTION "
        f"authority={relative(AUTHORITY_ROOT)} "
        f"terminal={relative(TERMINAL_ROOT)} "
        f"terminal_root={terminal_seal['sha256sums_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BoundaryError as error:
        print(f"ACE2_R5_CAUSAL_RESUME_SETUP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
