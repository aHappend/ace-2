#!/usr/bin/env python3
"""Inert create/use/prune regression for the Stage-1 package lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


class LifecycleFailure(RuntimeError):
    def __init__(self, message: str, *, phase: str, missing_path: Path) -> None:
        super().__init__(message)
        self.phase = phase
        self.missing_path = missing_path


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def durable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_layer(root: Path, layer: int, *, fault: str | None = None) -> dict[str, Any]:
    layer_root = root / f"layer-{layer:02d}"
    generated = layer_root / "vectors/generated-input.bin"
    result_path = layer_root / "result.json"
    payload = f"inert-layer-{layer:02d}\n".encode("ascii")
    phase = "create_generated_input"
    if fault != "missing_generated_input":
        generated.parent.mkdir(parents=True, exist_ok=False)
        generated.write_bytes(payload)
    if fault == "premature_prune" and generated.parent.exists():
        shutil.rmtree(generated.parent)
    phase = "use_generated_input"
    try:
        observed = generated.read_bytes()
    except FileNotFoundError as error:
        missing = Path(error.filename) if error.filename else generated
        raise LifecycleFailure(
            f"generated input is absent during {phase}: {missing}",
            phase=phase,
            missing_path=missing,
        ) from error
    if observed != payload:
        raise LifecycleFailure(
            f"generated input changed during {phase}: {generated}",
            phase=phase,
            missing_path=generated,
        )
    phase = "publish_durable_result"
    durable_json(
        result_path,
        {
            "layer": layer,
            "input_bytes": len(observed),
            "input_sha256": sha256_bytes(observed),
            "phase": phase,
            "status": "USED_BEFORE_PRUNE",
        },
    )
    phase = "prune_regenerable_input"
    shutil.rmtree(generated.parent)
    if not result_path.is_file() or generated.exists():
        raise LifecycleFailure(
            f"create/use/prune ordering failed during {phase}: {generated}",
            phase=phase,
            missing_path=generated,
        )
    return {
        "layer": layer,
        "created": True,
        "used": True,
        "durable_result_published": True,
        "pruned_after_use": True,
        "result_sha256": sha256_bytes(result_path.read_bytes()),
    }


def legacy_missing_path_reproduction(root: Path) -> dict[str, Any]:
    transient = root / "layer-11/vectors/generated-input.bin"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(b"inert-race\n")
    checked = transient.is_file()
    transient.unlink()
    try:
        transient.stat()
    except FileNotFoundError as error:
        missing = Path(error.filename) if error.filename else transient
        return {
            "status": "REPRODUCED",
            "exception_type": type(error).__name__,
            "missing_path": missing.as_posix(),
            "phase": "runtime_output_size_check",
            "sequence": [
                "supervisor_is_file_true",
                "child_prunes_transient",
                "supervisor_stat_missing_path",
            ],
            "is_file_before_prune": checked,
        }
    raise AssertionError("legacy check-then-stat race did not reproduce")


def exception_publication_negative(runner: Any, root: Path) -> dict[str, Any]:
    state = root / "exception-publication-state"
    state.mkdir(parents=True)
    (state / "fallback").mkdir()
    calls: list[str] = []

    def fail_primary(path: Path, value: object) -> None:
        calls.append(path.relative_to(state).as_posix())
        if path.name == "exception-record.json" and path.parent == state:
            raise OSError("injected primary exception-record publication failure")
        runner.publish(path, value)

    missing = root / "generated/missing-input.bin"
    try:
        missing.read_bytes()
    except FileNotFoundError as error:
        binding = runner.publish_exception_record(
            state,
            error,
            phase="use_generated_input",
            publisher=fail_primary,
        )
    else:
        raise AssertionError("missing-input negative control did not fail")
    selected = state / binding["path"]
    record = json.loads(selected.read_text(encoding="utf-8"))
    return {
        "status": "PASS_FALLBACK_DURABLE_BEFORE_TERMINAL",
        "primary_publication_failed": True,
        "selected_path": binding["path"],
        "selected_sha256": binding["sha256"],
        "exception_type": record["exception_type"],
        "missing_path": record["missing_path"],
        "phase": record["phase"],
        "traceback_present": bool(record["traceback"]["frames"]),
        "publisher_calls": calls,
    }


def run_all(runner: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=".stage1-lifecycle-r5-") as temporary:
        root = Path(temporary)
        reproduction = legacy_missing_path_reproduction(root / "reproduction")
        fixed_scan_bytes = runner.directory_size(root / "reproduction")
        layers_root = root / "layers"
        layers = [run_layer(layers_root, layer) for layer in range(24)]
        negative_controls: dict[str, Any] = {}
        for fault in ("premature_prune", "missing_generated_input"):
            fault_root = root / fault
            try:
                run_layer(fault_root, 0, fault=fault)
            except LifecycleFailure as error:
                negative_controls[fault] = {
                    "status": "REJECTED",
                    "exception_type": type(error).__name__,
                    "phase": error.phase,
                    "missing_path": error.missing_path.as_posix(),
                    "durable_result_exists": (fault_root / "layer-00/result.json").exists(),
                }
            else:
                raise AssertionError(f"{fault} negative control was accepted")
        negative_controls["exception_record_publication_failure"] = (
            exception_publication_negative(runner, root)
        )
        return {
            "schema": "ace2-stage1-inert-24-layer-lifecycle-regression-v1",
            "status": "PASS",
            "official": False,
            "model_executed": False,
            "rtl_executed": False,
            "layer_count": len(layers),
            "layer_ids": [record["layer"] for record in layers],
            "create_use_prune_order_exact": all(
                record["created"]
                and record["used"]
                and record["durable_result_published"]
                and record["pruned_after_use"]
                for record in layers
            ),
            "layers": layers,
            "legacy_failure_reproduction": reproduction,
            "fixed_supervisor_scan_completed": True,
            "fixed_supervisor_scan_bytes": fixed_scan_bytes,
            "negative_controls": negative_controls,
        }

