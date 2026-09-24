#!/usr/bin/env python3
"""Validate a future aggregate-only V28 B0 sidecar without opening protected tensors."""

from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

ACTION_ROOT = Path(__file__).resolve().parents[1]

def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError("module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module

def validate_path(document_path: Path, schema_path: Path):
    core = load(ACTION_ROOT / "tools/static_contract_matrix_v28.py", "v28_static_contract_for_sidecar")
    value = json.loads(document_path.read_text(encoding="ascii")); schema = json.loads(schema_path.read_text(encoding="ascii"))
    core.validate_sidecar(value, schema)
    return {"outcome": value["outcome"], "row_count": 574, "safe_aggregate_only": True}
