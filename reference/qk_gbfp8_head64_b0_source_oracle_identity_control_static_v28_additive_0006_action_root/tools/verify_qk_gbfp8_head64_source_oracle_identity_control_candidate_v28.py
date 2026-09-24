#!/usr/bin/env python3
"""Inert V28 candidate verifier; no official B0 payload or evaluator execution."""

from __future__ import annotations
import sys
EARLY_DONT_WRITE_BYTECODE = sys.dont_write_bytecode is True
sys.dont_write_bytecode = True

import argparse
import importlib.util
from pathlib import Path

ACTION_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_INSTALLED = False

def _audit_hook(event, args):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"}:
        raise RuntimeError(f"forbidden verifier process event: {event}")
    if event == "open" and args:
        path = args[0]
        if isinstance(path, (str, bytes)):
            text = path.decode("utf-8", "replace") if isinstance(path, bytes) else path
            lowered = text.lower()
            if lowered.endswith((".safetensors", ".npy", ".npz", ".pt", ".pth")) or "/protected-payload/" in lowered:
                raise RuntimeError("protected payload open forbidden")

def install_audit_hook():
    global _AUDIT_INSTALLED
    if not _AUDIT_INSTALLED:
        sys.addaudithook(_audit_hook); _AUDIT_INSTALLED = True

def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError("module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module

def verify(allow_acceptance: bool = False, launch_proof=None):
    core = load(ACTION_ROOT / "tools/static_contract_matrix_v28.py", "v28_static_contract_matrix")
    return core.verify_candidate(ACTION_ROOT, allow_acceptance, launch_proof)

def main() -> int:
    install_audit_hook()
    guard = load(ACTION_ROOT / "tools/literal_b_launch_guard_v28.py", "v28_bound_literal_b_guard")
    launch = {**guard.validate_current_process(Path(__file__), EARLY_DONT_WRITE_BYTECODE), **guard.prove_effective_suppression()}
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--negative-output", type=Path, required=True); args = parser.parse_args()
    report, negative = verify(False, launch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    core = load(ACTION_ROOT / "tools/static_contract_matrix_v28.py", "v28_static_contract_output")
    args.output.write_bytes(core.compact_bytes(report)); args.negative_output.write_bytes(core.compact_bytes(negative)); sys.stdout.buffer.write(core.compact_bytes(report)); return 0

if __name__ == "__main__": raise SystemExit(main())
